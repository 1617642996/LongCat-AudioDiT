"""
eval.py — 读取本地音频目录，推理并计算 WER。

samples_dir 格式：
    sample_00.wav  [+ sample_00.txt]
    sample_01.wav  [+ sample_01.txt]
    ...
    txt 缺失时自动用 Whisper 转录并写回。

Usage:
    # 评估基座模型:
    python eval.py --step 0 --samples_dir /path/to/wavs

    # 评估 LoRA checkpoint:
    python eval.py --step 5000 --checkpoint_dir ./checkpoints/step_005000 --samples_dir /path/to/wavs
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import librosa
import numpy as np
import soundfile as sf
import torch

# ── path setup ────────────────────────────────────────────────────────────────
FINETUNE_DIR = Path(__file__).resolve().parent
LONGCAT_DIR  = FINETUNE_DIR.parent.parent
sys.path.insert(0, str(LONGCAT_DIR))
sys.path.insert(0, str(FINETUNE_DIR))

import audiodit  # noqa: F401
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from batch_inference import infer_one
from lora_utils import load_lora

BASE_MODEL_DIR = "meituan-longcat/LongCat-AudioDiT-1B"


# ── sample loading ────────────────────────────────────────────────────────────

def load_samples(samples_dir: Path, asr) -> list[tuple[Path, str]]:
    """
    返回 (wav_path, text) 列表。递归搜索子目录下的 wav 文件。
    txt 文件不存在时用 Whisper 转录并写回。
    """
    wav_paths = sorted(samples_dir.rglob("*.wav"))
    assert wav_paths, f"No wav files found in {samples_dir}"

    items = []
    for wav_path in wav_paths:
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
        else:
            audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
            text = asr.transcribe(audio, language=None)["text"].strip()
            txt_path.write_text(text, encoding="utf-8")
        items.append((wav_path, text))

    return items


# ── WER ───────────────────────────────────────────────────────────────────────

def word_error_rate(hyp: str, ref: str) -> float:
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r:
        return 0.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost    = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(r)][len(h)] / len(r)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",            type=int, required=True)
    parser.add_argument("--samples_dir",     type=str, required=True,
                        help="本地音频目录，包含 *.wav（txt 缺失时自动转录）")
    parser.add_argument("--checkpoint_dir",  type=str, default=None,
                        help="LoRA adapter 目录（save_lora 保存的）。不传则评估基座模型")
    parser.add_argument("--output_dir",      type=str, default="eval_out")
    parser.add_argument("--metrics_csv",     type=str, default="eval_out/metrics.csv")
    parser.add_argument("--gen_text",        type=str,
                        default="A gentle breeze blew across the open field as the sun began to set in the west.",
                        help="固定合成文本")
    parser.add_argument("--nfe",             type=int, default=16)
    parser.add_argument("--cfg_strength",    type=float, default=4.0)
    parser.add_argument("--guidance_method", type=str, default="cfg", choices=["cfg", "apg"])
    parser.add_argument("--device",          type=str, default="cuda:0")
    parser.add_argument("--whisper_model",   type=str, default="base")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_dir  = Path(args.output_dir) / f"step_{args.step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    import whisper
    asr   = whisper.load_model(args.whisper_model, device=str(device))
    items = load_samples(samples_dir, asr)

    model = AudioDiTModel.from_pretrained(BASE_MODEL_DIR).to(device)
    model.vae.to_half()
    if args.checkpoint_dir:
        load_lora(model, args.checkpoint_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

    # 4. 推理 + WER
    wers    = []
    rows    = []          # 用于 step 目录内的 results.csv
    gen_text = args.gen_text
    for i, (wav_path, text) in enumerate(items):
        out_path = output_dir / (wav_path.parent.name + ".wav")
        wav_np = infer_one(
            gen_text        = gen_text,
            prompt_text     = text,
            prompt_wav_path = str(wav_path),
            model           = model,
            tokenizer       = tokenizer,
            device          = device,
            nfe             = args.nfe,
            cfg_strength    = args.cfg_strength,
            guidance_method = args.guidance_method,
        )
        sf.write(str(out_path), wav_np, model.config.sampling_rate)

        audio_16k, _ = librosa.load(str(out_path), sr=16000, mono=True)
        hyp = asr.transcribe(audio_16k, language=None)["text"].strip()
        wer = word_error_rate(hyp, gen_text)
        wers.append(wer)
        rows.append([wav_path.name, gen_text, hyp, f"{wer:.4f}"])
        print(f"  {wav_path.name}  WER={wer:.3f}", flush=True)

    mean_wer = float(np.mean(wers))
    print(f"step={args.step}  WER={mean_wer:.4f}  n={len(items)}", flush=True)

    # ── step 目录内写 results.csv（含推理文本和 ASR 转录）──────────────────────
    results_path = output_dir / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "gen_text", "hyp", "wer"])
        w.writerows(rows)
        w.writerow(["MEAN", gen_text, "", f"{mean_wer:.4f}"])

    # ── 全局汇总 CSV（跨 step 对比用）─────────────────────────────────────────
    csv_path = Path(args.metrics_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_hdr:
            w.writerow(["step", "mean_wer", "n_samples", "gen_text"])
        w.writerow([args.step, f"{mean_wer:.4f}", len(items), gen_text])


@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    samples_dir: str,
    output_dir: Path,
    step: int,
    gen_text: str,
    nfe: int = 16,
    cfg_strength: float = 4.0,
    guidance_method: str = "cfg",
    whisper_model_name: str = "base",
    device=None,
    writer=None,
    train_vae: bool = False,
) -> None:
    """
    在 checkpoint 时刻推理并记录日志。复用训练进程中已加载的 model，不重新加载权重。

    Args:
        model           : 已 unwrap 的 AudioDiTModel（eval.py 内不重新 from_pretrained）
        tokenizer       : 对应 tokenizer
        samples_dir     : 本地音频目录（递归搜索 *.wav）
        output_dir      : 基础输出目录；生成音频写入 output_dir/step_{step:07d}/
        step            : 当前训练步数（用于命名和 TensorBoard global_step）
        gen_text        : 固定的待合成文本
        writer          : SummaryWriter（传 None 则不写 TensorBoard）
        train_vae       : 与训练循环一致，决定 eval 后是否把 vae 恢复成 eval 模式
    """
    import whisper

    if device is None:
        device = next(model.parameters()).device

    samples_dir = Path(samples_dir)
    step_dir    = output_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    asr   = whisper.load_model(whisper_model_name, device=str(device))
    items = load_samples(samples_dir, asr)

    was_training = model.training
    model.eval()
    wers = []
    try:
        for i, (wav_path, prompt_text) in enumerate(items):
            out_path = step_dir / (wav_path.parent.name + ".wav")
            wav_np = infer_one(
                gen_text        = gen_text,
                prompt_text     = prompt_text,
                prompt_wav_path = str(wav_path),
                model           = model,
                tokenizer       = tokenizer,
                device          = device,
                nfe             = nfe,
                cfg_strength    = cfg_strength,
                guidance_method = guidance_method,
            )
            sf.write(str(out_path), wav_np, model.config.sampling_rate)

            audio_16k, _ = librosa.load(str(out_path), sr=16000, mono=True)
            hyp = asr.transcribe(audio_16k, language=None)["text"].strip()
            wer = word_error_rate(hyp, gen_text)
            wers.append(wer)

            category = wav_path.parent.parent.name   # Clean / High_Noise / ...
            print(f"  [{step}][{category}/{wav_path.parent.name}] WER={wer:.3f}", flush=True)

            if writer is not None:
                wav_t = torch.from_numpy(wav_np).unsqueeze(0)  # [1, T]
                writer.add_audio(
                    f"eval/{category}/{wav_path.parent.name}",
                    wav_t, step,
                    sample_rate=model.config.sampling_rate,
                )
    finally:
        del asr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if was_training:
            model.train()
            if not train_vae:
                model.vae.eval()
            model.text_encoder.eval()

    if wers:
        mean_wer = float(np.mean(wers))
        print(f"Eval step={step}  mean_WER={mean_wer:.4f}  n={len(wers)}", flush=True)
        if writer is not None:
            writer.add_scalar("eval/WER", mean_wer, step)


if __name__ == "__main__":
    main()
