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

import librosaC
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
    返回 (wav_path, text) 列表。
    txt 文件不存在时用 Whisper 转录并写回。
    """
    wav_paths = sorted(samples_dir.glob("*.wav"))
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
        out_path = output_dir / wav_path.name
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


if __name__ == "__main__":
    main()
