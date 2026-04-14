"""
train.py — Fine-tuning script for LongCat-AudioDiT.

Usage:
    python train.py --config config.yaml

Dataset format (data.data_dir):
    每个样本由同名的 .wav 和 .txt 组成，放在同一目录下。
    .wav  : 单声道 24kHz 音频（整段，包含 prompt + generation）
    .txt  : 对应的完整文本（prompt 文本 + 空格 + 生成文本）

    Prompt 边界在 Dataset 里随机切分（prompt_min_sec ~ prompt_max_sec）。
    如果你的数据已经明确区分 prompt/gen，可以直接修改 AudioDiTDataset._load_item。
"""

from __future__ import annotations

import argparse
import functools
import contextlib
import logging
import math
import random
import sys
from pathlib import Path
# ── path setup ────────────────────────────────────────────────────────────────
FINETUNE_DIR = Path(__file__).resolve().parent
LONGCAT_DIR  = FINETUNE_DIR.parent
sys.path.insert(0, str(LONGCAT_DIR))   # for: import audiodit
sys.path.insert(0, str(FINETUNE_DIR))  # for: from lora_utils import ...

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import audiodit  # noqa: F401 — registers AudioDiTConfig/AudioDiTModel
from audiodit import AudioDiTModel
from audiodit.modeling_audiodit import lens_to_mask
from eval import run_eval
from lora_utils import inject_lora, save_lora, load_lora

log = logging.getLogger(__name__)


# full-tune 组件的参数名子串（lora 注入已由 inject_lora 处理，不需要列在这里）
_FULL_SUBSTRINGS: dict[str, list[str]] = {
    "dit_adaln":            ["adaln_mlp.", "adaln_global_mlp."],
    "text_conv":            ["text_conv_layer."],
    "latent_embed":         ["latent_embed."],
    "latent_cond_embedder": ["latent_cond_embedder."],
    "input_embed":          ["input_embed."],
    "output_proj":          ["norm_out.", "proj_out"],
    "time_embed":           ["time_embed."],
}


# ─── model setup ─────────────────────────────────────────────────────────────

def setup_model(cfg: dict, device: torch.device) -> AudioDiTModel:
    comps    = cfg.get("components", {})
    lora_cfg = cfg.get("lora", {})

    model = AudioDiTModel.from_pretrained(cfg["model"]["model_dir"]).to(device)
    model.vae.to_half()

    # LoRA 注入（内部已做全局冻结 + lora_A/lora_B 初始化）
    inject_lora(
        model,
        r            = lora_cfg.get("r", 32),
        lora_alpha   = lora_cfg.get("alpha", lora_cfg.get("r", 32)),
        lora_dropout = lora_cfg.get("dropout", 0.0),
        include_ffn  = comps.get("dit_ffn", {}).get("mode") == "lora",
    )

    # 全量微调 transformer 子模块
    for comp, substrings in _FULL_SUBSTRINGS.items():
        if comps.get(comp, {}).get("mode") == "full":
            for name, p in model.transformer.named_parameters():
                if any(s in name for s in substrings) and "lora_" not in name:
                    p.requires_grad_(True)

    # text_encoder
    if comps.get("text_encoder", {}).get("mode") == "full":
        for p in model.text_encoder.parameters():
            p.requires_grad_(True)

    # vae
    if comps.get("vae", {}).get("mode") == "full":
        model.vae.float()
        for p in model.vae.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


# ─── CFM training step ────────────────────────────────────────────────────────

def cfm_step(
    model: AudioDiTModel,
    wav: torch.Tensor,                    # [B, 1, T_wav]
    wav_lens_samples: torch.Tensor,       # [B]  实际音频长度（采样点数，不含 padding）
    prompt_lens_samples: torch.Tensor,    # [B]  prompt 长度（采样点数）
    input_ids: torch.Tensor,              # [B, S]
    attention_mask: torch.Tensor,         # [B, S]
    device: torch.device,
    dtype: torch.dtype,
    train_vae: bool = False,
) -> torch.Tensor:
    """
    单步 CFM 训练，返回标量 loss。

    z1         = VAE.encode(full_audio)          [B, T_total, 64]
    z0        ~ N(0, I)                          [B, T_total, 64]
    t         ~ U(0, 1)                          [B]
    x_t        = (1-t)*z0 + t*z1                [B, T_total, 64]  ← Noisy Latent
    latent_cond: prompt区=z1_prompt, gen区=0    [B, T_total, 64]  ← Clean Latent
    v_target   = z1 - z0                        [B, T_total, 64]
    loss       = MSE(v_pred[:, T_p:], v_target[:, T_p:])
    """
    raw      = getattr(model, 'module', model)  # unwrap DDP / FSDP if needed
    full_hop = raw.config.latent_hop
    B = wav.shape[0]

    # 1. VAE encode — 逐样本 trim 到实际长度再编码
    #    复用 encode_prompt_audio（含 off=3 edge 修正、alignment pad），
    #    逐样本编码避免 batch zero-pad 污染非因果 WavVAE 边界帧。
    wav_dev  = wav.to(device)
    vae_ctx  = contextlib.nullcontext() if train_vae else torch.no_grad()
    with vae_ctx:
        z1_list: list[torch.Tensor] = []
        for i in range(B):
            wav_i        = wav_dev[i:i+1, :, :int(wav_lens_samples[i].item())]  # [1, 1, T_i]
            z1_i, _      = raw.encode_prompt_audio(wav_i)                         # [1, T_lat_i, 64]
            z1_list.append(z1_i[0])                                              # [T_lat_i, 64]

    T_lat_per = [z.shape[0] for z in z1_list]
    T_total   = max(T_lat_per)
    T_lat = torch.tensor(T_lat_per, dtype=torch.long, device=device)  # [B]
    T_p   = (prompt_lens_samples // full_hop).long().to(device)
    T_p   = T_p.clamp(max=T_lat - 1)

    z1 = torch.stack(
        [F.pad(z, (0, 0, 0, T_total - z.shape[0])) for z in z1_list], dim=0
    )                                                                   # [B, T_total, 64]

    # 2. 采样 z0 和 t
    z0   = torch.randn_like(z1)                                # [B, T_total, 64]
    t    = torch.rand(B, device=device, dtype=torch.float32)   # [B]
    t_bc = t[:, None, None]                                    # [B, 1, 1]

    # 3. Noisy Latent: x_t = (1-t)*z0 + t*z1
    x_t = (1.0 - t_bc) * z0 + t_bc * z1                       # [B, T_total, 64]

    # 4. Clean Latent: prompt 区用 z1，gen 区置零（复用 lens_to_mask）
    prompt_mask = lens_to_mask(T_p, T_total)                   # [B, T_total]
    latent_cond = z1 * prompt_mask.unsqueeze(-1)               # [B, T_total, 64]

    # 5. 目标速度场
    v_target = z1 - z0                                         # [B, T_total, 64]

    # 6. 文本编码 — encode_text 内部已有 torch.no_grad()
    text_emb = raw.encode_text(
        input_ids.to(device), attention_mask.to(device)
    )                                                           # [B, S, text_dim]
    text_len = attention_mask.sum(dim=1).to(device)            # [B]

    # 7. Masks
    audio_mask = lens_to_mask(T_lat, T_total)                  # [B, T_total]
    text_mask  = lens_to_mask(text_len, text_emb.shape[1])     # [B, S]

    # 8. Transformer forward
    output = raw.transformer(
        x=x_t.to(dtype),
        text=text_emb.to(dtype),
        text_len=text_len,
        time=t,
        mask=audio_mask,
        cond_mask=text_mask,
        latent_cond=latent_cond.to(dtype),
    )
    v_pred = output["last_hidden_state"].float()                # [B, T_total, 64]

    # 9. Loss：只在生成区（T_p <= frame_idx < T_lat）计算
    gen_mask = audio_mask & ~prompt_mask                        # [B, T_total]
    loss = F.mse_loss(
        v_pred[gen_mask].reshape(-1, z1.shape[-1]),
        v_target[gen_mask].reshape(-1, z1.shape[-1]),
    )
    return loss


# ─── dataset ─────────────────────────────────────────────────────────────────

from dataset_ps import PsStreamDataset, ps_collate_fn 


# ─── checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(model: AudioDiTModel, output_dir: Path, step: int) -> None:
    """保存 LoRA adapter + 全量组件权重（PEFT 格式，eval.py 的 load_lora 可直接加载）。"""
    ckpt_dir = output_dir / f"step_{step:07d}"
    save_lora(model, ckpt_dir)
    log.info(f"Saved → {ckpt_dir}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = FINETUNE_DIR / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    train_cfg = cfg["training"]
    data_cfg  = cfg["data"]
    dtype_str = cfg["model"].get("dtype", "bfloat16")
    dtype     = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_str]
    grad_acc  = train_cfg.get("gradient_accumulation", 1)
    mp_str    = {torch.float32: "no", torch.bfloat16: "bf16"}[dtype]

    accelerator = Accelerator(
        mixed_precision=mp_str,
        gradient_accumulation_steps=grad_acc,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )
    device = accelerator.device
    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)

    # 模型
    comps     = cfg.get("components", {})
    train_vae = comps.get("vae", {}).get("mode") != "frozen"

    model = setup_model(cfg, device)
    model.train()
    if not train_vae:
        model.vae.eval()
    model.text_encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

    # 数据
    dataset = PsStreamDataset(
        split          = data_cfg.get("hf_split", "train"),
        shard_frac     = data_cfg.get("shard_frac", 1.0) / accelerator.num_processes,
        shard_offset   = accelerator.process_index,
        sample_rate    = data_cfg.get("sample_rate", 24000),
        min_audio_sec  = data_cfg.get("min_audio_sec", 4.0),
        max_audio_sec  = data_cfg.get("max_audio_sec", 20.0),
        prompt_min_sec = data_cfg.get("prompt_min_sec", 2.0),
        prompt_max_sec = data_cfg.get("prompt_max_sec", 8.0),
        wer_max        = data_cfg.get("wer_max", None),
        seed           = train_cfg.get("seed", 42),
    )
    dl = DataLoader(
        dataset,
        batch_size   = train_cfg["batch_size"],
        num_workers  = train_cfg.get("num_workers", 0),
        pin_memory   = train_cfg.get("num_workers", 0) > 0,
        drop_last    = True,
        persistent_workers = train_cfg.get("num_workers", 0) > 0,
        collate_fn   = functools.partial(
            ps_collate_fn,
            tokenizer    = tokenizer,
            max_text_len = data_cfg.get("max_text_len", 512),
        ),
    )

    # 优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg["learning_rate"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.get("warmup_steps", 500),
        num_training_steps=train_cfg["steps"],
    )

    model, optimizer, dl, scheduler = accelerator.prepare(model, optimizer, dl, scheduler)

    # 续训
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        load_lora(accelerator.unwrap_model(model), resume_from)
        log.info(f"Resumed from {resume_from}")

    eval_cfg   = cfg.get("eval", {})

    # 训练循环
    log_every  = train_cfg.get("log_every", 50)
    save_every = train_cfg.get("save_every", 1000)
    max_steps  = train_cfg["steps"]
    max_norm   = train_cfg.get("max_grad_norm", 1.0)
    output_dir = Path(train_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = FINETUNE_DIR / output_dir
    use_amp    = (dtype != torch.float32)

    writer = SummaryWriter(log_dir=str(output_dir / "tb")) if accelerator.is_main_process else None

    step         = 0
    running_loss = 0.0
    optimizer.zero_grad()

    while step < max_steps:
        for batch in dl:
            if step >= max_steps:
                break

            with accelerator.accumulate(model):
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    loss = cfm_step(
                        model=model,
                        wav=batch["wav"],
                        wav_lens_samples=batch["wav_lens"],
                        prompt_lens_samples=batch["prompt_lens"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        device=device,
                        dtype=dtype,
                        train_vae=train_vae,
                    )
                    loss = loss / grad_acc
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_acc

            if accelerator.is_main_process:
                if step % log_every == 0 and step > 0:
                    lr   = scheduler.get_last_lr()[0]
                    avg  = running_loss / log_every
                    log.info(f"step={step:6d}  loss={avg:.4f}  lr={lr:.2e}")
                    writer.add_scalar("train/loss", avg, step)
                    writer.add_scalar("train/lr",   lr,  step)
                    running_loss = 0.0

                if step % save_every == 0:
                    save_checkpoint(accelerator.unwrap_model(model), output_dir, step)
                    if eval_cfg.get("samples_dir"):
                        run_eval(
                            model           = accelerator.unwrap_model(model),
                            tokenizer       = tokenizer,
                            samples_dir     = eval_cfg["samples_dir"],
                            output_dir      = output_dir,
                            step            = step,
                            gen_text        = eval_cfg.get("gen_text", "A gentle breeze blew across the open field as the sun began to set in the west."),
                            nfe             = eval_cfg.get("nfe", 16),
                            cfg_strength    = eval_cfg.get("cfg_strength", 4.0),
                            guidance_method = eval_cfg.get("guidance_method", "cfg"),
                            whisper_model_name = eval_cfg.get("whisper_model", "base"),
                            device          = device,
                            writer          = writer,
                            train_vae       = train_vae,
                        )

            step += 1

    if accelerator.is_main_process:
        save_checkpoint(accelerator.unwrap_model(model), output_dir, step)
        if eval_cfg.get("samples_dir"):
            run_eval(
                model           = accelerator.unwrap_model(model),
                tokenizer       = tokenizer,
                samples_dir     = eval_cfg["samples_dir"],
                output_dir      = output_dir,
                step            = step,
                gen_text        = eval_cfg.get("gen_text", "A gentle breeze blew across the open field as the sun began to set in the west."),
                nfe             = eval_cfg.get("nfe", 16),
                cfg_strength    = eval_cfg.get("cfg_strength", 4.0),
                guidance_method = eval_cfg.get("guidance_method", "cfg"),
                whisper_model_name = eval_cfg.get("whisper_model", "base"),
                device          = device,
                writer          = writer,
                train_vae       = train_vae,
            )
        writer.close()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
