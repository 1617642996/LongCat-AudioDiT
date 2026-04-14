"""
diagnose.py — 验证基座模型在 humanify/ps 数据集上的真实 FM 损失分布。

回答两个核心问题：
  Q1: step=0 的 loss=0.94 是幸运单批次，还是基座模型的真实水平？
  Q2: t 值采样方差对 loss 的影响有多大？

Usage:
    python diagnose.py --n_batches 200 --shard_frac 0.02
    python diagnose.py --n_batches 200 --fixed_t 0.5   # 固定 t=0.5
    python diagnose.py --n_batches 200 --fixed_t 0.9   # 固定 t=0.9（容易批次）
    python diagnose.py --n_batches 200 --fixed_t 0.1   # 固定 t=0.1（困难批次）
"""

import argparse
import contextlib
import functools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

FINETUNE_DIR = Path(__file__).resolve().parent
LONGCAT_DIR  = FINETUNE_DIR.parent
sys.path.insert(0, str(LONGCAT_DIR))
sys.path.insert(0, str(FINETUNE_DIR))

import audiodit  # noqa: F401
from audiodit import AudioDiTModel
from audiodit.modeling_audiodit import lens_to_mask
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset_ps import PsStreamDataset, ps_collate_fn


BASE_MODEL_DIR = "meituan-longcat/LongCat-AudioDiT-1B"


@torch.no_grad()
def cfm_loss_with_stats(
    model: AudioDiTModel,
    wav: torch.Tensor,
    wav_lens_samples: torch.Tensor,
    prompt_lens_samples: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    fixed_t: float | None = None,
) -> dict:
    """
    cfm_step 的诊断版本：返回 loss + t 均值 + gen_mask 统计。

    fixed_t: 若不为 None，则所有样本使用该固定 t 值（用于分析 t 方差影响）。
    """
    raw      = getattr(model, 'module', model)
    full_hop = raw.config.latent_hop
    B = wav.shape[0]

    wav_dev = wav.to(device)
    z1_list: list[torch.Tensor] = []
    for i in range(B):
        wav_i   = wav_dev[i:i+1, :, :int(wav_lens_samples[i].item())]
        z1_i, _ = raw.encode_prompt_audio(wav_i)
        z1_list.append(z1_i[0])

    T_lat_per = [z.shape[0] for z in z1_list]
    T_total   = max(T_lat_per)
    T_lat = torch.tensor(T_lat_per, dtype=torch.long, device=device)
    T_p   = (prompt_lens_samples // full_hop).long().to(device)
    T_p   = T_p.clamp(max=T_lat - 1)

    z1 = torch.stack(
        [F.pad(z, (0, 0, 0, T_total - z.shape[0])) for z in z1_list], dim=0
    )
    z0 = torch.randn_like(z1)

    if fixed_t is not None:
        t = torch.full((B,), fixed_t, device=device, dtype=torch.float32)
    else:
        t = torch.rand(B, device=device, dtype=torch.float32)
    t_bc = t[:, None, None]

    x_t = (1.0 - t_bc) * z0 + t_bc * z1

    prompt_mask = lens_to_mask(T_p, T_total)
    latent_cond = z1 * prompt_mask.unsqueeze(-1)
    v_target    = z1 - z0

    text_emb  = raw.encode_text(input_ids.to(device), attention_mask.to(device))
    text_len  = attention_mask.sum(dim=1).to(device)
    audio_mask = lens_to_mask(T_lat, T_total)
    text_mask  = lens_to_mask(text_len, text_emb.shape[1])

    output = raw.transformer(
        x=x_t.to(dtype),
        text=text_emb.to(dtype),
        text_len=text_len,
        time=t,
        mask=audio_mask,
        cond_mask=text_mask,
        latent_cond=latent_cond.to(dtype),
    )
    v_pred = output["last_hidden_state"].float()

    gen_mask = audio_mask & ~prompt_mask
    n_gen    = gen_mask.sum().item()

    loss = F.mse_loss(
        v_pred[gen_mask].reshape(-1, z1.shape[-1]),
        v_target[gen_mask].reshape(-1, z1.shape[-1]),
    )

    # 零预测基线：模型输出全零时的 MSE，理论值 ≈ 2.0
    zero_loss = F.mse_loss(
        torch.zeros_like(v_pred[gen_mask]),
        v_target[gen_mask].reshape(-1, z1.shape[-1]),
    )

    return {
        "loss":         loss.item(),
        "zero_loss":    zero_loss.item(),
        "t_mean":       t.mean().item(),
        "t_min":        t.min().item(),
        "t_max":        t.max().item(),
        "n_gen_frames": n_gen,
        "T_total":      T_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batches",   type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--shard_frac",  type=float, default=0.02)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--fixed_t",     type=float, default=None,
                        help="固定 t 值（None=随机）。尝试 0.1/0.5/0.9 查看方差影响")
    parser.add_argument("--device",      type=str,   default="cuda:0")
    parser.add_argument("--split",       type=str,   default="train")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype  = torch.bfloat16

    print(f"加载基座模型 {BASE_MODEL_DIR} ...", flush=True)
    model = AudioDiTModel.from_pretrained(BASE_MODEL_DIR).to(device)
    model.vae.to_half()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    print("模型加载完毕\n", flush=True)

    ds = PsStreamDataset(split=args.split, shard_frac=args.shard_frac, seed=42)
    dl = DataLoader(
        ds,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        collate_fn  = functools.partial(ps_collate_fn, tokenizer=tokenizer),
    )

    tag = f"t=固定{args.fixed_t}" if args.fixed_t is not None else "t=随机U[0,1]"
    print(f"=== 诊断：基座模型 × {args.n_batches} 批次 × {tag} ===\n", flush=True)

    losses      = []
    zero_losses = []
    t_means     = []

    for i, batch in enumerate(dl):
        if i >= args.n_batches:
            break
        stats = cfm_loss_with_stats(
            model               = model,
            wav                 = batch["wav"],
            wav_lens_samples    = batch["wav_lens"],
            prompt_lens_samples = batch["prompt_lens"],
            input_ids           = batch["input_ids"],
            attention_mask      = batch["attention_mask"],
            device              = device,
            dtype               = dtype,
            fixed_t             = args.fixed_t,
        )
        losses.append(stats["loss"])
        zero_losses.append(stats["zero_loss"])
        t_means.append(stats["t_mean"])

        if (i + 1) % 20 == 0 or i == 0:
            print(
                f"  batch {i+1:4d}  loss={stats['loss']:.4f}  "
                f"zero_baseline={stats['zero_loss']:.4f}  "
                f"t_mean={stats['t_mean']:.3f}",
                flush=True,
            )

    losses      = np.array(losses)
    zero_losses = np.array(zero_losses)
    t_means     = np.array(t_means)

    print(f"\n{'─'*56}")
    print(f"{'统计量':<18} {'模型loss':>10} {'零预测基线':>12}")
    print(f"{'─'*56}")
    print(f"{'均值':<18} {losses.mean():>10.4f} {zero_losses.mean():>12.4f}")
    print(f"{'标准差':<18} {losses.std():>10.4f} {zero_losses.std():>12.4f}")
    print(f"{'最小值':<18} {losses.min():>10.4f} {zero_losses.min():>12.4f}")
    print(f"{'最大值':<18} {losses.max():>10.4f} {zero_losses.max():>12.4f}")
    print(f"{'中位数':<18} {np.median(losses):>10.4f} {np.median(zero_losses):>12.4f}")
    print(f"{'─'*56}")
    print(f"t 采样范围: {t_means.min():.3f} ~ {t_means.max():.3f}  均值={t_means.mean():.3f}\n")

    ratio = losses.mean() / zero_losses.mean()
    print(f"模型loss / 零预测基线 = {ratio:.3f}")
    if ratio < 0.80:
        print("结论 → 基座模型预测良好（<80% 基线），step=0 的 0.94 是真实水平")
    elif ratio < 1.10:
        print("结论 → 基座模型接近零预测，对本数据集几乎无条件作用")
    else:
        print("结论 → 基座模型预测反向（>基线），Pashto 严重域外，训练从负值基线起步")

    if args.fixed_t is None:
        corr = np.corrcoef(t_means, losses)[0, 1]
        print(f"\nt均值与loss相关系数: {corr:.3f}")
        if corr < -0.3:
            print("→ 强负相关：t越大loss越低，t方差是训练loss高方差的主因")
            print("  （建议：换用 LogitNormal 或 CosineSchedule 采样 t，减小方差）")
        else:
            print("→ 弱相关：loss方差主要来自数据本身")


if __name__ == "__main__":
    main()
