"""
lora_utils.py — LoRA injection and adapter management for AudioDiT.

Public API:
    inject_lora(model, r, lora_alpha, lora_dropout) -> PeftModel
        Wraps model.transformer with a LoraConfig targeting all attention
        projection layers in every AudioDiTBlock.

    save_lora(peft_model, path)
        Saves only the LoRA adapter weights (no base weights).

    load_lora(base_model, path) -> PeftModel
        Loads a saved adapter back onto a freshly loaded base model.

    merge_and_unload(peft_model) -> AudioDiTModel
        Merges LoRA weights into the base weights and returns a plain
        AudioDiTModel (no longer wrapped by PEFT). Use before exporting.

LoRA target modules (8 per block × 24 blocks = 192 modules):
    self_attn.to_q / to_k / to_v / to_out.0
    cross_attn.to_q / to_k / to_v / to_out.0

Usage:
    from lora_utils import inject_lora, save_lora, load_lora, merge_and_unload

    model = AudioDiTModel.from_pretrained("meituan-longcat/LongCat-AudioDiT-1B")
    peft_model = inject_lora(model, r=32, lora_alpha=32, lora_dropout=0.05)

    # training loop ...

    save_lora(peft_model, "checkpoint/step_005000")

    # later, for inference with merged weights:
    merged = merge_and_unload(peft_model)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model

# ── target modules ────────────────────────────────────────────────────────────
# AudioDiTBlock: self_attn (AudioDiTSelfAttention) + cross_attn (AudioDiTCrossAttention)
# to_out is an nn.ModuleList; index 0 is the Linear projection, 1 is Dropout.
_ATTN_TARGET_MODULES = [
    "self_attn.to_q",
    "self_attn.to_k",
    "self_attn.to_v",
    "self_attn.to_out.0",
    "cross_attn.to_q",
    "cross_attn.to_k",
    "cross_attn.to_v",
    "cross_attn.to_out.0",
]

_FFN_TARGET_MODULES = [
    "ffn.ff.0",
    "ffn.ff.3",
]


def _make_lora_config(r: int, lora_alpha: int, lora_dropout: float, include_ffn: bool) -> LoraConfig:
    target_modules = _ATTN_TARGET_MODULES + (_FFN_TARGET_MODULES if include_ffn else [])
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )


def inject_lora(
    model,
    r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    include_ffn: bool = False,
) -> PeftModel:
    """
    Wrap model.transformer with LoRA adapters.

    The base model (text_encoder + vae) stays untouched: PEFT only wraps the
    sub-module we pass in. We pass model.transformer directly so LoRA modules
    are named relative to it (e.g. "blocks.0.self_attn.to_q").

    Returns the *full* AudioDiTModel, with model.transformer replaced by the
    PeftModel wrapper. Gradient checkpointing on the base transformer weights
    is disabled automatically by PEFT; only LoRA parameters require grad.

    Args:
        model        : AudioDiTModel (loaded from HF, already on target device)
        r            : LoRA rank (32 for 1B model, 64 for 3.5B)
        lora_alpha   : LoRA scaling alpha (set equal to r → scale=1.0)
        lora_dropout : dropout applied inside LoRA layers

    Returns:
        PeftModel wrapping model.transformer.
        model.transformer is updated in-place; the original model object is
        returned for convenience so callers can keep using model.vae, etc.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    config = _make_lora_config(r, lora_alpha, lora_dropout, include_ffn)
    peft_transformer = get_peft_model(model.transformer, config)
    model.transformer = peft_transformer

    # Print trainable parameter summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"[lora_utils] LoRA injected: r={r}, alpha={lora_alpha}, dropout={lora_dropout}\n"
        f"             trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)",
        flush=True,
    )
    return model


# ── save / load ───────────────────────────────────────────────────────────────

def save_lora(model, path: Union[str, Path], merged: bool = False) -> None:
    """
    Save trained weights.

    Args:
        model  : AudioDiTModel whose .transformer is a PeftModel
        path   : output directory
        merged : False (default) — save only adapter A/B matrices + any full-tune params
                                   (adapter_config.json + adapter_model.safetensors + extras.pt)
                 True             — merge LoRA into base weights first, then save the full
                                   transformer state dict (no PEFT dependency at load time)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if merged:
        # Merge LoRA B@A into base attention weights, then save full transformer
        # state dict (includes adaln/proj_out/etc. full-tune params as-is).
        # No PEFT dependency required at load time.
        from safetensors.torch import save_file
        import copy
        tmp = copy.deepcopy(model)
        tmp.transformer = tmp.transformer.merge_and_unload()
        sd = {k: v.contiguous() for k, v in tmp.transformer.state_dict().items()}
        save_file(sd, path / "transformer_merged.safetensors")
        del tmp
        print(f"[lora_utils] merged weights saved → {path / 'transformer_merged.safetensors'}", flush=True)
    else:
        # Save adapter (A, B matrices only)
        model.transformer.save_pretrained(str(path))

        # Also save any full-tune params not managed by PEFT (e.g. proj_out, norm_out)
        extras = {
            k: v.detach().cpu()
            for k, v in model.transformer.named_parameters()
            if v.requires_grad and "lora_" not in k
        }
        if extras:
            torch.save(extras, path / "extras.pt")
            print(f"[lora_utils] extra full-tune params saved ({len(extras)} tensors) → {path / 'extras.pt'}", flush=True)

        print(f"[lora_utils] adapter saved → {path}", flush=True)


def load_lora(base_model, path: Union[str, Path], trainable: bool = False) -> object:
    """
    Load a previously saved LoRA adapter (+ optional extras.pt) onto a base model.

    Args:
        base_model : AudioDiTModel (frozen base weights, no LoRA yet)
        path       : directory containing adapter_config.json
        trainable  : True  — re-enable requires_grad for LoRA + extras (resume training)
                     False — inference only (default)

    Returns:
        The same AudioDiTModel with model.transformer replaced by a PeftModel.
    """
    path = Path(path)
    for p in base_model.parameters():
        p.requires_grad_(False)

    peft_transformer = PeftModel.from_pretrained(
        base_model.transformer, str(path), is_trainable=trainable
    )
    base_model.transformer = peft_transformer

    # Restore full-tune params (adaln, proj_out, latent_embed, etc.)
    extras_path = path / "extras.pt"
    if extras_path.exists():
        extras = torch.load(extras_path, map_location="cpu", weights_only=True)
        missing, unexpected = base_model.transformer.load_state_dict(extras, strict=False)
        if missing:
            print(f"[lora_utils] WARNING: {len(missing)} missing keys in extras — "
                  f"first 5: {missing[:5]}", flush=True)
        if unexpected:
            print(f"[lora_utils] WARNING: {len(unexpected)} unexpected keys in extras — "
                  f"first 5: {unexpected[:5]}", flush=True)
        print(f"[lora_utils] extras loaded ({len(extras)} tensors, "
              f"{len(missing)} missing, {len(unexpected)} unexpected) ← {extras_path}", flush=True)
        # 恢复训练时重新开启梯度
        if trainable:
            for k in extras:
                # 通过参数名定位并重新启用 requires_grad
                try:
                    param = base_model.transformer.get_parameter(k)
                    param.requires_grad_(True)
                except AttributeError:
                    pass
    else:
        print(f"[lora_utils] no extras.pt found at {extras_path}", flush=True)

    print(f"[lora_utils] adapter loaded ← {path}", flush=True)
    return base_model


# ── merge & export ────────────────────────────────────────────────────────────

def merge_and_unload(model) -> object:
    """
    Merge LoRA weights into the base transformer weights and unwrap PEFT.

    After this call, model.transformer is a plain AudioDiTTransformer with no
    PEFT overhead — suitable for export or direct inference without PEFT.

    Args:
        model : AudioDiTModel whose .transformer is a PeftModel

    Returns:
        The same AudioDiTModel with model.transformer now a plain nn.Module.
    """
    model.transformer = model.transformer.merge_and_unload()
    print("[lora_utils] LoRA merged into base weights; PEFT wrapper removed.", flush=True)
    return model


# ── convenience: trainable parameter names ────────────────────────────────────


def print_trainable_params(model) -> None:
    """Print all trainable parameter names and shapes (for debugging)."""
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name:80s}  {tuple(p.shape)}")
