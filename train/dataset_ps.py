"""
dataset_ps.py — Streaming IterableDataset for humanify/ps (WebDataset tar shards).

Yields one dict per sample:
    wav  : FloatTensor [1, T]  — full audio waveform at 24 kHz
    text : str                 — transcript

Usage:
    ds = PsStreamDataset(split="train", shard_frac=0.1)
    dl = DataLoader(ds, batch_size=4, collate_fn=ps_collate_fn, num_workers=4)
    for batch in dl:
        wav  = batch["wav"]   # [B, 1, T_max]
        lens = batch["lens"]  # [B]  unpadded sample counts
        text = batch["text"]  # List[str]
"""

import io
import json
import math
import os
import random
import tarfile
from typing import Iterator, Optional, Tuple

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_url, list_repo_files
from torch.utils.data import IterableDataset

import requests

DATASET_REPO = "humanify/ps"
SAMPLE_RATE  = 24_000

AUDIO_EXTS = frozenset({".flac", ".wav", ".mp3", ".ogg"})
TEXT_KEYS  = ("text", "transcript", "transcription", "sentence", "content")


def _stem_ext(name: str) -> Tuple[str, str]:
    dot = name.rfind(".")
    if dot == -1:
        return name, ""
    return name[:dot], name[dot:].lower()


def _decode_audio(data: bytes, sr: int) -> Optional[torch.Tensor]:
    try:
        audio, _ = librosa.load(io.BytesIO(data), sr=sr, mono=True)
        return torch.from_numpy(audio).unsqueeze(0)  # [1, T]
    except Exception:
        return None


class PsStreamDataset(IterableDataset):
    """
    Streams humanify/ps shards from HuggingFace.

    Args:
        split          : "train" | "validation" | "test"
        shard_frac     : fraction of shards to use (e.g. 0.1 → first 10%)
        shard_offset   : which shard_frac-sized block to pick (0-indexed)
        shuffle_shards : shuffle shard list before slicing
        seed           : RNG seed for shard shuffle
        sample_rate    : target sample rate (default 24 kHz)
        min_audio_sec  : drop samples shorter than this (need room for prompt)
        max_audio_sec  : truncate samples longer than this
        prompt_min_sec : minimum prompt duration (seconds)
        prompt_max_sec : maximum prompt duration (seconds)
        wer_max        : drop samples whose json["wer"] exceeds this threshold
        hf_token       : HuggingFace token; falls back to $HF_TOKEN env var
    """

    def __init__(
        self,
        split: str = "train",
        shard_frac: float = 1.0,
        shard_offset: int = 0,
        shuffle_shards: bool = True,
        seed: int = 42,
        sample_rate: int = SAMPLE_RATE,
        min_audio_sec: float = 4.0,
        max_audio_sec: float = 20.0,
        prompt_min_sec: float = 2.0,
        prompt_max_sec: float = 8.0,
        wer_max: Optional[float] = None,
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.split          = split
        self.shard_frac     = shard_frac
        self.shard_offset   = shard_offset
        self.shuffle_shards = shuffle_shards
        self.seed           = seed
        self.sample_rate    = sample_rate
        self.min_samples    = int(min_audio_sec  * sample_rate)
        self.max_samples    = int(max_audio_sec  * sample_rate)
        self.prompt_min     = int(prompt_min_sec * sample_rate)
        self.prompt_max     = int(prompt_max_sec * sample_rate)
        self.wer_max        = wer_max
        self.hf_token       = hf_token or os.environ.get("HF_TOKEN")

        self._shard_names, self._shard_urls = self._collect_shards()

    def _collect_shards(self) -> Tuple[list, list]:
        prefix_map = {"train": "train-", "validation": "validation-", "test": "test-"}
        if self.split not in prefix_map:
            raise ValueError(f"split must be one of {list(prefix_map)}, got {self.split!r}")
        prefix = prefix_map[self.split]

        all_files = list_repo_files(DATASET_REPO, repo_type="dataset", token=self.hf_token)
        shards = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".tar"))
        if not shards:
            raise RuntimeError(f"No tar shards found for split={self.split!r} in {DATASET_REPO}")

        if self.shuffle_shards:
            random.Random(self.seed).shuffle(shards)

        n_total  = len(shards)
        n_select = max(1, round(n_total * self.shard_frac))
        start    = (self.shard_offset * n_select) % n_total
        shards   = shards[start : start + n_select]

        print(f"[PsStreamDataset] split={self.split}, shards={len(shards)}/{n_total}", flush=True)

        urls = [hf_hub_url(repo_id=DATASET_REPO, filename=f, repo_type="dataset") for f in shards]
        return shards, urls

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            names, urls = self._shard_names, self._shard_urls
        else:
            wid, nw = worker_info.id, worker_info.num_workers
            names   = self._shard_names[wid::nw]
            urls    = self._shard_urls[wid::nw]

        for name, url in zip(names, urls):
            try:
                yield from self._iter_shard(url)
            except Exception as e:
                print(f"[PsStreamDataset] skip shard {name}: {e}", flush=True)

    def _iter_shard(self, url: str) -> Iterator[dict]:
        headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}

        current_key: Optional[str] = None
        buf_json:    Optional[dict] = None
        buf_audio:   Optional[bytes] = None

        with requests.get(url, stream=True, timeout=300, headers=headers) as resp:
            resp.raise_for_status()
            resp.raw.decode_content = True
            with tarfile.open(fileobj=resp.raw, mode="r|*") as tf:
                for member in tf:
                    if not member.isfile():
                        continue

                    stem, ext = _stem_ext(member.name)

                    if stem != current_key:
                        if buf_json is not None and buf_audio is not None:
                            item = self._process(buf_json, buf_audio)
                            if item is not None:
                                yield item
                        current_key = stem
                        buf_json = buf_audio = None

                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    data = fobj.read()

                    if ext == ".json":
                        try:
                            buf_json = json.loads(data.decode("utf-8", errors="ignore"))
                        except json.JSONDecodeError:
                            buf_json = None
                    elif ext in AUDIO_EXTS:
                        buf_audio = data

                if buf_json is not None and buf_audio is not None:
                    item = self._process(buf_json, buf_audio)
                    if item is not None:
                        yield item

    def _process(self, json_obj: dict, audio_bytes: bytes) -> Optional[dict]:
        # ── 文本 ──────────────────────────────────────────────────────────
        text = None
        for key in TEXT_KEYS:
            if key in json_obj and isinstance(json_obj[key], str):
                text = json_obj[key].strip()
                break
        if not text:
            return None

        # ── WER 过滤 ───────────────────────────────────────────────────────
        if self.wer_max is not None:
            wer = json_obj.get("wer")
            if wer is not None and wer > self.wer_max:
                return None

        # ── 音频解码 ───────────────────────────────────────────────────────
        wav = _decode_audio(audio_bytes, self.sample_rate)
        if wav is None:
            return None

        # ── 长度过滤 & 截断 ────────────────────────────────────────────────
        T = wav.shape[-1]
        if T < self.min_samples:
            return None
        if T > self.max_samples:
            wav = wav[:, :self.max_samples]
            T   = self.max_samples

        # ── 随机切 prompt 边界 ─────────────────────────────────────────────
        max_prompt = min(self.prompt_max, T // 2)
        if max_prompt <= self.prompt_min:
            prompt_len = 0
        else:
            prompt_len = random.randint(self.prompt_min, max_prompt)

        return {"wav": wav, "text": text, "prompt_len": prompt_len}


def ps_collate_fn(batch: list[dict], tokenizer=None, max_text_len: int = 512) -> dict:
    """
    Pads variable-length audio to the longest sample in the batch.

    Args:
        tokenizer    : HuggingFace tokenizer（传入时额外返回 input_ids / attention_mask）
        max_text_len : tokenizer 截断长度

    Returns:
        wav          : FloatTensor [B, 1, T_max]
        wav_lens     : LongTensor  [B]  — 向下对齐到 2048 的实际长度（VAE encoder 要求）
        prompt_lens  : LongTensor  [B]  — prompt 区长度（采样点数）
        text         : List[str]
        input_ids    : LongTensor  [B, S]  （tokenizer 不为 None 时）
        attention_mask: LongTensor [B, S]  （tokenizer 不为 None 时）
    """
    wavs = [x["wav"] for x in batch]
    lens_raw = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    # 向下对齐到 2048（= VAE total stride），逐样本编码时 trim 长度必须是整数倍
    wav_lens = (lens_raw // 2048 * 2048).clamp(min=2048)
    max_len  = int(wav_lens.max().item())
    wav_padded = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in wavs])
    prompt_lens = torch.tensor([x["prompt_len"] for x in batch], dtype=torch.long)

    result = {
        "wav":         wav_padded,
        "wav_lens":    wav_lens,
        "prompt_lens": prompt_lens,
    }
    if tokenizer is not None:
        enc = tokenizer(
            [x["text"] for x in batch],
            padding="longest",
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        result["input_ids"]       = enc["input_ids"]
        result["attention_mask"]  = enc["attention_mask"]
    return result


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--split",       default="validation")
    parser.add_argument("--shard_frac",  type=float, default=0.01)
    parser.add_argument("--n_batches",   type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--num_workers", type=int,   default=0)
    args = parser.parse_args()

    ds = PsStreamDataset(split=args.split, shard_frac=args.shard_frac)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=ps_collate_fn, num_workers=args.num_workers)

    for i, batch in enumerate(dl):
        if i >= args.n_batches:
            break
        print(f"[batch {i}] wav={batch['wav'].shape}  wav_lens={batch['wav_lens'].tolist()}")
        print(f"           text[0]={batch['text'][0][:80]!r}")

    print("done")
