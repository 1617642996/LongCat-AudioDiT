"""
inspect_filter.py — 拉取 humanify/ps 少量样本，按峰值分桶保存，供人工收听。

输出目录结构：
    ./filter_inspect/
        passed/   peak < 0.99，会进入训练
        filtered/ peak >= 0.99，会被丢弃

用法：
    python inspect_filter.py --n_passed 10 --n_filtered 10 --shard_frac 0.05
"""

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import librosa
import requests
import soundfile as sf
import torch
from huggingface_hub import hf_hub_url, list_repo_files

DATASET_REPO = "humanify/ps"
SAMPLE_RATE  = 24_000
AUDIO_EXTS   = frozenset({".flac", ".wav", ".mp3", ".ogg"})


def stem_ext(name):
    dot = name.rfind(".")
    if dot == -1:
        return name, ""
    return name[:dot], name[dot:].lower()


def iter_shard(url, hf_token=None):
    """逐条 yield (audio_bytes, ext, json_obj)"""
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    current_key = None
    buf_json = buf_audio = buf_ext = None

    with requests.get(url, stream=True, timeout=300, headers=headers) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        with tarfile.open(fileobj=resp.raw, mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                stem, ext = stem_ext(member.name)
                if stem != current_key:
                    if buf_json is not None and buf_audio is not None:
                        yield buf_audio, buf_ext, buf_json
                    current_key = stem
                    buf_json = buf_audio = buf_ext = None

                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                data = fobj.read()

                if ext == ".json":
                    try:
                        buf_json = json.loads(data.decode("utf-8", errors="ignore"))
                    except Exception:
                        buf_json = {}
                elif ext in AUDIO_EXTS:
                    buf_audio = data
                    buf_ext   = ext

            if buf_json is not None and buf_audio is not None:
                yield buf_audio, buf_ext, buf_json


def decode(audio_bytes):
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        wav = torch.from_numpy(audio)
        peak = wav.abs().max().item()
        return wav.numpy(), peak
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",      default="train")
    parser.add_argument("--shard_frac", type=float, default=0.05)
    parser.add_argument("--n_passed",   type=int,   default=10, help="保存多少个通过过滤的样本")
    parser.add_argument("--n_filtered", type=int,   default=10, help="保存多少个被过滤的样本")
    parser.add_argument("--threshold",  type=float, default=0.99)
    parser.add_argument("--out_dir",    default="./filter_inspect")
    parser.add_argument("--hf_token",   default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    out_pass  = Path(args.out_dir) / "passed"
    out_filt  = Path(args.out_dir) / "filtered"
    out_pass.mkdir(parents=True, exist_ok=True)
    out_filt.mkdir(parents=True, exist_ok=True)

    # 收集 shard 列表
    prefix = {"train": "train-", "validation": "validation-", "test": "test-"}[args.split]
    all_files = list(list_repo_files(DATASET_REPO, repo_type="dataset", token=hf_token))
    shards = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".tar"))
    n_select = max(1, round(len(shards) * args.shard_frac))
    shards = shards[:n_select]
    print(f"使用 {len(shards)}/{len(all_files)} 个 shard，threshold={args.threshold}")

    n_pass = n_filt = 0
    need_pass = args.n_passed
    need_filt = args.n_filtered

    for shard in shards:
        if n_pass >= need_pass and n_filt >= need_filt:
            break
        url = hf_hub_url(repo_id=DATASET_REPO, filename=shard, repo_type="dataset")
        print(f"  shard: {shard}")
        try:
            for audio_bytes, ext, json_obj in iter_shard(url, hf_token):
                if n_pass >= need_pass and n_filt >= need_filt:
                    break
                wav, peak = decode(audio_bytes)
                if wav is None:
                    continue

                dur = len(wav) / SAMPLE_RATE
                text = ""
                for k in ("text", "transcript", "transcription", "sentence", "content"):
                    if k in json_obj and isinstance(json_obj[k], str):
                        text = json_obj[k][:80]
                        break

                if peak >= args.threshold and n_filt < need_filt:
                    fname = out_filt / f"filtered_{n_filt:03d}_peak{peak:.3f}.wav"
                    sf.write(str(fname), wav, SAMPLE_RATE)
                    print(f"    [FILTERED] peak={peak:.4f}  dur={dur:.1f}s  text={text!r}")
                    print(f"               → {fname}")
                    n_filt += 1

                elif peak < args.threshold and n_pass < need_pass:
                    fname = out_pass / f"passed_{n_pass:03d}_peak{peak:.3f}.wav"
                    sf.write(str(fname), wav, SAMPLE_RATE)
                    print(f"    [PASSED]   peak={peak:.4f}  dur={dur:.1f}s  text={text!r}")
                    print(f"               → {fname}")
                    n_pass += 1

        except Exception as e:
            print(f"    skip shard: {e}")

    print(f"\n完成：passed={n_pass}, filtered={n_filt}")
    print(f"输出目录：{Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
