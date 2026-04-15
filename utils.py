import re
import librosa
import torch

def load_audio(wavpath, sr):
    audio, _ = librosa.load(wavpath, sr=sr, mono=True)
    wav = torch.from_numpy(audio).unsqueeze(0)
    # Peak safe-normalize：仅当 |wav| > 1 时按 peak 缩放到 1。训练侧同步处理
    # （见 train/dataset_ps.py::_decode_audio），保证推理/训练面对一致的 VAE 输入。
    peak = wav.abs().max().item()
    if peak > 1.0:
        wav = wav / peak
    return wav

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'["“”‘’]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def approx_duration_from_text(text, max_duration=30.0):
    EN_DUR_PER_CHAR = 0.082
    ZH_DUR_PER_CHAR = 0.21
    text = re.sub(r"\s+", "", text)
    num_zh = num_en = num_other = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            num_zh += 1
        elif c.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, num_zh * ZH_DUR_PER_CHAR + num_en * EN_DUR_PER_CHAR)