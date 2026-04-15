# Clipped 音频过滤 —— 修复 humanify/ps 训练 loss 病态分布

## TL;DR

humanify/ps 数据集中约 5-15% 的样本（集中在法庭录音）在存储时已经发生**数字削波**（digital clipping），送入 WavVAE 后编码崩溃，把 FM 训练 loss 从正常 ~1 飙到 15-19。修复方法：在数据入口按峰值过滤，`peak ≥ 0.99` 的样本直接丢弃。

- **修改文件**：`train/dataset_ps.py::_decode_audio`
- **改动**：单行过滤 `if peak >= 0.99: return None`
- **效果**：50-batch loss 分布 `mean 3.91 → 1.20`、`std 4.60 → 0.54`、`max 19.73 → 4.65`

---

## 现象

用 LoRA fine-tune 跑 800 步，loss 曲线一路走高：

```
step=0    loss=0.98
step=100  loss=3.43
step=200  loss=4.23
step=400  loss=4.20
step=800  loss=4.96
```

直觉第一反应："训练在发散"。

## 诊断（逐层剥开）

### 1. 先确认是不是真的发散

加了一段"不更新参数、跑 50 batch 看 base model 的 loss 分布"的诊断代码。输出：

```
n=50  mean=3.933  median=1.823  std=4.602
min=0.943  max=19.165  p10=1.020  p90=10.734
```

**结论**：base model 自身的 loss 分布 mean 就在 ~4，max 到 19。训练 100-step 均值从 0.98 漂到 4.2 不是发散，是**抽样噪声 + 向真实均值回归**。问题不在优化器，在**数据分布本身**。

### 2. 定位：是 loss 分布里的长尾样本

每个 batch 在不同 run 里 loss 是可重现的（batch 30 两次跑都是 ~18），说明是 batch 内容决定 loss。打印 top-5 worst batch：

```
#1  batch 24  loss=19.73
#2  batch 33  loss=18.31
#3  batch 29  loss=18.25
#4  batch  8  loss=13.28
#5  batch 40  loss= 9.60
```

文本全是英语法庭/访谈内容，token 长度正常，UNK 比例 0%。文本层面看不出差异。

### 3. 继续：看 latent 统计

加上每样本 `‖z1‖²`（VAE 编出来的 latent 模长平方）：

| batch | loss | ‖z1‖² mean | ‖z1‖² max |
|---|---|---|---|
| best 20 | 0.94 | 55 | 62 |
| best 0  | 0.96 | 54 | 63 |
| **worst 24** | **19.7** | **4578** | **7996** |
| **worst 8**  | **13.3** | **2830** | **9450** |
| **worst 40** | **9.6**  | **2143** | **5275** |

正常样本 `‖z1‖² ≈ 55`（64 维 N(0,1) 期望 = 64，合理），**worst batch 的 z1 模长平方比正常大 40-80 倍**。FM loss 形式：

```
loss = MSE(v_pred, v_target = z1 - z0)
```

当 `‖z1‖²` 巨大时，`‖v_target‖²` 同比增大，MSE 的**绝对数值**天花板直接被抬高。即使模型预测的**相对误差比例**和正常 batch 差不多（~25%），绝对 loss 仍然 = 19。

### 4. 锁定：看原始波形

加上每样本的 peak / RMS 统计：

| batch | loss | peak mean | peak max |
|---|---|---|---|
| best 20 | 0.94 | 0.22 | 0.28 |
| best 0  | 0.96 | 0.23 | 0.37 |
| **worst 24** | **19.7** | **1.09** ⚠️ | **1.19** ⚠️ |
| **worst 8**  | **13.3** | **1.06** ⚠️ | **1.27** ⚠️ |
| **worst 40** | **9.6**  | **1.05** ⚠️ | **1.09** ⚠️ |

**peak > 1**。浮点 WAV 的合法范围是 [-1, 1]，peak 超出意味着音频在存储时已经被削波（或由 MP3/OGG 解码时产生过冲）。

## 根因：数字削波（Digital Clipping）

**削波**：信号幅度超过 DAC/容器允许范围时被硬性截断到最大值，波形顶部变成平面：

```
正常信号                被削波的信号
    ╱\                    ╱‾‾\
   ╱  \                  ╱    \
──────\──  ──────       ──────\──
       \  ╱                     \  ╱
        \╱                       \╱
```

平顶区域的高频谐波畸变是**不可逆**的 —— 任何 amplitude 归一化都救不回来。

### 为什么 WavVAE 会崩

WavVAE 训练时只见过 headroom 充足（peak < 0.95）的"干净"语音。碰到削波音频：

1. encoder 看到 OOD 的硬平顶波形 → 输出 `mean` / `stdev` 异常大
2. reparameterize 得到的 latent 模长爆炸（50-80× 正常）
3. `v_target = z1 - z0` 的 MSE 天花板同比飙升 → loss = 19

### 为什么法庭录音集中中招

被过滤样本的文本几乎全是法庭语料（"court reporter", "mr. hain", "police brutality cases"...）。合理解释：

1. **远距离麦克风 + 自动增益**：法庭录音从律师席/证人席远距离拾取，需要大幅提 gain，触碰 headroom
2. **后期 loudness normalization**：法庭录音常被 normalize 到 -1 / 0 dBFS 方便归档，存储时就是削波状态
3. **多人同时说话**：法官、律师、证人交叠发声时瞬时峰值超出单人 headroom

访谈 / 日常对话类录音 peak 都在 0.15-0.55，有足够 headroom。

## 修复

**只改一行**（`train/dataset_ps.py`）：

```python
_CLIP_PEAK_THRESHOLD = 0.99   # peak ≥ 此阈值视为 clipped，直接丢弃


def _decode_audio(data: bytes, sr: int) -> Optional[torch.Tensor]:
    """
    解码音频并过滤 clipped 样本。

    humanify/ps 内有一批样本存储时已发生数字削波（peak = 1.0 的硬平顶），
    其谐波畸变使非因果 WavVAE 编出来的 latent 能量比正常大 30-80×，
    直接导致 FM loss 飙到 15-19（正常 ~1）。裁剪失真是不可逆的，
    单纯 amplitude 归一化也修不回来 —— 只能在数据入口丢弃。

    正常录音留有 headroom，peak < 0.95；peak ≥ 0.99 几乎 100% 来自 clipping。
    """
    try:
        audio, _ = librosa.load(io.BytesIO(data), sr=sr, mono=True)
        wav = torch.from_numpy(audio).unsqueeze(0)
        peak = wav.abs().max().item()
        if peak >= _CLIP_PEAK_THRESHOLD:
            return None   # 丢弃 clipped 样本
        return wav
    except Exception:
        return None
```

### 为什么阈值选 0.99

- 正常录音典型 peak < 0.95（3dB+ headroom 是行业惯例）
- peak ≥ 0.99 的样本几乎 100% 来自削波（或 intentional 0dBFS normalize，效果一样）
- 0.95-0.99 之间是灰色区域，可能是"响但未削"的合法样本，保守起见不过滤

### 为什么 amplitude 归一化救不回来

我们先试过 peak safe-normalize（`if peak > 1: wav = wav / peak`）：

| 指标 | 不处理 | peak 归一化到 1.0 | 过滤 ≥ 0.99 |
|---|---|---|---|
| mean | 3.91 | 3.35 | **1.20** |
| std | 4.60 | 3.53 | **0.54** |

归一化只让 `peak=1.09` → `peak=1.00`，但削波产生的**谐波畸变仍然存在**，VAE 仍然 OOD，`‖z1‖²` 从 4578 只降到 3395。要彻底解决必须在数据入口丢弃。

## 效果

50-batch loss 分布对比：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| mean | 3.91 | **1.20** |
| median | 1.92 | **1.08** |
| std | 4.60 | **0.54** |
| min | 0.94 | 0.89 |
| max | 19.73 | **4.65** |
| p10 | 0.98 | 0.96 |
| p90 | 9.60 | **1.39** |

**意义**：std 下降 88% → LoRA 每步梯度不再被 20× 离群值淹没，训练信号可见。可以进入正经 LoRA fine-tune 了。

## 启发

1. **loss 分布病态不等于训练发散**。先用"不更新参数跑 N 个 batch"的诊断量化 base 分布，再决定动不动优化器
2. **从现象到根因的诊断顺序**：`loss → batch-level 可重现 → ‖z1‖² → peak`。每加一个统计排除一个假设
3. **数据问题优先查峰值 / RMS / 频谱**，这些比语义类分析更容易立刻命中根因
4. **削波不可逆**，任何 amplitude normalize 都修不回来，只能丢弃

## 推理侧同步

训练侧过滤了削波样本，推理侧（`utils.py::load_audio`）也应该保持一致的 peak 处理，以免用户 prompt_audio 是削波的导致生成崩坏。当前实现是 `if peak > 1: wav = wav / peak`（归一化到 1.0），**未过滤**，考虑到推理场景下丢弃用户输入不合适，保留归一化但前端需要给用户警告。
