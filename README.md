# LongCat-AudioDiT Fine-tuning

基于 [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT) 的 LoRA + 全量混合微调脚本，数据集为 [humanify/ps](https://huggingface.co/datasets/humanify/ps)（WebDataset tar 流式加载）。

## 文件结构

```
train/
├── train.py          # 主训练脚本（Accelerate + PEFT）
├── config.yaml       # 所有超参（模型/数据/训练/LoRA/组件）
├── dataset_ps.py     # 流式数据集（humanify/ps）
├── lora_utils.py     # LoRA 注入、保存、加载、merge
├── eval.py           # 推理评估 + WER 计算（每 save_every 步自动运行）
├── inspect_filter.py # 音频过滤可视化工具（听 passed/filtered 样本）
└── docs/
    └── clipping_filter.md  # 削波过滤的原因和诊断过程
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install peft accelerate soundfile jiwer
```

### 2. 配置

编辑 `config.yaml`，关键字段：

```yaml
model:
  model_dir: "meituan-longcat/LongCat-AudioDiT-1B"  # 或本地路径

training:
  steps: 10000
  batch_size: 16          # 单卡 batch，总 batch = batch_size × num_processes
  learning_rate: 1.0e-4
  output_dir: "./checkpoint"

lora:
  r: 32
  alpha: 32
```

### 3. 启动训练

```bash
accelerate launch --num_processes 4 train/train.py --config train/config.yaml
```

单卡调试：

```bash
accelerate launch --num_processes 1 train/train.py --config train/config.yaml
```

## 训练机制

### Conditional Flow Matching (CFM)

每个样本被随机切成 **prompt 区**（声纹参考）和 **gen 区**（训练目标）：

- `t ~ Uniform(0, 1)` 采样扩散时刻
- `x_t = (1-t) * z0 + t * z1`，其中 `z0 ~ N(0,I)`，`z1` = VAE 编码的干净 latent
- loss = gen 区的 `MSE(v_pred, z1 - z0)`

### CFG Dropout

训练时各自以 10% 概率丢弃文本条件或参考音频条件，使模型支持推理时的 classifier-free guidance：

| 丢弃情况 | 概率 |
|---|---|
| 保留两者 | 81% |
| 仅丢弃文本 | 9% |
| 仅丢弃 prompt | 9% |
| 两者都丢弃 | 1% |

### Prompt/Gen 切分

按比例随机切（`prompt_frac_lo ~ prompt_frac_hi`），再由 `prompt_min/max_sec` 和 `min_gen_sec` 约束，最终向下对齐到 `full_hop=2048`（VAE 下采样步长）：

```
T = 14s 时的典型分布：
  prompt: 2.8s ~ 9.8s（均匀）
  gen:    4.2s ~ 11.2s
```

## 可训练组件

在 `config.yaml` 的 `components` 段控制每个组件的训练模式：

| 组件 | 参数量 | 可选模式 |
|---|---|---|
| `dit_attn` | ~600M | `lora` / `full` / `frozen` |
| `dit_ffn` | ~340M | `lora` / `full` / `frozen` |
| `dit_adaln` | ~50M | `full` / `frozen` |
| `text_conv` | ~9M | `full` / `frozen` |
| `latent_embed` | ~9M | `full` / `frozen` |
| `latent_cond_embedder` | ~18M | `full` / `frozen` |
| `input_embed` | ~5M | `full` / `frozen` |
| `output_proj` | ~9M | `full` / `frozen` |
| `time_embed` | ~2M | `full` / `frozen` |
| `text_encoder` | — | `lora` / `frozen` |
| `vae` | — | `frozen`（不建议修改） |

**推荐配置**（底噪/音色适配）：

```yaml
components:
  dit_attn:   { mode: lora }   # 核心注意力，LoRA 省显存
  dit_ffn:    { mode: lora }   # 特征变换
  dit_adaln:  { mode: full }   # 参数小，full 效果更稳
  latent_embed:         { mode: full }  # 读 prompt latent 的关键路径
  latent_cond_embedder: { mode: full }
  # 其余 frozen
```

## 数据过滤

### 削波过滤（clip_peak_threshold）

humanify/ps 中约 5-15% 的样本存在**数字削波**（法庭录音为主），会导致 WavVAE latent 能量异常（‖z‖² 比正常大 40-80×），FM loss 飙到 15-19。

过滤条件：`peak = wav.abs().max() >= 0.99` → 丢弃。

效果：50-batch loss 分布 `mean 3.91 → 1.20`，`std 4.60 → 0.54`。

详见 [`docs/clipping_filter.md`](docs/clipping_filter.md)。

### 其他过滤条件

| 参数 | 默认值 | 作用 |
|---|---|---|
| `wer_max` | 0.3 | 丢弃转录质量差的样本 |
| `min_audio_sec` | 4.0 | 太短无法切出 prompt + gen |
| `max_audio_sec` | 20.0 | 超长音频截断 |
| `min_text_tokens` | 8 | 异常短文本丢弃 |

## Checkpoint 格式

每个 checkpoint 目录包含：

```
checkpoint/step_XXXXXXX/
├── adapter_config.json       # LoRA 配置（PEFT 标准格式）
├── adapter_model.safetensors # LoRA A/B 矩阵
└── extras.pt                 # 全量微调参数（adaln、proj_out 等）
```

### 加载 checkpoint 推理

```python
from audiodit import AudioDiTModel
from train.lora_utils import load_lora

model = AudioDiTModel.from_pretrained("meituan-longcat/LongCat-AudioDiT-1B")
model = load_lora(model, "train/checkpoint/step_0005000")
model.eval()
```

### 合并 LoRA 权重导出

```python
from train.lora_utils import merge_and_unload

merged = merge_and_unload(model)  # 返回普通 AudioDiTModel，无 PEFT 依赖
```

### 恢复训练

```yaml
# config.yaml
training:
  resume_from: "./checkpoint/step_0005000"
```

## 评估

每 `save_every` 步自动对 `eval.samples_dir` 下的 `.wav` 文件做推理并计算 WER。

手动运行：

```bash
python train/eval.py --step 5000 \
    --checkpoint_dir train/checkpoint/step_0005000 \
    --samples_dir /path/to/wavs
```

## 工具

### 检查过滤效果

```bash
# 拉取样本，分别保存 passed/filtered 到 ./filter_inspect/
python train/inspect_filter.py --n_passed 10 --n_filtered 10 --shard_frac 0.05
```

### TensorBoard

```bash
tensorboard --logdir train/checkpoint/tb
```
