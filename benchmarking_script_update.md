## Assignment 2 Systems - Benchmarking Script 进度更新（2026-06-07）

### 本次完成内容

- 修复并完善 `scripts/benchmark_transformer.py`。
- 支持 `small / medium / large / xl` 四种模型规格。
- 支持 `forward / forward_backward / train_step` 三种计时模式。
- 计时逻辑包含 warmup、正式测量、CUDA synchronize、mean/std 汇总。
- 新增 `scripts/run_benchmark_sweep.py`，自动跑四种模型和三种模式，并输出 CSV 与 Markdown 表格。
- 新增 `scripts/run_l20_benchmark_sweep.sh`，用于在 Merlin L20 worker 上创建 CUDA 12.1 兼容环境并跑实验。
- 新增 `scripts/check_gpu_env.py`，用于确认 GPU worker 上 PyTorch/CUDA 状态。

### GPU 提交与运行环境

使用 Merlin public Arnold L20 worker 跑实验。当前 agent 环境没有 TTY，普通 `mlx worker launch -- ...` 在 Login Worker 后会失败，因此使用 `--no-input`。

```bash
mlx worker launch --no-input \
  --resourcetype=public-arnold \
  --cluster=cloudnative-my \
  --queuename=compute-635-my2-cloudnative-ai-workspace.public-guarantee \
  --gpu=1 \
  --type=NVIDIA-L20 \
  -- bash /mnt/bn/ai-infra-aigc-my/mlx/users/zuotongbin.tobin/playground/assignment2-systems/scripts/run_l20_benchmark_sweep.sh
```

GPU 环境：

| Item | Value |
|-|-|
| GPU | NVIDIA L20 |
| GPU memory | 46068 MiB |
| Driver | 535.161.08 |
| CUDA reported by nvidia-smi | 12.2 |
| PyTorch used for benchmark | 2.5.1+cu121 |
| batch size | 4 |
| context length | 512 |
| warmup steps | 5 |
| measure steps | 10 |
| precision | fp32 |

说明：项目本地 `.venv` 里的 `torch 2.11.0+cu130` 与 L20 worker driver 不兼容，`torch.cuda.is_available()` 为 False。因此 GPU worker 内临时创建 `/tmp/cs336-systems-cu121-venv`，安装 `torch==2.5.1+cu121` 后执行 benchmark。

### Benchmark 主结果

单位：ms/step。

| Model | Forward Mean | Forward Std | Fwd+Bwd Mean | Fwd+Bwd Std | Train Step Mean | Train Step Std |
|-|-|-|-|-|-|-|
| small | 30.766 | 0.198 | 96.029 | 0.330 | 106.199 | 0.172 |
| medium | 92.536 | 0.312 | 292.050 | 0.394 | 323.746 | 0.531 |
| large | 208.267 | 0.371 | 644.069 | 1.274 | 738.555 | 1.397 |
| xl | 615.624 | 7.883 | 1723.526 | 6.391 | ERROR | ERROR |

`xl train_step` 失败原因：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 100.00 MiB.
GPU 0 has a total capacity of 44.53 GiB of which 93.94 MiB is free.
Process had 44.43 GiB memory in use.
```

解释：`xl` 在 fp32 + batch size 4 + context length 512 下，forward/backward 可以完成，但 train step 还需要 optimizer state / update 相关额外显存，L20 46GB 不足以完成。

### Warmup 对比

以下 warmup 对比先用 `small` 模型跑三种 mode。单位：ms/step。

| Warmup Steps | Forward Mean | Forward Std | Fwd+Bwd Mean | Fwd+Bwd Std | Train Step Mean | Train Step Std |
|-|-|-|-|-|-|
| 0 | 46.869 | 50.980 | 144.858 | 152.802 | 154.102 | 150.067 |
| 1 | 30.740 | 0.627 | 96.352 | 0.789 | 106.387 | 0.466 |
| 2 | 30.766 | 0.514 | 96.596 | 0.498 | 106.406 | 0.374 |
| 5 | 30.766 | 0.198 | 96.029 | 0.330 | 106.199 | 0.172 |

观察：

- 不做 warmup 时，首次 CUDA 初始化、kernel loading、allocator 初始化等冷启动开销会混进正式测量，导致 mean 偏高且 std 极大。
- 1-2 次 warmup 后结果已经接近稳定。
- 5 次 warmup 的 std 最小，更适合作为报告表格的正式结果。

### 当前产物

本地生成文件：

```text
benchmark_results.csv
benchmark_results.md
benchmark_warmup0.csv
benchmark_warmup0.md
benchmark_warmup1.csv
benchmark_warmup1.md
benchmark_warmup2.csv
benchmark_warmup2.md
```

### 结论

Benchmarking Script 这一部分的核心脚本能力已经完成，并已在 L20 GPU 上得到 fp32 timing 结果。报告中可以使用上面的主结果表和 warmup 对比表。唯一未完成的主表单元是 `xl train_step`，原因是 L20 46GB 显存不足；如果必须得到该单元，可以降低 batch size、改用 mixed precision，或换更大显存 GPU。
