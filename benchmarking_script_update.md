## Assignment 2 Systems - Benchmarking Script 进度更新（2026-06-07）

### 结论

Benchmarking Script 已完成并在 GPU worker 上验证。L20 46GB 可以完成 `small / medium / large` 的三种模式，以及 `xl forward / forward_backward`；`xl train_step` 在 L20 上 OOM。后续使用 H100 80GB 按同一配置补跑 user-adapters 版本，`xl train_step` 已成功完成，结果为 **1171.549 ms/step**。

报告中建议保留 L20 主结果表，并在 `xl train_step` 位置说明 L20 显存不足；同时使用 H100 补跑结果补全该单元或作为额外验证。

### 本次完成内容

- 修复并完善 `scripts/benchmark_transformer.py`。
- 支持 `small / medium / large / xl` 四种模型规格。
- 支持 `forward / forward_backward / train_step` 三种计时模式。
- 计时逻辑包含 warmup、正式测量、CUDA synchronize、mean/std 汇总。
- 新增 `scripts/run_benchmark_sweep.py`，自动跑四种模型和三种模式，并输出 CSV 与 Markdown 表格。
- 新增 `scripts/run_l20_benchmark_sweep.sh`，用于在 Merlin L20 worker 上创建 CUDA 12.1 兼容环境并跑主实验。
- 新增 `scripts/run_h100_xl_train_step.sh`，用于在 Merlin H100 worker 上补跑 L20 OOM 的 `xl train_step`。
- 新增 `scripts/check_gpu_env.py`，用于确认 GPU worker 上 PyTorch/CUDA 状态。

### 实验配置

主实验配置：

| Item | Value |
|-|-|
| batch size | 4 |
| context length | 512 |
| warmup steps | 5 |
| measure steps | 10 |
| precision | fp32 |
| optimizer | AdamW |

L20 worker 环境：

| Item | Value |
|-|-|
| GPU | NVIDIA L20 |
| GPU memory | 46068 MiB |
| Driver | 535.161.08 |
| CUDA reported by nvidia-smi | 12.2 |
| PyTorch used for benchmark | 2.5.1+cu121 |

H100 worker 环境：

| Item | Value |
|-|-|
| GPU | NVIDIA H100 80GB HBM3 |
| GPU memory | 81559 MiB |
| Driver | 535.129.03 |
| CUDA reported by nvidia-smi | 12.9 |
| PyTorch used for benchmark | 2.5.1+cu121 |

说明：项目本地 `.venv` 中的 PyTorch 版本与部分 worker driver 不兼容，因此 GPU worker 内统一临时创建 `/tmp/cs336-systems-*-venv`，安装 `torch==2.5.1+cu121` 后执行 benchmark。

### Benchmark 结果

单位：ms/step。

L20 主结果：

| Model | Forward Mean | Forward Std | Fwd+Bwd Mean | Fwd+Bwd Std | Train Step Mean | Train Step Std |
|-|-|-|-|-|-|-|
| small | 30.766 | 0.198 | 96.029 | 0.330 | 106.199 | 0.172 |
| medium | 92.536 | 0.312 | 292.050 | 0.394 | 323.746 | 0.531 |
| large | 208.267 | 0.371 | 644.069 | 1.274 | 738.555 | 1.397 |
| xl | 615.624 | 7.883 | 1723.526 | 6.391 | OOM | OOM |

H100 补跑结果：

| Model | Mode | Batch Size | Context Length | Warmup | Measure | Precision | Mean | Std |
|-|-|-|-|-|-|-|-|-|
| xl | train_step | 4 | 512 | 5 | 10 | fp32 | 1171.549 | 0.371 |

`xl train_step` 在 L20 上的典型错误：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 100.00 MiB.
GPU 0 has a total capacity of 44.53 GiB of which 93.94 MiB is free.
Process had 44.43 GiB memory in use.
```

解释：`xl` 在 fp32 + batch size 4 + context length 512 下，forward/backward 可以完成；但 `train_step` 会执行 AdamW optimizer step，需要额外创建 optimizer state。L20 46GB 显存不足，H100 80GB 可以完成同一配置。

### Warmup 对比

以下 warmup 对比使用 `small` 模型跑三种 mode。单位：ms/step。

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

### 提交命令

L20 主实验：

```bash
mlx worker launch --no-input \
  --resourcetype=public-arnold \
  --cluster=cloudnative-my \
  --queuename=compute-635-my2-cloudnative-ai-workspace.public-guarantee \
  --gpu=1 \
  --type=NVIDIA-L20 \
  -- bash /mnt/bn/ai-infra-aigc-my/mlx/users/zuotongbin.tobin/playground/assignment2-systems/scripts/run_l20_benchmark_sweep.sh
```

H100 补跑：

```bash
NO_COLOR=1 TERM=dumb mlx worker launch --no-input \
  --resourcetype=public-arnold \
  --cluster=cloudnative-maliva \
  --queuename=compute-635-aliyun.va-cloudnative-aigcp-workspace.public-guarantee \
  --gpu=1 \
  --type=H100-SXM-80GB \
  --workdir=/mnt/bn/ecomcommonnas/mlx/users/zuotongbin.tobin/playground/assignment2-systems \
  -- bash scripts/run_h100_xl_train_step.sh
```

### 当前产物

```text
benchmark_results.csv
benchmark_results.md
benchmark_warmup0.csv
benchmark_warmup0.md
benchmark_warmup1.csv
benchmark_warmup1.md
benchmark_warmup2.csv
benchmark_warmup2.md
benchmark_h100_xl_train_step.csv
benchmark_h100_xl_train_step.md
benchmark_h100_xl_train_step_user_assignment1.csv
benchmark_h100_xl_train_step_user_assignment1.md
```
