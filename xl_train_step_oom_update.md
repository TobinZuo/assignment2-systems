## XL train_step OOM 排查与 H100 复验（2026-06-07）

### 结论

`xl train_step` 在 L20 46GB 上 OOM，且 batch size 降到 1 仍然 OOM。主要瓶颈不是 batch activation，而是 `xl` 模型在 fp32 AdamW 下的参数、梯度和 optimizer state 显存。换用 H100 80GB 后，user-adapters 版本在同一实验配置下已成功跑通。

最终可用结果：

| GPU | Model | Mode | Batch Size | Context Length | Precision | Warmup | Measure | Mean | Std | Result |
|-|-|-|-|-|-|-|-|-|-|-|
| L20 46GB | xl | train_step | 4 | 512 | fp32 | 5 | 10 | - | - | OOM |
| H100 80GB | xl | train_step | 4 | 512 | fp32 | 5 | 10 | 1171.549 ms | 0.371 ms | PASS |

### L20 现象

Benchmarking Script 主表中，`xl` 模型的 `forward` 和 `forward_backward` 可以在 L20 上完成，但 `train_step` 失败：

| Model | Forward Mean | Fwd+Bwd Mean | Train Step |
|-|-|-|
| xl | 615.624 ms | 1723.526 ms | OOM |

L20 环境：

| Item | Value |
|-|-|
| GPU | NVIDIA L20 |
| GPU memory | 46068 MiB |
| precision | fp32 |
| optimizer | AdamW |
| context length | 512 |

典型错误：

```text
torch.OutOfMemoryError: CUDA out of memory.
GPU 0 total capacity: 44.53 GiB
process memory in use: about 44.5 GiB
```

### Batch Size 排查

为了判断 OOM 是否主要来自 activation，固定 `context_length=512 / fp32 / AdamW / warmup=5 / measure=10`，只改变 batch size。

| Batch Size | Result |
|-|-|
| 1 | OOM |
| 2 | OOM |
| 3 | OOM |
| 4 | OOM |

batch size 降到 1 仍然 OOM，说明瓶颈不只是 batch activation。

### 原因分析

`forward_backward` 在反向传播完成后已有参数和梯度；`train_step` 还会进入 optimizer step。AdamW 需要为每个参数维护两份 fp32 状态：

- first moment `m`
- second moment `v`

因此相比 forward/backward，train step 会额外引入大量 optimizer state 显存。对 `xl` 模型而言，fp32 参数、梯度和 AdamW state 加起来已经接近或超过 L20 46GB 显存上限。

### H100 复验

重新查询 Merlin quota 后，public Arnold 资源中可用 H100：

| Resource | GPU Type | Cluster | QueueName |
|-|-|-|-|
| Public Arnold | H100-SXM-80GB | cloudnative-maliva | compute-635-aliyun.va-cloudnative-aigcp-workspace.public-guarantee |

使用脚本：

```bash
scripts/run_h100_xl_train_step.sh
```

提交命令：

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

H100 远程环境：

| Item | Value |
|-|-|
| GPU | NVIDIA H100 80GB HBM3 |
| GPU memory | 81559 MiB |
| Driver | 535.129.03 |
| CUDA reported by nvidia-smi | 12.9 |
| PyTorch | 2.5.1+cu121 |
| torch.cuda.is_available() | True |

复验结果：

| Model | Mode | Batch Size | Context Length | Precision | Warmup | Measure | Mean | Std | Result |
|-|-|-|-|-|-|-|-|-|-|
| xl | train_step | 4 | 512 | fp32 | 5 | 10 | 1171.549 ms | 0.371 ms | PASS |

结果文件：

```text
benchmark_h100_xl_train_step.csv
benchmark_h100_xl_train_step.md
benchmark_h100_xl_train_step_user_assignment1.csv
benchmark_h100_xl_train_step_user_assignment1.md
```

### 报告建议

报告中可以保留 L20 46GB 下 `xl train_step` OOM 的记录，并解释原因是 fp32 AdamW 的参数、梯度和 optimizer state 显存不足。如果需要给出完整 benchmark 表，可以用 H100 80GB 的补跑结果填补 `xl train_step` 单元，并注明该单元来自 H100 复验。

不建议为了让 L20 跑通而改用 SGD、CPU offload 或 sharded optimizer，因为这些会改变 Benchmarking Script 的实验定义。
