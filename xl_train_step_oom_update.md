## XL train_step OOM 排查更新（2026-06-07）

### 问题

在 Benchmarking Script 主表中，`xl` 模型的 `forward` 和 `forward + backward` 可以在 L20 上完成，但 `train_step` 失败：

| Model | Forward Mean | Fwd+Bwd Mean | Train Step |
|-|-|-|
| xl | 615.624 ms | 1723.526 ms | OOM |

失败环境：

| Item | Value |
|-|-|
| GPU | NVIDIA L20 |
| GPU memory | 46068 MiB |
| precision | fp32 |
| optimizer | AdamW |
| context length | 512 |

### 尝试 1：降低 batch size

为了判断 OOM 是否主要来自 activation，继续在 L20 上测试 `xl train_step`，固定 `context_length=512 / fp32 / AdamW / warmup=5 / measure=10`，只改变 batch size。

| Batch Size | Result |
|-|-|
| 1 | OOM |
| 2 | OOM |
| 3 | OOM |
| 4 | OOM |

典型错误：

```text
torch.OutOfMemoryError: CUDA out of memory.
GPU 0 total capacity: 44.53 GiB
process memory in use: about 44.5 GiB
```

结论：降低 batch size 到 1 仍然 OOM，说明瓶颈不只是 batch activation，而是 `xl` 模型参数、梯度以及 AdamW optimizer state 本身已经接近或超过 L20 46GB 显存上限。

### 原因分析

`forward_backward` 在反向传播完成后已经有参数和梯度；`train_step` 还会进入 optimizer step。AdamW 需要为每个参数维护两份 fp32 状态：

- first moment `m`
- second moment `v`

因此相比 forward/backward，train step 会额外引入大量 optimizer state 显存。对 `xl` 模型而言，即使 batch size 降低到 1，AdamW state 仍然让 L20 显存不够。

### 尝试 2：查询更大显存 GPU

查询当前 Merlin quota：

```bash
mlx worker quota
```

当前可用资源：

| Resource | GPU Type | Available |
|-|-|-|
| Public Workspace | NVIDIA-A10 | 1 |
| Public Arnold | NVIDIA-L20 | 6 |

当前 public quota 中没有 H100，也没有 H20。

### H20 / H100 是否能解决

- H100 80GB：大概率可以缓解或解决 `xl train_step fp32 AdamW` 的显存问题。
- H20：如果是 96GB 规格，也比 L20 46GB 更有希望跑通。
- 当前账号/队列没有查到 H100/H20，所以暂时无法验证。

### 当前可行方案

1. 报告中保留 `xl train_step` OOM，并说明 L20 46GB 在 fp32 AdamW 下不足。
2. 后续实现 mixed precision benchmark，用 `fp16` 或 `bf16 autocast` 再测 `xl train_step`。
3. 如果拿到 H100/H20 或更大显存队列，再补跑 `xl train_step`。
4. 不建议为了这张基础 benchmark 表改用 SGD、CPU offload 或 sharded optimizer，因为这会改变实验定义。

### 结论

`xl train_step` 的 OOM 不是简单降低 batch size 可以解决的问题。当前证据表明主要瓶颈来自 `xl` 模型的 fp32 参数、梯度和 AdamW optimizer state。Benchmarking Script 部分可以将该项记录为 L20 46GB 下 OOM，并在报告中解释原因。
