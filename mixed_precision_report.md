# Mixed Precision Benchmark Update

## Status

Mixed precision support has been added to `scripts/benchmark_transformer.py`.

- `--precision fp32` keeps the original full precision path.
- `--precision bf16` runs forward/loss under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- `--precision fp16` runs forward/loss under `torch.autocast(device_type="cuda", dtype=torch.float16)` and uses `GradScaler` for backward and optimizer steps.
- Model parameters remain fp32; autocast changes eligible compute op outputs.

The full H100 sweep is complete via the `ecom_ai_platform` Arnold usergroup on maliva H100. A10 results are kept as a fallback / fit check, but the H100 table should be used as the main mixed precision result.

## DType Probe

Probe command output files:

- `mixed_precision_probe_a10_fp32.json`
- `mixed_precision_probe_a10_bf16.json`
- `mixed_precision_probe_a10_fp16.json`

Key observations:

| Precision | Parameters | Embedding output | Linear / matmul / logits | Softmax | Loss |
|-|-|-|-|-|-|
| fp32 | fp32 | fp32 | fp32 | fp32 | fp32 |
| bf16 | fp32 | fp32 | bf16 | fp32 | fp32 |
| fp16 | fp32 | fp32 | fp16 | fp32 | fp32 |

This matches the expected autocast behavior: weights stay fp32, matmul-heavy operations use the requested lower precision, and numerically sensitive reductions such as softmax / cross entropy remain fp32.

## A10 Timing Results

Configuration:

- GPU: NVIDIA A10
- implementation: `user-adapters`
- batch size: 4
- context length: 512
- warmup steps: 5
- measure steps: 10
- output: `benchmark_mixed_a10_small_medium_user-adapters.csv`

| Model | Mode | fp32 mean ms | bf16 mean ms | fp16 mean ms |
|-|-|-:|-:|-:|
| small | forward | 81.926 | 39.745 | 40.049 |
| small | forward_backward | 239.774 | 120.781 | 121.756 |
| small | train_step | 261.110 | 143.178 | 146.049 |
| medium | forward | 248.573 | 107.625 | 106.370 |
| medium | forward_backward | 700.953 | 324.931 | 323.651 |
| medium | train_step | 773.072 | 397.145 | 403.463 |

## A10 Large / XL Fit Check

Configuration:

- GPU: NVIDIA A10
- implementation: `user-adapters`
- batch size: 4
- context length: 512
- warmup steps: 1
- measure steps: 3
- output: `benchmark_mixed_a10_large_xl_try_user-adapters.csv`

| Model | Mode | fp32 | bf16 | fp16 |
|-|-|-:|-:|-:|
| large | forward | 511.328 ms | 215.823 ms | 209.741 ms |
| large | forward_backward | 1469.688 ms | 651.864 ms | 646.973 ms |
| large | train_step | OOM | OOM | OOM |
| xl | forward | 1525.427 ms | 440.928 ms | 448.017 ms |
| xl | forward_backward | OOM | OOM | OOM |
| xl | train_step | OOM | OOM | OOM |

Conclusion: A10 can still provide useful large/xl forward timing data, and large forward+backward fits. It is not enough for large train_step or xl backward/train_step at batch 4, context 512. Those cases still need L20/H20/H100-class memory.

## H100 Timing Results

Configuration:

- GPU: NVIDIA H100 80GB HBM3
- resource path: `--resourcetype=arnold --usergroup=ecom_ai_platform`
- cluster: `cloudnative-maliva`
- queue: `compute-763-aliyun.va-cloudnative-aigcp-ecom.ai.platform-guarantee`
- implementation: `user-adapters`
- batch size: 4
- context length: 512
- warmup steps: 5
- measure steps: 10
- outputs:
  - `benchmark_mixed_h100_small_medium_user-adapters.csv`
  - `benchmark_mixed_h100_large_xl_user-adapters.csv`

| Model | Mode | fp32 mean ms | bf16 mean ms | fp16 mean ms |
|-|-|-:|-:|-:|
| small | forward | 22.481 | 15.273 | 15.614 |
| small | forward_backward | 67.728 | 45.929 | 47.729 |
| small | train_step | 70.852 | 54.707 | 54.417 |
| medium | forward | 60.410 | 28.831 | 31.353 |
| medium | forward_backward | 178.310 | 88.595 | 91.863 |
| medium | train_step | 191.247 | 107.816 | 109.981 |
| large | forward | 131.682 | 52.921 | 51.036 |
| large | forward_backward | 398.386 | 162.387 | 156.194 |
| large | train_step | 428.946 | 184.044 | 185.410 |
| xl | forward | 345.371 | 86.203 | 86.897 |
| xl | forward_backward | 1071.001 | 244.939 | 247.456 |
| xl | train_step | 1174.974 | 347.213 | 360.949 |

All H100 cases fit at batch 4, context 512. bf16 is the best default reporting precision: it gives the same broad speedup as fp16 while preserving more exponent range and avoiding fp16-specific scaling concerns.

## Interpretation

bf16 and fp16 roughly halve runtime for the measured A10 small/medium cases because the dominant work is GEMM-heavy Transformer compute. The speedup is strongest in forward and forward+backward. Train step improves less because optimizer work and fp32 state updates remain part of the measured step.

bf16 is the safer default for reporting because it avoids most fp16 dynamic range issues while still using lower-precision tensor-core paths. fp16 is also fast here, but it needs gradient scaling for backward / optimizer steps.

## H100 Reproduction

Prepared scripts:

- `scripts/run_h100_mixed_precision_sweep.sh`
- `scripts/run_h100_mixed_precision_incremental_small_medium.sh`
- `scripts/run_mixed_precision_incremental.py`

Recommended command:

```bash
NO_COLOR=1 TERM=dumb mlx worker launch --no-input \
  --resourcetype=arnold \
  --usergroup=ecom_ai_platform \
  --cluster=cloudnative-maliva \
  --queuename=compute-763-aliyun.va-cloudnative-aigcp-ecom.ai.platform-guarantee \
  --gpu=1 \
  --type=H100-SXM-80GB \
  --workdir=/mlx/users/zuotongbin.tobin/playground/assignment2-systems \
  -- /bin/bash /mlx/users/zuotongbin.tobin/playground/assignment2-systems/scripts/run_h100_mixed_precision_incremental_small_medium.sh
```

For `large/xl`, run:

```bash
NO_COLOR=1 TERM=dumb mlx worker launch --no-input \
  --resourcetype=arnold \
  --usergroup=ecom_ai_platform \
  --cluster=cloudnative-maliva \
  --queuename=compute-763-aliyun.va-cloudnative-aigcp-ecom.ai.platform-guarantee \
  --gpu=1 \
  --type=H100-SXM-80GB \
  --workdir=/mlx/users/zuotongbin.tobin/playground/assignment2-systems \
  -- /bin/bash /mlx/users/zuotongbin.tobin/playground/assignment2-systems/scripts/run_h100_mixed_precision_incremental_large_xl.sh
```
