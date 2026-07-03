# Memory Profiling Results

Peak values are `torch.cuda.max_memory_allocated()` after warmup, measured over the profiled step(s). Snapshot pickle files can be opened at https://pytorch.org/memory_viz.

## fp32 peak memory

| Context Length | Forward Peak Memory | Train Step Peak Memory |
|-|-|-|
| 128 | 13133.0 MiB | 52247.8 MiB |
| 2048 | 21625.1 MiB | OOM/error |

## Mixed precision comparison

| Context Length | Mode | fp32 Peak | bf16 Peak | fp16 Peak |
|-|-|-|-|-|
| 128 | forward | 13133.0 MiB | 13138.0 MiB | - |
| 128 | train_step | 52247.8 MiB | 52247.8 MiB | - |
| 2048 | forward | 21625.1 MiB | 19457.1 MiB | - |
| 2048 | train_step | OOM/error | OOM/error | - |

## Residual stream activation size

`batch_size * context_length * d_model * bytes_per_element` for xl at batch 4, context 2048, d_model 2560, fp32:

`4 * 2048 * 2560 * 4 / 1024^2 = 80.0 MiB`
