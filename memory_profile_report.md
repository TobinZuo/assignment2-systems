# CS336 Assignment 2 Memory Profiling

Environment: Merlin worker on NVIDIA H100 80GB HBM3, PyTorch 2.5.1+cu121, `basics_impl=user-adapters`, batch size 4, one warmup step and one measured step. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was set. Peak values are `torch.cuda.max_memory_allocated()` after warmup.

## Peak memory

| Context Length | Forward fp32 | Train Step fp32 | Forward bf16 | Train Step bf16 |
|-|-|-|-|-|
| 128 | 13133.0 MiB | 52247.8 MiB | 13138.0 MiB | 52247.8 MiB |
| 2048 | 21625.1 MiB | OOM | 19457.1 MiB | OOM |

## Snapshot and timeline files

| Run | Snapshot | Active Memory Timeline HTML |
|-|-|-|
| ctx128 forward fp32 | `memory_profiles/xl_ctx128_forward_fp32.pickle` | `memory_profiles/html/xl_ctx128_forward_fp32_active_timeline.html` |
| ctx128 forward bf16 | `memory_profiles/xl_ctx128_forward_bf16.pickle` | `memory_profiles/html/xl_ctx128_forward_bf16_active_timeline.html` |
| ctx128 train_step fp32 | `memory_profiles/xl_ctx128_train_step_fp32.pickle` | `memory_profiles/html/xl_ctx128_train_step_fp32_active_timeline.html` |
| ctx128 train_step bf16 | `memory_profiles/xl_ctx128_train_step_bf16.pickle` | `memory_profiles/html/xl_ctx128_train_step_bf16_active_timeline.html` |
| ctx2048 forward fp32 | `memory_profiles/xl_ctx2048_forward_fp32.pickle` | `memory_profiles/html/xl_ctx2048_forward_fp32_active_timeline.html` |
| ctx2048 forward bf16 | `memory_profiles/xl_ctx2048_forward_bf16.pickle` | `memory_profiles/html/xl_ctx2048_forward_bf16_active_timeline.html` |

## OOM cases

- ctx2048 train_step fp32: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.11 GiB of which 1.69 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 76.53 GiB is allocated by PyTorch, and 222.82 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. 
- ctx2048 train_step bf16: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.11 GiB of which 157.00 MiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Process 3414 has 0 bytes memory in use. Of the allocated memory 66.45 GiB is allocated by PyTorch, and 161.19 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. 

## Residual stream activation size

`activation size = batch_size * context_length * d_model * bytes_per_element`

For xl, batch size 4, context length 2048, d_model 2560, fp32:

`4 * 2048 * 2560 * 4 / 1024^2 = 80.0 MiB`

The bf16/fp16 equivalent for the same residual tensor is 40.0 MiB.

## Largest allocations observed

The event table is sorted by individual allocation size in the measured window. The final-active-block table shows allocations still live at snapshot dump time; many large final blocks are model parameters/optimizer state chunks whose Python stack is not retained by the allocator snapshot.

### xl_ctx2048_forward_fp32

| Rank | Allocation event | Source |
|-|-|-|
| 1 | 2048.0 MiB | `adapters.py:124 run_scaled_dot_product_attention` |
| 2 | 2048.0 MiB | `adapters.py:126 run_scaled_dot_product_attention` |
| 3 | 2048.0 MiB | `adapters.py:733 run_softmax` |
| 4 | 2048.0 MiB | `adapters.py:734 run_softmax` |
| 5 | 2048.0 MiB | `adapters.py:735 run_softmax` |

| Rank | Active final block | Source |
|-|-|-|
| 1 | 100.0 MiB | `unknown` |
| 2 | 97.7 MiB | `unknown` |
| 3 | 32.0 MiB | `unknown` |
| 4 | 25.0 MiB | `unknown` |
| 5 | 0.1 MiB | `unknown` |

### xl_ctx128_train_step_fp32

| Rank | Allocation event | Source |
|-|-|-|
| 1 | 100.0 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::create_out(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::TensorOptions const&)` |
| 2 | 100.0 MiB | `adapters.py:844 step` |
| 3 | 97.7 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::create_out(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::TensorOptions const&)` |
| 4 | 97.7 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA_memory_format_empty(c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, std::optional<c10::MemoryFormat>)` |
| 5 | 97.7 MiB | `adapters.py:844 step` |

| Rank | Active final block | Source |
|-|-|-|
| 1 | 100.0 MiB | `unknown` |
| 2 | 100.0 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::create_out(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::TensorOptions const&)` |
| 3 | 97.7 MiB | `unknown` |
| 4 | 97.7 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::create_out(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::TensorOptions const&)` |
| 5 | 97.7 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA_memory_format_empty(c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, std::optional<c10::MemoryFormat>)` |

### xl_ctx128_train_step_bf16

| Rank | Allocation event | Source |
|-|-|-|
| 1 | 100.0 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__empty_strided(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)` |
| 2 | 100.0 MiB | `adapters.py:844 step` |
| 3 | 97.7 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__empty_strided(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)` |
| 4 | 97.7 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA_memory_format_empty(c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, std::optional<c10::MemoryFormat>)` |
| 5 | 97.7 MiB | `adapters.py:844 step` |

| Rank | Active final block | Source |
|-|-|-|
| 1 | 100.0 MiB | `unknown` |
| 2 | 100.0 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__empty_strided(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)` |
| 3 | 97.7 MiB | `unknown` |
| 4 | 97.7 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__empty_strided(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)` |
| 5 | 97.7 MiB | `RegisterCUDA.cpp:0 at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA_memory_format_empty(c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, std::optional<c10::MemoryFormat>)` |

