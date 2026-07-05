from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is only required for GPU tests.
    triton = None
    tl = None


def _causal_mask(scores: torch.Tensor, n_queries: int, n_keys: int) -> torch.Tensor:
    mask = torch.arange(n_queries, device=scores.device)[:, None] >= torch.arange(n_keys, device=scores.device)[None, :]
    return scores.masked_fill(~mask, -1e6)


def _attention_forward_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        scores = _causal_mask(scores, q.shape[-2], k.shape[-2])
    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, v)
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def _attention_backward_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    is_causal: bool,
):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        scores = _causal_mask(scores, q.shape[-2], k.shape[-2])

    p = torch.exp(scores - lse.unsqueeze(-1))
    dv = torch.matmul(p.transpose(-2, -1), dout)
    dp = torch.matmul(dout, v.transpose(-2, -1))
    d = (dout * out).sum(dim=-1, keepdim=True)
    ds = p * (dp - d)
    dq = torch.matmul(ds, k) * scale
    dk = torch.matmul(ds.transpose(-2, -1), q) * scale
    return dq, dk, dv


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        out, lse = _attention_forward_torch(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _attention_backward_torch(q, k, v, out, lse, dout, ctx.is_causal)
        return dq, dk, dv, None


if triton is not None:

    @triton.jit
    def _flash_attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        lse_ptr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        d_model: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_n: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_n: tl.constexpr,
        v_stride_d: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_d: tl.constexpr,
        lse_stride_b: tl.constexpr,
        lse_stride_n: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q = tl.load(
            q_ptr + pid_b * q_stride_b + offs_m[:, None] * q_stride_n + offs_d[None, :] * q_stride_d,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < d_model),
            other=0.0,
        )

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in tl.range(0, n_keys, BLOCK_N):
            curr_n = start_n + offs_n
            k = tl.load(
                k_ptr + pid_b * k_stride_b + curr_n[None, :] * k_stride_n + offs_d[:, None] * k_stride_d,
                mask=(curr_n[None, :] < n_keys) & (offs_d[:, None] < d_model),
                other=0.0,
            )
            scores = tl.dot(q, k, input_precision="ieee") * scale
            valid = (offs_m[:, None] < n_queries) & (curr_n[None, :] < n_keys)
            if is_causal:
                valid = valid & (offs_m[:, None] >= curr_n[None, :])
            scores = tl.where(valid, scores, -1.0e20)

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)

            v = tl.load(
                v_ptr + pid_b * v_stride_b + curr_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
                mask=(curr_n[:, None] < n_keys) & (offs_d[None, :] < d_model),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v, input_precision="ieee")
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        out = acc / l_i[:, None]
        tl.store(
            out_ptr + pid_b * out_stride_b + offs_m[:, None] * out_stride_n + offs_d[None, :] * out_stride_d,
            out,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < d_model),
        )
        tl.store(
            lse_ptr + pid_b * lse_stride_b + offs_m * lse_stride_n,
            tl.log(l_i) + m_i,
            mask=offs_m < n_queries,
        )

    @triton.jit
    def _flash_attention_backward_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        lse_ptr,
        dout_ptr,
        dq_ptr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        d_model: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_n: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_n: tl.constexpr,
        v_stride_d: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_d: tl.constexpr,
        lse_stride_b: tl.constexpr,
        lse_stride_n: tl.constexpr,
        dout_stride_b: tl.constexpr,
        dout_stride_n: tl.constexpr,
        dout_stride_d: tl.constexpr,
        dq_stride_b: tl.constexpr,
        dq_stride_n: tl.constexpr,
        dq_stride_d: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q = tl.load(
            q_ptr + pid_b * q_stride_b + offs_m[:, None] * q_stride_n + offs_d[None, :] * q_stride_d,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < d_model),
            other=0.0,
        )
        out = tl.load(
            out_ptr + pid_b * out_stride_b + offs_m[:, None] * out_stride_n + offs_d[None, :] * out_stride_d,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < d_model),
            other=0.0,
        )
        dout = tl.load(
            dout_ptr + pid_b * dout_stride_b + offs_m[:, None] * dout_stride_n + offs_d[None, :] * dout_stride_d,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < d_model),
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + pid_b * lse_stride_b + offs_m * lse_stride_n,
            mask=offs_m < n_queries,
            other=float("inf"),
        )
        delta = tl.sum(out * dout, axis=1)
        dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in tl.range(0, n_keys, BLOCK_N):
            curr_n = start_n + offs_n
            k = tl.load(
                k_ptr + pid_b * k_stride_b + curr_n[None, :] * k_stride_n + offs_d[:, None] * k_stride_d,
                mask=(curr_n[None, :] < n_keys) & (offs_d[:, None] < d_model),
                other=0.0,
            )
            v = tl.load(
                v_ptr + pid_b * v_stride_b + curr_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
                mask=(curr_n[:, None] < n_keys) & (offs_d[None, :] < d_model),
                other=0.0,
            )

            scores = tl.dot(q, k, input_precision="ieee") * scale
            valid = (offs_m[:, None] < n_queries) & (curr_n[None, :] < n_keys)
            if is_causal:
                valid = valid & (offs_m[:, None] >= curr_n[None, :])
            p = tl.exp(scores - lse[:, None])
            p = tl.where(valid, p, 0.0)

            dp = tl.dot(dout, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(tl.float32), tl.trans(k), input_precision="ieee")

        dq *= scale
        tl.store(
            dq_ptr + pid_b * dq_stride_b + offs_m[:, None] * dq_stride_n + offs_d[None, :] * dq_stride_d,
            dq,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < d_model),
        )

    @triton.jit
    def _flash_attention_backward_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        lse_ptr,
        dout_ptr,
        dk_ptr,
        dv_ptr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        d_model: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_n: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_n: tl.constexpr,
        v_stride_d: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_d: tl.constexpr,
        lse_stride_b: tl.constexpr,
        lse_stride_n: tl.constexpr,
        dout_stride_b: tl.constexpr,
        dout_stride_n: tl.constexpr,
        dout_stride_d: tl.constexpr,
        dk_stride_b: tl.constexpr,
        dk_stride_n: tl.constexpr,
        dk_stride_d: tl.constexpr,
        dv_stride_b: tl.constexpr,
        dv_stride_n: tl.constexpr,
        dv_stride_d: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        k = tl.load(
            k_ptr + pid_b * k_stride_b + offs_n[:, None] * k_stride_n + offs_d[None, :] * k_stride_d,
            mask=(offs_n[:, None] < n_keys) & (offs_d[None, :] < d_model),
            other=0.0,
        )
        v = tl.load(
            v_ptr + pid_b * v_stride_b + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
            mask=(offs_n[:, None] < n_keys) & (offs_d[None, :] < d_model),
            other=0.0,
        )

        dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

        for start_m in tl.range(0, n_queries, BLOCK_M):
            curr_m = start_m + offs_m
            q = tl.load(
                q_ptr + pid_b * q_stride_b + curr_m[:, None] * q_stride_n + offs_d[None, :] * q_stride_d,
                mask=(curr_m[:, None] < n_queries) & (offs_d[None, :] < d_model),
                other=0.0,
            )
            out = tl.load(
                out_ptr + pid_b * out_stride_b + curr_m[:, None] * out_stride_n + offs_d[None, :] * out_stride_d,
                mask=(curr_m[:, None] < n_queries) & (offs_d[None, :] < d_model),
                other=0.0,
            )
            dout = tl.load(
                dout_ptr + pid_b * dout_stride_b + curr_m[:, None] * dout_stride_n + offs_d[None, :] * dout_stride_d,
                mask=(curr_m[:, None] < n_queries) & (offs_d[None, :] < d_model),
                other=0.0,
            )
            lse = tl.load(lse_ptr + pid_b * lse_stride_b + curr_m * lse_stride_n)
            delta = tl.sum(out * dout, axis=1)

            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            valid = (curr_m[:, None] < n_queries) & (offs_n[None, :] < n_keys)
            if is_causal:
                valid = valid & (curr_m[:, None] >= offs_n[None, :])
            p = tl.exp(scores - lse[:, None])
            p = tl.where(valid, p, 0.0)

            dv += tl.dot(tl.trans(p).to(tl.float32), dout, input_precision="ieee")
            dp = tl.dot(dout, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds).to(tl.float32), q, input_precision="ieee")

        dk *= scale
        tl.store(
            dk_ptr + pid_b * dk_stride_b + offs_n[:, None] * dk_stride_n + offs_d[None, :] * dk_stride_d,
            dk,
            mask=(offs_n[:, None] < n_keys) & (offs_d[None, :] < d_model),
        )
        tl.store(
            dv_ptr + pid_b * dv_stride_b + offs_n[:, None] * dv_stride_n + offs_d[None, :] * dv_stride_d,
            dv,
            mask=(offs_n[:, None] < n_keys) & (offs_d[None, :] < d_model),
        )

    def _attention_backward_triton(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        dout: torch.Tensor,
        is_causal: bool,
    ):
        dout = dout.contiguous()

        batch_size, n_queries, d_model = q.shape
        n_keys = k.shape[-2]
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        block_m = 16
        block_n = 32
        block_d = triton.next_power_of_2(d_model)
        if block_d > 128:
            raise ValueError(f"FlashAttentionTriton currently supports d_model <= 128, got {d_model}.")

        scale = 1.0 / math.sqrt(d_model)
        grid_dq = (triton.cdiv(n_queries, block_m), batch_size)
        _flash_attention_backward_dq_kernel[grid_dq](
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            n_queries,
            n_keys,
            d_model,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            scale,
            bool(is_causal),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )

        grid_dkdv = (triton.cdiv(n_keys, block_n), batch_size)
        _flash_attention_backward_dkdv_kernel[grid_dkdv](
            q,
            k,
            v,
            out,
            lse,
            dout,
            dk,
            dv,
            n_queries,
            n_keys,
            d_model,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            scale,
            bool(is_causal),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return dq, dk, dv


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        if triton is None:
            raise RuntimeError("Triton is not available in this environment.")
        if not q.is_cuda:
            raise RuntimeError("FlashAttentionTriton requires CUDA tensors.")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Expected q, k, and v to have shape [batch, seq, d_model].")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
            raise ValueError("q, k, and v must have matching batch and d_model dimensions.")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        batch_size, n_queries, d_model = q.shape
        n_keys = k.shape[-2]
        out = torch.empty_like(q)
        lse = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

        block_m = 16
        block_n = 32
        block_d = triton.next_power_of_2(d_model)
        if block_d > 128:
            raise ValueError(f"FlashAttentionTriton currently supports d_model <= 128, got {d_model}.")

        grid = (triton.cdiv(n_queries, block_m), batch_size)
        _flash_attention_forward_kernel[grid](
            q,
            k,
            v,
            out,
            lse,
            n_queries,
            n_keys,
            d_model,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            1.0 / math.sqrt(d_model),
            bool(is_causal),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _attention_backward_triton(q, k, v, out, lse, dout, ctx.is_causal)
        return dq, dk, dv, None
