"""Custom Triton kernels for GPU-accelerated deep learning operations.
Implements fused softmax, FlashAttention, and layer norm using OpenAI Triton.
Reference: Triton (Tillet et al.), FlashAttention (Dao et al. NeurIPS 2022).
"""
import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    # Fallback stubs for environments without Triton
    class triton:
        @staticmethod
        def jit(fn): return fn
        @staticmethod
        def autotune(configs, key): return lambda fn: fn
        @staticmethod
        def heuristics(d): return lambda fn: fn
        class Config:
            def __init__(self, kw, num_warps=4, num_stages=2): pass
    class tl:
        constexpr = int
        @staticmethod
        def program_id(axis): return 0
        @staticmethod
        def arange(start, end): return None
        @staticmethod
        def load(ptr, mask=None, other=0.0): return None
        @staticmethod
        def store(ptr, val, mask=None): pass
        @staticmethod
        def exp(x): return None
        @staticmethod
        def sum(x, axis=0): return None
        @staticmethod
        def max(x, axis=0): return None
        @staticmethod
        def log(x): return None
        @staticmethod
        def dot(a, b): return None
        @staticmethod
        def zeros(shape, dtype): return None
        float32 = None; float16 = None


@triton.jit
def _fused_softmax_kernel(
    x_ptr, out_ptr,
    stride_row, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """Row-wise softmax: load row, compute max, subtract, exp, normalize."""
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    row_start = x_ptr + row_idx * stride_row
    x = tl.load(row_start + offs, mask=mask, other=-float('inf'))
    row_max = tl.max(x, axis=0)
    x = x - row_max
    x_exp = tl.exp(x)
    row_sum = tl.sum(x_exp, axis=0)
    out = x_exp / row_sum
    tl.store(out_ptr + row_idx * stride_row + offs, out, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Fused row-wise softmax using custom Triton kernel."""
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    if not TRITON_AVAILABLE or not x.is_cuda:
        return torch.softmax(x, dim=-1)
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    n_warps = max(4, min(32, BLOCK_SIZE // 32))
    _fused_softmax_kernel[(n_rows,)](x, out, x.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=n_warps)
    return out


@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, L_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, M, N, D,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """FlashAttention forward: tiled O(N) SRAM computation."""
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H; off_h = off_bh % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptrs = Q_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        k_ptrs = K_ptr + off_b * stride_kb + off_h * stride_kh + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + off_b * stride_vb + off_h * stride_vh + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=((start_n + offs_n[:, None]) < N) & (offs_d[None, :] < D), other=0.0)
        v = tl.load(v_ptrs, mask=((start_n + offs_n[:, None]) < N) & (offs_d[None, :] < D), other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        m_new = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_new)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    out_ptrs = Out_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))
    l_ptrs = L_ptr + off_bh * M + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < M)


def flash_attention_triton(q, k, v, causal=False):
    """FlashAttention with Triton kernel. q,k,v: (B, H, N, D)."""
    if not TRITON_AVAILABLE or not q.is_cuda:
        # Fallback: standard attention
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones(*scores.shape[-2:], device=q.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(attn, v)

    B, H, M, D = q.shape
    N = k.shape[2]
    BLOCK_M, BLOCK_N, BLOCK_D = 128, 64, triton.next_power_of_2(D)
    out = torch.empty_like(q)
    L = torch.empty(B * H, M, device=q.device, dtype=torch.float32)
    scale = D ** -0.5
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    _flash_attn_fwd_kernel[grid](
        q, k, v, out, L,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        B, H, M, N, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2
    )
    return out


@triton.jit
def _layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride_row, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_cols
    x_norm = (x - mean) / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = x_norm * w + b
    tl.store(out_ptr + row * stride_row + offs, out, mask=mask)


def fused_layer_norm(x, weight, bias, eps=1e-5):
    """Fused LayerNorm Triton kernel."""
    if not TRITON_AVAILABLE or not x.is_cuda:
        return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _layer_norm_kernel[(n_rows,)](x, weight, bias, out, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(orig_shape)
