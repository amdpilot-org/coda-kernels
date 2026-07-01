"""AMD MXFP4 GEMM + SwiGLU fused candidate path.

Fuses gate+up GEMMs into a single AITER gemm_a16wfp4 call by concatenating
weights along the N dimension, then applies SwiGLU activation.
Uses per-tensor-identity cache to avoid repeated concatenation overhead.
Optionally uses CUDA graph replay to amortize CPU launch overhead.
Shape-tuned for (M=8, N=8192, K=8192) concat-GEMM on gfx950 MI355X.
"""
import sys
import warnings

for p in ("/sgl-workspace/aiter",):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import aiter.ops.triton.gemm_a16wfp4 as _gemm_mod

_gemm_a16wfp4 = getattr(_gemm_mod, "gemm_a16wfp4")

# Tuned GEMM config for M=8, N=8192 (concat), K=8192 on gfx950 MI355X.
# Swept BLOCK_SIZE_N={32,64,128}, warps={2,4,8}, BK={256,512}.
# BN=64 with BM=4, BK=512, warps=4 is ~20% faster than default BN=128.
_TUNED_CONFIG = {
    "BLOCK_SIZE_M": 4,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg",
    "NUM_KSPLIT": 1,
}

# Module-level cache for concatenated weights/scales.
_CONCAT_CACHE: dict = {}

# CUDA graph cache: graph_key -> CUDAGraph
_GRAPH_CACHE: dict = {}

# CUDA graph result cache: graph_key -> captured result tensor
_RESULT_CACHE: dict = {}

# Stable input-buffer cache: (shape, dtype, device) -> buffer.
# The captured CUDA graph reads from this buffer's fixed address; each call
# copies the caller's `a` into it so the graph always sees current data.
_INPUT_BUF_CACHE: dict = {}


def _get_cached_concat(b_gate, b_up, scale_gate, scale_up):
    key = (id(b_gate), id(b_up), id(scale_gate), id(scale_up))
    if key not in _CONCAT_CACHE:
        b = torch.cat([b_gate, b_up], dim=0)
        scale = torch.cat([scale_gate, scale_up], dim=0)
        _CONCAT_CACHE[key] = (b, scale)
    return _CONCAT_CACHE[key]


def _get_input_buf(a):
    """Return a persistent buffer matching ``a``'s shape/dtype/device.

    The CUDA graph captures this buffer's address; copying each caller's ``a``
    into it makes replay correct regardless of the caller's allocation pattern
    (fresh tensor each call, id recycling, in-place updates, etc.).
    """
    key = (a.shape, a.dtype, str(a.device))
    buf = _INPUT_BUF_CACHE.get(key)
    if buf is None:
        buf = torch.empty_like(a)
        _INPUT_BUF_CACHE[key] = buf
    return buf


def _normal_fused(a, b, scale, N, dtype):
    """Normal (non-graph) fused execution."""
    out = _gemm_a16wfp4(a, b, scale, atomic_add=False, dtype=dtype, config=_TUNED_CONFIG)
    gate = out[:, :N]
    up = out[:, N:]
    return torch.nn.functional.silu(gate) * up


def _capture_and_cache(a, b, scale, N, dtype, graph_key):
    """Warm up kernels, capture a CUDA graph, and cache it for replay."""
    # Warm up Triton kernel compilation / lazy state.
    for _ in range(3):
        o = _gemm_a16wfp4(a, b, scale, atomic_add=False, dtype=dtype, config=_TUNED_CONFIG)
        _ = torch.nn.functional.silu(o[:, :N]) * o[:, N:]

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o_g = _gemm_a16wfp4(a, b, scale, atomic_add=False, dtype=dtype, config=_TUNED_CONFIG)
        result_g = torch.nn.functional.silu(o_g[:, :N]) * o_g[:, N:]

    # Validate with replay (first capture execution can be unreliable).
    g.replay()
    torch.cuda.synchronize()

    _GRAPH_CACHE[graph_key] = g
    _RESULT_CACHE[graph_key] = result_g
    return result_g


def gemm_a16wfp4_swiglu_fused(
    a: torch.Tensor,
    b_gate: torch.Tensor,
    scale_gate: torch.Tensor,
    b_up: torch.Tensor,
    scale_up: torch.Tensor,
    dtype=torch.bfloat16,
):
    """Fused MXFP4 GEMM(gate+up) + SwiGLU via single concat-GEMM.

    Captures a CUDA graph keyed on the *stable* input buffer (not the caller's
    ``a``), then replays it on subsequent calls to avoid CPU launch overhead.
    Each call copies ``a`` into the buffer so the graph always reads current
    data; the returned tensor is cloned so callers may keep multiple results.
    Falls back to normal execution if graph capture fails.

    Args:
        a:        (M, K)   BF16 activations
        b_gate:   (N, K/2) packed MXFP4 gate weights
        scale_gate: (N, K/32) e8m0 gate scales
        b_up:     (N, K/2) packed MXFP4 up weights
        scale_up:   (N, K/32) e8m0 up scales
        dtype:    output dtype (default bfloat16)

    Returns:
        (M, N) BF16 SwiGLU output
    """
    b, scale = _get_cached_concat(b_gate, b_up, scale_gate, scale_up)
    N = b_gate.shape[0]

    # Copy activations into a stable buffer so the captured graph always reads
    # from the same address.  Keying on id(a) (the previous approach) was
    # unsafe: a fresh `a` each call missed the cache and re-captured every
    # time, and a recycled id(a) replayed against freed memory.
    a_buf = _get_input_buf(a)
    a_buf.copy_(a)
    graph_key = (a_buf.shape, a_buf.dtype, str(a_buf.device), id(b), id(scale), dtype)

    # Fast path: graph replay.
    if graph_key in _GRAPH_CACHE:
        _GRAPH_CACHE[graph_key].replay()
        return _RESULT_CACHE[graph_key].clone()

    # New graph key: capture (warmup + capture + validate), then clone result.
    try:
        _capture_and_cache(a_buf, b, scale, N, dtype, graph_key)
        return _RESULT_CACHE[graph_key].clone()
    except Exception as exc:
        warnings.warn(f"CUDA graph capture failed ({exc}); falling back to normal execution.")
        return _normal_fused(a_buf, b, scale, N, dtype)


def fused_mxfp4_gemm_swiglu(
    a: torch.Tensor,
    b_gate: torch.Tensor,
    scale_gate: torch.Tensor,
    b_up: torch.Tensor,
    scale_up: torch.Tensor,
    dtype=torch.bfloat16,
):
    """Alias for gemm_a16wfp4_swiglu_fused."""
    return gemm_a16wfp4_swiglu_fused(a, b_gate, scale_gate, b_up, scale_up, dtype)
