"""AMD FP8 fused dequant + QKV projection candidate kernel (CODA backend).

Implements the FP8 block-scaled GEMM QKV pre-projection for MI300X (gfx942),
targeting the shape used by FP8 LLM inference:
  M = batch, K = hidden = 8192, N = q_out + k_out + v_out = 8192 + 1024 + 1024 = 10240
  (num_heads=64, num_kv_heads=8 GQA, head_dim=128).

The frozen Stage0 baseline is ``aiter.gemm_a8w8_blockscale``, which dispatches
to the fixed ``gemm_a8w8_blockscale_ck`` kernel (~32.7 us/call at batch=8)
because no tuned config exists for this shape in the blockscale tuned-CSV.
Sweeping ``gemm_a8w8_blockscale_tune`` over kernelIds (0..31) on gfx942
(cu_num=304) shows ``kernelId=8`` — tile config
``1x128x128_256x16x64x256_..._1x1_intrawave_v1`` — is the fastest valid
variant at ~26.3 us/call (0.807x baseline), with exact parity
(max abs err 0.0 vs the reference ck kernel on the harness inputs).

Note: the runtime ``gemm_a8w8_blockscale``/``gemm_a8w8_blockscale_ck`` path
does NOT consume ``kernelId`` from the tuned-CSV (it only selects ck vs
cktile, and cktile is ~6x slower here).  The tuned tile is therefore only
reachable through the ``gemm_a8w8_blockscale_tune`` entry point, which this
module calls directly.  This is the CODA AMD candidate path requested by
amdpilot-org/coda-kernels#7.

Alternatives ruled out by empirical sweep on gfx942 (cu_num=304):
  * ``gemm_a8w8_blockscale_cktile_tune`` — all kernelIds slower (best 0.409 ms
    vs 0.026 ms for ck kernelId=8).
  * ``gemm_a8w8_blockscale_bpreshuffle_ck`` with ``shuffle_weight(w, (16,16))``
    — parity-correct but slower (0.886x baseline vs 0.656x for kernelId=8).
    Layouts (32,16)/(32,32) fail parity; (16,32) also slower.
  * splitK sweep (0,1,2,3,4,8,16) — no gain (K=8192 already well-parallelized).
  * CUDA graph capture — counterproductive (0.955x; replay overhead exceeds
    the ~26 us kernel time).
"""
import sys

for p in ("/sgl-workspace/aiter",):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_tune

# Tuned kernelId for the QKV pre-projection shape (M=8, N=10240, K=8192) on
# gfx942 / MI300X (cu_num=304).  Selected by sweeping kernelIds 0..31 with
# gemm_a8w8_blockscale_tune; kernelId=8 is the fastest valid config at
# ~26.3 us vs ~32.7 us for the default runtime ck kernel.  Parity: max abs
# err 0.0 vs the reference ck kernel on the harness inputs (ones scales).
# splitK does not improve this shape (K=8192 is already well parallelized).
_TUNED_KERNEL_ID = 8
_TUNED_SPLITK = 0


def fp8_qkv_projection_fused(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    kernel_id: int = _TUNED_KERNEL_ID,
    splitK: int = _TUNED_SPLITK,
) -> torch.Tensor:
    """FP8 block-scaled GEMM QKV projection via the tuned CK kernel.

    Fuses the FP8 dequant (per-128 block scale) into the GEMM, as the baseline
    does, but selects the tuned tile config (kernelId=8) instead of the
    default runtime kernel.

    Args:
        x:        (M, K)        FP8 (e4m3fnuz) activations
        w:        (N, K)        FP8 (e4m3fnuz) weights, N = q_out + 2*kv_out
        x_scale:  (M, K/128)    float32 activation block scales
        w_scale:  (N/128, K/128) float32 weight block scales
        dtype:    output dtype (default bfloat16)
        kernel_id: tuned CK kernel id (default 8 for the QKV shape)
        splitK:    K-split factor (default 0; not beneficial for this shape)

    Returns:
        (M, N) output tensor in ``dtype``.
    """
    m = x.shape[0]
    n = w.shape[0]
    out = torch.empty(m, n, dtype=dtype, device=x.device)
    gemm_a8w8_blockscale_tune(
        x, w, x_scale, w_scale, out, kernelId=kernel_id, splitK=splitK
    )
    return out


def fp8_qkv_projection_split(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype = torch.bfloat16,
):
    """FP8 QKV projection returning the split (Q, K, V) tensors.

    Q: (M, num_heads * head_dim), K/V: (M, num_kv_heads * head_dim),
    where head_dim = x.shape[1] // num_heads.
    """
    head_dim = x.shape[1] // num_heads
    q_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim
    y = fp8_qkv_projection_fused(x, w, x_scale, w_scale, dtype)
    q, k, v = torch.split(y, [q_out, kv_out, kv_out], dim=1)
    return q, k, v
