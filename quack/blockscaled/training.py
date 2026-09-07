# Copyright (c) 2026, Tri Dao.
"""Block-scaled operand construction for variable-size training GEMMs.

The public dense quantizer cannot express expert boundaries along a GEMM
reduction dimension.  These helpers insert scale-only 128-element padding at
every boundary, matching the SM100/SM120 varlen mainloop contract.  Quantized
values remain densely concatenated, so no padding reaches the GEMM operands.
"""

from __future__ import annotations

import torch

from quack.blockscaled.operand import MXFP8_E4M3, BlockScaledOperand
from quack.blockscaled.quantize import (
    pack_scale_2d_to_blocked_contig,
    to_mx_compiled,
    to_mx_dim0_compiled,
)

_SF_VEC = MXFP8_E4M3.sf_vec_size
_SF_ATOM = 128


def _validate_inputs(x: torch.Tensor, cu_seqlens: torch.Tensor) -> int:
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"x must be a 2-D BF16/FP32 tensor, got {x.dtype} {tuple(x.shape)}")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA tensor")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must be a one-dimensional int32 tensor")
    if cu_seqlens.device != x.device:
        raise ValueError("x and cu_seqlens must be on the same device")
    experts = cu_seqlens.numel() - 1
    if experts <= 0:
        raise ValueError("cu_seqlens must describe at least one segment")
    return experts


def _padded_row_mapping(
    cu_seqlens: torch.Tensor, total_rows: int, experts: int
) -> tuple[torch.Tensor, int]:
    """Map dense segment rows into 128-aligned per-segment scale storage."""
    rows = torch.arange(total_rows, dtype=torch.int64, device=cu_seqlens.device)
    boundaries = cu_seqlens[1:].to(torch.int64)
    expert_ids = torch.bucketize(rows, boundaries, right=True)
    starts = cu_seqlens[:-1].to(torch.int64)
    padded_starts = (
        starts // _SF_ATOM + torch.arange(experts, dtype=torch.int64, device=cu_seqlens.device)
    ) * _SF_ATOM
    destination = rows + (padded_starts - starts)[expert_ids]
    padded_rows = ((total_rows + _SF_ATOM - 1) // _SF_ATOM + experts - 1) * _SF_ATOM
    return destination, padded_rows


def quantize_mxfp8_varlen_m(x: torch.Tensor, cu_seqlens_m: torch.Tensor) -> BlockScaledOperand:
    """Row-quantize ``x=(sum(M_e), K)`` for a variable-M grouped GEMM.

    FP8 values stay dense.  Only scale rows are padded so each expert begins
    on a 128-row scale atom, including when experts are empty or non-aligned.
    """
    experts = _validate_inputs(x, cu_seqlens_m)
    if x.shape[1] % _SF_VEC:
        raise ValueError(f"K={x.shape[1]} must be divisible by {_SF_VEC}")
    if x.shape[0] == 0:
        raise ValueError("a fully empty variable-M operand is not supported")
    qdata, linear_scale = to_mx_compiled(x)
    destination, padded_rows = _padded_row_mapping(cu_seqlens_m, x.shape[0], experts)
    padded_scale = torch.zeros(
        padded_rows,
        linear_scale.shape[1],
        dtype=linear_scale.dtype,
        device=x.device,
    )
    padded_scale.view(torch.uint8).index_copy_(
        0, destination, linear_scale.contiguous().view(torch.uint8)
    )
    scale = pack_scale_2d_to_blocked_contig(padded_scale.unsqueeze(0))
    return BlockScaledOperand.from_parts(qdata, scale, MXFP8_E4M3, orig_dtype=x.dtype)


def quantize_mxfp8_varlen_k(x: torch.Tensor, cu_seqlens_k: torch.Tensor) -> BlockScaledOperand:
    """Segment-quantize ``x=(sum(K_e), N)`` along its first dimension.

    The returned ``quant_dim=-2`` operand is directly usable as the B operand
    of a grouped wgrad GEMM; ``result.mT`` is its corresponding A operand.
    Every expert starts at a 128-element scale atom, so a 32-value MX block can
    never mix values from adjacent experts.
    """
    experts = _validate_inputs(x, cu_seqlens_k)
    if x.shape[0] == 0:
        raise ValueError("a fully empty variable-K operand is not supported")
    destination, padded_rows = _padded_row_mapping(cu_seqlens_k, x.shape[0], experts)
    padded = torch.zeros(padded_rows, x.shape[1], dtype=x.dtype, device=x.device)
    padded.index_copy_(0, destination, x)
    qdata_padded, scale_dim0 = to_mx_dim0_compiled(padded)
    qdata = qdata_padded.index_select(0, destination).contiguous()
    scale = pack_scale_2d_to_blocked_contig(scale_dim0.mT.contiguous().unsqueeze(0))
    return BlockScaledOperand.from_parts(
        qdata,
        scale,
        MXFP8_E4M3,
        orig_dtype=x.dtype,
        quant_dim=-2,
    )


__all__ = ["quantize_mxfp8_varlen_k", "quantize_mxfp8_varlen_m"]
