# Copyright (c) 2026, Tri Dao.
"""MXFP8 quantization layouts used by variable-size training GEMMs."""

import itertools

import pytest
import torch

from quack.blockscaled import (
    quantize_mxfp8_varlen_k,
    quantize_mxfp8_varlen_m,
    unpack_scale_blocked_to_2d,
)
from quack.gemm_interface import gemm, gemm_dact


def _skip_if_not_sm100():
    if torch.cuda.get_device_properties(0).major < 10:
        pytest.skip("SM100+ required")


def _active_varlen_m_scales(scale, cu_seqlens, sf_k):
    experts = cu_seqlens.numel() - 1
    total = int(cu_seqlens[-1])
    padded_rows = scale.shape[1] * 128
    linear = unpack_scale_blocked_to_2d(scale, padded_rows, sf_k)[0]
    cu = cu_seqlens.tolist()
    return torch.cat(
        [
            linear[(cu[e] // 128 + e) * 128 : (cu[e] // 128 + e) * 128 + cu[e + 1] - cu[e]]
            for e in range(experts)
        ]
    ).reshape(total, sf_k)


def _dequantize_varlen_k(operand, cu_seqlens):
    """Return one dense dequantized ``(K_e, N)`` tensor per segment."""
    total_sf_k = operand.scale.shape[2] * 4
    scales = unpack_scale_blocked_to_2d(operand.scale, operand.shape[1], total_sf_k)[0].float()
    cu = cu_seqlens.tolist()
    result = []
    for expert, (start, end) in enumerate(itertools.pairwise(cu)):
        length = end - start
        sf_start = (start // 128 + expert) * 4
        sf_count = (length + 31) // 32
        scale = scales[:, sf_start : sf_start + sf_count].repeat_interleave(32, dim=-1)
        values = operand.qdata[start:end].float()
        result.append(values * scale[:, :length].T)
    return result


def test_quantize_mxfp8_varlen_m_matches_rowwise_reference():
    _skip_if_not_sm100()
    from quack.blockscaled.quantize import to_mx_compiled

    torch.manual_seed(0)
    cu = torch.tensor([0, 0, 1, 129, 256, 385], dtype=torch.int32, device="cuda")
    x = torch.randn(385, 256, dtype=torch.bfloat16, device="cuda")
    result = quantize_mxfp8_varlen_m(x, cu)
    q_ref, sf_ref = to_mx_compiled(x)

    assert result.quant_dim == -1
    assert torch.equal(result.qdata, q_ref)
    active_sf = _active_varlen_m_scales(result.scale, cu, x.shape[1] // 32)
    assert torch.equal(active_sf.view(torch.uint8), sf_ref.view(torch.uint8))


def test_quantize_mxfp8_varlen_k_training_gemm():
    """Segmented dim-0 casts feed the wgrad orientation without allowing an
    MX scale block to cross an expert boundary, including empty experts and
    segment lengths that are not multiples of 32."""
    _skip_if_not_sm100()
    torch.manual_seed(1)
    cu = torch.tensor([0, 0, 33, 97, 97, 226], dtype=torch.int32, device="cuda")
    lhs_hp = torch.randn(226, 128, dtype=torch.bfloat16, device="cuda")
    rhs_hp = torch.randn(226, 256, dtype=torch.bfloat16, device="cuda")
    lhs = quantize_mxfp8_varlen_k(lhs_hp, cu)
    rhs = quantize_mxfp8_varlen_k(rhs_hp, cu)

    out = gemm(lhs.mT, rhs, cu_seqlens_k=cu, tuned=False)
    lhs_ref = _dequantize_varlen_k(lhs, cu)
    rhs_ref = _dequantize_varlen_k(rhs, cu)
    ref = torch.stack(
        [
            a.T @ b if a.shape[0] else torch.zeros(128, 256, dtype=torch.float32, device="cuda")
            for a, b in zip(lhs_ref, rhs_ref)
        ]
    )
    assert out.shape == (cu.numel() - 1, 128, 256)
    torch.testing.assert_close(out.float(), ref.float(), rtol=5e-3, atol=5e-3)
    assert torch.count_nonzero(out[0]) == 0
    assert torch.count_nonzero(out[3]) == 0


def test_blockscaled_varlen_m_dgated_training():
    """The MXFP8 FC2 dgrad path composes varlen-M, SwiGLU backward,
    per-token router scaling, and the router-score reduction."""
    _skip_if_not_sm100()
    torch.manual_seed(2)
    cu = torch.tensor([0, 1, 34, 34, 163], dtype=torch.int32, device="cuda")
    total, hidden, intermediate = 163, 256, 128
    dout_hp = torch.randn(total, hidden, dtype=torch.bfloat16, device="cuda") * 0.05
    weight_hp = torch.randn(4, hidden, intermediate, dtype=torch.bfloat16, device="cuda") * 0.05
    preact = torch.randn(total, 2 * intermediate, dtype=torch.bfloat16, device="cuda")
    score = torch.rand(total, dtype=torch.float32, device="cuda")
    dout = quantize_mxfp8_varlen_m(dout_hp, cu)
    weight = type(dout).quantize(weight_hp, "mxfp8", dim=-2)

    dh, postact, dscore = gemm_dact(
        dout,
        weight,
        PreAct=preact,
        activation="swiglu",
        colvec_scale=score,
        colvec_reduce=True,
        cu_seqlens_m=cu,
        dynamic_scheduler=False,
        tuned=False,
    )

    active_sf = _active_varlen_m_scales(dout.scale, cu, hidden // 32).float()
    dout_dq = dout.qdata.float() * active_sf.repeat_interleave(32, dim=-1)
    weight_dq = weight.dequantize(torch.float32)
    offsets = cu.tolist()
    acc = torch.cat([dout_dq[offsets[e] : offsets[e + 1]] @ weight_dq[e] for e in range(4)])
    preact_ref = preact.float().detach().requires_grad_()
    act_ref = torch.nn.functional.silu(preact_ref[:, 0::2]) * preact_ref[:, 1::2]
    (dh_ref,) = torch.autograd.grad(act_ref, preact_ref, acc * score[:, None])
    postact_ref = act_ref * score[:, None]
    dscore_ref = (act_ref * acc).sum(dim=-1)
    torch.testing.assert_close(dh.float(), dh_ref, rtol=8e-3, atol=2e-2)
    torch.testing.assert_close(postact.float(), postact_ref, rtol=8e-3, atol=2e-2)
    torch.testing.assert_close(dscore.float(), dscore_ref, rtol=8e-3, atol=2e-2)
