# SM100 MXFP8 variable-length training primitives

This branch adds the Quack building blocks used by Sonic MoE's SM100 MXFP8
training backend. Master tensors remain BF16 or FP32; GEMM operands use OCP
E4M3 values with one E8M0 scale per 32 values.

## Variable-length operands

```python
from quack.blockscaled import quantize_mxfp8_varlen_k, quantize_mxfp8_varlen_m

# x is the expert-sorted (sum(M_e), K) activation.
x_mx = quantize_mxfp8_varlen_m(x, expert_offsets)

# g is (sum(K_e), N). Scaling restarts at every expert boundary.
g_mx = quantize_mxfp8_varlen_k(g, expert_offsets)
```

Values remain densely concatenated. Scale storage inserts a 128-value atom at
each expert boundary, including repeated offsets for empty experts. This makes
the operands directly consumable by `gemm(..., cu_seqlens_m=...)` and
`gemm(..., cu_seqlens_k=...)` without mixing scale groups across experts.

## Fused FC1 epilogues

`gated_preact_quant_mod()` stores the BF16 preactivation required by backward
while directly emitting gated MXFP8 values and blocked scales for FC2.
`gated_quant_mod(..., has_rowvec=True)` is the inference form: it fuses expert
bias and gated activation but does not materialize a preactivation.

The SM100 path supports SwiGLU, GEGLU and ReGLU. Scale-vector and MMA layout
requirements are checked at dispatch; Sonic currently requires hidden and
intermediate dimensions divisible by 128.

## Validation

From the paired workspace root:

```bash
./scripts/run_sm100_container.sh python -m pytest -q \
  quack/tests/test_blockscaled_training.py \
  quack/tests/test_gemm_quant_out.py \
  quack/tests/test_gemm_blockscaled_interface.py
```

The tests cover empty experts, non-aligned segment lengths, both training GEMM
orientations, exact FP8/scale results, fused bias, and complete gated backward.
