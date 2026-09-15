# SM100 MXFP8 saved-activation epilogues

Date: 2026-09-15

## Scope

The SM100 training path can now quantize both outputs of grouped FC1 in one
epilogue:

- full-width gate/up preactivation saved as E4M3 plus canonical blocked E8M0;
- half-width gated activation consumed immediately by FC2 in the same format.

DGated backward can load the saved E4M3 preactivation through TMA, apply its
rowwise scale in registers, and produce BF16 dpreactivation. A second variant
also emits the rowwise MXFP8 dpreactivation used by full-MXFP8 FC1 dgrad.

For clustered-M kernels, phantom partner-CTA rows clamp their scale address to
the final valid expert row. Their output stores remain predicated. BF16-C
DGated remains available as the fallback.

## Layout contract

The public preactivation is N-contiguous `(total_M, 2N)` E4M3. At trace time,
adjacent gate/up bytes are recast as one `(total_M, N)` `Int16` C element so
the existing paired DGated epilogue can be reused. Scale storage remains the
canonical `(1, padded_rm, ceil(2N / 128), 32, 4, 4)` E8M0 layout.

Each SM100 epilogue thread owns contiguous pair columns. The implementation
loads one scale for 16 pairs on 128-aligned tile-N layouts and one for eight
pairs on other supported layouts. It applies that scale directly to the C
fragment before the vectorized dgate loop, avoiding a second live fragment and
the associated register-pressure cliff.

Correctness covers tile-N 64/128/192/256, cluster-M 1/2, cluster-N 1/2,
non-aligned and empty expert rows, score scaling, reduction, and fused
dpreactivation quantization.

## Reference and attribution

The design was informed by the Apache-2.0-licensed public SuperSonic-MoE
implementation, specifically
`sonicmoe/quack_utils/_gated_epilogues.py` at commit
`76b4f4f8c37e6f71bfac8fe9e85dd02005d59d8a` in
`PFCCLab/supersonic-moe`.

This implementation lives in Quack's native epilogue interfaces rather than a
Sonic-side monkey patch. It preserves exact non-aligned varlen expert rows and
standard OCP 1x32 scaling. Its scale reuse and shortened scale-fragment
lifetime are Quack-specific optimizations.

## Microbenchmark

Run with cache disabled when changing constexpr layout logic:

```bash
QUACK_CACHE_ENABLED=0 python benchmarks/benchmark_mxfp8_dgated_preact.py \
    --experts 8 --rows-per-expert 256 --n 2048 --k 2048 --tile-n 256
```

The benchmark quantizes C before timing and reports only grouped blockscaled
GEMM plus DGated. It submits a backlog for these sub-100-us kernels and
interleaves BF16-C and MXFP8-C samples. End-to-end Sonic training must include
the forward save, backward load, and removed standalone work before deciding
whether to enable the path by default.
