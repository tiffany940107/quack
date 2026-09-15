# Copyright (c) 2026, Tri Dao.
"""Benchmark BF16-C and MXFP8-C SM100 DGated epilogues.

The timed region contains only the grouped blockscaled GEMM and its DGated
epilogue.  Input quantization is intentionally outside the region.  Multiple
launches are submitted per sample so sub-100-us kernels are measured with a
backlog instead of one host-timed launch at a time.

Example:

    python benchmarks/benchmark_mxfp8_dgated_preact.py \
        --experts 8 --rows-per-expert 256 --n 2048 --k 2048
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from triton.testing import do_bench

import cutlass
from quack.blockscaled import quantize_mxfp8_varlen_m
from quack.blockscaled.utils import create_blockscaled_varlen_m_operands
from quack.epilogue.library import dgated_fp8_preact_mod, dgated_mod


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--rows-per-expert", type=int, default=256)
    parser.add_argument("--n", type=int, default=2048, help="DGated pair count")
    parser.add_argument("--k", type=int, default=2048, help="GEMM reduction width")
    parser.add_argument("--mainloop", choices=("bf16", "mxfp8"), default="mxfp8")
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--backlog", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    return parser.parse_args()


def _backlogged(fn, count: int) -> None:
    for _ in range(count):
        fn()


def main() -> None:
    args = _arguments()
    if torch.cuda.get_device_properties(0).major != 10:
        raise RuntimeError("this benchmark requires an SM100 GPU")
    seqlens = [args.rows_per_expert] * args.experts
    total_m = sum(seqlens)
    if args.mainloop == "mxfp8":
        operands = create_blockscaled_varlen_m_operands(
            args.experts,
            0,
            args.n,
            args.k,
            32,
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            seqlens_m=seqlens,
        )
        _, _, a, qb, sfa, sfb, offsets = operands
        weight = qb.permute(2, 0, 1)
    else:
        offsets = torch.arange(
            0,
            total_m + 1,
            args.rows_per_expert,
            device="cuda",
            dtype=torch.int32,
        )
        a = torch.randn(total_m, args.k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            args.experts,
            args.k,
            args.n,
            device="cuda",
            dtype=torch.bfloat16,
        )
        sfa = sfb = None
    preact_bf16 = (
        torch.randn(total_m, 2 * args.n, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    preact_mx = quantize_mxfp8_varlen_m(preact_bf16, offsets)
    dpreact_bf16 = torch.empty_like(preact_bf16)
    dpreact_fp8c = torch.empty_like(preact_bf16)
    postact_bf16 = torch.empty(
        total_m, args.n, device="cuda", dtype=torch.bfloat16
    )
    postact_fp8c = torch.empty_like(postact_bf16)
    common = {
        "tile_M": args.tile_m,
        "tile_N": args.tile_n,
        "cluster_M": args.cluster_m,
        "cluster_N": args.cluster_n,
        "pingpong": False,
        "persistent": True,
        "is_dynamic_persistent": False,
        "cu_seqlens_m": offsets,
    }
    if args.mainloop == "mxfp8":
        common.update(
            SFA=sfa,
            SFB=sfb,
            bs_format_a="mxfp8_e4m3",
            bs_format_b="mxfp8_e4m3",
        )
    bf16_mod = dgated_mod("swiglu", has_scale=False, has_reduce=False)
    fp8c_mod = dgated_fp8_preact_mod("swiglu", has_scale=False, has_reduce=False)

    def launch_bf16() -> None:
        bf16_mod.gemm(
            a,
            weight,
            dpreact_bf16,
            preact_bf16,
            epi_args={"mAuxOut": postact_bf16},
            **common,
        )

    def launch_fp8c() -> None:
        fp8c_mod.gemm(
            a,
            weight,
            dpreact_fp8c,
            preact_mx.qdata,
            epi_args={"preact_scale": preact_mx.scale, "mAuxOut": postact_fp8c},
            **common,
        )

    launch_bf16()
    launch_fp8c()
    torch.cuda.synchronize()
    samples = {"bf16_c_us": [], "mxfp8_c_us": []}
    orders = ((launch_bf16, launch_fp8c), (launch_fp8c, launch_bf16))
    for round_idx in range(args.rounds):
        for fn in orders[round_idx % len(orders)]:
            elapsed_ms = do_bench(
                lambda: _backlogged(fn, args.backlog),
                warmup=args.warmup_ms,
                rep=args.rep_ms,
            )
            samples[f"{'bf16_c' if fn is launch_bf16 else 'mxfp8_c'}_us"].append(
                elapsed_ms * 1000 / args.backlog
            )
    bf16_us = statistics.median(samples["bf16_c_us"])
    fp8c_us = statistics.median(samples["mxfp8_c_us"])
    print(
        json.dumps(
            {
                "shape": {
                    "experts": args.experts,
                    "rows_per_expert": args.rows_per_expert,
                    "n": args.n,
                    "k": args.k,
                    "tile_m": args.tile_m,
                    "tile_n": args.tile_n,
                    "cluster_m": args.cluster_m,
                    "cluster_n": args.cluster_n,
                    "mainloop": args.mainloop,
                },
                "backlog": args.backlog,
                "samples_us": samples,
                "p50_us": {"bf16_c": bf16_us, "mxfp8_c": fp8c_us},
                "bf16_c_over_mxfp8_c": bf16_us / fp8c_us,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
