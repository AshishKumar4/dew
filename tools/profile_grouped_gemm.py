#!/usr/bin/env python3
"""The grouped GEMM's launches against their references, kernel by kernel, in one capture.

    python tools/profile_grouped_gemm.py --out DIR [--rows NAME ...] [--products P ...] [--arms A ...]

Each scoreboard row's product (`ROWS`; forward, input_gradient, kernel_gradient) runs through each arm
`generation.chosen` names (`mosaic_gpu`, the template's launch; `cutlass` and `triton`, the references),
each arm a jitted function of its own name, warmed, then run three times under one `dew.Profiler`
capture, a region an arm. DIR/kernels.json attributes the capture's kernel_stats report to the arms by
those names: each kernel's time, registers, shared memory, occupancy and tensor-core use. Each arm's
device peak (`peak_bytes_in_use`) is measured apart, in a process of its own (`--peak`), since a
device's peak counts every allocation the process made. Profile with
`XLA_FLAGS=--xla_gpu_enable_command_buffer=` so the kernels are attributed to their ops
(`dew.telemetry.profile.COMMAND_BUFFERS`). The wheel CUTLASS needs is imported where it is importable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROWS = {
    "qwen3-30b-a3b gate_up": (65536, 2048, 1536, 128),
    "qwen3-30b-a3b down": (65536, 768, 2048, 128),
    "kimi-linear gate_up": (32768, 2304, 2048, 256),
    "kimi-linear down": (32768, 1024, 2304, 256),
    "mixtral up": (8192, 4096, 14336, 8),
    "mixtral down": (8192, 14336, 4096, 8),
    "deepseek-v3 down": (65536, 2048, 7168, 256),
}
"""The scoreboard's grouped rows: rows, depth, width and groups (the route benches')."""
PRODUCTS = ("forward", "input_gradient", "kernel_gradient")
REPEATS = 3


def program(name: str, product: str, arm: str):
    """The row's product through `arm`, jitted under a name of its own, and its operands."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from dew.nn.kernels import generation
    from dew.nn.kernels.templates.gemm import template

    rows, depth, width, groups = ROWS[name]
    rng = np.random.default_rng(0)
    sizes = np.floor(rng.dirichlet(np.ones(groups) * 2) * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    sizes = jnp.asarray(sizes)
    x = jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(groups, depth, width)) / np.sqrt(depth), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16)

    def run(x, w, dy):
        with generation.chosen(grouped_matmul=arm):
            if product == "forward":
                return template.gemm(x, w, sizes, out_dtype=jnp.bfloat16)
            if product == "input_gradient":
                return template.gemm(dy, w, sizes, transpose_rhs=True, out_dtype=jnp.bfloat16)
            return template.gemm_t(x, dy, sizes, out_dtype=jnp.bfloat16)

    run.__name__ = run.__qualname__ = label(name, product, arm)
    return jax.jit(run), (x, w, dy)


def label(name: str, product: str, arm: str) -> str:
    """An arm's jitted function's name, which its kernels' op names carry."""
    return "_".join("".join(c if c.isalnum() else "_" for c in part) for part in (name, product, arm))


def peak(name: str, product: str, arm: str) -> dict:
    """The arm's device peak, in this process: its operands, one run's outputs and its scratch."""
    import jax

    fn, operands = program(name, product, arm)
    jax.block_until_ready(fn(*operands))
    stats = jax.devices()[0].memory_stats() or {}
    return {"peak_bytes": stats.get("peak_bytes_in_use"), "operand_bytes": sum(a.nbytes for a in operands)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rows", nargs="*", default=list(ROWS))
    parser.add_argument("--products", nargs="*", default=list(PRODUCTS))
    parser.add_argument("--arms", nargs="*", default=["mosaic_gpu", "cutlass", "triton"])
    parser.add_argument("--peak", nargs=3, metavar=("ROW", "PRODUCT", "ARM"))
    args = parser.parse_args()
    if args.peak:
        print(json.dumps(peak(*args.peak)))
        return

    import jax

    from dew.telemetry.profile import Profiler, report_table

    args.out.mkdir(parents=True, exist_ok=True)
    cases = [(name, product, arm) for name in args.rows for product in args.products for arm in args.arms]
    compiled, failures = {}, {}
    for case in cases:
        try:
            fn, operands = program(*case)
            for _ in range(2):
                jax.block_until_ready(fn(*operands))
            compiled[case] = (fn, operands)
        except Exception as error:  # an arm that does not take the row records why
            failures[label(*case)] = f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"
    profiler = Profiler(args.out / "capture")
    with profiler:
        for case, (fn, operands) in compiled.items():
            with profiler.region(label(*case)):
                for _ in range(REPEATS):
                    jax.block_until_ready(fn(*operands))
    capture = profiler.directory
    columns, kernels = report_table(next(Path(capture).rglob("reports/kernel_stats.json")))
    report = {"device": jax.devices()[0].device_kind, "jax": jax.__version__, "capture": str(capture),
              "columns": columns, "failures": failures, "arms": {}}
    # XProf's kernel_stats table names a kernel's op and total time in these columns (xprof 2.23).
    named = next((column for column in ("op_name", "tf_op_name") if column in columns), None)
    timed = next((column for column in ("total_duration_us", "total_duration") if column in columns), None)
    if named is None or timed is None:
        raise RuntimeError(f"kernel_stats names no op or total time among its columns {columns}")
    for case in compiled:
        mine = [row for row in kernels if label(*case) in str(row.get(named) or "")]
        measured = subprocess.run([sys.executable, __file__, "--peak", *case], capture_output=True, text=True,
                                  env=os.environ)
        report["arms"][label(*case)] = {
            "row": case[0], "product": case[1], "arm": case[2], "kernels": mine,
            "kernel_us": sum(float(row.get(timed) or 0) for row in mine) / REPEATS,
            "peak": json.loads(measured.stdout.strip().splitlines()[-1]) if measured.returncode == 0
            else {"error": measured.stderr[-300:]},
        }
    (args.out / "kernels.json").write_text(json.dumps(report, indent=1))
    for key, arm in report["arms"].items():
        print(key, round(arm["kernel_us"], 1), "us", arm["peak"].get("peak_bytes"), flush=True)


if __name__ == "__main__":
    main()
