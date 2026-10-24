"""Times every fitting config of the grouped GEMM's launches on this GPU, the records the cost model's
ranking test reads (tests/fixtures/gemm_sweeps/<generation>.json).

    python tools/sweep_grouped_gemm.py --out records.json [--implementations mosaic_gpu triton]
        [--products forward ...] [--shapes NAME ...]

For each swept shape (the fixture's seven MoE projections) and product (forward, input_gradient,
kernel_gradient, kernel_gradient_fp32), each launch's fitting configs (`Template.fitting` on this device's
facts) are compiled in parallel and timed on the device (CUPTI, the median of five), each checked against
float32 host products. A config that fails to compile or run records why in place of its time. The output
is the fixture's form: provenance (the device, jax and the commit) and one record a shape, product and
implementation, its times by config repr, its worst error and the config the cost model picks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

SHAPES = {
    # rows (tokens x top-k), width in, width out, experts
    "qwen3-30b-a3b gate_up": (65536, 2048, 1536, 128),
    "qwen3-30b-a3b down": (65536, 768, 2048, 128),
    "kimi-linear gate_up": (32768, 2304, 2048, 256),
    "targets-comment": (8192, 2048, 768, 64),
    "kimi-linear down": (32768, 1024, 2304, 256),
    "mixtral up": (8192, 4096, 14336, 8),
    "mixtral down": (8192, 14336, 4096, 8),
}
PRODUCTS = ("forward", "input_gradient", "kernel_gradient", "kernel_gradient_fp32")
REPEATS = 5


def operands(rows: int, depth: int, width: int, groups: int):
    """The rows, each group's matrix, the cotangent and the group sizes, drawn as the fixture's were."""
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(groups, depth, width)) / np.sqrt(depth), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16)
    sizes = np.floor(rng.dirichlet(np.ones(groups) * 2) * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    return x, w, dy, jnp.asarray(sizes)


def references(x, w, dy, sizes) -> dict:
    """Each product in float32 on the host."""
    import numpy as np

    x, w, dy = (np.asarray(array, np.float32) for array in (x, w, dy))
    bounds = np.concatenate([[0], np.cumsum(np.asarray(sizes))])
    spans = list(itertools.pairwise(bounds.tolist()))
    return {"forward": np.concatenate([x[a:b] @ w[g] for g, (a, b) in enumerate(spans)]),
            "input_gradient": np.concatenate([dy[a:b] @ w[g].T for g, (a, b) in enumerate(spans)]),
            "kernel_gradient": np.stack([x[a:b].T @ dy[a:b] for a, b in spans])}


def device_ms(executable, *args) -> float:
    """The median over `REPEATS` runs of the executable's kernels' device time, in ms (CUPTI)."""
    from jax.experimental.mosaic.gpu import profiler

    times = []
    for _ in range(REPEATS):
        _, elapsed = profiler.measure(executable)(*args)
        if elapsed is None:
            raise RuntimeError("CUPTI timed no kernel of the executable")
        times.append(elapsed)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--implementations", nargs="+", default=["mosaic_gpu", "triton"])
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=PRODUCTS)
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    args = parser.parse_args()

    import jax
    import numpy as np

    from dew.nn.kernels import generation, grouped_matmul as gm

    target = generation.gpu_target()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    provenance = {"job": os.environ.get("JOB", "sweep_grouped_gemm"), "device": jax.devices()[0].device_kind,
                  "jax": jax.__version__, "commit": commit,
                  "bench": "tools/sweep_grouped_gemm.py: device ms (CUPTI, median of 5), bf16 operands"}
    records = []
    for name in args.shapes:
        x, w, dy, sizes = operands(*SHAPES[name])
        truth = references(x, w, dy, sizes)
        for product in args.products:
            if product == "forward":
                launches, call, options = (gm.MosaicGmm(), gm.Gmm()), (x, w, sizes), {"trans_rhs": False}
            elif product == "input_gradient":
                launches, call, options = (gm.MosaicGmm(), gm.Gmm()), (dy, w, sizes), {"trans_rhs": True}
            else:
                launches, call, options = (gm.MosaicTgmm(), gm.Tgmm()), (x, dy, sizes), {}
            options = {**options, "out_dtype": "float32" if product.endswith("fp32") else "bfloat16",
                       "interpret": False}
            want = truth[product.removesuffix("_fp32")]
            for launch in launches:
                if launch.name not in args.implementations:
                    continue
                ba = launch.bind(*call, **options)
                if not launch.eligible(ba, target):
                    continue
                fits = launch.fitting(ba, target)

                def compiled(config, launch=launch, call=call, options=options):
                    configured = launch.replace(config=config)
                    return jax.jit(lambda *a: configured(*a, **options)).lower(*call).compile()

                with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
                    futures = {config: pool.submit(compiled, config) for config in fits}
                times, errors = {}, {}
                for config, future in futures.items():
                    try:
                        executable = future.result()
                        times[repr(config)] = device_ms(executable, *call)
                        got = np.asarray(executable(*call), np.float32)
                        errors[repr(config)] = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
                    except Exception as failed:  # a config that fails to compile or run records why
                        times[repr(config)] = "".join(traceback.format_exception_only(failed))[-300:]
                record = {"shape": name, "product": product, "implementation": launch.name,
                          "heuristic": repr(launch.estimate(ba, target)[1]), "times": times,
                          "worst_error": max(errors.values(), default=None)}
                records.append(record)
                print(json.dumps({key: value for key, value in record.items() if key != "times"}), flush=True)
                args.out.write_text(json.dumps({"provenance": provenance, "records": records}, indent=1))
        del truth


if __name__ == "__main__":
    sys.exit(main())
