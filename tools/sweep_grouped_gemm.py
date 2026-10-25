"""Times every fitting config of the grouped GEMM's launches on this GPU, the records the cost model's
ranking test reads (tests/fixtures/gemm_sweeps/<generation>.json).

    python tools/sweep_grouped_gemm.py --out records.json [--implementations mosaic_gpu]
        [--products forward ...] [--shapes NAME ...] [--sample N]

For each swept shape (the fixture's seven MoE projections) and product (forward, input_gradient,
kernel_gradient, kernel_gradient_fp32), each launch's fitting configs (`Template.fitting` on this
device's facts) are timed as the tuner times them
(`tuning.measured`: compiled in parallel, each run after the L2 is flushed, the device time CUPTI reads)
on the fixture's draw, each checked against float32 host products. The output is the fixture's form:
provenance (the device and its facts, jax and the commit) and one record a shape, product and
implementation, its median times by config repr, the configs that failed apart, its worst error and the
config the cost model picks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
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


def operands(rows: int, depth: int, width: int, groups: int, *, host: bool = False):
    """The rows, each group's matrix, the cotangent and the group sizes, drawn as the fixture's were: on the
    device, or as host arrays where `host` (a process that places them under the GPU's lock,
    `tuning.measured`)."""
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(0)
    placed = (lambda array, dtype: np.asarray(array).astype(dtype)) if host else jnp.asarray
    x = placed(rng.normal(size=(rows, depth)), jnp.bfloat16)
    w = placed(rng.normal(size=(groups, depth, width)) / np.sqrt(depth), jnp.bfloat16)
    dy = placed(rng.normal(size=(rows, width)), jnp.bfloat16)
    sizes = np.floor(rng.dirichlet(np.ones(groups) * 2) * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    return x, w, dy, placed(sizes, np.int32)


def references(x, w, dy, sizes) -> dict:
    """Each product in float32 on the host."""
    import numpy as np

    x, w, dy = (np.asarray(array, np.float32) for array in (x, w, dy))
    bounds = np.concatenate([[0], np.cumsum(np.asarray(sizes))])
    spans = list(itertools.pairwise(bounds.tolist()))
    return {"forward": np.concatenate([x[a:b] @ w[g] for g, (a, b) in enumerate(spans)]),
            "input_gradient": np.concatenate([dy[a:b] @ w[g].T for g, (a, b) in enumerate(spans)]),
            "kernel_gradient": np.stack([x[a:b].T @ dy[a:b] for a, b in spans])}


def _summarized(timings) -> tuple[dict, dict]:
    """A record's median time and failure by config repr, from `tuning.measured`'s timings."""
    times, failures = {}, {}
    for config, timing in timings.items():
        if isinstance(timing, Exception):
            failures[repr(config)] = f"{type(timing).__name__}: {str(timing)[:300]}"
            continue
        times[repr(config)] = statistics.median(timing.evaluation_times_ms)
    return times, failures


def _checking(want, errors: dict[str, float]):
    """A `tuning.measured` outputs callback: each config's worst error against the truth `want`, relative to
    its largest magnitude, written into `errors` by the config's repr."""
    import numpy as np

    def check(config, got):
        got = np.asarray(got, np.float32)
        errors[repr(config)] = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))

    return check


def _record(name: str, product: str, implementation: str, *, host: bool = False):
    """A record's launch, its bound key, its call's arrays and options, from the fixture's own draw (host
    arrays where `host`)."""
    import jax

    from dew.nn.kernels import grouped_matmul as gm

    x, w, dy, sizes = operands(*SHAPES[name], host=host)
    if product == "forward":
        launches, call, options = (gm.MosaicGmm(), gm.Gmm()), (x, w, sizes), {"trans_rhs": False}
    elif product == "input_gradient":
        launches, call, options = (gm.MosaicGmm(), gm.Gmm()), (dy, w, sizes), {"trans_rhs": True}
    else:
        launches, call, options = (gm.MosaicTgmm(), gm.Tgmm()), (x, dy, sizes), {}
    options = {**options, "out_dtype": "float32" if product.endswith("fp32") else "bfloat16",
               "interpret": False}
    launch = next(launch for launch in launches if launch.name == implementation)
    ba = launch.bind(*(jax.ShapeDtypeStruct(array.shape, array.dtype) for array in call), **options)
    return launch, ba, call


def _part(name, product, implementation, sample, part, parts, lock, wanted, results) -> None:
    """One process's share of a record (`_timed_apart`): every `parts`-th of its sampled configs from
    `part`, compiled here and timed under `lock`, checked against the truth at `wanted`; its times, failures
    and errors put on `results`."""
    _shared_memory_policy(parts)
    import numpy as np

    from dew.nn.kernels import generation, tuning

    # Host arrays: their placement on the device is the measurement's, under the lock.
    launch, ba, call = _record(name, product, implementation, host=True)
    want = np.load(wanted, mmap_mode="r")
    target = generation.gpu_target()
    configs = list(sampled(launch, ba, target, sample) or launch.fitting(ba, target))
    errors: dict[str, float] = {}
    timings = tuning.measured(launch, ba, arrays=list(call), outputs=_checking(want, errors),
                              configs=configs[part::parts],
                              gpu=lambda: lock, compilers=max(1, (os.cpu_count() or 1) // parts))
    results.put((part, *_summarized(timings), errors))


GPU_SHARE = 0.9
"""The share of the GPU's memory a sweep's processes divide among them (`_shared_memory_policy`), the rest
left to the CUDA contexts and the driver."""


def _shared_memory_policy(parts: int) -> None:
    """This process's JAX memory policy where `parts` processes beside the sweep's own share one GPU, set
    before JAX initializes: no preallocation, and at most its even split of `GPU_SHARE`, so the first to
    allocate cannot take the memory its siblings need."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{GPU_SHARE / (parts + 1):.3f}"


class PartDied(RuntimeError):
    """A process of `_timed_apart` that exited without its share's results (killed out of memory, a
    compiler's abort): its part and exit code."""


def _timed_apart(name, product, implementation, sample, parts, want, *,
                 part=_part) -> tuple[dict, dict, dict]:
    """A record timed by `parts` processes at once (`--processes`), each compiling its share of the configs
    while another's run holds the GPU, the truth `want` read from a file they share. Results are collected
    while the processes' exits are watched: one that exits without its results (killed, aborted) raises
    `PartDied` once its siblings are stopped and reaped, never leaving the record waiting on it."""
    import contextlib
    import multiprocessing
    import queue
    import tempfile

    import numpy as np

    context = multiprocessing.get_context("spawn")
    lock, results = context.Lock(), context.Queue()
    gathered = {}
    with tempfile.NamedTemporaryFile(suffix=".npy") as file:
        np.save(file.name, np.asarray(want, np.float32))
        workers = [context.Process(target=part, args=(name, product, implementation, sample, index, parts,
                                                      lock, file.name, results)) for index in range(parts)]
        for worker in workers:
            worker.start()
        try:
            while len(gathered) < parts:
                with contextlib.suppress(queue.Empty):
                    # Five seconds of no result: the processes' exits are looked at.
                    index, *share = results.get(timeout=5)
                    gathered[index] = share
                    continue
                dead = [index for index, worker in enumerate(workers)
                        if index not in gathered and not worker.is_alive()]
                if dead:
                    # A result put just before its process exited may still be in the queue's pipe.
                    try:
                        index, *share = results.get(timeout=2)
                        gathered[index] = share
                        continue
                    except queue.Empty:
                        index = dead[0]
                        raise PartDied(f"{name} {product} {implementation}: part {index} of {parts} exited "
                                       f"{workers[index].exitcode} without its results") from None
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                worker.join()
    times, failures, errors = {}, {}, {}
    for part_times, part_failures, part_errors in gathered.values():
        times |= part_times
        failures |= part_failures
        errors |= part_errors
    return times, failures, errors


def sampled(launch, ba, target, count: int):
    """`launch`'s fitting configs at `ba` on `target` a sweep times: every one where `count` is 0, else the
    `count` the cost model estimates fastest, which the pick and its nearest rivals lie in, and `count`
    more spread evenly over the rest by estimate, from the next-fastest to the slowest."""
    fits = launch.fitting(ba, target)
    if not count or len(fits) <= 2 * count:
        return None
    ranked = sorted(fits, key=lambda config: launch.cost(ba, config, target))
    rest = ranked[count:]
    spacing = (len(rest) - 1) / max(count - 1, 1)
    return ranked[:count] + [rest[round(index * spacing)] for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    # The Triton launches are kept, not extended (ops/BIBLE.md): a sweep times the Mosaic mainloops.
    parser.add_argument("--implementations", nargs="+", default=["mosaic_gpu"], choices=["mosaic_gpu"])
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=PRODUCTS)
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    parser.add_argument("--sample", type=int, default=0,
                        help="time the model's N best fitting configs and N more spread over the rest by "
                             "estimate (0: every one)")
    parser.add_argument("--processes", type=int, default=1,
                        help="processes compiling a record's configs at once, each timing its own under one "
                             "lock on the GPU (compiles run beside runs, never two runs at once)")
    args = parser.parse_args()
    if args.processes > 1:
        _shared_memory_policy(args.processes)

    import jax

    from dew.nn.kernels import generation, grouped_matmul as gm, tuning

    target = generation.gpu_target()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # The device's facts, as its launches read them, so the ranking test costs the configs on this device's.
    facts = {name: getattr(target, name) for name in ("multiprocessors", "shared_limit", "clock_hz",
                                                       "memory_bandwidth")}
    provenance = {"job": os.environ.get("JOB", "sweep_grouped_gemm"), "device": jax.devices()[0].device_kind,
                  "jax": jax.__version__, "commit": commit, "generation": target.generation, "facts": facts,
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
                # The launch at its key's shapes, timed on the fixture's own draw (`tuning.measured`).
                specs = [jax.ShapeDtypeStruct(array.shape, array.dtype) for array in call]
                ba = launch.bind(*specs, **options)
                if not launch.eligible(ba, target):
                    continue
                # Each config's outputs from its first timed run (`tuning.measured`), checked as they come,
                # with no compile of their own and one config's outputs held at a time.
                errors: dict[str, float] = {}
                check = _checking(want, errors)

                configs = sampled(launch, ba, target, args.sample)
                if args.processes > 1:
                    try:
                        times, failures, errors = _timed_apart(name, product, launch.name, args.sample,
                                                               args.processes, want)
                    except PartDied as died:
                        # The record's failure written beside the records before it; the sweep stops.
                        records.append({"shape": name, "product": product, "implementation": launch.name,
                                        "times": {}, "failures": {"process": str(died)}, "worst_error": None})
                        args.out.write_text(json.dumps({"provenance": provenance, "records": records},
                                                       indent=1))
                        print(died, file=sys.stderr, flush=True)
                        return 1
                else:
                    timings = tuning.measured(launch, ba, arrays=list(call), outputs=check, configs=configs)
                    times, failures = _summarized(timings)
                record = {"shape": name, "product": product, "implementation": launch.name,
                          "heuristic": repr(launch.estimate(ba, target)[1]), "times": times,
                          "failures": failures, "worst_error": max(errors.values(), default=None)}
                records.append(record)
                print(json.dumps({key: value for key, value in record.items() if key != "times"}), flush=True)
                args.out.write_text(json.dumps({"provenance": provenance, "records": records}, indent=1))
        del truth
    return 0


if __name__ == "__main__":
    sys.exit(main())
