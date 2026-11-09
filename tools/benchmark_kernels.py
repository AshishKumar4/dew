"""Measure the per-generation kernel choices, one measurement per process.

Each mode prints one JSON line. `peak_bytes_in_use` is a process high-water
mark, so a row is one process.

- `projection`: `dew.nn.moe.expert_projection` forward and forward plus
  backward, at lm-moe's shape (8192 rows, 8 experts, bf16 compute, Dirichlet
  routing), against a float64 oracle of the rounded operands; with
  `--against`, the implementation against another in role-swapped pairs
  (`dew.telemetry.benchmark.compared`), each call at the next of a pool of
  routings, with its own A/A, which is the comparison to read across arms.
- `adam`: one AdamW update over the lm-dense parameter tree, fp32, bf16 or
  8-bit moment storage (`OptimConfig.state_dtype`).
- `step`: a whole training step of `lm-dense` (359.8M) or `lm-moe` (321.8M,
  8 experts top-2), through `tools/benchmark_step.py`'s trainer, with the
  grouped matmul, the state dtype and the vocabulary head chosen.

The docs/performance.md section "Kernel choices per generation" was measured
with these commands, for example:

    PYTHONPATH=src python tools/benchmark_kernels.py projection --implementation pallas
    PYTHONPATH=src python tools/benchmark_kernels.py projection --implementation mosaic_gpu \\
        --against cutlass
    PYTHONPATH=src python tools/benchmark_kernels.py adam --state-dtype bfloat16
    PYTHONPATH=src python tools/benchmark_kernels.py step --path lm-moe --batch 4 \\
        --implementation auto --state-dtype bfloat16
"""

import argparse
import itertools
import json

import grouped_oracle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from benchmark_cases import LM_DENSE, LM_MOE, Case
from benchmark_models import build_trainer, compiled_step, timed_steps

from dew.nn.kernels.options import program
from dew.nn.moe import expert_projection
from dew.optim import OptimConfig
from dew.telemetry.benchmark import compared, samples
from dew.telemetry.profile import Profiler


def timed(function, arguments, repeats: int) -> dict[str, float]:
    """Mean, minimum and median wall time of `function`, each call synced."""
    measured = np.asarray(samples(function, *arguments, warmup=5, repeats=repeats)) * 1e3
    return {"mean_ms": float(np.mean(measured)), "min_ms": float(np.min(measured)),
            "median_ms": float(np.median(measured))}


def peak_bytes() -> int | None:
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use")


ROUTINGS = 16
"""The routings a paired comparison's calls take in turn, one a call: a grouped product's time moved several
percent from one routing seed to another (row 3616), so each arm meets the same ones in the same order."""


def routed(rng: np.random.Generator, rows: int, experts: int) -> np.ndarray:
    """A Dirichlet routing of `rows` rows over `experts` experts, its group sizes summing to `rows`."""
    sizes = np.floor(rng.dirichlet(np.ones(experts)) * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    return sizes


def projections(implementation: str) -> tuple:
    """`implementation`'s expert projection compiled as the forward alone and as the forward plus backward
    of the product's sum against a cotangent, each called as `(x, kernel, group_sizes, ...)`."""
    # Arrays are arguments, not closures: `program` keeps each function it compiled.
    def project(x, kernel, group_sizes):
        return jnp.asarray(expert_projection(x, kernel, group_sizes, jnp.bfloat16, implementation, None))

    def loss(x, kernel, group_sizes, cotangent):
        return jnp.sum(project(x, kernel, group_sizes).astype(jnp.float32) * cotangent.astype(jnp.float32))

    return program(project), program(jax.value_and_grad(loss, argnums=(0, 1)))


def paired(args: argparse.Namespace, x, kernel, cotangent) -> dict[str, object]:
    """`args.implementation` against `args.against` in `args.pairs` role-swapped pairs (`compared`) for the
    forward and the forward plus backward, every call at the next of `ROUTINGS` routings: the median ratio of
    the first's seconds to the second's, its interval, and the second's against itself (its A/A). Its cache
    state: each arm compiled and warmed before its pairs, and no flush between calls, so a call reads
    whatever of its operands the call before it left in L2."""
    pool = [jnp.asarray(routed(np.random.default_rng(seed), args.rows, args.experts))
            for seed in range(ROUTINGS)]
    arms = [projections(implementation) for implementation in (args.implementation, args.against)]
    result: dict[str, object] = {"against": args.against, "routings": ROUTINGS,
                                 "cache": {"compile": "warm", "l2": "unflushed"}}
    for index, (name, extra) in enumerate((("forward", ()), ("forward_backward", (cotangent,)))):
        def drawn(function, extra=extra):
            draws = itertools.cycle(pool)
            return lambda: function(x, kernel, next(draws), *extra)

        comparison = compared(drawn(arms[0][index]), drawn(arms[1][index]), pairs=args.pairs)
        result[name] = {"ratio": comparison.ratio, "interval": comparison.interval, "alone": comparison.alone}
    return result


def projection(args: argparse.Namespace) -> dict[str, object]:
    rows, experts = args.rows, args.experts
    width, features = (args.width, args.features) if args.shape == "up" else (args.features, args.width)
    rng = np.random.default_rng(0)
    sizes = routed(rng, rows, experts)
    x = jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16)
    kernel = jnp.asarray(rng.normal(size=(experts, width, features)) / np.sqrt(width), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=(rows, features)), jnp.bfloat16)
    group_sizes = jnp.asarray(sizes)
    forward, both = projections(args.implementation)
    output = np.asarray(forward(x, kernel, group_sizes), np.float64)
    _, (dx, dk) = both(x, kernel, group_sizes, cotangent)
    oracle, dx_oracle, dk_oracle = grouped_oracle.products(x, kernel.astype(jnp.bfloat16), cotangent, sizes)

    def relative(value, reference):
        return float(np.abs(np.asarray(value, np.float64) - reference).max() / np.abs(reference).max())

    result = {"implementation": args.implementation, "shape": [rows, width, features, experts],
              "groups": sizes.tolist(), "forward": timed(forward, (x, kernel, group_sizes), args.repeats),
              "forward_backward": timed(both, (x, kernel, group_sizes, cotangent), args.repeats),
              "forward_error": relative(output, oracle), "input_gradient_error": relative(dx, dx_oracle),
              "kernel_gradient_error": relative(dk, dk_oracle)}
    if args.against is not None:
        result["paired"] = paired(args, x, kernel, cotangent)
    return result


def lm_dense_tree() -> dict[str, jax.Array]:
    """The lm-dense model's parameters as one flat tree, initialized."""
    case = case_for("lm-dense", 1, None)
    model = build_trainer(case).objective.model
    variables = jax.eval_shape(lambda: model.init(jax.random.key(0),
                                                  jnp.zeros((1, 8), jnp.int32)))
    leaves = jax.tree.leaves(variables["params"])
    keys = jax.random.split(jax.random.key(0), len(leaves))
    return {str(index): jax.random.normal(key, leaf.shape, jnp.float32) * 0.02
            for index, (key, leaf) in enumerate(zip(keys, leaves, strict=True))}


def adam(args: argparse.Namespace) -> dict[str, object]:
    params = lm_dense_tree()
    keys = jax.random.split(jax.random.key(1), len(params))
    grads = {name: jax.random.normal(key, leaf.shape, jnp.float32) * 1e-3
             for key, (name, leaf) in zip(keys, params.items(), strict=True)}
    solver = OptimConfig(optimizer="adamw", learning_rate=1e-4, weight_decay=0.1,
                                         state_dtype=args.state_dtype).build(1000)

    def update(params, state, grads):
        updates, state = solver.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    run = jax.jit(update, donate_argnums=(0, 1))
    state = solver.init(params)

    def step():
        # Each call donates the last call's parameters and state.
        nonlocal params, state
        params, state = run(params, state, grads)
        return params, state

    updates = [seconds * 1e3 for seconds in samples(step, warmup=3, repeats=args.repeats)]
    return {"state_dtype": args.state_dtype, "params": sum(leaf.size for leaf in params.values()),
            "update_mean_ms": float(np.mean(updates)), "update_min_ms": float(np.min(updates)),
            "state_bytes": sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(state))}


def case_for(path: str, batch: int, args: argparse.Namespace | None) -> Case:
    remat = None if args is None else args.remat
    if path == "dit":
        # DiT-L/2's width and depth on 64x64 inputs, 1024 tokens a sample.
        config: dict[str, object] = {"patch_size": 2, "emb_features": 1024, "num_layers": 24,
                                     "num_heads": 16, "mlp_ratio": 4,
                                     "remat": False if remat in (None, "none") else remat}
        return Case("simple_dit", config, dtype="bfloat16", batch_size=batch,
                                   image_size=64)
    if path == "lm-dense":
        config = dict(LM_DENSE)
    else:
        config = dict(LM_MOE)
        if args is not None:
            config["mixture"] = {**LM_MOE["mixture"], "implementation": args.implementation}
    config["remat"] = None if remat in (None, "none") else remat
    # The decoder's whole context, a window a row.
    return Case("causal_transformer", config, dtype="bfloat16",
                               batch_size=batch, seq_len=config["max_seq_len"])


def step(args: argparse.Namespace) -> dict[str, object]:
    case = case_for(args.path, args.batch, args)
    trainer = build_trainer(case, optimizer=OptimConfig(
        optimizer="adam", learning_rate=1e-4, state_dtype=args.state_dtype).build(1000))
    with compiled_step(case, trainer) as compiled:
        state, loss, _, _ = timed_steps(compiled.step, compiled.state, 5)
        losses = [float(loss)]
        state, loss, _, seconds = timed_steps(compiled.step, state, args.steps)
        step_ms = seconds / args.steps * 1e3
        if args.trace:
            # The trace alone, which XProf opens as it is.
            with Profiler(args.trace, reports=False):
                state, loss, _, _ = timed_steps(compiled.step, state, 3)
        losses.append(float(loss))
        compile_seconds = compiled.compile_seconds
    return {"path": args.path, "batch": args.batch, "implementation": args.implementation,
            "state_dtype": args.state_dtype, "remat": args.remat, "ms_per_step": step_ms,
            "compile_seconds": compile_seconds, "loss_after_warmup": losses[0],
            "loss_last": losses[-1]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    project = modes.add_parser("projection")
    project.add_argument("--implementation", default="auto")
    project.add_argument("--shape", choices=("up", "down"), default="up")
    project.add_argument("--rows", type=int, default=8192, help="routed tokens, each top-k choice a row")
    project.add_argument("--experts", type=int, default=8)
    project.add_argument("--width", type=int, default=768, help="the model width an up projection reads")
    project.add_argument("--features", type=int, default=2048, help="the expert width it writes")
    project.add_argument("--repeats", type=int, default=50)
    project.add_argument("--against", default=None,
                         help="an implementation to compare against in role-swapped pairs")
    project.add_argument("--pairs", type=int, default=400, help="the paired comparison's pairs")
    update = modes.add_parser("adam")
    update.add_argument("--state-dtype", choices=("float32", "bfloat16", "int8"), default="float32")
    update.add_argument("--repeats", type=int, default=30)
    run = modes.add_parser("step")
    run.add_argument("--path", choices=("lm-dense", "lm-moe", "dit"), required=True)
    run.add_argument("--remat", default="none",
                     help="none, full, a REMAT_POLICIES name (decoders) or dots (DiT)")
    run.add_argument("--batch", type=int, required=True)
    run.add_argument("--implementation", default="auto")
    run.add_argument("--state-dtype", choices=("float32", "bfloat16", "int8"), default="float32")
    run.add_argument("--steps", type=int, default=30)
    run.add_argument("--trace", default=None,
                     help="a directory for a dew.Profiler trace of three steady-state steps")
    args = parser.parse_args(argv)
    result = {"projection": projection, "adam": adam, "step": step}[args.mode](args)
    result.update(device_kind=jax.devices()[0].device_kind, jax=jax.__version__,
                  peak_bytes=peak_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
