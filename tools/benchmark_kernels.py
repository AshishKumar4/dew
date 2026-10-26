"""Measure the per-generation kernel choices, one measurement per process.

Each mode prints one JSON line. `peak_bytes_in_use` is a process high-water
mark, so a row is one process.

- `projection`: `dew.nn.moe.expert_projection` forward and forward plus
  backward, at lm-moe's shape (8192 rows, 8 experts, bf16 compute, Dirichlet
  routing), against a float64 oracle of the rounded operands.
- `adam`: one AdamW update over the lm-dense parameter tree, fp32, bf16 or
  8-bit moment storage (`OptimConfig.state_dtype`).
- `step`: a whole training step of `lm-dense` (359.8M) or `lm-moe` (321.8M,
  8 experts top-2), through `tools/benchmark_step.py`'s trainer, with the
  grouped matmul, the state dtype and the vocabulary head chosen.

The docs/performance.md section "Kernel choices per generation" was measured
with these commands, for example:

    PYTHONPATH=src python tools/benchmark_kernels.py projection --implementation pallas
    PYTHONPATH=src python tools/benchmark_kernels.py adam --state-dtype bfloat16
    PYTHONPATH=src python tools/benchmark_kernels.py step --path lm-moe --batch 4 \\
        --implementation auto --state-dtype bfloat16
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from benchmark_cases import LM_DENSE, LM_MOE, Case
from benchmark_models import build_trainer, compiled_step, timed_steps

from dew.nn.moe import expert_projection
from dew.optim import OptimConfig
from dew.telemetry.benchmark import samples
from dew.telemetry.profile import capture_options


def timed(function, arguments, repeats: int) -> dict[str, float]:
    """Mean, minimum and median wall time of `function`, each call synced."""
    measured = np.asarray(samples(function, *arguments, warmup=5, repeats=repeats)) * 1e3
    return {"mean_ms": float(np.mean(measured)), "min_ms": float(np.min(measured)),
            "median_ms": float(np.median(measured))}


def peak_bytes() -> int | None:
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use")


def projection(args: argparse.Namespace) -> dict[str, object]:
    rows, experts = args.rows, args.experts
    width, features = (args.width, args.features) if args.shape == "up" else (args.features, args.width)
    rng = np.random.default_rng(0)
    sizes = np.floor(rng.dirichlet(np.ones(experts)) * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    x = jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16)
    kernel = jnp.asarray(rng.normal(size=(experts, width, features)) / np.sqrt(width), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=(rows, features)), jnp.bfloat16)
    group_sizes = jnp.asarray(sizes)

    def project(x, kernel):
        return jnp.asarray(expert_projection(x, kernel, group_sizes, jnp.bfloat16,
                                             args.implementation, None))

    def loss(x, kernel):
        return jnp.sum(project(x, kernel).astype(jnp.float32) * cotangent.astype(jnp.float32))

    forward = jax.jit(project)
    both = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    output = np.asarray(forward(x, kernel), np.float64)
    _, (dx, dk) = both(x, kernel)
    xr = np.asarray(x, np.float64)
    kr = np.asarray(kernel.astype(jnp.bfloat16), np.float64)
    cr = np.asarray(cotangent, np.float64)
    starts = np.cumsum(sizes) - sizes
    oracle, dx_oracle = np.zeros((rows, features)), np.zeros((rows, width))
    dk_oracle = np.zeros((experts, width, features))
    for expert in range(experts):
        block = slice(starts[expert], starts[expert] + sizes[expert])
        oracle[block] = xr[block] @ kr[expert]
        dx_oracle[block] = cr[block] @ kr[expert].T
        dk_oracle[expert] = xr[block].T @ cr[block]

    def relative(value, reference):
        return float(np.abs(np.asarray(value, np.float64) - reference).max() / np.abs(reference).max())

    return {"implementation": args.implementation, "shape": [rows, width, features, experts],
            "groups": sizes.tolist(), "forward": timed(forward, (x, kernel), args.repeats),
            "forward_backward": timed(both, (x, kernel), args.repeats),
            "forward_error": relative(output, oracle), "input_gradient_error": relative(dx, dx_oracle),
            "kernel_gradient_error": relative(dk, dk_oracle)}


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
            with jax.profiler.trace(args.trace, profiler_options=capture_options()):
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
                     help="a directory for a jax.profiler trace of three steady-state steps")
    args = parser.parse_args(argv)
    result = {"projection": projection, "adam": adam, "step": step}[args.mode](args)
    result.update(device_kind=jax.devices()[0].device_kind, jax=jax.__version__,
                  peak_bytes=peak_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
