#!/usr/bin/env python3
"""The Dew side of a language-model reference run.

The same windows in the same order as `torch_lm.py`, through the trainer's
own compiled step: `LMObjective` over the checkpoint `load_pretrained`
reads (fp32 masters, `--dtype` compute), the solver `build_optimizer` makes
from an `OptimConfig` (global-norm clip, then AdamW on the cosine schedule),
and the mesh `MeshSpec` describes. One transformation is chained in front of
that solver, `recorded_norm`, which keeps the global norm of the gradient it
is handed in the optimizer state and passes the gradient on unchanged; it
is how the step's gradient norm leaves the compiled step.

No EMA (`ema_decay=None`): the reference keeps none, and the average would
cost the step memory and time the comparison does not ask about.

    python tools/reference_runs/dew_lm.py --model <hf dir> --data windows.npz \\
        --batch 4 --dtype bfloat16 --out dew-bf16.json
    python tools/reference_runs/dew_lm.py ... --mesh fsdp --devices 4
"""

import argparse
import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from common import (
    TracedWindow,
    attach_xplane_profile,
    git_head,
    host,
    peak_flops,
    sha256,
    steps_for,
    throughput,
    train_flops_per_token,
    window_rows,
    write_record,
)


class NormState(NamedTuple):
    norm: jax.Array


def recorded_norm() -> optax.GradientTransformation:
    """Pass the gradient through and keep its global norm in the state."""

    def init(params):
        del params
        return NormState(jnp.zeros((), jnp.float32))

    def update(updates, state, params=None):
        del state, params
        return updates, NormState(optax.tree.norm(updates).astype(jnp.float32))

    return optax.GradientTransformation(init, update)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, required=True, help="global rows per step")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--mesh", choices=("data", "fsdp", "expert"), default="data",
                        help="the axis the devices form; data=1 device is the single-GPU run")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Layout's sharding tolerance; an expert mesh keeps the dense weights "
                             "replicated, as DDP does, which the default 2%% refuses")
    parser.add_argument("--dispatch", choices=("global", "exchange"), default=None,
                        help="MoE: the mixture's token dispatch; the checkpoint's model runs 'global', "
                             "and 'exchange' sends tokens to the expert axis's devices (all-to-all)")
    parser.add_argument("--attention", default="auto")
    parser.add_argument("--head-chunks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr-peak", type=float, default=2e-5)
    parser.add_argument("--lr-init", type=float, default=2e-6)
    parser.add_argument("--lr-end", type=float, default=2e-6)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--aux-loss-alpha", type=float, default=None,
                        help="MoE: LMObjective's aux_loss_alpha (Switch balance loss per router)")
    parser.add_argument("--seq-aux", action="store_true",
                        help="MoE: the per-sequence balance loss instead of the batch-wide one")
    parser.add_argument("--probe-every", type=int, default=None,
                        help="MoE: router loads and entropy on a fixed batch every this many steps")
    parser.add_argument("--probe-rows", type=int, default=4)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--trace-dir", default=None)
    return parser.parse_args()


@dataclasses.dataclass
class Run:
    """A placed train state, its compiled step and the batches that feed it."""

    trainer: Any
    state: Any
    step: Any
    batch: Callable[[int], Any]
    rate: Callable[[int], Any]
    windows: np.ndarray
    seq: int
    total: int
    schedule_steps: int
    compile_seconds: float


def build(args: argparse.Namespace) -> Run:
    """The trainer's own objective, solver and mesh over the recorded
    windows, placed and compiled."""
    if args.dtype == "float32":
        # The fp32 run is the truth the bound is measured from: no TF32.
        jax.config.update("jax_default_matmul_precision", "highest")

    from dew.config import OptimConfig
    from dew.interop import load_pretrained
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import TEXT_KEY, LMObjective
    from dew.training import Layout, MeshSpec, Trainer
    from dew.training.distributed import batch_shardings
    from dew.training.optim import Cosine, build_optimizer

    data = np.load(args.data)
    windows, order = data["windows"], data["order"]
    seq = windows.shape[1] - 1
    # The schedule decays over every epoch; --steps only stops the run early.
    schedule_steps = steps_for(order, args.batch)
    total = schedule_steps if args.steps is None else min(schedule_steps, args.steps)

    pretrained = load_pretrained(args.model, dtype=args.dtype, param_dtype="float32",
                                 attention_impl=args.attention)
    model = pretrained.model
    if args.dispatch is not None:
        if not isinstance(model, CausalTransformer) or model.mixture is None:
            raise SystemExit(f"--dispatch sets a mixture's dispatch, and {args.model} builds no mixture")
        model = model.clone(mixture=dataclasses.replace(model.mixture, dispatch=args.dispatch))
    objective = LMObjective(model, seq, ema_decay=None, head_chunks=args.head_chunks,
                            pretrained=pretrained.variables, aux_loss_alpha=args.aux_loss_alpha,
                            seq_aux=args.seq_aux)
    schedule = Cosine(peak=args.lr_peak, warmup_steps=args.warmup, end=args.lr_end, init=args.lr_init)
    solver = build_optimizer(OptimConfig(
        optimizer="adamw", optimizer_opts={"b1": args.b1, "b2": args.b2, "eps": args.eps},
        schedule=schedule, weight_decay=args.weight_decay, clip_grads=args.clip), schedule_steps)
    devices = jax.device_count()
    mesh = {"data": MeshSpec(), "fsdp": MeshSpec(fsdp=devices), "expert": MeshSpec(expert=devices)}[args.mesh]
    layout = Layout() if args.tolerance is None else Layout(tolerance=args.tolerance)
    trainer = Trainer(objective, optax.chain(recorded_norm(), solver), key=jax.random.key(0),
                      mesh=mesh, layout=layout, checkpoints=None, tracker=None)
    state, _, _ = trainer.place()
    del pretrained
    placement = batch_shardings(trainer.device_mesh, {TEXT_KEY: windows[:args.batch]})

    def batch(step: int):
        return jax.device_put({TEXT_KEY: windows[window_rows(order, step, args.batch)]}, placement)

    compile_start = time.perf_counter()
    step = trainer.compile(state, batch(0))
    return Run(trainer, state, step, batch, schedule.schedule(schedule_steps), windows, seq, total,
               schedule_steps, time.perf_counter() - compile_start)


class RouterProbe:
    """Each router's share of top-k slots per expert and its mean entropy
    over one fixed batch, as `torch_lm.router_probe` computes, at the steps
    it is called on."""

    def __init__(self, run: Run, rows: int):
        from dew.objectives.lm.objective import _router_scores

        objective = run.trainer.objective
        self.mesh = run.trainer.device_mesh
        self.tokens = jnp.asarray(run.windows[:rows])
        self.records: list[dict] = []

        @jax.jit
        def scores(params, tokens):
            routing = objective.token_scores(params, tokens, routing=True).routing
            loads, entropies = [], []
            for probs, indices in _router_scores(routing):
                loads.append(jnp.bincount(indices.ravel(), length=probs.shape[-1]) / indices.size)
                p = probs.astype(jnp.float32)
                entropies.append(-jnp.mean(jnp.sum(p * jnp.log(jnp.maximum(p, 1e-30)), axis=-1)))
            return jnp.stack(loads), jnp.stack(entropies)

        self.scores = scores

    def __call__(self, step: int, params) -> float:
        """Probe at `step` and return the seconds it took, which the timed
        window leaves out."""
        jax.block_until_ready(params)
        start = time.perf_counter()
        with jax.set_mesh(self.mesh):
            loads, entropies = jax.device_get(self.scores(params, self.tokens))
        self.records.append({"step": step, "load": np.asarray(loads, np.float64).tolist(),
                             "entropy": np.asarray(entropies, np.float64).tolist()})
        return time.perf_counter() - start


@dataclasses.dataclass
class Curves:
    """What every step left, and the timed window's length."""

    losses: list
    objectives: list
    norms: list
    extras: list
    window_seconds: float
    timed_steps: int


def trace_directory(args: argparse.Namespace) -> Path:
    return Path(args.trace_dir or Path(args.out).with_suffix("")).resolve()


def train(args: argparse.Namespace, run: Run, probe: RouterProbe | None) -> tuple[Any, Curves]:
    """Every step in the recorded order: timed from `timing_warmup` to the
    profiled tail, which is traced, with the probes' time taken out."""
    losses, objectives, norms, extras = [], [], [], []
    window = TracedWindow(run.total, args.timing_warmup, args.profile_steps, trace_directory(args))
    state, loss = run.state, None
    for step in range(run.total):
        window.before(step, loss)
        if probe is not None and step % args.probe_every == 0:
            spent = probe(step, state.params)
            if window.times(step):
                window.excluded += spent
        state, loss, metrics, _, _ = run.step(state, run.batch(step))
        # The step's loss is the whole objective, balance terms included;
        # the cross entropy the reference records as its loss is `ce`.
        losses.append(metrics["ce"])
        objectives.append(loss)
        norms.append(jnp.copy(state.opt_state[0].norm))
        if args.aux_loss_alpha is not None:
            extras.append({"aux_loss": metrics["aux_loss"]})
    return state, Curves(losses, objectives, norms, extras, window.close(loss), window.timed_steps)


def versions() -> dict:
    """The versions a record names, read before the run so an untraceable
    checkout fails at once rather than after it."""
    return {"jax": jax.__version__, "dew": git_head(Path(__file__).resolve().parents[2]),
            "optax": optax.__version__, "flax": __import__("flax").__version__}


def record(args: argparse.Namespace, run: Run, curves: Curves, probe: RouterProbe | None,
           built_from: dict) -> dict:
    """The run's curves, conditions, versions, throughput and memory."""
    device_kind = jax.devices()[0].device_kind
    devices = jax.device_count()
    timed = throughput(curves.window_seconds, curves.timed_steps, args.batch * run.seq)
    hf_config = json.loads(Path(args.model, "config.json").read_text())
    flops_token = train_flops_per_token(hf_config, run.seq)
    peak_rate = peak_flops(device_kind)
    stats = [device.memory_stats() or {} for device in jax.local_devices()]
    result = {
        "framework": "dew",
        "precision": args.dtype,
        "parallel": f"{args.mesh}={devices}",
        "world": devices,
        "config": {**vars(args), "total_steps": run.total, "schedule_steps": run.schedule_steps, "seq": run.seq,
                   "data_sha256": sha256(args.data), "model_type": hf_config["model_type"],
                   "mesh": dict(run.trainer.device_mesh.shape),
                   "optimizer": "optax.chain(recorded_norm, clip_by_global_norm, adamw) via build_optimizer",
                   "schedule": "dew.training.optim.Cosine", "ema": None,
                   "model_config": {k: v for k, v in run.trainer.objective.model.__dict__.items()
                                    if isinstance(v, (int, float, str, bool, type(None)))}},
        "versions": built_from,
        "device": device_kind,
        "host": host(),
        "loss": np.asarray(jax.device_get(curves.losses), np.float64).tolist(),
        "objective": np.asarray(jax.device_get(curves.objectives), np.float64).tolist(),
        "grad_norm": np.asarray(jax.device_get(curves.norms), np.float64).tolist(),
        "lr": [float(run.rate(step)) for step in range(run.total)],
        "throughput": timed,
        "compile_seconds": run.compile_seconds,
        "flops_per_token": flops_token,
        "hlo_flops_per_step": run.trainer.flops_per_step,
        "mfu": None if peak_rate is None else flops_token * timed["tokens_per_s"] / (peak_rate * devices),
        "memory": {"peak_allocated_bytes": [int(s.get("peak_bytes_in_use", 0)) for s in stats],
                   "bytes_limit": [int(s.get("bytes_limit", 0)) for s in stats]},
    }
    for name in (curves.extras[0] if curves.extras else {}):
        result[name] = np.asarray(jax.device_get([entry[name] for entry in curves.extras]), np.float64).tolist()
    if probe is not None:
        result["router_probe"] = probe.records
    return result


def main() -> None:
    args = arguments()
    built_from = versions()
    run = build(args)
    probe = RouterProbe(run, args.probe_rows) if args.probe_every else None
    state, curves = train(args, run, probe)
    if probe is not None:
        probe(run.total, state.params)
    result = record(args, run, curves, probe, built_from)
    if args.profile_steps:
        write_record(args.out, result)
        attach_xplane_profile(result, trace_directory(args), args.profile_steps)
    write_record(args.out, result)
    print(json.dumps({"first_loss": result["loss"][0], "last_loss": result["loss"][-1],
                      "tokens_per_s": result["throughput"]["tokens_per_s"], "mfu": result["mfu"],
                      "compile_s": run.compile_seconds,
                      "peak_gib": max(result["memory"]["peak_allocated_bytes"]) / 2**30}, indent=1))


if __name__ == "__main__":
    main()
