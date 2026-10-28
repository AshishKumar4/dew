#!/usr/bin/env python3
"""The Dew side of a language-model reference run.

The same windows in the same order as `torch_lm.py`, through the trainer's
own compiled step: `LMObjective` over the checkpoint `Pretrained.load`
reads (fp32 masters, `--dtype` compute), the solver `OptimConfig.build` makes
from an `OptimConfig` (global-norm clip, then AdamW on the cosine schedule),
and the mesh `MeshSpec` describes. One transformation is chained in front of
that solver, `recorded_norm`, which keeps the global norm of the gradient it
is handed in the optimizer state and passes the gradient on unchanged; it
is how the step's gradient norm leaves the compiled step.

No EMA (`ema_decay=None`): the reference keeps none, and the average would
cost the step memory and time the comparison does not ask about.

    python tools/reference_runs/dew_lm.py --model <hf dir> --data windows.npz \\
        --batch 4 --dtype bfloat16 --work work.json --out dew-bf16.json
    python tools/reference_runs/dew_lm.py ... --mesh fsdp

The mesh axis spans every device the process sees (`jax.device_count()`);
choose them the way jax does, with CUDA_VISIBLE_DEVICES or JAX_PLATFORMS.
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
    WorkloadWork,
    attach_xplane_profile,
    documents,
    host,
    peak_flops,
    run_arguments,
    sha256,
    steps_for,
    throughput,
    window_rows,
    write_record,
)
from reference_source import dew_revision


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
    from dew.nn.backbones.decoder_block import REMAT_POLICIES

    parser = argparse.ArgumentParser(parents=[run_arguments()])
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--mesh", choices=("data", "fsdp", "expert"), default="data",
                        help="the axis the devices form; data=1 device is the single-GPU run")
    parser.add_argument("--dispatch", choices=("global", "exchange"), default=None,
                        help="MoE: the mixture's token dispatch; the checkpoint's model runs 'global', "
                             "and 'exchange' sends tokens to the expert axis's devices (all-to-all)")
    parser.add_argument("--attention", default="auto")
    parser.add_argument("--head-chunks", type=int, default=4)
    parser.add_argument("--split", type=int, default=None,
                        help="the microbatches a step's batch is read in (Trainer's split); unset plans it")
    parser.add_argument("--state-dtype", choices=("float32", "bfloat16", "int8"), default="float32",
                        help="Adam's moments in memory (OptimConfig.state_dtype)")
    parser.add_argument("--remat", default=None, choices=sorted(REMAT_POLICIES),
                        help="the decoder's starting remat policy; unset, the model's")
    parser.add_argument("--aux-loss-alpha", type=float, default=None,
                        help="MoE: ExpertBalance's aux_loss_alpha (Switch balance loss per router)")
    parser.add_argument("--seq-aux", action="store_true",
                        help="MoE: the per-sequence balance loss instead of the batch-wide one")
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
    work: WorkloadWork


def build(args: argparse.Namespace) -> Run:
    """The trainer's own objective, solver and mesh over the recorded
    windows, placed and compiled."""
    if args.dtype == "float32":
        # The fp32 run is the truth the bound is measured from: no TF32.
        jax.config.update("jax_default_matmul_precision", "highest")

    from work import measure

    from dew.batch import coordinates
    from dew.distributed.batches import batch_shardings
    from dew.interop import Pretrained
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.multimodal import MultimodalTransformer
    from dew.objectives.lm import TEXT_KEY, LMObjective
    from dew.objectives.routing import ExpertBalance
    from dew.optim import Cosine, OptimConfig, ParamGroup
    from dew.training import MeshSpec, Trainer
    from dew.training.planning import Trials

    data = np.load(args.data)
    windows, order = data["windows"], data["order"]
    seq = windows.shape[1] - 1
    segments = None if args.documents is None else documents(windows.shape, args.documents)
    # A packed batch's documents a row, the same each step, so one compile serves every step.
    most = None if segments is None else int(segments.max())
    # The schedule decays over every epoch; --steps only stops the run early.
    schedule_steps = steps_for(order, args.batch)
    total = schedule_steps if args.steps is None else min(schedule_steps, args.steps)

    # On the host: the trainer places the weights over its mesh.
    pretrained = Pretrained.load(args.model, dtype=args.dtype, param_dtype="float32",
                                 attention_impl=args.attention, on_host=True)
    model, variables = pretrained.model, pretrained.variables
    if isinstance(model, MultimodalTransformer):
        # A text run trains a multimodal checkpoint's decoder, its language model, alone: the towers
        # read no text, and would only carry their weights and moments through every step.
        model = model.language_model
        variables = {collection: tree["language_model"] for collection, tree in variables.items()
                     if "language_model" in tree}
    if args.remat is not None:
        model = model.clone(remat=args.remat)
    if args.dispatch is not None:
        if not isinstance(model, CausalTransformer) or model.mixture is None:
            raise SystemExit(f"--dispatch sets a mixture's dispatch, and {args.model} builds no mixture")
        model = model.clone(mixture=dataclasses.replace(model.mixture, dispatch=args.dispatch))
    balance = None if args.aux_loss_alpha is None else ExpertBalance(
        aux_loss_alpha=args.aux_loss_alpha, seq_aux=args.seq_aux)
    objective = LMObjective(model, seq, ema_decay=None, head_chunks=args.head_chunks,
                            variables=variables, balance=balance)
    def rows(indices) -> dict:
        """The windows' rows, a packed run's documents laid out beside them."""
        if segments is None:
            return {TEXT_KEY: windows[indices]}
        return {TEXT_KEY: windows[indices], "text_coordinates": coordinates(
            segments=segments[indices], max_segments=most, xp=np)}

    work = measure(objective, rows(np.arange(args.batch)))
    work.write(args.work)
    schedule = Cosine(peak_value=args.lr_peak, warmup_steps=args.warmup, end_value=args.lr_end,
                      init_value=args.lr_init)
    # One group over every parameter, as torch_lm.py's AdamW decays the norms
    # and biases too.
    solver = OptimConfig(
        optimizer="adamw", b1=args.b1, b2=args.b2, optimizer_opts={"eps": args.eps},
        state_dtype=args.state_dtype, learning_rate=schedule, weight_decay=args.weight_decay,
        clip_grads=args.clip,
        param_groups=(ParamGroup("all", ("*",)),)).build(schedule_steps)
    devices = jax.device_count()
    mesh = {"data": MeshSpec(), "fsdp": MeshSpec(fsdp=devices), "expert": MeshSpec(expert=devices)}[args.mesh]
    trainer = Trainer(objective, optax.chain(recorded_norm(), solver), key=jax.random.key(0),
                      mesh=mesh, checkpoints=None, tracker=None, split=args.split)
    state, _, _ = trainer.place()
    del pretrained
    placement = batch_shardings(trainer.device_mesh, rows(np.arange(args.batch)))

    def batch(step: int):
        return jax.device_put(rows(window_rows(order, step, args.batch)), placement)

    compile_start = time.perf_counter()
    # As `fit` plans: a frontier the plan opens is timed, and the run goes on from the state it timed.
    trials = Trials(state, batch(0))
    step = trainer.compile(state, batch(0), trials)
    return Run(trainer, trials.state, step, batch, schedule.schedule(schedule_steps), windows, seq, total,
               schedule_steps, time.perf_counter() - compile_start, work)


class RouterProbe:
    """Each router's share of top-k slots per expert and its mean entropy
    over one fixed batch, as `torch_lm.router_probe` computes, at the steps
    it is called on."""

    def __init__(self, run: Run, rows: int):
        from dew.objectives.routing import router_scores

        objective = run.trainer.objective
        self.mesh = run.trainer.device_mesh
        self.tokens = jnp.asarray(run.windows[:rows])
        self.records: list[dict] = []

        @jax.jit
        def scores(params, tokens):
            routing = objective.token_scores(params, tokens, routing=True).routing
            loads, entropies = [], []
            for probs, indices in router_scores(routing):
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
            spent = probe(step, state.variables)
            if window.times(step):
                window.excluded += spent
        state, loss, metrics, _, _, _ = run.step(state, run.batch(step))
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
    return {"jax": jax.__version__, "dew": dew_revision(),
            "optax": optax.__version__, "flax": __import__("flax").__version__}


def record(args: argparse.Namespace, run: Run, curves: Curves, probe: RouterProbe | None,
           built_from: dict) -> dict:
    """The run's curves, conditions, versions, throughput and memory."""
    device_kind = jax.devices()[0].device_kind
    devices = jax.device_count()
    timed = throughput(curves.window_seconds, curves.timed_steps, args.batch * run.seq)
    hf_config = json.loads(Path(args.model, "config.json").read_text())
    flops_token = run.work.training_flops_per_step / run.work.tokens_per_step
    peak_rate = peak_flops(device_kind)
    stats = [device.memory_stats() or {} for device in jax.local_devices()]
    result = {
        "framework": "dew",
        "precision": args.dtype,
        "parallel": f"{args.mesh}={devices}",
        "world": devices,
        "config": {**vars(args), "total_steps": run.total, "schedule_steps": run.schedule_steps,
                   "seq": run.seq,
                   "data_sha256": sha256(args.data), "model_type": hf_config["model_type"],
                   "mesh": dict(run.trainer.device_mesh.shape),
                   "optimizer": "optax.chain(recorded_norm, clip_by_global_norm, adamw) "
                                "via OptimConfig.build",
                   "schedule": "dew.optim.Cosine", "ema": None,
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
        "work": dataclasses.asdict(run.work),
        "hlo_flops_per_step": (run.trainer.step_flops and run.trainer.step_flops.hardware),
        "mfu": None if peak_rate is None else run.work.training_flops_per_step * curves.timed_steps
            / curves.window_seconds / (peak_rate * devices),
        "memory": {"peak_allocated_bytes": [int(s.get("peak_bytes_in_use", 0)) for s in stats],
                   "bytes_limit": [int(s.get("bytes_limit", 0)) for s in stats]},
    }
    for name in (curves.extras[0] if curves.extras else {}):
        result[name] = np.asarray(jax.device_get([entry[name] for entry in curves.extras]),
                                  np.float64).tolist()
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
        probe(run.total, state.variables)
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
