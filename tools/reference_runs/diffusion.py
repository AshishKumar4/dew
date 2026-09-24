#!/usr/bin/env python3
"""One diffusion reference run, flaxdiff's or Dew's, on fixed images and
fixed noise.

Both sides train the same DiT (the SimpleDiT both packages define, whose
parameter trees are the same leaf for leaf) on the EDM convention: log-normal
sigmas at P_mean -0.4 / P_std 1, Karras preconditioning, the EDM lambda
weight, the l2 loss. What would differ between two runs of the same model
is the randomness, so none is drawn inside either step: each step's noise
level draw `t` (the standard normal both EDM schedulers map through
sigma = exp(P_mean + P_std t)) and its Gaussian noise come from
`np.random.default_rng([seed, step])` and ride in the batch. Each side's
loss is its own objective's `loss` with the two draws read from the batch
instead of the step key; everything between them is the package's code.

The shared initial weights are flaxdiff's init at key 0, written by the
flaxdiff run as an npz (`--init`) and read by Dew's as its `pretrained` tree.
Each side's own trainer builds the step: flaxdiff's
`GeneralDiffusionTrainer._define_train_step` over its data/fsdp mesh, Dew's
`Trainer.compile` over `MeshSpec`. The optimizer is optax AdamW on the cosine
schedule with global-norm clipping on both, behind `recorded_norm` so the
gradient norm leaves the compiled step; the EMA runs on both, as both
trainers keep one.

    python tools/reference_runs/diffusion.py --framework flaxdiff \\
        --flaxdiff-path <dir holding flaxdiff/> --data flowers.npz --init init.npz \\
        --dtype bfloat16 --batch 64 --out flaxdiff-bf16.json
    python tools/reference_runs/diffusion.py --framework dew --data flowers.npz \\
        --init init.npz --dtype bfloat16 --batch 64 --out dew-bf16.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from common import (
    TracedWindow,
    attach_xplane_profile,
    dit_train_flops_per_image,
    git_head,
    host,
    peak_flops,
    sha256,
    source_version,
    steps_for,
    throughput,
    window_rows,
    write_record,
)
from dew_lm import recorded_norm
from flax.traverse_util import flatten_dict, unflatten_dict

# The DiT both packages define, at 64x64 pixels: 256 patches of 4x4.
MODEL = {"output_channels": 3, "patch_size": 4, "emb_features": 384, "num_layers": 8,
         "num_heads": 6, "mlp_ratio": 4, "norm_epsilon": 1e-5, "force_fp32_for_softmax": True}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("flaxdiff", "dew"), required=True)
    parser.add_argument("--flaxdiff-path", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--init", required=True, help="the shared initial weights (npz)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--attention", default=None,
                        help="attention kernel; unset is cudnn at bf16 and xla at fp32")
    parser.add_argument("--mesh", choices=("data", "fsdp"), default="data")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr-peak", type=float, default=2e-4)
    parser.add_argument("--lr-init", type=float, default=2e-5)
    parser.add_argument("--lr-end", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=5)
    args = parser.parse_args()
    if args.framework == "flaxdiff" and args.flaxdiff_path is None:
        parser.error("--framework flaxdiff needs --flaxdiff-path, the directory holding flaxdiff/")
    return args


def draws(seed: int, steps: int, batch: int, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Every step's noise-level draws and Gaussian noise, one generator a step."""
    levels = np.empty((steps, batch), np.float32)
    noise = np.empty((steps, batch, *shape), np.float32)
    for step in range(steps):
        generator = np.random.default_rng([seed, step])
        levels[step] = generator.standard_normal(batch, dtype=np.float32)
        noise[step] = generator.standard_normal((batch, *shape), dtype=np.float32)
    return levels, noise


def load_tree(path: str | Path) -> dict:
    """The npz of `/`-joined leaf paths `save_tree` writes, as its nested dict."""
    with np.load(path) as arrays:
        return unflatten_dict({name: np.asarray(arrays[name]) for name in arrays.files}, sep="/")


def save_tree(path: str | Path, tree) -> None:
    arrays: dict[str, Any] = {name: np.asarray(leaf) for name, leaf in flatten_dict(tree, sep="/").items()}
    np.savez(path, **arrays)


def solver(args, total: int) -> tuple[optax.GradientTransformation, optax.Schedule]:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr_init, peak_value=args.lr_peak, warmup_steps=args.warmup,
        decay_steps=total, end_value=args.lr_end)
    return optax.chain(recorded_norm(), optax.clip_by_global_norm(args.clip),
                       optax.adamw(schedule, b1=args.b1, b2=args.b2, eps=args.eps,
                                   weight_decay=args.weight_decay)), schedule


def flaxdiff_side(args, images, total, attention):
    """flaxdiff's trainer over its own mesh, with the draws read from the batch."""
    sys.path.insert(0, args.flaxdiff_path)
    from flaxdiff.inputs import DiffusionInputConfig
    from flaxdiff.models.simple_dit import SimpleDiT
    from flaxdiff.predictors import get_diffusion_preset
    from flaxdiff.schedulers import get_coeff_shapes_tuple
    from flaxdiff.trainer import DiffusionObjective, GeneralDiffusionTrainer

    class FixedDraws(DiffusionObjective):
        """flaxdiff's `DiffusionObjective.loss`, the two draws read from the batch."""

        def loss(self, params, ema_params, batch, rng, step):
            data = (jnp.asarray(batch["image"], dtype=jnp.float32) - 127.5) / 127.5
            noise_level, noise = batch["t"], batch["noise"]
            rates = self.noise_schedule.get_rates(noise_level, get_coeff_shapes_tuple(data))
            noisy_data, c_in, expected_output = self.model_output_transform.forward_diffusion(
                data, noise, rates)
            inputs = self.noise_schedule.transform_inputs(noisy_data * c_in, noise_level)
            preds = self.model.apply(params, *inputs, train=True, rngs={"dropout": rng})
            preds = self.model_output_transform.pred_transform(noisy_data, preds, rates)
            sample_losses = self.loss_fn(preds, expected_output)
            weights = self.noise_schedule.get_weights(noise_level, get_coeff_shapes_tuple(sample_losses))
            return jnp.mean(sample_losses * weights), {}

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else None
    model = SimpleDiT(**MODEL, dtype=dtype, attention_impl=attention)
    train_schedule, _, transform = get_diffusion_preset("edm")
    input_config = DiffusionInputConfig(sample_data_key="image", sample_data_shape=images.shape[1:],
                                        conditions=[])
    objective = FixedDraws(model=model, noise_schedule=train_schedule, model_output_transform=transform,
                           input_config=input_config, input_shapes=input_config.get_input_shapes(),
                           unconditional_prob=0.0, ema_decay=args.ema)
    optimizer, schedule = solver(args, total)
    trainer = GeneralDiffusionTrainer(
        model, optimizer, input_config, jax.random.PRNGKey(0), noise_schedule=train_schedule,
        objective=objective, unconditional_prob=0.0, model_output_transform=transform,
        ema_decay=args.ema, checkpoint_base_path=str(Path(args.out).with_suffix("").parent / "ckpt"),
        distributed_training=True, fsdp_size=jax.device_count() if args.mesh == "fsdp" else 1)
    state = trainer.state
    init = Path(args.init)
    if not init.exists():
        save_tree(init, jax.device_get(state.params))
    initial = load_tree(init)
    params = jax.device_put(initial, trainer.state_sharding.params)
    state = state.replace(
        params=params, ema_params=jax.device_put(initial, trainer.state_sharding.ema_params),
        opt_state=jax.jit(optimizer.init, out_shardings=trainer.state_sharding.opt_state)(params))
    step_fn = trainer._define_train_step(args.batch)
    rng_state = trainer.rngstate

    def run(state, batch):
        nonlocal rng_state
        state, loss, _, rng_state, _ = step_fn(state, rng_state, batch)
        return state, loss

    def place(batch):
        from flaxdiff.utils import shard_batch
        return shard_batch(trainer.batch_sharding, batch)

    return state, run, place, schedule, dict(trainer.mesh.shape), None


def dew_side(args, images, total, attention):
    """Dew's trainer over `MeshSpec`, with the draws read from the batch."""
    from dew.diffusion.presets import EDM
    from dew.diffusion.schedules import expand
    from dew.diffusion.transforms import broadcast_rates
    from dew.inputs import Field, InputSpec, unit_range
    from dew.objectives.base import Aux, Mean
    from dew.objectives.diffusion import DiffusionObjective
    from dew.registry import models
    from dew.training import MeshSpec, Trainer
    from dew.training.distributed import batch_shardings

    class FixedDraws(DiffusionObjective):
        """Dew's `DiffusionObjective.loss`, the two draws read from the batch."""

        def loss(self, params, batch, step):
            samples = unit_range(batch[self.inputs.sample.key])
            schedule = self.process.schedule
            t, noise = batch["t"], batch["noise"]
            rates = broadcast_rates(schedule, t, samples)
            noisy, c_in, target = self.process.prediction.forward_diffusion(samples, noise, rates)
            preds = self.model.apply(self.trainable(params), noisy * c_in, schedule.model_time(t),
                                     train=True, rngs={"dropout": step.key})
            preds = self.process.prediction.pred_transform(noisy, preds, rates, t)
            losses = optax.l2_loss(preds, target)
            weights = expand(self.process.weight(t), losses)
            return Mean(jnp.sum(losses * weights),
                        jnp.asarray(losses.size, jnp.promote_types(losses.dtype, jnp.float32))), Aux(metrics={})

    from dew.registry import with_precision

    model = models.build("simple_dit", with_precision("simple_dit", MODEL, dtype=args.dtype,
                                                      attention_impl=attention))
    # The npz holds flaxdiff's whole variables dict, {"params": ...}. flaxdiff
    # keeps its Fourier table out of that dict, and its main branch draws it
    # from numpy's RandomState(42) (commit 63f2427), as Dew's init does, so
    # Dew's own init supplies it. FlaxDiff 0.2 drew it from jax.random instead
    # (see FourierEmbedding).
    constants = jax.tree.map(np.asarray, model.init(
        jax.random.key(0), jnp.zeros((1, *images.shape[1:])), jnp.zeros((1,)))["constants"])
    initial = {**load_tree(args.init), "constants": constants, "encoders": {}}
    objective = FixedDraws(model, EDM()(), InputSpec(sample=Field("image", images.shape[1:])),
                           ema_decay=args.ema, guidance=None, pretrained=initial)
    optimizer, schedule = solver(args, total)
    devices = jax.device_count()
    mesh = MeshSpec(fsdp=devices) if args.mesh == "fsdp" else MeshSpec()
    trainer = Trainer(objective, optimizer, key=jax.random.key(0), mesh=mesh, checkpoints=None,
                      tracker=None)
    state, _, _ = trainer.place()
    layout = {}
    compiled = {}

    def place(batch):
        if not layout:
            layout["value"] = batch_shardings(trainer.device_mesh, batch)
        return jax.device_put(batch, layout["value"])

    def run(state, batch):
        if not compiled:
            compiled["step"] = trainer.compile(state, batch)
        state, loss, _, _, _ = compiled["step"](state, batch)
        return state, loss

    return state, run, place, schedule, dict(trainer.device_mesh.shape), trainer


def main() -> None:
    args = arguments()
    # Read before the run, so a source that cannot be traced fails at once.
    built_from = {"jax": jax.__version__, "optax": optax.__version__, "flax": __import__("flax").__version__,
                  "dew": git_head(Path(__file__).resolve().parents[2]),
                  "flaxdiff": source_version(args.flaxdiff_path) if args.flaxdiff_path else None}
    if args.dtype == "float32":
        jax.config.update("jax_default_matmul_precision", "highest")
    attention = args.attention or ("cudnn" if args.dtype == "bfloat16" else "xla")
    data = np.load(args.data)
    images, order = data["images"], data["order"]
    schedule_steps = steps_for(order, args.batch)
    total = schedule_steps if args.steps is None else min(schedule_steps, args.steps)
    levels, noise = draws(args.seed, total, args.batch, images.shape[1:])

    side = flaxdiff_side if args.framework == "flaxdiff" else dew_side
    state, run, place, schedule, mesh_shape, trainer = side(args, images, schedule_steps, attention)

    def batch(step: int):
        return place({"image": images[window_rows(order, step, args.batch)],
                      "t": levels[step], "noise": noise[step]})

    losses, norms = [], []
    trace_dir = Path(args.out).with_suffix("").resolve()
    window = TracedWindow(total, args.timing_warmup, args.profile_steps, trace_dir)
    loss = None
    compile_start = time.perf_counter()
    for step in range(total):
        window.before(step, loss)
        state, loss = run(state, batch(step))
        if step == 0:
            jax.block_until_ready(loss)
            first_step_seconds = time.perf_counter() - compile_start
        losses.append(loss)
        norms.append(jnp.copy(state.opt_state[0].norm))
    window_seconds = window.close(loss)

    device_kind = jax.devices()[0].device_kind
    devices = jax.device_count()
    timed = throughput(window_seconds, window.timed_steps, args.batch)
    timed["images_per_s"] = timed.pop("tokens_per_s")
    flops_image = dit_train_flops_per_image(MODEL, *images.shape[1:])
    peak_rate = peak_flops(device_kind)
    stats = [device.memory_stats() or {} for device in jax.local_devices()]
    record = {
        "framework": args.framework,
        "precision": args.dtype,
        "parallel": f"{args.mesh}={devices}",
        "world": devices,
        "config": {**vars(args), "attention": attention, "model": MODEL, "total_steps": total,
                   "schedule_steps": schedule_steps, "data_sha256": sha256(args.data),
                   "init_sha256": sha256(args.init), "mesh": mesh_shape,
                   "convention": "EDM (P_mean -0.4, P_std 1, sigma_data 0.5), l2 loss, EDM weight",
                   "optimizer": "optax.chain(recorded_norm, clip_by_global_norm, adamw)"},
        "versions": built_from,
        "device": device_kind,
        "host": host(),
        "loss": np.asarray(jax.device_get(losses), np.float64).tolist(),
        "grad_norm": np.asarray(jax.device_get(norms), np.float64).tolist(),
        "lr": [float(schedule(step)) for step in range(total)],
        "throughput": timed,
        "first_step_seconds": first_step_seconds,
        "flops_per_image": flops_image,
        "hlo_flops_per_step": None if trainer is None else trainer.flops_per_step,
        "mfu": None if peak_rate is None else flops_image * timed["images_per_s"] / (peak_rate * devices),
        "memory": {"peak_allocated_bytes": [int(s.get("peak_bytes_in_use", 0)) for s in stats]},
    }
    if args.profile_steps:
        write_record(args.out, record)
        attach_xplane_profile(record, trace_dir, args.profile_steps)
    write_record(args.out, record)
    print(json.dumps({"first_loss": record["loss"][0], "last_loss": record["loss"][-1],
                      "images_per_s": timed["images_per_s"],
                      "peak_gib": max(record["memory"]["peak_allocated_bytes"]) / 2**30}, indent=1))


if __name__ == "__main__":
    main()
