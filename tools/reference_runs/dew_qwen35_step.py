#!/usr/bin/env python3
"""The Dew twin of torch_qwen35_step, through Pretrained.load and Trainer.

The checkpoint's text backbone alone is trained: no vision tower, decision
head, vocabulary projection, EMA, accumulation, clipping or rematerialization.
The scalar projection and resident token rows are shared with torch. FP32
masters and AdamW moments support bf16 compute (fp32 for the tiny CPU proof).

Compilation and 20 warmup steps precede the endpoint-synchronized 200-step
window. As in benchmark_step.py, a later synchronized pass supplies diagnostic
median/p10/p90 samples, never the pairing's throughput or noise estimate.
The JAX peak is read before that pass and includes loading/compile/warmup;
JAX has no resettable steady-window peak. --smoke reads the untrained tiny
checkpoint that torch_qwen35_step --smoke --smoke-fixture DIR exported.
"""

import argparse
import dataclasses
import importlib.metadata
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from common import host, peak_flops, throughput, write_record
from qwen35 import MODEL, REVISION, WHOLE_PEAK_SCOPE, inputs, workload
from reference_source import git_head
from work import measure

from dew.config import OptimConfig
from dew.inputs import Field, InputSpec
from dew.interop import Pretrained
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.multimodal import MultimodalTransformer
from dew.objectives.base import Objective
from dew.optim import Cosine
from dew.training import Trainer
from dew.training.runtime import prepare_process


class BackboneLoss(Objective):
    """The trainer's ordinary gradient/update path over normalized hidden states.

    No objective-side gradient rule or performance-only optimizer is used.
    The fixed projection computes in the hidden state's dtype, then its scalar
    outputs are widened before squaring, exactly like torch's autocast linear.
    """

    def __init__(self, model, variables, projection, sequence):
        self.model = model
        self.variables = variables
        self.projection = projection
        self.inputs = InputSpec(sample=Field("tokens", (sequence,)))

    counts_tokens = True

    def real_lengths(self, batch):
        """The hidden-state loss sees each complete, unpadded input row."""
        rows, sequence = np.shape(batch["tokens"])
        return np.full((rows,), sequence, np.int64)

    def loss(self, variables, batch, step):
        hidden = self.model.apply(variables, batch["tokens"], train=step.training,
                                  method="hidden_states")
        projected = hidden @ jnp.asarray(self.projection, hidden.dtype)
        return jnp.mean(jnp.square(projected.astype(jnp.float32)))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dew-qwen35-step.json")
    parser.add_argument("--work", required=True, help="write the native count for this backbone workload")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-fixture", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--session")
    args = parser.parse_args()
    if args.warmup < 2 or args.steps < 1:
        parser.error("need at least two warmup steps and one timed step")
    if args.smoke != bool(args.smoke_fixture):
        parser.error("--smoke requires the torch-exported --smoke-fixture DIR, and vice versa")
    return args


def backbone(source, revision: str | None, dtype: str):
    """The unrematerialized text backbone of `source` through the recipe's loader: what this
    runner trains, and what work.py counts for the full workload."""
    bundle = Pretrained.load(source, revision=revision, dtype=dtype, param_dtype="float32")
    model, variables = bundle.model, bundle.variables
    if isinstance(model, MultimodalTransformer):
        model = model.language_model
        variables = {collection: tree["language_model"] for collection, tree in variables.items()
                     if "language_model" in tree}
    if not isinstance(model, CausalTransformer) or not model.tie_embeddings:
        raise RuntimeError("expected the tied-embedding Qwen3.5 text backbone")
    # The checkpoint's prediction depth (MTP) is no part of the backbone torch trains: its block would add
    # 20.5M parameters the step only carries and updates.
    depths = {f"mtp_{depth}" for depth in range(model.num_nextn_predict_layers)}
    variables = {collection: {name: value for name, value in tree.items() if name not in depths}
                 for collection, tree in variables.items()}
    return model.clone(remat=None, num_nextn_predict_layers=0), variables


def build(args):
    """Load through the recipe's loader and compile the real, unrematted trainer step."""
    if jax.device_count() != 1:
        raise RuntimeError("this comparison requires exactly one device")
    device = jax.local_devices()[0]
    if args.smoke and device.platform != "cpu":
        raise RuntimeError("--smoke runs only on armada CPU")
    if not args.smoke and "A100-SXM4-80GB" not in device.device_kind:
        raise RuntimeError(f"full reference requires A100-SXM4-80GB, found {device.device_kind}")
    if args.smoke:
        jax.config.update("jax_default_matmul_precision", "highest")
    model, variables = (backbone(args.smoke_fixture, None, "float32") if args.smoke
                        else backbone(MODEL, REVISION, "bfloat16"))
    return compile_step(args, model, variables)


def compile_step(args, model, variables):
    """The real, unrematted trainer step on `model`'s rows, compiled, and its work record written."""
    batch, sequence = (2, 16) if args.smoke else (8, 3072)
    rows, projection = inputs(args.seed, model.vocab_size, model.emb_features, batch, sequence)
    objective = BackboneLoss(model, variables, projection, sequence)
    work = measure(objective, {"tokens": rows})
    work.write(args.work)
    solver = OptimConfig(optimizer="adamw", b1=0.9, b2=0.999, optimizer_opts={"eps": 1e-8},
                          weight_decay=1e-4, clip_grads=0.0, state_dtype="float32",
                          learning_rate=Cosine(init_value=0.0, peak_value=2.5e-5, warmup_steps=10,
                                               end_value=0.0)).build(220)
    trainer = Trainer(objective, solver, key=jax.random.key(args.seed), checkpoints=None, tracker=None)
    state, _, _ = trainer.place()
    placed = jax.device_put({"tokens": rows})
    jax.block_until_ready((state, placed))
    started = time.perf_counter()
    compiled = trainer.compile(state, placed)
    compile_seconds = time.perf_counter() - started
    if objective.model.remat is not None or trainer.host_master:
        raise RuntimeError("fit ladder changed the matched no-checkpointing, device-resident workload")
    return trainer, state, compiled, placed, compile_seconds, work


def train(args, trainer, state, compiled, batch):
    """One asynchronous window, then per-step synchronized diagnostics outside its peak scope."""
    losses, finite = [], []
    first_before = np.asarray(jax.tree.leaves(state.variables)[0]).copy() if args.smoke else None
    for _ in range(args.warmup):
        state, loss, _, valid, _, _ = compiled(state, batch)
        losses.append(loss)
        finite.append(valid)
    jax.block_until_ready(state)
    started = time.perf_counter()
    for _ in range(args.steps):
        state, loss, _, valid, _, _ = compiled(state, batch)
        losses.append(loss)
        finite.append(valid)
    jax.block_until_ready(state)
    seconds = time.perf_counter() - started
    peak = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
    if not args.smoke and peak is None:
        raise RuntimeError("GPU allocator supplied no peak_bytes_in_use")
    samples = []
    for _ in range(args.steps):
        started = time.perf_counter()
        state, loss, _, valid, _, _ = compiled(state, batch)
        jax.block_until_ready(state)
        samples.append(time.perf_counter() - started)
        losses.append(loss)
        finite.append(valid)
    if not np.isfinite(np.asarray(losses)).all() or not np.asarray(finite).all():
        raise RuntimeError("non-finite loss or update")
    if args.smoke and np.array_equal(first_before, np.asarray(jax.tree.leaves(state.variables)[0])):
        raise RuntimeError("tiny proof failed to update the backbone")
    parameters = jax.tree.leaves(state.variables["params"])
    if any(leaf.dtype != jnp.float32 for leaf in parameters):
        raise RuntimeError("all master parameters must be fp32")
    floats = [leaf for leaf in jax.tree.leaves(state.opt_state) if jnp.issubdtype(leaf.dtype, jnp.floating)]
    if not floats or any(leaf.dtype != jnp.float32 for leaf in floats):
        raise RuntimeError("all optimizer moments must be fp32")
    return seconds, peak, samples, {
        "loss_first": float(losses[0]), "loss_last": float(losses[-1]), "finite_losses": True,
        "optimizer_state_dtypes": ["float32"], "no_timing_window_compilation": True,
        "parameter_count": sum(leaf.size for leaf in parameters),
    }


def run(args) -> dict:
    prepare_process(multi_host=False)
    trainer, state, compiled, batch, compile_seconds, work = build(args)
    seconds, peak_bytes, samples, checks = train(args, trainer, state, compiled, batch)
    rows, sequence = batch["tokens"].shape
    model = trainer.objective.model
    device_name = "CPU smoke (no MFU)" if args.smoke else jax.local_devices()[0].device_kind
    peak = None if args.smoke else peak_flops(device_name)
    median = statistics.median(samples)
    timed = throughput(seconds, args.steps, rows * sequence, samples)
    record = {
        "schema": "dew.attribution.torch-reference.v1", "framework": "dew", "smoke": args.smoke,
        "work": dataclasses.asdict(work),
        "session": args.session, "commit": git_head(Path.cwd()),
        "script_commit": git_head(Path(__file__).resolve().parents[2]), "host": host(),
        "environment": {"device": device_name,
                        "runtime": {name: os.environ.get(name) for name in
                                    ("XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                                     "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_PLATFORMS")}, "packages": {
            dist.metadata["Name"].lower().replace("_", "-"): dist.version
            for dist in importlib.metadata.distributions()}},
        "config": {**workload(args.seed, model.vocab_size, model.emb_features, rows, sequence,
                               args.smoke, args.warmup, args.steps),
                   "parameter_count": checks["parameter_count"], "compile_backend": "Trainer / XLA"},
        "steady": {**timed, "window_tokens_per_s": timed["tokens_per_s"], "step_ms": median * 1000,
                   "tokens_per_s": rows * sequence / median,
                   "mfu": None if peak is None else work.training_flops_per_step / median / peak,
                   "window_mfu": None if peak is None else work.training_flops_per_step
                       / (seconds / args.steps) / peak,
                   "step_ms_samples": [sample * 1000 for sample in samples],
                   "samples_scope": "diagnostic synchronized pass after timing; not gate samples"},
        "memory": {"whole_peak_bytes": peak_bytes, "whole_peak_scope": WHOLE_PEAK_SCOPE,
                   "whole_peak_metric": "jax.local_devices()[0].memory_stats()['peak_bytes_in_use']",
                   "peak_allocated_bytes": peak_bytes,
                   "peak_allocated_gib": None if peak_bytes is None else peak_bytes / 2**30,
                   "scope": WHOLE_PEAK_SCOPE},
        "time": {"compile_seconds": compile_seconds,
                 "model_flops_per_step": work.training_flops_per_step,
                 "flops_formula": work.basis},
        "checks": checks,
    }
    write_record(args.out, record)
    return record


if __name__ == "__main__":
    run(arguments())
