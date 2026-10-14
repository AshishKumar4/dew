#!/usr/bin/env python3
"""Compare row and token budgets through the real decision recipe and Trainer step."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import queue
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, TrainerConfig
from dew.data.dataset import (
    Batch,
    Checkpointable,
    Closeable,
    DataPartition,
    Loader,
    Loading,
    Reader,
    Stoppable,
)
from dew.decision.config import DecisionRunConfig, JointShape
from dew.decision.data import Decisions
from dew.decision.scoring import Brier, LogLoss, RankedProbability
from dew.interop import sources
from dew.objectives.base import Metric, Objective
from dew.telemetry.profile import Profiler, region
from dew.training import Trainer
from dew.training.distributed import DevicePrefetchIterator, Signature, batch_signature
from dew.training.runtime import prepare_process
from dew.training.state import TrainState
from dew.training.trainer import CompiledStep, Rollout

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


class Counted:
    def __init__(self, parent: Iterator[Batch]):
        self.parent = parent
        self.counts: queue.Queue[dict[str, float]] = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        started = time.perf_counter()
        batch = next(self.parent)
        lengths = np.sum(batch["valid"], axis=1, dtype=np.int64)
        self.counts.put({
            "source_read_seconds": time.perf_counter() - started,
            "real_tokens": int(np.sum(lengths)),
            "sum_lengths_squared": int(np.sum(np.square(lengths))),
            "padded_tokens": int(np.size(batch["tokens"])),
            "rows": len(lengths),
        })
        return batch

    def get_state(self):
        if not isinstance(self.parent, Checkpointable):
            raise TypeError("the benchmark requires an exact-position decision reader")
        return self.parent.get_state()

    def set_state(self, state) -> None:
        if not isinstance(self.parent, Checkpointable):
            raise TypeError("the benchmark requires an exact-position decision reader")
        self.parent.set_state(state)

    def request_stop(self) -> None:
        if isinstance(self.parent, Stoppable):
            self.parent.request_stop()

    def close(self) -> None:
        if isinstance(self.parent, Closeable):
            self.parent.close()


def model_flops(config: Mapping[str, object], real_tokens: float, squared_lengths: float) -> float:
    """Counted's recurrent convention, without padding, decision head or recomputation."""
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise TypeError("the model config needs a text_config record")
    width, inner = int(text["hidden_size"]), int(text["intermediate_size"])
    heads, kv, head = (int(text[name]) for name in
                       ("num_attention_heads", "num_key_value_heads", "head_dim"))
    gh, gk, gv, keys = (int(text[name]) for name in
                        ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
                         "linear_num_key_heads"))
    attention = 3 * width * heads * head + 2 * width * kv * head
    delta = width * (2 * keys * gk + 2 * gh * gv + 2 * gh) + gh * gv * width
    convolution = 2 * (2 * keys * gk + gh * gv) * int(text["linear_conv_kernel_dim"])
    total = 0.0
    for kind in text["layer_types"]:
        projection = attention if kind == "full_attention" else delta
        total += 6 * (projection + 3 * width * inner) * real_tokens
        total += (6 * heads * head * squared_lengths if kind == "full_attention" else
                  3 * (6 * gh * gk * gv + convolution) * real_tokens)
    return total


class Measurement:
    def __init__(self, trainer: Trainer, counted: Counted, config: BenchmarkRun):
        self.trainer, self.counted, self.config = trainer, counted, config
        self.compiled: dict[Signature, CompiledStep] = {}
        self.hardware: dict[Signature, float | None] = {}
        self.compiles: list[dict[str, object]] = []
        self.rows: list[dict[str, float]] = []
        self.started = time.perf_counter()
        self.first_step_seconds: float | None = None
        self.flop_config = None if config.smoke else json.loads(
            (sources.snapshot(config.pretrained, config.revision, weights=False) / "config.json").read_text())

    def step(self, state: TrainState, reader: DevicePrefetchIterator, index: int) -> TrainState:
        started = time.perf_counter()
        with region("input.wait"):
            batch = next(reader)
        wait = time.perf_counter() - started
        counted = self.counted.counts.get_nowait()
        shape = batch_signature(batch)
        compile_seconds = 0.0
        if shape not in self.compiled:
            began = time.perf_counter()
            with region("compile"):
                self.compiled[shape] = self.trainer.compile(state, batch)
            compile_seconds = time.perf_counter() - began
            self.hardware[shape] = (self.trainer.step_flops and self.trainer.step_flops.hardware)
            self.compiles.append({
                "microstep": index, "shape": list(batch["tokens"].shape), "seconds": compile_seconds,
                "hlo_flops": self.hardware[shape], "memory": self.trainer.memory,
            })
        with region("train.step"):
            state, loss, _, finite, accepted = self.compiled[shape](state, batch)
        loss.block_until_ready()
        values = jax.device_get((loss, finite, accepted))
        if not bool(values[1]) or not bool(values[2]) or not math.isfinite(float(values[0])):
            raise ValueError(f"nonfinite or rejected training microstep {index}")
        if self.first_step_seconds is None:
            self.first_step_seconds = time.perf_counter() - self.started
        self.rows.append({
            **counted, "microstep": index, "loss": float(values[0]), "input_wait_seconds": wait,
            "compile_seconds": compile_seconds, "seconds": time.perf_counter() - started,
            "hlo_flops": self.hardware[shape] or 0.0,
            "hlo_flops_available": float(self.hardware[shape] is not None),
            "model_flops": 0.0 if self.flop_config is None else model_flops(
                self.flop_config, counted["real_tokens"], counted["sum_lengths_squared"]),
        })
        return state

    def windows(self) -> list[dict[str, float | None]]:
        accumulation = self.config.trainer.accumulation
        warmup = self.config.warmup_updates * accumulation
        size = self.config.timed_updates * accumulation
        results = []
        for window in range(self.config.windows):
            rows = self.rows[warmup + window * size:warmup + (window + 1) * size]
            if not rows:
                continue
            inclusive = sum(row["seconds"] for row in rows)
            compiles = sum(row["compile_seconds"] for row in rows)
            seconds = inclusive - compiles
            real = sum(row["real_tokens"] for row in rows)
            middle = len(rows) // 2
            before = statistics.mean(row["seconds"] - row["compile_seconds"] for row in rows[:middle])
            after = statistics.mean(row["seconds"] - row["compile_seconds"] for row in rows[middle:])
            results.append({
                "updates": len(rows) / accumulation, "seconds": seconds, "inclusive_seconds": inclusive,
                "compile_seconds": compiles, "real_tokens": real, "real_tokens_per_second": real / seconds,
                "ms_per_update": seconds * 1000 * accumulation / len(rows),
                "input_wait_fraction": sum(row["input_wait_seconds"] for row in rows) / seconds,
                "padding_fraction": 1 - real / sum(row["padded_tokens"] for row in rows),
                "drift_fraction": after / before - 1,
                "model_mfu": (None if self.config.smoke else
                              sum(row["model_flops"] for row in rows) / seconds / 312e12),
                "hfu": (None if self.config.smoke or not all(row["hlo_flops_available"] for row in rows)
                        else sum(row["hlo_flops"] for row in rows) / seconds / 312e12),
            })
        return results

    def record(self, state: TrainState, peak: int | None) -> dict[str, object]:
        expected = len(self.rows) // self.config.trainer.accumulation
        if int(state.updates) != expected:
            raise ValueError(f"committed {int(state.updates)} optimizer updates, expected {expected}")
        leaves = jax.tree.leaves((state.variables, state.opt_state))
        if any(leaf.dtype != jnp.float32 for leaf in jax.tree.leaves(state.variables["params"])):
            raise ValueError("the comparison requires fp32 master parameters")
        if any(leaf.dtype != jnp.float32 for leaf in jax.tree.leaves(state.opt_state)
               if jnp.issubdtype(leaf.dtype, jnp.floating)):
            raise ValueError("the comparison requires fp32 optimizer moments")
        if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves):
            raise ValueError("nonfinite final parameters or optimizer state")
        measured = self.rows[self.config.warmup_updates * self.config.trainer.accumulation:]
        return {
            "arm": self.config.arm, "smoke": self.config.smoke,
            "loss_normalization": (
                "per real token" if self.config.trainer.batch_tokens else "per scored question"),
            "analytic_mfu_scope": "Counted recurrent trunk work at real lengths; head and remat excluded",
            "hfu_scope": "compiled HLO includes padding, head and recomputation; separate from model MFU",
            "model_flops_per_step": statistics.mean(row["model_flops"] for row in measured),
            "whole_run_compile_seconds": sum(float(row["seconds"]) for row in self.compiles),
            "compile_count": len(self.compiles), "compiles": self.compiles,
            "first_step_seconds": self.first_step_seconds, "windows": self.windows(), "microsteps": self.rows,
            "first_step_scope": "after state placement: reader first batch and first compile",
            "live_peak_bytes": peak,
            "live_peak_scope": "process through timed end; excludes finite-state diagnostics",
            "checks": {"finite_losses_and_final_state": True, "fp32_parameters_and_moments": True,
                       "optimizer_updates": expected},
        }


@dataclasses.dataclass(frozen=True)
class BenchmarkRun(DecisionRunConfig):
    output: str = "budget-benchmark.json"
    arm: Literal["rows", "tokens"] = "tokens"
    warmup_updates: int = 32
    timed_updates: int = 128
    windows: int = 3
    smoke: bool = False
    profile_directory: str | None = None

    def train[Loss, Effects](
        self, objective: Objective[Loss, Effects], dataset: Loader, *, name: str,
        metrics: Sequence[Metric] = (), rollout: Rollout | None = None,
        summary: Mapping[str, object] | None = None, validation: Mapping[str, Reader] | None = None,
        after: Callable[[TrainState, str], None] | None = None,
    ) -> TrainState:
        updates = self.warmup_updates + self.timed_updates * self.windows
        trainer = Trainer(objective, self.optim.build(updates), key=self.trainer.key,
                          mesh=self.trainer.mesh, layout=self.trainer.layout,
                          accumulation=self.trainer.accumulation, checkpoints=None, tracker=None)
        state, _, _ = trainer.place()
        jax.block_until_ready(state)
        counted = Counted(iter(dataset.train(DataPartition.of(trainer.device_mesh))))
        measured = Measurement(trainer, counted, self)
        with contextlib.ExitStack() as resources:
            reader = resources.enter_context(DevicePrefetchIterator(counted, trainer.device_mesh))
            for index in range(updates * self.trainer.accumulation):
                if self.profile_directory and index == self.warmup_updates * self.trainer.accumulation:
                    resources.enter_context(Profiler(self.profile_directory))
                with jax.profiler.StepTraceAnnotation("train", step_num=index):
                    state = measured.step(state, reader, index)
        peak = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
        if not self.smoke and peak is None:
            raise ValueError("the A100 must supply peak_bytes_in_use")
        result = measured.record(state, peak)
        result["data_plan"] = None if dataset.plan is None else dataset.plan.record()
        result["config"] = self.to_dict()
        Path(self.output).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return state


def configuration(args: argparse.Namespace) -> BenchmarkRun:
    tokens = args.arm == "tokens"
    micro = 2048 if args.smoke else 24576
    layout = 512 if args.smoke else 3072
    rows = tuple({"state": "ticket " + "charged twice " * (index % 6 + 1), "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {
            "billing": "payments", "technical": "outages", "sales": "pricing"}}},
        "answers": {"team": "billing"}} for index in range(48))
    return BenchmarkRun(
        output=str(args.output), arm=args.arm, smoke=args.smoke,
        warmup_updates=args.warmup_updates, timed_updates=args.timed_updates, windows=args.windows,
        profile_directory=args.profile_directory,
        pretrained="tests/fixtures/hf/qwen38-dense-tiny" if args.smoke else MODEL,
        revision=None if args.smoke else REVISION, param_dtype="float32", lora=None,
        head=(JointShape(width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
              if args.smoke else JointShape()),
        model=ModelConfig("causal_transformer", {"dtype": "float32" if args.smoke else "bfloat16"}),
        objective=ObjectiveConfig("decision", {
            "loss": LogLoss() + 0.5 * Brier() + RankedProbability(), "label_smoothing": 0.05,
            "bucket": None if tokens else (64 if args.smoke else 256), "shuffle_options": True}),
        data=Decisions(rows) if args.smoke else Decisions(args.source, descriptor=args.descriptor),
        data_read="memory" if args.smoke else "indexed", data_seed=0,
        loading=Loading(workers=args.workers, threads=4, read_buffer=16),
        trainer=TrainerConfig(batch_size=None if tokens else (4 if args.smoke else 8),
                              batch_tokens=micro if tokens else None,
                              update_tokens=micro * args.accumulation if tokens else None,
                              accumulation=args.accumulation, key=0, steps=1, eval_every=None,
                              checkpoint_every=None, log_every=1, multi_host=False,
                              compilation_cache_dir=None),
        optim=OptimConfig(learning_rate=2.5e-5, state_dtype="float32"), calibrate=False,
        max_len=layout, max_state_tokens=64 if args.smoke else 1024,
        bucket_lengths=(((128, 256, 512) if args.smoke else (512, 768, 1024, 1536, 2048, 3072))
                        if tokens else ()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("rows", "tokens"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup-updates", type=int, default=32)
    parser.add_argument("--timed-updates", type=int, default=128)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--profile-directory")
    parser.add_argument("--source", help="the decision mixture's root URL, for the full run")
    parser.add_argument("--descriptor", help="the URL of its immutable source descriptor, for the full run")
    args = parser.parse_args()
    if min(args.warmup_updates, args.timed_updates, args.windows, args.accumulation) < 1:
        parser.error("warmup, timing, windows and accumulation must be positive")
    if not args.smoke and (args.source is None or args.descriptor is None):
        parser.error("the full run reads the decision mixture --source names, through --descriptor")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepare_process(multi_host=False, compilation_cache_dir=None)
    device = jax.local_devices()[0]
    if len(jax.local_devices()) != 1 or (args.smoke and device.platform != "cpu"):
        parser.error("this comparison needs one device; the smoke must use armada CPU")
    if not args.smoke and ("A100" not in device.device_kind or "80GB" not in device.device_kind):
        parser.error(f"the approved full comparison needs A100-80GB, found {device.device_kind}")
    configuration(args).prepare().train(args.arm)


if __name__ == "__main__":
    main()
