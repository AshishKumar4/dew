#!/usr/bin/env python3
"""Compile time and step time of the decoder's layer stack, by depth and by
how the stack runs: the plain loop, flax's scan over runs of like layers,
and the pipeline over the stage axis.

Every case trains a causal_transformer of the given depth at one small width
through the real Trainer step (the objective, the sharding and the donated
state of a run), so the compile time is the step's, forward and backward,
and the executables XLA built for it are counted from jax's own compile log.

Usage:
    PYTHONPATH=src python tools/benchmark_stack.py
    PYTHONPATH=src python tools/benchmark_stack.py --depths 24 48 --scan True
    XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=src \\
        python tools/benchmark_stack.py --fsdp 4 --stage 2 --microbatches 4
    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 PYTHONPATH=src \\
        python tools/benchmark_stack.py --dtype bfloat16 --attention-impl cudnn
"""

import contextlib
import io
import json
import logging
import time
from dataclasses import dataclass, field

import jax
import optax
import tyro

from benchmark_step import Case, batches, parameter_count
from dew import models  # naming a registry fills it
from dew.objectives.lm import LMObjective
from dew.registry import with_precision
from dew.training import Layout, MeshSpec, Trainer
from dew.training.distributed import DevicePrefetchIterator

# One width for every depth, small enough that the step is compile-bound on
# a CPU and the depth is the only thing that changes between cases.
WIDTH = {"vocab_size": 512, "emb_features": 64, "num_heads": 4, "num_kv_heads": 2,
         "mlp_features": 128, "max_seq_len": 64}
SEQ_LEN = 64
BATCH = 8


@dataclass(frozen=True)
class StackConfig:
    """Which depths and modes to time, and how."""

    depths: list[int] = field(default_factory=lambda: [24, 48])
    scan: list[bool] = field(default_factory=lambda: [False, True])
    """The scan_layers settings to time at every depth."""
    stage: int = 1
    """Pipeline stages; above 1 the stack runs as a pipeline over the stage axis."""
    microbatches: int | None = None
    fsdp: int = 1
    steps: int = 20
    warmup: int = 2
    dtype: str = "float32"
    attention_impl: str = "xla"
    json_out: str | None = None


class CompileCounter(logging.Handler):
    """Counts the training steps XLA compiled, off jax's compile log."""

    def __init__(self):
        super().__init__()
        self.steps = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "Compiling jit(step)" in record.getMessage():
            self.steps += 1


def build_trainer(case: Case, config: StackConfig) -> Trainer:
    model = models.build(case.architecture, **with_precision(
        case.architecture, case.config, dtype=config.dtype,
        attention_impl=config.attention_impl))
    return Trainer(
        LMObjective(model, case.seq_len), optax.adam(1e-4), key=jax.random.key(0),
        mesh=MeshSpec(fsdp=config.fsdp, stage=config.stage, microbatches=config.microbatches),
        layout=Layout(min_shard=2 ** 8), checkpoints=None, tracker=None)


def measure(depth: int, scan: bool, config: StackConfig, counter: CompileCounter) -> dict:
    """One case: build, compile the step, time it, and count its compilations."""
    case = Case(architecture="causal_transformer",
                config={**WIDTH, "num_layers": depth, "scan_layers": scan},
                dtype=config.dtype, batch_size=BATCH, seq_len=SEQ_LEN)
    trainer = build_trainer(case, config)
    with DevicePrefetchIterator(batches(case), trainer.device_mesh) as source:
        abstract = jax.eval_shape(trainer.initial_state)
        state = jax.jit(trainer.initial_state, out_shardings=trainer.shardings(abstract))()
        counter.steps = 0

        started = time.perf_counter()
        compiled = trainer.compile(state, next(source))
        compile_seconds = time.perf_counter() - started

        def step(state):
            state, _, loss, _, _ = compiled(state, None, next(source))
            return state, loss

        # One warm step before the timed window; the first dispatch of the
        # executable stays outside it.
        state, loss = step(state)
        for _ in range(config.warmup - 1):
            state, loss = step(state)
        loss.block_until_ready()
        started = time.perf_counter()
        for _ in range(config.steps):
            state, loss = step(state)
        loss.block_until_ready()
        elapsed = time.perf_counter() - started
        return {
            "depth": depth,
            "scan_layers": scan,
            "stage": config.stage,
            "microbatches": config.microbatches,
            "fsdp": config.fsdp,
            "params": parameter_count(state.params),
            "compile_seconds": round(compile_seconds, 2),
            "compilations": counter.steps,
            "ms_per_step": round(elapsed / config.steps * 1e3, 2),
            "loss": float(loss),
            "device_kind": jax.devices()[0].device_kind,
        }


def main(config: StackConfig) -> list[dict]:
    jax.config.update("jax_log_compiles", True)
    counter = CompileCounter()
    logging.getLogger("jax._src.interpreters.pxla").addHandler(counter)
    rows = []
    for depth in config.depths:
        for scan in config.scan:
            # The trainer narrates the state and the shapes; the numbers are
            # what this tool prints.
            with contextlib.redirect_stdout(io.StringIO()):
                row = measure(depth, scan, config, counter)
            rows.append(row)
            print(f"depth {depth} scan_layers {scan} stage {config.stage}: compile "
                  f"{row['compile_seconds']} s in {row['compilations']} compilation(s), "
                  f"{row['ms_per_step']} ms/step, loss {row['loss']:.4f}")
    if config.json_out:
        with open(config.json_out, "w") as handle:
            json.dump(rows, handle, indent=2)
    return rows


if __name__ == "__main__":
    main(tyro.cli(StackConfig))
