"""Measure one vocabulary-head forward and backward variant per process.

`peak_bytes_in_use` is a process high-water mark, so a row is one process: run
this once per variant and read the JSON it prints.

The variants are baseline (one full-vocabulary logits tensor), stored (the
Python vocabulary loop holding its tiles), remat (that loop under
`jax.checkpoint`) and bounded (the shipped recomputing custom VJP), each with
an optional chunk count after the name (`bounded8`; 4 when absent) and the
suffixes -noacc, which drops the top-1 prediction to price the metric,
-fp32, which keeps the states in fp32, and -bf16, which multiplies the head
as bf16 (`CausalTransformer.bf16_head`). The table is held `[vocab, features]`,
the layout `embed_tokens.embedding` has, so the transpose the loss needs is
inside the measurement and not hidden by the setup.

`--token-tile` and `--vocab-tile` are the bounded backward's block and default
to the loss's own. The comparison the shipped default comes from is two runs:

    PYTHONPATH=src python tools/benchmark_lm_head.py remat4-fp32 \
        --batch 1 --sequence 1024 --features 2816 --vocab 262144 --softcap 30
    PYTHONPATH=src python tools/benchmark_lm_head.py bounded4-fp32 \
        --batch 1 --sequence 1024 --features 2816 --vocab 262144 --softcap 30

`process_peak_delta_bytes` includes setup and only ever grows, so it is an
upper bound; the compiled executable's `temp_size_in_bytes` is the temporary
figure. The full head gradient is an output of every variant, not a saving.
"""

import argparse
import inspect
import json
import time
from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp

from dew.objectives.lm.chunked import (
    chunked_cross_entropy,
    head_logits,
    vocabulary_chunks,
)

# The loss owns the measured default; reading it keeps one source for it.
DEFAULT_TILE = inspect.signature(chunked_cross_entropy).parameters["tile"].default


@dataclass(frozen=True)
class Variant:
    text: str
    head: str
    chunks: int
    accuracy: bool
    states_dtype: jnp.dtype
    bf16: bool = False
    softcap: float | None = None
    z_loss: float = 0.0
    tile: tuple[int, int] = DEFAULT_TILE


@dataclass(frozen=True)
class Shape:
    batch: int = 16
    sequence: int = 512
    features: int = 768
    vocab: int = 50304
    repeats: int = 50


def finish(losses, predicted, log_z, targets, variant: Variant):
    loss = jnp.mean(losses + variant.z_loss * jnp.square(log_z))
    accuracy = (
        jnp.mean((predicted == targets).astype(losses.dtype))
        if variant.accuracy
        else jnp.zeros((), jnp.float32)
    )
    return loss, accuracy


def baseline(states, table, targets, variant: Variant):
    logits = head_logits(states, table.T, softcap=variant.softcap, precision=None)
    log_z = jax.nn.logsumexp(logits, axis=-1)
    picked = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return finish(log_z - picked, jnp.argmax(logits, axis=-1), log_z, targets, variant)


def bounded(states, table, targets, variant: Variant):
    losses, predicted, log_z = chunked_cross_entropy(
        states, table.T, targets, variant.chunks, softcap=variant.softcap,
        tile=variant.tile, predict=variant.accuracy, bf16=variant.bf16
    )
    return finish(losses, predicted, log_z, targets, variant)


def retained(states, table, targets, variant: Variant):
    """The pre-bounded vocabulary loop, held here so the baseline does not
    move when the shipped kernel does; `remat` is it under `jax.checkpoint`."""
    flat = states.astype(jnp.float32).reshape(-1, states.shape[-1])
    labels = targets.reshape(-1)
    total = jnp.full(labels.shape, -jnp.inf, jnp.float32)
    target_logit = jnp.zeros(labels.shape, jnp.float32)
    best = jnp.full(labels.shape, -jnp.inf, jnp.float32)
    predicted = jnp.zeros(labels.shape, jnp.int32)

    def tile(hidden, matrix, column):
        logits = head_logits(hidden, matrix, softcap=variant.softcap, precision=None)
        picked = jnp.take_along_axis(logits, column[:, None], axis=-1)[:, 0]
        return (
            jax.nn.logsumexp(logits, axis=-1),
            picked,
            jnp.max(logits, axis=-1),
            jnp.argmax(logits, axis=-1),
        )

    if variant.head == "remat":
        tile = jax.checkpoint(tile)
    for start, stop in vocabulary_chunks(table.shape[0], variant.chunks):
        inside = (labels >= start) & (labels < stop)
        column = jnp.clip(labels - start, 0, stop - start - 1)
        chunk_z, picked, chunk_best, chunk_column = tile(
            flat, table[start:stop].T, column
        )
        total = jnp.logaddexp(total, chunk_z)
        target_logit = target_logit + jnp.where(inside, picked, 0.0)
        better = chunk_best > best
        best = jnp.where(better, chunk_best, best)
        predicted = jnp.where(better, chunk_column + start, predicted)
    return finish(
        (total - target_logit).reshape(targets.shape),
        predicted.reshape(targets.shape),
        total.reshape(targets.shape),
        targets,
        variant,
    )


Head = Callable[[jax.Array, jax.Array, jax.Array, Variant], tuple[jax.Array, jax.Array]]
HEADS: dict[str, Head] = {
    "baseline": baseline,
    "stored": retained,
    "remat": retained,
    "bounded": bounded,
}
SUFFIXES = ("noacc", "fp32", "bf16")


def parse_variant(text: str) -> Variant:
    base, *suffixes = text.split("-")
    head = base.rstrip("0123456789")
    if head not in HEADS:
        raise ValueError(f"{text!r} does not start with one of {sorted(HEADS)}")
    unknown = sorted(set(suffixes) - set(SUFFIXES))
    if unknown:
        raise ValueError(
            f"{text!r} carries unknown suffixes {unknown}; valid: {list(SUFFIXES)}"
        )
    return Variant(
        text,
        head,
        int(base[len(head) :] or "4"),
        accuracy="noacc" not in suffixes,
        states_dtype=jnp.dtype(jnp.float32 if "fp32" in suffixes else jnp.bfloat16),
        bf16="bf16" in suffixes,
    )


def timed(callable_, args, repeats: int) -> float:
    """Steady wall latency with no overlapping result buffers between calls."""
    for _ in range(3):
        out = jax.block_until_ready(callable_(*args))
        del out
    start = time.perf_counter()
    for _ in range(repeats):
        out = jax.block_until_ready(callable_(*args))
        del out
    return (time.perf_counter() - start) / repeats * 1e3


def executable_bytes(executable) -> dict[str, int] | None:
    analysis = executable.memory_analysis()
    if analysis is None:
        return None
    return {
        name: getattr(analysis, name)
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
        )
    }


def measure(variant: Variant, shape: Shape) -> dict[str, object]:
    if min(shape.batch, shape.sequence, shape.features, shape.vocab, shape.repeats) < 1:
        raise ValueError("All workload dimensions and repeats must be positive")
    vocabulary_chunks(shape.vocab, variant.chunks)
    hidden = jax.random.normal(
        jax.random.PRNGKey(0),
        (shape.batch, shape.sequence, shape.features),
        variant.states_dtype,
    )
    embedding = (
        jax.random.normal(
            jax.random.PRNGKey(1), (shape.vocab, shape.features), jnp.float32
        )
        * 0.02
    )
    targets = jax.random.randint(
        jax.random.PRNGKey(2), (shape.batch, shape.sequence), 0, shape.vocab
    )
    jax.block_until_ready((hidden, embedding, targets))
    function = HEADS[variant.head]
    device = jax.local_devices()[0]
    stats = device.memory_stats()
    before = stats.get("peak_bytes_in_use") if stats else None

    forward = jax.jit(lambda states, table: function(states, table, targets, variant))
    both = jax.jit(
        jax.value_and_grad(
            lambda states, table: function(states, table, targets, variant),
            argnums=(0, 1),
            has_aux=True,
        )
    )
    start = time.perf_counter()
    forward_executable = forward.lower(hidden, embedding).compile()
    both_executable = both.lower(hidden, embedding).compile()
    compile_seconds = time.perf_counter() - start
    args = (hidden, embedding)
    forward_ms = timed(forward_executable, args, shape.repeats)
    both_ms = timed(both_executable, args, shape.repeats)
    (loss, accuracy), (d_states, d_table) = jax.block_until_ready(
        both_executable(*args)
    )
    stats = device.memory_stats()
    peak = stats.get("peak_bytes_in_use") if stats else None

    return {
        "scope": "vocabulary_head_only",
        "timing": "steady_wall_synchronized_per_invocation",
        "variant": variant.text,
        "head": variant.head,
        "chunks": variant.chunks,
        "shape": {
            "batch": shape.batch,
            "sequence": shape.sequence,
            "features": shape.features,
            "vocab": shape.vocab,
        },
        "softcap": variant.softcap,
        "z_loss": variant.z_loss,
        "tile": list(variant.tile) if variant.head == "bounded" else None,
        "accuracy": variant.accuracy,
        "states_dtype": variant.states_dtype.name,
        "head_dtype": str(embedding.dtype),
        "jax": jax.__version__,
        "default_matmul_precision": jax.config.jax_default_matmul_precision,
        "device": str(device),
        "device_kind": device.device_kind,
        "repeats": shape.repeats,
        "compile_seconds": compile_seconds,
        "forward_ms": forward_ms,
        "forward_backward_ms": both_ms,
        "process_peak_bytes": peak,
        "setup_peak_bytes": before,
        "process_peak_delta_bytes": None
        if peak is None or before is None
        else max(0, peak - before),
        "forward_executable_bytes": executable_bytes(forward_executable),
        "forward_backward_executable_bytes": executable_bytes(both_executable),
        "required_head_gradient_bytes": d_table.size * d_table.dtype.itemsize,
        "required_hidden_gradient_bytes": d_states.size * d_states.dtype.itemsize,
        "loss": float(loss),
        "token_accuracy": float(accuracy),
        "d_states_sum": float(jnp.sum(d_states.astype(jnp.float32))),
        "d_table_sum": float(jnp.sum(d_table)),
        "d_states_absmax": float(jnp.abs(d_states.astype(jnp.float32)).max()),
        "d_table_absmax": float(jnp.abs(d_table).max()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "variant",
        type=parse_variant,
        help="baseline/stored/remat/bounded, count, -noacc, -fp32",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--features", type=int, default=768)
    parser.add_argument("--vocab", type=int, default=50304)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--softcap", type=float)
    parser.add_argument("--z-loss", type=float, default=0.0)
    parser.add_argument("--token-tile", type=int, default=DEFAULT_TILE[0])
    parser.add_argument("--vocab-tile", type=int, default=DEFAULT_TILE[1])
    args = parser.parse_args(argv)
    variant = replace(args.variant, softcap=args.softcap, z_loss=args.z_loss,
                      tile=(args.token_tile, args.vocab_tile))
    shape = Shape(args.batch, args.sequence, args.features, args.vocab, args.repeats)
    print(json.dumps(measure(variant, shape)))


if __name__ == "__main__":
    main()
