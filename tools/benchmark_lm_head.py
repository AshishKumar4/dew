"""Time the vocabulary head and its loss alone, one variant per process.

`peak_bytes_in_use` is a process high-water mark, so a row is one process:
run this once per variant and read the delta it prints.

The shape is the head work a training step does at 8,192 tokens (batch 16 x
sequence 512), width 768, vocabulary 50,304, bf16 states with an fp32 head and
an fp32 loss, forward and backward. The head is held as `[vocab, width]`, the
layout `embed_tokens.embedding` has, so the transpose the chunked path needs
is inside the measurement and not hidden by the setup.

  baseline    states -> full fp32 logits -> optax cross entropy plus the
              argmax accuracy, the unchunked head
  stored      dew.objectives.lm.chunked.chunked_cross_entropy, which keeps
              the tiles for the backward pass
  remat       the same loop with jax.checkpoint around the tile, which
              recomputes it in the backward pass instead

A variant is one of those names with an optional chunk count after it
(`stored8`; 4 when absent), then optional suffixes: `-noacc` drops the top-1
prediction, to price the metric, and `-fp32` keeps the states in fp32, so
the state gradient can be compared without bf16 rounding in the way.

Usage: PYTHONPATH=<checkout>/src python tools/benchmark_lm_head.py stored4-noacc
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from dew.objectives.lm.chunked import chunked_cross_entropy, vocabulary_chunks

B, S, D, V = 16, 512, 768, 50304
REPEATS = 50


@dataclass(frozen=True)
class Variant:
    """One row of the table, parsed from its name."""

    text: str
    head: str
    chunks: int
    accuracy: bool
    states_dtype: jnp.dtype


def baseline(states, table, targets, variant: Variant):
    """The unchunked head: one full-vocabulary logits tensor, then the loss."""
    logits = jnp.einsum('...d,vd->...v', states.astype(jnp.float32),
                        table.astype(jnp.float32))
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    correct = (jnp.argmax(logits, axis=-1) == targets).astype(losses.dtype)
    return jnp.mean(losses), jnp.mean(correct)


def stored(states, table, targets, variant: Variant):
    """The shipped kernel: the tiles stay live for the backward pass."""
    losses, predicted = chunked_cross_entropy(states, table.T, targets, variant.chunks)
    if not variant.accuracy:
        return jnp.mean(losses), jnp.zeros(())
    return jnp.mean(losses), jnp.mean((predicted == targets).astype(losses.dtype))


def remat(states, table, targets, variant: Variant):
    """The same arithmetic with each tile recomputed in the backward pass."""
    head = table.T
    flat = states.astype(jnp.float32).reshape(-1, states.shape[-1])
    flat_targets = targets.reshape(-1)
    total = jnp.full(flat_targets.shape, -jnp.inf, jnp.float32)
    target_logit = jnp.zeros(flat_targets.shape, jnp.float32)
    best = jnp.full(flat_targets.shape, -jnp.inf, jnp.float32)
    predicted = jnp.zeros(flat_targets.shape, jnp.int32)

    @jax.checkpoint
    def tile(hidden, head_chunk, column):
        logits = jnp.einsum('td,dv->tv', hidden, head_chunk,
                            preferred_element_type=jnp.float32)
        picked = jnp.take_along_axis(logits, column[:, None], axis=-1)[:, 0]
        return (jax.nn.logsumexp(logits, axis=-1), picked,
                jnp.max(logits, axis=-1), jnp.argmax(logits, axis=-1))

    for start, stop in vocabulary_chunks(head.shape[-1], variant.chunks):
        inside = (flat_targets >= start) & (flat_targets < stop)
        column = jnp.clip(flat_targets - start, 0, stop - start - 1)
        chunk_lse, picked, chunk_best, chunk_column = tile(
            flat, head[:, start:stop], column)
        total = jnp.logaddexp(total, chunk_lse)
        target_logit = target_logit + jnp.where(inside, picked, 0.0)
        if variant.accuracy:
            better = chunk_best > best
            best = jnp.where(better, chunk_best, best)
            predicted = jnp.where(better, chunk_column + start, predicted)
    losses = (total - target_logit).reshape(targets.shape)
    if not variant.accuracy:
        return jnp.mean(losses), jnp.zeros(())
    correct = (predicted.reshape(targets.shape) == targets).astype(losses.dtype)
    return jnp.mean(losses), jnp.mean(correct)


Head = Callable[[jax.Array, jax.Array, jax.Array, Variant], tuple[jax.Array, jax.Array]]
HEADS: dict[str, Head] = {"baseline": baseline, "stored": stored, "remat": remat}
SUFFIXES = ("noacc", "fp32")


def parse_variant(text: str) -> Variant:
    base, *suffixes = text.split("-")
    head = base.rstrip("0123456789")
    if head not in HEADS:
        raise ValueError(f"{text!r} does not start with one of {sorted(HEADS)}")
    unknown = sorted(set(suffixes) - set(SUFFIXES))
    if unknown:
        raise ValueError(f"{text!r} carries unknown suffixes {unknown}; valid: {list(SUFFIXES)}")
    return Variant(text, head, int(base[len(head):] or "4"), accuracy="noacc" not in suffixes,
                   states_dtype=jnp.dtype(jnp.float32 if "fp32" in suffixes else jnp.bfloat16))


def timed(callable_, *args) -> float:
    """Milliseconds per call over REPEATS calls, dispatched asynchronously."""
    for _ in range(3):
        callable_(*args)
    jax.block_until_ready(callable_(*args))
    start = time.perf_counter()
    out = None
    for _ in range(REPEATS):
        out = callable_(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / REPEATS * 1e3


def measure(variant: Variant) -> dict[str, object]:
    hidden = jax.random.normal(jax.random.PRNGKey(0), (B, S, D), variant.states_dtype)
    # [vocab, width]: the embedding table's own layout.
    embedding = jax.random.normal(jax.random.PRNGKey(1), (V, D), jnp.float32) * 0.02
    targets = jax.random.randint(jax.random.PRNGKey(2), (B, S), 0, V)
    function = HEADS[variant.head]

    device = jax.local_devices()[0]
    stats = device.memory_stats()
    before = stats.get("peak_bytes_in_use", 0) if stats else 0

    forward = jax.jit(lambda states, table: function(states, table, targets, variant))
    both = jax.jit(jax.value_and_grad(
        lambda states, table: function(states, table, targets, variant),
        argnums=(0, 1), has_aux=True))
    jax.block_until_ready(forward(hidden, embedding))
    jax.block_until_ready(both(hidden, embedding))

    forward_ms = timed(forward, hidden, embedding)
    both_ms = timed(both, hidden, embedding)
    stats = device.memory_stats()
    peak = stats.get("peak_bytes_in_use", 0) if stats else 0
    (loss, accuracy), (d_states, d_table) = both(hidden, embedding)

    return {
        "variant": variant.text,
        "head": variant.head,
        "chunks": variant.chunks,
        "accuracy": variant.accuracy,
        "states_dtype": variant.states_dtype.name,
        "forward_ms": round(forward_ms, 2),
        "forward_backward_ms": round(both_ms, 2),
        "peak_gib": round(peak / 2 ** 30, 3),
        "peak_delta_gib": round(max(0, peak - before) / 2 ** 30, 3),
        "loss": float(loss),
        "token_accuracy": float(accuracy),
        "d_states_sum": float(jnp.sum(d_states.astype(jnp.float32))),
        "d_table_sum": float(jnp.sum(d_table)),
        "d_states_absmax": float(jnp.abs(d_states.astype(jnp.float32)).max()),
        "d_table_absmax": float(jnp.abs(d_table).max()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", type=parse_variant,
                        help="baseline, stored or remat, a chunk count, -noacc, -fp32")
    variant = parser.parse_args(argv).variant
    print(json.dumps(measure(variant)))


if __name__ == "__main__":
    main()
