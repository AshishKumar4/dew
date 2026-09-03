"""Isolated head+loss timing for the LM vocabulary head, small-LM shape.

One variant per process, because `peak_bytes_in_use` is a process high-water
mark: run this once per row and read the delta it prints.

The shape is the head work a training step does at 8,192 tokens (batch 16 x
sequence 512), width 768, vocabulary 50,304, bf16 states with an fp32 head and
an fp32 loss, forward and backward. The head is held as `[vocab, width]`, the
layout `embed_tokens.embedding` really has, so the transpose the chunked path
needs is inside the measurement and not hidden by the setup.

  baseline    states -> full fp32 logits -> optax cross entropy plus the
              argmax accuracy: the code this branch replaced, verbatim
  stored      dew.objectives.lm.chunked.chunked_cross_entropy, which keeps
              the tiles for the backward pass
  remat       the same loop with jax.checkpoint around the tile, which
              recomputes it in the backward pass instead
  -noacc      drops the top-1 prediction, to price the metric
  -fp32       fp32 states, so the state gradient can be compared without
              bf16 rounding in the way

Usage: PYTHONPATH=<checkout>/src python tools/benchmark_lm_head.py <variant>
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import optax

from dew.objectives.lm.chunked import chunked_cross_entropy, vocabulary_chunks

B, S, D, V = 16, 512, 768, 50304
REPEATS = 50

variant = sys.argv[1]
chunks = next((int(c) for c in ("4", "8") if c in variant), 4)
accuracy_wanted = "noacc" not in variant
states_dtype = jnp.float32 if "fp32" in variant else jnp.bfloat16

hidden = jax.random.normal(jax.random.PRNGKey(0), (B, S, D), states_dtype)
# [vocab, width]: the embedding table's own layout.
embedding = jax.random.normal(jax.random.PRNGKey(1), (V, D), jnp.float32) * 0.02
targets = jax.random.randint(jax.random.PRNGKey(2), (B, S), 0, V)


def baseline(states, table):
    """The unchunked head: one full-vocabulary logits tensor, then the loss."""
    logits = jnp.einsum('...d,vd->...v', states.astype(jnp.float32),
                        table.astype(jnp.float32))
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    correct = (jnp.argmax(logits, axis=-1) == targets).astype(losses.dtype)
    return jnp.mean(losses), jnp.mean(correct)


def stored(states, table):
    """The shipped kernel: the tiles stay live for the backward pass."""
    losses, predicted = chunked_cross_entropy(states, table.T, targets, chunks)
    if not accuracy_wanted:
        return jnp.mean(losses), jnp.zeros(())
    return jnp.mean(losses), jnp.mean((predicted == targets).astype(losses.dtype))


def remat(states, table):
    """The same arithmetic with each tile recomputed in the backward pass."""
    head = table.T
    flat = states.astype(jnp.float32).reshape(-1, D)
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

    for start, stop in vocabulary_chunks(V, chunks):
        inside = (flat_targets >= start) & (flat_targets < stop)
        column = jnp.clip(flat_targets - start, 0, stop - start - 1)
        chunk_lse, picked, chunk_best, chunk_column = tile(
            flat, head[:, start:stop], column)
        total = jnp.logaddexp(total, chunk_lse)
        target_logit = target_logit + jnp.where(inside, picked, 0.0)
        if accuracy_wanted:
            better = chunk_best > best
            best = jnp.where(better, chunk_best, best)
            predicted = jnp.where(better, chunk_column + start, predicted)
    losses = (total - target_logit).reshape(targets.shape)
    if not accuracy_wanted:
        return jnp.mean(losses), jnp.zeros(())
    correct = (predicted.reshape(targets.shape) == targets).astype(losses.dtype)
    return jnp.mean(losses), jnp.mean(correct)


FUNCTIONS = {"baseline": baseline, "stored": stored, "remat": remat}
function = FUNCTIONS[variant.split("-")[0].rstrip("48")]

device = jax.local_devices()[0]
before = device.memory_stats().get("peak_bytes_in_use", 0)

forward = jax.jit(function)
both = jax.jit(jax.value_and_grad(function, argnums=(0, 1), has_aux=True))
jax.block_until_ready(forward(hidden, embedding))
jax.block_until_ready(both(hidden, embedding))


def timed(callable_):
    for _ in range(3):
        callable_(hidden, embedding)
    jax.block_until_ready(callable_(hidden, embedding))
    start = time.perf_counter()
    for _ in range(REPEATS):
        out = callable_(hidden, embedding)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / REPEATS * 1e3


forward_ms = timed(forward)
both_ms = timed(both)
peak = device.memory_stats().get("peak_bytes_in_use", 0)
(loss, accuracy), (d_states, d_table) = both(hidden, embedding)

print(json.dumps({
    "variant": variant,
    "chunks": chunks,
    "accuracy": accuracy_wanted,
    "states_dtype": str(states_dtype.__name__),
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
}))
