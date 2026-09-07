"""Cached text generation with explicit lengths and sampling likelihoods."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from jax import lax
from jax.experimental import multihost_utils
from jax.typing import ArrayLike

from dew.objectives.base import Variables


@dataclass(frozen=True)
class Sampling:
    """Token selection and termination. Zero temperature is deterministic argmax.

    ``top_k=None`` keeps the vocabulary. EOS counts as a sampled action;
    subsequent output slots contain ``pad_id`` and have no likelihood.
    """

    temperature: float = 1.0
    top_k: int | None = None
    eos_id: int | None = None
    pad_id: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        for name, value in (("pad_id", self.pad_id), ("eos_id", self.eos_id)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative token id")


@struct.dataclass
class Generation:
    """Prompt plus padded continuation, and response-aligned likelihoods.

    ``lengths`` counts response actions, including EOS. ``terminated`` marks
    EOS termination; false marks a length limit. Both log-probability arrays
    have shape [B, max_new_tokens]. Only positions below ``lengths`` are valid.
    ``behavior_log_probs`` describes the temperature/top-k distribution that
    drew each action. ``raw_log_probs`` describes the unmodified model policy.
    """

    tokens: jax.Array
    lengths: jax.Array
    terminated: jax.Array
    behavior_log_probs: jax.Array
    raw_log_probs: jax.Array


def _sample_token(logits: jax.Array, keys: jax.Array, sampling: Sampling
                  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    logits = logits.astype(jnp.float32)
    raw = jax.nn.log_softmax(logits)
    if sampling.temperature == 0:
        token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        behavior = jnp.zeros(token.shape, jnp.float32)
    else:
        scores = logits / sampling.temperature
        if sampling.top_k is not None:
            keep = min(sampling.top_k, scores.shape[-1])
            cutoff = lax.top_k(scores, keep)[0][..., -1:]
            scores = jnp.where(scores < cutoff, -jnp.inf, scores)
        token = jax.vmap(jax.random.categorical)(keys, scores).astype(jnp.int32)
        behavior = jnp.take_along_axis(jax.nn.log_softmax(scores), token[:, None], -1)[:, 0]
    selected_raw = jnp.take_along_axis(raw, token[:, None], -1)[:, 0]
    return token, behavior, selected_raw


def _generate(model: nn.Module, params: Variables, prompt: jax.Array, key_data: jax.Array,
              live: jax.Array, max_new_tokens: int, sampling: Sampling) -> Generation:
    """Decode a fixed trip count so every device runs identical collectives.

    ``live`` marks real rows; filler rows exist only to keep global shapes
    equal across processes. Finished rows keep decoding with pad input, and
    their outputs are masked, so no collective depends on sampled content.
    """
    batch = prompt.shape[0]
    keys = jax.random.wrap_key_data(key_data)
    tokens = jnp.full((batch, max_new_tokens), sampling.pad_id, jnp.int32)
    lengths = jnp.zeros(batch, jnp.int32)
    done = jnp.zeros(batch, bool)
    behavior = jnp.zeros((batch, max_new_tokens), jnp.float32)
    raw = jnp.zeros_like(behavior)
    if max_new_tokens == 0:
        return Generation(prompt, lengths, done, behavior, raw)
    cache = model.apply(params, batch, method="init_cache", mutable=["cache"])[1]["cache"]
    logits, mutated = model.apply({**params, "cache": cache}, prompt,
                                  decode=True, mutable=["cache"])

    def step(carry, index):
        logits, cache, tokens, lengths, finished, behavior, raw = carry
        step_keys = jax.vmap(lambda key: jax.random.fold_in(key, index))(keys)
        chosen, chosen_behavior, chosen_raw = _sample_token(logits, step_keys, sampling)
        active = live & ~finished
        token = jnp.where(active, chosen, sampling.pad_id)
        tokens = tokens.at[:, index].set(token)
        behavior = behavior.at[:, index].set(jnp.where(active, chosen_behavior, 0.0))
        raw = raw.at[:, index].set(jnp.where(active, chosen_raw, 0.0))
        lengths = lengths + active.astype(jnp.int32)
        if sampling.eos_id is not None:
            finished = finished | (active & (chosen == sampling.eos_id))
        next_logits, updated = model.apply(
            {**params, "cache": cache}, token[:, None], decode=True, mutable=["cache"])
        return (next_logits[:, -1], updated["cache"], tokens, lengths, finished,
                behavior, raw), None

    (_, _, tokens, lengths, done, behavior, raw), _ = lax.scan(
        step, (logits[:, -1], mutated["cache"], tokens, lengths, done, behavior, raw),
        jnp.arange(max_new_tokens))
    return Generation(jnp.concatenate([prompt, tokens], axis=1), lengths, done, behavior, raw)


def _mesh(params: Variables) -> jax.sharding.Mesh | None:
    for leaf in jax.tree.leaves(params):
        mesh = getattr(getattr(leaf, "sharding", None), "mesh", None)
        if mesh is not None and not mesh.empty:
            return mesh
    return None


@functools.lru_cache(maxsize=None)
def _compiled(rows: jax.sharding.NamedSharding | None):
    """The decode loop with its batch rows split over the mesh, parameters as placed."""
    return jax.jit(_generate, static_argnames=("model", "max_new_tokens", "sampling"),
                   in_shardings=(None, rows, rows, rows), out_shardings=rows)


def _agreed_lengths(lengths: np.ndarray, processes: int) -> np.ndarray:
    """Every process's prompt lengths, as one [processes, rows] table.

    Buckets are chosen from this shared table, so all ranks execute the same
    compiled shapes in the same order even when their own rows differ.
    """
    if processes == 1:
        return lengths[None, :]
    counts = multihost_utils.process_allgather(np.asarray(len(lengths), np.int64))
    if np.any(counts != len(lengths)):
        raise ValueError("every process must generate the same number of rows")
    return np.asarray(multihost_utils.process_allgather(lengths.astype(np.int64)))


def generate(model: nn.Module, params: Variables, prompt: ArrayLike | Sequence[Sequence[int]],
             max_new_tokens: int,
             *, key: jax.Array, sampling: Sampling = Sampling(),
             prompt_lengths: ArrayLike | Sequence[int] | None = None) -> Generation:
    """Generate from non-empty left-padded [B, P] token rows on the host.

    ``params`` is the full Flax variables mapping. Missing ``prompt_lengths``
    means all P tokens are real. Length buckets remove padding before prefill,
    so attention, latent and recurrent caches see the same tokens as an
    unpadded call. Each bucket compiles the same cached decoder at its shape.
    The original padded prompt stays in the returned ``tokens``. Random keys
    fold in the global row index and the response position, so a pool draws
    what one process draws over the same rows.

    On a mesh the bucket's rows split over the batch axes the trainer uses,
    parameters stay where they were placed, and the decode runs a fixed trip
    count. In a multi-process run each process passes its own rows and gets
    its own back. Bucket shapes come from every process's prompt lengths, so
    cooperating ranks issue identical collectives; a process short of rows in
    a bucket fills the shape with rows whose outputs are discarded.
    """
    ids = np.asarray(prompt)
    if ids.ndim != 2 or min(ids.shape) < 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("prompt must be non-empty [B, P] integer token ids")
    if "params" not in params:
        raise ValueError("generate takes the full variables dict ({'params': ...})")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    batch, width = ids.shape
    lengths = np.full(batch, width, np.int32) if prompt_lengths is None else np.asarray(prompt_lengths)
    if (lengths.shape != (batch,) or not np.issubdtype(lengths.dtype, np.integer)
            or np.any(lengths < 1) or np.any(lengths > width)):
        raise ValueError("prompt_lengths must be [B] integers between 1 and the prompt width")
    cache_len = getattr(model, "max_seq_len", None)
    if cache_len is not None and int(lengths.max()) + max_new_tokens > cache_len:
        raise ValueError("prompt plus max_new_tokens exceeds max_seq_len; raise the cache capacity")
    vocab = getattr(model, "vocab_size", None)
    if np.any(ids < 0) or (vocab is not None and np.any(ids >= vocab)):
        raise ValueError("prompt token ids must be inside the vocabulary")
    if vocab is not None and (sampling.pad_id >= vocab or
                             (sampling.eos_id is not None and sampling.eos_id >= vocab)):
        raise ValueError("sampling token ids must be inside the vocabulary")
    mesh = _mesh(params)
    processes = jax.process_count() if mesh is not None else 1
    process = jax.process_index() if mesh is not None else 0
    rows_sharding = None
    per_process = 1
    if mesh is not None:
        from dew.training.distributed import BATCH_SPEC

        rows_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(BATCH_SPEC[0]))
        batch_devices = math.prod(mesh.shape[axis] for axis in BATCH_SPEC[0])
        if batch_devices % processes:
            raise ValueError("the batch axes of the mesh do not split evenly over the processes")
        per_process = batch_devices // processes
    table = _agreed_lengths(lengths.astype(np.int64), processes)
    compiled = _compiled(rows_sharding)
    typed = key if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else jax.random.wrap_key_data(key)
    row_keys = np.asarray(jax.random.key_data(jax.vmap(
        lambda row: jax.random.fold_in(typed, row))(jnp.arange(batch) + process * batch)))
    tokens = np.concatenate([ids.astype(np.int32),
                             np.full((batch, max_new_tokens), sampling.pad_id, np.int32)], axis=1)
    response_lengths = np.zeros(batch, np.int32)
    terminated = np.zeros(batch, bool)
    behavior = np.zeros((batch, max_new_tokens), np.float32)
    raw = np.zeros_like(behavior)
    for length in np.unique(table):
        length = int(length)
        rows = np.flatnonzero(lengths == length)
        # Every process contributes the same row count, rounded up to whole
        # device shards, so the global shape agrees; the filler rows repeat
        # a real row's prompt where this process has one.
        widest = int(np.max(np.sum(table == length, axis=1)))
        count = -(-widest // per_process) * per_process
        picks = (rows if rows.size else np.zeros(1, np.int64))[np.arange(count) % max(rows.size, 1)]
        bucket_prompt = (ids[picks, width - length:].astype(np.int32) if rows.size
                         else np.zeros((count, length), np.int32))
        inputs = (bucket_prompt, row_keys[picks], np.arange(count) < rows.size)
        if rows_sharding is not None:
            inputs = tuple(jax.make_array_from_process_local_data(rows_sharding, value)
                           for value in inputs)
        output = compiled(model, params, *inputs, max_new_tokens, sampling)
        if not rows.size:
            continue
        from dew.training.distributed import local_rows

        keep = slice(0, rows.size)
        tokens[rows, width:] = local_rows(output.tokens)[keep, length:]
        response_lengths[rows] = local_rows(output.lengths)[keep]
        terminated[rows] = local_rows(output.terminated)[keep]
        behavior[rows] = local_rows(output.behavior_log_probs)[keep]
        raw[rows] = local_rows(output.raw_log_probs)[keep]
    result = Generation(jnp.asarray(tokens), jnp.asarray(response_lengths), jnp.asarray(terminated),
                        jnp.asarray(behavior), jnp.asarray(raw))
    if mesh is None or processes > 1:
        # A pool hands each process its own rows as host data; the trainer
        # reassembles them. One process on a mesh gets rows every device holds.
        return result
    return jax.device_put(result, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
