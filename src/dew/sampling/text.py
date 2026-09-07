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


def _generate(model: nn.Module, params: Variables, prompt: jax.Array,
              keys: jax.Array, max_new_tokens: int, sampling: Sampling) -> Generation:
    batch = prompt.shape[0]
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

    def condition(carry):
        index, _, _, _, _, finished, _, _ = carry
        return (index < max_new_tokens) & ~jnp.all(finished)

    def step(carry):
        index, logits, cache, tokens, lengths, finished, behavior, raw = carry
        step_keys = jax.vmap(lambda key: jax.random.fold_in(key, index))(keys)
        chosen, chosen_behavior, chosen_raw = _sample_token(logits, step_keys, sampling)
        active = ~finished
        token = jnp.where(active, chosen, sampling.pad_id)
        tokens = tokens.at[:, index].set(token)
        behavior = behavior.at[:, index].set(jnp.where(active, chosen_behavior, 0.0))
        raw = raw.at[:, index].set(jnp.where(active, chosen_raw, 0.0))
        lengths = lengths + active.astype(jnp.int32)
        if sampling.eos_id is not None:
            finished = finished | (chosen == sampling.eos_id)

        def advance(_):
            next_logits, updated = model.apply(
                {**params, "cache": cache}, token[:, None], decode=True, mutable=["cache"])
            return next_logits[:, -1], updated["cache"]

        logits, cache = lax.cond(
            (index + 1 < max_new_tokens) & ~jnp.all(finished),
            advance, lambda _: (logits, cache), operand=None)
        return index + 1, logits, cache, tokens, lengths, finished, behavior, raw

    _, _, _, tokens, lengths, done, behavior, raw = lax.while_loop(
        condition, step, (jnp.asarray(0), logits[:, -1], mutated["cache"],
                          tokens, lengths, done, behavior, raw))
    return Generation(jnp.concatenate([prompt, tokens], axis=1), lengths, done, behavior, raw)


def _replication(params: Variables) -> jax.sharding.NamedSharding | None:
    for leaf in jax.tree.leaves(params):
        mesh = getattr(getattr(leaf, "sharding", None), "mesh", None)
        if mesh is not None and not mesh.empty:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return None


@functools.lru_cache(maxsize=None)
def _compiled(out_sharding):
    return jax.jit(_generate, static_argnames=("model", "max_new_tokens", "sampling"),
                   out_shardings=out_sharding)


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
    fold in the original row index and response position, independent of buckets.
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
    sharding = _replication(params)
    compiled = _compiled(sharding)
    row_keys = jax.vmap(lambda row: jax.random.fold_in(key, row))(jnp.arange(batch))
    if np.all(lengths == width):
        return compiled(model, params, jnp.asarray(ids, jnp.int32), row_keys,
                        max_new_tokens, sampling)
    result = Generation(
        jnp.concatenate([jnp.asarray(ids, jnp.int32),
                         jnp.full((batch, max_new_tokens), sampling.pad_id, jnp.int32)], axis=1),
        jnp.zeros(batch, jnp.int32), jnp.zeros(batch, bool),
        jnp.zeros((batch, max_new_tokens), jnp.float32),
        jnp.zeros((batch, max_new_tokens), jnp.float32))
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        output = compiled(model, params, jnp.asarray(ids[rows, width - int(length):], jnp.int32),
                          row_keys[rows], max_new_tokens, sampling)
        result = Generation(
            result.tokens.at[rows, width:].set(output.tokens[:, int(length):]),
            result.lengths.at[rows].set(output.lengths),
            result.terminated.at[rows].set(output.terminated),
            result.behavior_log_probs.at[rows].set(output.behavior_log_probs),
            result.raw_log_probs.at[rows].set(output.raw_log_probs))
    return result if sharding is None else jax.device_put(result, sharding)
