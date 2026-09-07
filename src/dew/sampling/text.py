"""Cached text generation with explicit lengths and sampling likelihoods."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from jax import lax
from jax.experimental import multihost_utils
from jax.typing import ArrayLike

from dew.nn.inputs import ModelInputs
from dew.objectives.base import Variables


@dataclass(frozen=True)
class Sampling:
    """Token selection and termination. Zero temperature is deterministic argmax.

    ``top_k=None`` keeps the vocabulary. EOS counts as a sampled action;
    subsequent output slots contain ``pad_id`` and have no likelihood.
    """

    temperature: float = 1.0
    top_k: int | None = None
    eos_id: int | tuple[int, ...] | None = None
    pad_id: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        if type(self.pad_id) is not int or self.pad_id < 0:
            raise ValueError("pad_id must be a non-negative token id")
        if self.eos_id is not None:
            stops = (self.eos_id,) if isinstance(self.eos_id, int) else tuple(self.eos_id)
            if not stops or any(type(token) is not int or token < 0 for token in stops):
                raise ValueError("eos_id must contain non-negative token ids")
            object.__setattr__(self, "eos_id", stops)


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


@struct.dataclass
class DecoderState:
    """Functional autoregressive state shared by batch generation and scheduling."""

    cache: Variables
    logits: jax.Array
    lengths: jax.Array
    finished: jax.Array
    positions: jax.Array | None


@struct.dataclass
class TokenSample:
    """One response action per row; validity determines which rows emitted."""

    tokens: jax.Array
    valid: jax.Array
    behavior_log_probs: jax.Array
    raw_log_probs: jax.Array


def _prefill(model: nn.Module, params: Variables, inputs: ModelInputs) -> DecoderState:
    batch, width = inputs.tokens.shape
    cache = model.apply(params, batch, method="init_cache", mutable=["cache"])[1]["cache"]
    logits, updated = model.apply(
        {**params, "cache": cache}, inputs.tokens, decode=True, mutable=["cache"],
        rngs=None, method=None, capture_intermediates=False, **inputs.kwargs())
    valid = inputs.token_fields["attention_mask"]
    last = jnp.max(jnp.where(valid, jnp.arange(width)[None, :], -1), axis=1)
    positions = inputs.token_fields.get("positions")
    if positions is not None:
        positions = positions[jnp.arange(batch), jnp.maximum(last, 0)] + 1
    return DecoderState(
        updated["cache"], logits[jnp.arange(batch), jnp.maximum(last, 0)],
        jnp.zeros(batch, jnp.int32), last < 0, positions)


def _decode(model: nn.Module, params: Variables, state: DecoderState,
            keys: jax.Array, sampling: Sampling) -> tuple[DecoderState, TokenSample]:
    step_keys = jax.vmap(jax.random.fold_in)(keys, state.lengths)
    chosen, behavior, raw = _sample_token(state.logits, step_keys, sampling)
    active = ~state.finished
    token = jnp.where(active, chosen, sampling.pad_id)
    stopped = (jnp.zeros_like(active) if sampling.eos_id is None else
               active & jnp.isin(chosen, jnp.asarray(sampling.eos_id)))
    positions = {} if state.positions is None else {"positions": state.positions[:, None]}
    logits, updated = model.apply(
        {**params, "cache": state.cache}, token[:, None], decode=True,
        attention_mask=active[:, None], mutable=["cache"], rngs=None,
        method=None, capture_intermediates=False, **positions)
    following = DecoderState(updated["cache"], logits[:, -1],
                             state.lengths + active.astype(jnp.int32),
                             state.finished | stopped,
                             None if state.positions is None else state.positions + active)
    return following, TokenSample(token, active, jnp.where(active, behavior, 0.0),
                                   jnp.where(active, raw, 0.0))


def _generate(model: nn.Module, params: Variables, inputs: ModelInputs,
              keys: jax.Array, max_new_tokens: int, sampling: Sampling) -> Generation:
    """One fixed compiled scan; finished rows do not mutate their cache state."""
    batch = inputs.tokens.shape[0]
    if max_new_tokens == 0:
        empty = jnp.zeros((batch, 0), jnp.float32)
        return Generation(inputs.tokens, jnp.zeros(batch, jnp.int32),
                          jnp.zeros(batch, bool), empty, empty)
    initial = _prefill(model, params, inputs)

    def step(state, _):
        return _decode(model, params, state, keys, sampling)

    state, samples = lax.scan(step, initial, None, length=max_new_tokens)
    samples = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), samples)
    return Generation(jnp.concatenate([inputs.tokens, samples.tokens], axis=1),
                      state.lengths, state.finished & jnp.any(inputs.token_fields["attention_mask"], axis=1),
                      samples.behavior_log_probs, samples.raw_log_probs)


def _mesh(params: Variables) -> jax.sharding.Mesh | None:
    for leaf in jax.tree.leaves(params):
        mesh = getattr(getattr(leaf, "sharding", None), "mesh", None)
        if mesh is not None and not mesh.empty:
            return mesh
    return None


@functools.lru_cache(maxsize=None)
def _compiled(rows: jax.sharding.NamedSharding | None):
    return jax.jit(_generate, static_argnames=("model", "max_new_tokens", "sampling"),
                   in_shardings=(None, rows, rows), out_shardings=rows)


def generate(model: nn.Module, params: Variables,
             inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
             *, key: jax.Array, sampling: Sampling = Sampling()) -> Generation:
    """Generate from numeric model inputs, with an array shorthand for text.

    ModelInputs.token_fields["attention_mask"] identifies real tokens. Missing masks mean all
    tokens are real. Every row must contain a real token. Prefill evaluates
    conditioning once; decode reuses the model-owned cache and logical-position
    state. Each cache compacts real input tokens and leaves paused rows intact.

    Parameters keep their placement. On a mesh, rows split over its batch
    axes. All cooperating processes use the same input shapes and sampling
    value, and execute a fixed decode trip count. Each process receives only
    its own rows, with keys folded by global row index and response position.
    """
    from dew.training.distributed import BATCH_SPEC, local_rows

    mesh = _mesh(params)
    processes = jax.process_count() if mesh is not None else 1
    process = jax.process_index() if mesh is not None else 0
    error = None
    prepared = None
    try:
        if isinstance(inputs, ModelInputs):
            inputs.validate()
            ids = local_rows(inputs.tokens)
            fields = {name: local_rows(value) for name, value in inputs.token_fields.items()}
            conditioning = {name: local_rows(value) for name, value in inputs.conditioning.items()}
        else:
            ids, fields, conditioning = np.asarray(inputs), {}, {}
        if ids.ndim != 2 or min(ids.shape) < 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("inputs must contain non-empty [B, P] integer token ids")
        if "params" not in params:
            raise ValueError("generate takes the full variables dict ({'params': ...})")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        valid = np.asarray(fields.get("attention_mask", np.ones(ids.shape, bool)))
        if valid.shape != ids.shape or not np.all((valid == 0) | (valid == 1)):
            raise ValueError("attention_mask must be binary [B, P] aligned with token ids")
        if not np.all(valid.any(axis=1)):
            raise ValueError("each prompt must contain at least one valid token")
        cache_len = getattr(model, "max_seq_len", None)
        if cache_len is not None and int(valid.sum(axis=1).max()) + max_new_tokens > cache_len:
            raise ValueError("prompt plus max_new_tokens exceeds max_seq_len; raise the cache capacity")
        vocab = getattr(model, "vocab_size", None)
        if np.any(ids < 0):
            raise ValueError("token ids must be non-negative")
        media = np.asarray(fields.get("image_indices", np.full(ids.shape, -1))) >= 0
        if vocab is not None and np.any((ids >= vocab) & ~media):
            raise ValueError("text token ids must be inside the vocabulary")
        if vocab is not None and (sampling.pad_id >= vocab or
                                 (sampling.eos_id is not None and np.any(np.asarray(sampling.eos_id) >= vocab))):
            raise ValueError("sampling token ids must be inside the vocabulary")
        fields["attention_mask"] = valid.astype(bool)
        prepared = ModelInputs(jnp.asarray(ids, jnp.int32),
                               {name: jnp.asarray(value) for name, value in fields.items()},
                               {name: jnp.asarray(value) for name, value in conditioning.items()})
        prepared.validate()
    except BaseException as failure:
        error = failure
    if processes > 1:
        from dew.artifacts import agree_process_phase
        agree_process_phase(error, phase="generation input validation")
    elif error is not None:
        raise error
    assert prepared is not None
    batch = prepared.tokens.shape[0]
    if processes > 1:
        # Compare fixed-size hashes before creating distributed input arrays.
        # The schema covers all conditioning and token fields, not token length
        # alone; different traced shapes would issue mismatched collectives.
        import hashlib
        schema = (str(jax.tree.structure(prepared)),
                  [(leaf.shape, str(leaf.dtype)) for leaf in jax.tree.leaves(prepared)],
                  max_new_tokens, sampling)
        digest = np.frombuffer(hashlib.sha256(repr(schema).encode()).digest(), np.uint8)
        multihost_utils.assert_equal(digest, "generation input shapes and sampling must agree across processes")
    rows_sharding = None
    if mesh is not None:
        rows_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(BATCH_SPEC[0]))
        batch_devices = math.prod(mesh.shape[axis] for axis in BATCH_SPEC[0])
        if batch_devices % processes:
            raise ValueError("the batch axes of the mesh do not split evenly over the processes")
        local_devices = batch_devices // processes
        count = -(-batch // local_devices) * local_devices
        if count != batch:
            indices = jnp.arange(count) % batch
            prepared = prepared.take_rows(indices)
            valid = prepared.token_fields["attention_mask"] & (jnp.arange(count)[:, None] < batch)
            prepared = replace(prepared, token_fields={**prepared.token_fields, "attention_mask": valid})
        prepared = jax.tree.map(
            lambda leaf: jax.make_array_from_process_local_data(rows_sharding, np.asarray(leaf)), prepared)
    else:
        count = batch
    row_keys = jax.vmap(lambda row: jax.random.fold_in(key, row))(
        jnp.arange(count) + process * batch)
    if rows_sharding is not None:
        key_data = jax.make_array_from_process_local_data(rows_sharding, np.asarray(jax.random.key_data(row_keys)))
        row_keys = jax.random.wrap_key_data(key_data, impl=jax.random.key_impl(key))
    output = _compiled(rows_sharding)(model, params, prepared, row_keys, max_new_tokens, sampling)
    if mesh is None:
        return output
    if processes == 1:
        output = jax.tree.map(lambda leaf: leaf[:batch], output)
        return jax.device_put(output, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
    return jax.tree.map(lambda leaf: jnp.asarray(local_rows(leaf)[:batch]), output)
