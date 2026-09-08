"""Cached text generation with explicit lengths and sampling likelihoods."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Generic, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from jax.experimental import checkify, multihost_utils
from jax.typing import ArrayLike

from dew.nn.inputs import (
    ArrayT, ModelInputs, RowPlan, agreed_validity, generation_signature,
    local_rows, mesh_of, request_key,
)
from dew.objectives.base import Variables
from dew.sampling import decoding
from dew.sampling.decoding import (
    EndOfSequence, Greedy, LogitsTransform, MinP, StepState, Stopping, Temperature, TopK, TopP,
)
from dew.sampling import strategies
from dew.sampling.strategies import DecodeOps, DecoderState, Draws, Sample, Strategy

Transforms = LogitsTransform | Sequence[LogitsTransform]
Criteria = Stopping | Sequence[Stopping]


@dataclass(frozen=True)
class Sampling:
    """Token selection and termination. Zero temperature is deterministic argmax.

    ``top_k=None`` keeps the vocabulary. EOS counts as a sampled action;
    subsequent output slots contain ``pad_id`` and have no likelihood.

    A ``Sampling`` value is a convenience over the general decoding
    components: it compiles to the temperature, top-k, top-p and min-p
    transforms in that order plus an EOS criterion, which `generate` appends
    after any transforms and criteria a caller passes.
    """

    temperature: float = 1.0
    top_k: int | None = None
    eos_id: int | tuple[int, ...] | None = None
    pad_id: int = 0
    top_p: float = 1.0
    min_p: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        for name, value in (("top_p", self.top_p), ("min_p", self.min_p)):
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between zero and one")
        if type(self.pad_id) is not int or self.pad_id < 0:
            raise ValueError("pad_id must be a non-negative token id")
        if self.eos_id is not None:
            stops = (self.eos_id,) if isinstance(self.eos_id, int) else tuple(self.eos_id)
            if not stops or any(type(token) is not int or token < 0 for token in stops):
                raise ValueError("eos_id must contain non-negative token ids")
            object.__setattr__(self, "eos_id", stops)

    def transforms(self) -> tuple[LogitsTransform, ...]:
        """The filtering tail this policy adds after a caller's transforms.

        Zero temperature is the argmax, and the sample-only filters are
        inactive there, which is what `generate()` does with `do_sample=False`.
        """
        if self.temperature == 0:
            return (Greedy(),)
        tail: list[LogitsTransform] = []
        if self.temperature != 1.0:
            tail.append(Temperature(self.temperature))
        if self.top_k is not None:
            tail.append(TopK(self.top_k))
        if self.top_p < 1.0:
            tail.append(TopP(self.top_p))
        if self.min_p > 0.0:
            tail.append(MinP(self.min_p))
        return tuple(tail)

    def criteria(self) -> tuple[Stopping, ...]:
        """The EOS criterion this policy adds after a caller's criteria."""
        if self.eos_id is None:
            return ()
        return (EndOfSequence(jnp.asarray(self.eos_id, jnp.int32)),)


@struct.dataclass
class Generation(Generic[ArrayT]):
    """Prompt plus padded continuation, and response-aligned likelihoods.

    ``lengths`` counts response actions, including EOS. ``terminated`` marks a
    stopping criterion, EOS by default; false marks a length limit. Both
    log-probability arrays have shape [B, max_new_tokens]. Only positions
    below ``lengths`` are valid. ``behavior_log_probs`` describes the
    distribution that actually drew each action, after the whole transform
    chain. ``raw_log_probs`` describes the unmodified model policy.

    A request for ``n`` continuations per prompt gives every array
    ``[B * n, ...]`` rows: prompt zero's ``n`` continuations, then prompt
    one's. Each row carries its own length, termination and likelihoods.

    Arrays keep the placement the task ran with: on a mesh they are global
    arrays whose rows split over the batch axes, padded to the device count.
    ``host()`` reads this process's ``rows`` real rows back as host arrays.
    ``text`` decodes them through the processor the task was bound to.
    """

    tokens: ArrayT
    lengths: ArrayT
    terminated: ArrayT
    behavior_log_probs: ArrayT
    raw_log_probs: ArrayT
    rows: int | None = struct.field(pytree_node=False, default=None)
    decoder: Callable[[ArrayLike, ArrayLike, int], tuple[str, ...]] | None = struct.field(
        pytree_node=False, default=None)

    @property
    def prompt_width(self) -> int:
        return self.tokens.shape[1] - self.behavior_log_probs.shape[1]

    def host(self) -> Generation[np.ndarray]:
        """This process's real rows as host arrays."""
        return jax.tree.map(lambda leaf: local_rows(leaf)[:self.rows], self)

    @functools.cached_property
    def text(self) -> tuple[str, ...]:
        """Each real row's valid continuation, decoded on first access."""
        if self.decoder is None:
            raise ValueError("this generation carries no processor to decode with")
        rows = self.host()
        return self.decoder(rows.tokens, rows.lengths, self.prompt_width)


def _prefill(model: nn.Module, params: Variables, inputs: ModelInputs
             ) -> tuple[DecoderState, jax.Array]:
    """The state after the prompt, and which rows hold a real token."""
    batch, width = inputs.tokens.shape
    cache = model.apply(params, batch, method="init_cache", mutable=["cache"])[1]["cache"]
    logits, updated = model.apply(
        {**params, "cache": cache}, inputs.tokens, decode=True, mutable=["cache"],
        rngs=None, method=None, capture_intermediates=False, **inputs.kwargs())
    # An unpadded prompt carries no validity field, and its last real token is
    # the last slot.
    valid = inputs.token_fields.get("attention_mask")
    last = (jnp.full((batch,), width - 1, jnp.int32) if valid is None else
            jnp.max(jnp.where(valid, jnp.arange(width)[None, :], -1), axis=1))
    positions = inputs.token_fields.get("positions")
    if positions is not None:
        positions = positions[jnp.arange(batch), jnp.maximum(last, 0)] + 1
    return DecoderState(updated["cache"], logits[jnp.arange(batch), jnp.maximum(last, 0)],
                        positions), last >= 0


def _operations(model: nn.Module, params: Variables, pad_id: int) -> DecodeOps:
    """The model operations a strategy may run, bound to these weights.

    Parameters stay unmapped: every operation reads the same tree, and only
    the cache moves with the rows.
    """

    def advance(state: DecoderState, token: jax.Array, active: jax.Array) -> DecoderState:
        positions = {} if state.positions is None else {"positions": state.positions[:, None]}
        logits, updated = model.apply(
            {**params, "cache": state.cache}, jnp.where(active, token, pad_id)[:, None],
            decode=True, attention_mask=active[:, None], mutable=["cache"], rngs=None,
            method=None, capture_intermediates=False, **positions)
        return DecoderState(updated["cache"], logits[:, -1],
                            None if state.positions is None else state.positions + active)

    def reindex(state: DecoderState, rows: jax.Array) -> DecoderState:
        return jax.tree.map(lambda leaf: jnp.take(leaf, rows, axis=0), state)

    return DecodeOps(advance, reindex)


def _generate(model: nn.Module, params: Variables, inputs: ModelInputs, keys: jax.Array,
              max_new_tokens: int, sampling: Sampling, n: int,
              transforms: tuple[LogitsTransform, ...], stopping: tuple[Stopping, ...],
              strategy: Strategy) -> Generation:
    """One fixed compiled loop; finished rows do not mutate their cache state.

    The continuations share one prefill. Parameters remain unmapped, and the
    strategy owns whatever loop the request asked for. Results leave in prompt
    order, each prompt's continuations together.
    """
    batch, width = inputs.tokens.shape
    prompt = inputs.tokens if n == 1 else jnp.repeat(inputs.tokens, n, axis=0)
    valid = inputs.token_fields.get("attention_mask")
    if max_new_tokens == 0:
        empty = jnp.zeros((batch * n, 0), jnp.float32)
        return Generation(prompt, jnp.zeros(batch * n, jnp.int32),
                          jnp.zeros(batch * n, bool), empty, empty)
    state, real = _prefill(model, params, inputs)
    start = StepState(
        tokens=jnp.concatenate([inputs.tokens, jnp.zeros((batch, max_new_tokens), jnp.int32)], axis=1),
        valid=jnp.concatenate([jnp.ones((batch, width), bool) if valid is None else valid.astype(bool),
                               jnp.zeros((batch, max_new_tokens), bool)], axis=1),
        step=jnp.zeros(batch, jnp.int32), active=real, keys=keys, prompt_width=width)
    drawn: Draws = strategy(state, start, _operations(model, params, sampling.pad_id),
                            decoding.chain(transforms), decoding.criterion(stopping),
                            max_new_tokens, n)
    return Generation(
        jnp.concatenate([prompt, jnp.where(drawn.valid, drawn.tokens, sampling.pad_id)], axis=1),
        jnp.sum(drawn.valid, axis=1, dtype=jnp.int32), drawn.terminated,
        drawn.behavior_log_probs, drawn.raw_log_probs)


def _validated(model: nn.Module, ids: np.ndarray, fields: dict[str, np.ndarray],
               conditioning: dict[str, np.ndarray], max_new_tokens: int, sampling: Sampling,
               n: int) -> ModelInputs:
    """Host checks shared by every caller; returns device inputs whose validity
    field is present only where a prompt is actually padded."""
    if ids.ndim != 2 or min(ids.shape) < 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("inputs must contain non-empty [B, P] integer token ids")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer number of continuations")
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
    token_fields = {name: jnp.asarray(value) for name, value in fields.items()
                    if name != "attention_mask"}
    if not valid.all():
        token_fields["attention_mask"] = jnp.asarray(valid.astype(bool))
    prepared = ModelInputs(jnp.asarray(ids, jnp.int32), token_fields,
                           {name: jnp.asarray(value) for name, value in conditioning.items()})
    prepared.validate()
    return prepared


def resolve(sampling: Sampling, logits: Transforms, stopping: Criteria,
            strategy: Strategy | None
            ) -> tuple[tuple[LogitsTransform, ...], tuple[Stopping, ...], Strategy]:
    """The one chain, criterion and strategy a request runs.

    A caller's transforms run first, in the order given, and the policy's
    filtering tail runs last, which is the order `generate()` builds its
    processor list in. Criteria combine with OR. No strategy means `Sample`.
    """
    return (decoding.components(logits, "logits") + sampling.transforms(),
            decoding.components(stopping, "stopping") + sampling.criteria(),
            Sample() if strategy is None else strategies.as_pytree(strategy))


def _digest(components: object) -> tuple[object, ...]:
    """A stable description of resolved components, without their payloads."""
    return (str(jax.tree.structure(components)),
            tuple((jnp.shape(leaf), str(jnp.result_type(leaf))) for leaf in jax.tree.leaves(components)))


def _checked(model: nn.Module, params: Variables, inputs: ModelInputs, keys: jax.Array,
             max_new_tokens: int, sampling: Sampling, n: int,
             transforms: tuple[LogitsTransform, ...], stopping: tuple[Stopping, ...],
             strategy: Strategy) -> tuple[checkify.Error, Generation]:
    """`_generate` with its device checks discharged into a returned error.

    An undefined draw has to reach the caller as an exception rather than a
    fabricated token, and a device loop cannot raise. `checkify` carries the
    failure out of the computation; with no check emitted the error is empty
    and throwing it costs nothing.
    """

    def run(params, inputs, keys, transforms, stopping, strategy):
        return _generate(model, params, inputs, keys, max_new_tokens, sampling, n,
                         transforms, stopping, strategy)

    return checkify.checkify(run, errors=checkify.user_checks)(
        params, inputs, keys, transforms, stopping, strategy)


@functools.lru_cache(maxsize=None)
def _compiled(rows: jax.sharding.NamedSharding | None):
    return jax.jit(_checked, static_argnames=("model", "max_new_tokens", "sampling", "n"),
                   in_shardings=(None, rows, rows, None, None, None),
                   out_shardings=(None, rows))


@overload
def generate(model: nn.Module, params: Variables,
             inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
             *, key: jax.Array, sampling: Sampling = Sampling(), n: int = 1,
             logits: Transforms = (), stopping: Criteria = (),
             strategy: Strategy | None = None) -> Generation: ...


@overload
def generate(model: nn.Module, params: Variables,
             inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
             *, seed: int, sampling: Sampling = Sampling(), n: int = 1,
             logits: Transforms = (), stopping: Criteria = (),
             strategy: Strategy | None = None) -> Generation: ...


def generate(model: nn.Module, params: Variables,
             inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
             *, key: jax.Array | None = None, seed: int | None = None,
             sampling: Sampling = Sampling(), n: int = 1, logits: Transforms = (),
             stopping: Criteria = (), strategy: Strategy | None = None) -> Generation:
    """Generate from numeric model inputs, with an array shorthand for text.

    ModelInputs.token_fields["attention_mask"] identifies real tokens. Missing masks mean all
    tokens are real. Every row must contain a real token. Prefill evaluates
    conditioning once; decode reuses the model-owned cache and logical-position
    state. Each cache compacts real input tokens and leaves paused rows intact.

    Parameters keep their placement. On a mesh, rows split over its batch
    axes and the result keeps that sharding; ``Generation.host()`` reads a
    process's own rows back. All cooperating processes use the same input
    shapes, sampling value, decoding components and continuation count, and
    execute a fixed decode trip count. Keys fold in the global row index and
    the response position, so a pool draws what one process draws for the
    same rows.

    ``n`` continuations of each prompt share its prefill and leave as ``n``
    consecutive rows of every array, in prompt order. Continuation zero of a
    prompt draws with that prompt's own key, so ``n=1`` and continuation zero
    of any larger request are the same draw.

    ``logits`` and ``stopping`` extend decoding with transforms from
    ``dew.sampling.decoding`` or plain callables of the same shape; the
    ``sampling`` tail and its EOS criterion follow them. ``strategy`` replaces
    the per-row draw loop; ``None`` uses ``Sample``.
    """
    mesh = mesh_of(params)
    processes = jax.process_count() if mesh is not None else 1
    error = None
    prepared = None
    random_key = None
    components = None
    try:
        random_key = request_key(key, seed)
        canonical = ModelInputs.from_value(inputs)
        ids = local_rows(canonical.tokens)
        fields = {name: local_rows(value) for name, value in canonical.token_fields.items()}
        conditioning = {name: local_rows(value) for name, value in canonical.conditioning.items()}
        if "params" not in params:
            raise ValueError("generate takes the full variables dict ({'params': ...})")
        prepared = _validated(model, ids, fields, conditioning, max_new_tokens, sampling, n)
        components = resolve(sampling, logits, stopping, strategy)
    except BaseException as failure:
        error = failure
    if processes > 1:
        from dew.artifacts import agree_process_phase
        agree_process_phase(error, phase="generation input validation")
    elif error is not None:
        raise error
    assert prepared is not None and random_key is not None and components is not None
    controls = (max_new_tokens, n, sampling, _digest(components))
    if processes > 1:
        # Whether this process's own prompts needed padding is rank-local, and
        # the digest below would refuse a pool that disagrees only about that,
        # so the pool agrees one validity schema first.
        prepared = agreed_validity(prepared, processes, controls=controls, phase="generation input")
        # Compare fixed-size hashes before creating distributed input arrays.
        # The schema covers all conditioning and token fields, not token length
        # alone; different traced shapes would issue mismatched collectives.
        digest = generation_signature(prepared, controls)
        multihost_utils.assert_equal(
            digest, "generation input shapes, continuations, decoding components and sampling "
                    "must agree across processes")
    plan = RowPlan.over(mesh, prepared.tokens.shape[0])
    padded = plan.pad(prepared)
    if plan.count != plan.rows:
        # Repeated rows carry no real token, so they finish at once and emit
        # nothing. Their validity is the field an unpadded request omitted.
        existing = padded.token_fields.get("attention_mask")
        valid = (np.ones(padded.tokens.shape, bool) if existing is None
                 else np.asarray(existing))
        padded = replace(padded, token_fields={**padded.token_fields,
                                               "attention_mask": valid & ~plan.padding[:, None]})
    failure, output = _compiled(plan.sharding)(model, params, plan.place(padded),
                                               plan.keys(random_key), max_new_tokens, sampling, n,
                                               *components)
    failure.throw()
    return replace(output, rows=plan.rows * n)
