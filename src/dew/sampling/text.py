"""Cached text generation with explicit lengths and sampling likelihoods.

`generate` validates a request on the host, prefills the prompt once, and
hands the decode loop to a `Strategy`. `Sampling` is the default policy:
temperature, top-k, top-p and min-p with an EOS criterion. `Generation`
carries the tokens with one length and two log probabilities per row, the
behaviour policy's and the model's own.

On a pool every process resolves the same request. `_request` raises what a
rank can get wrong on its own, `_digest` reduces the resolved components to a
value every rank can compare, and the ranks agree on both before any of them
enters a collective.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Generic, Protocol, overload, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import checkify, multihost_utils
from jax.typing import ArrayLike

from dew.artifacts import agreed
from dew.nn.backbones.causal_transformer import gather_cache_rows
from dew.nn.inputs import (
    ArrayT,
    ModelInputs,
    PredictionPhase,
    RowPlan,
    agreed_validity,
    generation_signature,
    local_rows,
    mesh_of,
    request_key,
)
from dew.objectives.base import Variables
from dew.sampling import decoding, strategies
from dew.sampling.decoding import (
    EndOfSequence,
    Greedy,
    LogitsTransform,
    MinP,
    StepState,
    Stopping,
    Temperature,
    TopK,
    TopP,
)
from dew.sampling.strategies import DecodeOps, DecoderState, Draws, Sample, Strategy

Transforms = LogitsTransform | Sequence[LogitsTransform]
Criteria = Stopping | Sequence[Stopping]
Components = tuple[tuple[LogitsTransform, ...], tuple[Stopping, ...], Strategy]

# A resolved component's configuration as every process can spell it: a repr,
# a number (an array's shape is a tuple of them), or a tuple of either nested
# as deep as the component is. Only `repr` and `==` are applied to one, by the
# pool agreement in `dew.nn.inputs` and by the digest that precedes it.
type Identity = str | int | tuple[Identity, ...]
# What carries its own qualified name, and so names a component: a class, a
# function, a bound method, a library builtin. Anything else is named by its
# class, which is one of these.
type SelfNaming = type | types.FunctionType | types.MethodType | types.BuiltinFunctionType
SELF_NAMING = (type, types.FunctionType, types.MethodType, types.BuiltinFunctionType)


@runtime_checkable
class Bounded(Protocol):
    """A model that declares its vocabulary and the capacity of its cache.

    `nn.Module` declares neither; a decoder (`CausalTransformer`, the
    multimodal wrapper, a test policy) declares both, and the host checks
    read them to refuse an id outside the vocabulary or a request that
    would overflow the cache. A model that declares neither is checked
    for neither.
    """

    @property
    def vocab_size(self) -> int: ...

    @property
    def max_seq_len(self) -> int | None: ...


@runtime_checkable
class Exposing(Protocol):
    """A decoder that hands back its hidden states beside its logits.

    A strategy that drafts seeds its draft from them. A model without the
    method still decodes; it only cannot draft.
    """

    def states_and_logits(self, tokens: jax.Array, **kwargs: jax.Array | bool | None
                          ) -> tuple[jax.Array, jax.Array]: ...


@runtime_checkable
class Selective(Protocol):
    """A decoder that scores one position per row instead of them all.

    Its prefill runs the head on the slot the first draw reads and nothing
    else; a model without the method scores every prompt position and pays
    for the ones it discards.
    """

    def states_and_logits_at(self, tokens: jax.Array, slots: jax.Array, **kwargs: jax.Array | bool | None
                             ) -> tuple[jax.Array, jax.Array]: ...


@runtime_checkable
class Predicting(Protocol):
    """A decoder that declares how many multi-token prediction depths it carries."""

    @property
    def num_nextn_predict_layers(self) -> int: ...


@dataclass(frozen=True)
class Sampling:
    """Token selection and termination. Zero temperature is deterministic argmax.

    ``top_k=None`` keeps the vocabulary. EOS counts as a sampled action;
    subsequent output slots contain ``pad_id`` and have no likelihood.

    A ``Sampling`` value compiles to temperature, top-k, top-p and min-p
    transforms when a request has no explicit logits chain. An explicit
    chain replaces those transforms. The EOS criterion still joins the
    request's stopping criteria.
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

    @property
    def stops(self) -> tuple[int, ...]:
        """The EOS ids that end a draw, none when the policy names no EOS."""
        eos = self.eos_id
        return () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)

    def transforms(self) -> tuple[LogitsTransform, ...]:
        """The complete default chain for a request without explicit transforms.

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
        """This process's real rows as host arrays, without the padding a
        row plan added to fill the devices."""
        return jax.tree.map(lambda leaf: local_rows(leaf)[:self.rows], self)

    @functools.cached_property
    def text(self) -> tuple[str, ...]:
        """Each real row's valid continuation, decoded on first access."""
        if self.decoder is None:
            raise ValueError("this generation carries no processor to decode with")
        rows = self.host()
        return self.decoder(rows.tokens, rows.lengths, self.prompt_width)


def _prefill(model: nn.Module, params: Variables, inputs: ModelInputs, ops: DecodeOps,
             cache: Variables | None = None) -> tuple[DecoderState, jax.Array]:
    """The state after the prompt, and which rows hold a real token.

    A decoder that scores one position per row runs its head on the slot the
    first draw reads and nothing else: the head over a whole prompt is the
    largest array a prefill allocates, [rows, width, vocab], and the loop
    keeps one row of it. A decoder without that method scores every position
    and pays for the ones it discards.

    A model with prediction depths also gets their independent cache, seeded
    over the prompt: each depth reads the target's hidden state at one
    position with the token at the next, at that token's own position, which
    is the history a checkpoint's predictor was trained behind. A depth left
    empty would draft the first block from nothing.

    `cache` continues a cache the caller already holds, whose cursors say
    where each row's prompt resumes; None allocates an empty one.
    """
    batch, width = inputs.tokens.shape
    held = _empty_cache(model, params, batch, ops.depths) if cache is None else cache
    exposed = isinstance(model, Exposing)
    selective = isinstance(model, Selective)
    # An unpadded prompt carries no validity field, and its last real token is
    # the last slot.
    valid = inputs.token_fields.get("attention_mask")
    last = (jnp.full((batch,), width - 1, jnp.int32) if valid is None else
            jnp.max(jnp.where(valid, jnp.arange(width)[None, :], -1), axis=1))
    rows, slot = jnp.arange(batch), jnp.maximum(last, 0)
    scored = (inputs.tokens, slot) if selective else (inputs.tokens,)
    answer, updated = model.apply(
        {**params, "cache": held}, *scored, decode=True,
        mutable=["cache", "embeddings"], rngs=None,
        method=("states_and_logits_at" if selective else
                "states_and_logits" if exposed else None), capture_intermediates=False,
        **inputs.kwargs())
    states, logits = answer if exposed or selective else (None, answer)
    drawn = logits if selective else logits[rows, slot]
    supplied = inputs.token_fields.get("positions")
    rotary = inputs.token_fields.get("rotary_positions")
    logical = rotary if rotary is not None else supplied
    # The model's own next coordinate, as its cache records it: the largest
    # coordinate a real token holds on any axis, one on. A drawn token
    # continues from there on every axis.
    real = jnp.ones((batch, width), bool) if valid is None else valid.astype(bool)
    positions = None if logical is None else jnp.max(
        jnp.where(jnp.reshape(real, real.shape + (1,) * (logical.ndim - 2)), logical, -1),
        axis=tuple(range(1, logical.ndim))) + 1
    state = DecoderState(updated["cache"], drawn, positions,
                         None if states is None else states[rows, slot])
    prepared = jax.tree.leaves(updated.get("embeddings", {}))
    if ops.depths > 1 and states is not None:
        # Every later depth needs a predecessor slot in the carry from the
        # start, so a block's loop keeps one structure whatever the prompt
        # held; a depth without enough history never reads it.
        state = dataclasses.replace(state, drafts=(states[rows, slot],) * (ops.depths - 1))
    if ops.depths and width > 1 and states is not None and prepared:
        state = _seeded_depths(ops, state, states, prepared[0], real, logical, batch, width)
    return state, last >= 0


def _empty_cache(model: nn.Module, params: Variables, batch: int, depths: int) -> Variables:
    """A zeroed decode cache for `batch` rows, with the prediction depths' own beside it."""
    cache = model.apply(params, batch, method="init_cache", mutable=["cache"])[1]["cache"]
    if depths:
        drafting = model.apply(params, batch, method="init_mtp_cache", mutable=["cache"])[1]["cache"]
        cache = unflatten_dict({**flatten_dict(dict(cache)), **flatten_dict(dict(drafting))})
    return cache


def _seeded_depths(ops: DecodeOps, state: DecoderState, states: jax.Array,
                   embeddings: jax.Array, real: jax.Array, logical: jax.Array | None,
                   batch: int, width: int) -> DecoderState:
    """`state` with each prediction depth's cache seeded over the prompt.

    A padded prompt's real tokens are compacted to the front first, so a
    depth reads the target's hidden state at one position with the token at
    the next, at that token's own coordinate, which is the history a
    checkpoint's predictor was trained behind.
    """
    order = jnp.argsort(~real, axis=1, stable=True)[..., None]
    lengths = jnp.sum(real, axis=1, dtype=jnp.int32)
    # Without supplied coordinates a token's position is its rank among
    # the row's real tokens, which is what the cache assigns; the physical
    # slot a padded prompt put it in is not a coordinate.
    coordinates = (jnp.take_along_axis(logical.astype(jnp.int32), order, axis=1)
                   if logical is not None and logical.ndim == 3 else
                   jnp.broadcast_to(jnp.arange(width)[None, :], (batch, width))
                   if logical is None
                   else jnp.take_along_axis(logical.astype(jnp.int32), order[..., 0], axis=1))
    compact = states[jnp.arange(batch)[:, None], order[..., 0]]
    state, _, carried = strategies.reseed(
        ops, state, (compact[:, 0],) + (None,) * (ops.depths - 1), compact[:, 1:],
        jnp.take_along_axis(embeddings, order, axis=1)[:, 1:],
        jnp.arange(width - 1)[None, :] < (lengths - 1)[:, None],
        coordinates[:, 1:], jnp.maximum(lengths - 2, 0),
        prior_tokens=jnp.ones(batch, jnp.int32))
    return dataclasses.replace(state, drafts=carried[1:])


def prediction_depths(model: nn.Module) -> int:
    """How many multi-token prediction depths the model declares; none unless it is `Predicting`."""
    return model.num_nextn_predict_layers if isinstance(model, Predicting) else 0


def _operations(model: nn.Module, params: Variables, pad_id: int, depths: int) -> DecodeOps:
    """The model operations a strategy may run, bound to these weights.

    Parameters stay unmapped: every operation reads the same tree, and only
    the cache moves with the rows.
    """
    exposed = isinstance(model, Exposing)

    def run(state: DecoderState, tokens: jax.Array, valid: jax.Array
            ) -> tuple[DecoderState, jax.Array, jax.Array | None]:
        width = tokens.shape[1]
        positions = ({} if state.positions is None else
                     {"positions": state.positions[:, None] + jnp.arange(width)[None, :]})
        answer, updated = model.apply(
            {**params, "cache": state.cache}, jnp.where(valid, tokens, pad_id),
            decode=True, attention_mask=valid, mutable=["cache"], rngs=None,
            method="states_and_logits" if exposed else None,
            capture_intermediates=False, **positions)
        states, logits = answer if exposed else (None, answer)
        moved = (None if state.positions is None else
                 state.positions + jnp.sum(valid, axis=1, dtype=state.positions.dtype))
        return (dataclasses.replace(state, cache=updated["cache"], logits=logits[:, -1],
                                    positions=moved,
                                    hidden=None if states is None else states[:, -1]),
                logits, states)

    def advance(state: DecoderState, token: jax.Array, active: jax.Array) -> DecoderState:
        return run(state, token[:, None], active[:, None])[0]

    def reindex(state: DecoderState, rows: jax.Array) -> DecoderState:
        # The whole carry moves, the prediction states with it; a leaf left
        # behind would hand a reparented row another row's history.
        return dataclasses.replace(
            jax.tree.map(lambda leaf: jnp.take(leaf, rows, axis=0),
                         dataclasses.replace(state, cache={})),
            cache=gather_cache_rows(state.cache, rows))

    if not depths or not exposed:
        return DecodeOps(advance, reindex, run)
    return DecodeOps(advance, reindex, run, *_drafting(model, params, pad_id), depths)


def _drafting(model: nn.Module, params: Variables, pad_id: int):
    """The `(propose, embed)` pair a drafting strategy runs the depths with.

    `propose` runs one prediction depth over a candidate token at an explicit
    target position; `embed` prepares the token embeddings a replayed block
    hands back to the depths.
    """
    def propose(state: DecoderState, hidden: jax.Array, tokens: jax.Array | None,
                embeds: jax.Array | None, valid: jax.Array, positions: jax.Array,
                depth: int, prediction_phase: PredictionPhase) -> tuple[DecoderState, jax.Array, jax.Array]:
        ids = jnp.zeros(valid.shape, jnp.int32) if tokens is None else jnp.where(valid, tokens, pad_id)
        # Multi-axis rotary metadata is not a scalar position, and a depth
        # takes it under its own name.
        placed = ({"rotary_positions": positions} if positions.ndim == 3
                  else {"positions": positions})
        (logits, states), updated = model.apply(
            {**params, "cache": state.cache}, hidden, ids, depth=depth, attention_mask=valid,
            input_embeddings=embeds, decode=True, mutable=["cache"], rngs=None,
            prediction_phase=prediction_phase,
            method="mtp_step", capture_intermediates=False, **placed)
        return dataclasses.replace(state, cache=updated["cache"]), logits, states

    def embed(tokens: jax.Array) -> jax.Array:
        prepared = model.apply(params, tokens, method="token_embeddings")
        assert isinstance(prepared, jax.Array)
        return prepared

    return propose, embed


def _generate(model: nn.Module, params: Variables, inputs: ModelInputs, keys: jax.Array,
              max_new_tokens: int, pad_id: int, n: int,
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
    ops = _operations(model, params, pad_id, prediction_depths(model))
    state, real = _prefill(model, params, inputs, ops)
    start = StepState(
        tokens=jnp.concatenate([inputs.tokens, jnp.zeros((batch, max_new_tokens), jnp.int32)], axis=1),
        valid=jnp.concatenate([jnp.ones((batch, width), bool) if valid is None else valid.astype(bool),
                               jnp.zeros((batch, max_new_tokens), bool)], axis=1),
        step=jnp.zeros(batch, jnp.int32), active=real, keys=keys, prompt_width=width)
    drawn: Draws = strategy(state, start, ops, decoding.chain(transforms),
                            decoding.criterion(stopping), max_new_tokens, n)
    return Generation(
        jnp.concatenate([prompt, jnp.where(drawn.valid, drawn.tokens, pad_id)], axis=1),
        jnp.sum(drawn.valid, axis=1, dtype=jnp.int32), drawn.terminated,
        drawn.behavior_log_probs, drawn.raw_log_probs)


def _validated(model: nn.Module, ids: np.ndarray, fields: dict[str, np.ndarray],
               conditioning: dict[str, jax.Array | np.ndarray], max_new_tokens: int, sampling: Sampling,
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
    cache_len = model.max_seq_len if isinstance(model, Bounded) else None
    if cache_len is not None and int(valid.sum(axis=1).max()) + max_new_tokens > cache_len:
        raise ValueError("prompt plus max_new_tokens exceeds max_seq_len; raise the cache capacity")
    vocab = model.vocab_size if isinstance(model, Bounded) else None
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


def resolve(sampling: Sampling, logits: Transforms | None, stopping: Criteria | None,
            strategy: Strategy | None) -> Components:
    """The one chain, criterion and strategy a request runs.

    `logits=None` runs the policy's own filtering tail. An explicit sequence
    is the whole chain instead, in the order given, so a caller that needs a
    different order than `Sampling` produces writes the order it wants, and
    `()` runs no transform at all.

    Criteria always compose: an explicit sequence runs alongside the policy's
    EOS criterion rather than replacing it, so naming a criterion cannot drop
    termination by accident. No strategy means `Sample`.
    """
    transforms = sampling.transforms() if logits is None else logits
    return (decoding.components(transforms, "logits"),
            (() if stopping is None else decoding.components(stopping, "stopping"))
            + sampling.criteria(),
            Sample() if strategy is None else strategies.as_pytree(strategy))


def _named(value: SelfNaming) -> str:
    """A component's identity, the same string in every process.

    A repr would carry the object's address, and two processes that resolved
    the same chain would then look like they disagreed.
    """
    return f"{value.__module__}.{value.__qualname__}"


def _stable(value: object, seen: frozenset[int] = frozenset()) -> Identity:
    """A component's configuration as something every process can compare.

    The description reaches every part a rank could differ in: a dataclass's
    fields including its array shapes, a partial's function and bound
    arguments, and a function's captured cells and defaults. A name alone is
    not enough, because two ranks can hold the same nested function closed
    over different values. Module-level helpers and library references keep
    working, since globals are not captured cells.

    Anything whose text carries an address cannot be compared across
    processes, so it is refused here rather than after the ranks have entered
    different device loops.
    """
    if isinstance(value, (bool, int, float, complex, str, bytes)) or value is None:
        return repr(value)
    if id(value) in seen:
        return "cycle"
    seen = seen | {id(value)}
    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_stable(entry, seen) for entry in value))
    if isinstance(value, Mapping):
        return ("mapping", tuple(sorted((repr(name), _stable(entry, seen))
                                        for name, entry in value.items())))
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        return ("array", *_hashed(value))
    if isinstance(value, functools.partial):
        return ("partial", _stable(value.func, seen), _stable(value.args, seen),
                _stable(dict(value.keywords), seen))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (_named(type(value)), *tuple(
            (field.name, _stable(getattr(value, field.name), seen))
            for field in dataclasses.fields(value)))
    if isinstance(value, types.FunctionType):
        cells = []
        for cell in value.__closure__ or ():
            try:
                captured = cell.cell_contents
            except ValueError:
                cells.append("empty")
                continue
            cells.append(_stable(captured, seen))
        return ("function", _named(value), tuple(cells), _stable(value.__defaults__ or (), seen),
                _stable(value.__kwdefaults__ or {}, seen))
    kind = value if isinstance(value, SELF_NAMING) else type(value)
    text = repr(value)
    if "0x" in text:
        raise ValueError(
            f"a decoding component holds {_named(kind)}, whose identity is this process's "
            "memory address; give the configuration as arrays or plain values so a pool can "
            "agree on it")
    return (_named(kind), text)


def _hashed(value: np.ndarray | np.generic | jax.Array) -> tuple[tuple[int, ...], str, str]:
    """An array's shape, dtype and contents, as a comparable triple."""
    array = np.ascontiguousarray(np.asarray(value))
    return (array.shape, str(array.dtype), hashlib.sha256(array.tobytes()).hexdigest())


def _digest(components: Components) -> tuple[Identity, str]:
    """Resolved components as a value every process can compare.

    The configuration arrays are part of the identity: two processes holding
    `EndOfSequence` over different ids build the same tree of the same shapes
    and would agree on a schema alone. Each leaf contributes its shape, dtype
    and contents separately, so moving a value from one leaf to the next
    changes the digest too, and an array a plain function captured is covered
    by the description even though no pytree leaf holds it.
    """
    payload = hashlib.sha256()
    for leaf in jax.tree.leaves(components):
        payload.update(repr(_hashed(leaf)).encode())
    return (_stable(components), payload.hexdigest())


def _request(model: nn.Module, params: Variables,
             inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
             key: jax.Array | None, seed: int | None, sampling: Sampling, n: int,
             logits: Transforms | None, stopping: Criteria | None, strategy: Strategy | None,
             *, pooled: bool) -> tuple[ModelInputs, jax.Array, Components, tuple[Identity, ...]]:
    """This process's validated request and the controls a pool compares.

    Everything a rank can get wrong on its own is raised from here, the
    component digest included, so a pool agrees on the failure before any
    rank enters a collective. A single process never digests: refusing a
    component only a pool could disagree about would cost it nothing.
    """
    random_key = request_key(key, seed)
    canonical = ModelInputs.from_value(inputs)
    ids = local_rows(canonical.tokens)
    fields = {name: local_rows(value) for name, value in canonical.token_fields.items()}
    conditioning = {name: local_rows(value, host=False) for name, value in canonical.conditioning.items()}
    if "params" not in params:
        raise ValueError("generate takes the full variables dict ({'params': ...})")
    prepared = _validated(model, ids, fields, conditioning, max_new_tokens, sampling, n)
    components = resolve(sampling, logits, stopping, strategy)
    controls = (max_new_tokens, n, sampling.pad_id) + ((_digest(components),) if pooled else ())
    return prepared, random_key, components, controls


def _checked(model: nn.Module, params: Variables, inputs: ModelInputs, keys: jax.Array,
             max_new_tokens: int, pad_id: int, n: int,
             transforms: tuple[LogitsTransform, ...], stopping: tuple[Stopping, ...],
             strategy: Strategy) -> tuple[checkify.Error, Generation]:
    """`_generate` with its device checks discharged into a returned error.

    An undefined draw has to reach the caller as an exception rather than a
    fabricated token, and a device loop cannot raise. `checkify` carries the
    failure out of the computation; with no check emitted the error is empty
    and throwing it costs nothing.
    """

    def run(params, inputs, keys, transforms, stopping, strategy):
        return _generate(model, params, inputs, keys, max_new_tokens, pad_id, n,
                         transforms, stopping, strategy)

    return checkify.checkify(run, errors=checkify.user_checks)(
        params, inputs, keys, transforms, stopping, strategy)


@functools.cache
def _compiled(rows: jax.sharding.NamedSharding | None):
    return jax.jit(_checked, static_argnames=("model", "max_new_tokens", "pad_id", "n"),
                   in_shardings=(None, rows, rows, None, None, None),
                   out_shardings=(None, rows))


def _agreed_request(prepared: ModelInputs, processes: int, controls) -> ModelInputs:
    """`prepared` under one validity schema, with the pool agreed on the rest.

    Whether this process's own prompts needed padding is rank-local, and the
    digest would refuse a pool that disagrees only about that, so the schema
    is agreed first. The digest then covers all conditioning and token
    fields, not token length alone, because different traced shapes would
    issue mismatched collectives.
    """
    prepared = agreed_validity(prepared, processes, controls=controls, phase="generation input")
    multihost_utils.assert_equal(
        generation_signature(prepared, controls),
        "generation input shapes, continuations, decoding components and padding "
        "must agree across processes")
    return prepared


def _padded(plan: RowPlan, prepared: ModelInputs) -> ModelInputs:
    """`prepared` filled out to the rows the devices take.

    Repeated rows carry no real token, so they finish at once and emit
    nothing. Their validity is the field an unpadded request omitted.
    """
    padded = plan.pad(prepared)
    if plan.count == plan.rows:
        return padded
    existing = padded.token_fields.get("attention_mask")
    valid = jnp.ones(padded.tokens.shape, bool) if existing is None else existing
    return replace(padded, token_fields={**padded.token_fields,
                                         "attention_mask": valid & ~plan.padding[:, None]})


@overload
def generate(model: nn.Module, params: Variables,
             inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
             *, key: jax.Array, sampling: Sampling = Sampling(), n: int = 1,
             logits: Transforms | None = None, stopping: Criteria | None = None,
             strategy: Strategy | None = None) -> Generation: ...


@overload
def generate(model: nn.Module, params: Variables,
             inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
             *, seed: int, sampling: Sampling = Sampling(), n: int = 1,
             logits: Transforms | None = None, stopping: Criteria | None = None,
             strategy: Strategy | None = None) -> Generation: ...


def generate(model: nn.Module, params: Variables,
             inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
             *, key: jax.Array | None = None, seed: int | None = None,
             sampling: Sampling = Sampling(), n: int = 1, logits: Transforms | None = None,
             stopping: Criteria | None = None, strategy: Strategy | None = None) -> Generation:
    """Generate from numeric model inputs, with an array shorthand for text.

    ModelInputs.token_fields["attention_mask"] identifies real tokens. Missing masks mean all
    tokens are real. Every row must contain a real token. Prefill evaluates
    conditioning once; decode reuses the model-owned cache and logical-position
    state. Each cache compacts real input tokens and leaves paused rows intact.

    Parameters keep their placement. On a mesh, rows split over its batch
    axes and the result keeps that sharding; ``Generation.host()`` reads a
    process's own rows back. All cooperating processes use the same input
    shapes, effective decoding components, padding id and continuation count.
    Decode loops have fixed bounds; any skipped blocks are globally agreed.
    Keys fold in the global row index and response position, so a pool draws
    what one process draws for the same rows.

    ``n`` continuations of each prompt share its prefill and leave as ``n``
    consecutive rows of every array, in prompt order. Continuation zero of a
    prompt draws with that prompt's own key, so ``n=1`` and continuation zero
    of any larger request are the same draw.

    ``logits`` is the whole transform chain, in the order it runs. Left as
    ``None`` it is what ``sampling`` compiles to, and ``()`` runs no
    transform. ``stopping`` adds criteria beside the policy's EOS one rather
    than replacing it. ``strategy`` replaces the per-row draw loop; ``None``
    uses ``Sample``.
    """
    mesh = mesh_of(params)
    processes = jax.process_count() if mesh is not None else 1

    def resolve():
        return _request(model, params, inputs, max_new_tokens, key, seed, sampling, n,
                        logits, stopping, strategy, pooled=processes > 1)

    request = (agreed("generation input validation", resolve) if processes > 1 else resolve())
    prepared, random_key, components, controls = request
    if processes > 1:
        prepared = _agreed_request(prepared, processes, controls)
    plan = RowPlan.over(mesh, prepared.tokens.shape[0])
    padded = _padded(plan, prepared)
    failure, output = _compiled(plan.sharding)(model, params, plan.place(padded),
                                               plan.keys(random_key), max_new_tokens, sampling.pad_id, n,
                                               *components)
    failure.throw()
    return replace(output, rows=plan.rows * n)
