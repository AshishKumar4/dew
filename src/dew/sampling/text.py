"""Cached text generation with explicit lengths and sampling likelihoods.

`generate` validates a request on the host, prefills the prompt once, and
hands the decode loop to a `Strategy`. `Sampling` is the default policy: the
common generation controls as one value, compiled in Transformers' order,
with an EOS criterion. `Generation`
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
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Generic, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import checkify
from jax.typing import ArrayLike
from typing_extensions import TypeVar

from dew._hashing import array_fingerprint
from dew._records import finite
from dew.batch import Variables, real_slots, token_positions, valid_order, with_validity
from dew.interop.weights import decoding_storage
from dew.nn.inputs import ModelInputs, PredictionPhase, Request, host_rows, local_rows, mesh_of, next_position
from dew.nn.kernels.options import program
from dew.nn.kv_cache import gather_cache_rows, refuse_unassigned, write_cache
from dew.nn.protocols import (
    BlockDenoiser,
    BlockDrafting,
    Bounded,
    Decoding,
    Exposing,
    Generating,
    Lacks,
    Layered,
    Predicting,
    ProjectionGroup,
    Selective,
    Serving,
    refuse_prefix,
)
from dew.nn.visibility import with_served_blocks
from dew.objectives.base import ROUNDING_STREAM
from dew.sampling import decoding, strategies
from dew.sampling.decoding import (
    EndOfSequence,
    FrequencyPenalty,
    Greedy,
    LogitsTransform,
    MinNewTokens,
    MinP,
    NoRepeatNGram,
    PresencePenalty,
    RepetitionPenalty,
    StepState,
    Stopping,
    Temperature,
    TopK,
    TopP,
    Typical,
)
from dew.sampling.strategies import DecodeOps, DecoderState, Draws, Sample, Speculative, Strategy

ArrayT = TypeVar("ArrayT", bound=jax.Array | np.ndarray, default=jax.Array, covariant=True)

Transforms = LogitsTransform | Sequence[LogitsTransform]
Criteria = Stopping | Sequence[Stopping]
Components = tuple[tuple[LogitsTransform, ...], decoding.Finishing, Strategy]

# A resolved component's configuration as every process can spell it: a repr,
# a number (an array's shape is a tuple of them), or a tuple of either nested
# as deep as the component is. Only `repr` and `==` are applied to one, by the
# pool agreement in `dew.nn.inputs` and by the digest that precedes it.
type Identity = str | int | tuple[Identity, ...]
# What carries its own qualified name, and so names a component: a class, a
# function, a bound method, a library builtin. Anything else is named by its
# class, which is one of these.
type SelfNaming = type | types.FunctionType | types.MethodType | types.BuiltinFunctionType
SELF_NAMING: tuple[type[SelfNaming], ...] = get_args(SelfNaming.__value__)


# Transformers' `_get_logits_processor` order, by its own control names: the
# processors, then the sampling warpers. Presence and frequency penalties are
# vLLM's (and OpenAI's) and sit beside the repetition penalty, where vLLM
# applies its penalties together. `Sampling` holds the common controls; a
# source's generation config hands the rest in as built transforms.
PROCESSORS = ("sequence_bias", "encoder_repetition_penalty", "repetition_penalty", "presence_penalty",
              "frequency_penalty", "no_repeat_ngram_size", "encoder_no_repeat_ngram_size", "bad_words_ids",
              "min_length", "min_new_tokens", "forced_bos_token_id", "forced_eos_token_id",
              "remove_invalid_values", "exponential_decay_length_penalty", "suppress_tokens",
              "begin_suppress_tokens")
WARPERS = ("temperature", "top_h", "top_k", "top_p", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff")


@dataclass(frozen=True)
class Sampling:
    """Common generation controls that decide how tokens are drawn and when a row stops.

    Zero temperature is deterministic argmax, and ``top_k=None`` keeps the
    whole vocabulary. EOS counts as a sampled action; the output slots after
    it contain ``pad_token_id`` and have no likelihood.

    The repetition penalty, ``no_repeat_ngram_size``, ``min_new_tokens`` and
    ``typical_p`` are Transformers' controls of the same names, and
    ``presence_penalty`` and ``frequency_penalty`` are vLLM's. Each one's
    default changes nothing. ``stop`` ends a row whose text ends with one of
    the strings; a task compiles them against its processor.

    ``eos_token_ids`` and ``pad_token_id``, Transformers' names, belong to the
    model and its tokenizer, so where a policy leaves them None, a task fills
    in its own, as Transformers does. Plain `generate` has no task, so it
    stops on EOS only when the policy names one, and pads with 0 unless the
    policy names a ``pad_token_id``. The block and masked decoders take the
    same two names.

    A ``Sampling`` value compiles to its transforms in Transformers' order
    when a request has no explicit logits chain. An explicit chain replaces
    those transforms. The EOS and stop-string criteria still join the
    request's stopping criteria.
    """

    temperature: float = 1.0
    top_k: int | None = None
    eos_token_ids: int | tuple[int, ...] | None = None
    pad_token_id: int | None = None
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    min_new_tokens: int = 0
    typical_p: float = 1.0
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # A rate is any real number, Python's or NumPy's (a config's or a sweep's), held as a float; zero
        # is greedy decoding, so the temperature is the one rate the controls' own checks do not cover.
        rates = {name: decoding.unit_interval(name, getattr(self, name))
                 for name in ("top_p", "min_p", "typical_p")}
        rates["temperature"] = finite(self.temperature, "temperature")
        if rates["temperature"] < 0:
            raise ValueError("temperature must be finite and non-negative")
        rates["repetition_penalty"] = decoding.positive("repetition_penalty", self.repetition_penalty)
        for name in ("presence_penalty", "frequency_penalty"):
            rates[name] = finite(getattr(self, name), name)
        if self.top_k is not None:
            object.__setattr__(self, "top_k", decoding.positive_size("top_k", self.top_k))
        for name, rate in rates.items():
            object.__setattr__(self, name, rate)
        for name, value in (("no_repeat_ngram_size", self.no_repeat_ngram_size),
                            ("min_new_tokens", self.min_new_tokens)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.pad_token_id is not None and (type(self.pad_token_id) is not int or self.pad_token_id < 0):
            raise ValueError("pad_token_id must be a non-negative token id")
        if self.eos_token_ids is not None:
            object.__setattr__(self, "eos_token_ids", decoding.id_row("eos_token_ids", self.eos_token_ids))
        if isinstance(self.stop, str):
            raise ValueError(f"stop is a tuple of strings; write stop=({self.stop!r},)")
        if any(not isinstance(string, str) or not string for string in self.stop):
            raise ValueError("stop must hold non-empty strings")
        # A record reads a JSON list back; the policy holds a tuple.
        object.__setattr__(self, "stop", tuple(self.stop))

    @property
    def pad(self) -> int:
        """The id in the output slots after EOS, which is `pad_token_id`, or 0 when it is None."""
        return 0 if self.pad_token_id is None else self.pad_token_id

    def active(self, names: Sequence[str]) -> list[str]:
        """Return the controls in `names` that this policy sets away from their defaults.

        A client for a remote engine that cannot apply some controls the way
        Dew does passes their names here, and refuses a policy that sets any
        of them so the engine never draws without them.
        """
        neutral = Sampling()
        return [name for name in names if getattr(self, name) != getattr(neutral, name)]

    @property
    def stops(self) -> tuple[int, ...]:
        """The EOS ids that end a draw, or an empty tuple when the policy names no EOS."""
        eos = self.eos_token_ids
        return () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)

    def transforms(self) -> tuple[LogitsTransform, ...]:
        """Return the full default chain that a request without explicit transforms runs.

        At zero temperature the chain ends in the argmax and runs none of the
        sampling-only filters, as Transformers does with `do_sample=False`.
        To add a transform of your own, pass
        `logits=policy.transforms() + (mine,)`.
        """
        return ordered_transforms(self)

    def criteria(self) -> tuple[Stopping, ...]:
        """Return the EOS criterion this policy adds after a caller's criteria, empty when it names no EOS."""
        if self.eos_token_ids is None:
            return ()
        return (EndOfSequence(jnp.asarray(self.eos_token_ids, jnp.int32)),)


def with_ids_of(policy: Sampling, defaults: Sampling) -> Sampling:
    """Return `policy` with the EOS and pad ids it leaves None taken from `defaults`.

    A task's policy holds its model's and tokenizer's ids, so a call that
    replaces the policy keeps stopping and padding where the task does.
    """
    return replace(policy, eos_token_ids=(defaults.eos_token_ids if policy.eos_token_ids is None
                                          else policy.eos_token_ids),
                   pad_token_id=defaults.pad_token_id if policy.pad_token_id is None else policy.pad_token_id)


def ordered_transforms(policy: Sampling, extra: Mapping[str, LogitsTransform] = types.MappingProxyType({}),
                       *, searching: bool = False) -> tuple[LogitsTransform, ...]:
    """Build `policy`'s chain, with `extra` beside it, in Transformers' order.

    `extra` holds transforms for the controls a `Sampling` does not carry, by
    their Transformers names in `PROCESSORS` and `WARPERS`, plus
    `renormalize_logits`, which runs last; a source's generation config is
    what names them. Zero temperature ends the processors with the argmax
    and runs no warper, and a beam search (`searching`) picks its own
    continuations, so its chain ends after the processors.
    """
    eos = None if policy.eos_token_ids is None else jnp.asarray(policy.eos_token_ids, jnp.int32)
    if policy.min_new_tokens and eos is None:
        raise ValueError("min_new_tokens holds EOS back, and this policy names no eos_token_ids; "
                         "set it, or let a task fill it")
    own: dict[str, LogitsTransform | None] = {
        "repetition_penalty": (RepetitionPenalty(policy.repetition_penalty)
                               if policy.repetition_penalty != 1.0 else None),
        "presence_penalty": PresencePenalty(policy.presence_penalty) if policy.presence_penalty else None,
        "frequency_penalty": FrequencyPenalty(policy.frequency_penalty) if policy.frequency_penalty else None,
        "no_repeat_ngram_size": (NoRepeatNGram(policy.no_repeat_ngram_size)
                                 if policy.no_repeat_ngram_size else None),
        "min_new_tokens": (MinNewTokens(policy.min_new_tokens, eos)
                           if eos is not None and policy.min_new_tokens else None),
        "temperature": Temperature(policy.temperature) if policy.temperature not in (0.0, 1.0) else None,
        "top_k": None if policy.top_k is None else TopK(policy.top_k),
        "top_p": TopP(policy.top_p) if policy.top_p < 1.0 else None,
        "min_p": MinP(policy.min_p) if policy.min_p > 0.0 else None,
        "typical_p": Typical(policy.typical_p) if policy.typical_p < 1.0 else None,
    }
    misplaced = sorted(set(extra) & set(own) | set(extra) - {*PROCESSORS, *WARPERS, "renormalize_logits"})
    if misplaced:
        raise ValueError(f"{misplaced} are not controls a chain takes beside a Sampling policy")
    present = {**extra, **{name: transform for name, transform in own.items() if transform is not None}}
    chain = [present[name] for name in PROCESSORS if name in present]
    if not searching:
        if policy.temperature == 0:
            chain.append(Greedy())
        else:
            chain += [present[name] for name in WARPERS if name in present]
    if "renormalize_logits" in extra:
        chain.append(extra["renormalize_logits"])
    return tuple(chain)


@struct.dataclass
class Generation(Generic[ArrayT]):
    """The prompt and padded continuation of each row, with response-aligned log-probabilities.

    ``lengths`` counts each row's response actions, including EOS.
    ``terminated`` is true where a stopping criterion (EOS by default) ended
    the row and false where the length limit did. Both log-probability
    arrays have shape [B, max_new_tokens], and only positions below
    ``lengths`` are valid. ``behavior_log_probs`` describes the distribution
    that actually drew each action, after the whole transform chain.
    ``raw_log_probs`` describes the unmodified model policy.

    A request for ``n`` continuations per prompt gives every array
    ``[B * n, ...]`` rows: prompt zero's ``n`` continuations, then prompt
    one's. Each row has its own length, termination and likelihoods.

    The arrays keep the placement the task ran with. On a mesh they are
    global arrays whose rows are split over the batch axes and padded to the
    device count. ``host()`` reads this process's ``rows`` real rows back as
    host arrays, and ``text`` decodes them through the task's processor.
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
        """Return this process's real rows as host arrays, without the padding rows that fill the devices."""
        return jax.tree.map(lambda leaf: host_rows(leaf, self.rows), self)

    @functools.cached_property
    def text(self) -> tuple[str, ...]:
        """Each real row's valid continuation, decoded on first access (`row_texts`)."""
        return row_texts(self.decoder, self, self.prompt_width)


def decoded_rows(processor, tokens: ArrayLike, lengths: ArrayLike, width: int) -> tuple[str, ...]:
    """Each row's `lengths` tokens past `width`, as text, by `processor`: what a task's `decode` and a
    generation's `text` both return (`row_texts`). Without a processor there is no text to return, and
    it raises."""
    if processor is None:
        raise ValueError("this generation carries no processor to decode with")
    rows, counts = np.asarray(tokens), np.asarray(lengths)
    return tuple(processor.decode(rows[row:row + 1, width:width + int(counts[row])])[0]
                 for row in range(rows.shape[0]))


def row_texts(decoder, generation, prompt_width: int | None) -> tuple[str, ...]:
    """A text or canvas generation's real rows decoded past their prompt by `decoder`, `decoded_rows`
    over a processor: its rows on the host (`generation.host()`), each row's tokens past
    `prompt_width`. A generation with no decoder, or with no prompt width to decode from, is refused
    (ValueError) rather than decoded from the start of its tokens."""
    if decoder is None:
        raise ValueError("this generation carries no processor to decode with")
    if prompt_width is None:
        raise ValueError("this generation has no prompt width")
    rows = generation.host()
    return decoder(rows.tokens, rows.lengths, prompt_width)


def rounding_keys(keys: jax.Array) -> jax.Array:
    """The key data of each row's stochastic-rounding stream (`DecoderState.rounding`): a default-impl
    key, `ROUNDING_STREAM`'s, folded with each word of the row's request key in turn, so it is apart
    from the sampling stream whatever that key's implementation, as a training step's rounding stream
    is apart from its dropout (`dew.objectives.base.training_rngs`)."""
    def derived(words: jax.Array) -> jax.Array:
        key = jax.random.key(ROUNDING_STREAM)
        for index in range(words.shape[0]):
            key = jax.random.fold_in(key, words[index])
        return jax.random.key_data(key)

    return jax.vmap(derived)(jax.random.key_data(keys))


def rounding_rngs(rounding: jax.Array | None) -> dict[str, jax.Array] | None:
    """A model call's RNG streams from the rows' rounding key data (`DecoderState.rounding`): its
    `stochastic_rounding` stream, the first row's key; none where the decode carries none. A cache's
    allocation takes it too, as its dummy decode step runs the model."""
    return None if rounding is None else {"stochastic_rounding": jax.random.wrap_key_data(rounding)[0]}


def _rounded(rounding: jax.Array | None) -> tuple[dict[str, jax.Array] | None, jax.Array | None]:
    """A model call's streams (`rounding_rngs`) and every row's key for the next call."""
    if rounding is None:
        return None, None
    keys = jax.random.wrap_key_data(rounding)
    following = jax.random.key_data(jax.vmap(lambda key: jax.random.fold_in(key, 1))(keys))
    return rounding_rngs(rounding), following


def prefill_state(model: nn.Module, params: Variables, inputs: ModelInputs, ops: DecodeOps,
                  cache: Variables | None = None, rounding: jax.Array | None = None
                  ) -> tuple[DecoderState, jax.Array]:
    """The state after the prompt, and which rows hold a real token.

    A `Selective` decoder skips the prompt-wide head, the largest array a
    prefill allocates, `[rows, width, vocab]`. Prediction depths get their
    cache seeded over the prompt (`_seeded_depths`), so the first block does
    not draft from nothing, and a block drafter's windows take every real
    prompt position's context. `cache` continues a cache the caller holds,
    whose cursors say where each prompt resumes; None allocates an empty one. `rounding` is each
    row's stochastic-rounding key data (`rounding_keys`), which the prefill and every later call draws
    from (`DecoderState.rounding`).
    """
    batch, width = inputs.tokens.shape
    rngs, rounding = _rounded(rounding)
    held = _empty_cache(model, params, batch, ops, rngs) if cache is None else cache
    exposed = isinstance(model, Exposing)
    selective = isinstance(model, Selective)
    real = real_slots(inputs.token_fields, (batch, width))
    last = jnp.max(jnp.where(real, jnp.arange(width)[None, :], -1), axis=1)
    rows, slot = jnp.arange(batch), jnp.maximum(last, 0)
    scored = (inputs.tokens, slot) if selective else (inputs.tokens,)
    answer, updated = model.apply(
        {**params, "cache": held},
        *scored,
        decode=True,
        mutable=["cache", "embeddings", *(["prediction_inputs"] if ops.record is not None else [])],
        rngs=rngs,
        method=("states_and_logits_at" if selective else "states_and_logits" if exposed else None),
        capture_intermediates=False,
        **inputs.kwargs(),
    )
    states, logits = answer if exposed or selective else (None, answer)
    if ops.record is not None:
        states = _context(model, params, updated)
    drawn = logits if selective else logits[rows, slot]
    rotary = inputs.token_fields.get("rotary_positions")
    logical = rotary if rotary is not None else token_positions(inputs.token_fields)
    state = DecoderState(updated["cache"], drawn, next_position(inputs.token_fields, real),
                         None if states is None else states[rows, slot], rounding=rounding)
    if rebuild_position(model) is not None:
        if cache is not None or logical is not None or inputs.conditioning:
            raise ValueError('LongRoPE cache rebuilding requires a fresh text prompt '
                             'without custom coordinates')
        if ops.depths or ops.record is not None:
            raise ValueError("LongRoPE cache rebuilding recomputes the trunk's prefix alone, not a "
                             "prediction depth's or a block drafter's cache, so it decodes without drafting")
        lengths = jnp.sum(real, axis=1, dtype=jnp.int32)
        slots = jnp.where(real, jnp.cumsum(real, axis=1, dtype=jnp.int32) - 1, -1)
        history = write_cache(jnp.zeros((batch, model.max_seq_len), inputs.tokens.dtype),
                              inputs.tokens, slots)
        state = dataclasses.replace(state, tokens=history, lengths=lengths)
    prepared = jax.tree.leaves(updated.get("embeddings", {}))
    if ops.depths > 1 and states is not None:
        # Every later depth needs a predecessor slot in the carry from the
        # start, so a block's loop keeps one structure whatever the prompt
        # held; a depth without enough history never reads it.
        state = dataclasses.replace(state, drafts=(states[rows, slot],) * (ops.depths - 1))
    if ops.depths and width > 1 and states is not None and prepared:
        state = _seeded_depths(ops, state, states, prepared[0], real, logical, batch, width)
    if ops.record is not None and states is not None:
        state = ops.record(state, states, real)
    return state, last >= 0


def _empty_cache(model: nn.Module, params: Variables, batch: int, ops: DecodeOps,
                 rngs: dict[str, jax.Array] | None = None) -> Variables:
    """A zeroed decode cache for `batch` rows, with the drafter's own beside it, its dummy step drawing
    from `rngs` (`rounding_rngs`). A decoder whose every mixer is tokenwise (`MLPMixer`) keeps no
    cache, and its collection is empty."""
    def initialized(method: str) -> dict:
        _, mutated = model.apply(params, batch, method=method, mutable=["cache"], rngs=rngs)
        return flatten_dict(dict(mutated["cache"])) if "cache" in mutated else {}

    cache = initialized("init_cache")
    for method in ("init_mtp_cache",) * bool(ops.depths) + ("init_draft_cache",) * (ops.record is not None):
        cache.update(initialized(method))
    return unflatten_dict(cache)


def _context(model: nn.Module, params: Variables, updated: Mapping) -> jax.Array:
    """The block drafter's context `[B, S, targets * D]` a forward recorded."""
    context = model.apply(params, updated["prediction_inputs"], method="draft_context")
    assert isinstance(context, jax.Array)
    return context


def _seeded_depths(ops: DecodeOps, state: DecoderState, states: jax.Array,
                   embeddings: jax.Array, real: jax.Array, logical: jax.Array | None,
                   batch: int, width: int) -> DecoderState:
    """`state` with each prediction depth's cache seeded over the prompt.

    A padded prompt's real tokens are compacted to the front first, so a
    depth reads the target's hidden state at one position with the token at
    the next, at that token's own coordinate, which is the history a
    checkpoint's predictor was trained behind.
    """
    order, lengths = valid_order(real)
    order = order[..., None]
    # Without supplied coordinates a token's position is its rank among
    # the row's real tokens, which is what the cache assigns; the physical
    # slot a padded prompt put it in is not a coordinate.
    coordinates = (jnp.take_along_axis(logical.astype(jnp.int32), order, axis=1)
                   if logical is not None and logical.ndim == 3 else
                   jnp.broadcast_to(jnp.arange(width)[None, :], (batch, width))
                   if logical is None
                   else jnp.take_along_axis(logical.astype(jnp.int32), order[..., 0], axis=1))
    compact = states[jnp.arange(batch)[:, None], order[..., 0]]
    state, carried = strategies.reseed(
        ops, state, (compact[:, 0],) + (None,) * (ops.depths - 1), compact[:, 1:],
        jnp.take_along_axis(embeddings, order, axis=1)[:, 1:],
        jnp.arange(width - 1)[None, :] < (lengths - 1)[:, None],
        coordinates[:, 1:], jnp.maximum(lengths - 2, 0),
        prior_tokens=jnp.ones(batch, jnp.int32))
    return dataclasses.replace(state, drafts=carried[1:])


def prediction_depths(model: nn.Module) -> int:
    """How many multi-token prediction depths the model declares; none unless it is `Predicting`."""
    return model.num_nextn_predict_layers if isinstance(model, Predicting) else 0


def rebuild_position(model: nn.Module) -> int | None:
    """The position past which the model's cached keys go stale, so a row
    crossing it recomputes its prefix (LongRoPE's switch of factors), or
    None where the model's cache stays valid (`Serving`)."""
    return model.cache_rebuild_position if isinstance(model, Serving) else None


def _refuse_generation(model: nn.Module) -> None:
    """Refuse a model that says why a generation loop cannot continue its rows (`Generating`)."""
    refusal = model.generation_refusal() if isinstance(model, Generating) else None
    if refusal is not None:
        raise ValueError(refusal)


def decode_ops(model: nn.Module, params: Variables, pad_id: int, depths: int | None) -> DecodeOps:
    """The model operations a strategy may run, bound to these weights.

    `depths` is the prediction depths a drafting strategy runs, beside the
    model's block drafter where it has one; None, for a strategy that does
    not draft (any but `Speculative`), runs neither, so the prefill
    allocates and seeds no drafting cache.

    Parameters stay unmapped: every operation reads the same tree, and only
    the cache moves with the rows. A model with a block drafter hands its
    drafter's context back as the states `verify` returns.
    A LongRoPE crossing costs one full-prefix forward over the whole batch
    to keep static shapes; it happens once per request.
    """
    exposed = isinstance(model, Exposing)
    blocks = (_block_drafting(model, params)
              if depths is not None and isinstance(model, BlockDrafting) and model.dspark else None)
    depths = depths or 0
    recorded = [] if blocks is None else ["prediction_inputs"]

    def run(state: DecoderState, tokens: jax.Array, valid: jax.Array
            ) -> tuple[DecoderState, jax.Array, jax.Array | None]:
        width = tokens.shape[1]
        positions = ({} if state.positions is None else
                     {"positions": state.positions[:, None] + jnp.arange(width)[None, :]})
        rngs, rounding = _rounded(state.rounding)
        def append():
            return model.apply(
                {**params, "cache": state.cache}, jnp.where(valid, tokens, pad_id),
                decode=True, attention_mask=valid, mutable=["cache", *recorded], rngs=rngs,
                method="states_and_logits" if exposed else None,
                capture_intermediates=False, **positions)

        history, lengths = state.tokens, state.lengths
        original = rebuild_position(model)
        if original is not None and history is not None and lengths is not None:
            slots = lengths[:, None] + jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1
            history = write_cache(history, tokens, jnp.where(valid, slots, -1))
            following = lengths + jnp.sum(valid, axis=1, dtype=jnp.int32)
            crossing = (lengths <= original) & (following > original)
            ordinary = append()

            def rebuild():
                """Recompute the prefix at its new table, Phi-3's intended policy.

                Transformers 5.16.1 slices to one token before clearing KV:
                at the tiny fixture's position 8 its hook sees [[32], [59]],
                no cache, and predicts 2 where its full-prefix forward predicts
                32. Rebuilding the prefix retains the prompt's information.
                https://github.com/huggingface/transformers/issues/49334
                """
                empty = model.apply(params, tokens.shape[0], method='init_cache', mutable=['cache'],
                                    rngs=rngs)[1]['cache']
                answer, updated = model.apply(
                    {**params, 'cache': empty}, history,
                    attention_mask=jnp.arange(history.shape[1])[None, :] < following[:, None],
                    decode=True, mutable=['cache'], rngs=rngs,
                    method='states_and_logits' if exposed else None)
                rows = jnp.arange(tokens.shape[0])[:, None]
                # The same answer `append` gives: states beside the logits where the model exposes them.
                picked = jax.tree.map(lambda leaf: leaf[rows, slots], answer)
                rebuilt = picked, updated

                def replaced(fresh, held):
                    selected = crossing.reshape(-1, *(1,) * (fresh.ndim - 1))
                    return jnp.where(selected, fresh, held)

                return jax.tree.map(replaced, rebuilt, ordinary)

            answer, updated = jax.lax.cond(
                jnp.any(crossing), rebuild, lambda: ordinary)
            lengths = following
        else:
            answer, updated = append()
        states, logits = answer if exposed else (None, answer)
        if blocks is not None:
            states = _context(model, params, updated)
        moved = (None if state.positions is None else
                 state.positions + jnp.sum(valid, axis=1, dtype=state.positions.dtype))
        return (dataclasses.replace(state, cache=updated["cache"], logits=logits[:, -1],
                                    positions=moved, rounding=rounding,
                                    tokens=history, lengths=lengths,
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

    if blocks is not None:
        return DecodeOps(advance, reindex, run, record=blocks[0], draft=blocks[1])
    if not depths or not exposed:
        return DecodeOps(advance, reindex, run)
    return DecodeOps(advance, reindex, run, *_drafting(model, params, pad_id), depths)


def _drafting(model: nn.Module, params: Variables, pad_id: int):
    """`DecodeOps.propose` and `DecodeOps.embed` over the model's prediction depths."""
    def propose(state: DecoderState, hidden: jax.Array, tokens: jax.Array | None,
                embeds: jax.Array | None, valid: jax.Array, positions: jax.Array,
                depth: int, prediction_phase: PredictionPhase) -> tuple[DecoderState, jax.Array, jax.Array]:
        ids = jnp.zeros(valid.shape, jnp.int32) if tokens is None else jnp.where(valid, tokens, pad_id)
        # Multi-axis rotary metadata is not a scalar position, and a depth
        # takes it under its own name.
        placed = ({"rotary_positions": positions} if positions.ndim == 3
                  else {"positions": positions})
        rngs, rounding = _rounded(state.rounding)
        (logits, states), updated = model.apply(
            {**params, "cache": state.cache}, hidden, ids, depth=depth, attention_mask=valid,
            input_embeddings=embeds, decode=True, mutable=["cache"], rngs=rngs,
            prediction_phase=prediction_phase,
            method="mtp_step", capture_intermediates=False, **placed)
        return dataclasses.replace(state, cache=updated["cache"], rounding=rounding), logits, states

    def embed(tokens: jax.Array) -> jax.Array:
        prepared = model.apply(params, tokens, method="token_embeddings")
        assert isinstance(prepared, jax.Array)
        return prepared

    return propose, embed


def _block_drafting(model: nn.Module, params: Variables):
    """`DecodeOps.record` and `DecodeOps.draft` over the model's block drafter."""
    def record(state: DecoderState, context: jax.Array, valid: jax.Array) -> DecoderState:
        _, updated = model.apply({**params, "cache": state.cache}, context, None, valid=valid,
                                 method="draft", mutable=["cache"])
        return dataclasses.replace(state, cache=updated["cache"])

    def draft(state: DecoderState, tokens: jax.Array,
              choose: Callable[[int, jax.Array], jax.Array]) -> None:
        model.apply({**params, "cache": state.cache}, None, tokens, choose=choose,
                    method="draft", mutable=["cache"])

    return record, draft


def _generate(model: nn.Module, params: Variables, inputs: ModelInputs, keys: jax.Array,
              trips: int, budget: jax.Array, pad_id: int, n: int,
              transforms: tuple[LogitsTransform, ...], stopping: decoding.Finishing,
              strategy: Strategy) -> Generation:
    """One fixed compiled loop of `trips` draws; finished rows do not mutate
    their cache state. `budget` is the tokens the request may draw, at most
    `trips`, which the steps read (`StepState.budget`).

    The continuations share one prefill. Parameters remain unmapped, and the
    strategy owns whatever loop the request asked for. Results leave in prompt
    order, each prompt's continuations together.
    """
    batch, width = inputs.tokens.shape
    prompt = inputs.tokens if n == 1 else jnp.repeat(inputs.tokens, n, axis=0)
    if trips == 0:
        empty = jnp.zeros((batch * n, 0), jnp.float32)
        return Generation(prompt, jnp.zeros(batch * n, jnp.int32),
                          jnp.zeros(batch * n, bool), empty, empty)
    drafting = isinstance(strategy, Speculative)
    ops = decode_ops(model, params, pad_id, prediction_depths(model) if drafting else None)
    state, real = prefill_state(model, params, inputs, ops, rounding=rounding_keys(keys))
    start = StepState(
        tokens=jnp.concatenate([inputs.tokens, jnp.zeros((batch, trips), jnp.int32)], axis=1),
        valid=jnp.concatenate([real_slots(inputs.token_fields, (batch, width)),
                               jnp.zeros((batch, trips), bool)], axis=1),
        step=jnp.zeros(batch, jnp.int32), active=real, keys=keys, prompt_width=width,
        budget=jnp.full(batch, budget, jnp.int32))
    drawn: Draws = strategy(state, start, ops, decoding.chain(transforms),
                            decoding.finish(stopping), trips, n)
    return Generation(
        jnp.concatenate([prompt, jnp.where(drawn.valid, drawn.tokens, pad_id)], axis=1),
        jnp.sum(drawn.valid, axis=1, dtype=jnp.int32), drawn.terminated,
        drawn.behavior_log_probs, drawn.raw_log_probs)


def check_inputs(model: nn.Module, ids: np.ndarray, fields: dict[str, np.ndarray], max_new_tokens: int,
                 sampling: Sampling, n: int, *, capacity: int | None = None) -> np.ndarray:
    """Shared host validation; return the real slots (`real_slots`) without placing unused device inputs.

    `capacity` is the cache slots a row holds, which a prompt's real tokens
    and `max_new_tokens` must fit: a server's own capacity, or the model's
    `max_seq_len` where None.
    """
    if isinstance(model, BlockDenoiser):
        raise Lacks("cache", f"{type(model).__name__}'s cache commits whole canvases (BlockDenoiser), not "
                             "a token at a time; generate with BlockGeneration")
    if not isinstance(model, Decoding):
        raise Lacks("cache", f"{type(model).__name__} opens no decode cache (Decoding.init_cache), so it "
                             "cannot continue a prompt token by token")
    if ids.ndim != 2 or min(ids.shape) < 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("inputs must contain non-empty [B, P] integer token ids")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer number of continuations")
    mask = fields.get("attention_mask")
    if mask is not None and (mask.shape != ids.shape or not np.all((mask == 0) | (mask == 1))):
        raise ValueError("attention_mask must be binary [B, P] aligned with token ids")
    valid = real_slots(fields, ids.shape, xp=np)
    if not np.all(valid.any(axis=1)):
        raise ValueError("each prompt must contain at least one valid token")
    cache_len = capacity if capacity is not None or not isinstance(model, Bounded) else model.max_seq_len
    if cache_len is not None and int(valid.sum(axis=1).max()) + max_new_tokens > cache_len:
        raise ValueError("prompt plus max_new_tokens exceeds max_seq_len; raise the cache capacity")
    vocab = model.vocab_size if isinstance(model, Bounded) else None
    if np.any(ids < 0):
        raise ValueError("token ids must be non-negative")
    media = np.asarray(fields.get("image_indices", np.full(ids.shape, -1))) >= 0
    if vocab is not None and np.any((ids >= vocab) & ~media):
        raise ValueError("text token ids must be inside the vocabulary")
    if vocab is not None and (sampling.pad >= vocab or any(token >= vocab for token in sampling.stops)):
        raise ValueError("sampling token ids must be inside the vocabulary")
    return valid


def _validated(model: nn.Module, ids: np.ndarray, fields: dict[str, np.ndarray],
               conditioning: dict[str, jax.Array | np.ndarray], max_new_tokens: int, sampling: Sampling,
               n: int) -> ModelInputs:
    """Host checks shared by every caller; returns device inputs whose validity
    field is present only where a prompt is actually padded."""
    valid = check_inputs(model, ids, fields, max_new_tokens, sampling, n)
    token_fields = with_validity({name: jax.tree.map(jnp.asarray, value) for name, value in fields.items()},
                                 None if valid.all() else jnp.asarray(valid))
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
    given = () if stopping is None else tuple(stopping) if isinstance(stopping, Sequence) else (stopping,)
    return (decoding.components(transforms), decoding.Finishing.of((*given, *sampling.criteria())),
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
        return ("array", _hashed(value))
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


def _hashed(value: np.ndarray | np.generic | jax.Array) -> str:
    """An array's dtype, shape and contents, fingerprinted (`dew._hashing.array_fingerprint`)."""
    digest = hashlib.sha256()
    array_fingerprint(digest, np.asarray(value))
    return digest.hexdigest()


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


def _check(model: nn.Module, params: Variables, inputs: ModelInputs, max_new_tokens: int, trips: int,
           sampling: Sampling, n: int, logits: Transforms | None, stopping: Criteria | None,
           strategy: Strategy | None, prefix: int | Literal["prompt"] | None,
           pooled: bool) -> tuple[ModelInputs, tuple, Components]:
    """This process's validated request, the controls a pool compares, and
    the decoding components.

    The component digest is raised from here with the rest, so a pool agrees
    on the failure before any rank enters a collective. A single process
    never digests: refusing a component only a pool could disagree about
    would cost it nothing. A prefix run's blocks are made here of this
    process's own rows, and a prompt too short for them refused, so a pool
    refuses it together.
    """
    ids = local_rows(inputs.tokens)
    fields = {name: jax.tree.map(local_rows, value) for name, value in inputs.token_fields.items()}
    refuse_prefix(model, prefix, longest=None)
    fields = with_served_blocks(fields, real_slots(fields, ids.shape, xp=np), prefix)
    conditioning = {name: local_rows(value, host=False) for name, value in inputs.conditioning.items()}
    if "params" not in params:
        raise ValueError("generate takes the full variables dict ({'params': ...})")
    _refuse_generation(model)
    if trips < max_new_tokens:
        raise ValueError(f"trips must cover the budget of {max_new_tokens} tokens, got {trips}")
    prepared = _validated(model, ids, fields, conditioning, trips, sampling, n)
    components = resolve(sampling, logits, stopping, strategy)
    controls = (max_new_tokens, trips, n, sampling.pad) + ((_digest(components),) if pooled else ())
    return prepared, controls, components


def _checked(model: nn.Module, params: Variables, inputs: ModelInputs, keys: jax.Array,
             budget: jax.Array, trips: int, pad_id: int, n: int,
             transforms: tuple[LogitsTransform, ...], stopping: decoding.Finishing,
             strategy: Strategy,
             packing: tuple[ProjectionGroup, ...] = ()) -> tuple[checkify.Error, Generation]:
    """`_generate` with its device checks discharged into a returned error.

    An undefined draw has to reach the caller as an exception rather than a
    fabricated token, and a device loop cannot raise. `checkify` carries the
    failure out of the computation; with no check emitted the error is empty
    and throwing it costs nothing.
    """

    # dew.inference imports this module, so its packing is imported here.
    from dew.inference.projections import packed_in_call

    def run(params, inputs, keys, budget, transforms, stopping, strategy):
        # The groups a borrowed tree holds unpacked pack here, once a call (`packed_in_call`).
        return _generate(model, packed_in_call(params, packing), inputs, keys, trips, budget, pad_id, n,
                         transforms, stopping, strategy)

    return checkify.checkify(run, errors=checkify.user_checks)(
        params, inputs, keys, budget, transforms, stopping, strategy)




_DEFAULT_SAMPLING = Sampling()


def generate(model: nn.Module, params: Variables,
             inputs: ModelInputs | ArrayLike | Sequence[Sequence[int]], max_new_tokens: int,
             *, key: int | jax.Array,
             sampling: Sampling = _DEFAULT_SAMPLING, n: int = 1, logits: Transforms | None = None,
             stopping: Criteria | None = None, strategy: Strategy | None = None,
             trips: int | None = None, prefix: int | Literal["prompt"] | None = None) -> Generation:
    """Generate continuations from model inputs, or from token ids for a text-only prompt.

    ``params`` is the full variables dict, ``{'params': ...}``.
    ``ModelInputs.token_fields["attention_mask"]`` marks the real tokens;
    without a mask, every token is real. Every row must contain at least one
    real token. Prefill evaluates the conditioning once, and decoding reuses
    the cache and logical-position state the model keeps. Each cache
    compacts the real input tokens and leaves paused rows unchanged.

    Parameters keep their placement, and decode at the model's compute
    dtype: a call casts a parameter stored wider to it once, before its
    loop (`dew.interop.weights.decoding_storage`), and holds that copy
    beside the caller's weights while it runs (fp32 weights under bf16
    compute: half their bytes again). On a mesh, the rows are split over
    its batch axes and the result keeps that sharding; ``Generation.host()``
    reads a process's own rows back. All cooperating processes must use the
    same input shapes, effective decoding components, padding id and
    continuation count. The decode loops have fixed bounds, and the
    processes agree on any block they skip. Keys fold in the global row
    index and response position, so a pool draws what one process draws for
    the same rows.

    The ``n`` continuations of each prompt share its prefill and come back
    as ``n`` consecutive rows of every array, in prompt order. Continuation
    zero of a prompt draws with that prompt's own key, so ``n=1`` and
    continuation zero of any larger request are the same draw.

    ``logits`` is the whole transform chain, in the order it runs. ``None``
    means the chain ``sampling`` compiles to, and ``()`` runs no transform.
    ``stopping`` adds criteria beside the policy's EOS criterion and never
    replaces it. ``strategy`` replaces the per-row draw loop; ``None`` uses
    ``Sample``. A policy with ``stop`` strings raises ``ValueError``,
    because ``generate`` has no tokenizer to compile them against.

    ``trips``, at least ``max_new_tokens``, scans that many draws, so calls
    of several budgets share one compiled loop (``TextGeneration`` buckets
    them); the budget still bounds what the request draws as far as a
    transform or a search reads it (``ForcedEOS``, ``Beam``), and the
    columns past it come back for the caller to cut.

    ``prefix`` serves a prefix run (``LMObjective``'s): each prompt that
    carries no blocks of its own reads its first ``prefix`` real tokens, or
    with ``"prompt"`` all of them, as one block (``served_blocks``). They are
    made of each process's own rows inside the request's agreed check, and a
    prompt with fewer real tokens than an int ``prefix`` is refused there, by
    every process of a pool together.
    """
    if sampling.stop:
        raise ValueError("stop strings compile against a tokenizer's vocabulary, which generate does not "
                         "have; generate through a TextGeneration with a processor, or pass "
                         "stopping=(decoding.stop_strings(tokenizer, strings, vocab_size),)")
    scanned = max_new_tokens if trips is None else trips
    request, components = Request.prepare(
        inputs, key, mesh_of(params),
        lambda canonical, pooled: _check(model, params, canonical, max_new_tokens, scanned, sampling, n,
                                         logits, stopping, strategy, prefix, pooled),
        phase="generation")
    plan = request.plan
    capacity = model.max_seq_len if isinstance(model, Bounded) else None
    if isinstance(model, Layered) and capacity is not None:
        refuse_unassigned(model.kv_cache, plan.count, capacity)
    from dew.inference.projections import pending

    compiled = program(_checked, model, static_argnames=("model", "trips", "pad_id", "n", "packing"),
                       in_shardings=(None, plan.sharding, plan.sharding, None, None, None, None),
                       out_shardings=(None, plan.sharding))
    # A tree a loader owns arrives packed; one borrowed from a live objective packs in the call.
    failure, output = compiled(model, decoding_storage(model, params), plan.place(request.padded()),
                               plan.keys(request.key), jnp.int32(max_new_tokens), scanned, sampling.pad, n,
                               *components, pending(model, params))
    # The error's flags are the one read a request waits on.
    jax.device_get(failure).throw()
    return replace(output, rows=plan.rows * n)
