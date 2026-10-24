"""Logit transforms and stopping criteria for the decode loop.

Three concepts extend decoding. A `LogitsTransform` is a pure callable from
the step state and `[rows, vocab]` logits to new logits. A `Stopping` is a
pure callable from the step state and the tokens just drawn to a per-row
finished flag; criteria combine with OR after every committed token. A
`Strategy` in `dew.sampling.strategies` runs the device loop.

Transforms and criteria read `StepState`, which holds the token history and
nothing about the model: no parameters and no cache. Every built-in here is a
pytree, so a configuration holding arrays crosses `jax.jit` as data instead of
entering a compilation cache key, and so do its numbers where they fix no
shape (`as_pytree`): another temperature, penalty or length runs the same
program. A plain function works as well, and `jax.tree_util.Partial` can hold
array configuration for one.

The numerical reference is Transformers 5.16.1
`generation/logits_process.py` and `generation/stopping_criteria.py`, with two
differences that follow from Dew's decode loop. Each row reads its own
unpadded history rather than the batch's padded width, and frequency and
presence penalties follow vLLM's formula
(`model_executor/layers/utils.py`), which Transformers does not implement.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Callable, Sequence
from typing import NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax

from dew._distributions import wellformed
from dew._records import finite
from dew.batch import token_ids, valid_order
from dew.nn import categorical
from dew.nn.scatter import DROPPED
from dew.sampling.vocabulary import (
    Referencing,
    Tokenizing,
    Vocabulary,
    matching_mode,
    vocabulary_of,
    vocabulary_pieces,
)

FILTER = -jnp.inf
"""The score a removed token keeps, as `logits_process.py`'s filter value."""


@struct.dataclass
class StepState:
    """What a transform or a criterion sees at one decode step.

    `tokens` is the fixed-capacity buffer of the prompt followed by the draw
    slots, `[rows, prompt_width + budget]`, and `valid` marks the slots that
    hold a real token. A row's history is therefore its own, whatever padding
    the prompt batch needed. `step` counts the tokens the row has committed,
    `active` marks the rows still generating, and `keys` holds one PRNG key
    per row. `budget` is the tokens each row's request may draw, `[rows]`,
    which can be fewer than the draw slots: a task scans a bucket of trips
    past it so budgets share one compiled loop, and a server's slots hold
    its whole capacity. None leaves it the draw slots.
    """

    tokens: jax.Array
    valid: jax.Array
    step: jax.Array
    active: jax.Array
    keys: jax.Array
    prompt_width: int = struct.field(pytree_node=False, default=0)
    budget: jax.Array | None = None

    @property
    def width(self) -> int:
        """Slots in the buffer, prompt plus budget."""
        return self.tokens.shape[1]

    @property
    def rows(self) -> int:
        return self.tokens.shape[0]

    def budgets(self) -> jax.Array:
        """Return the tokens each row's request may draw (`budget`), `[rows]`."""
        if self.budget is not None:
            return self.budget
        return jnp.full(self.rows, self.width - self.prompt_width, jnp.int32)

    def total(self) -> jax.Array:
        """Return the number of real tokens each row holds, prompt and generated together."""
        return jnp.sum(self.valid, axis=-1, dtype=jnp.int32)

    def history(self) -> tuple[jax.Array, jax.Array]:
        """Return each row's real tokens left-aligned, and how many there are.

        Prompts pad wherever their batch needed it, so a transform that reads
        order (n-grams, biased sequences, stop strings) needs the row's own
        tokens without holes. Padding slots hold -1, which no token id equals.
        """
        return _compact(self.tokens, self.valid)

    def prompt_history(self) -> tuple[jax.Array, jax.Array]:
        """Return the prompt region's real tokens left-aligned, and how many there are."""
        width = self.prompt_width
        return _compact(self.tokens[:, :width], self.valid[:, :width])

    def commit(self, tokens: jax.Array, drawn: jax.Array) -> StepState:
        """Return the state after the `drawn` rows append `tokens` at their next slot."""
        rows = jnp.arange(self.rows)
        slot = self.prompt_width + self.step
        slot = jnp.where(slot < self.tokens.shape[1], slot, DROPPED)
        return dataclasses.replace(
            self,
            tokens=self.tokens.at[rows, slot].set(jnp.where(drawn, tokens, 0), mode="drop"),
            valid=self.valid.at[rows, slot].set(drawn, mode="drop"),
            step=self.step + drawn.astype(jnp.int32))


def _compact(tokens: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array]:
    order, lengths = valid_order(valid)
    kept = jnp.arange(tokens.shape[1])[None, :] < lengths[:, None]
    return jnp.where(kept, jnp.take_along_axis(tokens, order, axis=-1), -1), lengths


def _suffix(tokens: jax.Array, lengths: jax.Array, width: int) -> jax.Array:
    """The last `width` tokens of each left-aligned row, right aligned.

    Slots before a row's first token hold -1, so a comparison against them
    never matches a token id.
    """
    if width <= 0:
        return jnp.zeros((tokens.shape[0], 0), tokens.dtype)
    index = lengths[:, None] - width + jnp.arange(width)[None, :]
    taken = jnp.take_along_axis(tokens, jnp.clip(index, 0, tokens.shape[1] - 1), axis=-1)
    return jnp.where((index >= 0) & (index < lengths[:, None]), taken, -1)


def _windows(tokens: jax.Array, size: int) -> jax.Array:
    """Every length-`size` window of each row, `[rows, starts, size]`."""
    starts = tokens.shape[1] - size + 1
    index = jnp.arange(starts)[:, None] + jnp.arange(size)[None, :]
    return tokens[:, index]


def _present(tokens: jax.Array, valid: jax.Array, vocab: int) -> jax.Array:
    """Which vocabulary entries each row's valid tokens contain."""
    rows = jnp.arange(tokens.shape[0])[:, None]
    slots = jnp.where(valid & (tokens < vocab), tokens, DROPPED)
    return jnp.zeros((tokens.shape[0], vocab), bool).at[rows, slots].set(True, mode="drop")


def _counts(tokens: jax.Array, valid: jax.Array, vocab: int) -> jax.Array:
    """How often each vocabulary entry appears among a row's valid tokens."""
    rows = jnp.arange(tokens.shape[0])[:, None]
    slots = jnp.where(valid & (tokens < vocab), tokens, DROPPED)
    return jnp.zeros((tokens.shape[0], vocab), jnp.float32).at[rows, slots].add(1.0, mode="drop")


def id_row(name: str, value: int | Sequence[int]) -> tuple[int, ...]:
    """`value`, one token id or a sequence of them (a range, an array's row), as a tuple, read by the
    one law of token ids (`dew.batch.token_id`)."""
    row = (value,) if isinstance(value, int | np.integer) else value
    return tuple(token_ids(row if isinstance(row, np.ndarray) else tuple(row), where=name))


def positive(name: str, value: float) -> float:
    rate = finite(value, name)
    if rate <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return rate


def unit_interval(name: str, value: float) -> float:
    rate = finite(value, name)
    if not 0 <= rate <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return rate


def positive_size(name: str, value: object) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@runtime_checkable
class LogitsTransform(Protocol):
    """A pure rewrite of `[rows, vocab]` scores, applied before the draw."""

    def __call__(self, state: StepState, logits: jax.Array, /) -> jax.Array: ...


@runtime_checkable
class Stopping(Protocol):
    """A pure per-row finish test over the tokens a step just drew.

    A criterion that cuts a row short at a length rather than ending it, as
    `MaxNewTokens` and `MaxLength` do, is a `Truncates`: a row it finishes
    is not terminated (`Finish`)."""

    def __call__(self, state: StepState, tokens: jax.Array, /) -> jax.Array: ...


class Truncates:
    """Marks a stopping criterion that cuts a row short at a length rather than ending it."""


@struct.dataclass
class Greedy:
    """The argmax as a distribution: zero on the best token, `-inf` elsewhere.

    `Sampling(temperature=0)` compiles to this, so the draw stays the
    deterministic argmax with a behaviour log probability of exactly zero; a
    terminal Greedy lets the sampler take the argmax without building the
    point mass. Transforms before it still shape the argmax, as in greedy
    search. A row without a distribution (all removed, or NaN or `+inf`)
    passes through for the draw to refuse, rather than becoming a confident
    token.
    """

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        best = jnp.argmax(logits, axis=-1)[:, None]
        point = jnp.where(jnp.arange(logits.shape[-1])[None, :] == best, 0.0, FILTER)
        return jnp.where(wellformed(logits)[:, None], point, logits)


class _Numerical:
    """A built-in whose numbers fix no shape. `as_pytree` passes it as
    `numerical()`, its `apply` with those numbers bound as data, so another
    value runs the program the first compiled; the built-in applies alike."""

    def numerical(self) -> jax.tree_util.Partial:
        raise NotImplementedError

    def __call__(self, state: StepState, values: jax.Array) -> jax.Array:
        return self.numerical()(state, values)


@struct.dataclass
class Temperature(_Numerical):
    """Divides the logits by `value`, as `TemperatureLogitsWarper`."""

    value: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        positive("temperature", self.value)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(Temperature.apply, self.value)

    @staticmethod
    def apply(value: float, state: StepState, logits: jax.Array) -> jax.Array:
        return logits / value


@struct.dataclass
class TopK:
    """Keeps the `k` highest scores, as `TopKLogitsWarper`."""

    k: int = struct.field(pytree_node=False, default=1)

    def __post_init__(self) -> None:
        positive_size("top_k", self.k)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        keep = min(self.k, logits.shape[-1])
        cutoff = lax.top_k(logits, keep)[0][..., -1:]
        return jnp.where(logits < cutoff, FILTER, logits)


@struct.dataclass
class TopP(_Numerical):
    """Nucleus filtering, as `TopPLogitsWarper`.

    The ascending tail holding cumulative mass at most `1 - p` is removed and
    the best token always survives.
    """

    p: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        unit_interval("top_p", self.p)

    def numerical(self) -> jax.tree_util.Partial:
        # The tail's mass is subtracted on the host, as the reference does, before it is traced.
        return jax.tree_util.Partial(TopP.apply, 1.0 - self.p)

    @staticmethod
    def apply(excluded_mass: float, state: StepState, logits: jax.Array) -> jax.Array:
        order = jnp.argsort(logits, axis=-1, stable=True)
        ascending = jnp.take_along_axis(logits, order, axis=-1)
        tail = jnp.cumsum(jax.nn.softmax(ascending), axis=-1) <= excluded_mass
        tail = tail.at[:, -1].set(False)
        rows = jnp.arange(logits.shape[0])[:, None]
        removed = jnp.zeros_like(tail).at[rows, order].set(tail)
        return jnp.where(removed, FILTER, logits)


@struct.dataclass
class MinP(_Numerical):
    """Relative filtering at `p` times the top probability, as `MinPLogitsWarper`."""

    p: float = struct.field(pytree_node=False, default=0.0)

    def __post_init__(self) -> None:
        unit_interval("min_p", self.p)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(MinP.apply, self.p)

    @staticmethod
    def apply(p: float, state: StepState, logits: jax.Array) -> jax.Array:
        probabilities = jax.nn.softmax(logits)
        cutoff = jnp.max(probabilities, axis=-1, keepdims=True) * p
        return jnp.where(probabilities < cutoff, FILTER, logits)


@struct.dataclass
class Typical(_Numerical):
    """Locally typical filtering, as `TypicalLogitsWarper`."""

    mass: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        unit_interval("typical_p", self.mass)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(Typical.apply, self.mass)

    @staticmethod
    def apply(mass: float, state: StepState, logits: jax.Array) -> jax.Array:
        normalized = jax.nn.log_softmax(logits)
        entropy = categorical.entropy(logits)[..., None]
        shifted = jnp.abs(-normalized - entropy)
        order = jnp.argsort(shifted, axis=-1, stable=True)
        sorted_shift = jnp.take_along_axis(shifted, order, axis=-1)
        cumulative = jnp.cumsum(jax.nn.softmax(jnp.take_along_axis(logits, order, axis=-1)), axis=-1)
        last = jnp.clip(jnp.sum(cumulative < mass, axis=-1), 0, logits.shape[-1] - 1)
        removed = sorted_shift > jnp.take_along_axis(sorted_shift, last[:, None], axis=-1)
        removed = removed.at[:, 0].set(False)
        rows = jnp.arange(logits.shape[0])[:, None]
        scattered = jnp.zeros_like(removed).at[rows, order].set(removed)
        return jnp.where(scattered, FILTER, logits)


@struct.dataclass
class EpsilonCutoff(_Numerical):
    """Removes tokens below an absolute probability, as `EpsilonLogitsWarper`."""

    epsilon: float = struct.field(pytree_node=False, default=0.0)

    def __post_init__(self) -> None:
        unit_interval("epsilon_cutoff", self.epsilon)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(EpsilonCutoff.apply, self.epsilon)

    @staticmethod
    def apply(epsilon: float, state: StepState, logits: jax.Array) -> jax.Array:
        best = jnp.max(logits, axis=-1, keepdims=True)
        removed = (jax.nn.softmax(logits) < epsilon) & (logits < best)
        return jnp.where(removed, FILTER, logits)


@struct.dataclass
class EtaCutoff(_Numerical):
    """Entropy-scaled cutoff, as `EtaLogitsWarper`, which takes the square
    root of its float32 epsilon on the device."""

    epsilon: float = struct.field(pytree_node=False, default=0.0)

    def __post_init__(self) -> None:
        unit_interval("eta_cutoff", self.epsilon)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(EtaCutoff.apply, self.epsilon)

    @staticmethod
    def apply(epsilon: float, state: StepState, logits: jax.Array) -> jax.Array:
        probabilities = jnp.exp(jax.nn.log_softmax(logits))
        entropy = categorical.entropy(logits)
        cutoff = jnp.asarray(epsilon, jnp.float32)
        eta = jnp.minimum(cutoff, jnp.sqrt(cutoff) * jnp.exp(-entropy))[:, None]
        best = jnp.max(logits, axis=-1, keepdims=True)
        return jnp.where((probabilities < eta) & (logits < best), FILTER, logits)


@struct.dataclass
class TopH(_Numerical):
    """Entropy-budget filtering, as `TopHLogitsWarper`.

    Tokens enter in probability order while the cumulative entropy of the
    truncated head stays within `fraction` of its total entropy, and the best
    token always enters, out of the reference's fixed head of 100.

    The two entropies are computed the way the reference computes them, and
    they are not the same expression. The budget is
    `torch.distributions.Categorical.entropy`, which clamps the log
    probabilities to the dtype's minimum so a removed token contributes
    nothing. The running sum is the reference's own `-p * log(p)`, whose
    removed tokens are NaN, and a NaN ends the selection because every
    comparison against it is false. Substituting one for the other keeps a
    token the reference drops.
    """

    fraction: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        if unit_interval("top_h", self.fraction) <= 0:
            raise ValueError("top_h must be finite and above zero up to one")

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(TopH.apply, self.fraction)

    @staticmethod
    def apply(fraction: float, state: StepState, logits: jax.Array) -> jax.Array:
        head, index = lax.top_k(logits, min(100, logits.shape[-1]))
        normalized = head - jax.scipy.special.logsumexp(head, axis=-1, keepdims=True)
        probabilities = jax.nn.softmax(normalized)
        budget = -jnp.sum(jnp.maximum(normalized, jnp.finfo(head.dtype).min) * probabilities,
                          axis=-1, keepdims=True) * fraction
        selected = jnp.cumsum(-probabilities * jnp.log(probabilities), axis=-1) <= budget
        selected = selected.at[:, 0].set(True)
        rows = jnp.arange(logits.shape[0])[:, None]
        keep = jnp.zeros(logits.shape, bool).at[rows, index].set(selected)
        return jnp.where(keep, logits, FILTER)


@struct.dataclass
class Renormalize:
    """Replaces scores by their log softmax, as `LogitNormalization`."""

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return jax.nn.log_softmax(logits)


@struct.dataclass
class RemoveInvalidValues:
    """Maps NaN to zero and infinities to the float range, as `InfNanRemoveLogitsProcessor`.

    Nothing else in the chain repairs a broken distribution; an undefined
    draw raises instead. Add this transform if you want one sanitized.
    """

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return jnp.nan_to_num(logits, nan=0.0)


@struct.dataclass
class RepetitionPenalty(_Numerical):
    """Divides the positive scores of seen tokens by the penalty and multiplies negative ones.

    The history is the row's valid prompt and drawn tokens, as
    `RepetitionPenaltyLogitsProcessor` reads the whole `input_ids`.
    """

    penalty: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        positive("repetition_penalty", self.penalty)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(RepetitionPenalty.apply, self.penalty)

    @staticmethod
    def apply(penalty: float, state: StepState, logits: jax.Array) -> jax.Array:
        seen = _present(state.tokens, state.valid, logits.shape[-1])
        return jnp.where(seen, _scaled(logits, penalty), logits)


@struct.dataclass
class PromptRepetitionPenalty(_Numerical):
    """Raises the scores of prompt tokens, as `EncoderRepetitionPenaltyLogitsProcessor`.

    The reference inverts its argument, so a penalty above one rewards
    repeating the prompt. A decoder-only prompt is the encoder input here.
    """

    penalty: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        positive("encoder_repetition_penalty", self.penalty)

    def numerical(self) -> jax.tree_util.Partial:
        # The reference's inversion, on the host.
        return jax.tree_util.Partial(PromptRepetitionPenalty.apply, 1.0 / self.penalty)

    @staticmethod
    def apply(inverted: float, state: StepState, logits: jax.Array) -> jax.Array:
        width = state.prompt_width
        seen = _present(state.tokens[:, :width], state.valid[:, :width], logits.shape[-1])
        return jnp.where(seen, _scaled(logits, inverted), logits)


def _scaled(logits: jax.Array, penalty: float) -> jax.Array:
    return jnp.where(logits < 0, logits * penalty, logits / penalty)


@struct.dataclass
class FrequencyPenalty(_Numerical):
    """Subtracts `penalty` times each token's count among the drawn tokens.

    This is vLLM's formula, `logits -= frequency_penalties * output_bin_counts`
    (`model_executor/layers/utils.py`), which is also OpenAI's
    `frequency_penalty`. It counts generated tokens only, not the prompt.
    """

    penalty: float = struct.field(pytree_node=False, default=0.0)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(FrequencyPenalty.apply, self.penalty)

    @staticmethod
    def apply(penalty: float, state: StepState, logits: jax.Array) -> jax.Array:
        width = state.prompt_width
        counts = _counts(state.tokens[:, width:], state.valid[:, width:], logits.shape[-1])
        return logits - penalty * counts


@struct.dataclass
class PresencePenalty(_Numerical):
    """Subtracts `penalty` from every token already drawn.

    This is vLLM's `logits -= presence_penalties * output_mask`, which is also
    OpenAI's `presence_penalty`. It reads generated tokens only, not the prompt.
    """

    penalty: float = struct.field(pytree_node=False, default=0.0)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(PresencePenalty.apply, self.penalty)

    @staticmethod
    def apply(penalty: float, state: StepState, logits: jax.Array) -> jax.Array:
        width = state.prompt_width
        seen = _present(state.tokens[:, width:], state.valid[:, width:], logits.shape[-1])
        return logits - penalty * seen.astype(logits.dtype)


@struct.dataclass
class NoRepeatNGram:
    """Bans tokens that would repeat an n-gram of the row's own history.

    This is the tensorised form of `NoRepeatNGramLogitsProcessor`: the current
    suffix is matched against every window, and a matching window bans the
    token that followed it. The window starting at the suffix itself needs one token more
    than the row has, so a suffix never bans its own successor.
    """

    size: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self) -> None:
        positive_size("no_repeat_ngram_size", self.size)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        tokens, lengths = state.history()
        return _ngram_ban(logits, tokens, lengths, _suffix(tokens, lengths, self.size - 1),
                          lengths, self.size)


@struct.dataclass
class PromptNoRepeatNGram:
    """Bans tokens that would repeat an n-gram of the prompt.

    `EncoderNoRepeatNGramLogitsProcessor` builds its table from the encoder
    input and matches it against the decoder's suffix. The prompt is the
    encoder input of a decoder-only model.
    """

    size: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self) -> None:
        positive_size("encoder_no_repeat_ngram_size", self.size)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        tokens, lengths = state.history()
        prompt, prompt_lengths = state.prompt_history()
        return _ngram_ban(logits, prompt, prompt_lengths, _suffix(tokens, lengths, self.size - 1),
                          lengths, self.size)


def _ngram_ban(logits: jax.Array, table: jax.Array, table_lengths: jax.Array,
               prefix: jax.Array, lengths: jax.Array, size: int) -> jax.Array:
    """Ban the successor of every `table` window whose start matches `prefix`."""
    windows = _windows(table, size)
    starts = jnp.arange(windows.shape[1])[None, :]
    inside = starts <= (table_lengths[:, None] - size)
    matched = jnp.all(windows[..., :-1] == prefix[:, None, :], axis=-1) & inside
    matched = matched & (lengths >= size - 1)[:, None]
    rows = jnp.arange(logits.shape[0])[:, None]
    slots = jnp.where(matched & (windows[..., -1] < logits.shape[-1]), windows[..., -1], DROPPED)
    banned = jnp.zeros(logits.shape, bool).at[rows, slots].set(True, mode="drop")
    return jnp.where(banned, FILTER, logits)


@struct.dataclass
class SequenceBias:
    """Adds a bias to the token that would complete each biased sequence.

    This is `SequenceBiasLogitsProcessor` as a table: `sequences` is `[count, width]`
    right-aligned token ids, `lengths` their real lengths and `bias` the value
    added to the last id when the row's suffix matches the preceding ones. A
    sequence longer than the row's history is skipped, as the reference skips
    one longer than the context.
    """

    sequences: jax.Array
    lengths: jax.Array
    bias: jax.Array

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        tokens, lengths = state.history()
        width = self.sequences.shape[1]
        prefix = self.sequences[:, :-1]
        needed = self.lengths - 1
        held = jnp.arange(width - 1)[None, :] >= (width - 1 - needed[:, None])
        suffix = _suffix(tokens, lengths, width - 1)
        matched = jnp.all((suffix[:, None, :] == prefix[None, :, :]) | ~held[None, :, :], axis=-1)
        matched = matched & (self.lengths[None, :] <= lengths[:, None])
        added = jnp.where(matched, self.bias[None, :], 0.0)
        rows = jnp.arange(logits.shape[0])[:, None]
        last = jnp.broadcast_to(self.sequences[None, :, -1], added.shape)
        last = jnp.where(last < logits.shape[-1], last, DROPPED)
        return logits + jnp.zeros(logits.shape, jnp.float32).at[rows, last].add(added, mode="drop")


def sequence_bias(entries: Sequence[tuple[Sequence[int], float]]) -> SequenceBias:
    """A `SequenceBias` table from `(token ids, bias)` pairs."""
    pairs = [(tuple(ids), float(value)) for ids, value in entries]
    for ids, _ in pairs:
        id_row("sequence_bias", ids)
    width = max((len(ids) for ids, _ in pairs), default=1)
    table = np.zeros((len(pairs), width), np.int32)
    for row, (ids, _) in enumerate(pairs):
        table[row, width - len(ids):] = ids
    return SequenceBias(jnp.asarray(table),
                        jnp.asarray([len(ids) for ids, _ in pairs], jnp.int32),
                        jnp.asarray([value for _, value in pairs], jnp.float32))


def bad_words(ids: Sequence[Sequence[int]], eos_token_ids: int | Sequence[int] | None = None) -> SequenceBias:
    """A `-inf` `SequenceBias` over forbidden sequences, as `NoBadWordsLogitsProcessor`.

    Single-token sequences that name an EOS id are dropped, as the reference
    drops them, so banning bad words cannot ban termination.
    """
    stops = () if eos_token_ids is None else id_row("eos_token_ids", eos_token_ids)
    kept = [tuple(word) for word in ids if not (len(word) == 1 and word[0] in stops)]
    return sequence_bias([(word, -math.inf) for word in kept])


@struct.dataclass
class SuppressTokens:
    """Removes a fixed set of tokens, as `SuppressTokensLogitsProcessor`."""

    tokens: jax.Array

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return jnp.where(_membership(self.tokens, logits.shape[-1]), FILTER, logits)


@struct.dataclass
class BeginSuppressTokens(_Numerical):
    """Removes tokens at one generated position, as `SuppressTokensAtBeginLogitsProcessor`.

    The reference suppresses the first drawn token, or the one after it where
    `forced_bos` (`ForcedBOS`) forces the first: transformers moves the begin
    index past the forced token only for a one-token prompt
    (generation/utils.py:1278-1282, 5.16.1), the rows where `ForcedBOS` fires.
    Each row decides by its own prompt, so rows of mixed prompt lengths
    suppress each at its own position.
    """

    tokens: jax.Array
    forced_bos: bool = struct.field(pytree_node=False, default=False)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(BeginSuppressTokens.apply, self.tokens, self.forced_bos)

    @staticmethod
    def apply(tokens: jax.Array, forced_bos: bool, state: StepState, logits: jax.Array) -> jax.Array:
        # A row's prompt is its real tokens less the ones it has drawn.
        offset = (forced_bos & (state.total() - state.step == 1)).astype(state.step.dtype)
        removed = _membership(tokens, logits.shape[-1])[None, :] & (state.step == offset)[:, None]
        return jnp.where(removed, FILTER, logits)


def _membership(tokens: jax.Array, vocab: int) -> jax.Array:
    return jnp.zeros(vocab, bool).at[tokens].set(True, mode="drop")


@struct.dataclass
class ForcedBOS(_Numerical):
    """Forces one token as the first of the whole sequence, as `ForcedBOSTokenLogitsProcessor`."""

    token: int = struct.field(pytree_node=False, default=0)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(ForcedBOS.apply, self.token)

    @staticmethod
    def apply(token: int, state: StepState, logits: jax.Array) -> jax.Array:
        return _forced(logits, token, state.total() == 1)


@struct.dataclass
class ForcedEOS(_Numerical):
    """Forces EOS one step before the end, as `ForcedEOSTokenLogitsProcessor`.

    `eos` may name several ids, all of which the forced step allows, as the
    reference allows every id its tensor holds. `max_length` counts prompt and
    generated tokens together. Left as None, the end is the request's own
    token budget (`StepState.budgets`), so a caller that changes the budget
    per call forces EOS at the new end rather than at a length the task was
    built with, and a loop that scans past the budget forces it inside it.
    """

    eos: jax.Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.int32))
    max_length: int | None = struct.field(pytree_node=False, default=None)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(ForcedEOS.apply, self.eos, self.max_length)

    @staticmethod
    def apply(eos: jax.Array, max_length: int | None, state: StepState, logits: jax.Array) -> jax.Array:
        rows = (state.step == state.budgets() - 1 if max_length is None
                else state.total() == max_length - 1)
        only = jnp.where(_membership(eos, logits.shape[-1])[None, :], 0.0, FILTER)
        return jnp.where(rows[:, None], only, logits)


def _forced(logits: jax.Array, token: int, rows: jax.Array) -> jax.Array:
    only = jnp.where(jnp.arange(logits.shape[-1])[None, :] == token, 0.0, FILTER)
    return jnp.where(rows[:, None], only, logits)


@struct.dataclass
class MinLength(_Numerical):
    """Suppresses EOS until the whole sequence reaches `length`, as `MinLengthLogitsProcessor`."""

    length: int = struct.field(pytree_node=False, default=0)
    eos: jax.Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.int32))

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(MinLength.apply, self.length, self.eos)

    @staticmethod
    def apply(length: int, eos: jax.Array, state: StepState, logits: jax.Array) -> jax.Array:
        return _suppress_eos(logits, eos, state.total() < length)


@struct.dataclass
class MinNewTokens(_Numerical):
    """Suppresses EOS until `count` tokens are drawn, as `MinNewTokensLengthLogitsProcessor`."""

    count: int = struct.field(pytree_node=False, default=0)
    eos: jax.Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.int32))

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(MinNewTokens.apply, self.count, self.eos)

    @staticmethod
    def apply(count: int, eos: jax.Array, state: StepState, logits: jax.Array) -> jax.Array:
        return _suppress_eos(logits, eos, state.step < count)


def _suppress_eos(logits: jax.Array, eos: jax.Array, rows: jax.Array) -> jax.Array:
    return jnp.where(rows[:, None] & _membership(eos, logits.shape[-1])[None, :], FILTER, logits)


@struct.dataclass
class ExponentialDecayLengthPenalty(_Numerical):
    """Raises the EOS score after `start` drawn tokens, as `ExponentialDecayLengthPenalty`.

    The reference measures from `start_index + prompt_width`, which is the
    generated count used here, and adds `|score| * (factor ** index - 1)` so a
    negative score also rises. A removed EOS (-inf, a grammar's mask) stays
    removed.
    """

    start: int = struct.field(pytree_node=False, default=0)
    factor: float = struct.field(pytree_node=False, default=1.0)
    eos: jax.Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.int32))

    def __post_init__(self) -> None:
        positive("exponential_decay_length_penalty factor", self.factor)

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(ExponentialDecayLengthPenalty.apply, self.start, self.factor, self.eos)

    @staticmethod
    def apply(start: int, factor: float, eos: jax.Array, state: StepState, logits: jax.Array) -> jax.Array:
        index = jnp.maximum(state.step - start, 0)
        growth = jnp.where(state.step > start, factor ** index.astype(jnp.float32) - 1.0, 0.0)
        selected = _membership(eos, logits.shape[-1])[None, :]
        return logits + jnp.where(selected & jnp.isfinite(logits), jnp.abs(logits) * growth[:, None], 0.0)


@struct.dataclass
class EndOfSequence:
    """Finishes a row that drew one of the EOS ids, as `EosTokenCriteria`."""

    eos: jax.Array

    def __call__(self, state: StepState, tokens: jax.Array) -> jax.Array:
        return jnp.isin(tokens, self.eos)


@struct.dataclass
class MaxNewTokens(_Numerical, Truncates):
    """Finishes a row once it has drawn `count` tokens, which truncates it."""

    count: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ValueError("max new tokens must be a non-negative integer")

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(MaxNewTokens.apply, self.count)

    @staticmethod
    def apply(count: int, state: StepState, tokens: jax.Array) -> jax.Array:
        return state.step >= count


@struct.dataclass
class MaxLength(_Numerical, Truncates):
    """Finishes a row once prompt and generated tokens reach `length`, as `MaxLengthCriteria`, which
    truncates it."""

    length: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self) -> None:
        if type(self.length) is not int or self.length < 0:
            raise ValueError("max length must be a non-negative integer")

    def numerical(self) -> jax.tree_util.Partial:
        return jax.tree_util.Partial(MaxLength.apply, self.length)

    @staticmethod
    def apply(length: int, state: StepState, tokens: jax.Array) -> jax.Array:
        return state.total() >= length


@struct.dataclass
class StopStrings:
    """Finishes a row whose text ends with one of the compiled stop strings.

    The tables come from `stop_strings`, which reads the tokenizer once. The
    device check is `StopStringCriteria`'s: it goes through the row's tokens
    backwards, requires the last token to overlap the end of a stop string,
    and keeps matching earlier tokens against the positions they can occupy. A match
    counts only when the string touches the final token, so a string produced
    earlier does not stop the row later.
    """

    table: jax.Array
    targets: jax.Array
    positions: int = struct.field(pytree_node=False, default=1)
    ends: int = struct.field(pytree_node=False, default=1)
    span: int = struct.field(pytree_node=False, default=1)

    def __call__(self, state: StepState, tokens: jax.Array) -> jax.Array:
        history, lengths = state.history()
        flipped = jnp.flip(_suffix(history, lengths, self.span), axis=1)
        rows, span = flipped.shape
        strings = self.targets.shape[0]
        embedded = jnp.take(self.table, jnp.where(flipped < 0, self.table.shape[0] - 1, flipped), axis=0)
        valid_positions = embedded[:, 1:, :self.positions * strings].reshape(
            (rows, span - 1, strings, self.positions))
        end_lengths = embedded[:, :1, self.positions * strings:-1].reshape(
            (rows, 1, strings, self.ends))
        widths = jnp.broadcast_to(embedded[:, 1:, None, -1:], (rows, span - 1, strings, self.ends))
        cumulative = jnp.cumsum(jnp.concatenate([end_lengths, widths], axis=1), axis=1)
        later = jnp.any(cumulative[:, :-1, :, None, :] == valid_positions[..., None], axis=-2)
        matched = jnp.concatenate([end_lengths > 0, later], axis=1)
        reached = jnp.where(jnp.cumsum((~matched).astype(jnp.int32), axis=1) == 0, cumulative, 0)
        return jnp.any(jnp.max(reached, axis=(1, 3)) >= self.targets[None, :], axis=-1)


def stop_strings(tokenizer: Vocabulary | Referencing | Tokenizing, strings: str | Sequence[str],
                 vocab_size: int | None = None) -> StopStrings:
    """Compile a tokenizer's vocabulary against `strings` into a `StopStrings`.

    The tables record, once on the host, where each token's piece can sit
    inside a stop string and how many of its trailing units the piece's start
    covers, so the criterion never decodes. A byte-level or byte-fallback
    vocabulary matches over UTF-8 bytes, so a code point two tokens split
    still ends a row. `vocab_size` sizes the table for a model's padded head.
    """
    wanted = (strings,) if isinstance(strings, str) else tuple(strings)
    if not wanted or any(not isinstance(value, str) or not value for value in wanted):
        raise ValueError("stop_strings needs non-empty strings")
    source = vocabulary_of(tokenizer, "stop_strings")
    mode = matching_mode(source)
    pieces, ids = vocabulary_pieces(source, mode)
    targets = [value.encode("utf-8") if mode is not None else value for value in wanted]
    width = max(len(ids) + 1, 1 if vocab_size is None else vocab_size + 1)
    return _stop_string_tables(pieces, ids, targets, width)


def _stop_string_tables(pieces: Sequence[str | bytes], ids: Sequence[int],
                        strings: Sequence[str | bytes], rows: int) -> StopStrings:
    """`StopStringCriteria._stop_string_create_embedding_vec` over the pieces."""
    valid: list[dict[int, list[int]]] = []
    overlaps: list[dict[int, list[int]]] = []
    for target in strings:
        backwards = target[::-1]
        inside: dict[int, list[int]] = {}
        ending: dict[int, list[int]] = {}
        for piece, index in zip(pieces, ids, strict=True):
            reversed_piece = piece[::-1]
            for start in range(1 - len(piece), len(target)):
                if start < 0:
                    part, position = reversed_piece[-start:], 0
                else:
                    part, position = reversed_piece, start
                # Pieces and targets are both bytes or both text (`stop_strings`).
                window = backwards[position:position + len(part)]
                if part[:len(window)] == window:
                    if position == 0:
                        ending.setdefault(index, []).append(min(len(part), len(target)))
                    else:
                        inside.setdefault(index, []).append(position)
        valid.append(inside)
        overlaps.append(ending)
    if not any(overlaps):
        raise ValueError("no token in the vocabulary can end any of the stop strings")
    positions = max((len(entry) for row in valid for entry in row.values()), default=1)
    ends = max(len(entry) for row in overlaps for entry in row.values())
    width = len(strings) * (positions + ends) + 1
    table = np.full((max(rows, max(ids) + 2), width), -1, np.int32)
    for order, (inside, ending) in enumerate(zip(valid, overlaps, strict=True)):
        for index, entries in inside.items():
            table[index, positions * order:positions * order + len(entries)] = entries
        for index, entries in ending.items():
            start = positions * len(strings) + ends * order
            table[index, start:start + len(entries)] = entries
    for piece, index in zip(pieces, ids, strict=True):
        table[index, -1] = len(piece)
    return StopStrings(jnp.asarray(table), jnp.asarray([len(value) for value in strings], jnp.int32),
                       positions, ends, max(len(value) for value in strings))


def as_pytree(value: LogitsTransform) -> LogitsTransform:
    """`value` in a form `jax.jit` accepts as data.

    A built-in whose numbers fix no shape lowers to its `apply` with them
    bound as dynamic leaves (`_Numerical`): a temperature, a filter's mass, a
    penalty, a length or count. Other built-ins retain their registered
    pytrees, with what fixes a shape (a top-k width, an n-gram size) static.
    A plain function is wrapped in a Partial that keeps the function static
    and its bound arguments as data; strategies use that same callable rule.
    """
    if isinstance(value, _Numerical):
        return value.numerical()
    return jax.tree_util.Partial(value) if jax.tree_util.all_leaves([value]) else value


def components(values: LogitsTransform | Sequence[LogitsTransform]) -> tuple[LogitsTransform, ...]:
    """`values` as a tuple of pytrees `jax.jit` accepts as data.

    Takes one transform or criterion or a sequence of either.
    """
    if callable(values):
        return (as_pytree(values),)
    return tuple(as_pytree(value) for value in values)


@dataclasses.dataclass(frozen=True, eq=False)
class LogitsChain:
    """The ordered transforms, keeping a terminal Greedy for the sampler.

    An arbitrary transform after Greedy may restore a nondegenerate
    distribution, so only the final built-in Greedy proves an argmax draw.
    Like a function closure, the callable is hashed by identity, not by its
    captured processor arrays, when passed directly to `jax.jit`.
    """

    transforms: tuple[LogitsTransform, ...]

    @property
    def greedy(self) -> bool:
        return bool(self.transforms) and type(self.transforms[-1]) is Greedy

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        for transform in self.transforms:
            logits = transform(state, logits).astype(jnp.float32)
        return logits


def chain(transforms: Sequence[LogitsTransform]) -> LogitsChain:
    """The transforms as one callable, applied in order."""
    return LogitsChain(tuple(transforms))


class Finish(NamedTuple):
    """The rows a step's criteria finished, and those of them a criterion ended, as a stop does,
    rather than one cut short at a length (`Stopping`'s `truncates`): the terminated rows."""

    done: jax.Array
    ended: jax.Array


class Finishing(NamedTuple):
    """A request's criteria as pytrees (`components`), those that end a row apart from those that
    cut it short at a length, which is told from each criterion as given (`Truncates`) before it is
    one: a built-in's numbers ride as data (`as_pytree`)."""

    ending: tuple[Stopping, ...]
    truncating: tuple[Stopping, ...]

    @classmethod
    def of(cls, criteria: Sequence[Stopping]) -> Finishing:
        ending = tuple(as_pytree(stop) for stop in criteria if not isinstance(stop, Truncates))
        return cls(ending, tuple(as_pytree(stop) for stop in criteria if isinstance(stop, Truncates)))


def finish(stopping: Finishing) -> Callable[[StepState, jax.Array], Finish]:
    """The criteria as one callable: which rows any of them finished, and which one that ends a row
    finished, each combined with OR."""

    def finished(state: StepState, tokens: jax.Array) -> Finish:
        ended = cut = jnp.zeros(tokens.shape, bool)
        for stop in stopping.ending:
            ended = ended | stop(state, tokens)
        for stop in stopping.truncating:
            cut = cut | stop(state, tokens)
        return Finish(ended | cut, ended)

    return finished


def criterion(stopping: Finishing | Sequence[Stopping]) -> Callable[[StepState, jax.Array], jax.Array]:
    """The criteria as one callable, combined with OR: which rows any of them finished (`finish`)."""
    finished = finish(stopping if isinstance(stopping, Finishing) else Finishing(tuple(stopping), ()))
    return lambda state, tokens: finished(state, tokens).done


__all__ = [
    "BeginSuppressTokens",
    "EndOfSequence",
    "EpsilonCutoff",
    "EtaCutoff",
    "ExponentialDecayLengthPenalty",
    "Finish",
    "Finishing",
    "ForcedBOS",
    "ForcedEOS",
    "FrequencyPenalty",
    "Greedy",
    "LogitsChain",
    "LogitsTransform",
    "MaxLength",
    "MaxNewTokens",
    "MinLength",
    "MinNewTokens",
    "MinP",
    "NoRepeatNGram",
    "PresencePenalty",
    "PromptNoRepeatNGram",
    "PromptRepetitionPenalty",
    "RemoveInvalidValues",
    "Renormalize",
    "RepetitionPenalty",
    "SequenceBias",
    "StepState",
    "StopStrings",
    "Stopping",
    "SuppressTokens",
    "Temperature",
    "TopH",
    "TopK",
    "TopP",
    "Truncates",
    "Typical",
]
