"""Logit transforms and stopping criteria for the decode loop.

Three concepts extend decoding. A `LogitsTransform` is a pure callable from
the step state and `[rows, vocab]` logits to new logits. A `Stopping` is a
pure callable from the step state and the tokens just drawn to a per-row
finished flag; criteria combine with OR after every committed token. A
`Strategy` in `dew.sampling.strategies` owns the device loop.

Transforms and criteria read `StepState`, which carries the token history and
nothing about the model: no parameters, no cache. Every built-in here is a
pytree, so a configuration holding arrays crosses `jax.jit` as data instead of
entering a compilation cache key. A plain function works as well, and
`jax.tree_util.Partial` carries array configuration for one.

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
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax
from jax.typing import ArrayLike

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
    per row.
    """

    tokens: jax.Array
    valid: jax.Array
    step: jax.Array
    active: jax.Array
    keys: jax.Array
    prompt_width: int = struct.field(pytree_node=False, default=0)

    @property
    def width(self) -> int:
        """Slots in the buffer, prompt plus budget."""
        return self.tokens.shape[1]

    @property
    def rows(self) -> int:
        return self.tokens.shape[0]

    def total(self) -> jax.Array:
        """Real tokens each row holds, prompt and generated together."""
        return jnp.sum(self.valid, axis=-1, dtype=jnp.int32)

    def history(self) -> tuple[jax.Array, jax.Array]:
        """Each row's real tokens left aligned, and how many there are.

        Prompts pad wherever their batch needed it, so a transform that reads
        order (n-grams, biased sequences, stop strings) needs the row's own
        tokens without holes. Padding slots hold -1, which no token id equals.
        """
        return _compact(self.tokens, self.valid)

    def prompt_history(self) -> tuple[jax.Array, jax.Array]:
        """The prompt region's real tokens left aligned, and how many."""
        width = self.prompt_width
        return _compact(self.tokens[:, :width], self.valid[:, :width])

    def generated(self) -> tuple[jax.Array, jax.Array]:
        """The drawn region's real tokens left aligned, and how many."""
        width = self.prompt_width
        return _compact(self.tokens[:, width:], self.valid[:, width:])

    def commit(self, tokens: jax.Array, drawn: jax.Array) -> StepState:
        """The state after `drawn` rows appended `tokens` at their next slot."""
        rows = jnp.arange(self.rows)
        slot = self.prompt_width + self.step
        return dataclasses.replace(
            self,
            tokens=self.tokens.at[rows, slot].set(jnp.where(drawn, tokens, 0), mode="drop"),
            valid=self.valid.at[rows, slot].set(drawn, mode="drop"),
            step=self.step + drawn.astype(jnp.int32))


def _compact(tokens: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array]:
    order = jnp.argsort(~valid, axis=-1, stable=True)
    lengths = jnp.sum(valid, axis=-1, dtype=jnp.int32)
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
    slots = jnp.where(valid, tokens, vocab)
    return jnp.zeros((tokens.shape[0], vocab), bool).at[rows, slots].set(True, mode="drop")


def _counts(tokens: jax.Array, valid: jax.Array, vocab: int) -> jax.Array:
    """How often each vocabulary entry appears among a row's valid tokens."""
    rows = jnp.arange(tokens.shape[0])[:, None]
    slots = jnp.where(valid, tokens, vocab)
    return jnp.zeros((tokens.shape[0], vocab), jnp.float32).at[rows, slots].add(1.0, mode="drop")


def _ids(name: str, value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must hold non-negative token ids")
    ids = (value,) if isinstance(value, int) else tuple(value)
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError(f"{name} must hold non-negative token ids")
    return ids


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _unit(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return float(value)


def _size(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@runtime_checkable
class LogitsTransform(Protocol):
    """A pure `[rows, vocab]` score rewrite, applied before the draw."""

    def __call__(self, state: StepState, logits: jax.Array, /) -> jax.Array: ...


@runtime_checkable
class Stopping(Protocol):
    """A pure per-row finish test over the tokens a step just drew."""

    def __call__(self, state: StepState, tokens: jax.Array, /) -> jax.Array: ...


@struct.dataclass
class Greedy:
    """The argmax as a distribution: zero on the best token, `-inf` elsewhere.

    `Sampling(temperature=0)` compiles to this, so a zero-temperature draw
    stays the deterministic argmax and its behaviour log probability stays
    exactly zero while running through the same categorical draw as any other
    policy. Transforms placed before it still shape the argmax, which is what
    greedy search does with a processor list.
    """

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        best = jnp.argmax(logits, axis=-1)[:, None]
        return jnp.where(jnp.arange(logits.shape[-1])[None, :] == best, 0.0, FILTER)


@struct.dataclass
class Temperature:
    """`logits / value`, as `TemperatureLogitsWarper`."""

    value: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        _positive("temperature", self.value)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return logits / self.value


@struct.dataclass
class TopK:
    """Keep the `k` highest scores, as `TopKLogitsWarper`."""

    k: int = struct.field(pytree_node=False, default=1)

    def __post_init__(self) -> None:
        _size("top_k", self.k)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        keep = min(self.k, logits.shape[-1])
        cutoff = lax.top_k(logits, keep)[0][..., -1:]
        return jnp.where(logits < cutoff, FILTER, logits)


@struct.dataclass
class TopP:
    """Nucleus filtering, as `TopPLogitsWarper`.

    The ascending tail holding cumulative mass at most `1 - p` is removed and
    the best token always survives.
    """

    p: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        _unit("top_p", self.p)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        order = jnp.argsort(logits, axis=-1, stable=True)
        ascending = jnp.take_along_axis(logits, order, axis=-1)
        tail = jnp.cumsum(jax.nn.softmax(ascending), axis=-1) <= 1.0 - self.p
        tail = tail.at[:, -1].set(False)
        rows = jnp.arange(logits.shape[0])[:, None]
        removed = jnp.zeros_like(tail).at[rows, order].set(tail)
        return jnp.where(removed, FILTER, logits)


@struct.dataclass
class MinP:
    """Relative filtering at `p` times the top probability, as `MinPLogitsWarper`."""

    p: float = struct.field(pytree_node=False, default=0.0)

    def __post_init__(self) -> None:
        _unit("min_p", self.p)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        probabilities = jax.nn.softmax(logits)
        cutoff = jnp.max(probabilities, axis=-1, keepdims=True) * self.p
        return jnp.where(probabilities < cutoff, FILTER, logits)


@struct.dataclass
class Typical:
    """Locally typical filtering, as `TypicalLogitsWarper`."""

    mass: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        _unit("typical_p", self.mass)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        normalized = jax.nn.log_softmax(logits)
        probabilities = jnp.exp(normalized)
        entropy = -jnp.sum(jnp.where(jnp.isneginf(normalized), 0.0, normalized * probabilities),
                           axis=-1, keepdims=True)
        shifted = jnp.abs(-normalized - entropy)
        order = jnp.argsort(shifted, axis=-1, stable=True)
        sorted_shift = jnp.take_along_axis(shifted, order, axis=-1)
        cumulative = jnp.cumsum(jax.nn.softmax(jnp.take_along_axis(logits, order, axis=-1)), axis=-1)
        last = jnp.clip(jnp.sum(cumulative < self.mass, axis=-1), 0, logits.shape[-1] - 1)
        removed = sorted_shift > jnp.take_along_axis(sorted_shift, last[:, None], axis=-1)
        removed = removed.at[:, 0].set(False)
        rows = jnp.arange(logits.shape[0])[:, None]
        scattered = jnp.zeros_like(removed).at[rows, order].set(removed)
        return jnp.where(scattered, FILTER, logits)


@struct.dataclass
class EpsilonCutoff:
    """Remove tokens below an absolute probability, as `EpsilonLogitsWarper`."""

    epsilon: float = struct.field(pytree_node=False, default=0.0)

    def __post_init__(self) -> None:
        _unit("epsilon_cutoff", self.epsilon)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        best = jnp.max(logits, axis=-1, keepdims=True)
        removed = (jax.nn.softmax(logits) < self.epsilon) & (logits < best)
        return jnp.where(removed, FILTER, logits)


@struct.dataclass
class EtaCutoff:
    """Entropy-scaled cutoff, as `EtaLogitsWarper`."""

    epsilon: float = struct.field(pytree_node=False, default=0.0)

    def __post_init__(self) -> None:
        _unit("eta_cutoff", self.epsilon)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        normalized = jax.nn.log_softmax(logits)
        probabilities = jnp.exp(normalized)
        entropy = -jnp.sum(jnp.where(jnp.isneginf(normalized), 0.0, normalized * probabilities), axis=-1)
        eta = jnp.minimum(self.epsilon, math.sqrt(self.epsilon) * jnp.exp(-entropy))[:, None]
        best = jnp.max(logits, axis=-1, keepdims=True)
        return jnp.where((probabilities < eta) & (logits < best), FILTER, logits)


@struct.dataclass
class TopH:
    """Entropy-budget filtering, as `TopHLogitsWarper`.

    Tokens enter in probability order while the cumulative entropy of the
    truncated head stays within `h` times its total entropy, and the best
    token always enters. `n` is the head the reference fixes at 100.
    """

    h: float = struct.field(pytree_node=False, default=1.0)
    n: int = struct.field(pytree_node=False, default=100)

    def __post_init__(self) -> None:
        if _unit("top_h", self.h) <= 0:
            raise ValueError("top_h must be finite and above zero up to one")
        _size("top_h candidates", self.n)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        head, index = lax.top_k(logits, min(self.n, logits.shape[-1]))
        normalized = jax.nn.log_softmax(head)
        probabilities = jnp.exp(normalized)
        terms = -probabilities * normalized
        threshold = jnp.sum(terms, axis=-1, keepdims=True) * self.h
        selected = jnp.cumsum(terms, axis=-1) <= threshold
        selected = selected.at[:, 0].set(True)
        rows = jnp.arange(logits.shape[0])[:, None]
        keep = jnp.zeros(logits.shape, bool).at[rows, index].set(selected)
        return jnp.where(keep, logits, FILTER)


@struct.dataclass
class Renormalize:
    """Replace scores by their log softmax, as `LogitNormalization`."""

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return jax.nn.log_softmax(logits)


@struct.dataclass
class RemoveInvalidValues:
    """Map NaN to zero and infinities to the float range, as `InfNanRemoveLogitsProcessor`.

    Nothing else in the chain repairs a broken distribution: an undefined draw
    raises instead. Ask for this transform to sanitize one.
    """

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        info = jnp.finfo(logits.dtype)
        cleaned = jnp.where(jnp.isnan(logits), 0.0, logits)
        cleaned = jnp.where(logits == jnp.inf, info.max, cleaned)
        return jnp.where(logits == -jnp.inf, info.min, cleaned)


@struct.dataclass
class RepetitionPenalty:
    """Divide positive scores of seen tokens and multiply negative ones.

    The history is the row's valid prompt and drawn tokens, as
    `RepetitionPenaltyLogitsProcessor` reads the whole `input_ids`.
    """

    penalty: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        _positive("repetition_penalty", self.penalty)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        seen = _present(state.tokens, state.valid, logits.shape[-1])
        return jnp.where(seen, _scaled(logits, self.penalty), logits)


@struct.dataclass
class PromptRepetitionPenalty:
    """Raise the scores of prompt tokens, as `EncoderRepetitionPenaltyLogitsProcessor`.

    The reference inverts its argument, so a penalty above one rewards
    repeating the prompt. A decoder-only prompt is the encoder input here.
    """

    penalty: float = struct.field(pytree_node=False, default=1.0)

    def __post_init__(self) -> None:
        _positive("encoder_repetition_penalty", self.penalty)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        width = state.prompt_width
        seen = _present(state.tokens[:, :width], state.valid[:, :width], logits.shape[-1])
        return jnp.where(seen, _scaled(logits, 1.0 / self.penalty), logits)


def _scaled(logits: jax.Array, penalty: float) -> jax.Array:
    return jnp.where(logits < 0, logits * penalty, logits / penalty)


@struct.dataclass
class FrequencyPenalty:
    """Subtract `penalty` times each token's count among the drawn tokens.

    vLLM's formula, `logits -= frequency_penalties * output_bin_counts`
    (`model_executor/layers/utils.py`), which is also OpenAI's
    `frequency_penalty`. It counts generated tokens, not the prompt.
    """

    penalty: float = struct.field(pytree_node=False, default=0.0)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        width = state.prompt_width
        counts = _counts(state.tokens[:, width:], state.valid[:, width:], logits.shape[-1])
        return logits - self.penalty * counts


@struct.dataclass
class PresencePenalty:
    """Subtract `penalty` from every token already drawn.

    vLLM's `logits -= presence_penalties * output_mask`, which is OpenAI's
    `presence_penalty`. It reads generated tokens, not the prompt.
    """

    penalty: float = struct.field(pytree_node=False, default=0.0)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        width = state.prompt_width
        seen = _present(state.tokens[:, width:], state.valid[:, width:], logits.shape[-1])
        return logits - self.penalty * seen.astype(logits.dtype)


@struct.dataclass
class NoRepeatNGram:
    """Ban tokens that would repeat an n-gram of the row's own history.

    The tensorised form of `NoRepeatNGramLogitsProcessor`: the current suffix
    is matched against every window, and a matching window bans the token that
    followed it. The window starting at the suffix itself needs one token more
    than the row has, so a suffix never bans its own successor.
    """

    size: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self) -> None:
        _size("no_repeat_ngram_size", self.size)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        tokens, lengths = state.history()
        return _ngram_ban(logits, tokens, lengths, _suffix(tokens, lengths, self.size - 1),
                          lengths, self.size)


@struct.dataclass
class PromptNoRepeatNGram:
    """Ban tokens that would repeat an n-gram of the prompt.

    `EncoderNoRepeatNGramLogitsProcessor` builds its table from the encoder
    input and matches it against the decoder's suffix. The prompt is the
    encoder input of a decoder-only model.
    """

    size: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self) -> None:
        _size("encoder_no_repeat_ngram_size", self.size)

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
    slots = jnp.where(matched, windows[..., -1], logits.shape[-1])
    banned = jnp.zeros(logits.shape, bool).at[rows, slots].set(True, mode="drop")
    return jnp.where(banned, FILTER, logits)


@struct.dataclass
class SequenceBias:
    """Add a bias to the token that would complete each biased sequence.

    `SequenceBiasLogitsProcessor` as a table: `sequences` is `[count, width]`
    right-aligned token ids, `lengths` their real lengths and `bias` the value
    added to the last id when the row's suffix matches the preceding ones. A
    sequence longer than the row's history is skipped, as the reference skips
    one longer than the context.
    """

    sequences: jax.Array
    lengths: jax.Array
    bias: jax.Array

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        tokens, history = state.history()
        width = self.sequences.shape[1]
        prefix = self.sequences[:, :-1]
        needed = self.lengths - 1
        held = jnp.arange(width - 1)[None, :] >= (width - 1 - needed[:, None])
        suffix = _suffix(tokens, history, width - 1)
        matched = jnp.all((suffix[:, None, :] == prefix[None, :, :]) | ~held[None, :, :], axis=-1)
        matched = matched & (self.lengths[None, :] <= history[:, None])
        added = jnp.where(matched, self.bias[None, :], 0.0)
        rows = jnp.arange(logits.shape[0])[:, None]
        last = jnp.broadcast_to(self.sequences[None, :, -1], added.shape)
        return logits + jnp.zeros(logits.shape, jnp.float32).at[rows, last].add(added, mode="drop")


def sequence_bias(entries: Sequence[tuple[Sequence[int], float]]) -> SequenceBias:
    """A `SequenceBias` table from `(token ids, bias)` pairs."""
    pairs = [(tuple(ids), float(value)) for ids, value in entries]
    for ids, _ in pairs:
        _ids("sequence_bias", ids)
    if not pairs:
        raise ValueError("sequence_bias needs at least one sequence")
    width = max(len(ids) for ids, _ in pairs)
    table = np.zeros((len(pairs), width), np.int32)
    for row, (ids, _) in enumerate(pairs):
        table[row, width - len(ids):] = ids
    return SequenceBias(jnp.asarray(table),
                        jnp.asarray([len(ids) for ids, _ in pairs], jnp.int32),
                        jnp.asarray([value for _, value in pairs], jnp.float32))


def bad_words(ids: Sequence[Sequence[int]], eos_id: int | Sequence[int] | None = None) -> SequenceBias:
    """A `-inf` `SequenceBias` over forbidden sequences, as `NoBadWordsLogitsProcessor`.

    Single-token sequences that name an EOS id are dropped, as the reference
    drops them, so banning bad words cannot ban termination.
    """
    stops = () if eos_id is None else _ids("eos_id", eos_id)
    kept = [tuple(word) for word in ids if not (len(word) == 1 and word[0] in stops)]
    if not kept:
        raise ValueError("bad_words_ids holds no sequence outside the EOS ids")
    return sequence_bias([(word, -math.inf) for word in kept])


@struct.dataclass
class SuppressTokens:
    """Remove a fixed set of tokens, as `SuppressTokensLogitsProcessor`."""

    tokens: jax.Array

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return jnp.where(_membership(self.tokens, logits.shape[-1]), FILTER, logits)


@struct.dataclass
class BeginSuppressTokens:
    """Remove tokens at one generated position, as `SuppressTokensAtBeginLogitsProcessor`.

    The reference suppresses where the sequence still has its prompt width, so
    `offset` is the generated index the suppression applies at, zero for the
    first drawn token and one where a forced BOS occupies that slot.
    """

    tokens: jax.Array
    offset: int = struct.field(pytree_node=False, default=0)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        removed = _membership(self.tokens, logits.shape[-1])[None, :] & (state.step == self.offset)[:, None]
        return jnp.where(removed, FILTER, logits)


def _membership(tokens: jax.Array, vocab: int) -> jax.Array:
    return jnp.zeros(vocab, bool).at[tokens].set(True, mode="drop")


@struct.dataclass
class ForcedBOS:
    """Force one token as the first of the whole sequence, as `ForcedBOSTokenLogitsProcessor`."""

    token: int = struct.field(pytree_node=False, default=0)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return _forced(logits, self.token, state.total() == 1)


@struct.dataclass
class ForcedEOS:
    """Force EOS one step before the length limit, as `ForcedEOSTokenLogitsProcessor`."""

    token: int = struct.field(pytree_node=False, default=0)
    max_length: int = struct.field(pytree_node=False, default=1)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return _forced(logits, self.token, state.total() == self.max_length - 1)


def _forced(logits: jax.Array, token: int, rows: jax.Array) -> jax.Array:
    only = jnp.where(jnp.arange(logits.shape[-1])[None, :] == token, 0.0, FILTER)
    return jnp.where(rows[:, None], only, logits)


@struct.dataclass
class MinLength:
    """Suppress EOS until the whole sequence reaches `length`, as `MinLengthLogitsProcessor`."""

    length: int = struct.field(pytree_node=False, default=0)
    eos: jax.Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.int32))

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return _suppress_eos(logits, self.eos, state.total() < self.length)


@struct.dataclass
class MinNewTokens:
    """Suppress EOS until `count` tokens are drawn, as `MinNewTokensLengthLogitsProcessor`."""

    count: int = struct.field(pytree_node=False, default=0)
    eos: jax.Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.int32))

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        return _suppress_eos(logits, self.eos, state.step < self.count)


def _suppress_eos(logits: jax.Array, eos: jax.Array, rows: jax.Array) -> jax.Array:
    return jnp.where(rows[:, None] & _membership(eos, logits.shape[-1])[None, :], FILTER, logits)


@struct.dataclass
class ExponentialDecayLengthPenalty:
    """Grow the EOS score after `start` drawn tokens, as `ExponentialDecayLengthPenalty`.

    The reference measures from `start_index + prompt_width`, which is the
    generated count used here, and adds `|score| * (factor ** index - 1)` so a
    negative score also rises.
    """

    start: int = struct.field(pytree_node=False, default=0)
    factor: float = struct.field(pytree_node=False, default=1.0)
    eos: jax.Array = struct.field(default_factory=lambda: jnp.zeros((0,), jnp.int32))

    def __post_init__(self) -> None:
        _positive("exponential_decay_length_penalty factor", self.factor)

    def __call__(self, state: StepState, logits: jax.Array) -> jax.Array:
        index = jnp.maximum(state.step - self.start, 0)
        growth = jnp.where(state.step > self.start, self.factor ** index.astype(jnp.float32) - 1.0, 0.0)
        selected = _membership(self.eos, logits.shape[-1])[None, :]
        return logits + jnp.where(selected, jnp.abs(logits) * growth[:, None], 0.0)


@struct.dataclass
class EndOfSequence:
    """Finish a row that drew one of the EOS ids, as `EosTokenCriteria`."""

    eos: jax.Array

    def __call__(self, state: StepState, tokens: jax.Array) -> jax.Array:
        return jnp.isin(tokens, self.eos)


@struct.dataclass
class MaxNewTokens:
    """Finish a row once it has drawn `count` tokens."""

    count: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ValueError("max new tokens must be a non-negative integer")

    def __call__(self, state: StepState, tokens: jax.Array) -> jax.Array:
        return state.step >= self.count


@struct.dataclass
class MaxLength:
    """Finish a row once prompt and generated tokens reach `length`, as `MaxLengthCriteria`."""

    length: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self) -> None:
        if type(self.length) is not int or self.length < 0:
            raise ValueError("max length must be a non-negative integer")

    def __call__(self, state: StepState, tokens: jax.Array) -> jax.Array:
        return state.total() >= self.length


@struct.dataclass
class StopStrings:
    """Finish a row whose text ends with one of the compiled stop strings.

    The tables come from `stop_strings`, which reads the tokenizer once. The
    device check is `StopStringCriteria`'s: walk the row's tokens backwards,
    require the last token to overlap the end of a stop string, and keep
    matching earlier tokens against the positions where they can sit. A match
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


class Vocabulary(Protocol):
    """What `stop_strings` needs from a tokenizer: text for token rows."""

    def decode(self, tokens: ArrayLike) -> list[str]: ...


def stop_strings(processor: Vocabulary, strings: str | Sequence[str], vocab_size: int,
                 *, probe: Sequence[int] = (1,)) -> StopStrings:
    """Compile a tokenizer's vocabulary against `strings` into a `StopStrings`.

    The tables record, for every token, where it can sit inside a stop string
    and how many of the string's trailing characters its start can cover. This
    runs once on the host; the criterion itself never decodes.

    A token's text is read by decoding `probe` followed by that token and
    dropping the text `probe` alone produces, because a tokenizer's decoder
    adds or removes a leading space depending on what came before.
    """
    wanted = (strings,) if isinstance(strings, str) else tuple(strings)
    if not wanted or any(not isinstance(value, str) or not value for value in wanted):
        raise ValueError("stop_strings needs non-empty strings")
    prefix = list(probe)
    rows = np.asarray([prefix + [token] for token in range(vocab_size)], np.int32)
    base = processor.decode(np.asarray([prefix], np.int32))[0]
    texts = [text[len(base):] if text.startswith(base) else text
             for text in processor.decode(rows)]
    return _stop_string_tables(texts, wanted)


def _stop_string_tables(texts: Sequence[str], strings: Sequence[str]) -> StopStrings:
    """`StopStringCriteria._stop_string_create_embedding_vec` over plain text."""
    valid: list[dict[int, list[int]]] = []
    overlaps: list[dict[int, list[int]]] = []
    for target in strings:
        reversed_target = target[::-1]
        inside: dict[int, list[int]] = {}
        ending: dict[int, list[int]] = {}
        for index, token in enumerate(texts):
            reversed_token = token[::-1]
            for start in range(1 - len(token), len(target)):
                if start < 0:
                    piece, position = reversed_token[-start:], 0
                else:
                    piece, position = reversed_token, start
                if piece.startswith(reversed_target[position:position + len(piece)]):
                    if position == 0:
                        ending.setdefault(index, []).append(min(len(piece), len(target)))
                    else:
                        inside.setdefault(index, []).append(position)
        valid.append(inside)
        overlaps.append(ending)
    if not any(overlaps):
        raise ValueError("no token in the vocabulary can end any of the stop strings")
    positions = max((len(item) for row in valid for item in row.values()), default=1)
    ends = max(len(item) for row in overlaps for item in row.values())
    width = len(strings) * (positions + ends) + 1
    table = np.full((len(texts) + 1, width), -1, np.int32)
    for order, (inside, ending) in enumerate(zip(valid, overlaps)):
        for index, items in inside.items():
            table[index, positions * order:positions * order + len(items)] = items
        for index, items in ending.items():
            start = positions * len(strings) + ends * order
            table[index, start:start + len(items)] = items
    for index, token in enumerate(texts):
        table[index, -1] = len(token)
    return StopStrings(jnp.asarray(table), jnp.asarray([len(value) for value in strings], jnp.int32),
                       positions, ends, max(len(value) for value in strings))


def as_pytree(value: LogitsTransform) -> LogitsTransform:
    """`value` in a form `jax.jit` accepts as data.

    A built-in transform or criterion is already a pytree. A plain function is
    a leaf that `jit` cannot take, so it is wrapped in
    `jax.tree_util.Partial`, which keeps the function in the tree structure
    and any array arguments as data. `dew.sampling.strategies` applies the
    same rule to a strategy.
    """
    if not callable(value):
        raise TypeError(f"a decoding component must be callable, got {type(value).__name__}")
    return jax.tree_util.Partial(value) if jax.tree_util.all_leaves([value]) else value


def components(values: LogitsTransform | Sequence[LogitsTransform],
               kind: str) -> tuple[LogitsTransform, ...]:
    """A user-supplied transform or criterion, or a sequence of them, for `jit`."""
    if callable(values):
        return (as_pytree(values),)
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{kind} must be a callable or a sequence of callables")
    return tuple(as_pytree(value) for value in values)


def chain(transforms: Sequence[LogitsTransform]) -> Callable[[StepState, jax.Array], jax.Array]:
    """The transforms as one callable, applied in order."""

    def apply(state: StepState, logits: jax.Array) -> jax.Array:
        for transform in transforms:
            logits = transform(state, logits).astype(jnp.float32)
        return logits

    return apply


def criterion(stopping: Sequence[Stopping]) -> Callable[[StepState, jax.Array], jax.Array]:
    """The criteria as one callable, combined with OR."""

    def finished(state: StepState, tokens: jax.Array) -> jax.Array:
        done = jnp.zeros(tokens.shape, bool)
        for stop in stopping:
            done = done | stop(state, tokens)
        return done

    return finished
