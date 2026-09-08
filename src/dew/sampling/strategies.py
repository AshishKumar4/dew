"""Decoding strategies: the device loop a generation request runs.

A strategy receives the prefilled `DecoderState`, the initial `StepState`, the
typed operations that move the model forward and reparent its cache rows, the
composed transform chain and stopping criterion, the token budget and the
number of continuations. It returns one `Draws` record per output row.

`Sample` draws each row independently. `Beam` and `Speculative` in this module
use the same operations, so a user strategy is a callable with this signature
and nothing else: no registry, no server object, no model access.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax
from jax.experimental import checkify

from dew.nn.inputs import continuation_keys, prompt_major
from dew.objectives.base import Variables
from dew.sampling.decoding import StepState


@struct.dataclass
class DecoderState:
    """The model-owned decode carry.

    `logits` scores the next position of every row. `positions` continues
    explicitly supplied scalar rotary coordinates; the physical cursors live
    inside the cache. `hidden` holds the final states of the last step for a
    strategy that drafts from them, and is None otherwise.
    """

    cache: Variables
    logits: jax.Array
    positions: jax.Array | None = None
    hidden: jax.Array | None = None


@struct.dataclass
class Draws:
    """One strategy's output rows: `[rows, budget]` tokens and likelihoods.

    `valid` marks the slots a row actually emitted, so its length is that
    row's count and the slots above it carry no likelihood. `terminated`
    marks a row a stopping criterion ended, as opposed to one that ran out
    of budget.
    """

    tokens: jax.Array
    valid: jax.Array
    behavior_log_probs: jax.Array
    raw_log_probs: jax.Array
    terminated: jax.Array


Advance = Callable[[DecoderState, jax.Array, jax.Array], DecoderState]
Reindex = Callable[[DecoderState, jax.Array], DecoderState]
Propose = Callable[[DecoderState, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


@dataclasses.dataclass(frozen=True)
class DecodeOps:
    """What a strategy may do to the model, and nothing more.

    `advance` feeds one token per row and returns the state scoring the next
    position; inactive rows leave their cache untouched. `reindex` gathers
    cache rows, so a strategy can duplicate, reorder or drop a row's whole
    decode state. `propose` runs the model's prediction depths over a
    candidate token, returning its logits and hidden state, and is None on a
    model without them.
    """

    advance: Advance
    reindex: Reindex
    propose: Propose | None = None


class Strategy(Protocol):
    """The device loop of one generation request."""

    def __call__(self, state: DecoderState, start: StepState, ops: DecodeOps,
                 transform: Callable[[StepState, jax.Array], jax.Array],
                 stopping: Callable[[StepState, jax.Array], jax.Array],
                 budget: int, n: int, /) -> Draws: ...


def as_pytree(value: Strategy) -> Strategy:
    """The strategy as data for `jax.jit`, by `decoding.as_pytree`'s rule."""
    if not callable(value):
        raise TypeError(f"a strategy must be callable, got {type(value).__name__}")
    return jax.tree_util.Partial(value) if jax.tree_util.all_leaves([value]) else value


def draw(state: StepState, logits: jax.Array,
         transform: Callable[[StepState, jax.Array], jax.Array]
         ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One categorical draw per row, with its behaviour and raw likelihood.

    The raw log probabilities come from the model's own distribution, before
    any transform; the behaviour ones from the distribution that actually
    draws, after the whole chain. An active row whose chain left no finite
    score has no distribution to draw from, so the call raises instead of
    returning the first index.
    """
    scores = transform(state, logits.astype(jnp.float32))
    defined = jnp.any(jnp.isfinite(scores), axis=-1) & ~jnp.any(jnp.isnan(scores), axis=-1)
    checkify.check(jnp.all(defined | ~state.active),
                   "the transform chain left an active row without a distribution to draw from")
    keys = jax.vmap(jax.random.fold_in)(state.keys, state.step)
    token = jax.vmap(jax.random.categorical)(keys, scores).astype(jnp.int32)
    behavior = jnp.take_along_axis(jax.nn.log_softmax(scores), token[:, None], -1)[:, 0]
    raw = jnp.take_along_axis(jax.nn.log_softmax(logits.astype(jnp.float32)), token[:, None], -1)[:, 0]
    return token, behavior, raw


@struct.dataclass
class Sample:
    """Draw every row independently, one token per step.

    This is the loop `generate` runs when a request names no strategy.
    Continuations of a prompt share its prefill and run one after another, so
    decode memory does not grow with `n` and a routed-expert forward sees the
    same batch as a single continuation.
    """

    def __call__(self, state: DecoderState, start: StepState, ops: DecodeOps,
                 transform: Callable[[StepState, jax.Array], jax.Array],
                 stopping: Callable[[StepState, jax.Array], jax.Array],
                 budget: int, n: int) -> Draws:
        if n == 1:
            return _sample_rows(state, start, ops, transform, stopping, budget)
        return prompt_major(lax.map(
            lambda keys: _sample_rows(state, dataclasses.replace(start, keys=keys), ops, transform,
                                      stopping, budget),
            continuation_keys(start.keys, n)))


def _sample_rows(state: DecoderState, start: StepState, ops: DecodeOps,
                 transform: Callable[[StepState, jax.Array], jax.Array],
                 stopping: Callable[[StepState, jax.Array], jax.Array],
                 budget: int) -> Draws:
    """The fixed-trip decode scan from one prefilled state."""

    def step(carry, _):
        state, step_state, terminated = carry
        active = step_state.active
        token, behavior, raw = draw(step_state, state.logits, transform)
        committed = step_state.commit(token, active)
        stopped = active & stopping(committed, token)
        following = ops.advance(state, token, active)
        carry = (following, dataclasses.replace(committed, active=active & ~stopped),
                 terminated | stopped)
        return carry, (token, active, jnp.where(active, behavior, 0.0), jnp.where(active, raw, 0.0))

    initial = (state, start, jnp.zeros(start.rows, bool))
    (_, _, terminated), columns = lax.scan(step, initial, None, length=budget)
    tokens, valid, behavior, raw = (jnp.swapaxes(value, 0, 1) for value in columns)
    return Draws(tokens, valid, behavior, raw, terminated)
