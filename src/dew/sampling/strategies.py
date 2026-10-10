"""Decoding strategies: the device loop a generation request runs.

A strategy receives the prefilled `DecoderState`, the initial `StepState`, the
typed operations that move the model forward and reparent its cache rows, the
composed transform chain and stopping criterion, the token budget and the
number of continuations. It returns one `Draws` record per output row.

`Sample`, `Beam` and `Speculative` use only those operations, so a user
strategy is a callable with the same signature: no registry, no server object,
no model access.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax
from jax.experimental import checkify

from dew.nn.inputs import PredictionPhase, continuation_keys, prompt_major
from dew.nn.scatter import DROPPED
from dew.objectives.base import Variables, token_log_probs
from dew.sampling import decoding
from dew.sampling.decoding import StepState
from dew.sampling.guided import Grammar


@struct.dataclass
class DecoderState:
    """The model-owned decode carry.

    `logits` scores the next position of every row. `positions` is the
    logical coordinate the next token of every row sits at, when the prompt
    supplied its coordinates; None leaves the cache to count real tokens.
    Target advancement and draft proposals both read it, and `advance` moves
    it. `hidden` holds the final states of the last step for a strategy that
    drafts from them, and is None otherwise.
    """

    cache: Variables
    logits: jax.Array
    positions: jax.Array | None = None
    hidden: jax.Array | None = None
    drafts: tuple[jax.Array, ...] = ()
    """Each later prediction depth's predecessor state at the last real token:
    entry `d` feeds depth `d + 1`, as `reseed` carries them; `hidden` is
    depth zero's."""
    tokens: jax.Array | None = None
    lengths: jax.Array | None = None
    """Compact token history for a positional table change that invalidates KV."""



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
Verify = Callable[[DecoderState, jax.Array, jax.Array],
                  tuple[DecoderState, jax.Array, jax.Array | None]]
Propose = Callable[[DecoderState, jax.Array, jax.Array | None, jax.Array | None, jax.Array,
                    jax.Array, int, PredictionPhase], tuple[DecoderState, jax.Array, jax.Array]]
Record = Callable[[DecoderState, jax.Array, jax.Array], DecoderState]
DraftBlock = Callable[[DecoderState, jax.Array, Callable[[int, jax.Array], jax.Array]], None]


@dataclasses.dataclass(frozen=True)
class DecodeOps:
    """What a strategy may do to the model, and nothing more.

    `advance` feeds one token per row and returns the state scoring the next
    position, leaving inactive rows' caches untouched. `reindex` gathers
    cache rows, so a strategy can duplicate, reorder or drop a row's decode
    state. `verify` feeds a block of tokens per row behind a validity mask and
    returns every position's logits and hidden states. `propose` runs one of
    the model's `depths` prediction depths over a candidate token at an
    explicit target position, in the given prediction phase, and returns its
    logits and hidden state. `record` and `draft` are a block drafter's:
    `record` appends the states `verify` returned for a stretch's real
    positions to the drafter's windows, and `draft` drafts the next block from
    each row's first token, handing each position's logits to a `choose`.
    """

    advance: Advance
    reindex: Reindex
    verify: Verify | None = None
    propose: Propose | None = None
    embed: Callable[[jax.Array], jax.Array] | None = None
    depths: int = 0
    record: Record | None = None
    draft: DraftBlock | None = None


class Strategy(Protocol):
    """The device loop that runs one generation request.

    A strategy is called with the prefilled `DecoderState`, the initial
    `StepState`, the `DecodeOps` that advance the model and gather its cache
    rows, the composed transform chain, the stopping criterion, the token
    budget and the number of continuations. It returns one `Draws` record
    per output row. `Sample`, `Beam` and `Speculative` use nothing else, so
    your own strategy is a callable with the same signature.
    """

    def __call__(self, state: DecoderState, start: StepState, ops: DecodeOps,
                 transform: Callable[[StepState, jax.Array], jax.Array],
                 stopping: Callable[[StepState, jax.Array], jax.Array],
                 budget: int, n: int, /) -> Draws: ...


def reseed(ops: DecodeOps, state: DecoderState, carry: Sequence[jax.Array | None],
           states: jax.Array, embeds: jax.Array, valid: jax.Array, positions: jax.Array,
           last: jax.Array, *, prior_tokens: jax.Array
           ) -> tuple[DecoderState, tuple[jax.Array, ...]]:
    """Write the prediction cache a stretch of history leaves behind, and
    return each depth's predecessor at `last`, the target's first.

    The entry for token `t` at depth `d` consumes depth `d - 1`'s hidden
    state at `t - 1` with `t`'s own embedding, at `t`'s target coordinate,
    and depth zero's predecessor is the target model, as `mtp_hidden_states`
    trains the depths and `Qwen3_5MultiTokenPredictor.forward` reads them.

    `carry` holds each depth's predecessor at the position before this
    stretch, so a block continues across the boundary. `prior_tokens` counts
    the real tokens before it: depth `d` needs `d + 1` predecessors, which
    rotary coordinates cannot count, and a newly available predecessor is
    kept even before its next depth can write. `embeds` are the prepared
    embeddings, media included. `last` is each row's final written slot; a
    row that wrote nothing keeps its carry.
    """
    propose = ops.propose
    if propose is None:
        return state, tuple(entry for entry in carry if entry is not None)
    ordinal = prior_tokens[:, None] + jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1
    upstream, tails = states, []
    for depth in range(ops.depths):
        head = carry[depth] if depth < len(carry) else None
        before = jnp.concatenate(
            [(jnp.zeros_like(upstream[:, 0]) if head is None else head)[:, None],
             upstream[:, :-1]], axis=1)
        ready = valid & (ordinal > depth)
        state, _, out = propose(state, before, None, embeds, ready, positions, depth, "extend")
        held = jnp.zeros_like(upstream[:, 0]) if head is None else head
        predecessor_ready = jnp.any(valid & (ordinal >= depth), axis=1)
        tail = upstream[jnp.arange(upstream.shape[0]), last]
        mask = predecessor_ready.reshape((predecessor_ready.shape[0],) + (1,) * (tail.ndim - 1))
        tails.append(jnp.where(mask, tail, held))
        upstream = out
    return state, tuple(tails)


def as_pytree(value: Strategy) -> Strategy:
    """The strategy as data for `jax.jit`, by `decoding.as_pytree`'s rule."""
    return jax.tree_util.Partial(value) if jax.tree_util.all_leaves([value]) else value


def wellformed(scores: jax.Array) -> jax.Array:
    """Which rows hold a distribution: every score finite or `-inf`, one finite.

    A NaN or a `+inf` beside a finite score makes the softmax undefined and
    the draw arbitrary; a row with nothing finite has no distribution at all.
    """
    return (jnp.all(jnp.isfinite(scores) | jnp.isneginf(scores), axis=-1)
            & jnp.any(jnp.isfinite(scores), axis=-1))


def select(keys: jax.Array, scores: jax.Array, active: jax.Array) -> jax.Array:
    """One categorical draw per row from `scores`, refusing an undefined row."""
    checkify.check(jnp.all(wellformed(scores) | ~active),
                   "the transform chain left an active row without a distribution to draw from")
    return jax.vmap(jax.random.categorical)(keys, scores).astype(jnp.int32)


def draw(state: StepState, logits: jax.Array,
         transform: Callable[[StepState, jax.Array], jax.Array]
         ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One categorical draw per row, with its behaviour and raw likelihood.

    The raw log probabilities come from the model's own distribution, before
    any transform; the behaviour ones from the distribution that actually
    draws, after the whole chain. A model distribution that is itself
    undefined cannot be reported truthfully, so an active row with one raises
    rather than returning a NaN, whatever a later repair does to the scores.
    """
    raw = logits.astype(jnp.float32)
    checkify.check(jnp.all(wellformed(raw) | ~state.active),
                   "the model produced an active row without a distribution to score")
    if isinstance(transform, decoding.LogitsChain) and transform.greedy:
        scores = decoding.chain(transform.transforms[:-1])(state, raw)
        checkify.check(jnp.all(wellformed(scores) | ~state.active),
                       "the transform chain left an active row without a distribution to draw from")
        token = jnp.argmax(scores, axis=-1).astype(jnp.int32)
        behavior = jnp.zeros(token.shape, jnp.float32)
    else:
        scores = transform(state, raw)
        keys = jax.vmap(jax.random.fold_in)(state.keys, state.step)
        token = select(keys, scores, state.active)
        behavior = token_log_probs(scores, token)
    selected = token_log_probs(raw, token)
    return token, behavior, selected


def continuations(start: StepState, n: int,
                  run: Callable[[StepState], Draws]) -> Draws:
    """`run` over `n` continuations of every prompt, in prompt order.

    One continuation is the request itself, with the prompt's own keys, so
    `n=1` and continuation zero of a larger request are the same draw. The
    rest run as a mapped loop over fresh keys rather than as a wider batch,
    which keeps decode memory and every collective the size of one request.
    """
    if n == 1:
        return run(start)
    return prompt_major(lax.map(
        lambda keys: run(dataclasses.replace(start, keys=keys)),
        continuation_keys(start.keys, n)))


class _Emitted(NamedTuple):
    """The rows a speculative loop has written so far, by response position.

    `valid` marks the slots a row actually emitted, so a block that kept two
    tokens writes two columns and the rest of the block writes none.
    """

    tokens: jax.Array
    valid: jax.Array
    behavior_log_probs: jax.Array
    raw_log_probs: jax.Array


@struct.dataclass
class Sample:
    """Draws every row independently, one token per step.

    This is the loop `generate` runs when a request names no strategy. A
    prompt's continuations share its prefill (`continuations`).

    `grammar` restricts every draw to a regex or JSON schema
    (`dew.sampling.guided`). Each row keeps its own automaton state, and the
    tokens the automaton forbids get the score -inf before the transform
    chain runs. So the chain filters and samples within the language, and
    the raw likelihood is still the model's own.
    """

    grammar: Grammar | None = None

    def __call__(self, state: DecoderState, start: StepState, ops: DecodeOps,
                 transform: Callable[[StepState, jax.Array], jax.Array],
                 stopping: Callable[[StepState, jax.Array], jax.Array],
                 budget: int, n: int) -> Draws:
        return continuations(start, n, lambda drawn: _sample_rows(
            state, drawn, ops, transform, stopping, budget, self.grammar))


def _sample_rows(state: DecoderState, start: StepState, ops: DecodeOps,
                 transform: Callable[[StepState, jax.Array], jax.Array],
                 stopping: Callable[[StepState, jax.Array], jax.Array],
                 budget: int, grammar: Grammar | None = None) -> Draws:
    """The fixed-trip decode scan from one prefilled state, guided when a grammar is given."""

    def step(carry, _):
        state, step_state, terminated, automaton = carry
        active = step_state.active
        chain = transform if grammar is None else grammar.guiding(transform, automaton)
        token, behavior, raw = draw(step_state, state.logits, chain)
        committed = step_state.commit(token, active)
        stopped = active & stopping(committed, token)
        following = ops.advance(state, token, active)
        carry = (following, dataclasses.replace(committed, active=active & ~stopped),
                 terminated | stopped,
                 None if grammar is None else grammar.advanced(automaton, token, active))
        return carry, (token, active, jnp.where(active, behavior, 0.0), jnp.where(active, raw, 0.0))

    initial = (state, start, jnp.zeros(start.rows, bool),
               None if grammar is None else grammar.start(start.rows))
    (_, _, terminated, _), columns = lax.scan(step, initial, None, length=budget)
    tokens, valid, behavior, raw = (jnp.swapaxes(value, 0, 1) for value in columns)
    return Draws(tokens, valid, behavior, raw, terminated)


DEAD = -1.0e9
"""The score that takes a beam out of a selection, as `utils.py` uses."""


@struct.dataclass
class Completed:
    """A prompt's completed hypotheses, best score first."""

    score: jax.Array
    tokens: jax.Array
    raw: jax.Array
    flag: jax.Array
    length: jax.Array
    terminated: jax.Array


@struct.dataclass
class Beam:
    """Runs deterministic beam search over one shared prefill.

    The bookkeeping follows `_beam_search` in Transformers 5.16.1. Each step
    scores every live beam's continuations and keeps the best `keep` of
    them, `max(2, 1 + stop_ids) * width`, so that `width` live beams always
    remain. `stop_ids` is the number of tokens that can end a beam. The kept
    continuations that a stopping criterion ended move into the completed
    set, with their score divided by their generated length raised to
    `length_penalty`, and the search continues with the rest.

    `early_stopping` takes the reference's three settings, which decide when
    a prompt is no longer worth extending from an estimate of the best score
    its live beams can still reach. False estimates it from the current
    length. True does the same and also stops recording completed beams once
    every slot of the completed set is full. "never" estimates it from the
    whole budget when the penalty rewards length.

    The prompt is prefilled once and its cache row is copied into `width`
    rows. Every step gathers each beam's parent row through
    `DecodeOps.reindex`, so a branched beam decodes exactly like a
    separately selected prefix. Parameters are never mapped.

    `n` is how many completed beams to return, which is separate from the
    search width, and an `n` larger than `width` raises `ValueError`. A
    selected path is a search result and not a draw, so its behaviour log
    probability is zero. The raw log probabilities are still the model's own
    for the tokens on the path.
    """

    width: int = struct.field(pytree_node=False, default=1)
    length_penalty: float = struct.field(pytree_node=False, default=1.0)
    early_stopping: bool | str = struct.field(pytree_node=False, default=False)
    stop_ids: int = struct.field(pytree_node=False, default=1)

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 1:
            raise ValueError("beam width must be a positive integer")
        if self.early_stopping not in (True, False, "never"):
            raise ValueError("early_stopping is True, False or 'never'")
        if type(self.stop_ids) is not int or self.stop_ids < 0:
            raise ValueError("stop_ids counts the tokens that can end a beam")

    @property
    def keep(self) -> int:
        """The number of continuations a step keeps, as `beams_to_keep` upstream."""
        return max(2, 1 + self.stop_ids) * self.width

    def __call__(self, state: DecoderState, start: StepState, ops: DecodeOps,
                 transform: Callable[[StepState, jax.Array], jax.Array],
                 stopping: Callable[[StepState, jax.Array], jax.Array],
                 budget: int, n: int) -> Draws:
        if n > self.width:
            raise ValueError(f"beam search returns at most its width; asked for {n} of {self.width}")
        return _beam_search(state, start, ops, transform, stopping, budget, n, self)


def _pick(leaf, index):
    """`leaf[:, index]` per prompt, over a `[prompts, count, ...]` leaf."""
    return jnp.take_along_axis(leaf, index.reshape(index.shape + (1,) * (leaf.ndim - 2)),
                               axis=1)


def _beam_rows(parent, prompts: int, width: int):
    """Cache rows of a `[prompts, count]` parent-beam index."""
    return (parent + jnp.arange(prompts)[:, None] * width).reshape(-1)


def _beam_continue(state: DecoderState, beams: StepState, ops: DecodeOps, parent, token,
                   forward, real, prompts: int, width: int):
    """The `width` beams the step carries on with, each parent's decode state
    gathered onto its child's row before the child's token is committed."""
    parents = _pick(parent, forward)
    selected = _pick(token, forward).reshape(-1)
    state = ops.reindex(state, _beam_rows(parents, prompts, width))
    beams = jax.tree.map(lambda leaf: jnp.take(leaf, _beam_rows(parents, prompts, width), axis=0),
                         beams)
    return state, beams.commit(selected, real), selected


def _completed(done: Completed, search: Beam, top, extending, position: int, *,
               best, hit, ended, tokens, raw) -> Completed:
    """The `width` best completed hypotheses, after this step's arrivals.

    A candidate joins the set at its length-penalized score, and only the
    ones a criterion ended this step, in the live half of the ranking, can
    join at all. `early_stopping` True stops recording once every slot is
    full, and a prompt that is no longer worth extending records nothing.
    """
    prompts, width, keep = done.score.shape[0], search.width, search.keep
    normalized = best / jnp.power(position + 1.0, search.length_penalty)
    blocked = jnp.all(done.flag, axis=-1, keepdims=True) & (search.early_stopping is True)
    normalized = normalized + (blocked | ~extending).astype(jnp.float32) * DEAD
    normalized = normalized + (~(hit & top)).astype(jnp.float32) * DEAD
    merged = jnp.concatenate([done.score, normalized], axis=1)
    order = lax.top_k(merged, width)[1]
    return Completed(
        score=jnp.take_along_axis(merged, order, axis=1),
        tokens=_pick(jnp.concatenate([done.tokens, tokens], axis=1), order),
        raw=_pick(jnp.concatenate([done.raw, raw], axis=1), order),
        flag=jnp.take_along_axis(jnp.concatenate([done.flag, hit & top], axis=1), order, axis=1),
        length=jnp.take_along_axis(
            jnp.concatenate([done.length,
                             jnp.full((prompts, keep), position + 1, jnp.int32)], axis=1),
            order, axis=1),
        terminated=jnp.take_along_axis(
            jnp.concatenate([done.terminated, ended & top], axis=1), order, axis=1))


class _BeamCarry(NamedTuple):
    """The live beams, their model state and scores, and the completed hypotheses."""

    state: DecoderState
    beams: StepState
    live: jax.Array
    drawn: jax.Array
    scored: jax.Array
    extending: jax.Array
    done: Completed


def _beam_search(state: DecoderState, start: StepState, ops: DecodeOps,
                 transform: Callable[[StepState, jax.Array], jax.Array],
                 stopping: Callable[[StepState, jax.Array], jax.Array],
                 budget: int, n: int, search: Beam) -> Draws:
    """`Beam`'s scan: one step per position, keeping `width` live beams.

    The carry is the model state, the beams' `StepState`, each beam's running
    score, the tokens and raw log probabilities drawn so far, whether the
    prompt is still worth extending, and the completed set.
    """
    prompts, width, keep = start.rows, search.width, search.keep
    penalty, never = search.length_penalty, search.early_stopping == "never"
    real = jnp.repeat(start.active, width)
    # The request's own budget, which the scan's `budget` trips can exceed:
    # a hypothesis ends at it and none longer is recorded.
    limit = start.budgets()[:, None]
    top = (jnp.arange(keep) < width)[None, :]

    def step(carry: _BeamCarry, position):
        best, index, chosen, vocab = _beam_candidates(carry.state, carry.beams, transform, carry.live, real,
                                                      carry.extending, prompts, width, keep)
        parent, token = index // vocab, (index % vocab).astype(jnp.int32)

        branch = _beam_rows(parent, prompts, width)
        candidates = jax.tree.map(lambda leaf: jnp.take(leaf, branch, axis=0), carry.beams)
        flat = token.reshape(-1)
        ended = stopping(candidates.commit(flat, jnp.repeat(start.active, keep)), flat)
        ended = ended.reshape(prompts, keep) & jnp.repeat(start.active, keep).reshape(prompts, keep)
        hit = ended | (position + 1 >= limit)

        grown = jnp.take(carry.drawn, branch, axis=0).reshape(prompts, keep, budget)
        grown = grown.at[:, :, position].set(token)
        traced = jnp.take(carry.scored, branch, axis=0).reshape(prompts, keep, budget)
        traced = traced.at[:, :, position].set(chosen)

        alive = best + hit.astype(jnp.float32) * DEAD
        forward = lax.top_k(alive, width)[1]
        state, beams, selected = _beam_continue(carry.state, carry.beams, ops, parent, token, forward,
                                                real, prompts, width)

        done = _completed(carry.done, search, top, carry.extending & (position < limit), position,
                          best=best, hit=hit, ended=ended, tokens=grown, raw=traced)

        live = _pick(alive, forward)
        # `never` estimates the best score still reachable from the whole
        # budget where the penalty rewards length; otherwise from here.
        reach = limit.astype(jnp.float32) if never and penalty > 0 else (position + 1.0)
        worst = jnp.where(done.flag, jnp.min(done.score, axis=1, keepdims=True), DEAD)
        extending = carry.extending & jnp.any(live[:, :1] / jnp.power(reach, penalty) > worst,
                                axis=-1, keepdims=True)
        state = ops.advance(state, selected, real)
        drawn = _pick(grown, forward).reshape(prompts * width, budget)
        scored = _pick(traced, forward).reshape(prompts * width, budget)
        return _BeamCarry(state, beams, live, drawn, scored, extending, done), None

    initial = _beam_start(state, start, ops, prompts, width, budget)
    finished, _ = lax.scan(step, initial, jnp.arange(budget))
    return _beam_draws(finished.done, start, prompts, n, budget)


def _beam_candidates(state: DecoderState, beams: StepState, transform, live, real, extending,
                     prompts: int, width: int, keep: int):
    """The best `keep` continuations of the live beams, over all of them.

    Returns each candidate's running score, the flat `beam * vocab` index it
    came from, the model's own log probability of its token, and the
    vocabulary that index is read against. A beam that is real, still scored
    and worth extending has to hold a distribution; one that is not is
    refused rather than searched.
    """
    genuine = real.reshape(prompts, width) & (live > DEAD / 2) & extending
    checkify.check(jnp.all(wellformed(state.logits.astype(jnp.float32)).reshape(
        prompts, width) | ~genuine),
        "the model produced a live beam without a distribution to score")
    raw = jax.nn.log_softmax(state.logits.astype(jnp.float32))
    scores = transform(beams, raw)
    checkify.check(jnp.all(wellformed(scores).reshape(prompts, width) | ~genuine),
                   "the transform chain left a live beam without a distribution to search")
    vocab = scores.shape[-1]
    best, index = lax.top_k((scores.reshape(prompts, width, vocab)
                             + live[:, :, None]).reshape(prompts, width * vocab), keep)
    return best, index, jnp.take_along_axis(raw.reshape(prompts, width * vocab), index, axis=1), vocab


def _beam_start(state: DecoderState, start: StepState, ops: DecodeOps, prompts: int,
                width: int, budget: int) -> _BeamCarry:
    """The carry the search scans from: `width` copies of each prompt.

    Only beam zero starts alive, so the first step's continuations all come
    from the one prefilled row and the search does not begin by scoring
    `width` copies of the same distribution.
    """
    return _BeamCarry(
        ops.reindex(state, jnp.repeat(jnp.arange(prompts), width)),
        jax.tree.map(lambda leaf: jnp.repeat(leaf, width, axis=0), start),
        jnp.broadcast_to(jnp.where(jnp.arange(width) == 0, 0.0, DEAD), (prompts, width)),
        jnp.zeros((prompts * width, budget), jnp.int32),
        jnp.zeros((prompts * width, budget), jnp.float32),
        jnp.ones((prompts, 1), bool),
        Completed(jnp.full((prompts, width), DEAD, jnp.float32),
                  jnp.zeros((prompts, width, budget), jnp.int32),
                  jnp.zeros((prompts, width, budget), jnp.float32),
                  jnp.zeros((prompts, width), bool),
                  jnp.zeros((prompts, width), jnp.int32),
                  jnp.zeros((prompts, width), bool)))


def _beam_draws(done: Completed, start: StepState, prompts: int, n: int, budget: int) -> Draws:
    """The best `n` completed hypotheses per prompt, as output rows."""
    lengths = jnp.where(start.active[:, None], done.length, 0)[:, :n]
    valid = jnp.arange(budget)[None, None, :] < lengths[:, :, None]
    return Draws(done.tokens[:, :n].reshape(prompts * n, budget),
                 valid.reshape(prompts * n, budget),
                 jnp.zeros((prompts * n, budget), jnp.float32),
                 jnp.where(valid, done.raw[:, :n], 0.0).reshape(prompts * n, budget),
                 (done.terminated[:, :n] & start.active[:, None]).reshape(prompts * n))


@struct.dataclass
class Speculative:
    """Drafts with the model's prediction depths or block drafter, and verifies with the model itself.

    The acceptance rule is algorithm 1 of arXiv 2211.17192, as
    `_speculative_sampling` in Transformers 5.16.1 applies it. The first
    candidate of a block is an ordinary target draw, so it is always
    accepted. The model's prediction depths chain the remaining candidates
    from the target's last hidden state and each candidate's embedding, as
    vLLM's `Qwen3_5MultiTokenPredictor` does. A block drafter (DeepSeek-V4.1's
    DSpark) instead drafts all the candidates after the first in one pass,
    drawing each from its position's logits as the pass reaches it.

    A proposed token `x` is accepted with probability `min(1, p(x) / q(x))`,
    where `p` is the target's post-transform distribution and `q` is the
    draft's actual one, compared as a log ratio. At the first rejection, the
    replacement is drawn from the normalized positive part of `p - q`, and a
    block with nothing rejected draws a bonus token from `p`. So the emitted
    tokens have exactly the distribution `Sample` would give them, token for
    token, though not the same draws at one seed.

    Every emitted token, including a replacement or a bonus, records the
    target's post-transform log probability as its behaviour log probability
    and the model's own log probability as its raw value. The draft's `q`,
    the acceptance probability and the residual are never recorded, because
    none of them is the distribution the emitted token came from.

    Each iteration proposes `block` candidates (at least 2), so every
    collective keeps the same size. The target cache is saved before the
    block and the accepted prefix is replayed into it, because a recurrent
    mixer's state is a running summary that no cursor can rewind. The
    prediction cache is rebuilt the same way. A block that continues emits
    two or more tokens unless the budget ends first, so `ceil(budget / 2)`
    iterations bound the loop.

    `confidence` stops the draft after the first candidate to which the
    draft itself gives a probability below `confidence`, as the reference's
    `ConfidenceCriteria` does. The later candidates are still computed, at
    the same shapes, but cannot be accepted. Calling the strategy on a model
    that has neither prediction depths nor a block drafter raises
    `ValueError`.
    """

    block: int = struct.field(pytree_node=False, default=4)
    confidence: float = struct.field(pytree_node=False, default=0.0)

    def __post_init__(self) -> None:
        if type(self.block) is not int or self.block < 2:
            raise ValueError("a speculative block proposes at least two candidates")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence is a probability")

    def __call__(self, state: DecoderState, start: StepState, ops: DecodeOps,
                 transform: Callable[[StepState, jax.Array], jax.Array],
                 stopping: Callable[[StepState, jax.Array], jax.Array],
                 budget: int, n: int) -> Draws:
        depths = ops.propose is not None and ops.embed is not None and ops.depths
        if ops.verify is None or not (depths or ops.draft is not None):
            raise ValueError("speculative decoding drafts with the model's prediction depths or "
                             "its block drafter, and this model has neither; load a checkpoint "
                             "with MTP or DSpark weights or choose another strategy")
        return continuations(start, n, lambda drawn: _speculate(
            state, drawn, ops, transform, stopping, budget, self))


def _coordinates(state: DecoderState, step: StepState, slots: jax.Array) -> jax.Array:
    """The logical coordinate of each slot a block may emit.

    Without supplied coordinates the count of real tokens is what the cache
    assigns; a drawn token sits at the same coordinate on every axis.
    """
    base = step.total() if state.positions is None else state.positions
    return base[:, None] + slots


class _Drafted(NamedTuple):
    """What one drafting pass leaves for the verification that follows it.

    `candidates` is the block's proposed tokens, `scores` the distribution
    each was drawn from, and `steps` the `StepState` at each slot. `offered`
    marks the candidates the confidence criterion still let the draft make,
    and `ending` whether a stopping criterion fired on each of them.
    """

    state: DecoderState
    candidates: list[jax.Array]
    scores: list[jax.Array]
    steps: list[StepState]
    offered: list[jax.Array]
    ending: list[jax.Array]


def _drafted(plan: Speculative, ops: DecodeOps, state: DecoderState, step: StepState,
             keys: jax.Array, base: jax.Array, active: jax.Array, budget: int,
             transform, stopping) -> _Drafted:
    """One block's candidates: an ordinary target draw, then the draft's,
    chained through the prediction depths or drawn as a block drafter's pass
    reaches each position."""
    block_size, rows = plan.block, step.rows

    def asked(at):
        """Rows genuinely drawing at slot `at`, not walking past the budget."""
        return active & (step.step + at < budget)

    opening = dataclasses.replace(step, active=asked(0))
    scores = [transform(opening, state.logits.astype(jnp.float32))]
    candidates = [select(keys[:, 0], scores[0], opening.active)]
    offered = [jnp.ones(rows, bool)]
    states = [opening, opening.commit(candidates[0], active)]
    live, sure, ending = [opening.active], [jnp.ones(rows, bool)], []

    def reached(depth: int) -> jax.Array:
        """The rows still drafting at `depth`, once the criteria saw the candidate before it."""
        ending.append(stopping(states[depth], candidates[depth - 1]))
        live.append(live[-1] & ~ending[depth - 1] & asked(depth))
        states[depth] = dataclasses.replace(states[depth], active=live[-1])
        return live[-1]

    def drawn_at(depth: int, logits: jax.Array) -> jax.Array:
        """Candidate `depth`, drawn from the draft's logits for it."""
        drawn = transform(states[depth], logits.astype(jnp.float32))
        token = select(keys[:, depth], drawn, live[-1])
        scores.append(drawn)
        candidates.append(token)
        offered.append(sure[-1])
        sure.append(sure[-1] & (jnp.exp(token_log_probs(drawn, token)) >= plan.confidence))
        states.append(states[depth].commit(token, active))
        return token

    drafting = state
    if ops.draft is not None:
        def choose(index: int, logits: jax.Array) -> jax.Array:
            depth = index + 1
            if depth >= block_size:
                return jnp.argmax(logits, axis=-1)
            reached(depth)
            return drawn_at(depth, logits)

        ops.draft(state, candidates[0], choose)
        if len(candidates) < block_size:
            raise ValueError(f"the model's block drafter proposes {len(candidates) - 1} tokens a "
                             f"pass, fewer than a block of {block_size} candidates needs")
    else:
        propose = ops.propose
        assert propose is not None and state.hidden is not None, \
            "a drafting pass reads the last hidden state"
        hidden = state.hidden
        for depth in range(1, block_size):
            drafting, logits, produced = propose(
                drafting, hidden[:, None], candidates[depth - 1][:, None], None, reached(depth)[:, None],
                base[:, depth - 1][:, None], (depth - 1) % ops.depths, "draft")
            hidden = produced[:, 0]
            drawn_at(depth, logits[:, 0])
    # Every candidate gets the real criterion, the last one included: a
    # block accepted whole must not draw its bonus behind a stop.
    ending.append(stopping(states[block_size], candidates[block_size - 1]))
    return _Drafted(drafting, candidates, scores, states, offered, ending)


def _accepted(block_size: int, keys: jax.Array, targets: list[jax.Array], drafted: _Drafted,
              step: StepState, active: jax.Array, budget: int) -> tuple[jax.Array, jax.Array]:
    """How many candidates the block keeps, and the token it ends on.

    A candidate the draft never offered was not rejected, so the block ends
    on an ordinary target draw there; only a rejected offer draws from the
    positive part of `p - q`, the distinction algorithm 1 rests on.
    """
    candidates, drafts, offered, ending = (drafted.candidates, drafted.scores,
                                           drafted.offered, drafted.ending)
    accepted = []
    for at in range(1, block_size):
        ratio = (token_log_probs(targets[at], candidates[at])
                 - token_log_probs(drafts[at], candidates[at]))
        uniform = jax.vmap(jax.random.uniform)(keys[:, block_size + at])
        accepted.append((jnp.log(uniform) <= ratio) & offered[at])
    available = 1 + sum(offered_flag.astype(jnp.int32) for offered_flag in offered[1:])
    matched = (1 + jnp.sum(jnp.cumprod(jnp.stack(accepted, axis=1), axis=1), axis=1)).astype(jnp.int32)
    turned_down = matched < available
    target = jnp.take_along_axis(jnp.stack(targets, axis=1), matched[:, None, None], axis=1)[:, 0]
    draft = jnp.take_along_axis(jnp.stack(drafts, axis=1),
                                jnp.minimum(matched, block_size - 1)[:, None, None], axis=1)[:, 0]
    residual = jnp.maximum(jax.nn.softmax(target) - jax.nn.softmax(draft), 0.0)
    weight = jnp.sum(residual, axis=-1, keepdims=True)
    rest = jnp.log(residual / jnp.where(weight > 0, weight, 1.0))
    stopped = jnp.any(jnp.stack(ending, axis=1)
                      & (jnp.arange(block_size)[None, :] < matched[:, None]), axis=1)
    replacement = select(keys[:, 2 * block_size], jnp.where(turned_down[:, None], rest, target),
                         active & ~stopped & (step.step + matched < budget))
    return matched, replacement


def _emission(block_size: int, slots: jax.Array, matched: jax.Array, replacement: jax.Array,
              proposed: jax.Array, targets: list[jax.Array], raw: list[jax.Array],
              rows: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The block's tokens and the two log probabilities recorded for each,
    the target's post-transform and the model's own (`Speculative`)."""
    emitted = jnp.where(slots < matched[:, None],
                        jnp.concatenate([proposed, jnp.zeros((rows, 1), jnp.int32)], axis=1),
                        replacement[:, None])
    behavior = jnp.stack([token_log_probs(targets[at], emitted[:, at])
                          for at in range(block_size + 1)], axis=1)
    original = jnp.stack([token_log_probs(raw[at], emitted[:, at])
                          for at in range(block_size + 1)], axis=1)
    return emitted, behavior, original


def _emitted_count(block_size: int, step: StepState, emitted: jax.Array, active: jax.Array,
                   terminated: jax.Array, budget: int, stopping,
                   matched: jax.Array) -> tuple[jax.Array, jax.Array]:
    """How many of the block's slots each row keeps, and which rows ended.

    The accepted prefix and the token after it are walked one slot at a time
    so every criterion sees the history it would have seen in an ordinary
    loop; the row stops at the first slot one fires on.
    """
    count = jnp.minimum(matched + 1, budget - step.step)
    walked, hits = step, []
    for at in range(block_size + 1):
        walked = walked.commit(emitted[:, at], active & (at < count))
        hits.append(stopping(walked, emitted[:, at]) & active & (at < count))
    firing = jnp.stack(hits, axis=1)
    anywhere = jnp.any(firing, axis=1)
    count = jnp.where(anywhere, jnp.minimum(count, jnp.argmax(firing, axis=1) + 1), count)
    count = jnp.where(active, count, 0)
    return count, terminated | (anywhere & (count == jnp.argmax(firing, axis=1) + 1))


def _recorded(block_size: int, step: StepState, emitted: jax.Array, behavior: jax.Array,
              original: jax.Array, count: jax.Array, slots: jax.Array, index: jax.Array,
              active: jax.Array, terminated: jax.Array, budget: int, out):
    """The kept slots, the `StepState` they leave, and the output rows.

    Each kept slot lands at its own response position, so a block that
    emitted two tokens writes two columns and the rest of the block writes
    none. A row that ran out of budget or ended stops being active.
    """
    keep = slots < count[:, None]
    committed = step
    for at in range(block_size + 1):
        committed = committed.commit(emitted[:, at], keep[:, at])
    committed = dataclasses.replace(
        committed, active=active & ~terminated & (step.step + count < budget))
    landing = jnp.where(keep, step.step[:, None] + slots, DROPPED)
    return keep, committed, _Emitted(
        out.tokens.at[index, landing].set(emitted, mode="drop"),
        out.valid.at[index, landing].set(True, mode="drop"),
        out.behavior_log_probs.at[index, landing].set(behavior, mode="drop"),
        out.raw_log_probs.at[index, landing].set(original, mode="drop"))


class _SpeculativeCarry(NamedTuple):
    """The target state and accepted history, row termination, and emitted response slots."""

    state: DecoderState
    step: StepState
    terminated: jax.Array
    out: _Emitted


def _block(carry: _SpeculativeCarry, plan: Speculative, ops: DecodeOps, transform, stopping, budget: int,
           slots: jax.Array, index: jax.Array):
    """One speculative block: draft `plan.block` candidates, verify them,
    accept the longest prefix the test allows, and emit what follows it,
    replaying the accepted prefix into the cache saved before the block."""
    verify, block_size = ops.verify, plan.block
    assert verify is not None
    state, step = carry.state, carry.step
    active, saved, rows = step.active, state.cache, step.rows
    base = _coordinates(state, step, slots)
    keys = jax.vmap(lambda key, count: jax.random.split(
        jax.random.fold_in(key, count), 2 * block_size + 1))(step.keys, step.step)

    drafted = _drafted(plan, ops, state, step, keys, base, active, budget,
                       transform, stopping)
    states, candidates = drafted.steps, drafted.candidates
    proposed = jnp.stack(candidates, axis=1)
    _, verified, _ = verify(
        drafted.state, proposed,
        jnp.stack([active & (step.step + at < budget) for at in range(block_size)], axis=1))
    raw = [state.logits.astype(jnp.float32)] + [verified[:, at].astype(jnp.float32)
                                                for at in range(block_size)]
    targets = [drafted.scores[0]] + [transform(states[at], raw[at])
                                     for at in range(1, block_size + 1)]

    matched, replacement = _accepted(block_size, keys, targets, drafted, step, active, budget)

    emitted, behavior, original = _emission(block_size, slots, matched, replacement, proposed,
                                            targets, raw, rows)
    count, terminated = _emitted_count(block_size, step, emitted, active, carry.terminated, budget,
                                       stopping, matched)
    checkify.check(
        jnp.all(jnp.stack([wellformed(raw[at]) | ~(at < count) for at in range(block_size + 1)])),
        "the model produced an emitted position without a distribution to score")
    keep, committed, out = _recorded(block_size, step, emitted, behavior, original, count,
                                     slots, index, active, terminated, budget, carry.out)

    following, again, seen = verify(dataclasses.replace(state, cache=saved), emitted, keep)
    assert seen is not None
    last = jnp.maximum(count - 1, 0)
    if ops.record is not None:
        following = ops.record(following, seen, keep)
    else:
        assert ops.embed is not None
        # The tails reseed hands back are every depth's predecessor at the
        # last emitted slot, the target's own state first; a row that
        # emitted nothing keeps what it had.
        following, carried = reseed(
            ops, following, (state.hidden, *state.drafts), seen, ops.embed(emitted),
            keep, base, last, prior_tokens=step.total())
        following = dataclasses.replace(following, hidden=carried[0], drafts=carried[1:])
    following = dataclasses.replace(
        following, logits=jnp.where((count > 0)[:, None],
                                    jnp.take_along_axis(again, last[:, None, None], axis=1)[:, 0],
                                    state.logits))
    return _SpeculativeCarry(following, committed, terminated, out), None


def _speculate(state: DecoderState, start: StepState, ops: DecodeOps,
               transform: Callable[[StepState, jax.Array], jax.Array],
               stopping: Callable[[StepState, jax.Array], jax.Array],
               budget: int, plan: Speculative) -> Draws:
    """`Speculative`'s loop of `_block`s. The carry is the model state, the
    `StepState` the accepted tokens leave behind, which rows ended, and the
    rows emitted so far."""
    block_size, rows = plan.block, start.rows
    slots = jnp.arange(block_size + 1)[None, :]
    index = jnp.arange(rows)[:, None]

    def outer(carry: _SpeculativeCarry, _):
        # Every rank reduces the same global mask, so the pool skips the same
        # blocks. A skipped block runs no model call at all, which is where
        # the saved target forwards come from.
        return lax.cond(jnp.any(carry.step.active),
                        lambda held: _block(held, plan, ops, transform, stopping, budget,
                                            slots, index),
                        lambda held: (held, None), carry)

    empty = _Emitted(jnp.zeros((rows, budget), jnp.int32), jnp.zeros((rows, budget), bool),
                     jnp.zeros((rows, budget), jnp.float32), jnp.zeros((rows, budget), jnp.float32))
    finished, _ = lax.scan(
        outer, _SpeculativeCarry(state, start, jnp.zeros(rows, bool), empty),
        None, length=-(-budget // 2))
    out = finished.out
    return Draws(out.tokens, out.valid,
                 jnp.where(out.valid, out.behavior_log_probs, 0.0),
                 jnp.where(out.valid, out.raw_log_probs, 0.0), finished.terminated)
