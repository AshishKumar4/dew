"""Direct preference optimization over chosen and rejected completions.

A `DPOObjective` scores tokens as `LMObjective` does (`TokenScorer`), and its
loss is the preference term from `dew.rl`: policy and reference log-probabilities
as negated per-token cross entropies, summed over each half's completion tokens,
through `preference_logsigmoid_terms`, averaged over pairs. The reference is the objective's own
frozen tree, `step.ema` at unit decay, so the run carries no second model.
Batches hold windows of packed pairs (`dew.data.preferences`): each pair one document, its halves'
shared prompt once with each half's rest continuing it (`dew.batch.Forks`), so a prompt is computed once
for both halves and several pairs fill a window; each pair's halves are summed apart (`PAIRS_KEY`,
`HALVES_KEY`), so shuffling or packing never fuses two pairs.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from dew.artifacts import TokenScores
from dew.batch import COORDINATES_FIELD, VALID_ROWS, Coordinates, Forks, Variables, coordinates
from dew.data.plan import PAIR_WINDOWS
from dew.data.preferences import (
    DOCUMENTS_KEY,
    HALVES_KEY,
    IDS_KEY,
    MASK_KEY,
    PAIRS_KEY,
    PARENTS_KEY,
    POSITIONS_KEY,
    REAL_KEY,
    window_shape,
)
from dew.inputs import Field, InputSpec
from dew.nn.inspection import watched_positions
from dew.objectives.base import Aux, EMASpec, Objective
from dew.objectives.routing import ExpertBalance
from dew.rl.surrogate import preference_logsigmoid_terms

from ..lm.objective import TokenScorer


def window_coordinates(batch, finals: int, pairs: int = 1) -> Coordinates | None:
    """A batch's windows laid out as the model reads them: each pair's documents, each token at its own
    half's place, a half's rest continuing the prompt it shares (`dew.data.preferences`), each half's
    final id in the windows' tail of `finals` (`Coordinates.finals`); halves laid out whole
    (`share=False`, no `PARENTS_KEY`) are documents of their own; a window of one half (no
    `DOCUMENTS_KEY`) is a plain row, its half first and padding after, None, as a model that reads one
    document a row takes it. A window holds at most `pairs` pairs, each prompt shared one document its
    halves continue (`Forks.held`)."""
    if DOCUMENTS_KEY not in batch:
        return None
    real = jnp.asarray(batch[REAL_KEY]) > 0
    positions = jnp.asarray(batch[POSITIONS_KEY], jnp.int32)
    held = batch.get(PARENTS_KEY)
    documents = jnp.asarray(batch[DOCUMENTS_KEY], jnp.int32)
    laid = replace(coordinates(segments=documents, valid=real, positions=positions,
                               max_segments=documents.shape[1] if held is None else jnp.shape(held)[1],
                               right_padded=True), finals=finals)
    # Each pair's prompt the one document its halves continue (`Forks.held`).
    forks = None if held is None else Forks(jnp.asarray(held, jnp.int32), depth=1, held=pairs)
    return laid if forks is None else replace(laid, forks=forks)


def _completions(batch, scored: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Each target `[B, W - 1]` (the token at `t + 1`'s at `t`) that counts in its chosen and in its
    rejected half's sum: a completion token its half marks that the scorer scores (`scored`, its weights:
    a token with a predecessor in its half; a half's first token has none)."""
    half = jnp.asarray(batch[HALVES_KEY], jnp.int32)[:, 1:]
    marked = (jnp.asarray(batch[MASK_KEY])[:, 1:] != 0) & (scored > 0)
    return marked & (half == 1), marked & (half == 2)


def pair_numbers(batch, halved: bool = False) -> tuple[jax.Array, int]:
    """Each token's pair among the batch's windows `[R, W]`, from 1, 0 on padding, and the most pairs a
    batch of its shape holds: each window's own pairs (`PAIRS_KEY`, numbered from 1 in the window, its
    tail's finals by their pair's number) after the windows' before it; where `halved`, each odd window
    the rejected half of the pair the one before it holds the chosen of (`DPOObjective.windows`, a pair's
    halves a window each), the pair its two windows' own whether or not they hold one. Numbered from the
    windows a step reads alone, so rows stacked from several batches or repeated to fill one stay pairs of
    their own."""
    local = jnp.asarray(batch[PAIRS_KEY], jnp.int32)
    rows, width = local.shape
    count = rows * max(width // 2, 1)
    real = local > 0
    if halved:
        # A pair's two windows its own number, a pair a row as the batch held it.
        return jnp.where(real, (jnp.arange(rows, dtype=jnp.int32) // 2 + 1)[:, None], 0), count
    # The pairs a window opens, and the last number before it.
    opened = jnp.max(local, axis=1)
    return jnp.where(real, (jnp.cumsum(opened) - opened)[:, None] + local, 0), count


def _halves(values: jax.Array, batch, scored: jax.Array, halved: bool) -> tuple[jax.Array, jax.Array,
                                                                                 jax.Array]:
    """Each of the batch's pairs' chosen and rejected sums of the per-target `values` `[B, W - 1]` (the
    token at `t + 1`'s at `t`) over its completion tokens the scorer scores (`_completions`), and each
    pair's weight, the window's it opens in (`Objective.row_weights`), 0 for a number no pair takes; each
    `[P]`, a pair by its number (`pair_numbers`, `halved`)."""
    numbers, count = pair_numbers(batch, halved)
    chosen_marks, rejected_marks = _completions(batch, scored)
    targets = numbers[:, 1:]

    def summed(weights, held):
        # A value outside the half's completion counts nothing, a NaN there neither.
        kept = jnp.where(weights, held, 0).reshape(-1)
        return jax.ops.segment_sum(kept, jnp.where(weights, targets, 0).reshape(-1),
                                   num_segments=count + 1)[1:]

    weights = Objective.row_weights(batch, numbers.shape[0]).astype(values.dtype)
    # A pair's weight is the window's it opens in, found by index: a number no pair takes has none.
    held = jnp.broadcast_to(jnp.arange(numbers.shape[0], dtype=jnp.int32)[:, None], numbers.shape)
    first = jax.ops.segment_min(held.reshape(-1), numbers.reshape(-1), num_segments=count + 1)[1:]
    taken = first < numbers.shape[0]
    weight = jnp.where(taken, weights[jnp.where(taken, first, 0)], 0)
    return summed(chosen_marks, values), summed(rejected_marks, values), weight


class DPOObjective(TokenScorer):
    """Trains a policy on preference pairs with the DPO loss (arXiv:2305.18290, equation 7).

    The reference is the starting policy, frozen: the objective's average at
    unit decay. `beta` is the KL strength and must be positive. `model`,
    `seq_len` and the scoring `options` are `TokenScorer`'s, with `seq_len`
    one less than a half's longest; a window holds two halves' tokens, several
    pairs packed (`dew.data.preferences.windows`), or with `segments=1`, for a
    model that reads one document a row, a pair a row, its halves each a
    window on an axis of the pair's (`[B, 2, W]`, `PreferencePair`), so whatever
    deals the rows deals whole pairs. The loss reads both as windows `[R, W]`
    (`windows`) and reduces each pair's halves by its number across them
    (`pair_numbers`), one path. The completion mask says which targets
    count. Validation scores the chosen responses' perplexity under the
    policy.

    `balance` trains the routers of a mixture as `ExpertBalance` describes
    (`dew.objectives.routing`), over the completion tokens each pair's forward
    reads: both halves of every pair route, and the counts sum over them.
    """

    ema_is_reference = True
    counts_tokens = False
    """Preference pairs need their windows' geometry, not LM's text rows."""

    keeps_whole_logits = False

    scored_fields = None
    """Validation scores the chosen responses of the pairs alone, for their perplexity, reading the windows'
    layout keys whole (`_evaluation_scores`)."""

    def __init__(self, model, seq_len: int, beta: float = 0.1, *, balance: ExpertBalance | None = None,
                 segments: int | None = None, pairs: int = 1, **options):
        if beta <= 0:
            raise ValueError(f"beta scales the KL term, so it is positive, got {beta}")
        if segments not in (None, 1):
            raise ValueError(f"segments=1 lays a pair a row, its halves a window each; got {segments}")
        if pairs < 1:
            raise ValueError(f"a window holds at least one pair; got pairs={pairs}")
        # A token's place is its own half's, at most `seq_len`, which the model's positions cover.
        super().__init__(model, seq_len, balance=balance, **options)
        self.ema = EMASpec.constant(1.0)
        self.beta = beta
        self.pairs = pairs
        self._stated = segments
        self._lay(segments)

    def _lay(self, segments: int | None) -> None:
        """Read batches laid out with `segments` (`segments=1` a pair a row, else windows of packed pairs):
        the window's inputs (`context`) and finals, and the batch it takes."""
        inputs, self.finals = window_shape(self.seq_len + 1, pairs=self.pairs, segments=segments)
        # A window reads its inputs, its finals a tail of targets no input reads (`Coordinates.finals`).
        self.context = inputs
        width = inputs + self.finals
        # The batch is windows of packed pairs, two halves' tokens a window, or a pair a row, its halves
        # a window each (`segments=1`); windows of pairs a caller holds are as wide as their pairs read
        # (`preferences.windows`), at most this.
        self.inputs = InputSpec(sample=Field(IDS_KEY, (2, width)) if segments == 1
                                else Field(IDS_KEY, (width,), variable_axes=(0,)))
        self.segments = segments

    def bind_plan(self, version: int) -> None:
        """Read its batches as data plan `version` lays them out: a plan from before preference pairs packed
        into windows (`dew.data.plan.PAIR_WINDOWS`) lays a pair a row where the run states no `segments`,
        so the objective reads a pair a row (`segments=1`, which `traced` reads), as the run it resumes
        did; a later plan, as stated."""
        self._lay(1 if version < PAIR_WINDOWS and self._stated is None else self._stated)

    @property
    def halved(self) -> bool:
        """Whether a batch holds a pair a row, its halves a window each (`segments=1`)."""
        return self.segments == 1

    def windows(self, batch):
        """The batch as the windows `[R, W]` the loss reads, each its inputs, at most the objective's
        (`context`), and its finals' tail: a pair a row (`halved`) as its two windows in turn, each row's
        weight its windows' (`VALID_ROWS`)."""
        ids = jnp.asarray(batch[IDS_KEY])
        if ids.ndim != (3 if self.halved else 2) or (self.halved and ids.shape[1] != 2) or (
                ids.shape[-1] <= self.finals):
            shape = "batch, 2, W" if self.halved else "batch, W"
            raise ValueError(f"a DPO batch holds windows of packed pairs [{IDS_KEY} shape ({shape}), W past "
                             f"its {self.finals} finals] (dew.data.preferences), got {tuple(ids.shape)}")
        # A pair a row holds its halves' ids and marks alone (`PreferencePair`); its pair, halves and places
        # are its axes'.
        held = ((IDS_KEY, MASK_KEY, REAL_KEY) if self.halved else
                (IDS_KEY, MASK_KEY, REAL_KEY, PAIRS_KEY, HALVES_KEY, POSITIONS_KEY,
                 *((DOCUMENTS_KEY,) if DOCUMENTS_KEY in batch else ())))
        for key in held:
            if jnp.shape(batch[key]) != ids.shape:
                raise ValueError(f"{key} has shape {tuple(jnp.shape(batch[key]))} for {tuple(ids.shape)} "
                                 f"{IDS_KEY}; a window's fields hold one entry a token")
        if not self.halved:
            return batch
        rows = ids.shape[0]
        real = jnp.asarray(batch[REAL_KEY]) > 0
        axes = {PAIRS_KEY: real.astype(jnp.int32),
                HALVES_KEY: jnp.where(real, jnp.arange(1, 3, dtype=jnp.int32)[None, :, None], 0),
                POSITIONS_KEY: jnp.where(real, jnp.arange(ids.shape[-1], dtype=jnp.int32), 0)}
        laid = {key: jnp.reshape(value, (2 * rows, -1))
                for key, value in {**{key: jnp.asarray(batch[key]) for key in held}, **axes}.items()}
        if VALID_ROWS in batch:
            laid[VALID_ROWS] = jnp.repeat(jnp.asarray(batch[VALID_ROWS]).reshape(rows), 2)
        return {**batch, **laid}

    traced = ("seq_len", "segments", "pairs", "balance", "head_chunks", "head_tile", "head_fused", "beta")
    """Its window, balance, head policy and its beta; its reference is the step's EMA, an argument."""

    def loss(self, variables, batch, step):
        """Compute the preference term over each pair's completion tokens.

        It also reports `rewards/chosen`, `rewards/rejected` and `accuracy`,
        the fraction of pairs whose chosen reward is higher.
        """
        statistics, aux, _ = self._preference(variables, batch, step)
        return statistics, aux

    def _preference(self, variables, batch, step, *, predict: bool | None = None):
        """The preference loss and its reports, with the policy's scores of the windows behind them
        (`token_scores`' `predict`)."""
        if step.ema is None:
            raise ValueError(
                "the DPO reference reads step.ema, but the objective keeps no EMA; "
                "a DPO run always freezes one")
        batch = self.windows(batch)
        ids = jnp.asarray(batch[IDS_KEY])
        laid = window_coordinates(batch, self.finals, self.pairs)
        # The policy and its reference score the windows in one forward each, through the same function,
        # so a policy equal to its reference earns zero rewards wherever the two compile alike.
        real = jnp.asarray(batch[REAL_KEY])[:, :-self.finals] > 0
        with watched_positions(real if step.watched is not None else None):
            policy = self.token_scores(variables, ids, coordinates=laid, qk_stats=step.qk_stats,
                                       predict=predict, progress=self.progress(step),
                                       routing=self.balance is not None, watched=step.watched is not None)
        reference = self.token_scores(step.ema, ids, coordinates=laid, progress=self.progress(step))
        policy_chosen, policy_rejected, weights = _halves(-policy.losses, batch, policy.weights, self.halved)
        ref_chosen, ref_rejected, _ = _halves(-reference.losses, batch, policy.weights, self.halved)
        ones = jnp.ones((*weights.shape, 1), policy_chosen.dtype)
        terms, (pair_chosen, pair_rejected) = preference_logsigmoid_terms(
            policy_chosen[..., None], policy_rejected[..., None], ref_chosen[..., None],
            ref_rejected[..., None], ones, ones, self.beta)
        # The batch's pairs, a row of them, each weighed by its window's weight (`_halves`).
        counted = weights.astype(terms.dtype)[None]
        accuracy = self.accuracy((pair_chosen > pair_rejected).astype(jnp.float32)[None], {}, counted)
        statistics = self.row_mean(terms[None], {}, weights=counted)
        reported = {
            "rewards/chosen": self.row_mean(pair_chosen[None], {}, weights=counted).mean()[0],
            "rewards/rejected": self.row_mean(pair_rejected[None], {}, weights=counted).mean()[0],
            "accuracy": accuracy.mean()[0],
        }
        effects = None
        if self.balance is not None:
            statistics, effects, metrics = self.balance.compose(
                variables, statistics, policy.routing or {}, batch, self._routed_queries(batch),
                {} if laid is None else {COORDINATES_FIELD: laid})
            reported.update(metrics)
        return statistics, Aux[Variables](reported, qk_stats=policy.qk, effects=effects,
                                          inspected=policy.inspected), policy

    def _routed_queries(self, batch) -> jax.Array:
        """The router positions that count, `[B, W - finals]`: each half's completion tokens among the
        inputs, none in a repeat window (`VALID_ROWS`)."""
        mask = jnp.asarray(batch[MASK_KEY])
        rows = self.row_weights(batch, mask.shape[0]) > 0
        return (mask[:, :-self.finals] != 0) & rows[:, None]

    def _evaluation_scores(self, params, batch, step) -> TokenScores:
        """The chosen responses' per-token cross entropies, with their completion marks as weights."""
        laid = self.windows(batch)
        scores = self.token_scores(params, jnp.asarray(laid[IDS_KEY]), predict=True,
                                   coordinates=window_coordinates(laid, self.finals, self.pairs))
        assert scores.correct is not None
        return self._chosen(scores.losses, scores.correct, scores.weights, batch)

    def _validation_scores(self, params, batch, step):
        """The preference loss and the chosen responses' scores, from the
        policy's one scoring of the windows."""
        statistics, _, policy = self._preference(params, batch, step, predict=True)
        assert policy.correct is not None
        return statistics, self._chosen(policy.losses, policy.correct, policy.weights, batch)

    def _chosen(self, losses, correct, scored, batch) -> TokenScores:
        """The chosen halves' scores over the windows (`windows`), with their completion marks the scorer
        scores as weights, a row of `batch`'s: a pair's two windows end to end where `halved`."""
        chosen, _ = _completions(self.windows(batch), scored)
        scores = TokenScores(losses=losses, weights=chosen.astype(losses.dtype), correct=correct)
        if not self.halved:
            return scores
        rows = jnp.shape(batch[IDS_KEY])[0]
        return jax.tree.map(lambda leaf: jnp.reshape(leaf, (rows, -1)), scores)
