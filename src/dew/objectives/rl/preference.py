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
from dew.batch import COORDINATES_FIELD, Coordinates, Forks, Variables, coordinates
from dew.data.pipeline import ObjectiveData
from dew.data.preferences import (
    DOCUMENTS_KEY,
    HALVES_KEY,
    IDS_KEY,
    MASK_KEY,
    PAIRS_KEY,
    PARENTS_KEY,
    POSITIONS_KEY,
    REAL_KEY,
    PairIds,
    window_width,
)
from dew.inputs import Field, InputSpec
from dew.nn.inspection import watched_positions
from dew.objectives.base import Aux, EMASpec, Objective
from dew.objectives.routing import ExpertBalance
from dew.rl.surrogate import preference_logsigmoid_terms

from ..lm.objective import TokenScorer


def window_coordinates(batch) -> Coordinates | None:
    """A batch's windows laid out as the model reads them: each pair's documents, each token at its own
    half's place, a half's rest continuing the prompt it shares (`dew.data.preferences`); halves laid out
    whole (`share=False`, no `PARENTS_KEY`) are documents of their own; a window of one half (no
    `DOCUMENTS_KEY`) is a plain row, its half first and padding after, None, as a model that reads one
    document a row takes it."""
    if DOCUMENTS_KEY not in batch:
        return None
    real = jnp.asarray(batch[REAL_KEY]) > 0
    positions = jnp.asarray(batch[POSITIONS_KEY], jnp.int32)
    held = batch.get(PARENTS_KEY)
    documents = jnp.asarray(batch[DOCUMENTS_KEY], jnp.int32)
    laid = coordinates(segments=documents, valid=real, positions=positions,
                       max_segments=documents.shape[1] if held is None else jnp.shape(held)[1],
                       right_padded=True)
    return laid if held is None else replace(laid, forks=Forks(jnp.asarray(held, jnp.int32), depth=1))


def _completions(batch, scored: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Each target `[B, W - 1]` (the token at `t + 1`'s at `t`) that counts in its chosen and in its
    rejected half's sum: a completion token its half marks that the scorer scores (`scored`, its weights:
    a token with a predecessor in its half; a half's first token has none)."""
    half = jnp.asarray(batch[HALVES_KEY], jnp.int32)[:, 1:]
    marked = (jnp.asarray(batch[MASK_KEY])[:, 1:] != 0) & (scored > 0)
    return marked & (half == 1), marked & (half == 2)


def pair_numbers(batch) -> tuple[jax.Array, int]:
    """Each token's pair among the batch's `[B, W]`, from 1, 0 on padding, and the most pairs a batch of its
    shape holds: the pairs its windows number (`PAIRS_KEY`, each its own across the batch, `PairIds`)
    numbered again in their order within whatever part of the batch a step reads, so one reduction reads a
    pair's halves whether a window holds them both or each holds one; a step's parts hold whole pairs
    (`UNIT_ROWS`, `dew.training.planning.splits`)."""
    pair = jnp.asarray(batch[PAIRS_KEY], jnp.int32)
    rows, width = pair.shape
    count = rows * max(width // 2, 1)
    # Padding (0) ahead of every pair, so it numbers 0 whatever the part holds.
    flat = jnp.concatenate([jnp.zeros((1,), jnp.int32), pair.reshape(-1)])
    _, numbered = jnp.unique(flat, size=count + 1, fill_value=jnp.iinfo(jnp.int32).max, return_inverse=True)
    return numbered.reshape(-1)[1:].reshape(rows, width).astype(jnp.int32), count


def _halves(values: jax.Array, batch, scored: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each of the batch's pairs' chosen and rejected sums of the per-target `values` `[B, W - 1]` (the
    token at `t + 1`'s at `t`) over its completion tokens the scorer scores (`_completions`), and each
    pair's weight, its window's (`Objective.row_weights`) where it holds a pair, 0 for a number no pair
    takes; each `[P]`, a pair by its number (`pair_numbers`)."""
    numbers, count = pair_numbers(batch)
    chosen_marks, rejected_marks = _completions(batch, scored)
    targets = numbers[:, 1:]

    def summed(weights, held):
        # A value outside the half's completion counts nothing, a NaN there neither.
        kept = jnp.where(weights, held, 0).reshape(-1)
        return jax.ops.segment_sum(kept, jnp.where(weights, targets, 0).reshape(-1),
                                   num_segments=count + 1)[1:]

    rows = Objective.row_weights(batch, numbers.shape[0])
    weight = jax.ops.segment_max(jnp.broadcast_to(rows[:, None], numbers.shape).reshape(-1),
                                 numbers.reshape(-1), num_segments=count + 1)[1:]
    return summed(chosen_marks, values), summed(rejected_marks, values), jnp.maximum(weight, 0)


class DPOObjective(TokenScorer):
    """Trains a policy on preference pairs with the DPO loss (arXiv:2305.18290, equation 7).

    The reference is the starting policy, frozen: the objective's average at
    unit decay. `beta` is the KL strength and must be positive. `model`,
    `seq_len` and the scoring `options` are `TokenScorer`'s, with `seq_len`
    one less than a half's longest; a window holds two halves' tokens, several
    pairs packed (`dew.data.preferences.windows`), or with `segments=1`, for a
    model that reads one document a row, one half's, a pair's two windows in
    turn: the loss reduces each pair's halves by its number across the batch
    (`pair_numbers`, `PairIds`), so either reads alike, and a step reads a
    pair's two windows in one of its parts (`UNIT_ROWS`, which the trainer
    lays its rows by for its split). The completion mask says which targets
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
                 segments: int | None = None, **options):
        if beta <= 0:
            raise ValueError(f"beta scales the KL term, so it is positive, got {beta}")
        if segments is not None and segments < 1:
            raise ValueError(f"a window holds at least one document; got segments={segments}")
        width = window_width(seq_len + 1, segments)
        super().__init__(model, width - 1, balance=balance, **options)
        self.ema = EMASpec.constant(1.0)
        # The batch is windows of packed pairs, two halves' tokens a window, or one half's (`segments=1`).
        self.inputs = InputSpec(sample=Field(IDS_KEY, (width,)))
        self.beta = beta
        self.segments = segments

    def data(self, tokenizer=None) -> ObjectiveData:
        """Each batch's pairs numbered across it, each pair's windows one unit (`PairIds`)."""
        return ObjectiveData(train_batch=(PairIds(),), evaluate_batch=(PairIds(),))

    def _windows(self, batch):
        """The batch's ids, refused unless they are windows of the objective's width."""
        ids = jnp.asarray(batch[IDS_KEY])
        if ids.ndim != 2 or ids.shape[1] != self.seq_len + 1:
            raise ValueError(f"a DPO batch holds windows of packed pairs [{IDS_KEY} shape (batch, "
                             f"{self.seq_len + 1})] (dew.data.preferences), got {tuple(ids.shape)}")
        for key in (MASK_KEY, REAL_KEY, PAIRS_KEY, HALVES_KEY, POSITIONS_KEY,
                    *((DOCUMENTS_KEY,) if DOCUMENTS_KEY in batch else ())):
            if jnp.shape(batch[key]) != ids.shape:
                raise ValueError(f"{key} has shape {tuple(jnp.shape(batch[key]))} for {tuple(ids.shape)} "
                                 f"{IDS_KEY}; a window's fields hold one entry a token")
        return ids

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
        ids = self._windows(batch)
        laid = window_coordinates(batch)
        # The policy and its reference score the windows in one forward each, through the same function,
        # so a policy equal to its reference earns zero rewards wherever the two compile alike.
        real = jnp.asarray(batch[REAL_KEY])[:, :-1] > 0
        with watched_positions(real if step.watched is not None else None):
            policy = self.token_scores(variables, ids, coordinates=laid, qk_stats=step.qk_stats,
                                       predict=predict, progress=self.progress(step),
                                       routing=self.balance is not None, watched=step.watched is not None)
        reference = self.token_scores(step.ema, ids, coordinates=laid, progress=self.progress(step))
        policy_chosen, policy_rejected, weights = _halves(-policy.losses, batch, policy.weights)
        ref_chosen, ref_rejected, _ = _halves(-reference.losses, batch, policy.weights)
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
        """The router positions that count, `[B, W - 1]`: each half's completion tokens among the inputs, none
        in a repeat window (`VALID_ROWS`)."""
        mask = jnp.asarray(batch[MASK_KEY])
        rows = self.row_weights(batch, mask.shape[0]) > 0
        return (mask[:, :-1] != 0) & rows[:, None]

    def _evaluation_scores(self, params, batch, step) -> TokenScores:
        """The chosen responses' per-token cross entropies, with their completion marks as weights."""
        ids = self._windows(batch)
        scores = self.token_scores(params, ids, coordinates=window_coordinates(batch), predict=True)
        assert scores.correct is not None
        return self._chosen(scores.losses, scores.correct, scores.weights, batch)

    def _validation_scores(self, params, batch, step):
        """The preference loss and the chosen responses' scores, from the
        policy's one scoring of the windows."""
        statistics, _, policy = self._preference(params, batch, step, predict=True)
        assert policy.correct is not None
        return statistics, self._chosen(policy.losses, policy.correct, policy.weights, batch)

    def _chosen(self, losses, correct, scored, batch) -> TokenScores:
        """The chosen halves' scores, with their completion marks the scorer scores as weights."""
        chosen, _ = _completions(batch, scored)
        return TokenScores(losses=losses, weights=chosen.astype(losses.dtype), correct=correct)
