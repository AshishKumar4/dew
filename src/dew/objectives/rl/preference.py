"""Direct preference optimization over chosen and rejected completions.

A `DPOObjective` scores tokens as `LMObjective` does (`TokenScorer`), and its
loss is the preference term from `dew.rl`: policy and reference log-probabilities
as negated per-token cross entropies, summed under the shifted completion
mask, through `preference_logsigmoid_terms`, averaged over pairs. The reference is the objective's own
frozen tree, `step.ema` at unit decay, so the run carries no second model.
Batches hold pairs, `[B, 2, S]` with the chosen row at index 0, so shuffling
never separates a pair; the loss reads them in TRL's stacked order.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dew.artifacts import TokenScores
from dew.batch import VALID_ROWS, Variables
from dew.data.preferences import IDS_KEY, MASK_KEY
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, EMASpec
from dew.objectives.routing import ExpertBalance
from dew.rl.surrogate import preference_logsigmoid_terms

from ..lm.objective import TokenScorer


class DPOObjective(TokenScorer):
    """Trains a policy on preference pairs with the DPO loss (arXiv:2305.18290, equation 7).

    The reference is the starting policy, frozen: the objective's average at
    unit decay. `beta` is the KL strength and must be positive. `model`,
    `seq_len` and the scoring `options` are `TokenScorer`'s, with `seq_len`
    one less than the row width; the completion mask says which targets
    count. Validation scores the chosen responses' perplexity under the
    policy.

    `balance` trains the routers of a mixture as `ExpertBalance` describes
    (`dew.objectives.routing`), over the completion tokens each half's forward
    reads: both halves of every pair route, and the counts sum over them.
    """

    ema_is_reference = True
    counts_tokens = False
    """Preference pairs need their two-sequence geometry, not LM's text rows."""

    keeps_whole_logits = False

    scored_fields = (IDS_KEY, MASK_KEY)
    """Validation scores the chosen responses of the pairs alone, for their perplexity."""

    def __init__(self, model, seq_len: int, beta: float = 0.1, *, balance: ExpertBalance | None = None,
                 **options):
        if beta <= 0:
            raise ValueError(f"beta scales the KL term, so it is positive, got {beta}")
        super().__init__(model, seq_len, balance=balance, **options)
        self.ema = EMASpec.constant(1.0)
        # The batch is `input_ids` pairs, chosen then rejected, not text windows.
        self.inputs = InputSpec(sample=Field(IDS_KEY, (2, seq_len + 1)))
        self.beta = beta

    def _halves(self, batch):
        """Split the batch into chosen and rejected halves with the shifted mask.

        The pair index is read out explicitly, so shuffling never fuses
        two pairs. Flat stacks, misaligned masks and rows outside the
        window are refused.
        """
        ids = jnp.asarray(batch[IDS_KEY])
        mask = jnp.asarray(batch[MASK_KEY])
        if ids.ndim != 3 or ids.shape[1] != 2:
            raise ValueError(
                f"a DPO batch holds pairs [{IDS_KEY} shape (batch, 2, length)], "
                f"got {tuple(ids.shape)}")
        if ids.shape != mask.shape:
            raise ValueError(
                f"{IDS_KEY} has shape {tuple(ids.shape)} for {tuple(mask.shape)} "
                f"{MASK_KEY}; ids and mask align, one mark per token")
        if ids.shape[2] != self.seq_len + 1:
            raise ValueError(
                f"a {self.seq_len}-token context needs {self.seq_len + 1} ids per row, "
                f"got {ids.shape[2]}")
        # Axis 1 is the pair, not a stack: a reshape would interleave two
        # pairs into the halves and compare across them.
        return (ids[:, 0], ids[:, 1], mask[:, 0, 1:], mask[:, 1, 1:])

    def loss(self, variables, batch, step):
        """Compute the preference term over each pair's completion tokens.

        It also reports `rewards/chosen`, `rewards/rejected` and `accuracy`,
        the fraction of pairs whose chosen reward is higher.
        """
        statistics, aux, _ = self._preference(variables, batch, step)
        return statistics, aux

    def _preference(self, variables, batch, step, *, predict: bool | None = None):
        """The preference loss and its reports, with the policy's scores of
        the pairs behind them, chosen halves first (`token_scores`' `predict`)."""
        if step.ema is None:
            raise ValueError(
                "the DPO reference reads step.ema, but the objective keeps no EMA; "
                "a DPO run always freezes one")
        chosen_ids, rejected_ids, chosen_mask, rejected_mask = self._halves(batch)
        # The policy and its reference score both halves in one forward each, through the same
        # function, so a policy equal to its reference earns zero rewards wherever the two compile alike.
        pairs = jnp.concatenate((chosen_ids, rejected_ids))
        policy = self.token_scores(variables, pairs, qk_stats=step.qk_stats, predict=predict,
                                   routing=self.balance is not None)
        policy_chosen, policy_rejected = jnp.split(-policy.losses, 2)
        ref_chosen, ref_rejected = jnp.split(-self.token_scores(step.ema, pairs).losses, 2)
        terms, (pair_chosen, pair_rejected) = preference_logsigmoid_terms(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected,
            chosen_mask, rejected_mask, self.beta)
        accuracy, _ = self.accuracy((pair_chosen > pair_rejected).astype(jnp.float32), batch).mean()
        statistics = self.row_mean(terms, batch)
        reported = {
            "rewards/chosen": pair_chosen.mean(),
            "rewards/rejected": pair_rejected.mean(),
            "accuracy": accuracy,
        }
        effects = None
        if self.balance is not None:
            statistics, effects, metrics = self.balance.compose(
                variables, statistics, policy.routing or {}, self._routed_batch(batch),
                self._routed_queries(batch), {})
            reported.update(metrics)
        return statistics, Aux[Variables](reported, qk_stats=policy.qk, effects=effects), policy

    def _routed_queries(self, batch) -> jax.Array:
        """The router positions that count, `[2 * pairs, seq_len]`: each half's completion tokens among its
        inputs, the chosen halves first as `pairs` stacks them, and none in a repeat row (`VALID_ROWS`)."""
        mask = jnp.asarray(batch[MASK_KEY])
        rows = self.row_weights(batch, mask.shape[0]) > 0
        completed = jnp.concatenate((mask[:, 0, :-1], mask[:, 1, :-1])) != 0
        return completed & jnp.concatenate((rows, rows))[:, None]

    def _routed_batch(self, batch):
        """The batch with each pair's row weight on both of its halves' rows, the rows the routers' statistics
        count (`ExpertBalance.compose`); a batch without `VALID_ROWS` is read as it is."""
        if VALID_ROWS not in batch:
            return batch
        rows = self.row_weights(batch, batch[IDS_KEY].shape[0])
        return {**batch, VALID_ROWS: jnp.concatenate((rows, rows))}

    def _evaluation_scores(self, params, batch, step) -> TokenScores:
        """The chosen responses' per-token cross entropies, with the shifted
        completion mask as weights."""
        chosen_ids, _, _, _ = self._halves(batch)
        scores = self.token_scores(params, chosen_ids, predict=True)
        assert scores.correct is not None
        return self._chosen(scores.losses, scores.correct, batch)

    def _validation_scores(self, params, batch, step):
        """The preference loss and the chosen responses' scores, from the
        policy's one scoring of the pairs."""
        statistics, _, policy = self._preference(params, batch, step, predict=True)
        assert policy.correct is not None
        rows = policy.losses.shape[0] // 2
        return statistics, self._chosen(policy.losses[:rows], policy.correct[:rows], batch)

    def _chosen(self, losses, correct, batch) -> TokenScores:
        """The chosen halves' scores, with the shifted completion mask as weights."""
        _, _, chosen_mask, _ = self._halves(batch)
        return TokenScores(losses=losses, weights=chosen_mask.astype(losses.dtype), correct=correct)
