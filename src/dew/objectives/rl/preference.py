"""Direct preference optimization over chosen and rejected completions.

A `DPOObjective` is an `LMObjective` whose loss is the preference term from
`dew.rl` instead of the cross entropy: policy and reference log-probabilities
as negated per-token cross entropies, summed under the shifted completion
mask, through `preference_logsigmoid_terms`, averaged over pairs. The reference is the objective's own
frozen tree, `step.ema` at unit decay, so the run carries no second model.
Batches hold pairs, `[B, 2, S]` with the chosen row at index 0, so shuffling
never separates a pair; the loss reads them in TRL's stacked order.
"""

from __future__ import annotations

import jax.numpy as jnp

from dew.artifacts import TokenScores
from dew.data.preferences import IDS_KEY, MASK_KEY
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, Variables
from dew.rl.surrogate import preference_logsigmoid_terms

from ..lm import LMObjective


class DPOObjective(LMObjective):
    """Trains a policy on preference pairs with the DPO loss (arXiv:2305.18290, equation 7).

    The reference is the starting policy, frozen. `beta` is the KL strength
    and must be positive. `model` and `seq_len` are as in `LMObjective`, with
    `seq_len` one less than the row width. The reference never moves, so an
    `ema_decay` other than None is refused, and so is `loss_role`, because the
    completion mask already says which targets count. Validation scores the
    chosen responses' perplexity under the policy.
    """

    _ema_is_reference = True

    keeps_whole_logits = False

    def __init__(self, model, seq_len: int, beta: float = 0.1, **kwargs):
        if beta <= 0:
            raise ValueError(f"beta scales the KL term, so it is positive, got {beta}")
        if kwargs.get("ema_decay") is not None:
            raise ValueError(
                "the DPO reference is frozen at unit decay, so ema_decay is refused")
        if kwargs.get("loss_role") is not None:
            raise ValueError(
                "the completion mask already says which targets count, "
                "so loss_role is refused on a DPO objective")
        kwargs["ema_decay"] = 1.0
        super().__init__(model, seq_len, **kwargs)
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
        if step.ema is None:
            raise ValueError(
                "the DPO reference reads step.ema, but the objective keeps no EMA; "
                "a DPO run always freezes one")
        chosen_ids, rejected_ids, chosen_mask, rejected_mask = self._halves(batch)
        # The policy and its reference score both halves in one forward each, through the same
        # program, so a policy equal to its reference earns rewards of exactly zero, not ulps.
        pairs = jnp.concatenate((chosen_ids, rejected_ids))
        policy = self.token_scores(variables, pairs, qk_stats=self.qk_stats)
        policy_chosen, policy_rejected = jnp.split(-policy.losses, 2)
        ref_chosen, ref_rejected = jnp.split(-self.token_scores(step.ema, pairs).losses, 2)
        terms, (pair_chosen, pair_rejected) = preference_logsigmoid_terms(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected,
            chosen_mask, rejected_mask, self.beta)
        accuracy, _ = self.accuracy((pair_chosen > pair_rejected).astype(jnp.float32), batch).mean()
        return self.row_mean(terms, batch), Aux[Variables]({
            "rewards/chosen": pair_chosen.mean(),
            "rewards/rejected": pair_rejected.mean(),
            "accuracy": accuracy,
        }, qk_stats=policy.qk)

    def evaluate(self, params, batch, step):
        """Return per-token scores of the chosen responses under the policy, for their perplexity.

        The per-token cross entropies take the shifted completion mask as
        weights.
        """
        chosen_ids, _, chosen_mask, _ = self._halves(batch)
        scores = self.token_scores(params, chosen_ids, predict=True)
        losses = scores.losses
        assert scores.correct is not None
        weights = chosen_mask.astype(losses.dtype)
        return TokenScores(losses=losses, weights=weights, correct=scores.correct)
