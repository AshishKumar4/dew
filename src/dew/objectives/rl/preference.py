"""Direct preference optimization over chosen and rejected completions.

A `DPOObjective` is an `LMObjective` whose loss is the preference term from
`dew.rl` instead of the cross entropy: policy and reference log-probabilities
as negated per-token cross entropies, summed under the shifted completion
mask, through `preference_logsigmoid`. The reference is the objective's own
frozen tree, `step.ema` at unit decay, so the run carries no second model.
Batches hold pairs, `[B, 2, S]` with the chosen row at index 0, so shuffling
never separates a pair; the loss reads them in TRL's stacked order.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from dew.objectives.base import Variables

from dew.artifacts import TokenScores
from dew.data.preferences import IDS_KEY, MASK_KEY
from dew.objectives.base import Aux, Mean
from dew.registry import objectives
from dew.rl.surrogate import preference_logsigmoid_terms

from ..lm import LMObjective


@objectives("dpo")
class DPOObjective(LMObjective):
    """The DPO loss (arXiv:2305.18290, eq. 7) on a frozen reference.

    `beta` is the KL strength; `model` and `seq_len` are the LMObjective's,
    with `seq_len` one below the row width. The reference never moves, so an
    `ema_decay` argument is refused, and `loss_role` is refused with it: the
    completion mask already says which targets count. Validation scores the
    chosen responses' perplexity under the policy.
    """

    def __init__(self, model, seq_len: int, beta: float = 0.1, **kwargs):
        if beta <= 0:
            raise ValueError(f"beta scales the KL term, so it is positive, got {beta}")
        if "ema_decay" in kwargs:
            raise ValueError(
                "the DPO reference is frozen at unit decay, so ema_decay is refused")
        if kwargs.get("loss_role") is not None:
            raise ValueError(
                "the completion mask already says which targets count, "
                "so loss_role is refused on a DPO objective")
        kwargs["ema_decay"] = 1.0
        super().__init__(model, seq_len, **kwargs)
        self.beta = beta

    def _halves(self, batch):
        """The batch as chosen and rejected halves with the shifted mask: the
        pair index read out explicitly, so shuffling never fuses two pairs.
        Refuses flat stacks, misaligned masks and rows outside the window."""
        try:
            ids = jnp.asarray(batch[IDS_KEY])
        except KeyError:
            raise ValueError(
                f"a DPO batch carries {IDS_KEY} with one pair per row; "
                f"the batch has {sorted(batch)}") from None
        try:
            mask = jnp.asarray(batch[MASK_KEY])
        except KeyError:
            raise ValueError(
                f"a DPO batch carries {MASK_KEY} marking the completion tokens; "
                f"the batch has {sorted(batch)}") from None
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

    def loss(self, params, batch, step):
        if step.ema is None:
            raise ValueError(
                "the DPO reference reads step.ema, but the objective keeps no EMA; "
                "a DPO run always freezes one")
        chosen_ids, rejected_ids, chosen_mask, rejected_mask = self._halves(batch)
        policy_chosen = self.per_token_log_probs(params, chosen_ids)
        policy_rejected = self.per_token_log_probs(params, rejected_ids)
        ref_chosen = self.per_token_log_probs(step.ema, chosen_ids)
        ref_rejected = self.per_token_log_probs(step.ema, rejected_ids)
        terms = preference_logsigmoid_terms(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected,
            chosen_mask, rejected_mask, self.beta)
        pair_chosen = self.beta * (policy_chosen * chosen_mask).sum(-1)
        pair_rejected = self.beta * (policy_rejected * rejected_mask).sum(-1)
        accuracy = (pair_chosen > pair_rejected).astype(jnp.float32).mean()
        return Mean(jnp.sum(terms), jnp.asarray(terms.size)), Aux[Variables]({
            "rewards/chosen": pair_chosen.mean(),
            "rewards/rejected": pair_rejected.mean(),
            "accuracy": accuracy,
        })

    def preview(self, params, batch, step, *, scored=None):
        """Draw policy text; this objective's EMA holds the frozen reference."""
        return super().preview(params, batch, dataclasses.replace(step, ema=None), scored=scored)

    def evaluate(self, params, batch, step):
        """The chosen responses' perplexity under the policy: the per-token
        cross entropies with the shifted completion mask as weights."""
        chosen_ids, _, chosen_mask, _ = self._halves(batch)
        losses = -self.per_token_log_probs(params, chosen_ids)
        weights = chosen_mask.astype(losses.dtype)
        return TokenScores(losses=losses, weights=weights)
