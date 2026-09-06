"""Group-relative policy optimization over sampled rollouts.

A `GRPOObjective` is an `LMObjective` whose loss is the section 6
composition from `dew.rl`: the clipped surrogate of the token log-ratio plus
`beta` times the token-mean k3 KL against the frozen reference. It reads the
rolled-out batch the `SampledRollout` packs: the concatenation for the
current log-probabilities, `old_log_probs` for the ratio, `advantages` and
`response_mask` for the masked terms. The reference is the objective's own
frozen tree, `step.ema` at unit decay, rescored only when `beta` is positive.
Validation scores the prompts' own perplexity, since a validation pass never
samples.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from dew.artifacts import TokenScores
from dew.data.prompts import LENGTH_KEY, PROMPT_KEY
from dew.objectives.base import Aux
from dew.objectives.lm.chunked import chunked_cross_entropy
from dew.registry import objectives
from dew.rl import clipped_surrogate, k3_kl, token_log_ratio, token_mean

from ..lm import LMObjective
from .rollout import ADVANTAGES_KEY, IDS_KEY, OLD_LOG_PROBS_KEY, RESPONSE_MASK_KEY


@objectives("grpo")
class GRPOObjective(LMObjective):
    """The GRPO loss (arXiv:2402.03300, eq. 4) with verl's presentation: the
    dual-clipped surrogate token-meaned over the response mask, plus `beta`
    times the token-mean k3 KL to the frozen reference
    (`verl/trainer/ppo/core_algos.py`, `compute_policy_loss_vanilla` with
    `token-mean` and `kl_penalty_forward` with `k3`).

    `beta` is the KL strength, 0.0 leaving the reference unread;
    `epsilon_low`, `epsilon_high` and `dual_clip` are the clip points;
    `model` and `seq_len` are the LMObjective's, with `seq_len` one below the
    prompt width plus the response width. An `ema_decay` argument is refused,
    and `loss_role` is refused with it: the response mask already says which
    targets count.
    """

    def __init__(self, model, seq_len: int, beta: float = 0.0,
                 epsilon_low: float = 0.2, epsilon_high: float = 0.2,
                 dual_clip: float = 3.0, **kwargs):
        if beta < 0:
            raise ValueError(f"beta scales the KL penalty, so it is non-negative, got {beta}")
        if "ema_decay" in kwargs:
            raise ValueError(
                "the GRPO reference is frozen at unit decay, so ema_decay is refused")
        if kwargs.get("loss_role") is not None:
            raise ValueError(
                "the response mask already says which targets count, "
                "so loss_role is refused on a GRPO objective")
        kwargs["ema_decay"] = 1.0
        super().__init__(model, seq_len, **kwargs)
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.dual_clip = dual_clip

    def _window(self, batch):
        """The rollout batch validated: the concatenation width, the response
        width, and the response slice. Position p of the concatenation
        predicts token p + 1, so the response starts one before the prompt
        width."""
        try:
            ids = jnp.asarray(batch[IDS_KEY])
        except KeyError:
            raise ValueError(
                f"a GRPO batch carries {IDS_KEY} with the prompt concatenation; "
                f"the batch has {sorted(batch)}") from None
        for key in (OLD_LOG_PROBS_KEY, ADVANTAGES_KEY, RESPONSE_MASK_KEY):
            if key not in batch:
                raise ValueError(
                    f"a GRPO batch carries {key} from the rollout; "
                    f"the batch has {sorted(batch)}")
        old = jnp.asarray(batch[OLD_LOG_PROBS_KEY])
        if old.shape != jnp.asarray(batch[ADVANTAGES_KEY]).shape:
            raise ValueError(
                f"{OLD_LOG_PROBS_KEY} {tuple(old.shape)} and {ADVANTAGES_KEY} "
                f"{tuple(jnp.asarray(batch[ADVANTAGES_KEY]).shape)} share one shape: "
                "one term per response token")
        if old.shape != jnp.asarray(batch[RESPONSE_MASK_KEY]).shape:
            raise ValueError(
                f"{OLD_LOG_PROBS_KEY} {tuple(old.shape)} and {RESPONSE_MASK_KEY} "
                f"{tuple(jnp.asarray(batch[RESPONSE_MASK_KEY]).shape)} share one shape: "
                "one term per response token")
        if ids.shape[1] != self.seq_len + 1:
            raise ValueError(
                f"a {self.seq_len}-token context needs {self.seq_len + 1} ids per row, "
                f"got {ids.shape[1]}")
        if old.shape[1] >= ids.shape[1]:
            raise ValueError(
                f"the response is {old.shape[1]} tokens wide for a {ids.shape[1]}-wide "
                "concatenation; the rollout sizes the objective one below prompt "
                "plus response")
        start = ids.shape[1] - old.shape[1] - 1
        return ids, start, old.shape[1]

    def loss(self, params, batch, step):
        ids, start, width = self._window(batch)
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY])
        policy = self.per_token_log_probs(params, ids)[:, start:start + width]
        ratio = token_log_ratio(policy, jnp.asarray(batch[OLD_LOG_PROBS_KEY]))
        pg_loss, aux = clipped_surrogate(
            ratio, jnp.asarray(batch[ADVANTAGES_KEY]), mask,
            epsilon_low=self.epsilon_low, epsilon_high=self.epsilon_high,
            dual_clip=self.dual_clip)
        metrics = {"pg": pg_loss, **{f"actor/{k}": v for k, v in aux.items()}}
        if self.beta > 0:
            if step.ema is None:
                raise ValueError(
                    "the KL term reads step.ema, but the objective keeps no EMA; "
                    "a GRPO run with beta above zero always freezes one")
            ref = self.per_token_log_probs(step.ema, ids)[:, start:start + width]
            kl = token_mean(k3_kl(policy, ref), mask)
            metrics["kl"] = kl
            return pg_loss + self.beta * kl, Aux(metrics)
        return pg_loss, Aux(metrics)

    def preview(self, params, batch, step, *, scored=None):
        """Draw policy text; this objective's EMA holds the frozen reference."""
        return super().preview(params, batch, dataclasses.replace(step, ema=None), scored=scored)

    def evaluate(self, params, batch, step):
        """The prompts' perplexity under the policy: each row's shifted
        cross entropy with the real suffix as weights, off the row's
        `prompt_length`. Pads predict nothing and count nothing."""
        try:
            prompts = jnp.asarray(batch[PROMPT_KEY])
        except KeyError:
            raise ValueError(
                f"GRPO validation scores {PROMPT_KEY} batches; "
                f"the batch has {sorted(batch)}") from None
        try:
            lengths = jnp.asarray(batch[LENGTH_KEY]).reshape(-1)
        except KeyError:
            raise ValueError(
                f"GRPO validation weights with {LENGTH_KEY}; "
                f"the batch has {sorted(batch)}") from None
        if prompts.shape[1] < 2:
            raise ValueError(
                f"a prompt needs two tokens to score one target, got {prompts.shape[1]}")
        hidden = self.model.apply(params, prompts[:, :-1], train=False,
                                  method=type(self.model).hidden_states)
        head = self.model.apply(params, params["params"],
                                method=type(self.model).head_weight)
        losses, _ = chunked_cross_entropy(
            hidden, head, prompts[:, 1:], self.head_chunks,
            softcap=self.model.final_logit_softcap,
            precision=self.model.precision)
        positions = jnp.broadcast_to(
            jnp.arange(prompts.shape[1] - 1), losses.shape)
        starts = (prompts.shape[1] - lengths - 1)[:, None]
        weights = (positions >= starts).astype(losses.dtype)
        return TokenScores(losses=losses, weights=weights)
