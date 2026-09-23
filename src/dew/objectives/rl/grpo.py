"""Group-relative policy optimization over sampled rollouts.

A `GRPOObjective` is an `LMObjective` whose loss is the section 6
composition from `dew.rl`: a clipped policy surrogate plus `beta` times the
k3 KL against the frozen reference. It reads one of two batch layouts.

- The windowed layout `SampledRollout` builds: each row
  is `[left-padded prompt | response]`, and `old_log_probs`, `advantages`
  and `response_mask` are response-width.
- The packed layout `rollouts.pack` builds: rows of strictly merged chains
  with `text_segment_ids` and `text_positions`, every column `[rows, width]`
  and aligned with `input_ids`. `behavior_log_probs` stands in for
  `old_log_probs` when no proximal rescoring supplied one.

The reference is the objective's own frozen tree, `step.ema` at unit decay,
rescored only when `beta` is positive. Validation scores the prompts' own
perplexity, since a validation pass never samples.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from dew.artifacts import TokenScores
from dew.data.prompts import LENGTH_KEY, PROMPT_KEY
from dew.objectives.base import Aux, Mean, Variables, mean_loss
from dew.objectives.lm.chunked import chunked_cross_entropy
from dew.registry import objectives
from dew.rl import behavior_importance_weights, k3_kl, sequence_log_ratio, token_log_ratio
from dew.rl.surrogate import (
    behavior_band_weights,
    cispo_terms,
    clipped_surrogate_terms,
    mismatch_metrics,
    sequence_rejection_mask,
)

from ..lm import LMObjective
from ..lm.objective import _shift_rows, _unpadded
from .rollout import ADVANTAGES_KEY, BEHAVIOR_LOG_PROBS_KEY, IDS_KEY, OLD_LOG_PROBS_KEY, RESPONSE_MASK_KEY
from .rollouts import POSITIONS_KEY, ROLLOUT_WEIGHTS_KEY, SEGMENT_IDS_KEY

POLICY_LOSSES = ("ppo", "gspo", "cispo")
AGGREGATIONS = ("token-mean", "rollout-mean")


class _Terms(NamedTuple):
    """The loss's inputs on one grid, target-aligned: current, proximal and
    behavior log-probabilities, advantages, the trainable mask, chain ids
    (None on a windowed row, which is one sequence), the packed per-rollout
    weights (None when the batch carries none) and how to rescore the
    reference."""

    policy: jax.Array
    old: jax.Array
    behavior: jax.Array | None
    advantages: jax.Array
    mask: jax.Array
    segments: jax.Array | None
    rollout_weights: jax.Array | None
    proximal: bool


def _band(name: str, band) -> tuple[float, float] | None:
    if band is None:
        return None
    low, high = (float(value) for value in band)
    if not 0 < low <= high:
        raise ValueError(f"{name} is a (low, high) ratio band with 0 < low <= high, got {band}")
    return low, high


@objectives("grpo")
class GRPOObjective(LMObjective):
    """Train a policy on sampled rollouts with the GRPO loss (arXiv:2402.03300, eq. 4).

    The default composition is verl's: the dual-clipped surrogate, token-meaned
    over the response mask, plus `beta` times the token-mean k3 KL to the
    frozen reference (`verl/trainer/ppo/core_algos.py`,
    `compute_policy_loss_vanilla` with `token-mean` and
    `kl_penalty_forward` with `k3`).

    `beta` is the KL strength; 0.0 allocates no frozen reference.
    `epsilon_low`, `epsilon_high` and `dual_clip` are the clip points;
    `model` and `seq_len` are the LMObjective's, with `seq_len` one below the
    row width. An `ema_decay` argument is refused, and `loss_role` is refused
    with it: the response mask already says which targets count.

    `policy_loss` picks the surrogate, each checked against verl 12ebe0c:
    `"ppo"` (`compute_policy_loss_vanilla`), `"gspo"`
    (`compute_policy_loss_gspo`: the sequence ratio of `sequence_log_ratio`,
    clipped, no dual clip) and `"cispo"` (`compute_policy_loss_cispo`). A
    sequence is a windowed row or a packed chain.

    `aggregation` is `"token-mean"` (verl's default) or `"rollout-mean"`:
    each rollout's token mean, averaged over rollouts, so a long rollout or
    one split over several rows weighs as one (Agent Lightning's
    `per_rollout_mean`, verl's `seq-mean-token-mean` when a rollout is one
    row). Packed batches carry the weights in `rollout_weights`; a windowed
    row is its own rollout.

    Behavior corrections read `behavior_log_probs` against the proximal
    policy (`old_log_probs`), all detached, from verl's
    `rollout_corr_helper`: `behavior_importance_cap` caps the token ratio
    (TIS); `behavior_band` zeroes it outside `(low, high)` instead (IcePop);
    `sequence_mask` and `geometric_mask` reject every token of a sequence
    whose summed (`seq_sum_k1`) or mean (`seq_mean_k1`) k1 statistic lies
    outside `(log low, log high)`. Metrics add `mismatch/kl`,
    `mismatch/k3_kl` and `mismatch/ess` whenever behavior likelihoods are
    present, and the fraction of trainable tokens each correction masked.
    """

    _ema_is_reference = True

    def __init__(self, model, seq_len: int, beta: float = 0.0,
                 epsilon_low: float = 0.2, epsilon_high: float = 0.2,
                 dual_clip: float = 3.0, behavior_importance_cap: float | None = None, *,
                 policy_loss: str = "ppo", aggregation: str = "token-mean",
                 behavior_band: tuple[float, float] | None = None,
                 sequence_mask: tuple[float, float] | None = None,
                 geometric_mask: tuple[float, float] | None = None, **kwargs):
        if beta < 0:
            raise ValueError(f"beta scales the KL penalty, so it is non-negative, got {beta}")
        if "ema_decay" in kwargs:
            raise ValueError(
                "the GRPO reference is frozen at unit decay, so ema_decay is refused")
        if kwargs.get("loss_role") is not None:
            raise ValueError(
                "the response mask already says which targets count, "
                "so loss_role is refused on a GRPO objective")
        if policy_loss not in POLICY_LOSSES:
            raise ValueError(f"policy_loss must be one of {POLICY_LOSSES}, got {policy_loss!r}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        kwargs["ema_decay"] = 1.0 if beta > 0 else None
        super().__init__(model, seq_len, **kwargs)
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.dual_clip = dual_clip
        if behavior_importance_cap is not None and (isinstance(behavior_importance_cap, bool)
                                                    or not behavior_importance_cap > 0):
            raise ValueError("behavior_importance_cap must be positive, or None to disable correction")
        self.behavior_band = _band("behavior_band", behavior_band)
        if behavior_importance_cap is not None and self.behavior_band is not None:
            raise ValueError("behavior_importance_cap and behavior_band are two token corrections; pick one")
        self.behavior_importance_cap = behavior_importance_cap
        self.sequence_mask = _band("sequence_mask", sequence_mask)
        self.geometric_mask = _band("geometric_mask", geometric_mask)
        self.policy_loss = policy_loss
        self.aggregation = aggregation

    def _window(self, batch):
        """Validate the rollout batch and locate its response slice.

        Returns the concatenation, where the response starts and how wide
        it is. Position p of the concatenation predicts token p + 1, so
        the response starts one before the prompt width.
        """
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

    def packed_log_probs(self, params: Variables, batch) -> jax.Array:
        """Score each packed id given its own chain's prefix, `[rows, width]`.

        Entry t is `log pi(input_ids[t] | chain prefix)`, aligned with
        `input_ids`; it is zero where `response_mask` is zero, which covers
        every chain start and all padding. The loss and a proximal rescoring
        read this one function.
        """
        for key in (IDS_KEY, SEGMENT_IDS_KEY, POSITIONS_KEY, RESPONSE_MASK_KEY):
            if key not in batch:
                raise ValueError(f"a packed GRPO batch carries {key}; the batch has {sorted(batch)}")
        ids = jnp.asarray(batch[IDS_KEY], jnp.int32)
        segments = jnp.asarray(batch[SEGMENT_IDS_KEY], jnp.int32)
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY])
        for key in (SEGMENT_IDS_KEY, POSITIONS_KEY, RESPONSE_MASK_KEY):
            if jnp.shape(batch[key]) != ids.shape:
                raise ValueError(f"{key} has shape {jnp.shape(batch[key])}; a packed column has "
                                 f"the shape of {IDS_KEY}, {ids.shape}")
        scores = self.token_scores(params, ids, segment_ids=segments,
                                   positions=jnp.asarray(batch[POSITIONS_KEY], jnp.int32))
        scored = jnp.concatenate([jnp.zeros((ids.shape[0], 1), jnp.float32),
                                  -scores.losses.astype(jnp.float32)], axis=1)
        return jnp.where(mask != 0, scored, 0.0)

    def _packed_terms(self, params, batch) -> _Terms:
        """Read a packed batch onto its own `[rows, width]` grid.

        The proximal policy is `old_log_probs` when a rescoring or the
        sampler supplied it, and the recorded behavior otherwise, so one of
        the two is required.
        """
        if ADVANTAGES_KEY not in batch:
            raise ValueError(f"a packed GRPO batch carries {ADVANTAGES_KEY}; the batch has {sorted(batch)}")
        if OLD_LOG_PROBS_KEY not in batch and BEHAVIOR_LOG_PROBS_KEY not in batch:
            raise ValueError(f"a packed GRPO batch carries {BEHAVIOR_LOG_PROBS_KEY} or {OLD_LOG_PROBS_KEY}; "
                             f"the batch has {sorted(batch)}")
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY], jnp.float32)
        for key in (ADVANTAGES_KEY, BEHAVIOR_LOG_PROBS_KEY, OLD_LOG_PROBS_KEY, ROLLOUT_WEIGHTS_KEY):
            if key in batch and jnp.shape(batch[key]) != mask.shape:
                raise ValueError(f"{key} has shape {jnp.shape(batch[key])}; a packed column has "
                                 f"the shape of {IDS_KEY}, {mask.shape}")
        behavior = (jnp.asarray(batch[BEHAVIOR_LOG_PROBS_KEY], jnp.float32)
                    if BEHAVIOR_LOG_PROBS_KEY in batch else None)
        proximal = OLD_LOG_PROBS_KEY in batch
        old = jnp.asarray(batch[OLD_LOG_PROBS_KEY], jnp.float32) if proximal else behavior
        assert old is not None
        weights = batch.get(ROLLOUT_WEIGHTS_KEY)
        return _Terms(self.packed_log_probs(params, batch), old, behavior,
                      jnp.asarray(batch[ADVANTAGES_KEY], jnp.float32), mask,
                      jnp.asarray(batch[SEGMENT_IDS_KEY], jnp.int32),
                      None if weights is None else jnp.asarray(weights, jnp.float32), proximal)

    def _windowed_terms(self, params, batch) -> _Terms:
        """Read a windowed batch onto its response slice."""
        ids, start, width = self._window(batch)
        behavior = None
        if BEHAVIOR_LOG_PROBS_KEY in batch:
            behavior = jnp.asarray(batch[BEHAVIOR_LOG_PROBS_KEY], jnp.float32)
            if behavior.shape != jnp.shape(batch[OLD_LOG_PROBS_KEY]):
                raise ValueError("behavior_log_probs must have the same shape as old_log_probs")
        padding = (None if LENGTH_KEY not in batch else
                   start + 1 - jnp.asarray(batch[LENGTH_KEY], jnp.int32))
        policy = self.per_token_log_probs(params, ids, left_padding=padding)[:, start:start + width]
        return _Terms(policy, jnp.asarray(batch[OLD_LOG_PROBS_KEY], jnp.float32), behavior,
                      jnp.asarray(batch[ADVANTAGES_KEY], jnp.float32),
                      jnp.asarray(batch[RESPONSE_MASK_KEY]), None, None, proximal=True)

    def _reference(self, params, batch) -> jax.Array:
        """The frozen reference's log-probabilities on the loss's grid."""
        if SEGMENT_IDS_KEY in batch:
            return self.packed_log_probs(params, batch)
        ids, start, width = self._window(batch)
        padding = (None if LENGTH_KEY not in batch else
                   start + 1 - jnp.asarray(batch[LENGTH_KEY], jnp.int32))
        return self.per_token_log_probs(params, ids, left_padding=padding)[:, start:start + width]

    def loss(self, params, batch, step):
        """Score the policy surrogate over the trainable tokens, plus the KL to the reference.

        The policy is rescored from the rollout's own ids, so every term
        reads the tokens that were actually drawn.
        """
        terms = (self._packed_terms if SEGMENT_IDS_KEY in batch else self._windowed_terms)(params, batch)
        mask = terms.mask
        corrected = (self.behavior_importance_cap, self.behavior_band, self.sequence_mask, self.geometric_mask)
        if any(option is not None for option in corrected) and terms.behavior is None:
            raise ValueError("behavior corrections require recorded behavior_log_probs")
        if self.behavior_importance_cap is not None and not terms.proximal:
            raise ValueError(
                "without old_log_probs the ratio is already current over behavior, so a TIS cap "
                "would count the correction twice (verl's bypass mode applies no IS weight); "
                "rescore old_log_probs or drop behavior_importance_cap")
        # Without a proximal rescoring, behavior stands in for the old policy,
        # and the corrections compare the detached current policy with behavior,
        # as verl's compute_policy_loss_bypass_mode does.
        proximal = terms.old if terms.proximal else jax.lax.stop_gradient(terms.policy)
        metrics: dict[str, jax.Array] = {}
        keep = jnp.ones_like(mask, jnp.float32)
        for name, band, geometric in (("sequence", self.sequence_mask, False),
                                      ("geometric", self.geometric_mask, True)):
            if band is not None:
                assert terms.behavior is not None
                rejected = sequence_rejection_mask(proximal, terms.behavior, mask, *band,
                                                   geometric=geometric, segments=terms.segments)
                metrics[f"masked/{name}"] = _fraction(1 - rejected, mask)
                keep = keep * rejected
        # Weights and diagnostics read every trainable token, as verl's do;
        # rejection reaches the loss through `effective` alone.
        importance = None
        if terms.behavior is not None:
            if self.behavior_importance_cap is not None:
                importance = behavior_importance_weights(proximal, terms.behavior, mask,
                                                         self.behavior_importance_cap)
            elif self.behavior_band is not None:
                importance = behavior_band_weights(proximal, terms.behavior, mask, *self.behavior_band)
                metrics["masked/band"] = _fraction(importance == 0, mask)
            cap = (self.behavior_importance_cap if self.behavior_band is None else self.behavior_band[1])
            mismatch = mismatch_metrics(proximal, terms.behavior, mask, importance, cap)
            metrics.update({f"mismatch/{key}": value for key, value in mismatch.items()})
            if importance is not None and not terms.proximal:
                # The ratio already carries current over behavior, so the band
                # acts as a keep mask and applies no weight.
                keep = keep * (importance != 0)
                importance = None
        effective = mask * keep
        per_token, aux = self._policy_terms(terms, effective)
        if importance is not None:
            per_token = per_token * importance
        weights = self._weights(terms, effective, keep)
        mass = jax.lax.stop_gradient(jnp.sum(weights))
        pg = Mean(jnp.sum(jnp.where(weights != 0, per_token, 0) * weights), mass)
        pg_loss, _ = mean_loss(pg)
        metrics = {"pg": pg_loss, **{f"actor/{k}": v for k, v in aux.items()}, **metrics}
        if self.beta > 0:
            if step.ema is None:
                raise ValueError(
                    "the KL term reads step.ema, but the objective keeps no EMA; "
                    "a GRPO run with beta above zero always freezes one")
            kl_terms = k3_kl(terms.policy, self._reference(step.ema, batch))
            kl = Mean(jnp.sum(jnp.where(weights != 0, kl_terms, 0) * weights), mass)
            metrics["kl"], _ = mean_loss(kl)
            return Mean(pg.total + self.beta * kl.total, mass), Aux[Variables](metrics)
        return pg, Aux[Variables](metrics)

    def _policy_terms(self, terms: _Terms, mask: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """The chosen surrogate's per-token terms and verl's three actor metrics."""
        if self.policy_loss == "cispo":
            return cispo_terms(terms.policy, terms.old, terms.advantages, mask,
                               self.epsilon_low, self.epsilon_high)
        if self.policy_loss == "gspo":
            ratio = sequence_log_ratio(terms.policy, terms.old, mask, terms.segments)
            per_token, aux = clipped_surrogate_terms(ratio, terms.advantages, mask,
                                                     self.epsilon_low, self.epsilon_high, dual_clip=None)
            aux["ppo_kl"] = _fraction(-token_log_ratio(terms.policy, terms.old), mask)
            return per_token, aux
        return clipped_surrogate_terms(token_log_ratio(terms.policy, terms.old), terms.advantages, mask,
                                       epsilon_low=self.epsilon_low, epsilon_high=self.epsilon_high,
                                       dual_clip=self.dual_clip)

    def _weights(self, terms: _Terms, effective: jax.Array, keep: jax.Array) -> jax.Array:
        """Each token's share of the loss mass under the chosen aggregation.

        Token-mean weighs every kept token 1. Rollout-mean weighs it one
        over its rollout's trainable count: the packed batch's own
        `rollout_weights` (rejected sequences drop out of the numerator and
        the mass alike), or the kept count of its row on a windowed batch.
        """
        effective = effective.astype(jnp.float32)
        if self.aggregation == "token-mean":
            return effective
        if terms.rollout_weights is not None:
            return terms.rollout_weights * keep * (effective != 0)
        if terms.segments is not None:
            raise ValueError("rollout-mean on a packed batch needs rollout_weights from pack")
        counts = jnp.sum(effective, axis=-1, keepdims=True)
        return effective / jnp.clip(counts, min=1.0)

    def evaluate(self, params, batch, step):
        """Score the prompts' perplexity under the policy.

        Each row is its shifted cross entropy with the real suffix as
        weights, taken off the row's `prompt_length`. Pads predict nothing
        and count nothing.
        """
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
        padding = prompts.shape[1] - lengths
        aligned = _shift_rows(prompts, padding)
        hidden = self.model.apply(params, aligned[:, :-1], train=False,
                                  method=type(self.model).hidden_states)
        head = self.model.apply(params, params["params"],
                                method=type(self.model).head_weight)
        losses, _, _ = chunked_cross_entropy(
            hidden, head, aligned[:, 1:], self.head_chunks,
            softcap=self.model.final_logit_softcap,
            precision=self.model.precision)
        losses, valid = _unpadded(losses, padding)
        return TokenScores(losses=losses, weights=valid.astype(losses.dtype))


def _fraction(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean of `values` over the positions `mask` keeps; zero when it keeps none."""
    keep = jnp.asarray(mask, jnp.float32)
    return jnp.sum(jnp.where(keep != 0, jnp.asarray(values, jnp.float32), 0) * keep) / jnp.clip(
        jnp.sum(keep), min=1.0)
