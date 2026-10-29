"""Group-relative policy optimization over sampled rollouts.

A `GRPOObjective` scores tokens as `LMObjective` does (`TokenScorer`), and its
loss is the section 6 composition from `dew.rl`: a clipped policy surrogate plus `beta` times the
k3 KL against the frozen reference. It reads the one batch layout
`sessions.pack` builds, which every rollout in Dew produces: rows of strictly
merged chains with `text_segment_ids` and `text_positions`, every column
`[rows, width]` and aligned with `input_ids`. `behavior_log_probs` stands in
for `old_log_probs` when no proximal rescoring supplied one.

The reference is the objective's own frozen tree, `step.ema` at unit decay,
rescored only when `beta` is positive. Validation scores the prompts' own
perplexity, since a validation pass never samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from dew.artifacts import TokenScores
from dew.batch import Variables, real_slots, thaw
from dew.data.prompts import LENGTH_KEY, PROMPT_KEY
from dew.inputs import Field, InputSpec
from dew.nn.inputs import rotate_rows
from dew.nn.precision import at_least_fp32, compensated_sum, weighted_total
from dew.objectives.base import Aux, EMASpec, Ratio, Shown
from dew.objectives.lm.chunked import final_read
from dew.objectives.routing import ExpertBalance
from dew.rl import behavior_importance_weights, k3_kl, masked_mean, sequence_log_ratio, token_log_ratio
from dew.rl.surrogate import (
    behavior_band_weights,
    cispo_terms,
    clipped_surrogate_terms,
    gmpo_terms,
    mismatch_metrics,
    reinforce_terms,
    sequence_rejection_mask,
    sequence_sum,
)

from ..lm.objective import Scores, TokenScorer, _unpadded
from .sessions import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    IDS_KEY,
    OLD_LOG_PROBS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    ROUTED_EXPERTS_KEY,
    ROUTED_KEY,
    SEGMENT_IDS_KEY,
    SESSION_WEIGHTS_KEY,
    SUPPORT_COLUMNS_KEY,
    SUPPORT_KEY,
)

# ppo, gspo and cispo were checked against verl at 12ebe0c; every method against its paper's form
# (tests/test_rl_methods.py).
POLICY_LOSSES = ("ppo", "gspo", "cispo", "gmpo", "reinforce")
AGGREGATIONS = ("token-mean", "session-mean", "constant")
"""How the per-token terms reduce: over every kept token, over each session's then the sessions', or
each session's sum over a constant number of tokens (`aggregation_tokens`, Dr. GRPO's generation
budget, one for RLOO's sequence sums) averaged over the sessions."""
KL_TERMS = ("k3", "k2")
"""The reference penalty's estimator: k3 (GRPO's), or k2, half the squared log-ratio (REINFORCE++'s loss,
arXiv:2501.03262 equation 8 and appendix B.1)."""


class _Terms(NamedTuple):
    """The loss's inputs on the packed grid, aligned with `input_ids`: current,
    old and behavior log-probabilities, advantages, the trainable mask, chain
    ids, the per-session weights when the batch carries them, and whether the
    old policy was rescored rather than standing in for behavior. The policy's
    live QK collection and routers' choices come with them where the step asks."""

    policy: jax.Array
    old: jax.Array
    behavior: jax.Array
    advantages: jax.Array
    mask: jax.Array
    segments: jax.Array
    session_weights: jax.Array | None
    proximal: bool
    qk_stats: Variables | None
    routing: Variables | None


def onto_ids(batch, scores: jax.Array) -> jax.Array:
    """Move `[rows, width - 1]` scores of each prefix onto the id it predicts, as `[rows, width]`.

    Entry t then scores `input_ids[t]` from its chain's prefix, and it is zero
    where `response_mask` is, every chain start and all padding included.
    """
    wide = at_least_fp32(scores.dtype)
    shifted = jnp.concatenate([jnp.zeros_like(scores[:, :1], wide), scores.astype(wide)], axis=1)
    return jnp.where(jnp.asarray(batch[RESPONSE_MASK_KEY]) != 0, shifted, 0.0)


def _band(name: str, band) -> tuple[float, float] | None:
    if band is None:
        return None
    low, high = (float(value) for value in band)
    if not 0 < low <= high:
        raise ValueError(f"{name} is a (low, high) ratio band with 0 < low <= high, got {band}")
    return low, high


class GRPOObjective(TokenScorer):
    """Trains a policy on sampled rollouts with the GRPO loss (arXiv:2402.03300, equation 4).

    By default the loss is verl's: the dual-clipped surrogate, averaged over
    the tokens of the response mask, plus `beta` times the token-mean k3 KL
    to the frozen reference (`verl/trainer/ppo/core_algos.py`,
    `compute_policy_loss_vanilla` with `token-mean` and `kl_penalty_forward`
    with `k3`).

    `beta` is the KL strength, and at 0.0 no frozen reference is allocated.
    `kl_term` is its estimator (`KL_TERMS`). `reward_kl` is a reference
    penalty in the reward instead (RLOO's, arXiv:2402.14740 equation 3): the
    rollout subtracts `reward_kl` times each call's summed log-ratio of the
    proximal policy to the reference from that call's reward before its
    estimator compares returns (`RolloutScheduler`), and the loss adds no term.
    `epsilon_low`, `epsilon_high` and `dual_clip` are the clip points.
    `model`, `seq_len` and the scoring `options` are `TokenScorer`'s, with
    `seq_len` one less than the row width; the response mask says which
    targets count. A positive `beta` reads the starting policy, frozen at unit
    decay, as the reference.

    `balance` trains the routers of a mixture as `ExpertBalance` describes
    (`dew.objectives.routing`), over the response tokens the policy's forward
    reads; each chain of a packed row is a document of its own.

    `policy_loss` picks the surrogate:

    - `"ppo"`, verl's `compute_policy_loss_vanilla`.
    - `"gspo"`, verl's `compute_policy_loss_gspo`. It clips the sequence ratio
      from `sequence_log_ratio` and has no dual clip. A sequence here is a
      packed chain.
    - `"cispo"`, verl's `compute_policy_loss_cispo`.

    `aggregation` is `"token-mean"` (verl's default) or `"session-mean"`.
    Session-mean takes each session's token mean and averages those over
    sessions, so a long session, or one split over several rows, counts as
    one. That is Agent Lightning's `per_rollout_mean`, and verl's
    `seq-mean-token-mean` when a session is one row. The batch holds the
    weights in `session_weights`.

    Behavior corrections compare `behavior_log_probs` with the proximal
    policy (`old_log_probs`), all detached, as verl's `rollout_corr_helper`
    does. `behavior_importance` is a single threshold, like verl's
    `rollout_is_threshold`: a number caps the token ratio (TIS), and a
    `(low, high)` pair zeroes the ratio outside the band (IcePop).
    `sequence_mask` and `geometric_mask` reject every token of a sequence
    whose summed (`seq_sum_k1`) or mean (`seq_mean_k1`) k1 statistic lies
    outside `(log low, log high)`. Whenever behavior likelihoods are present,
    the metrics include `mismatch/kl`, `mismatch/k3_kl` and `mismatch/ess`,
    and the fraction of trainable tokens each correction masked.

    Without `old_log_probs`, the corrections follow verl's bypass mode. They
    compare the detached current policy with behavior, and the band only
    masks. A TIS cap is refused in that mode, because the ratio is already
    current over behavior.

    When a packed batch holds engine records, the loss replays them.
    `routed_experts`/`routed` make every router use the experts the engine
    used (R3). `support_ids`/`support_columns` renormalize each sampled id
    over the ids its top-k/top-p sampler kept, at `sampling_temperature`
    (applied after any final softcap), so the policy likelihood is compared
    with a filtered behavior likelihood (DeepSeek-V3.2 section 3.1). Set
    `sampling_temperature` to the engine's when the engine reports processed
    likelihoods; raw ones need 1.0.
    """

    # The loss is a policy-gradient surrogate: its value is no measure of
    # progress, so it is shown without a direction.
    shown: Mapping[str, Shown] = {"loss": Shown()}

    ema_is_reference = True

    counts_tokens = False
    """Rollout/packing geometry is supplied by its own objective count, not inherited LM rows."""

    keeps_whole_logits = False

    scored_fields = (PROMPT_KEY, LENGTH_KEY)
    """Validation scores the prompts alone, for their perplexity under the policy."""

    def __init__(self, model, seq_len: int, beta: float = 0.0,
                 epsilon_low: float = 0.2, epsilon_high: float = 0.2,
                 dual_clip: float = 3.0, *, policy_loss: str = "ppo", aggregation: str = "token-mean",
                 behavior_importance: float | tuple[float, float] | None = None,
                 sequence_mask: tuple[float, float] | None = None,
                 geometric_mask: tuple[float, float] | None = None,
                 sampling_temperature: float = 1.0, balance: ExpertBalance | None = None,
                 aggregation_tokens: int | None = None, kl_term: str = "k3", reward_kl: float = 0.0,
                 **options):
        if beta < 0 or reward_kl < 0:
            raise ValueError(f"beta and reward_kl scale a KL penalty, so they are non-negative, got {beta} "
                             f"and {reward_kl}")
        if policy_loss not in POLICY_LOSSES:
            raise ValueError(f"policy_loss must be one of {POLICY_LOSSES}, got {policy_loss!r}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        if (aggregation == "constant") != (aggregation_tokens is not None):
            raise ValueError("aggregation_tokens is the constant a session's sum is divided by, which the "
                             "constant aggregation alone reads and needs")
        if aggregation_tokens is not None and (type(aggregation_tokens) is not int or aggregation_tokens < 1):
            raise ValueError("aggregation_tokens is a positive number of tokens")
        if kl_term not in KL_TERMS:
            raise ValueError(f"kl_term must be one of {KL_TERMS}, got {kl_term!r}")
        self.aggregation_tokens = aggregation_tokens
        self.kl_term = kl_term
        self.reward_kl = reward_kl
        super().__init__(model, seq_len, balance=balance, **options)
        # The reference is the starting policy, frozen at unit decay, where a KL term reads it.
        self.ema = EMASpec.constant(1.0 if beta > 0 or reward_kl > 0 else None)
        # The batch is packed `input_ids` rows, not the LM's text windows.
        self.inputs = InputSpec(sample=Field(IDS_KEY, (seq_len + 1,)))
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.dual_clip = dual_clip
        if isinstance(behavior_importance, tuple):
            behavior_importance = _band("behavior_importance", behavior_importance)
        elif behavior_importance is not None:
            if isinstance(behavior_importance, bool) or not behavior_importance > 0:
                raise ValueError("behavior_importance is a positive TIS cap, a (low, high) IcePop band, "
                                 "or None to disable correction")
            behavior_importance = float(behavior_importance)
        self.behavior_importance: float | tuple[float, float] | None = behavior_importance
        """The off-policy correction, normalised: a TIS cap (a float), an IcePop band (low, high),
        or None; the scheduler and the loss read this one value."""
        self.sequence_mask = _band("sequence_mask", sequence_mask)
        self.geometric_mask = _band("geometric_mask", geometric_mask)
        self.policy_loss = policy_loss
        self.aggregation = aggregation
        if not sampling_temperature > 0:
            raise ValueError(f"sampling_temperature is a positive temperature, got {sampling_temperature}")
        self.sampling_temperature = sampling_temperature

    def reference_penalty(self, policy: jax.Array, reference: jax.Array, weights: jax.Array,
                          mass: jax.Array) -> Ratio:
        """The loss's reference penalty over the current likelihoods `policy` and the reference's, by
        `kl_term` (k3, or k2: half the squared log-ratio), reduced as the policy term is: `weights` and
        `mass` are `policy_ratio`'s, so it is a token, session or constant mean as the method's is."""
        delta = policy - jax.lax.stop_gradient(reference)
        # Halved before it is squared out: k2 is representable wherever its log-ratio's square halved is.
        terms = k3_kl(policy, reference) if self.kl_term == "k3" else delta * (0.5 * delta)
        return Ratio(weighted_total(terms, weights)[0], mass)

    def packed_log_probs(self, params: Variables, batch) -> jax.Array:
        """Return each packed id's log-probability given its own chain's prefix, placed by `onto_ids`.

        The loss and a proximal rescoring share its scoring path. A
        packed column whose shape differs from `input_ids` raises `ValueError`.
        Where its step asks for the attention logit maxima (`Step.qk_stats`), or the
        routers' choices for a balance (`balance`), the loss scores through
        `_packed_scores`, which returns them from the same forward, so an override
        of this method must override that one too.
        """
        return self._packed_scores(params, batch)[0]

    def _packed_scores(self, params: Variables, batch, *, qk_stats: bool = False, routing: bool = False
                       ) -> tuple[jax.Array, Scores]:
        """The packed policy likelihoods and their live `Scores`: the QK collection where `qk_stats` asks
        for it, and the routers' choices where `routing` does."""
        ids = jnp.asarray(batch[IDS_KEY], jnp.int32)
        segments = jnp.asarray(batch[SEGMENT_IDS_KEY], jnp.int32)
        for key in (SEGMENT_IDS_KEY, POSITIONS_KEY, RESPONSE_MASK_KEY):
            if jnp.shape(batch[key]) != ids.shape:
                raise ValueError(f"{key} has shape {jnp.shape(batch[key])}; a packed column has "
                                 f"the shape of {IDS_KEY}, {ids.shape}")
        routes = (None if ROUTED_EXPERTS_KEY not in batch
                  else (batch[ROUTED_EXPERTS_KEY], batch.get(ROUTED_KEY)))
        scores = self.token_scores(params, ids, segment_ids=segments,
                                   positions=jnp.asarray(batch[POSITIONS_KEY], jnp.int32), routes=routes,
                                   qk_stats=qk_stats, routing=routing)
        support = (None if SUPPORT_KEY not in batch
                   else (batch[SUPPORT_KEY], batch[SUPPORT_COLUMNS_KEY]))
        sampled = self.sampled_log_probs(params, scores, ids, support, self.sampling_temperature)
        return onto_ids(batch, sampled), scores

    def _terms(self, params, batch, *, qk_stats: bool = False) -> _Terms:
        """Read a packed batch onto its own `[rows, width]` grid.

        The old policy is `old_log_probs` when a rescoring or the sampler
        supplied it, and the recorded behavior otherwise. The rollout's columns
        are read at the policy's precision, at least fp32, so a float64 twin's
        terms are its own float64 and a run's its float32.
        """
        shape = jnp.shape(batch[RESPONSE_MASK_KEY])
        for key in (ADVANTAGES_KEY, BEHAVIOR_LOG_PROBS_KEY, OLD_LOG_PROBS_KEY, SESSION_WEIGHTS_KEY):
            if key in batch and jnp.shape(batch[key]) != shape:
                raise ValueError(f"{key} has shape {jnp.shape(batch[key])}; a packed column has "
                                 f"the shape of {IDS_KEY}, {shape}")
        # `packed_log_probs` is the scoring seam a subclass may replace; the QK
        # collection MuonClip reads and the routers the balance reads come from the
        # same live forward when asked for.
        routed = self.balance is not None
        if qk_stats or routed:
            policy, scores = self._packed_scores(params, batch, qk_stats=qk_stats, routing=routed)
            qk, sown = scores.qk, scores.routing
        else:
            policy, qk, sown = self.packed_log_probs(params, batch), None, None
        dtype = at_least_fp32(policy.dtype)
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY], dtype)
        behavior = jnp.asarray(batch[BEHAVIOR_LOG_PROBS_KEY], dtype)
        proximal = OLD_LOG_PROBS_KEY in batch
        old = jnp.asarray(batch[OLD_LOG_PROBS_KEY], dtype) if proximal else behavior
        weights = batch.get(SESSION_WEIGHTS_KEY)
        return _Terms(policy, old, behavior,
                      jnp.asarray(batch[ADVANTAGES_KEY], dtype), mask,
                      jnp.asarray(batch[SEGMENT_IDS_KEY], jnp.int32),
                      None if weights is None else jnp.asarray(weights, dtype), proximal, qk, sown)

    def loss(self, variables, batch, step):
        """Compute the policy surrogate over the trainable tokens, plus the KL to the reference.

        The policy's log-probabilities are recomputed from the rollout's own
        ids, so every term uses the tokens that were actually sampled.
        """
        terms = self._terms(variables, batch, qk_stats=step.qk_stats)
        mask = terms.mask
        correction = self.behavior_importance
        if isinstance(correction, float) and not terms.proximal:
            raise ValueError(
                "without old_log_probs the ratio is already current over behavior, so a TIS cap "
                "would count the correction twice (verl's bypass mode applies no IS weight); "
                "rescore old_log_probs or give behavior_importance a (low, high) band")
        # Without a proximal rescoring, behavior stands in for the old policy,
        # and the corrections compare the detached current policy with behavior,
        # as verl's compute_policy_loss_bypass_mode does.
        proximal = terms.old if terms.proximal else jax.lax.stop_gradient(terms.policy)
        metrics: dict[str, jax.Array] = {}
        keep = jnp.ones_like(mask)
        for name, band, geometric in (("sequence", self.sequence_mask, False),
                                      ("geometric", self.geometric_mask, True)):
            if band is not None:
                rejected = sequence_rejection_mask(proximal, terms.behavior, mask, *band,
                                                   geometric=geometric, segments=terms.segments)
                metrics[f"masked/{name}"] = masked_mean(1 - rejected, mask)
                keep = keep * rejected
        # Weights and diagnostics read every trainable token, as verl's do;
        # rejection reaches the loss through `effective` alone.
        importance = None
        if isinstance(correction, float):
            importance = behavior_importance_weights(proximal, terms.behavior, mask, correction)
        elif isinstance(correction, tuple):
            importance = behavior_band_weights(proximal, terms.behavior, mask, *correction)
            metrics["masked/band"] = masked_mean(importance == 0, mask)
        cap = correction[1] if isinstance(correction, tuple) else correction
        mismatch = mismatch_metrics(proximal, terms.behavior, mask, importance, cap)
        metrics.update({f"mismatch/{key}": value for key, value in mismatch.items()})
        if importance is not None and not terms.proximal:
            # The ratio already carries current over behavior, so the band
            # acts as a keep mask and applies no weight.
            keep = keep * (importance != 0)
            importance = None
        effective = mask * keep
        pg, weights, mass, aux = self.policy_ratio(terms, effective, keep, importance)
        pg_loss, _ = pg.mean()
        metrics = {"pg": pg_loss, **{f"actor/{k}": v for k, v in aux.items()}, **metrics}
        if self.beta > 0:
            if step.ema is None:
                raise ValueError(
                    "the KL term reads step.ema, but the objective keeps no EMA; "
                    "a GRPO run with beta above zero always freezes one")
            kl = self.reference_penalty(terms.policy, self.packed_log_probs(step.ema, batch), weights, mass)
            metrics["kl"], _ = kl.mean()
            objective = Ratio(pg.total + self.beta * kl.total, mass)
        else:
            objective = pg
        statistics, effects = objective, None
        if self.balance is not None:
            # The router positions that count are the response tokens the policy's forward reads: a token's
            # input (`[:, :-1]`) whose own target is trained, in a real chain.
            fields = {"segment_ids": terms.segments}
            queries = (real_slots(fields, mask.shape)[:, :-1] & (mask[:, :-1] != 0)
                       & (self.row_weights(batch, mask.shape[0]) > 0)[:, None])
            statistics, effects, router_metrics = self.balance.compose(
                variables, objective, terms.routing or {}, batch, queries, fields)
            metrics.update(router_metrics)
        return statistics, Aux[Variables](metrics, qk_stats=terms.qk_stats, effects=effects)

    def policy_ratio(self, terms: _Terms, effective: jax.Array, keep: jax.Array, importance=None
                     ) -> tuple[Ratio, jax.Array, jax.Array, dict[str, jax.Array]]:
        """The method's policy term as the loss reduces it (`Ratio`), each token's weight, the mass and
        the actor metrics, over `effective`, the trained tokens `keep` admits, each term scaled by the
        behavior `importance` where given: the policy loss's per-token terms (`_policy_terms`) under
        the aggregation (`_weights`)."""
        per_token, aux = self._policy_terms(terms, effective)
        if importance is not None:
            per_token = per_token * importance
        weights = self._weights(terms, effective, keep)
        # Summed once rounded (`compensated_sum`): the session weights' sum is their count.
        mass = jax.lax.stop_gradient(compensated_sum(weights, tuple(range(weights.ndim))))
        if self.aggregation_tokens is not None:
            # The constant aggregation: each session's sum over the constant, averaged over the sessions
            # (their session weights sum to their count), a kept token weighing one over the constant.
            weights = effective.astype(weights.dtype) / self.aggregation_tokens
        if self.policy_loss in ("gspo", "gmpo"):
            # A sequence's tokens share its term: summed once a sequence, as its paper's form sums it.
            total = sequence_sum(per_token, weights, terms.segments)
        else:
            total, _ = weighted_total(per_token, weights)
        return Ratio(total, mass), weights, mass, aux

    def _policy_terms(self, terms: _Terms, mask: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """The chosen surrogate's per-token terms and verl's three actor metrics."""
        if self.policy_loss == "cispo":
            return cispo_terms(terms.policy, terms.old, terms.advantages, mask,
                               self.epsilon_low, self.epsilon_high)
        if self.policy_loss == "gmpo":
            return gmpo_terms(terms.policy, terms.old, terms.advantages, mask, terms.segments,
                              self.epsilon_low, self.epsilon_high)
        if self.policy_loss == "reinforce":
            return reinforce_terms(terms.policy, terms.advantages, mask)
        if self.policy_loss == "gspo":
            ratio = sequence_log_ratio(terms.policy, terms.old, mask, terms.segments)
            per_token, aux = clipped_surrogate_terms(ratio, terms.advantages, mask,
                                                     self.epsilon_low, self.epsilon_high, dual_clip=None)
            aux["ppo_kl"] = masked_mean(-token_log_ratio(terms.policy, terms.old), mask)
            return per_token, aux
        return clipped_surrogate_terms(token_log_ratio(terms.policy, terms.old), terms.advantages, mask,
                                       epsilon_low=self.epsilon_low, epsilon_high=self.epsilon_high,
                                       dual_clip=self.dual_clip)

    def _weights(self, terms: _Terms, effective: jax.Array, keep: jax.Array) -> jax.Array:
        """Each token's share of the loss mass under the chosen aggregation.

        Token-mean weighs every kept token 1. Session-mean weighs it one
        over its session's trainable count, the batch's own `session_weights`;
        rejected sequences drop out of the numerator and the mass alike.
        """
        effective = effective.astype(at_least_fp32(terms.policy.dtype))
        if self.aggregation == "token-mean":
            return effective
        if terms.session_weights is None:
            raise ValueError(f"{self.aggregation} aggregation reads {SESSION_WEIGHTS_KEY} from pack")
        return terms.session_weights * keep * (effective != 0)

    def validation_loss(self, variables, batch, step):
        """Score the source prompts' NLL, not the surrogate's rollout-only fields."""
        return self._validation_scores(variables, batch, step)[0]

    def _validation_scores(self, params, batch, step):
        """The prompts' NLL with the per-token scores it is the mean of."""
        scores = self._evaluation_scores(params, batch, step)
        return self.row_mean(scores.losses, batch, weights=scores.weights), scores

    def _evaluation_scores(self, params, batch, step) -> TokenScores:
        """The prompts' per-token scores.

        Prompts are left-padded, so a row's real tokens are its last
        `prompt_length` columns. Each row's score is its shifted cross entropy
        with those real tokens as the weights; padding neither predicts nor
        counts.
        """
        params = thaw(params)
        prompts = jnp.asarray(batch[PROMPT_KEY])
        lengths = jnp.asarray(batch[LENGTH_KEY]).reshape(-1)
        if prompts.shape[1] < 2:
            raise ValueError(
                f"a prompt needs two tokens to score one target, got {prompts.shape[1]}")
        padding = prompts.shape[1] - lengths
        aligned = rotate_rows(prompts, padding)
        read = final_read(self.model, params, aligned[:, :-1], train=False)
        losses, predicted, _ = self.head_terms(self.model, params, read, aligned[:, 1:], predict=True)
        assert predicted is not None
        correct, _ = _unpadded(predicted == aligned[:, 1:], padding)
        losses, valid = _unpadded(losses, padding)
        return TokenScores(losses=losses, weights=valid.astype(losses.dtype), correct=correct)

