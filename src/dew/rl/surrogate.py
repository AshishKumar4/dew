# Copyright 2026 Google LLC
# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Update rules, which turn log-probabilities and advantages into a loss.

A port of the array math inside Tunix's `grpo_loss_fn`
(`tunix/rl/algo_core.py`, commit b9f5e65: the importance ratio, the clipped
surrogate, the dual clip, the sequence-level ratio of GSPO with its
stop-gradient trick) and of Tunix's `compute_kl_divergence`
(`tunix/rl/common.py`), read against verl's `compute_policy_loss_vanilla`,
`compute_policy_loss_gspo`, `agg_loss` and `kl_penalty_forward`
(`verl/trainer/ppo/core_algos.py`, commit 896a9bb). Both projects are
Apache-2.0, and this file carries their notice.

Where the two references disagree, this file follows verl, whose functions
tools/parity_rl.py can call directly and tests/test_rl_surrogate.py compares
against. Both disagreements are written where they happen. Tunix's surrogate
lives behind a model forward inside `grpo_loss_fn`. It is a reading
reference; the callable checks run against verl.

Everything here is array math on `[B, T]` arrays. A policy forward, a
reference forward and a reward all happen outside.
"""


import math

import jax
import jax.numpy as jnp

from dew.rl.advantage import MEAN_EPS, masked_mean

LOG_RATIO_CLAMP = 20.0
"""Bound on a token's log importance ratio before it is exponentiated. Both
references clamp here, symmetrically, so `exp` cannot overflow fp32 on a
policy that has drifted."""

SEQUENCE_RATIO_CLAMP = 10.0
"""Upper bound on the sequence-level log ratio, verl's and Tunix's. Only the
upper side is clamped, since only an exploding ratio threatens what is
exponentiated."""

KL_DIFF_CLAMP = 20.0
KL_CLAMP = 10.0
"""Bounds inside the k3 estimator, verl's `kl_penalty_forward`. Tunix's
`compute_kl_divergence` leaves both off by default and clamps symmetrically
when asked, so the two agree wherever the estimate stays under 10."""


def token_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Sum over the unmasked positions of `x`, divided by how many there are.

    verl's `agg_loss(loss_agg_mode="token-mean")` and Tunix's
    `aggregate_loss("token-mean")`. The denominator is the exact token count,
    without `masked_mean`'s 1e-8. A loss that is 1e-8 off scales the gradient
    by the same factor, so both references keep the two reductions separate.
    A batch with no unmasked token divides by zero and surfaces as a nan.
    """
    weights = mask.astype(x.dtype)
    return jnp.sum(jnp.where(weights != 0, x, 0) * weights) / jnp.sum(weights)


def token_log_ratio(log_probs: jax.Array, old_log_probs: jax.Array) -> jax.Array:
    """Per-token `log pi(a) - log pi_old(a)`, clamped to +-20.

    verl calls this `negative_approx_kl` and negates it for its `ppo_kl`
    metric, which is `-token_mean(token_log_ratio(...), mask)`.
    """
    log_ratio = jnp.asarray(log_probs, jnp.float32) - jnp.asarray(old_log_probs, jnp.float32)
    return jnp.clip(log_ratio, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)


def _segment_totals(values: jax.Array, mask: jax.Array,
                    segments: jax.Array | None) -> tuple[jax.Array, jax.Array]:
    """Each position's sequence masked sum and unmasked count, broadcast to `[B, T]`.

    A sequence never crosses a row, so each row sums its own segments: under
    data parallelism the reduction stays on the device that holds the row.
    Without `segments` each row is one sequence.
    """
    values = jnp.asarray(values, jnp.float32)
    keep = mask.astype(jnp.float32)
    kept = jnp.where(keep != 0, values, 0) * keep
    width = values.shape[1]
    keys = (jnp.ones(values.shape, jnp.int32) if segments is None
            else jnp.asarray(segments, jnp.int32))
    per_row = jax.vmap(lambda row, ids: jax.ops.segment_sum(row, ids, num_segments=width + 1))
    total, counted = per_row(kept, keys), per_row(keep, keys)
    return jnp.take_along_axis(total, keys, axis=1), jnp.take_along_axis(counted, keys, axis=1)


def segment_mean(values: jax.Array, mask: jax.Array, segments: jax.Array | None = None) -> jax.Array:
    """Each position's masked mean over its sequence, broadcast back to `[B, T]`.

    A sequence is a row, or with `segments` one run of equal nonzero ids in
    a row, the packed layout's chain. The denominator is the sequence's
    unmasked count clipped at one, verl's `clamp(min=1)`, so a fully masked
    sequence pools to zero.
    """
    total, count = _segment_totals(values, mask, segments)
    return total / jnp.clip(count, min=1.0)


def sequence_log_ratio(log_probs: jax.Array, old_log_probs: jax.Array,
                       mask: jax.Array, segments: jax.Array | None = None) -> jax.Array:
    """GSPO's sequence-level log ratio, carrying a per-token gradient.

    The sequence ratio is the geometric mean of the token ratios, so its log is
    the masked mean of the token log ratios (arXiv:2507.18071, equation 6).
    Written as `logp - sg(logp) + sg(mean)` the value of every token in a
    sequence is that one mean, while the derivative with respect to each token's
    log-probability is that token's own. Dropping either stop-gradient leaves
    the value untouched and changes every gradient. The test pins the
    gradients for that reason.

    A sequence is a row, or with `segments` one packed chain of it
    (`segment_mean`), which is what verl's per-row pooling sees when each
    chain is its own row.

    Tunix clamps the token log ratios to +-20 before pooling them. This pools
    the raw difference, as verl's `compute_policy_loss_gspo` does. The clamp
    at 10 on the result bounds what is exponentiated either way.
    """
    log_probs = jnp.asarray(log_probs, jnp.float32)
    log_ratio = log_probs - jnp.asarray(old_log_probs, jnp.float32)
    pooled = segment_mean(log_ratio, mask, segments)
    sequence = (log_probs - jax.lax.stop_gradient(log_probs)
                + jax.lax.stop_gradient(pooled))
    return jnp.clip(sequence, max=SEQUENCE_RATIO_CLAMP)


def clipped_surrogate_terms(log_ratio: jax.Array, advantages: jax.Array, mask: jax.Array,
                            epsilon_low: float = 0.2, epsilon_high: float = 0.2,
                            dual_clip: float | None = 3.0) -> tuple[jax.Array, dict[str, jax.Array]]:
    """PPO policy terms before normalization, with the dual clip.

    `max(-A r, -A clip(r, 1 - eps_low, 1 + eps_high))` per token, and for a
    negative advantage the dual clip caps the term at `-A * dual_clip`
    (arXiv:1912.09729). Without the cap one token's ratio can dominate a step.
    `advantages` is `[B]`, one per completion, or `[B, T]` when a run
    scores tokens, the shape branch Tunix's `grpo_loss_fn` carries; a `[B]`
    column broadcasts over the sequence.

    Aux carries `pg_clipfrac`, `pg_clipfrac_lower` and `ppo_kl`, verl's three
    metrics, each read over the unmasked positions. They describe the ratio
    handed in, so with `sequence_log_ratio` the `ppo_kl` entry is the
    sequence-pooled quantity, and a GSPO run reads its `ppo_kl` from
    `token_log_ratio`, as verl's GSPO loss does.

    `dual_clip=None` leaves the negative side uncapped, verl's
    `compute_policy_loss_gspo`, and reports `pg_clipfrac_lower` as zero.
    """
    if dual_clip is not None and dual_clip <= 1.0:
        raise ValueError("the dual clip caps a negative advantage, so it needs "
                         f"dual_clip > 1, got {dual_clip}")

    log_ratio = jnp.asarray(log_ratio, jnp.float32)
    advantages = jnp.asarray(advantages, jnp.float32)
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    keep = mask.astype(jnp.float32)

    ratio = jnp.exp(log_ratio)
    unclipped = -advantages * ratio
    clipped = -advantages * jnp.clip(ratio, 1 - epsilon_low, 1 + epsilon_high)
    worse = jnp.maximum(unclipped, clipped)
    aux = {
        "pg_clipfrac": masked_mean(jnp.greater(clipped, unclipped).astype(jnp.float32), keep),
        "pg_clipfrac_lower": jnp.zeros((), jnp.float32),
        "ppo_kl": masked_mean(-log_ratio, keep),
    }
    if dual_clip is None:
        return worse, aux
    capped = -advantages * dual_clip
    negative = advantages < 0.0
    per_token = jnp.where(negative, jnp.minimum(capped, worse), worse)
    aux["pg_clipfrac_lower"] = masked_mean(
        jnp.greater(worse, capped).astype(jnp.float32) * negative.astype(jnp.float32), keep)
    return per_token, aux


def cispo_terms(log_probs: jax.Array, old_log_probs: jax.Array, advantages: jax.Array,
                mask: jax.Array, epsilon_low: float = 0.2,
                epsilon_high: float = 0.2) -> tuple[jax.Array, dict[str, jax.Array]]:
    """CISPO's policy terms: `-sg(clip(r, 1 - eps_low, 1 + eps_high)) * A * log pi`.

    MiniMax-M1 section 3.1 (arXiv:2506.13585), as verl's
    `compute_policy_loss_cispo` at 12ebe0c writes it: the ratio is clamped in
    log space to +-20, clipped, then detached, so every token keeps the
    gradient of its own log-probability, however far its ratio moved.
    `pg_clipfrac` counts tokens whose ratio the clip changed.
    """
    log_probs = jnp.asarray(log_probs, jnp.float32)
    log_ratio = token_log_ratio(log_probs, old_log_probs)
    advantages = jnp.asarray(advantages, jnp.float32)
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    keep = mask.astype(jnp.float32)
    ratio = jnp.exp(log_ratio)
    clipped = jnp.clip(ratio, 1 - epsilon_low, 1 + epsilon_high)
    terms = -jax.lax.stop_gradient(clipped) * advantages * log_probs
    aux = {
        "pg_clipfrac": masked_mean((ratio != clipped).astype(jnp.float32), keep),
        "pg_clipfrac_lower": jnp.zeros((), jnp.float32),
        "ppo_kl": masked_mean(-log_ratio, keep),
    }
    return terms, aux


def clipped_surrogate(log_ratio: jax.Array, advantages: jax.Array, mask: jax.Array,
                      epsilon_low: float = 0.2, epsilon_high: float = 0.2,
                      dual_clip: float | None = 3.0) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Token-mean reduction of the dual-clipped policy terms."""
    terms, aux = clipped_surrogate_terms(
        log_ratio, advantages, mask, epsilon_low, epsilon_high, dual_clip)
    return token_mean(terms, mask), aux


def k3_kl(log_probs: jax.Array, ref_log_probs: jax.Array) -> jax.Array:
    """Schulman's k3 estimator of `KL(pi || pi_ref)`, per token.

    `exp(d) - d - 1` for `d = log pi_ref - log pi`, which is non-negative,
    unbiased and lower variance than `-d` (http://joschu.net/blog/kl-approx.html).
    verl's `kl_penalty_forward("k3")` clamps `d` to +-20 before the exponential
    and the estimate to +-10 after it, and this follows verl. Without the
    second clamp one drifted token contributes `exp(20)` to the penalty and
    dominates the step.

    Aggregate it the way the policy loss is aggregated, `token_mean(kl, mask)`,
    and add `beta` times that.
    """
    diff = jnp.asarray(ref_log_probs, jnp.float32) - jnp.asarray(log_probs, jnp.float32)
    diff = jnp.clip(diff, -KL_DIFF_CLAMP, KL_DIFF_CLAMP)
    return jnp.clip(jnp.exp(diff) - diff - 1, -KL_CLAMP, KL_CLAMP)


def preference_logsigmoid_terms(policy_chosen: jax.Array, policy_rejected: jax.Array,
                                ref_chosen: jax.Array, ref_rejected: jax.Array,
                                mask_chosen: jax.Array, mask_rejected: jax.Array,
                                beta: float) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Per-pair DPO sigmoid terms and chosen/rejected reference-relative rewards.

    Equation 7 of arXiv:2305.18290 uses the difference of masked sequence
    policy/reference log-ratios. Rewards use beta times each log-ratio.
    Masks arrive shifted, one value per scored token.
    """
    chosen = (jnp.sum(policy_chosen * mask_chosen, axis=-1)
              - jnp.sum(ref_chosen * mask_chosen, axis=-1))
    rejected = (jnp.sum(policy_rejected * mask_rejected, axis=-1)
                - jnp.sum(ref_rejected * mask_rejected, axis=-1))
    terms = -jax.nn.log_sigmoid(beta * (chosen - rejected))
    return terms, (beta * chosen, beta * rejected)


def preference_logsigmoid(policy_chosen: jax.Array, policy_rejected: jax.Array,
                          ref_chosen: jax.Array, ref_rejected: jax.Array,
                          mask_chosen: jax.Array, mask_rejected: jax.Array,
                          beta: float) -> jax.Array:
    """Pair-mean reduction of the DPO sigmoid loss."""
    terms, _ = preference_logsigmoid_terms(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected,
        mask_chosen, mask_rejected, beta)
    return jnp.mean(terms)


def behavior_importance_weights(old_log_probs: jax.Array, behavior_log_probs: jax.Array,
                                mask: jax.Array, cap: float) -> jax.Array:
    """Detached token TIS weights from recorded raw and behavior policies.

    Port of verl compute_rollout_correction_weights(token), revision
    d040717b21af2e23e8e789a3e354cff2394ae2de: exponentiate the log ratio
    clamped to +-20, mask padding, then cap the weight. No batch normalization
    or rejection sampling is implied. Token TIS and filtered sampling do
    not recover an unbiased full-trajectory raw-policy expectation.
    """
    if type(cap) is bool or not cap > 0:
        raise ValueError("behavior importance cap must be positive")
    ratio = token_log_ratio(old_log_probs, behavior_log_probs)
    weights = jnp.where(mask != 0, jnp.exp(ratio) * mask, 0)
    return jax.lax.stop_gradient(jnp.minimum(weights, cap))


def behavior_band_weights(old_log_probs: jax.Array, behavior_log_probs: jax.Array,
                          mask: jax.Array, low: float, high: float) -> jax.Array:
    """Detached IcePop weights: the token ratio inside `[low, high]`, zero outside.

    verl 12ebe0c `compute_rollout_correction_weights(rollout_is="token")`
    with a `"low_high"` threshold: exp of the proximal-over-behavior log
    ratio clamped to +-20, masked, then zeroed outside the band instead of
    capped (arXiv:2510.18855). The band is inclusive at both ends.
    """
    if not 0 < low <= high:
        raise ValueError(f"an IcePop band needs 0 < low <= high, got [{low}, {high}]")
    ratio = jnp.exp(token_log_ratio(old_log_probs, behavior_log_probs))
    weights = jnp.where(mask != 0, ratio * mask, 0)
    return jax.lax.stop_gradient(jnp.where((weights >= low) & (weights <= high), weights, 0))


def sequence_rejection_mask(old_log_probs: jax.Array, behavior_log_probs: jax.Array,
                            mask: jax.Array, low: float, high: float, *, geometric: bool,
                            segments: jax.Array | None = None) -> jax.Array:
    """Keep 1 for every token of a sequence whose k1 statistic lies in `[log low, log high]`.

    verl 12ebe0c `compute_rollout_rejection_mask` with `seq_sum_k1`
    (`geometric=False`) or `seq_mean_k1` (`geometric=True`). verl's k1 is the
    negated log ratio, `log behavior - log proximal`, clamped to +-20 per
    token, then summed or averaged over the sequence; a sequence outside the
    band is rejected whole. The geometric form with a band near one is
    SkyRL's geometric sequence mask (0.99 to 1.01). A sequence is a row or a
    packed chain, as in `segment_mean`; outside `mask` the result is 1.
    """
    if not 0 < low <= high:
        raise ValueError(f"a rejection band needs 0 < low <= high, got [{low}, {high}]")
    k1 = -token_log_ratio(old_log_probs, behavior_log_probs)
    total, count = _segment_totals(k1, mask, segments)
    statistic = total / (count + MEAN_EPS) if geometric else total
    keep = (statistic >= math.log(low)) & (statistic <= math.log(high))
    return (keep | (mask == 0)).astype(jnp.float32)


def mismatch_metrics(proximal_log_probs: jax.Array, behavior_log_probs: jax.Array,
                     mask: jax.Array, weights: jax.Array | None = None,
                     cap: float | None = None) -> dict[str, jax.Array]:
    """Trainer-versus-engine diagnostics over the trainable tokens.

    verl 12ebe0c `compute_offpolicy_metrics`: `kl` is the direct estimate
    `mean(log behavior - log proximal)` and `k3_kl` the mean of
    `r - log r - 1` for `r = proximal / behavior`, the quantity prime-rl logs
    as `mismatch_kl`. `ess` is verl's `rollout_is_eff_sample_size`, one over
    the mean square of the applied weights clamped to `[0, cap]` and divided
    by their mean plus 1e-8; without applied `weights` it reads the raw
    token ratios. A value of one means every token weighs alike.
    """
    proximal = jnp.asarray(proximal_log_probs, jnp.float32)
    behavior = jnp.asarray(behavior_log_probs, jnp.float32)
    log_ratio = proximal - behavior
    keep = mask.astype(jnp.float32)
    if weights is None:
        weights = jnp.exp(jnp.clip(log_ratio, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)) * keep
    if cap is not None:
        weights = jnp.clip(weights, 0.0, cap)
    normalized = weights / (masked_mean(weights, keep) + MEAN_EPS)
    spread = masked_mean(jnp.square(normalized), keep)
    return {"kl": masked_mean(behavior - proximal, keep),
            "k3_kl": masked_mean(jnp.exp(log_ratio) - log_ratio - 1, keep),
            "ess": jnp.where(spread > 0, 1.0 / jnp.where(spread > 0, spread, 1.0), 0.0)}


def clipped_value_loss_terms(predicted: jax.Array, returns: jax.Array, old_values: jax.Array,
                             clip: float = 0.2) -> jax.Array:
    """verl d040717 compute_value_loss, before its token-mask reduction.

    The larger squared error of the live prediction and the prediction
    clipped around recorded values is multiplied by one half. Targets and
    recorded values are detached rollout data.
    """
    returns = jax.lax.stop_gradient(jnp.asarray(returns, jnp.float32))
    old_values = jax.lax.stop_gradient(jnp.asarray(old_values, jnp.float32))
    predicted = jnp.asarray(predicted, jnp.float32)
    clipped = jnp.clip(predicted, old_values - clip, old_values + clip)
    return 0.5 * jnp.maximum(jnp.square(predicted - returns), jnp.square(clipped - returns))
