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

from dew.nn.precision import at_least_fp32
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



def token_log_ratio(log_probs: jax.Array, old_log_probs: jax.Array) -> jax.Array:
    """Return the per-token log ratio `log pi(a) - log pi_old(a)`, clamped to +-20.

    verl calls this `negative_approx_kl`, and its `ppo_kl` metric is the
    masked token mean of its negation. The result is fp32, or the
    log-probabilities' dtype when that is wider (`at_least_fp32`).
    """
    log_probs = jnp.asarray(log_probs)
    work = at_least_fp32(log_probs.dtype)
    log_ratio = log_probs.astype(work) - jnp.asarray(old_log_probs, work)
    return jnp.clip(log_ratio, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)


def _segment_totals(values: jax.Array, mask: jax.Array,
                    segments: jax.Array | None) -> tuple[jax.Array, jax.Array]:
    """Each position's sequence masked sum and unmasked count, broadcast to `[B, T]`.

    A sequence never crosses a row, so each row sums its own segments: under
    data parallelism the reduction stays on the device that holds the row.
    Without `segments` each row is one sequence.
    """
    values = jnp.asarray(values)
    values = values.astype(at_least_fp32(values.dtype))
    keep = mask.astype(values.dtype)
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
    """Return GSPO's sequence-level log ratio, with a per-token gradient.

    The sequence ratio is the geometric mean of the token ratios, so its log is
    the masked mean of the token log ratios (arXiv:2507.18071, equation 6).
    The function computes it as `logp - sg(logp) + sg(mean)`, where `sg`
    stops the gradient. Every token in a sequence then has that one mean as
    its value, while the derivative with respect to each token's
    log-probability is that token's own. Dropping either stop-gradient
    leaves the value unchanged and changes every gradient.

    A sequence is a row or, with `segments`, one packed chain in a row (see
    `segment_mean`). That matches verl's per-row pooling when each chain is
    its own row.

    Tunix clamps the token log ratios to +-20 before pooling them. This
    function pools the raw difference, as verl's `compute_policy_loss_gspo`
    does. Either way, the result is clamped above at 10, which bounds what
    is exponentiated.
    """
    log_probs = jnp.asarray(log_probs)
    log_probs = log_probs.astype(at_least_fp32(log_probs.dtype))
    log_ratio = log_probs - jnp.asarray(old_log_probs, log_probs.dtype)
    pooled = segment_mean(log_ratio, mask, segments)
    # A test pins these gradients, because dropping a stop_gradient would not change the value.
    sequence = (log_probs - jax.lax.stop_gradient(log_probs)
                + jax.lax.stop_gradient(pooled))
    return jnp.clip(sequence, max=SEQUENCE_RATIO_CLAMP)


def clipped_surrogate_terms(log_ratio: jax.Array, advantages: jax.Array, mask: jax.Array,
                            epsilon_low: float = 0.2, epsilon_high: float = 0.2,
                            dual_clip: float | None = 3.0) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return PPO's per-token policy terms with the dual clip, before normalization, and their metrics.

    Each token's term is `max(-A r, -A clip(r, 1 - eps_low, 1 + eps_high))`,
    where `r = exp(log_ratio)` and the epsilons are `epsilon_low` and
    `epsilon_high`. For a negative advantage, the dual clip caps the term at
    `-A * dual_clip` (arXiv:1912.09729), because without the cap one token's
    ratio can dominate a step. `advantages` is `[B]`, one per completion, or
    `[B, T]` when a run scores tokens, a shape Tunix's `grpo_loss_fn` also
    handles. A `[B]` advantage broadcasts over the sequence.

    The metrics are verl's three, `pg_clipfrac`, `pg_clipfrac_lower` and
    `ppo_kl`, each averaged over the unmasked positions. They describe the
    ratio passed in, so with `sequence_log_ratio` the `ppo_kl` entry is the
    sequence-pooled quantity. A GSPO run therefore takes its `ppo_kl` from
    `token_log_ratio`, as verl's GSPO loss does.

    `dual_clip=None` leaves the negative side uncapped, as verl's
    `compute_policy_loss_gspo` does, and reports `pg_clipfrac_lower` as
    zero. A `dual_clip` of 1 or less raises ValueError.
    """
    if dual_clip is not None and dual_clip <= 1.0:
        raise ValueError("the dual clip caps a negative advantage, so it needs "
                         f"dual_clip > 1, got {dual_clip}")

    log_ratio = jnp.asarray(log_ratio)
    log_ratio = log_ratio.astype(at_least_fp32(log_ratio.dtype))
    advantages = jnp.asarray(advantages, log_ratio.dtype)
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
    log_probs = jnp.asarray(log_probs)
    log_probs = log_probs.astype(at_least_fp32(log_probs.dtype))
    log_ratio = token_log_ratio(log_probs, old_log_probs)
    advantages = jnp.asarray(advantages, log_probs.dtype)
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



def k3_kl(log_probs: jax.Array, ref_log_probs: jax.Array) -> jax.Array:
    """Return Schulman's k3 estimate of `KL(pi || pi_ref)` for each token.

    The estimate is `exp(d) - d - 1` for `d = log pi_ref - log pi`. It is
    non-negative, unbiased, and has lower variance than `-d`
    (http://joschu.net/blog/kl-approx.html). This follows verl's
    `kl_penalty_forward("k3")`, which clamps `d` to +-20 before the
    exponential and the estimate to +-10 after it. Without the second
    clamp, one drifted token would contribute `exp(20)` to the penalty and
    dominate the step.

    Aggregate it over the same tokens and in the same way as the policy
    loss, and add `beta` times the result.
    """
    log_probs = jnp.asarray(log_probs)
    work = at_least_fp32(log_probs.dtype)
    diff = jnp.asarray(ref_log_probs, work) - log_probs.astype(work)
    diff = jnp.clip(diff, -KL_DIFF_CLAMP, KL_DIFF_CLAMP)
    return jnp.clip(jnp.exp(diff) - diff - 1, -KL_CLAMP, KL_CLAMP)


def preference_logsigmoid_terms(policy_chosen: jax.Array, policy_rejected: jax.Array,
                                ref_chosen: jax.Array, ref_rejected: jax.Array,
                                mask_chosen: jax.Array, mask_rejected: jax.Array,
                                beta: float) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Return the per-pair DPO sigmoid loss terms and the chosen and rejected rewards.

    Each completion's log-ratio is the masked sum of its policy
    log-probabilities minus the masked sum of its reference
    log-probabilities. The term is `-log_sigmoid(beta * (chosen - rejected))`
    on those log-ratios (arXiv:2305.18290, equation 7), and each reward is
    `beta` times its log-ratio. The masks come already shifted, with one
    value per scored token.
    """
    chosen = (jnp.sum(policy_chosen * mask_chosen, axis=-1)
              - jnp.sum(ref_chosen * mask_chosen, axis=-1))
    rejected = (jnp.sum(policy_rejected * mask_rejected, axis=-1)
                - jnp.sum(ref_rejected * mask_rejected, axis=-1))
    terms = -jax.nn.log_sigmoid(beta * (chosen - rejected))
    return terms, (beta * chosen, beta * rejected)



def behavior_importance_weights(old_log_probs: jax.Array, behavior_log_probs: jax.Array,
                                mask: jax.Array, cap: float) -> jax.Array:
    """Return detached per-token TIS weights from the recorded raw-policy and behavior log-probabilities.

    This ports verl's `compute_rollout_correction_weights` for token-level
    weights. Each weight is the exponential of the log ratio
    `old_log_probs - behavior_log_probs`, clamped to +-20; padding gets
    weight zero, and every weight is capped at `cap`. There is no batch
    normalization and no rejection sampling. Token TIS and filtered sampling
    do not recover an unbiased raw-policy expectation over the full
    trajectory. A `cap` that is not positive raises ValueError.
    """
    # Ported from verl compute_rollout_correction_weights(token) at d040717b21af2e23e8e789a3e354cff2394ae2de.
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
    """Return PPO's clipped value-loss terms per token, before the token-mask reduction.

    This ports verl's `compute_value_loss`. Each term is half the larger of
    two squared errors against `returns`: the live prediction's, and that of
    the prediction clipped to within `clip` of the recorded `old_values`.
    `returns` and `old_values` are rollout data, so no gradient flows into
    them.
    """
    # Ported from verl compute_value_loss at d040717.
    predicted = jnp.asarray(predicted)
    predicted = predicted.astype(at_least_fp32(predicted.dtype))
    returns = jax.lax.stop_gradient(jnp.asarray(returns, predicted.dtype))
    old_values = jax.lax.stop_gradient(jnp.asarray(old_values, predicted.dtype))
    clipped = jnp.clip(predicted, old_values - clip, old_values + clip)
    return 0.5 * jnp.maximum(jnp.square(predicted - returns), jnp.square(clipped - returns))
