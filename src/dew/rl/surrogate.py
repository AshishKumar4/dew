# Copyright 2026 Google LLC
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Apache-2.0, which is why this file carries their notice.

Where the two references disagree, this file follows verl, whose functions
tools/parity_rl.py can call directly and tests/test_rl_surrogate.py compares
against. Both disagreements are written where they happen. Tunix's surrogate
lives behind a model forward inside `grpo_loss_fn`, so it is a reading
reference here and not a callable one.

Everything here is array math on `[B, T]` arrays. A policy forward, a
reference forward and a reward all happen outside.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from dew.rl.advantage import masked_mean

LOG_RATIO_CLAMP = 20.0
"""Bound on a token's log importance ratio before it is exponentiated. Both
references clamp here, symmetrically, so `exp` cannot overflow fp32 on a
policy that has drifted."""

SEQUENCE_RATIO_CLAMP = 10.0
"""Upper bound on the sequence-level log ratio, verl's and Tunix's. Only the
upper side, because a vanishing ratio is harmless and an exploding one is
not."""

KL_DIFF_CLAMP = 20.0
KL_CLAMP = 10.0
"""Bounds inside the k3 estimator, verl's `kl_penalty_forward`. Tunix's
`compute_kl_divergence` leaves both off by default and clamps symmetrically
when asked, so the two agree wherever the estimate stays under 10."""


def token_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Sum over the unmasked positions of `x`, divided by how many there are.

    verl's `agg_loss(loss_agg_mode="token-mean")` and Tunix's
    `aggregate_loss("token-mean")`. The denominator is the exact token count,
    with none of `masked_mean`'s 1e-8, because a loss that is 1e-8 off scales
    the gradient by the same factor. Both references keep the two reductions
    separate for that reason. A batch with no unmasked token divides by zero,
    which is a loud nan rather than a step on nothing.
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


def sequence_log_ratio(log_probs: jax.Array, old_log_probs: jax.Array,
                       mask: jax.Array) -> jax.Array:
    """GSPO's sequence-level log ratio, carrying a per-token gradient.

    The sequence ratio is the geometric mean of the token ratios, so its log is
    the masked mean of the token log ratios (arXiv:2507.18071, equation 6).
    Written as `logp - sg(logp) + sg(mean)` the value of every token in a
    sequence is that one mean, while the derivative with respect to each token's
    log-probability stays that token's own. Dropping either stop-gradient
    leaves the value untouched and changes every gradient, which is why the
    test that defends this asserts gradients.

    Tunix clamps the token log ratios to +-20 before pooling them and this does
    not, because verl's `compute_policy_loss_gspo` pools the raw difference.
    The clamp at 10 on the result bounds what is exponentiated either way.
    """
    log_probs = jnp.asarray(log_probs, jnp.float32)
    log_ratio = log_probs - jnp.asarray(old_log_probs, jnp.float32)
    keep = mask.astype(jnp.float32)
    pooled = (jnp.sum(log_ratio * keep, axis=-1)
              / jnp.clip(jnp.sum(keep, axis=-1), min=1.0))
    sequence = (log_probs - jax.lax.stop_gradient(log_probs)
                + jax.lax.stop_gradient(pooled)[:, None])
    return jnp.clip(sequence, max=SEQUENCE_RATIO_CLAMP)


def clipped_surrogate(log_ratio: jax.Array, advantages: jax.Array, mask: jax.Array,
                      epsilon_low: float = 0.2, epsilon_high: float = 0.2,
                      dual_clip: float = 3.0) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """PPO's clipped policy loss, token-mean, with the dual clip.

    `max(-A r, -A clip(r, 1 - eps_low, 1 + eps_high))` per token, and for a
    negative advantage the dual clip caps the term at `-A * dual_clip`
    (arXiv:1912.09729), which is what stops one token's ratio from dominating a
    step. `advantages` is `[B]`, one per completion, or `[B, T]` when a run
    scores tokens, the shape branch Tunix's `grpo_loss_fn` carries; a `[B]`
    column broadcasts over the sequence.

    Aux carries `pg_clipfrac`, `pg_clipfrac_lower` and `ppo_kl`, verl's three
    metrics, each read over the unmasked positions. They describe the ratio
    handed in, so with `sequence_log_ratio` the `ppo_kl` entry is the
    sequence-pooled quantity, and a GSPO run reads its `ppo_kl` from
    `token_log_ratio`, as verl's GSPO loss does.
    """
    if dual_clip <= 1.0:
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
    capped = -advantages * dual_clip
    negative = advantages < 0.0
    per_token = jnp.where(negative, jnp.minimum(capped, worse), worse)

    aux = {
        "pg_clipfrac": masked_mean(jnp.greater(clipped, unclipped).astype(jnp.float32), keep),
        "pg_clipfrac_lower": masked_mean(
            jnp.greater(worse, capped).astype(jnp.float32) * negative.astype(jnp.float32),
            keep),
        "ppo_kl": masked_mean(-log_ratio, keep),
    }
    return token_mean(per_token, keep), aux


def k3_kl(log_probs: jax.Array, ref_log_probs: jax.Array) -> jax.Array:
    """Schulman's k3 estimator of `KL(pi || pi_ref)`, per token.

    `exp(d) - d - 1` for `d = log pi_ref - log pi`, which is non-negative,
    unbiased and lower variance than `-d` (http://joschu.net/blog/kl-approx.html).
    verl's `kl_penalty_forward("k3")` clamps `d` to +-20 before the exponential
    and the estimate to +-10 after it, and this follows verl. Without the
    second clamp one drifted token contributes `exp(20)` to the penalty and
    owns the step.

    Aggregate it the way the policy loss is aggregated, `token_mean(kl, mask)`,
    and add `beta` times that.
    """
    diff = jnp.asarray(ref_log_probs, jnp.float32) - jnp.asarray(log_probs, jnp.float32)
    diff = jnp.clip(diff, -KL_DIFF_CLAMP, KL_DIFF_CLAMP)
    return jnp.clip(jnp.exp(diff) - diff - 1, -KL_CLAMP, KL_CLAMP)
