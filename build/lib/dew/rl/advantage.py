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
"""Advantage estimators, which turn a batch's rewards into its advantages.

A port of Tunix's `tunix/rl/algo_core.py` (`masked_mean`, `masked_var`,
`masked_whiten`, `compute_advantages`, `compute_drgrpo_advantages`,
`compute_rloo_advantages`, `compute_gae_advantages`, commit b9f5e65), read
against verl's `verl/trainer/ppo/core_algos.py`
(`compute_grpo_outcome_advantage`, `compute_rloo_outcome_advantage`,
`compute_gae_advantage_return`, commit 896a9bb) and verl's
`verl/utils/torch_functional.py` masked reductions. Both projects are
Apache-2.0, which is why this file carries their notice and Dew's other files
do not. tools/parity_rl.py runs both references on fixed tensors and
tests/test_rl_advantage.py holds the tolerances.

The constants are the references' own, and an argument wherever the references
disagree. Both use ddof 1 on the group deviation, 1e-6 under it, 1e-8 in every
masked mean, and the Bessel correction inside the whitening.

Everything here is array math. A function takes arrays and returns arrays, so
nothing in this file knows what a model, an objective or a trainer is.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

MEAN_EPS = 1e-8
"""Added to a masked mean's denominator, as both references add it. A fully
masked row then reads 0 rather than a nan that spreads through the batch."""

WHITEN_EPS = 1e-8
"""Added under the whitening's inverse square root, as both references add
it."""

GROUP_EPS = 1e-6
"""Default guard under the group deviation, verl's `epsilon` default and the
value Tunix hardcodes. verl-omni and TRL use 1e-4, so it is an argument."""


def masked_mean(x: jax.Array, mask: jax.Array, axis=None) -> jax.Array:
    """Mean of `x` over the positions `mask` keeps.

    Positions outside the mask are replaced rather than multiplied by zero,
    which is verl's form (`masked_sum`). A nan in a padded position survives a
    multiply by a zero mask and reaches the loss, and a padded position is
    exactly where an uninitialised value sits.
    """
    weights = mask.astype(x.dtype)
    kept = jnp.where(weights != 0, x, 0)
    return jnp.sum(kept * weights, axis=axis) / (jnp.sum(weights, axis=axis) + MEAN_EPS)


def masked_whiten(x: jax.Array, mask: jax.Array) -> jax.Array:
    """`x` centred and scaled by its masked mean and unbiased deviation.

    Both references whiten GAE advantages this way, Bessel correction and all,
    and both leave the positions outside the mask in the output for the loss to
    mask again. With one unmasked position the correction divides by zero.
    verl raises there, and this cannot, because it runs under jit.
    """
    mean = masked_mean(x, mask)
    variance = masked_mean(jnp.square(x - mean), mask)
    kept = jnp.sum(mask.astype(x.dtype))
    return (x - mean) * jax.lax.rsqrt(variance * (kept / (kept - 1)) + WHITEN_EPS)


def _grouped(rewards: jax.Array, group: int) -> jax.Array:
    """`[prompts, group]` float32 rewards, or a refusal that names the reason.

    A group of one has no baseline, and the three references answer it three
    ways (verl with the raw reward, Tunix with a nan from ddof 1 or with zeros,
    TRL with zeros), so a run that asks for one has a misconfigured group size.
    """
    if group < 2:
        raise ValueError(f"a group baseline needs group >= 2, got {group}")
    if rewards.ndim != 1:
        raise ValueError(
            f"rewards are one scalar per completion, [B], got {rewards.shape}")
    if rewards.shape[0] % group:
        raise ValueError(
            f"{rewards.shape[0]} rewards do not divide into groups of {group}")
    return jnp.asarray(rewards, jnp.float32).reshape(-1, group)


def group_advantage(rewards: jax.Array, group: int, normalise_by_std: bool = True,
                    eps: float = GROUP_EPS) -> jax.Array:
    """Group-relative advantage of `[B]` rewards, `group` completions per prompt.

    The rollout expands each prompt into `group` rows next to each other, so a
    group is a reshape and not an index array. verl groups by a `uid` column
    instead, which buys ragged groups Dew's rollout cannot produce.

    `normalise_by_std=False` is Dr.GRPO (arXiv:2503.20783), which subtracts the
    group mean and stops there. verl spells the same switch
    `norm_adv_by_std_in_grpo`.
    """
    grouped = _grouped(rewards, group)
    centred = grouped - jnp.mean(grouped, axis=-1, keepdims=True)
    if not normalise_by_std:
        return centred.reshape(-1)
    # ddof 1 is both references' choice, and jnp.std defaults to 0.
    deviation = jnp.std(grouped, axis=-1, ddof=1, keepdims=True)
    return (centred / (deviation + eps)).reshape(-1)


def rloo_advantage(rewards: jax.Array, group: int) -> jax.Array:
    """Each completion against the mean of the rest of its group.

    `r_i - mean(r_j, j != i)`, which is `group / (group - 1)` times the centred
    reward. Tunix writes the first form and verl the second, they are the same
    number, and this is Tunix's because the subtraction it performs is the
    definition (arXiv:2402.14740).
    """
    grouped = _grouped(rewards, group)
    others = (jnp.sum(grouped, axis=-1, keepdims=True) - grouped) / (group - 1)
    return (grouped - others).reshape(-1)


def gae(token_rewards: jax.Array, values: jax.Array, mask: jax.Array,
        gamma: float, lam: float) -> Tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over `[B, T]` rewards and values.

    `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)` and
    `A_t = delta_t + gamma * lam * A_{t+1}`, backwards from the last step
    (arXiv:1506.02438). A masked step contributes nothing, passing the running
    advantage and the next value through unchanged, so a padded tail cannot
    discount the real steps before it. Positions outside the mask hold the
    neighbouring step's carry rather than zero, which is what both references
    return and what the loss masks again.

    Returns the whitened advantages and the unwhitened returns, in that order
    and computed in that order. `returns = A + V` happens before the whitening
    in both references.

    The recursion runs in float32 whatever the caller's dtype, the way Tunix's
    GRPO loss casts its log-probabilities. The whitening subtracts two nearby
    numbers, and bf16 leaves nothing behind.
    """
    rewards = jnp.asarray(token_rewards, jnp.float32)
    values = jnp.asarray(values, jnp.float32)
    keep = jnp.asarray(mask, jnp.float32)

    def step(carry, inputs):
        advantage, next_value = carry
        reward_t, value_t, keep_t = inputs
        delta = reward_t + gamma * next_value - value_t
        candidate = delta + gamma * lam * advantage
        next_value = value_t * keep_t + (1 - keep_t) * next_value
        advantage = candidate * keep_t + (1 - keep_t) * advantage
        return (advantage, next_value), advantage

    zeros = jnp.zeros(values.shape[0], jnp.float32)
    _, transposed = jax.lax.scan(
        step, init=(zeros, zeros),
        xs=(rewards.T, values.T, keep.T), reverse=True)
    advantages = transposed.T
    return masked_whiten(advantages, keep), advantages + values
