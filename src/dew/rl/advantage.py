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
Apache-2.0, and this file carries their notice. tools/parity_rl.py
runs both references on fixed tensors and
tests/test_rl_advantage.py holds the tolerances.

The constants are the references' own, and an argument wherever the references
disagree. Both use ddof 1 on the group deviation, 1e-6 under it, 1e-8 in every
masked mean, and the Bessel correction inside the whitening.

Everything here is array math. A function takes arrays and returns arrays, so
nothing in this file knows what a model, an objective or a trainer is.
"""


from collections.abc import Callable

import jax
import jax.numpy as jnp

from dew.nn.precision import at_least_fp32, weighted_total

MEAN_EPS = 1e-8
"""Added to a masked mean's denominator, as both references add it. Without
it a fully masked row is a nan that spreads through the batch."""

WHITEN_EPS = 1e-8
"""Added under the whitening's inverse square root, as both references add
it."""

GROUP_EPS = 1e-6
"""Default guard under the group deviation, verl's `epsilon` default and the
value Tunix hardcodes. verl-omni and TRL use 1e-4, so it is an argument."""


def masked_mean(x: jax.Array, mask: jax.Array, axis=None) -> jax.Array:
    """Return the mean of `x` over the positions `mask` keeps, along `axis` (all axes by default).

    Values outside the mask are replaced with zero through `jnp.where` before
    the multiply, as verl's `masked_sum` does. A nan in a padded position
    would survive a multiply by a zero mask and reach the loss, and padded
    positions are where uninitialised values sit. The denominator adds
    `MEAN_EPS`, so a fully masked input gives zero.
    """
    total, mass = weighted_total(x, mask, axis=axis)
    return total / (mass + MEAN_EPS)


def masked_whiten(x: jax.Array, mask: jax.Array,
                  total: Callable[[jax.Array], jax.Array] = lambda moments: moments) -> jax.Array:
    """Return `x` centred on its masked mean and scaled by its unbiased masked deviation.

    Tunix and verl both whiten GAE advantages this way, Bessel correction
    included. Both also leave the positions outside the mask in the output,
    for the loss to mask again. With only one unmasked position the
    correction divides by zero. verl raises an error there; this function
    runs under jit and cannot.

    `total` sums the moments over every share of the data the whitening
    reads, a pool's processes for PPO's rollout; by default `x` is all of it.
    The moments are centred, the mean first and then the squared deviations
    from it, so a shift of every value moves no whitened one.
    """
    weights = mask.astype(x.dtype)
    kept, count = weighted_total(x, weights)
    count, kept = total(jnp.stack([count, kept]))
    mean = kept / (count + MEAN_EPS)
    deviations = jnp.where(weights != 0, x - mean, 0)
    variance = total(weighted_total(jnp.square(deviations), weights)[0]) / (count + MEAN_EPS)
    return (x - mean) * jax.lax.rsqrt(variance * (count / (count - 1)) + WHITEN_EPS)


def _grouped(rewards: jax.Array, group: int) -> jax.Array:
    """`[prompts, group]` rewards, at least float32 (a float64 caller's stay float64).

    A group of one has no baseline, and the three references answer it three
    ways (verl with the raw reward, Tunix with a nan from ddof 1 or with zeros,
    TRL with zeros), so a run that asks for one has a misconfigured group size.
    """
    if group < 2:
        raise ValueError(f"a group baseline needs group >= 2, got {group}")
    if rewards.ndim != 1:
        raise ValueError(
            f"rewards are one scalar per completion, [B], got {rewards.shape}")
    return jnp.asarray(rewards, at_least_fp32(jnp.asarray(rewards).dtype)).reshape(-1, group)


def group_advantage(rewards: jax.Array, group: int, normalise_by_std: bool = True,
                    eps: float = GROUP_EPS) -> jax.Array:
    """Return the group-relative advantage of `[B]` rewards, with `group` completions per prompt.

    Each reward has its group's mean subtracted and is divided by the
    group's standard deviation (ddof 1) plus `eps`. The rollout puts each
    prompt's `group` rows next to each other, so a reshape forms the groups.
    verl groups by a `uid` column, which allows ragged groups, but Dew's
    rollout cannot produce them. A `group` below 2 raises ValueError, because
    a group of one has no baseline.

    `normalise_by_std=False` is Dr.GRPO (arXiv:2503.20783), which subtracts the
    group mean without scaling by the deviation. verl calls the same switch
    `norm_adv_by_std_in_grpo`.
    """
    grouped = _grouped(rewards, group)
    if not normalise_by_std:
        return (grouped - jnp.mean(grouped, axis=-1, keepdims=True)).reshape(-1)
    # ddof 1 is both references' choice, and jnp.std defaults to 0.
    return group_normalised(grouped, ddof=1, eps=eps).reshape(-1)


def group_normalised(grouped, *, ddof: int, eps: float, xp=jnp):
    """Each row of `grouped` `[groups, members]` less its mean, over its standard deviation
    (`ddof`) plus `eps`, in `grouped`'s dtype and array library (`xp`, NumPy for a host reference's
    float64): the arithmetic every group-relative advantage shares, each reference its own
    `ddof`, `eps` and dtype."""
    centred = grouped - xp.mean(grouped, axis=-1, keepdims=True)
    return centred / (xp.std(grouped, axis=-1, ddof=ddof, keepdims=True) + eps)


def rloo_advantage(rewards: jax.Array, group: int) -> jax.Array:
    """Return each completion's reward minus the mean reward of the rest of its group.

    That is `r_i - mean(r_j, j != i)`, which equals `group / (group - 1)`
    times the centred reward (arXiv:2402.14740). Tunix writes the first form
    and verl the second; this function follows Tunix.
    """
    grouped = _grouped(rewards, group)
    others = (jnp.sum(grouped, axis=-1, keepdims=True) - grouped) / (group - 1)
    return (grouped - others).reshape(-1)


def gae_advantages(token_rewards: jax.Array, values: jax.Array, mask: jax.Array,
                   gamma: float, lam: float) -> jax.Array:
    """The unwhitened generalized advantage estimates of `[B, T]` rewards and values.

    The recursion runs backwards from the last step, with
    `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)` and
    `A_t = delta_t + gamma * lam * A_{t+1}` (arXiv:1506.02438). A masked step
    contributes nothing and passes the running advantage and the next value
    through unchanged, so a padded tail does not discount the real steps
    before it, and its reward and value are never read, so a NaN there
    reaches no real step. Positions outside the mask hold the neighbouring
    step's carry, as in both references, and the loss masks them again.

    The recursion runs in at least float32, the way Tunix's GRPO loss casts
    its log-probabilities (a float64 caller's, a float64 truth's, in float64). A caller whitens the advantages
    itself (`gae` over its batch, `PPORollout` over every process's), from
    these rather than from returns less values: at values far above the
    advantages, float32 rounds those differences away.
    """
    keep = jnp.asarray(mask) != 0
    wide = at_least_fp32(jnp.result_type(token_rewards, values))
    rewards = jnp.where(keep, jnp.asarray(token_rewards, wide), 0)
    values = jnp.where(keep, jnp.asarray(values, wide), 0)

    def step(carry, inputs):
        advantage, next_value = carry
        reward_t, value_t, keep_t = inputs
        delta = reward_t + gamma * next_value - value_t
        candidate = delta + gamma * lam * advantage
        advantage = jnp.where(keep_t, candidate, advantage)
        next_value = jnp.where(keep_t, value_t, next_value)
        return (advantage, next_value), advantage

    zeros = jnp.zeros(values.shape[0], wide)
    _, transposed = jax.lax.scan(
        step, init=(zeros, zeros),
        xs=(rewards.T, values.T, keep.T), reverse=True)
    return transposed.T


def gae(token_rewards: jax.Array, values: jax.Array, mask: jax.Array,
        gamma: float, lam: float) -> tuple[jax.Array, jax.Array]:
    """Compute generalized advantage estimates for `[B, T]` rewards and values
    (`gae_advantages`), whitened over the mask.

    Returns the whitened advantages and the unwhitened returns, in that order.
    Both references compute `returns = A + V` before whitening. The whitening
    subtracts two nearby numbers, and bf16 rounds the difference away, so it
    runs in at least float32 as the recursion does.
    """
    advantages = gae_advantages(token_rewards, values, mask, gamma, lam)
    returns = advantages + jnp.asarray(values, advantages.dtype)
    return masked_whiten(advantages, jnp.asarray(mask, advantages.dtype)), returns
