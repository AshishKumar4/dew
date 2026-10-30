"""PPO for continuous control in a `dew.rl.env.Env`: an actor-critic policy, its rollout and its loss.

The policy is Brax's PPO policy (google/brax, brax/training/agents/ppo): a tanh-squashed normal over
the action, its location and scale from one MLP (`softplus(raw) + 0.001`), and a separate value
MLP; the loss is the clipped surrogate over advantages whitened within the minibatch, half the
squared error of the value against its GAE target, and an entropy bonus, the advantages computed in
the loss from the live critic over each stored sequence as Brax's loss computes them. A step that
ended its episode at a time limit bootstraps from the observation it reached; one that ended in a
terminal state takes no value after it.

`ControlRollout` collects with the live policy and serves the collection as `epochs` passes of
`minibatches` minibatches, one an update, collecting again once they are spent, as Brax's
`num_updates_per_batch` passes over `num_minibatches` do.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.batch import Batch, Variables
from dew.objectives.base import Aux, Objective, Ratio, Shown, Step
from dew.rl.env import Carry, Env, Observation, Transition, env_rollout, start
from dew.rl.surrogate import clipped_surrogate_terms

if TYPE_CHECKING:
    from dew.state import TrainState

_MIN_STD = 1e-3
"""The floor of the policy's scale (Brax's NormalTanhDistribution `min_std`)."""


class _MLP(nn.Module):
    layers: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for width in self.layers:
            x = nn.swish(nn.Dense(width, kernel_init=nn.initializers.lecun_uniform())(x))
        return nn.Dense(self.out, kernel_init=nn.initializers.lecun_uniform())(x)


class ActorCritic(nn.Module):
    """Brax PPO's networks over a flat observation: the policy MLP's location and raw scale of each
    of `action_size` action dimensions, and the value MLP's one value. Brax's DM Control widths:
    four layers of 32 for the policy, five of 256 for the value."""

    action_size: int
    policy_layers: Sequence[int] = (32, 32, 32, 32)
    value_layers: Sequence[int] = (256, 256, 256, 256, 256)

    def setup(self) -> None:
        self.policy_net = _MLP(tuple(self.policy_layers), 2 * self.action_size)
        self.value_net = _MLP(tuple(self.value_layers), 1)

    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        location, scale = self.policy(observation)
        return location, scale, self.value(observation)

    def policy(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The normal's location and scale before the tanh, each `[..., action_size]`."""
        location, raw = jnp.split(self.policy_net(observation), 2, axis=-1)
        return location, jax.nn.softplus(raw) + _MIN_STD

    def value(self, observation: jax.Array) -> jax.Array:
        return self.value_net(observation)[..., 0]


def tanh_normal_log_prob(raw: jax.Array, location: jax.Array, scale: jax.Array) -> jax.Array:
    """The log-density of the action tanh(`raw`), `raw` drawn from N(location, scale), summed over
    the action: the normal's log-density less log(1 - tanh(raw)^2), in its stable form
    2 (log 2 - raw - softplus(-2 raw))."""
    normal = -0.5 * jnp.square((raw - location) / scale) - jnp.log(scale) - 0.5 * math.log(2 * math.pi)
    jacobian = 2 * (math.log(2.0) - raw - jax.nn.softplus(-2 * raw))
    return jnp.sum(normal - jacobian, axis=-1)


def tanh_normal_entropy(location: jax.Array, scale: jax.Array, key: jax.Array) -> jax.Array:
    """The squashed normal's entropy as Brax estimates it: the normal's, plus the tanh's log-Jacobian
    at one draw, summed over the action."""
    raw = location + scale * jax.random.normal(key, location.shape, location.dtype)
    jacobian = 2 * (math.log(2.0) - raw - jax.nn.softplus(-2 * raw))
    return jnp.sum(0.5 + 0.5 * math.log(2 * math.pi) + jnp.log(scale) + jacobian, axis=-1)


class ControlPPOObjective(Objective[Ratio]):
    """PPO over an `ActorCritic` (the module docstring). A batch holds sequences `[B, T]`: each step's
    `observation`, the pre-squash `raw_action` taken and its `log_prob` under the policy that took
    it, the `reward`, whether the step `terminated` or was `truncated`, and the `following`
    observation it reached. `discount` and `lam` are GAE's, `clip` the surrogate's,
    `reward_scaling` multiplies each reward, `value_cost` weighs the value loss and `entropy_cost`
    the entropy bonus. Brax's `ppo.train` defaults, but the discount, Playground's DM Control one."""

    shown: Mapping[str, Shown] = {"loss": Shown(), "episode_return": Shown(better="higher")}

    def __init__(self, model: ActorCritic, observation_size: int, *, discount: float = 0.995,
                 lam: float = 0.95, clip: float = 0.3, reward_scaling: float = 1.0, value_cost: float = 0.5,
                 entropy_cost: float = 1e-4, variables: Variables | None = None):
        if not 0 <= discount <= 1 or not 0 <= lam <= 1:
            raise ValueError("discount and lam lie in [0, 1]")
        if not clip > 0:
            raise ValueError("clip must be positive")
        self.model = model
        self.observation_size = observation_size
        self.discount, self.lam, self.clip = discount, lam, clip
        self.reward_scaling, self.value_cost, self.entropy_cost = reward_scaling, value_cost, entropy_cost
        self.variables = variables

    def held_variables(self) -> Variables | None:
        return self.variables

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, self.observation_size)))

    @property
    def validation_loss_refusal(self) -> str | None:
        return "control PPO scores its policy by its episodes' returns, not a validation loss"

    def act(self, params: Variables, observation: Observation, key: jax.Array,
            memory: None) -> tuple[jax.Array, dict, None]:
        """An action for each observation, tanh of a draw from the policy, beside the draw and its
        log-probability (`dew.rl.env.Policy`)."""
        location, scale = self.model.apply(params, observation, method=ActorCritic.policy)
        raw = location + scale * jax.random.normal(key, location.shape, location.dtype)
        extras = {"raw_action": raw, "log_prob": tanh_normal_log_prob(raw, location, scale)}
        return jnp.tanh(raw), extras, None

    def greedy(self, params: Variables, observation: Observation, key: jax.Array,
               memory: None) -> tuple[jax.Array, dict, None]:
        """The policy's deterministic action, tanh of its location, as Brax evaluates it."""
        location, _ = self.model.apply(params, observation, method=ActorCritic.policy)
        return jnp.tanh(location), {}, None

    def advantages(self, values: jax.Array, following: jax.Array,
                   batch: Batch) -> tuple[jax.Array, jax.Array]:
        """Brax's two passes over each sequence `[B, T]` (brax/training/agents/ppo/losses.py:59-98)
        from the critic's `values` at the observations and at the `following` ones: the value targets,
        GAE's lambda-returns, and the actor's advantages, a TD step from each value through the next
        step's target, or through the following value where the episode ended (none after a
        terminal state)."""
        rewards = jnp.asarray(batch["reward"], values.dtype) * self.reward_scaling
        terminated = jnp.asarray(batch["terminated"], values.dtype)
        ended = jnp.maximum(terminated, jnp.asarray(batch["truncated"], values.dtype))
        deltas = rewards + self.discount * (1 - terminated) * following - values

        def back(advantage, inputs):
            delta, end = inputs
            advantage = delta + self.discount * self.lam * (1 - end) * advantage
            return advantage, advantage

        _, returns = jax.lax.scan(back, jnp.zeros(values.shape[0], values.dtype), (deltas.T, ended.T),
                                  reverse=True)
        targets = returns.T + values
        onward = jnp.concatenate([targets[:, 1:], following[:, -1:]], 1)
        after = jnp.where(ended > 0, following, onward)
        return rewards + self.discount * (1 - terminated) * after - values, targets

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        observation, following = batch["observation"], batch["following"]
        location, scale = self.model.apply(variables, observation, method=ActorCritic.policy)
        values = self.model.apply(variables, observation, method=ActorCritic.value)
        after = jax.lax.stop_gradient(self.model.apply(variables, following, method=ActorCritic.value))
        advantages, targets = jax.lax.stop_gradient(
            self.advantages(jax.lax.stop_gradient(values), after, batch))
        whitened = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_ratio = tanh_normal_log_prob(batch["raw_action"], location, scale) - batch["log_prob"]
        mask = jnp.ones(log_ratio.shape, log_ratio.dtype)
        policy, metrics = clipped_surrogate_terms(log_ratio, whitened, mask, self.clip, self.clip,
                                                  dual_clip=None)
        value = self.value_cost * 0.5 * jnp.square(targets - values)
        entropy = tanh_normal_entropy(location, scale, step.key)
        per_step = policy + value - self.entropy_cost * entropy
        metrics = {**metrics, "policy": policy.mean(), "value": value.mean(), "entropy": entropy.mean()}
        return Ratio(per_step.sum(), jnp.asarray(per_step.size, per_step.dtype)), Aux(metrics)


@dataclass
class ControlRollout:
    """The trainer's rollout (`dew.training.rollout.Rollout`) of a `ControlPPOObjective` in `env`, a
    batched `Env`: every `epochs` x `minibatches` updates it collects `unrolls` sequences of
    `unroll_length` steps from each environment with the live policy, then serves them as `epochs`
    shuffled passes of `minibatches` minibatches, one an update. Its carry, the environments mid
    episode, stays on the device between collections. `metrics` reports the mean return of the
    episodes that ended in the last collection."""

    objective: ControlPPOObjective
    env: Env
    unroll_length: int = 30
    unrolls: int = 1
    epochs: int = 4
    minibatches: int = 4
    _carry: Carry | None = field(default=None, init=False, repr=False)
    _returns: jax.Array | None = field(default=None, init=False, repr=False)
    _collected: Mapping[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _order: np.ndarray | None = field(default=None, init=False, repr=False)
    _ended: float = field(default=float("nan"), init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.unroll_length, self.unrolls, self.epochs, self.minibatches) < 1:
            raise ValueError("a control rollout collects and serves at least one of each")
        self._collect = jax.jit(self._collected_impl)

    @property
    def metrics(self) -> Mapping[str, float]:
        return {"episode_return": self._ended}

    def _collected_impl(self, params: Variables, carry: Carry, returns: jax.Array):
        def unroll(state, _):
            carry, returns = state
            carry, steps = env_rollout(self.env, self.objective.act, params, carry, self.unroll_length)
            # Each environment's running return, read where its episode ends.
            def tally(held, step: Transition):
                total = held + step.following.reward
                return jnp.where(step.following.done, 0, total), (total, step.following.done)

            returns, (totals, ended) = jax.lax.scan(tally, returns, steps)
            return (carry, returns), (steps, totals, ended)

        (carry, returns), (steps, totals, ended) = jax.lax.scan(unroll, (carry, returns), None,
                                                                length=self.unrolls)
        # [unrolls, T, envs, ...] to sequences [unrolls * envs, T, ...].
        def sequences(leaf):
            leaf = jnp.moveaxis(leaf, 2, 1)
            return leaf.reshape(-1, *leaf.shape[2:])

        batch = {"observation": steps.observation, "raw_action": steps.extras["raw_action"],
                 "log_prob": steps.extras["log_prob"], "reward": steps.following.reward,
                 "terminated": steps.following.terminated, "truncated": steps.following.truncated,
                 "following": steps.following.observation}
        ended_returns = jnp.sum(jnp.where(ended, totals, 0)) / jnp.maximum(jnp.sum(ended), 1)
        return carry, returns, jax.tree.map(sequences, batch), ended_returns, jnp.sum(ended)

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch:
        served = int(state.updates) % (self.epochs * self.minibatches)
        if self._carry is None:
            self._carry = start(self.env, key)
            self._returns = jnp.zeros(self._carry.observation.shape[0], jnp.float32)
        if served == 0 or self._collected is None:
            self._carry, self._returns, collected, ended, count = self._collect(state.variables, self._carry,
                                                                                self._returns)
            self._collected = jax.device_get(collected)
            self._ended = float(ended) if int(count) else float("nan")
            sequences = len(self._collected["reward"])
            if sequences % self.minibatches:
                raise ValueError(f"{sequences} sequences do not split into {self.minibatches} minibatches")
            rng = np.random.default_rng(np.asarray(jax.random.key_data(key)).ravel())
            self._order = np.concatenate([rng.permutation(sequences) for _ in range(self.epochs)])
        size = len(self._collected["reward"]) // self.minibatches
        rows = self._order[served * size:(served + 1) * size]
        return {name: values[rows] for name, values in self._collected.items()}


def episode_return(objective: ControlPPOObjective, env: Env, params: Variables, key: jax.Array,
                   steps: int, *, greedy: bool = True) -> float:
    """The mean undiscounted return of `env`'s environments over `steps` steps of the policy, its
    deterministic action where `greedy`, from fresh episodes: one episode each where `steps` is an
    episode's length."""
    policy = objective.greedy if greedy else objective.act

    @jax.jit
    def run(params: Variables, key: jax.Array) -> jax.Array:
        _, steps_taken = env_rollout(env, policy, params, start(env, key), steps, reset=False)
        return jnp.mean(jnp.sum(steps_taken.following.reward, axis=0))

    return float(run(params, key))


__all__ = ["ActorCritic", "ControlPPOObjective", "ControlRollout", "episode_return", "tanh_normal_entropy",
           "tanh_normal_log_prob"]
