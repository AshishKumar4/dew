"""Environments as two pure functions over a pytree, and the one loop over them.

An `Env` is `reset(key) -> (state, TimeStep)` and `step(state, action) -> (state, TimeStep)`, jittable,
the shape every JAX-native simulator has (MuJoCo Playground's, through `dew.rl.playground`) and the
shape a learned world model has too: a Dreamer world model is an `Env` whose state is its latent and
whose step is the learned dynamics (`dew.objectives.rl.dreamer`). So acting in a simulator and
imagining in a world model are the same `env_rollout`, a `lax.scan` of a policy and a step.

A batch of environments is `Batched`, `jax.vmap` over the state, whose leading axis shards like any
batch axis. An episode that ends restarts in place (`env_rollout`'s `reset`), as Brax's and
Playground's auto-reset wrappers restart it, the transition that ended it recorded first.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from flax import struct

type Observation = Any
"""An environment's observation: an array, or a pytree of arrays (a dict of named views)."""

type EnvState = Any
"""An environment's state, a pytree of its own."""


@struct.dataclass
class TimeStep:
    """What an environment answers a reset or a step with: the `observation` reached, the `reward`
    the step earned (zero at a reset), `terminated` where the episode ended in a terminal state,
    whose value is zero, and `truncated` where it was cut at a time limit, whose value is the
    state's own. Every leaf leads with the environment's batch axes."""

    observation: Observation
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array

    @property
    def done(self) -> jax.Array:
        """Where the episode ended, by either cause."""
        return self.terminated | self.truncated


class Env(Protocol):
    """An environment as two jittable functions over its state, a pytree: `reset(key)` starts an
    episode and `step(state, action)` advances it, each answering a `TimeStep`. `action_size` is the
    width of a continuous action in [-1, 1]."""

    action_size: int

    def reset(self, key: jax.Array) -> tuple[EnvState, TimeStep]: ...

    def step(self, state: EnvState, action: jax.Array) -> tuple[EnvState, TimeStep]: ...


@struct.dataclass
class _Limited:
    inner: EnvState
    steps: jax.Array


@struct.dataclass
class TimeLimit:
    """`env` whose episodes are cut after `episode_length` steps, each step repeating its action
    `action_repeat` times and summing their rewards (Brax's EpisodeWrapper, which Playground's
    training wraps its environments in): a cut episode is `truncated`, not `terminated`."""

    env: Env = struct.field(pytree_node=False)
    episode_length: int = struct.field(pytree_node=False, default=1000)
    action_repeat: int = struct.field(pytree_node=False, default=1)

    @property
    def action_size(self) -> int:
        return self.env.action_size

    def reset(self, key: jax.Array) -> tuple[_Limited, TimeStep]:
        state, step = self.env.reset(key)
        return _Limited(state, jnp.zeros((), jnp.int32)), step

    def step(self, state: _Limited, action: jax.Array) -> tuple[_Limited, TimeStep]:
        def repeated(carry, _):
            inner, reward, terminated = carry
            inner, step = self.env.step(inner, action)
            # A terminated episode earns nothing more within the repeat.
            return (inner, reward + jnp.where(terminated, 0, step.reward), terminated | step.terminated), step

        first, initial = self.env.step(state.inner, action)
        (inner, reward, terminated), last = jax.lax.scan(
            repeated, (first, initial.reward, initial.terminated), None, length=self.action_repeat - 1)
        last = initial if self.action_repeat == 1 else jax.tree.map(lambda leaf: leaf[-1], last)
        steps = state.steps + 1
        return _Limited(inner, steps), TimeStep(last.observation, reward, terminated,
                                               ~terminated & (steps >= self.episode_length))


@struct.dataclass
class Batched:
    """`count` copies of `env` stepped together: `jax.vmap` over their states, each reset from its
    own key, an action and a `TimeStep` leading with the `count` axis."""

    env: Env = struct.field(pytree_node=False)
    count: int = struct.field(pytree_node=False)

    @property
    def action_size(self) -> int:
        return self.env.action_size

    def reset(self, key: jax.Array) -> tuple[EnvState, TimeStep]:
        return jax.vmap(self.env.reset)(jax.random.split(key, self.count))

    def step(self, state: EnvState, action: jax.Array) -> tuple[EnvState, TimeStep]:
        return jax.vmap(self.env.step)(state, action)


@struct.dataclass
class Transition:
    """One step of a rollout, leading with the environment's batch axes: the `observation` acted
    on, the `action` taken, what the policy reported beside it (`extras`, its log-probability and
    value), and the `TimeStep` that answered it, `following`, its observation the one the step
    reached before any restart."""

    observation: Observation
    action: jax.Array
    extras: Any
    following: TimeStep


@struct.dataclass
class Carry:
    """A rollout's running state between calls: the environments' state, the observation each
    acts on next, and the key the next restarts and actions draw from."""

    state: EnvState
    observation: Observation
    key: jax.Array


type Params = Any
"""A policy's weights, a pytree."""

type Policy = Callable[[Params, Observation, jax.Array], tuple[jax.Array, Any]]
"""`policy(params, observation, key) -> (action, extras)`: an action for each environment."""


def start(env: Env, key: jax.Array) -> Carry:
    """A rollout's carry at fresh episodes of `env`."""
    reset_key, key = jax.random.split(key)
    state, step = env.reset(reset_key)
    return Carry(state, step.observation, key)


def env_rollout(env: Env, policy: Policy, params: Params, carry: Carry, steps: int, *,
                reset: bool = True) -> tuple[Carry, Transition]:
    """`steps` steps of `policy` under `params` in `env` from `carry`, as one `lax.scan`: the carry
    after them and each step's `Transition`, leading with the step axis `[steps, ...]`.

    With `reset`, an environment whose episode ended restarts from a fresh reset of its own before
    the next step, its transition keeping the observation the episode ended at; a learned
    environment, whose imagined episodes never restart, passes False."""
    def one(carry: Carry, _):
        key, act_key, reset_key = jax.random.split(carry.key, 3)
        action, extras = policy(params, carry.observation, act_key)
        state, following = env.step(carry.state, action)
        observation = following.observation
        if reset:
            done = following.done

            def restart(state, observation):
                fresh_state, fresh = env.reset(reset_key)

                def chosen(new, old):
                    return jnp.where(done.reshape(done.shape + (1,) * (new.ndim - done.ndim)), new, old)

                return jax.tree.map(chosen, fresh_state, state), jax.tree.map(chosen, fresh.observation,
                                                                             observation)

            # A reset runs only at a step where some episode ended.
            state, observation = jax.lax.cond(jnp.any(done), restart, lambda *held: held, state, observation)
        return Carry(state, observation, key), Transition(carry.observation, action, extras, following)

    return jax.lax.scan(one, carry, None, length=steps)


__all__ = ["Batched", "Carry", "Env", "EnvState", "Observation", "Params", "Policy", "TimeLimit", "TimeStep",
           "Transition", "env_rollout", "start"]
