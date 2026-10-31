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

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
from flax import struct

from dew.batch import Variables

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

type Tree = jax.Array | DataclassInstance | Mapping[str, Tree] | tuple[Tree, ...] | list[Tree] | None
"""A pytree of arrays: an array, a dataclass of them (a `flax.struct` one), a dict, tuple or list of
them, or nothing."""

type Observation = jax.Array | Mapping[str, Observation]
"""An environment's observation: an array, or a dict of named views, each an array or a dict again."""

type EnvState = Tree
"""An environment's state, a pytree of its own."""

type Memory = Tree
"""A policy's memory between steps (a recurrent policy's state), None for one that keeps none."""


def views(observation: Observation) -> Mapping[str, Observation]:
    """An observation's named views: it, where it is a dict of them."""
    if not isinstance(observation, Mapping):
        raise TypeError("an array observation has no named views")
    return observation


def view(observation: Observation, name: str | None = None) -> jax.Array:
    """An observation's array: it whole, where `name` is None, or its view `name`."""
    found = observation if name is None else views(observation)[name]
    if isinstance(found, Mapping):
        raise TypeError(f"{'the observation' if name is None else f'the view {name!r}'} is a dict of views, "
                        f"{sorted(found)}, not an array")
    return found


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


class Env[State: EnvState](Protocol):
    """An environment as two jittable functions over its state, a pytree: `reset(key)` starts an
    episode and `step(state, action)` advances it, each answering a `TimeStep`. `action_size` is the
    width of a continuous action in [-1, 1]."""

    @property
    def action_size(self) -> int: ...

    def reset(self, key: jax.Array) -> tuple[State, TimeStep]: ...

    def step(self, state: State, action: jax.Array) -> tuple[State, TimeStep]: ...


@struct.dataclass
class _Limited:
    inner: EnvState
    steps: jax.Array


def chosen[T](mask: jax.Array, new: T, old: T) -> T:
    """`new` where `mask` holds, `old` elsewhere, leaf by leaf, `mask` along the leaves' leading axes and a
    key leaf chosen by its data."""
    def pick(fresh: jax.Array, held: jax.Array) -> jax.Array:
        if jnp.issubdtype(fresh.dtype, jax.dtypes.prng_key):
            bits = pick(jax.random.key_data(fresh), jax.random.key_data(held))
            return jax.random.wrap_key_data(bits, impl=jax.random.key_impl(fresh))
        return jnp.where(mask.reshape(mask.shape + (1,) * (fresh.ndim - mask.ndim)), fresh, held)

    return jax.tree.map(pick, new, old)


@struct.dataclass
class TimeLimit:
    """`env` whose episodes are cut after `episode_length` steps, each step repeating its action
    `action_repeat` times and summing their rewards (Brax's EpisodeWrapper, which Playground's
    training wraps its environments in): a cut episode is `truncated`, not `terminated`. An episode `env`
    itself ends, terminated or truncated, ends there: within a repeat it is held as it ended, earning
    nothing more."""

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
            inner, step = carry
            ended = step.terminated | step.truncated
            following, stepped = self.env.step(inner, action)
            # An episode that ended within the repeat is held as it ended and earns nothing more.
            inner = chosen(ended, inner, following)
            step = TimeStep(chosen(ended, step.observation, stepped.observation),
                            step.reward + jnp.where(ended, 0, stepped.reward),
                            step.terminated | (~ended & stepped.terminated),
                            step.truncated | (~ended & stepped.truncated))
            return (inner, step), None

        (inner, step), _ = jax.lax.scan(repeated, self.env.step(state.inner, action), None,
                                        length=self.action_repeat - 1)
        steps = state.steps + 1
        cut = ~step.terminated & (step.truncated | (steps >= self.episode_length))
        return _Limited(inner, steps), TimeStep(step.observation, step.reward, step.terminated, cut)


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
    extras: Tree
    following: TimeStep


@struct.dataclass
class Carry:
    """A rollout's running state between calls: the environments' state, the observation each
    acts on next, the key the next restarts and actions draw from, and the policy's own `memory`
    (a recurrent policy's state, None for one that keeps none)."""

    state: EnvState
    observation: Observation
    key: jax.Array
    memory: Memory = None


type Params = Variables
"""A policy's weights."""

type Policy = Callable[[Params, Observation, jax.Array, Memory], tuple[jax.Array, Tree, Memory]]
"""`policy(params, observation, key, memory) -> (action, extras, memory)`: an action for each
environment, what to record beside it, and the policy's memory after it (None for one that keeps
none)."""


def start(env: Env, key: jax.Array, memory: Memory = None) -> Carry:
    """A rollout's carry at fresh episodes of `env`, the policy's `memory` as it starts."""
    reset_key, key = jax.random.split(key)
    state, step = env.reset(reset_key)
    return Carry(state, step.observation, key, memory)


def env_rollout(env: Env, policy: Policy, params: Params, carry: Carry, steps: int, *,
                reset: bool = True) -> tuple[Carry, Transition]:
    """`steps` steps of `policy` under `params` in `env` from `carry`, as one `lax.scan`: the carry
    after them and each step's `Transition`, leading with the step axis `[steps, ...]`.

    With `reset`, an environment whose episode ended restarts from a fresh reset of its own before
    the next step, its transition keeping the observation the episode ended at; a learned
    environment, whose imagined episodes never restart, passes False."""
    def one(carry: Carry, _):
        key, act_key, reset_key = jax.random.split(carry.key, 3)
        action, extras, memory = policy(params, carry.observation, act_key, carry.memory)
        state, following = env.step(carry.state, action)
        observation = following.observation
        if reset:
            done = following.done

            def restart(state, observation):
                fresh_state, fresh = env.reset(reset_key)
                return chosen(done, fresh_state, state), chosen(done, fresh.observation, observation)

            # A reset runs only at a step where some episode ended.
            state, observation = jax.lax.cond(jnp.any(done), restart, lambda *held: held, state, observation)
        recorded = Transition(carry.observation, action, extras, following)
        return Carry(state, observation, key, memory), recorded

    return jax.lax.scan(one, carry, None, length=steps)


def driven_step(step: TimeStep, first: jax.Array) -> TimeStep:
    """`step` as a world-model learner's driver records it (`Driven`): its observation the dict of the
    `observation`, the `reward` that came with it, and whether it is its episode's `first`, its `last`
    or `terminal`."""
    observed = {"observation": step.observation, "reward": step.reward, "first": first, "last": step.done,
                "terminal": step.terminated}
    return TimeStep(observed, step.reward, step.terminated, step.truncated)


@struct.dataclass
class _Driven:
    inner: EnvState
    last: jax.Array
    key: jax.Array


@struct.dataclass
class Driven:
    """`env` (batched) stepped as an agent's driver steps it: its observation is the whole step a
    world-model learner records, a dict of the `observation`, the `reward` that came with it, and
    whether it is the episode's `first`, its `last` or `terminal`; an episode's last observation is
    answered as a step of its own, and the step after it, whatever its action, starts the next
    episode (DreamerV3's driver, embodied/core/driver.py). A rollout over it passes `reset=False`."""

    env: Env = struct.field(pytree_node=False)

    @property
    def action_size(self) -> int:
        return self.env.action_size

    def reset(self, key: jax.Array) -> tuple[_Driven, TimeStep]:
        reset_key, key = jax.random.split(key)
        state, step = self.env.reset(reset_key)
        observed = driven_step(step, jnp.ones_like(step.done))
        return _Driven(state, jnp.zeros_like(step.done), key), observed

    def step(self, state: _Driven, action: jax.Array) -> tuple[_Driven, TimeStep]:
        key, reset_key = jax.random.split(state.key)
        inner, step = self.env.step(state.inner, action)

        def restart(inner, step):
            fresh_state, fresh = self.env.reset(reset_key)
            return chosen(state.last, fresh_state, inner), chosen(state.last, fresh, step)

        inner, step = jax.lax.cond(jnp.any(state.last), restart, lambda *held: held, inner, step)
        return _Driven(inner, step.done, key), driven_step(step, state.last)


__all__ = ["Batched", "Carry", "Driven", "Env", "EnvState", "Memory", "Observation", "Params", "Policy",
           "TimeLimit", "TimeStep", "Transition", "Tree", "chosen", "driven_step", "env_rollout", "start",
           "view", "views"]
