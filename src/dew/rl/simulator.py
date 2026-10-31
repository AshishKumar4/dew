"""A simulator outside JAX as an environment: the boundary any GPU simulator plugs into.

A JAX-native simulator is an `Env` and runs inside `env_rollout`'s scan. A simulator written in another
framework (a CUDA, Warp, PyTorch or C++ engine) cannot step inside a jitted program, so it is stepped
from the host, a batch at a time. Its arrays cross the boundary by DLPack, without a copy where both
sides hold them on one device: an action is a JAX array, which a simulator takes with its own
`from_dlpack` (`torch.from_dlpack`, `wp.from_dlpack`, `cupy.from_dlpack`); an observation, reward or flag
is any array exposing `__dlpack__`, which `jax.dlpack.from_dlpack` takes.

A simulator steps every environment and restarts the episodes it is told to (`Simulator.restart`), so a
driver decides when an ended episode restarts: `simulator_rollout` restarts it at once, recording what
`env_rollout` records, the same `Transition` and `Carry`; `DrivenSimulator` restarts it a step later, as
`Driven` drives an `Env` for a world-model learner. `rollout_start` starts either kind of environment.
"""

from __future__ import annotations

import functools
from typing import Protocol, overload, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from typing_extensions import CapsuleType

from dew.rl.env import (
    Carry,
    Env,
    Memory,
    Observation,
    Params,
    Policy,
    TimeStep,
    Transition,
    Tree,
    chosen,
    driven_step,
    start,
)


class Shared(Protocol):
    """An array a simulator hands over: anything exposing the DLPack protocol (a PyTorch tensor, a Warp
    or CuPy array), which JAX takes without a copy where both sides hold it on one device."""

    def __dlpack__(self, *, stream: int | None = None) -> CapsuleType: ...

    def __dlpack_device__(self) -> tuple[int, int]: ...


@runtime_checkable
class Stepped(Protocol):
    """A batch of `count` environments stepped from the host. `reset(seed)` starts every episode and
    answers each first observation; `step(action)` advances every environment by its row of a
    `[count, action_size]` action and answers a `TimeStep`. Arrays cross by DLPack (`Shared`,
    `received`)."""

    @property
    def action_size(self) -> int: ...

    @property
    def count(self) -> int: ...

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: jax.Array) -> TimeStep: ...


@runtime_checkable
class Simulator(Stepped, Protocol):
    """A batch of environments stepped from the host (`Stepped`) whose `step` restarts none:
    `restart(mask)` starts a new episode in each environment `mask` `[count]` selects and answers every
    environment's observation, the selected ones' their new episodes' first."""

    def restart(self, mask: jax.Array) -> Observation: ...


type Handed = jax.Array | np.ndarray | Shared
"""An array as a simulator hands it over: a JAX or NumPy array, or another framework's (`Shared`)."""


def taken(leaf: Handed | np.generic) -> jax.Array:
    """A simulator's array as a JAX array: a device array through DLPack, on the device it is held on and
    without a copy where the producer allows it; a NumPy array copied to the default device, as a
    learner reads it."""
    if isinstance(leaf, jax.Array):
        return leaf
    if isinstance(leaf, np.ndarray | np.generic):
        return jnp.asarray(leaf)
    return jax.dlpack.from_dlpack(leaf)


@overload
def received(tree: Handed) -> jax.Array: ...


@overload
def received[T: Tree](tree: T) -> T: ...


def received(tree: Handed | Tree) -> jax.Array | Tree:
    """A simulator's array, or a tree of them (an observation, a step, a state), each array taken as a JAX
    array (`taken`)."""
    return jax.tree.map(taken, tree)


@functools.partial(jax.jit, static_argnums=0)
def _acted(policy: Policy, params: Params, carry: Carry) -> tuple[jax.Array, Tree, Memory, jax.Array]:
    """A policy's step from a rollout's carry, compiled once for each policy: its action, extras and
    memory, and the carry's next key."""
    key, act_key = jax.random.split(carry.key)
    action, extras, memory = policy(params, carry.observation, act_key, carry.memory)
    return action, extras, memory, key


def simulator_start(simulator: Stepped, key: jax.Array, memory: Memory = None) -> Carry:
    """A rollout's carry at fresh episodes of `simulator`: its first observations, seeded from `key`,
    and the policy's `memory` as it starts."""
    seed_key, key = jax.random.split(key)
    seed = int(jax.random.randint(seed_key, (), 0, jnp.iinfo(jnp.int32).max))
    return Carry(None, received(simulator.reset(seed)), key, memory)


def simulator_rollout(simulator: Stepped, policy: Policy, params: Params, carry: Carry, steps: int, *,
                      reset: bool = True) -> tuple[Carry, Transition]:
    """`steps` steps of `policy` under `params` in `simulator` from `carry`: the policy jitted, the
    simulator stepped from the host, each step's `Transition` stacked along a leading step axis
    `[steps, ...]`, as `env_rollout` records them; the carry's state is the simulator's own, held by it,
    so the carry holds None there. With `reset`, an episode that ended restarts before the next step,
    its transition keeping the observation it ended at, which takes a `Simulator`; one that restarts its
    episodes itself (`DrivenSimulator`) passes False. The policy acts where the simulator's arrays are, its
    `params` and the carry's key and memory put there whole (a no-op where they already are), replicated
    over a mesh the observations are laid out on, the observations keeping their layout."""
    restarting = simulator if isinstance(simulator, Simulator) else None
    if reset and restarting is None:
        raise TypeError("a rollout that restarts ended episodes needs a Simulator, which restarts them")
    held = jax.tree.leaves(carry.observation)[0].sharding
    whole = NamedSharding(held.mesh, PartitionSpec()) if isinstance(held, NamedSharding) else held
    params, key, memory = jax.device_put((params, carry.key, carry.memory), whole)
    carry = Carry(carry.state, carry.observation, key, memory)
    recorded = []
    for _ in range(steps):
        action, extras, memory, key = _acted(policy, params, carry)
        following = received(simulator.step(action))
        following = TimeStep(following.observation, following.reward, following.terminated.astype(bool),
                             following.truncated.astype(bool))
        observation = following.observation
        if restarting is not None and reset and bool(jnp.any(following.done)):
            observation = chosen(following.done, received(restarting.restart(following.done)), observation)
        recorded.append(Transition(carry.observation, action, extras, following))
        carry = Carry(None, observation, key, memory)
    return carry, jax.tree.map(lambda *leaves: jnp.stack(leaves), *recorded)


class DrivenSimulator:
    """`simulator` driven as `Driven` drives an `Env`, for a world-model learner: its observation is the
    whole step the learner records, a dict of the `observation`, the `reward` that came with it, and
    whether it is the episode's `first`, its `last` or `terminal`; an episode's last observation is
    answered as a step of its own, and the step after it, whatever its action, restarts the episode
    (DreamerV3's driver, embodied/core/driver.py). It restarts its episodes itself: a rollout over it
    passes `reset=False`."""

    def __init__(self, simulator: Simulator):
        self.simulator = simulator
        self.action_size, self.count = simulator.action_size, simulator.count
        self._last = jnp.zeros(simulator.count, bool)

    def reset(self, seed: int) -> Observation:
        observation = received(self.simulator.reset(seed))
        self._last = jnp.zeros(self.count, bool)
        step = TimeStep(observation, jnp.zeros(self.count), self._last, self._last)
        return driven_step(step, jnp.ones(self.count, bool)).observation

    def step(self, action: jax.Array) -> TimeStep:
        step = received(self.simulator.step(action))
        step = TimeStep(step.observation, step.reward, step.terminated.astype(bool),
                        step.truncated.astype(bool))
        restarted = self._last
        if bool(jnp.any(restarted)):
            fresh = received(self.simulator.restart(restarted))
            nothing = jnp.zeros_like(restarted)
            step = chosen(restarted, TimeStep(fresh, jnp.zeros_like(step.reward), nothing, nothing), step)
        self._last = step.done
        return driven_step(step, restarted)


def rollout_start(env: Env | Simulator, key: jax.Array, memory: Memory = None) -> Carry:
    """A rollout's carry at fresh episodes of `env`, an `Env` (`start`) or a `Simulator`
    (`simulator_start`)."""
    return simulator_start(env, key, memory) if isinstance(env, Simulator) else start(env, key, memory)


__all__ = ["DrivenSimulator", "Handed", "Shared", "Simulator", "Stepped", "received", "rollout_start",
           "simulator_rollout", "simulator_start", "taken"]
