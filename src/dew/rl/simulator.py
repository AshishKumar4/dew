"""A simulator outside JAX as an environment: the boundary any GPU simulator plugs into.

A JAX-native simulator is an `Env` and runs inside `env_rollout`'s scan. A simulator written in another
framework (a CUDA, Warp, PyTorch or C++ engine) cannot step inside a jitted program, so it is stepped
from the host, a batch at a time. Its arrays cross the boundary by DLPack, without a copy where both
sides hold them on one device: an action is a JAX array, which a simulator takes with its own
`from_dlpack` (`torch.from_dlpack`, `wp.from_dlpack`, `cupy.from_dlpack`); an observation, reward or flag
is any array exposing `__dlpack__`, which `jax.dlpack.from_dlpack` takes. `simulator_rollout` runs the
policy jitted and records what `env_rollout` records, the same `Transition` and `Carry`, so a learner
reads a simulator's rollouts as it reads an `Env`'s.

A simulator resets an episode that ends itself, in place, as GPU simulators batched over thousands of
environments do. It answers a step with the observation each episode ended on and, apart, the one each
next action reads (`SimulatorStep.next_observation`, the next episode's first where one ended).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from flax import struct

from dew.rl.env import Carry, Observation, Params, Policy, TimeStep, Transition


class Arrays(Protocol):
    """An array a simulator hands over: anything exposing the DLPack protocol (a PyTorch tensor, a Warp
    or CuPy array, a NumPy array, a JAX array)."""

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401

    def __dlpack_device__(self) -> tuple[int, int]: ...


@struct.dataclass
class SimulatorStep:
    """A simulator's answer to a step of every environment: what a `TimeStep` holds (the observation
    each episode reached, its reward, and whether it ended terminal or cut), and `next_observation`,
    what each next action reads: the next episode's first observation where an episode ended and was
    reset in place, the observation reached elsewhere. Every leaf leads with the batch axis."""

    observation: Observation
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    next_observation: Observation


class Simulator(Protocol):
    """A batch of environments stepped from the host: `reset(seed)` starts every episode and answers each
    first observation, `step(action)` advances every one by its row of a `[batch, action_size]` action
    and answers a `SimulatorStep` (or a mapping of its fields), resetting ended episodes itself. Arrays
    cross by DLPack (`Arrays`)."""

    action_size: int

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: jax.Array) -> SimulatorStep | Mapping[str, Any]: ...


def received(tree: Any) -> Any:  # noqa: ANN401
    """A simulator's arrays as JAX arrays, through DLPack: on the device they are held on, without a
    copy where the producer allows it, a host array as JAX takes host memory."""
    return jax.tree.map(lambda leaf: leaf if isinstance(leaf, jax.Array) else jax.dlpack.from_dlpack(leaf),
                        tree)


def simulator_start(simulator: Simulator, key: jax.Array, memory: Any = None) -> Carry:  # noqa: ANN401
    """A rollout's carry at fresh episodes of `simulator`: its first observations, seeded from `key`,
    and the policy's `memory` as it starts."""
    seed_key, key = jax.random.split(key)
    seed = int(jax.random.randint(seed_key, (), 0, jnp.iinfo(jnp.int32).max))
    return Carry(None, received(simulator.reset(seed)), key, memory)


def simulator_rollout(simulator: Simulator, policy: Policy, params: Params, carry: Carry,
                      steps: int) -> tuple[Carry, Transition]:
    """`steps` steps of `policy` under `params` in `simulator` from `carry`: the policy jitted, the
    simulator stepped from the host, each step's `Transition` stacked along a leading step axis
    `[steps, ...]`, as `env_rollout` records them; the carry's state is the simulator's own, held by it,
    so the carry holds None there."""
    @jax.jit
    def act(carry: Carry) -> tuple[jax.Array, Any, Any, jax.Array]:
        key, act_key = jax.random.split(carry.key)
        action, extras, memory = policy(params, carry.observation, act_key, carry.memory)
        return action, extras, memory, key

    recorded = []
    for _ in range(steps):
        action, extras, memory, key = act(carry)
        answer = simulator.step(action)
        fields = answer if isinstance(answer, Mapping) else {
            name: getattr(answer, name) for name in ("observation", "reward", "terminated", "truncated",
                                                     "next_observation")}
        stepped = received(dict(fields))
        following = TimeStep(stepped["observation"], stepped["reward"], stepped["terminated"].astype(bool),
                             stepped["truncated"].astype(bool))
        recorded.append(Transition(carry.observation, action, extras, following))
        carry = Carry(None, stepped["next_observation"], key, memory)
    return carry, jax.tree.map(lambda *leaves: jnp.stack(leaves), *recorded)


__all__ = ["Arrays", "Simulator", "SimulatorStep", "received", "simulator_rollout", "simulator_start"]
