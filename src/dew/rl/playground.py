"""MuJoCo Playground's MJX environments as `dew.rl.env.Env`s.

A Playground environment (google-deepmind/mujoco_playground, the `playground` distribution) already
answers a jittable `reset(rng)` and `step(state, action)` with a `State` holding `obs`, `reward` and
`done`, so the adapter renames those into a `TimeStep` and nothing more; its episodes run unbounded,
and Playground's training cuts them with Brax's episode wrapper, which `TimeLimit` stands for here.
Playground answers `done` as a float, 1 where the physics went non-finite in the DM Control suite's
tasks, which end only by their time limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct

from .env import Batched, Env, TimeLimit, TimeStep

if TYPE_CHECKING:
    from mujoco_playground import MjxEnv, State

type Override = str | int | float | bool
"""A Playground config field's value."""


@struct.dataclass
class Playground:
    """One Playground environment `env` (an `MjxEnv`) as an `Env`."""

    env: MjxEnv = struct.field(pytree_node=False)

    @classmethod
    def load(cls, name: str, *, impl: str = "jax", **overrides: Override) -> Playground:
        """Playground's registered environment `name` under its default config, MJX's JAX
        implementation by default, which runs on any JAX backend (the default config of a DM Control
        task names Warp's, which runs on CUDA alone), and `overrides`."""
        from mujoco_playground import registry

        return cls(registry.load(name, config_overrides={"impl": impl, **overrides}))

    @property
    def action_size(self) -> int:
        return self.env.action_size

    def _answer(self, state: State) -> TimeStep:
        terminated = jnp.asarray(state.done) > 0
        return TimeStep(state.obs, jnp.asarray(state.reward, jnp.float32), terminated,
                        jnp.zeros_like(terminated))

    def reset(self, key: jax.Array) -> tuple[State, TimeStep]:
        state = self.env.reset(key)
        return state, self._answer(state)

    def step(self, state: State, action: jax.Array) -> tuple[State, TimeStep]:
        state = self.env.step(state, action)
        return state, self._answer(state)


def playground(name: str, count: int, *, episode_length: int = 1000, action_repeat: int = 1,
               impl: str = "jax", **overrides: Override) -> Env:
    """`count` copies of Playground's environment `name` stepped together, each episode cut after
    `episode_length` steps of `action_repeat` repeats, as Playground trains them
    (`wrap_for_brax_training`)."""
    return Batched(TimeLimit(Playground.load(name, impl=impl, **overrides), episode_length, action_repeat),
                   count)


__all__ = ["Playground", "playground"]
