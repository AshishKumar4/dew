"""A replay of environment steps held on the device, written and sampled by jittable functions.

The replay keeps the last `capacity` steps of each of `envs` environments, a ring per environment
over a pytree of step fields `[capacity, envs, ...]`, its leading axes shardable over the
environment axis as the batch is. `sample` draws uniformly among the windows of consecutive steps
the replay holds whole (DreamerV3's uniform replay, arXiv 2301.04104, of `length`-step chunks).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from dew.rl.env import Tree

type Steps = Tree
"""A pytree of step fields, each leaf leading with its step and environment axes."""


@struct.dataclass
class ReplayBuffer:
    """The replay's state: the `held` steps' leaves `[capacity, envs, ...]`, the slot the next step writes
    (`cursor`) and the steps written so far, at most `capacity` (`filled`)."""

    held: Steps
    cursor: jax.Array
    filled: jax.Array

    @property
    def capacity(self) -> int:
        return jax.tree.leaves(self.held)[0].shape[0]

    @classmethod
    def empty(cls, example: Steps, capacity: int) -> ReplayBuffer:
        """An empty replay of `capacity` steps of what `example`, one step `[envs, ...]`, holds."""
        if capacity < 1:
            raise ValueError("a replay holds at least one step")
        held = jax.tree.map(lambda leaf: jnp.zeros((capacity, *jnp.shape(leaf)), jnp.result_type(leaf)),
                            example)
        return cls(held, jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32))

    def add(self, steps: Steps) -> ReplayBuffer:
        """The replay after `steps`, leaves `[T, envs, ...]` in order, the oldest overwritten."""
        count = jax.tree.leaves(steps)[0].shape[0]
        if count > self.capacity:
            raise ValueError(f"{count} steps overflow a replay of {self.capacity}")
        slots = (self.cursor + jnp.arange(count)) % self.capacity
        written = jax.tree.map(lambda old, new: old.at[slots].set(new.astype(old.dtype)), self.held, steps)
        return ReplayBuffer(written, (self.cursor + count) % self.capacity,
                            jnp.minimum(self.filled + count, self.capacity))

    def sample(self, key: jax.Array, batch: int, length: int) -> Steps:
        """`batch` windows of `length` consecutive steps `[batch, length, ...]`, each from an
        environment and a start drawn uniformly among those the replay holds whole, its steps in the
        order they were written. The replay must hold `length` steps."""
        if length > self.capacity:
            raise ValueError(f"a window of {length} steps overflows a replay of {self.capacity}")
        envs = jax.tree.leaves(self.held)[0].shape[1]
        env_key, start_key = jax.random.split(key)
        rows = jax.random.randint(env_key, (batch,), 0, envs)
        # The oldest step held is `filled` steps behind the cursor; a window starts at most
        # `filled - length` steps after it.
        offsets = jax.random.randint(start_key, (batch,), 0, jnp.maximum(self.filled - length + 1, 1))
        oldest = self.cursor - self.filled
        times = (oldest + offsets[:, None] + jnp.arange(length)[None, :]) % self.capacity
        return jax.tree.map(lambda held: held[times, rows[:, None]], self.held)


__all__ = ["ReplayBuffer", "Steps"]
