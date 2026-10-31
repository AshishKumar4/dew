"""An MJX environment and its state, the fields `dew.rl.playground` reads."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax

@dataclass
class State:
    obs: jax.Array | Mapping[str, jax.Array]
    reward: jax.Array
    done: jax.Array

class MjxEnv:
    @property
    def action_size(self) -> int: ...
    @property
    def observation_size(self) -> int | Mapping[str, tuple[int, ...]]: ...
    def reset(self, rng: jax.Array) -> State: ...
    def step(self, state: State, action: jax.Array) -> State: ...
