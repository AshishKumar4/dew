"""A policy scored by the episodes it flies rather than by records it reads: `episodes(policy, env,
steps)`, the scorer `dew.training.evaluation.Episodes` runs on a run's validation cadence, beside its
data splits. Each score is a mean over the environments' first episodes from fresh starts: the
undiscounted `return`, the `steps` the episode lasted, and the mean over those steps of each named
observation view's last step (a race's gates passed, `passed`), so a score is one a drone's episode
earns once, whatever the steps the rollout runs past it."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp

from dew.nn.kernels.options import OuterProgram
from dew.rl.env import Env, Memory, Policy, TimeStep, env_rollout, start, view

Params = Mapping[str, object]


def first_episodes(following: TimeStep) -> jax.Array:
    """Where each step `[T, envs]` belongs to its environment's first episode: the step that ended it
    included, the steps after it not."""
    return (jnp.cumsum(following.done, 0) - following.done) == 0


def first_episode_rows(following: TimeStep,
                       reached: Mapping[str, jax.Array] | None = None) -> dict[str, jax.Array]:
    """Each environment's first episode from the steps that followed its actions `[T, envs]`: its
    undiscounted `return`, its `steps`, whether it was `terminated` (not cut), the reward of its last step
    (`last_reward`), and each array of `reached` (`[T, envs, ...]`, an observation view) at its last step.
    The one reading of first episodes, which `episodes` averages and a tool's judge reads whole."""
    inside = first_episodes(following)
    last = jnp.sum(inside, 0) - 1

    def at_last(value: jax.Array) -> jax.Array:
        value = jnp.asarray(value)
        return jnp.take_along_axis(value, last.reshape(1, -1, *([1] * (value.ndim - 2))), 0)[0]

    rows = {"return": jnp.sum(jnp.where(inside, following.reward, 0), 0), "steps": jnp.sum(inside, 0),
            "terminated": jnp.any(inside & jnp.asarray(following.terminated, bool), 0),
            "last_reward": at_last(following.reward)}
    return {**rows, **{name: at_last(value) for name, value in (reached or {}).items()}}


def episodes(policy: Policy, env: Env, steps: int, *, memory: Callable[[int], Memory] | None = None,
             reached: Sequence[str] = ()) -> Callable[[Params, jax.Array], dict[str, float]]:
    """A scorer of `policy` in fresh environments of `env`, flown `steps` steps without restarting
    (`dew.rl.env.env_rollout`, one program, `OuterProgram`): called with the policy's variables and a key, it
    returns the first episodes' mean `return` and `steps`, and for each observation view in `reached` the
    mean of its value at each first episode's last step. `env` is batched, or one environment, scored as a
    batch of one. `memory(count)` is a recurrent policy's starting state over a batched env's `count`
    environments (`DreamerV3Objective.memory`)."""
    reward = jax.eval_shape(env.reset, jax.random.key(0))[1].reward
    count = reward.shape[0] if reward.ndim else 1
    starting = None if memory is None else memory(count)

    @OuterProgram
    def scored(variables: Params, key: jax.Array) -> dict[str, jax.Array]:
        taken = env_rollout(env, policy, variables, start(env, key, starting), steps, reset=False)[1]
        # One environment's steps `[T]` taken as a batch of one's `[T, 1]`.
        following = (taken.following if reward.ndim
                     else jax.tree.map(lambda leaf: leaf[:, None], taken.following))
        rows = first_episode_rows(following, {name: view(following.observation, name) for name in reached})
        # The return at the reward's own width; counts and views as floats.
        return {"return": jnp.mean(rows["return"]),
                **{name: jnp.mean(rows[name].astype(jnp.float32)) for name in ("steps", *reached)}}

    def score(variables: Params, key: jax.Array) -> dict[str, float]:
        return {name: float(value) for name, value in jax.device_get(scored(variables, key)).items()}

    return score


__all__ = ["episodes", "first_episode_rows", "first_episodes"]
