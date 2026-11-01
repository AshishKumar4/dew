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

from dew.rl.env import Env, Memory, Policy, TimeStep, env_rollout, start, view

Params = Mapping[str, object]


def first_episodes(following: TimeStep) -> jax.Array:
    """Where each step `[T, envs]` belongs to its environment's first episode: the step that ended it
    included, the steps after it not."""
    return (jnp.cumsum(following.done, 0) - following.done) == 0


def episodes(policy: Policy, env: Env, steps: int, *, memory: Callable[[int], Memory] | None = None,
             reached: Sequence[str] = ()) -> Callable[[Params, jax.Array], dict[str, float]]:
    """A scorer of `policy` in fresh environments of `env`, flown `steps` steps without restarting
    (`dew.rl.env.env_rollout`, one jitted program): called with the policy's variables and a key, it
    returns the first episodes' mean `return` and `steps`, and for each observation view in `reached` the
    mean of its value at each first episode's last step. `env` is batched, or one environment, scored as a
    batch of one. `memory(count)` is a recurrent policy's starting state over a batched env's `count`
    environments (`DreamerV3Objective.memory`)."""
    reward = jax.eval_shape(env.reset, jax.random.key(0))[1].reward
    count = reward.shape[0] if reward.ndim else 1
    starting = None if memory is None else memory(count)

    @jax.jit
    def scored(variables: Params, key: jax.Array) -> dict[str, jax.Array]:
        taken = env_rollout(env, policy, variables, start(env, key, starting), steps, reset=False)[1]
        # One environment's steps `[T]` taken as a batch of one's `[T, 1]`.
        following = (taken.following if reward.ndim
                     else jax.tree.map(lambda leaf: leaf[:, None], taken.following))
        inside = first_episodes(following)
        scores = {"return": jnp.mean(jnp.sum(jnp.where(inside, following.reward, 0), axis=0)),
                  "steps": jnp.mean(jnp.sum(inside, axis=0).astype(jnp.float32))}
        last = jnp.sum(inside, axis=0) - 1
        for name in reached:
            value = view(following.observation, name)
            ended = jnp.take_along_axis(value.reshape(value.shape[0], count, -1), last[None, :, None], 0)[0]
            scores[name] = jnp.mean(ended.astype(jnp.float32))
        return scores

    def score(variables: Params, key: jax.Array) -> dict[str, float]:
        return {name: float(value) for name, value in jax.device_get(scored(variables, key)).items()}

    return score


__all__ = ["episodes", "first_episodes"]
