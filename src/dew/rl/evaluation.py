"""A policy scored by the episodes it flies rather than by records it reads: `episodes(policy, env,
steps)`, the scorer `dew.training.evaluation.Episodes` runs on a run's validation cadence, beside its
data splits. Each score is a mean over the environments' first episodes from fresh starts: the
undiscounted `return`, the `steps` the episode lasted, and the mean over those steps of each named
observation view's last step (a race's gates passed, `passed`), so a score is one a drone's episode
earns once, whatever the steps the rollout runs past it.

Every such flight is `flights`': a JAX `Env` flown in one program, a host-stepped batch (a `Simulator`, or a
`DrivenSimulator` a world-model learner acts in) from the host, each answering the environment's own steps.
A judge that reads each environment's episode whole reads them with `first_episode_rows`."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp

from dew.nn.kernels.options import Held, owned_program
from dew.rl.env import Driven, Env, Memory, Policy, TimeStep, env_rollout, start, view, views
from dew.rl.simulator import DrivenSimulator, Simulator, simulator_rollout, simulator_start

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


def flights(policy: Policy, env: Env | Simulator | DrivenSimulator, steps: int, *,
            memory: Callable[[int], Memory] | None = None,
            models: tuple[Held, ...] = ()) -> Callable[[Params, jax.Array], TimeStep]:
    """A flier of `policy` in fresh environments of `env`, flown `steps` steps without restarting: called
    with the policy's variables and a key, it answers the steps that followed each action `[steps, envs]`,
    their observations the environment's own (a driver's, `Driven` or `DrivenSimulator`, taken off). A JAX
    `Env` is flown in one program (`dew.rl.env.env_rollout`), one environment as a batch of one; a
    host-stepped batch, a `Simulator` or a `DrivenSimulator`, from the host (`simulator_rollout`), its
    ended episodes left ended or, driven, restarted, which a first episode's reading leaves out.
    `memory(count)` is a recurrent policy's starting state over the `count` environments
    (`DreamerV3Objective.memory`), and `models` the networks the policy runs, whose compiler options its
    programs take (`dew.nn.kernels.options.owned_program`; an objective's `towers()`)."""
    driven = isinstance(env, Driven | DrivenSimulator)

    def own(following: TimeStep) -> TimeStep:
        # A driver's step holds the environment's observation under its `observation` entry.
        if not driven:
            return following
        return dataclasses.replace(following, observation=views(following.observation)["observation"])

    if isinstance(env, Simulator | DrivenSimulator):
        stepped = env
        starting = None if memory is None else memory(stepped.count)

        def flown_host(variables: Params, key: jax.Array) -> TimeStep:
            carry = simulator_start(stepped, key, starting)
            taken = simulator_rollout(stepped, policy, variables, carry, steps, reset=False, models=models)[1]
            return own(taken.following)

        return flown_host
    jax_env = env
    reward = jax.eval_shape(jax_env.reset, jax.random.key(0))[1].reward
    starting = None if memory is None else memory(reward.shape[0] if reward.ndim else 1)

    def flown(variables: Params, key: jax.Array) -> TimeStep:
        taken = env_rollout(jax_env, policy, variables, start(jax_env, key, starting), steps, reset=False)[1]
        # One environment's steps `[T]` taken as a batch of one's `[T, 1]`.
        following = taken.following
        return own(following if reward.ndim else jax.tree.map(lambda leaf: leaf[:, None], following))

    return owned_program(flown, *models)


def episodes(policy: Policy, env: Env | Simulator | DrivenSimulator, steps: int, *,
             memory: Callable[[int], Memory] | None = None, reached: Sequence[str] = (),
             models: tuple[Held, ...] = ()) -> Callable[[Params, jax.Array], dict[str, float]]:
    """A scorer of `policy` in fresh environments of `env`, flown `steps` steps without restarting
    (`flights`): called with the policy's variables and a key, it returns the first episodes' mean `return`
    and `steps`, and for each observation view in `reached` the mean of its value at each first episode's
    last step. `memory(count)` is a recurrent policy's starting state (`DreamerV3Objective.memory`), and
    `models` the networks whose compiler options its programs take."""
    flown = flights(policy, env, steps, memory=memory, models=models)

    def means(variables: Params, key: jax.Array) -> dict[str, jax.Array]:
        following = flown(variables, key)
        rows = first_episode_rows(following, {name: view(following.observation, name) for name in reached})
        # The return at the reward's own width; counts and views as floats.
        return {"return": jnp.mean(rows["return"]),
                **{name: jnp.mean(rows[name].astype(jnp.float32)) for name in ("steps", *reached)}}

    # A JAX Env's flight and its reading are one program; a host-stepped flight is read as it lands.
    read = means if isinstance(env, Simulator | DrivenSimulator) else owned_program(means, *models)

    def score(variables: Params, key: jax.Array) -> dict[str, float]:
        return {name: float(value) for name, value in jax.device_get(read(variables, key)).items()}

    return score


__all__ = ["episodes", "first_episode_rows", "first_episodes", "flights"]
