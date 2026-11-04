"""PPO for continuous control in a `dew.rl.env.Env`: an actor-critic policy, its rollout and its loss.

The policy is Brax's PPO policy (google/brax, brax/training/agents/ppo): a tanh-squashed normal over
the action, its location and scale from one MLP (`softplus(raw) + 0.001`), and a separate value
MLP; the loss is the clipped surrogate over advantages whitened within the minibatch, half the
squared error of the value against its GAE target, and an entropy bonus, the advantages computed in
the loss from the live critic over each stored sequence as Brax's loss computes them. A step that
ended its episode at a time limit bootstraps from the observation it reached; one that ended in a
terminal state takes no value after it.

`ControlRollout` collects with the live policy and serves the collection as `epochs` passes of
`minibatches` minibatches, one an update, collecting again once they are spent, as Brax's
`num_updates_per_batch` passes over `num_minibatches` do.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.batch import Batch, Variables, unboxed
from dew.nn.kernels.options import program
from dew.objectives.base import Aux, Objective, Ratio, Shown, Step, traced_update_flops
from dew.rl.env import Carry, Env, Memory, Observation, Params, Transition, Tree, env_rollout, view
from dew.rl.evaluation import episodes
from dew.rl.simulator import Simulator, branched, rollout_start, simulator_rollout
from dew.rl.surrogate import clipped_surrogate_terms

if TYPE_CHECKING:
    from dew.state import TrainState

_MIN_STD = 1e-3
"""The floor of the policy's scale (Brax's NormalTanhDistribution `min_std`)."""


class _MLP(nn.Module):
    layers: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for width in self.layers:
            x = nn.swish(nn.Dense(width, kernel_init=nn.initializers.lecun_uniform())(x))
        return nn.Dense(self.out, kernel_init=nn.initializers.lecun_uniform())(x)


class ActorCritic(nn.Module):
    """Brax PPO's networks over a flat observation: the policy MLP's location and raw scale of each
    of `action_size` action dimensions, and the value MLP's one value. Brax's DM Control widths:
    four layers of 32 for the policy, five of 256 for the value. `spread` multiplies the policy's scale:
    below 1, a policy explores less from the start (a drone's first rollouts fly rather than tumble)."""

    action_size: int
    policy_layers: Sequence[int] = (32, 32, 32, 32)
    value_layers: Sequence[int] = (256, 256, 256, 256, 256)
    spread: float = 1.0

    def setup(self) -> None:
        self.policy_net = _MLP(tuple(self.policy_layers), 2 * self.action_size)
        self.value_net = _MLP(tuple(self.value_layers), 1)

    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        location, scale = self.policy(observation)
        return location, scale, self.value(observation)

    def policy(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The normal's location and scale before the tanh, each `[..., action_size]`."""
        location, raw = jnp.split(self.policy_net(observation), 2, axis=-1)
        return location, (jax.nn.softplus(raw) + _MIN_STD) * self.spread

    def value(self, observation: jax.Array) -> jax.Array:
        return self.value_net(observation)[..., 0]


def tanh_normal_log_prob(raw: jax.Array, location: jax.Array, scale: jax.Array) -> jax.Array:
    """The log-density of the action tanh(`raw`), `raw` drawn from N(location, scale), summed over
    the action: the normal's log-density less log(1 - tanh(raw)^2), in its stable form
    2 (log 2 - raw - softplus(-2 raw))."""
    normal = -0.5 * jnp.square((raw - location) / scale) - jnp.log(scale) - 0.5 * math.log(2 * math.pi)
    jacobian = 2 * (math.log(2.0) - raw - jax.nn.softplus(-2 * raw))
    return jnp.sum(normal - jacobian, axis=-1)


def tanh_normal_entropy(location: jax.Array, scale: jax.Array, key: jax.Array) -> jax.Array:
    """The squashed normal's entropy as Brax estimates it: the normal's, plus the tanh's log-Jacobian
    at one draw, summed over the action."""
    raw = location + scale * jax.random.normal(key, location.shape, location.dtype)
    jacobian = 2 * (math.log(2.0) - raw - jax.nn.softplus(-2 * raw))
    return jnp.sum(0.5 + 0.5 * math.log(2 * math.pi) + jnp.log(scale) + jacobian, axis=-1)


class ControlPPOObjective(Objective[Ratio]):
    """PPO over an `ActorCritic` (the module docstring). A batch holds sequences `[B, T]`: each step's
    `observation`, the pre-squash `raw_action` taken and its `log_prob` under the policy that took
    it, the `reward`, whether the step `terminated` or was `truncated`, and the `following`
    observation it reached. `discount` and `lam` are GAE's, `clip` the surrogate's,
    `reward_scaling` multiplies each reward, `value_cost` weighs the value loss and `entropy_cost`
    the entropy bonus. Brax's `ppo.train` defaults, but the discount, Playground's DM Control one."""

    shown: Mapping[str, Shown] = {"loss": Shown(), "episode_return": Shown(better="higher")}

    def __init__(self, model: ActorCritic, observation_size: int, *, discount: float = 0.995,
                 lam: float = 0.95, clip: float = 0.3, reward_scaling: float = 1.0, value_cost: float = 0.5,
                 entropy_cost: float = 1e-4, variables: Variables | None = None, reads: tuple[str, ...] = ()):
        """`reads` names, where observations are dicts, the entries the actor and critic read, joined in
        order: a privileged teacher's (`dew.rl.flight`'s `state` and `goal`)."""
        if not 0 <= discount <= 1 or not 0 <= lam <= 1:
            raise ValueError("discount and lam lie in [0, 1]")
        if not clip > 0:
            raise ValueError("clip must be positive")
        self.model = model
        self.observation_size = observation_size
        self.discount, self.lam, self.clip = discount, lam, clip
        self.reward_scaling, self.value_cost, self.entropy_cost = reward_scaling, value_cost, entropy_cost
        # A held tree enters plain: a draw's boxes (`nn.with_logical_partitioning`) are its table's.
        self.variables = unboxed(variables)
        self.reads = reads

    def seen(self, observation: Observation) -> jax.Array:
        """What the actor and critic read of an observation: its `reads` views joined, or it whole."""
        if not self.reads:
            return view(observation)
        return jnp.concatenate([jnp.asarray(view(observation, name), jnp.float32) for name in self.reads], -1)

    def held_variables(self) -> Variables | None:
        return self.variables

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, self.observation_size)))

    @property
    def validation_loss_refusal(self) -> str | None:
        return "control PPO scores its policy by its episodes' returns, not a validation loss"

    def act(self, params: Params, observation: Observation, key: jax.Array,
            memory: Memory) -> tuple[jax.Array, Tree, Memory]:
        """An action for each observation, tanh of a draw from the policy, beside the draw and its
        log-probability (`dew.rl.env.Policy`)."""
        location, scale = self.model.bind(params).policy(self.seen(observation))
        raw = location + scale * jax.random.normal(key, location.shape, location.dtype)
        extras = {"raw_action": raw, "log_prob": tanh_normal_log_prob(raw, location, scale)}
        return jnp.tanh(raw), extras, None

    def greedy(self, params: Params, observation: Observation, key: jax.Array,
               memory: Memory) -> tuple[jax.Array, Tree, Memory]:
        """The policy's deterministic action, tanh of its location, as Brax evaluates it."""
        location, _ = self.model.bind(params).policy(self.seen(observation))
        return jnp.tanh(location), {}, None

    def advantages(self, values: jax.Array, following: jax.Array,
                   batch: Batch) -> tuple[jax.Array, jax.Array]:
        """Brax's two passes over each sequence `[B, T]` (brax/training/agents/ppo/losses.py:59-98)
        from the critic's `values` at the observations and at the `following` ones: the value targets,
        GAE's lambda-returns, and the actor's advantages, a TD step from each value through the next
        step's target, or through the following value where the episode ended (none after a
        terminal state)."""
        rewards = jnp.asarray(batch["reward"], values.dtype) * self.reward_scaling
        terminated = jnp.asarray(batch["terminated"], values.dtype)
        ended = jnp.maximum(terminated, jnp.asarray(batch["truncated"], values.dtype))
        deltas = rewards + self.discount * (1 - terminated) * following - values

        def back(advantage, inputs):
            delta, end = inputs
            advantage = delta + self.discount * self.lam * (1 - end) * advantage
            return advantage, advantage

        _, returns = jax.lax.scan(back, jnp.zeros(values.shape[0], values.dtype), (deltas.T, ended.T),
                                  reverse=True)
        targets = returns.T + values
        onward = jnp.concatenate([targets[:, 1:], following[:, -1:]], 1)
        after = jnp.where(ended > 0, following, onward)
        return rewards + self.discount * (1 - terminated) * after - values, targets

    def model_flops(self, model, batch: Batch) -> float:
        """One update's forward work on `batch`, its loss's traced program's (`traced_update_flops`)."""
        return traced_update_flops(self, batch)[0]

    def training_flops(self, model, batch: Batch) -> float:
        """One update's work on `batch`, its loss's gradient's traced program's (`traced_update_flops`)."""
        return traced_update_flops(self, batch)[1]

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        observation, following = self.seen(batch["observation"]), self.seen(batch["following"])
        model = self.model.bind(variables)
        location, scale = model.policy(observation)
        values = model.value(observation)
        after = jax.lax.stop_gradient(model.value(following))
        advantages, targets = jax.lax.stop_gradient(
            self.advantages(jax.lax.stop_gradient(values), after, batch))
        whitened = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_ratio = tanh_normal_log_prob(batch["raw_action"], location, scale) - batch["log_prob"]
        mask = jnp.ones(log_ratio.shape, log_ratio.dtype)
        policy, metrics = clipped_surrogate_terms(log_ratio, whitened, mask, self.clip, self.clip,
                                                  dual_clip=None)
        value = self.value_cost * 0.5 * jnp.square(targets - values)
        entropy = tanh_normal_entropy(location, scale, step.key)
        per_step = policy + value - self.entropy_cost * entropy
        metrics = {**metrics, "policy": policy.mean(), "value": value.mean(), "entropy": entropy.mean()}
        return Ratio(per_step.sum(), jnp.asarray(per_step.size, per_step.dtype)), Aux(metrics)


def _summary(steps: Transition, returns: jax.Array):
    """A collection `steps` `[unrolls, T, envs, ...]` as the sequences its minibatches are drawn from, and
    each environment's running return `returns`, read where its episode ends, over the unrolls in order:
    the returns, the sequences, the mean return of the episodes that ended and their count."""
    def tally(held, step: Transition):
        total = held + step.following.reward
        return jnp.where(step.following.done, 0, total), (total, step.following.done)

    flat = jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), steps)
    returns, (totals, ended) = jax.lax.scan(tally, returns, flat)

    # [unrolls, T, envs, ...] to sequences [unrolls * envs, T, ...].
    def sequences(leaf):
        leaf = jnp.moveaxis(leaf, 2, 1)
        return leaf.reshape(-1, *leaf.shape[2:])

    extras = steps.extras
    if not isinstance(extras, Mapping):
        raise TypeError("a control rollout records its policy's raw actions and log-probabilities")
    batch = {"observation": steps.observation, "raw_action": extras["raw_action"],
             "log_prob": extras["log_prob"], "reward": steps.following.reward,
             "terminated": steps.following.terminated, "truncated": steps.following.truncated,
             "following": steps.following.observation}
    ended_returns = jnp.sum(jnp.where(ended, totals, 0)) / jnp.maximum(jnp.sum(ended), 1)
    return returns, jax.tree.map(sequences, batch), ended_returns, jnp.sum(ended)


class Saved(TypedDict, total=False):
    """What a `ControlRollout` carries between calls, as a checkpoint holds it (`ControlRollout.saved`):
    every key but `served`, which one written before the rollout counted its own lacks."""

    carry: Carry
    returns: jax.Array
    collected: Mapping[str, np.ndarray]
    order: np.ndarray
    ended: np.ndarray
    served: np.ndarray


@dataclass
class ControlRollout:
    """The trainer's rollout (`dew.training.rollout.Rollout`) of a `ControlPPOObjective` in `env`, a
    batched `Env` or a `Simulator`: it collects `unrolls` sequences of `unroll_length` steps from each
    environment with the live policy, then serves them as `epochs` shuffled passes of `minibatches`
    minibatches, the next one each call, and collects again once it has served them all. Its count of
    those served is its own, so a run whose update pools several calls (`accumulation`), or whose
    optimizer holds or rejects an update, serves each collection whole, once a pass. Its carry, the
    environments mid episode, stays on the device between collections. `metrics` reports the mean
    return of the episodes that ended in the last collection."""

    objective: ControlPPOObjective
    env: Env | Simulator
    unroll_length: int = 30
    unrolls: int = 1
    epochs: int = 4
    minibatches: int = 4
    _carry: Carry | None = field(default=None, init=False, repr=False)
    _returns: jax.Array | None = field(default=None, init=False, repr=False)
    _collected: Mapping[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _order: np.ndarray | None = field(default=None, init=False, repr=False)
    _ended: float = field(default=float("nan"), init=False, repr=False)
    _served: int | None = field(default=None, init=False, repr=False)
    """The minibatches of the collection served; None before a call that set it: a run's first, or the
    first of a run resumed from a checkpoint written before the rollout counted them, which reads it off
    the updates as that run did."""

    def __post_init__(self) -> None:
        if min(self.unroll_length, self.unrolls, self.epochs, self.minibatches) < 1:
            raise ValueError("a control rollout collects and serves at least one of each")

    @property
    def metrics(self) -> Mapping[str, float]:
        return {"episode_return": self._ended}

    @contextlib.contextmanager
    def branch(self, name: str) -> Iterator[None]:
        """A cooldown branch's calls (`Rollout`'s `branch`), after which the rollout stands as the run left
        it: its environments mid episode, the collection it serves and their order. Nothing it carries is
        donated, so what the run left is kept as it is; a begun `Simulator` keeps its own environments
        (`dew.rl.simulator.branched`)."""
        env = self.env
        kept = (self._carry, self._returns, self._collected, self._order, self._ended, self._served)
        try:
            with (branched(env) if isinstance(env, Simulator) and self._carry is not None
                  else contextlib.nullcontext()):
                yield
        finally:
            self._carry, self._returns, self._collected, self._order, self._ended, self._served = kept

    def saved(self) -> Saved | None:
        """What the rollout carries between calls, for the run's checkpoint (`Rollout`'s `saved`): every
        environment mid episode and its running return, the collection it serves its minibatches from,
        their order, the collection's mean ended return and how many of them it served; None before its
        first call, and over a `Simulator`, whose environments live outside JAX and start afresh on a
        resume."""
        carry, returns, collected, order = self._carry, self._returns, self._collected, self._order
        if carry is None or returns is None or collected is None or order is None or self._served is None or (
                isinstance(self.env, Simulator)):
            return None
        return Saved(carry=carry, returns=returns, collected=dict(collected), order=order,
                     ended=np.asarray(self._ended, np.float32), served=np.asarray(self._served, np.int32))

    def saved_shapes(self) -> Mapping[str, object] | None:
        """`saved`'s leaves' shapes and dtypes, before any call; None over a `Simulator`."""
        env = self.env
        if isinstance(env, Simulator):
            return None
        key = jax.random.key(0)
        carry = jax.eval_shape(functools.partial(rollout_start, env), key)
        count = len(jax.tree.leaves(carry.observation)[0])

        def collected(variables, carry):
            _, steps = self._unrolled(env, variables, carry)
            return _summary(steps, jnp.zeros(count, jnp.float32))[1]

        gathered = jax.eval_shape(collected, jax.eval_shape(self.objective.init, key), carry)
        order = jax.ShapeDtypeStruct((self.epochs * len(gathered["reward"]),), np.int32)
        return {"carry": carry, "returns": jax.ShapeDtypeStruct((count,), jnp.float32), "collected": gathered,
                "order": order, "ended": jax.ShapeDtypeStruct((), jnp.float32),
                "served": jax.ShapeDtypeStruct((), jnp.int32)}

    def restored(self, tree: Saved) -> None:
        """The rollout carrying on from a `saved` tree; one saved before the rollout counted the minibatches
        it served reads them off the updates at its next call, as the run that saved it did."""
        self._carry, self._returns = tree["carry"], tree["returns"]
        self._collected = jax.tree.map(np.asarray, dict(tree["collected"]))
        self._order, self._ended = np.asarray(tree["order"]), float(tree["ended"])
        self._served = int(tree["served"]) if "served" in tree else None

    def _collector(self) -> Callable[[Variables, Carry], tuple[Carry, Transition]]:
        """The collection for the rollout's environment as it is now: a run may change it between
        collections (a curriculum), its collection made again; episodes under way carry on. Over a JAX
        environment it is the objective's program (`Objective.program`), which runs the policy's modules."""
        env = self.env
        if isinstance(env, Simulator):
            return functools.partial(self._simulated, env)
        return self.objective.program("collection", functools.partial(self._unrolled, env), holds=(env,))

    def _unrolled(self, env: Env, params: Variables, carry: Carry) -> tuple[Carry, Transition]:
        """The carry after `unrolls` rollouts of the live policy in `env` and their steps
        `[unrolls, T, envs, ...]`, one scan, jitted whole."""
        def unroll(carry: Carry, _) -> tuple[Carry, Transition]:
            return env_rollout(env, self.objective.act, params, carry, self.unroll_length)

        return jax.lax.scan(unroll, carry, None, length=self.unrolls)

    def _simulated(self, simulator: Simulator, params: Variables, carry: Carry) -> tuple[Carry, Transition]:
        """The carry after `unrolls` rollouts of the live policy in `simulator`, stepped from the host,
        and their steps `[unrolls, T, envs, ...]`."""
        unrolled = []
        for _ in range(self.unrolls):
            carry, steps = simulator_rollout(simulator, self.objective.act, params, carry, self.unroll_length,
                                             models=self.objective.towers())
            unrolled.append(steps)
        return carry, jax.tree.map(lambda *leaves: jnp.stack(leaves), *unrolled)

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch:
        passes = self.epochs * self.minibatches
        # The rollout's own count of the minibatches served (row 3932); none yet, the updates' as before.
        served = (int(state.updates) % passes or passes) if self._served is None else self._served
        carry, returns = self._carry, self._returns
        if carry is None or returns is None:
            carry = rollout_start(self.env, key)
            returns = jnp.zeros(len(jax.tree.leaves(carry.observation)[0]), jnp.float32)
        collected, order = self._collected, self._order
        if served >= passes or collected is None or order is None:
            carry, steps = self._collector()(state.variables, carry)
            returns, gathered, ended, count = program(_summary)(steps, returns)
            collected = jax.device_get(gathered)
            self._ended = float(ended) if int(count) else float("nan")
            sequences = len(collected["reward"])
            if sequences % self.minibatches:
                raise ValueError(f"{sequences} sequences do not split into {self.minibatches} minibatches")
            rng = np.random.default_rng(np.asarray(jax.random.key_data(key)).ravel())
            order = np.concatenate([rng.permutation(sequences) for _ in range(self.epochs)]).astype(np.int32)
            served = 0
        self._carry, self._returns, self._collected, self._order = carry, returns, collected, order
        self._served = served + 1
        size = len(collected["reward"]) // self.minibatches
        rows = order[served * size:(served + 1) * size]
        return jax.tree.map(lambda values: values[rows], collected)


def episode_return(objective: ControlPPOObjective, env: Env | Simulator, params: Variables, key: jax.Array,
                   steps: int, *, greedy: bool = True) -> float:
    """The mean undiscounted return of the first episode of each of `env`'s environments, an `Env` or a
    `Simulator`, over `steps` steps of the policy, its deterministic action where `greedy`: the whole
    episode where `steps` is an episode's length (`dew.rl.evaluation.episodes`)."""
    return episodes(objective.greedy if greedy else objective.act, env, steps)(params, key)["return"]


__all__ = ["ActorCritic", "ControlPPOObjective", "ControlRollout", "Saved", "episode_return",
           "tanh_normal_entropy", "tanh_normal_log_prob"]
