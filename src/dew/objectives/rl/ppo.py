"""PPO with a separate trainable critic, clipped value loss and episode GAE."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental import multihost_utils

from dew.artifacts import agreed
from dew.inference.tasks import Processor, TextGeneration
from dew.nn.inputs import ModelInputs, local_rows, mesh_of
from dew.objectives.base import Aux, EMASpec, Mean, Objective, Step, Variables, mean_loss
from dew.registry import objectives
from dew.rl import gae
from dew.rl.advantage import MEAN_EPS, WHITEN_EPS
from dew.rl.surrogate import clipped_value_loss_terms
from dew.sampling.text import Generation, Sampling
from dew.training.distributed import shard_batch
from dew.training.state import TrainState

from .episodes import EpisodeInference, EpisodeRollout
from .grpo import GRPOObjective
from .rollouts import (
    ADVANTAGES_KEY,
    CALL_INDEX_KEY,
    IDS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    ROLLOUT_INDEX_KEY,
    SEGMENT_IDS_KEY,
)

OLD_VALUES_KEY = "old_values"
RETURNS_KEY = "returns"


class ValueBackbone(Protocol):
    def hidden_states(self, tokens: jax.Array, train: bool = False, *,
                      segment_ids: jax.Array | None = None,
                      positions: jax.Array | None = None) -> jax.Array: ...


class ValueHead(nn.Module):
    """Project a decoder's hidden states to one float32 value per position.

    Packed rows pass their chains' `segment_ids` and `positions`, so no
    state reads another chain.
    """

    backbone: ValueBackbone

    @nn.compact
    def __call__(self, tokens: jax.Array, *, segment_ids: jax.Array | None = None,
                 positions: jax.Array | None = None) -> jax.Array:
        hidden = self.backbone.hidden_states(tokens, train=False, segment_ids=segment_ids, positions=positions)
        return nn.Dense(1, dtype=jnp.float32, name="value")(hidden)[..., 0]


def _part(variables: Variables, name: str) -> Variables:
    """Cut the `name` subtree out of every collection that holds one.

    The joint tree nests the collection above the side, `params/policy`,
    so a side's own tree is the same collections one level down.
    """
    return {collection: subtree[name] for collection, subtree in variables.items() if name in subtree}


def _join(policy: Variables, critic: Variables) -> Variables:
    """Nest a policy and a critic tree under one collection per side.

    The inverse of `_part`: each collection the two share becomes a
    `{"policy": ..., "critic": ...}` node.
    """
    return {collection: {name: tree[collection] for name, tree in (("policy", policy), ("critic", critic))
                         if collection in tree} for collection in policy.keys() | critic.keys()}


@dataclass(frozen=True)
class _Policy:
    task: EpisodeInference

    def bind(self, variables: Variables, /) -> EpisodeInference:
        return self.task.bind(_part(variables, "policy"))

    def __call__(self, inputs: ModelInputs | Sequence[Sequence[int]], max_new_tokens: int, /,
                 *, key: jax.Array, sampling: Sampling) -> Generation:
        return self.task(inputs, max_new_tokens, key=key, sampling=sampling)


@objectives("ppo")
class PPOObjective(Objective[Mean, Variables]):
    """Train a policy and a critic together on one token mass.

    The params collection holds policy and critic subtrees, both optimized by
    the ordinary Trainer. The unit-decay reference selects only policy leaves.
    Rollout targets are detached. beta and policy clip controls are GRPO's
    existing composition; value_coefficient weights verl's half-squared,
    clipped value error. A critic consumes packed token rows with their
    segment_ids and positions and returns [B, T] values; ValueHead supplies
    that interface for a decoder.
    """

    _ema_is_reference = True

    def __init__(self, model, seq_len: int, *, critic: nn.Module,
                 value_coefficient: float = .5, value_clip: float = .2, **policy_options):
        for name, value in (("value_coefficient", value_coefficient), ("value_clip", value_clip)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (policy_options.get("aggregation", "token-mean") != "token-mean"
                or any(policy_options.get(name) is not None for name in ("sequence_mask", "geometric_mask"))):
            raise ValueError("the critic shares the actor's token mass, so PPO keeps token-mean "
                             "aggregation and no sequence masks")
        self.actor = GRPOObjective(model, seq_len, **policy_options)
        self.critic, self.seq_len = critic, seq_len
        self.value_coefficient, self.value_clip = value_coefficient, value_clip
        self.inputs, self.artifact = self.actor.inputs, self.actor.artifact
        reference = self.actor.ema
        self.ema = None if reference is None else EMASpec(reference.decay,
            lambda path: len(path) > 1 and path[1] == "policy" and reference.select((path[0], *path[2:])))

    def held_variables(self) -> Variables | None:
        """Return whatever the actor starts from: a loaded policy checkpoint.

        The critic is drawn from the key, so the actor's tree is the only
        held data here, and it reaches the trainer's state JIT as the
        initializer's argument rather than as a captured constant.
        """
        return self.actor.held_variables()

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        critic = self.critic.init(jax.random.fold_in(key, 1),
                                  jnp.zeros((1, self.seq_len), jnp.int32))
        return _join(self.actor.init(key, variables), critic)

    def policy(self, variables: Variables) -> EpisodeInference:
        """Bind the policy subtree when an episode collector supplies the full tree."""
        return _Policy(self.actor.policy(_part(variables, "policy")))

    def pipeline(self, state: TrainState, *, ema: bool = True, processor: Processor | None = None) -> TextGeneration:
        """Publish the trained actor, without the critic or the frozen KL reference."""
        actor_state = replace(state, params=_part(state.params, "policy"))
        return self.actor.pipeline(actor_state, ema=ema, processor=processor)

    def values(self, variables: Variables, batch: Mapping[str, object]) -> jax.Array:
        """Score the state before each packed id, `[rows, width]` aligned with `input_ids`.

        Entry t is the critic's value of the chain prefix that predicts id
        t, the state its action was taken from; chain starts and padding,
        which no action follows, are zero.
        """
        ids = jnp.asarray(batch[IDS_KEY], jnp.int32)
        segments = jnp.asarray(batch[SEGMENT_IDS_KEY], jnp.int32)
        values = self.critic.apply(_part(variables, "critic"), ids[:, :-1], segment_ids=segments[:, :-1],
                                   positions=jnp.asarray(batch[POSITIONS_KEY], jnp.int32)[:, :-1])
        if not isinstance(values, jax.Array) or values.shape != ids[:, :-1].shape:
            raise ValueError("PPO critic must return one scalar value per input position")
        aligned = jnp.concatenate([jnp.zeros((ids.shape[0], 1), jnp.float32), values.astype(jnp.float32)], axis=1)
        return jnp.where(jnp.asarray(batch[RESPONSE_MASK_KEY]) != 0, aligned, 0.0)

    def loss(self, params: Variables, batch, step: Step) -> tuple[Mean, Aux[Variables]]:
        """Add the actor's policy loss to the clipped value error on the same mass."""
        for field in (OLD_VALUES_KEY, RETURNS_KEY):
            if field not in batch or jnp.shape(batch[field]) != jnp.shape(batch[RESPONSE_MASK_KEY]):
                raise ValueError(f"PPO requires response-aligned {field} from the rollout")
        policy_step = replace(step, ema=None if step.ema is None else _part(step.ema, "policy"))
        pg, aux = self.actor.loss(_part(params, "policy"), batch, policy_step)
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY])
        terms = clipped_value_loss_terms(self.values(params, batch), jnp.asarray(batch[RETURNS_KEY]),
                                         jnp.asarray(batch[OLD_VALUES_KEY]), self.value_clip)
        critic = Mean(jnp.sum(jnp.where(mask != 0, terms, 0) * mask), pg.mass)
        metrics = {**aux.metrics, "critic/loss": mean_loss(critic)[0]}
        return Mean(pg.total + self.value_coefficient * critic.total, pg.mass), Aux(metrics)

    def evaluate(self, params: Variables, batch, step: Step):
        return self.actor.evaluate(_part(params, "policy"), batch, replace(step, ema=None))

    def preview(self, params: Variables, batch, step: Step, *, scored=None):
        return self.actor.preview(_part(params, "policy"), batch, replace(step, ema=None), scored=scored)


@dataclass(frozen=True)
class PPORollout:
    """Collect episodes, then add critic baselines and verl's masked GAE.

    GAE continues across the action tokens of all turns in one episode,
    wherever the packer placed them. Tool observations and padding have no
    support. The terminal verifier reward lands on the last action with zero
    tail bootstrap, matching the pinned verl GAE input convention; truncated
    episodes are masked by the packer and take no targets; a cohort with no
    trainable token at all returns zero targets and zero mass, while a single
    trainable token, whose whitening is undefined, is refused.
    """

    objective: PPOObjective
    episodes: EpisodeRollout
    gamma: float = 1.
    lam: float = .95

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (self.gamma, self.lam)):
            raise ValueError("PPO gamma and lam must be finite values in [0, 1]")
        if self.objective.seq_len != self.episodes.max_prompt_tokens + self.episodes.max_new_tokens - 1:
            raise ValueError("PPO objective and episode token budgets must agree")

    @cached_property
    def _compiled_values(self):
        return jax.jit(self.objective.values)

    def _order(self, batch: Mapping[str, np.ndarray], count: int) -> np.ndarray:
        """Per episode, the flat packed positions of its action ids in the order they were drawn.

        `[episodes, max_turns * max_new_tokens]`, padded with -1. A call's
        ids sit in one chain in order, so sorting by call then position
        reads an episode's actions across its chains and rows.
        """
        mask = np.asarray(batch[RESPONSE_MASK_KEY]).reshape(-1) != 0
        owner = np.asarray(batch[ROLLOUT_INDEX_KEY]).reshape(-1)
        call = np.asarray(batch[CALL_INDEX_KEY]).reshape(-1).astype(np.int64)
        order = np.full((count, self.episodes.max_turns * self.episodes.max_new_tokens), -1, np.int64)
        where = np.flatnonzero(mask)
        for episode in range(count):
            mine = where[owner[where] == episode]
            mine = mine[np.lexsort((mine, call[mine]))]
            order[episode, :mine.size] = mine
        return order

    def _targets(self, values: np.ndarray, batch: Mapping[str, np.ndarray],
                 rewards: np.ndarray) -> dict[str, np.ndarray]:
        """Run verl's masked GAE over each episode's actions, then whiten over every process.

        The reward of an episode lands on its last action id, and GAE runs
        across the actions of all its turns at once. Whitening reads the
        global action count and moments, as it did over one device batch.
        """
        order = self._order(batch, rewards.shape[0])
        keep = order >= 0
        flat = values.reshape(-1)
        baselines = np.where(keep, flat[np.maximum(order, 0)], 0).astype(np.float32)
        last = np.where(keep.any(axis=1), keep.shape[1] - 1 - np.argmax(keep[:, ::-1], axis=1), -1)
        token_rewards = np.zeros_like(baselines)
        scored = last >= 0
        token_rewards[scored, last[scored]] = rewards[scored]
        _, returns = gae(jnp.asarray(token_rewards), jnp.asarray(baselines), jnp.asarray(keep, jnp.float32),
                         self.gamma, self.lam)
        returns = np.asarray(returns)
        raw = returns - baselines
        moments = np.asarray([np.sum(keep), np.sum(raw * keep), np.sum(raw * raw * keep)], np.float64)
        if jax.process_count() > 1:
            moments = np.sum(multihost_utils.process_allgather(moments), axis=0)
        count, total, squares = moments
        mean = total / (count + MEAN_EPS)
        variance = (squares - 2 * mean * total + mean * mean * count) / (count + MEAN_EPS)
        advantages = (raw - mean) / np.sqrt(variance * (count / (count - 1)) + WHITEN_EPS)
        shape = values.shape
        placed_advantages = np.zeros(flat.shape, np.float32)
        placed_returns = np.zeros(flat.shape, np.float32)
        placed_advantages[order[keep]] = advantages[keep]
        placed_returns[order[keep]] = returns[keep]
        return {OLD_VALUES_KEY: values.astype(np.float32), ADVANTAGES_KEY: placed_advantages.reshape(shape),
                RETURNS_KEY: placed_returns.reshape(shape)}

    def __call__(self, state: TrainState, batch: Mapping[str, object], key: jax.Array) -> dict[str, np.ndarray]:
        """Collect one cohort of episodes and return its packed rows with critic targets."""
        episodes = self.episodes.collect(state, batch, key)
        projected = agreed("PPO episode projection", lambda: self.episodes.project(episodes))
        count = np.asarray(min(2, np.count_nonzero(projected[RESPONSE_MASK_KEY])), np.int32)
        if jax.process_count() > 1:
            count = np.sum(multihost_utils.process_allgather(count))
        if int(count) == 0:
            # Every episode was masked (truncated): zero mass, so the step makes no update.
            zeros = np.zeros(projected[IDS_KEY].shape, np.float32)
            return {**projected, OLD_VALUES_KEY: zeros, ADVANTAGES_KEY: zeros, RETURNS_KEY: zeros}
        if int(count) < 2:
            raise ValueError("PPO GAE whitening requires at least two action tokens globally")
        mesh = mesh_of(state.params)
        device = agreed("PPO critic inputs", lambda: projected if mesh is None else shard_batch(mesh, projected))
        values = agreed("PPO critic values", lambda: local_rows(self._compiled_values(state.params, device)))
        rewards = np.asarray([0.0 if episode.reward is None else episode.reward for episode in episodes], np.float32)
        return {**projected, **self._targets(np.asarray(values), projected, rewards)}
