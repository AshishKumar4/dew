"""PPO with a separate trainable critic, clipped value loss and episode GAE."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental import multihost_utils

from dew.coordination import agreed
from dew.inference.tasks import Processor, TextGeneration
from dew.nn.inputs import ModelInputs, local_rows, mesh_of
from dew.nn.protocols import HiddenStates
from dew.objectives.base import (
    OMITTED,
    Aux,
    EMASpec,
    Objective,
    Omitted,
    ProgramModule,
    Ratio,
    Shown,
    Step,
    Variables,
    joined,
    part,
)
from dew.records import JSON, json_value, record
from dew.registry import objectives
from dew.rl import gae
from dew.rl.advantage import MEAN_EPS, WHITEN_EPS
from dew.rl.surrogate import clipped_value_loss_terms
from dew.sampling.text import Generation, Sampling
from dew.training.distributed import shard_batch
from dew.training.state import TrainState

from .episodes import EpisodeInference, EpisodeRollout
from .grpo import GRPOObjective, onto_ids
from .sessions import (
    ADVANTAGES_KEY,
    CALL_INDEX_KEY,
    IDS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
    SESSION_INDEX_KEY,
)

OLD_VALUES_KEY = "old_values"
RETURNS_KEY = "returns"


class ValueHead(nn.Module):
    """Projects a decoder's hidden states to one float32 value per position.

    Packed rows pass their chains' `segment_ids` and `positions`, so no
    hidden state sees another chain.
    """

    backbone: HiddenStates

    @nn.compact
    def __call__(self, tokens: jax.Array, *, segment_ids: jax.Array | None = None,
                 positions: jax.Array | None = None) -> jax.Array:
        hidden = self.backbone.hidden_states(
            tokens, train=False, segment_ids=segment_ids, positions=positions
        )
        return nn.Dense(1, dtype=jnp.float32, name="value")(hidden)[..., 0]


@dataclass(frozen=True)
class _Policy:
    task: EpisodeInference

    def bind(self, variables: Variables, /) -> EpisodeInference:
        return self.task.bind(part(variables, "policy"))

    def __call__(self, inputs: ModelInputs | Sequence[Sequence[int]], max_new_tokens: int, /,
                 *, key: jax.Array, sampling: Sampling) -> Generation:
        return self.task(inputs, max_new_tokens, key=key, sampling=sampling)


@objectives("ppo")
class PPOObjective(Objective[Ratio, Variables]):
    """Trains a policy and a critic together, with both losses over the same token mass.

    The `params` collection holds a `policy` subtree and a `critic` subtree,
    and the ordinary `Trainer` optimizes both. The frozen reference, an EMA
    at unit decay, tracks only the policy leaves. Rollout targets are
    detached.

    The policy is a `GRPOObjective` built from `model`, `seq_len` and the
    remaining keyword arguments, so `beta` and the clip settings work as they
    do there. It keeps `"token-mean"` aggregation and takes no sequence
    masks, because the critic shares the actor's token mass.
    `value_coefficient` weights verl's half-squared, clipped value error, and
    `value_clip` is that error's clip range.

    `critic` reads packed token rows with their `segment_ids` and
    `positions` and returns `[B, T]` values; `ValueHead` gives a decoder that
    interface.
    """

    # The loss is a policy-gradient surrogate plus the critic's, so only the
    # critic's own has a direction.
    shown: Mapping[str, Shown] = {"loss": Shown(), "critic/loss": Shown(better="lower")}

    saved_task = TextGeneration
    _ema_is_reference = True
    _model_part = "policy"

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

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The actor's modules, then the critic."""
        return (*self.actor.program_key(), ProgramModule(self.critic, None, trained=True))

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        *actor, self.critic = modules
        self.actor.substitute(actor)

    def held_variables(self) -> Variables | None:
        """Return the actor's held variables, such as a loaded policy checkpoint, or None.

        The critic is initialized from the key, so the actor's tree is the
        only held data. It reaches the trainer's state JIT as the
        initializer's argument, not as a captured constant.
        """
        return self.actor.held_variables()

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        critic = self.critic.init(jax.random.fold_in(key, 1),
                                  jnp.zeros((1, self.seq_len), jnp.int32))
        return joined({"policy": self.actor.init(key, variables), "critic": critic})

    def policy(self, variables: Variables) -> EpisodeInference:
        """Return the actor's episode inference task, whose `bind` reads only a tree's `policy` subtree."""
        return _Policy(self.actor.policy(part(variables, "policy")))

    def inference_record(self) -> JSON:
        """Return the actor's decoder record under PPO's registered name, or None when the actor has none.

        A loader rebuilds the decoder from it and takes the policy half of
        the saved tree."""
        actor = self.actor.inference_record()
        if actor is None:
            return None
        return json_value({**record(actor, 'inference record'), 'objective': objectives.name_of(type(self))},
                          'inference record')

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> TextGeneration:
        """Return the trained actor as a `TextGeneration`, without the critic or the frozen KL reference."""
        return self.actor.build_task(part(variables, "policy"), processor=processor)

    def values(self, variables: Variables, batch: Mapping[str, object]) -> jax.Array:
        """Return the critic's float32 value of each packed id's prefix, placed by `onto_ids`.

        A critic that does not return one value per input position raises
        `ValueError`.
        """
        ids = jnp.asarray(batch[IDS_KEY], jnp.int32)
        segments = jnp.asarray(batch[SEGMENT_IDS_KEY], jnp.int32)
        values = self.critic.apply(part(variables, "critic"), ids[:, :-1], segment_ids=segments[:, :-1],
                                   positions=jnp.asarray(batch[POSITIONS_KEY], jnp.int32)[:, :-1])
        if not isinstance(values, jax.Array) or values.shape != ids[:, :-1].shape:
            raise ValueError("PPO critic must return one scalar value per input position")
        return onto_ids(batch, values.astype(jnp.float32))

    def loss(self, variables: Variables, batch, step: Step) -> tuple[Ratio, Aux[Variables]]:
        """Return the actor's policy loss plus the weighted, clipped value error, over the same mass.

        The batch must hold `old_values` and `returns` from the rollout,
        aligned with `response_mask`, or the loss raises `ValueError`.
        """
        for field in (OLD_VALUES_KEY, RETURNS_KEY):
            if field not in batch or jnp.shape(batch[field]) != jnp.shape(batch[RESPONSE_MASK_KEY]):
                raise ValueError(f"PPO requires response-aligned {field} from the rollout")
        policy_step = replace(step, ema=None if step.ema is None else part(step.ema, "policy"))
        pg, aux = self.actor.loss(part(variables, "policy"), batch, policy_step)
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY])
        terms = clipped_value_loss_terms(self.values(variables, batch), jnp.asarray(batch[RETURNS_KEY]),
                                         jnp.asarray(batch[OLD_VALUES_KEY]), self.value_clip)
        critic = Ratio(jnp.sum(jnp.where(mask != 0, terms, 0) * mask), pg.mass)
        metrics = {**aux.metrics, "critic/loss": critic.mean()[0]}
        return Ratio(pg.total + self.value_coefficient * critic.total, pg.mass), Aux(metrics)

    def evaluate(self, params: Variables, batch, step: Step):
        return self.actor.evaluate(part(params, "policy"), batch, replace(step, ema=None))

    def preview(self, params: Variables, batch, step: Step, *, scored=None):
        return self.actor.preview(part(params, "policy"), batch, replace(step, ema=None), scored=scored)


@dataclass(frozen=True)
class PPORollout:
    """Collects episodes, then adds critic baselines and targets from verl's masked GAE.

    GAE runs across the action tokens of all turns in one episode, wherever
    the packer placed them. Tool observations and padding are outside the
    mask, so they get no targets. The verifier's terminal reward goes on the
    last action, with no bootstrap value after it, which matches the input
    convention of the pinned verl GAE. The packer masks truncated episodes,
    so they take no targets either.

    A cohort with no trainable token at all returns zero targets and zero
    mass. A cohort with a single trainable token raises `ValueError`,
    because whitening is undefined for one token.

    `gamma` and `lam` are GAE's discount and lambda, each in [0, 1]. The
    objective's `seq_len` must equal the episodes' `max_prompt_tokens` plus
    `max_new_tokens` minus one.
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
        owner = np.asarray(batch[SESSION_INDEX_KEY]).reshape(-1)
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

    def __call__(
        self, state: TrainState, batch: Mapping[str, object], key: jax.Array
    ) -> dict[str, np.ndarray]:
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
        mesh = mesh_of(state.variables)
        device = agreed(
            "PPO critic inputs", lambda: projected if mesh is None else shard_batch(mesh, projected)
        )
        values = agreed(
            "PPO critic values", lambda: local_rows(self._compiled_values(state.variables, device)))
        rewards = np.asarray(
            [0.0 if episode.reward is None else episode.reward for episode in episodes], np.float32
        )
        return {**projected, **self._targets(np.asarray(values), projected, rewards)}
