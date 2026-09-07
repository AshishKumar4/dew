"""PPO with a separate trainable critic, clipped value loss and episode GAE."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
import math
from typing import Protocol

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
from jax.experimental import multihost_utils

from dew.data.prompts import LENGTH_KEY
from dew.nn.inputs import ModelInputs
from dew.objectives.base import (Aux, EMASpec, Initializer, Mean, Objective, Step, Variables,
                                 mean_loss)
from dew.objectives.lm.objective import _shift_rows
from dew.registry import objectives
from dew.rl import gae
from dew.rl.surrogate import clipped_value_loss_terms
from dew.sampling.text import Generation, Sampling, _mesh
from dew.training.distributed import local_rows, shard_batch
from dew.training.state import TrainState
from .episodes import EpisodeInference, EpisodeRollout, _phase
from .grpo import GRPOObjective
from .rollout import ADVANTAGES_KEY, IDS_KEY, RESPONSE_MASK_KEY, REWARDS_KEY

OLD_VALUES_KEY = "old_values"
RETURNS_KEY = "returns"


class ValueBackbone(Protocol):
    def hidden_states(self, tokens: jax.Array, train: bool = False, *,
                      attention_mask: jax.Array | None = None) -> jax.Array: ...


class ValueHead(nn.Module):
    """A decoder's hidden states projected to one float32 value per position."""

    backbone: ValueBackbone

    @nn.compact
    def __call__(self, tokens: jax.Array, *, attention_mask: jax.Array | None = None) -> jax.Array:
        hidden = self.backbone.hidden_states(tokens, train=False, attention_mask=attention_mask)
        return nn.Dense(1, dtype=jnp.float32, name="value")(hidden)[..., 0]


def _part(variables: Variables, name: str) -> Variables:
    return {collection: subtree[name] for collection, subtree in variables.items() if name in subtree}


def _join(policy: Variables, critic: Variables) -> Variables:
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
    """The PPO policy objective and clipped critic error on the same token mass.

    The params collection holds policy and critic subtrees, both optimized by
    the ordinary Trainer. The unit-decay reference selects only policy leaves.
    Rollout targets are detached. beta and policy clip controls are GRPO's
    existing composition; value_coefficient weights verl's half-squared,
    clipped value error. A critic consumes token rows and attention_mask and
    returns [B, T] values; ValueHead supplies that interface for a decoder.
    """

    def __init__(self, model, seq_len: int, *, critic: nn.Module,
                 value_coefficient: float = .5, value_clip: float = .2, **policy_options):
        for name, value in (("value_coefficient", value_coefficient), ("value_clip", value_clip)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.actor = GRPOObjective(model, seq_len, **policy_options)
        self.critic, self.seq_len = critic, seq_len
        self.value_coefficient, self.value_clip = value_coefficient, value_clip
        self.inputs, self.artifact = self.actor.inputs, self.actor.artifact
        reference = self.actor.ema
        self.ema = None if reference is None else EMASpec(reference.decay,
            lambda path: len(path) > 1 and path[1] == "policy" and reference.select((path[0], *path[2:])))

    @property
    def initializer(self) -> Initializer:
        """The actor's initializer bound as this one's argument.

        A `Partial` is itself a pytree, so whatever the actor holds - a
        loaded policy checkpoint - travels as data through this objective's
        initializer too, rather than being captured when the trainer traces
        the initial state.
        """
        return Partial(self._initialize, self.actor.initializer)

    def init(self, key: jax.Array) -> Variables:
        return self._initialize(self.actor.initializer, key)

    def _initialize(self, policy: Initializer, key: jax.Array) -> Variables:
        """The one initialization implementation `init` and `initializer` share."""
        critic = self.critic.init(jax.random.fold_in(key, 1),
                                  jnp.zeros((1, self.seq_len), jnp.int32))
        return _join(policy(key), critic)

    def policy(self, variables: Variables) -> EpisodeInference:
        """Bind the policy subtree when an episode collector supplies the full tree."""
        return _Policy(self.actor.policy(_part(variables, "policy")))

    def values(self, variables: Variables, batch: Mapping[str, object]) -> jax.Array:
        """Values of states before each response action, with left padding removed."""
        ids = jnp.asarray(batch[IDS_KEY], jnp.int32)
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY])
        start = ids.shape[1] - mask.shape[1] - 1
        padding = (jnp.zeros(ids.shape[0], jnp.int32) if LENGTH_KEY not in batch else
                   start + 1 - jnp.asarray(batch[LENGTH_KEY], jnp.int32))
        aligned = _shift_rows(ids, padding)[:, :-1]
        valid = jnp.arange(aligned.shape[1])[None, :] < aligned.shape[1] - padding[:, None]
        values = self.critic.apply(_part(variables, "critic"), aligned, attention_mask=valid)
        if not isinstance(values, jax.Array) or values.shape != aligned.shape:
            raise ValueError("PPO critic must return one scalar value per input position")
        return _shift_rows(values, -padding)[:, start:start + mask.shape[1]]

    def loss(self, params: Variables, batch, step: Step) -> tuple[Mean, Aux[Variables]]:
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
    """Episode collection followed by critic baselines and verl's masked GAE.

    GAE continues across the action tokens of all turns in one episode. Tool
    observations and unused slots have no support. The terminal verifier
    reward lands on the last action; completed and budget-truncated episodes
    have zero tail bootstrap, matching the pinned verl GAE input convention.
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

    def _targets(self, variables: Variables, batch) -> dict[str, jax.Array]:
        values = self.objective.values(variables, batch)
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY]).reshape(-1, self.episodes.max_turns * values.shape[1])
        baselines = values.reshape(mask.shape)
        reward = jnp.asarray(batch[REWARDS_KEY]).reshape(-1, self.episodes.max_turns)[:, 0]
        positions = jnp.arange(mask.shape[1])[None, :]
        last = jnp.max(jnp.where(mask != 0, positions, -1), axis=1)
        rewards = jnp.where(positions == last[:, None], reward[:, None], 0)
        advantage, returns = gae(rewards, baselines, mask, self.gamma, self.lam)
        return {OLD_VALUES_KEY: values, ADVANTAGES_KEY: advantage.reshape(values.shape),
                RETURNS_KEY: returns.reshape(values.shape)}

    @cached_property
    def _compiled_targets(self):
        return jax.jit(self._targets)

    def __call__(self, state: TrainState, batch: Mapping[str, object], key: jax.Array) -> dict[str, np.ndarray]:
        episodes = self.episodes.collect(state, batch, key)
        projected = _phase(lambda: self.episodes.tensors(episodes), "PPO episode tensors")
        count = np.asarray(min(2, np.count_nonzero(projected[RESPONSE_MASK_KEY])), np.int32)
        if jax.process_count() > 1:
            count = np.sum(multihost_utils.process_allgather(count))
        if int(count) < 2:
            raise ValueError("PPO GAE whitening requires at least two action tokens globally")
        mesh = _mesh(state.params)
        device = _phase(lambda: projected if mesh is None else shard_batch(mesh, projected), "PPO critic inputs")
        targets = _phase(lambda: self._compiled_targets(state.params, device), "PPO critic targets")
        return {**projected, **{name: local_rows(value) for name, value in targets.items()}}
