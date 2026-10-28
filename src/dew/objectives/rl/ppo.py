"""PPO with a separate trainable critic, clipped value loss and episode GAE."""

import contextlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct

from dew._coordination import agreed, from_every_process
from dew.batch import Batch, Variables, part
from dew.distributed.batches import shard_batch
from dew.inference.tasks import Processor, TextGeneration, TextSettings
from dew.nn.dense import Dense
from dew.nn.inputs import ModelInputs, local_rows, mesh_of
from dew.nn.precision import at_least_fp32
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
    joined,
    projected_draw,
    projected_parts,
)
from dew.objectives.routing import ExpertBalance, RoutedStatistics
from dew.rl.advantage import gae_advantages, masked_whiten
from dew.rl.surrogate import clipped_value_loss_terms
from dew.sampling.text import Generation, Sampling
from dew.state import TrainState

from .episodes import EpisodeCollector, EpisodeInference
from .grpo import GRPOObjective, onto_ids
from .scheduler import RolloutScheduler, SchedulerRecord, Synchronous, task_ids
from .sessions import (
    ADVANTAGES_KEY,
    CALL_INDEX_KEY,
    IDS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
    SESSION_INDEX_KEY,
    TOKEN_REWARDS_KEY,
)

if TYPE_CHECKING:
    from dew.objectives.base import SavedTask, Task

OLD_VALUES_KEY = "old_values"
RETURNS_KEY = "returns"


class ValueHead(nn.Module):
    """Projects a decoder's hidden states to one value per position, at least float32: a bf16 or
    float32 backbone's in float32, a float64 twin's in float64.

    Packed rows pass their chains' `segment_ids` and `positions`, so no
    hidden state sees another chain.
    """

    backbone: HiddenStates

    @nn.nowrap
    def projected(self, params: Variables, *, updated: bool) -> Variables:
        """The backbone's projection of its own parameters (`ProjectsParameters`); the value map's
        as they are."""
        return projected_parts(params, {"backbone": self.backbone}, updated=updated)

    @nn.compact
    def __call__(self, tokens: jax.Array, *, segment_ids: jax.Array | None = None,
                 positions: jax.Array | None = None) -> jax.Array:
        hidden = self.backbone.hidden_states(
            tokens, train=False, segment_ids=segment_ids, positions=positions
        )
        return Dense(1, dtype=at_least_fp32(hidden.dtype), name="value")(hidden)[..., 0]


@dataclass(frozen=True)
class _Policy:
    task: EpisodeInference

    def bind(self, variables: Variables, /) -> EpisodeInference:
        return self.task.bind(part(variables, "policy"))

    def __call__(self, inputs: ModelInputs | Sequence[Sequence[int]], max_new_tokens: int, /,
                 *, key: jax.Array, sampling: Sampling) -> Generation:
        return self.task(inputs, max_new_tokens, key=key, sampling=sampling)


@struct.dataclass
class PPOStatistics:
    """The actor's loss statistics, with the router terms its policy adds where it balances, and the
    critic's independently normalized value error."""

    actor: Ratio | RoutedStatistics
    critic: Ratio


class PPOObjective(Objective[PPOStatistics | Ratio, Variables]):
    """Trains a policy and a critic together, each over its own loss mass.

    The `params` collection holds a `policy` subtree and a `critic` subtree,
    and the ordinary `Trainer` optimizes both. The frozen reference, an EMA
    at unit decay, tracks only the policy leaves. Rollout targets are
    detached.

    The policy is a `GRPOObjective` built from `model`, `seq_len`, `balance` and the remaining keyword
    arguments, so its aggregation, sequence masks, `beta` and clip settings work as they do there.
    `balance` (`ExpertBalance`, `dew.objectives.routing`) is named here rather than left to
    `policy_options`, so a run's record and its flags read it as the `ExpertBalance` it is. The critic
    always averages over the response tokens, independently of the actor's weighting.
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
    settings = TextSettings
    ema_is_reference = True
    _model_part = "policy"

    def __init__(self, model, seq_len: int, *, critic: nn.Module,
                 value_coefficient: float = .5, value_clip: float = .2,
                 balance: ExpertBalance | None = None, **policy_options):
        for name, value in (("value_coefficient", value_coefficient), ("value_clip", value_clip)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.actor = GRPOObjective(model, seq_len, balance=balance, **policy_options)
        self.critic, self.seq_len = critic, seq_len
        self.value_coefficient, self.value_clip = value_coefficient, value_clip
        self.inputs = self.actor.inputs
        reference = self.actor.ema
        self.ema = None if reference is None else EMASpec(reference.decay,
            lambda path: len(path) > 1 and path[1] == "policy" and reference.select((path[0], *path[2:])))

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The actor's modules, then the critic."""
        return (*self.actor.program_key(), ProgramModule(self.critic, None, trained=True))

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        *actor, self.critic = modules
        self.actor.substitute(actor)

    def row_tokens(self, batch: Batch) -> int | None:
        """The actor's tokens a row, of the rows the actor and the critic both read."""
        return self.actor.row_tokens(batch)

    def projected(self, params: Variables, *, updated: bool) -> Variables:
        """The actor's projection of the policy's tree, and the critic's of its own."""
        return projected_parts(params, {"policy": self.actor, "critic": self.critic}, updated=updated)

    def tile_head(self, tile: tuple[int, int] | None = None) -> str | None:
        """Tile the actor's head, the vocabulary head this objective trains."""
        return self.actor.tile_head(tile)

    def restore_head(self, tile: tuple[int, int] | None, *, fused: bool) -> None:
        self.actor.restore_head(tile, fused=fused)

    def held_variables(self) -> Variables | None:
        """Return the actor's held variables, such as a loaded policy checkpoint, or None.

        The critic is initialized from the key, so the actor's tree is the
        only held data. It reaches the trainer's state JIT as the
        initializer's argument, not as a captured constant.
        """
        return self.actor.held_variables()

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        critic = projected_draw(self.critic, self.critic.init(jax.random.fold_in(key, 1),
                                                              jnp.zeros((1, self.seq_len), jnp.int32)))
        return joined({"policy": self.actor.init(key, variables), "critic": critic})

    def policy(self, variables: Variables) -> EpisodeInference:
        """Return the actor's episode inference task, whose `bind` reads only a tree's `policy` subtree."""
        return _Policy(self.actor.policy(part(variables, "policy")))

    def task_settings(self) -> TextSettings | None:
        """The actor's decoder settings; a loader rebuilds the decoder from them under PPO's
        own record and takes the policy half of the saved tree."""
        return self.actor.task_settings()

    def recorded_model(self) -> nn.Module:
        """The actor's decoder, without the critic."""
        return self.actor.recorded_model()

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> "Task | SavedTask":
        """Return the trained actor's task, its `TextGeneration`, without the critic or the frozen KL
        reference."""
        return self.actor.build_task(part(variables, "policy"), processor=processor)

    def values(self, variables: Variables, batch: Mapping[str, object]) -> jax.Array:
        """Return the critic's value of each packed id's prefix, at least float32, placed by `onto_ids`.

        A critic that does not return one value per input position raises
        `ValueError`.
        """
        return self._critic_scores(variables, batch)[0]

    def _critic_scores(self, variables: Variables, batch: Mapping[str, object], *, qk_stats: bool = False
                       ) -> tuple[jax.Array, Variables | None]:
        """The critic's values and, when requested, the QK collection from that same forward."""
        ids = jnp.asarray(batch[IDS_KEY], jnp.int32)
        segments = jnp.asarray(batch[SEGMENT_IDS_KEY], jnp.int32)
        values = self.critic.apply(part(variables, "critic"), ids[:, :-1], segment_ids=segments[:, :-1],
                                   positions=jnp.asarray(batch[POSITIONS_KEY], jnp.int32)[:, :-1],
                                   mutable=["qk"] if qk_stats else False)
        qk = None
        if qk_stats:
            values, gathered = values
            qk = gathered.get("qk", {})
        if not isinstance(values, jax.Array) or values.shape != ids[:, :-1].shape:
            raise ValueError("PPO critic must return one scalar value per input position")
        return onto_ids(batch, values.astype(at_least_fp32(values.dtype))), qk

    def loss(self, variables: Variables, batch, step: Step) -> tuple[PPOStatistics, Aux[Variables]]:
        """Return independent statistics for the actor and the clipped value error.

        The batch must hold `old_values` and `returns` from the rollout,
        aligned with `response_mask`, or the loss raises `ValueError`.
        """
        for column in (OLD_VALUES_KEY, RETURNS_KEY):
            if column not in batch or jnp.shape(batch[column]) != jnp.shape(batch[RESPONSE_MASK_KEY]):
                raise ValueError(f"PPO requires response-aligned {column} from the rollout")
        policy_step = replace(step, ema=None if step.ema is None else part(step.ema, "policy"))
        pg, aux = self.actor.loss(part(variables, "policy"), batch, policy_step)
        mask = jnp.asarray(batch[RESPONSE_MASK_KEY])
        values, critic_qk = self._critic_scores(variables, batch, qk_stats=step.qk_stats)
        terms = clipped_value_loss_terms(values, jnp.asarray(batch[RETURNS_KEY]),
                                         jnp.asarray(batch[OLD_VALUES_KEY]), self.value_clip)
        critic = Ratio(jnp.sum(jnp.where(mask != 0, terms, 0) * mask), jnp.sum(mask))
        metrics = {**aux.metrics, "critic/loss": critic.mean()[0]}
        qk = None if aux.qk_stats is None else {"policy": aux.qk_stats}
        if critic_qk is not None:
            qk = {**(qk or {}), "critic": critic_qk}
        return PPOStatistics(pg, critic), Aux(metrics, qk_stats=qk, effects=aux.effects)

    def reduce_loss(self, stats: PPOStatistics | Ratio) -> tuple[jax.Array, jax.Array]:
        """Combine the independently normalized actor and weighted critic losses, or the validation NLL."""
        if isinstance(stats, Ratio):
            return stats.mean()
        actor, actor_active = self.actor.reduce_loss(stats.actor)
        critic, _ = stats.critic.mean()
        critic_active = self.value_coefficient * stats.critic.mass > 0
        return actor + self.value_coefficient * critic, actor_active | critic_active

    def apply_effects(self, variables: Variables, effects: Variables) -> Variables:
        """The policy's router counts applied (`GRPOObjective.apply_effects`), in the policy's own
        collections and the critic's left as they are."""
        policy = self.actor.apply_effects(part(variables, "policy"), effects)
        return {name: {**variables[name], "policy": tree} for name, tree in policy.items()}

    def validation_loss(self, variables: Variables, batch, step: Step):
        """Score source prompts under the live actor, without rollout targets or the critic."""
        return self.actor.validation_loss(part(variables, "policy"), batch, replace(step, ema=None))

    def _validation_scores(self, params: Variables, batch, step: Step):
        return self.actor._validation_scores(part(params, "policy"), batch, replace(step, ema=None))

    def evaluate(self, params: Variables, batch, step: Step):
        return self.actor.evaluate(part(params, "policy"), batch, replace(step, ema=None))

    def preview(self, params: Variables, batch, step: Step, *, scored=None):
        return self.actor.preview(part(params, "policy"), batch, replace(step, ema=None), scored=scored)


def _over_processes(moments: jax.Array) -> jax.Array:
    """`moments` summed over every process of the pool (`from_every_process`), in float64."""
    shared = from_every_process(np.asarray(moments, np.float64).tolist())
    return jnp.asarray(np.sum(np.asarray(shared, np.float64), axis=0), moments.dtype)


@dataclass(frozen=True)
class PPORollout:
    """Collects episodes through `rollout`, then adds critic baselines and targets from verl's masked GAE.

    `episodes` collects `groups` episodes of each task through the
    rollout scheduler (`RolloutScheduler`, `Synchronous`), its policy
    `objective.policy({})`, which binds each step's policy subtree; `log`
    receives the scheduler's record of each call. GAE runs across the
    action tokens of all turns in one episode, wherever the packer placed
    them. Tool observations and padding are
    outside the mask, so they get no targets. The rewards are the packed
    `token_rewards`: each step reward the environment paid
    (`Observation.reward`) on the last action id of the call it paid for,
    and the verifier's terminal reward on the episode's last action, with no
    bootstrap value after it, which matches the input convention of the
    pinned verl GAE. The packer masks truncated episodes, so they take no
    targets either.

    A cohort with no trainable token at all returns zero targets and zero
    mass. A cohort with a single trainable token raises `ValueError`,
    because whitening is undefined for one token.

    `gamma` and `lam` are GAE's discount and lambda, each in [0, 1]. The
    objective's `seq_len` must equal the episodes' `max_prompt_tokens` plus
    `max_new_tokens` minus one.
    """

    objective: PPOObjective
    episodes: EpisodeCollector
    groups: int = 2
    gamma: float = 1.
    lam: float = .95
    log: Callable[[SchedulerRecord], None] | None = None
    rollout: RolloutScheduler = field(init=False)

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (self.gamma, self.lam)):
            raise ValueError("PPO gamma and lam must be finite values in [0, 1]")
        if self.objective.seq_len != self.episodes.width - 1:
            raise ValueError("PPO objective and episode token budgets must agree")
        object.__setattr__(self, "rollout", RolloutScheduler(Synchronous(self.episodes), groups=self.groups,
                                                             log=self.log))

    @property
    def metrics(self) -> Mapping[str, float]:
        return self.rollout.metrics

    @property
    def shown(self) -> Mapping[str, Shown]:
        return self.rollout.shown

    def branch(self, name: str) -> contextlib.AbstractContextManager[None]:
        """The cooldown branch `name`'s calls (`dew.training.rollout`), as its scheduler draws them
        (`RolloutScheduler.branch`)."""
        return self.rollout.branch(name)

    def _order(self, batch: Mapping[str, np.ndarray], count: int) -> np.ndarray:
        """Per episode, the flat packed positions of its action ids in the order they were drawn.

        `[episodes, max_turns * max_new_tokens]`, padded with -1. A call's ids
        sit in one chain in order, so sorting by call then position reads an
        episode's actions across its chains and rows.
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
                 count: int) -> dict[str, np.ndarray]:
        """Run verl's masked GAE over each episode's actions, then whiten over every process.

        GAE runs across the actions of all an episode's turns at once, on
        the packed `token_rewards`. Whitening reads the global action count
        and moments, as it did over one device batch.
        """
        order = self._order(batch, count)
        keep = order >= 0
        flat = values.reshape(-1)
        placed = np.maximum(order, 0)
        baselines = np.where(keep, flat[placed], 0).astype(np.float32)
        token_rewards = np.where(keep, np.asarray(batch[TOKEN_REWARDS_KEY]).reshape(-1)[placed], 0
                                 ).astype(np.float32)
        raw = np.asarray(gae_advantages(jnp.asarray(token_rewards), jnp.asarray(baselines),
                                        jnp.asarray(keep, jnp.float32), self.gamma, self.lam))
        returns = raw + baselines
        advantages = np.asarray(masked_whiten(jnp.asarray(raw), jnp.asarray(keep), _over_processes))
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
        projected = self.rollout(state, batch, key)
        episodes = len(task_ids(batch)) * self.groups
        count = sum(from_every_process(min(2, int(np.count_nonzero(projected[RESPONSE_MASK_KEY])))))
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
        critic = self.objective.program("values", self.objective.values)
        values = agreed("PPO critic values", lambda: local_rows(critic(state.variables, device)))
        return {**projected, **self._targets(np.asarray(values), projected, episodes)}
