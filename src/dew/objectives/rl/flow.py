"""Flow-GRPO over recorded rectified-flow transitions.

The policy ratio uses the mean log likelihood over coordinates, as the
released implementation does, and not the product of every coordinate's
likelihood ratio. The KL term is the conditional Gaussian KL from
arXiv:2505.05470v5 section 4, also averaged over coordinates, and its
variance includes the elapsed time. The released scripts/train_sd3.py
computes a different term: it divides the squared mean displacement by the
squared diffusion coefficient, which equals the elapsed time times this
conditional KL. The two regularizers therefore weight the steps differently.

Callback scores stay in float64 through the host grouping, where the
released trainer converted them to float32 earlier.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.coordination import agreed, agreed_same, broadcast_from_process_zero, collective_host
from dew.data.dataset import rows_of
from dew.diffusion.presets import Preset
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import SourceGrid
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.nn.precision import at_least_fp32, weighted_total
from dew.objectives.base import OMITTED, Aux, Batch, Omitted, Ratio, Shown, Step, Variables
from dew.objectives.diffusion.objective import DiffusionObjective
from dew.rl.advantage import group_normalised
from dew.rl.surrogate import clipped_surrogate_terms
from dew.sampling.flow import FlowSDE, FlowTrajectory, GaussianTransition
from dew.sampling.guidance import CFG, Walk
from dew.sampling.solvers import Euler, Solver
from dew.training.distributed import row_origins, shared_host_fields, split_fields, with_host

from .sessions import ADVANTAGES_KEY, OLD_LOG_PROBS_KEY

REWARDS_KEY = "rewards"
"""The batch field that holds each image's reward, the score its trajectory got."""

if TYPE_CHECKING:
    from dew.training.state import TrainState

Predictor = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]



def _source(inputs: InputSpec, batch: Batch) -> jax.Array:
    """Read the real row-bearing field used by rollout, scoring and preview."""
    conditions = inputs.conditions
    name = next(iter(conditions.values())).field if conditions else inputs.sample.key
    source = jnp.asarray(jax.tree.leaves(batch[name])[0])
    if source.ndim == 0 or source.shape[0] == 0:
        raise ValueError("flow sampling needs a non-empty leading batch dimension")
    return source



_DEFAULT_SDE = FlowSDE()
_DEFAULT_GUIDANCE = CFG(3.0)
_DEFAULT_SOLVER = Euler()


class FlowGRPOObjective(DiffusionObjective):
    """Trains a rectified-flow policy with a clipped policy gradient, normalized per coordinate.

    A conditional transition KL regularizes it.

    A batch holds `latents` and `next_latents` as `[N, K, ...]`; `timesteps`,
    `next_timesteps`, the joint `old_log_probs` and `transition_mask` as
    `[N, K]`; and `advantages` as `[N]` or `[N, K]`. K is the number of
    selected transitions. The loss's denominator counts the kept stochastic
    transitions, and deterministic intervals contribute no policy loss.

    With `beta` > 0, the objective freezes the initial denoiser in the
    existing EMA slot, which then holds a reference instead of an average.
    The task a run publishes and restores, its evaluation and its previews
    all use the live policy. `solver` and `steps` configure evaluation, and
    `sde` configures both the rollout and the rescoring. `variables` is the
    whole variables tree the policy starts from, as `DiffusionObjective`
    takes it: the model's collections, `encoders` and any `autoencoder`.
    The batches it trains on are a `FlowRollout`'s, which a run builds from
    its `FlowRolloutConfig`.
    """

    # The loss is a policy-gradient surrogate, shown without a direction.
    shown: Mapping[str, Shown] = {"loss": Shown(), "reward": Shown(better="higher")}
    # With beta > 0 the EMA slot holds the frozen reference, so pipelines,
    # evaluation and previews read the live weights.
    _ema_is_reference = True

    def __init__(self, model: nn.Module, process: Process | Preset, inputs: InputSpec, *,
                 sde: FlowSDE = _DEFAULT_SDE, beta: float = 0.0,
                 clip_range: float = 1e-4, adv_clip_max: float = 5.0,
                 autoencoder: AutoEncoder | None = None,
                 guidance: CFG | None = _DEFAULT_GUIDANCE, solver: Solver = _DEFAULT_SOLVER,
                 steps: int = 41, variables: Variables | None = None,
                 grid: SourceGrid | None | Omitted = OMITTED):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if not math.isfinite(clip_range) or not 0 <= clip_range < 1:
            raise ValueError("clip_range must be finite and in [0, 1)")
        if not math.isfinite(adv_clip_max) or adv_clip_max <= 0:
            raise ValueError("adv_clip_max must be finite and positive")
        if steps < 2:
            raise ValueError("evaluation needs at least two time points")
        if variables is not None and "params" not in variables:
            raise ValueError("variables must be a variables tree with a params collection")
        super().__init__(model, process, inputs, autoencoder=autoencoder,
                         unconditional_prob=0, ema_decay=1.0, solver=solver,
                         guidance=guidance, steps=steps, variables=variables, grid=grid)
        if beta == 0:
            self.ema = None
        self.sde = sde
        self.beta = beta
        self.clip_range = clip_range
        self.adv_clip_max = adv_clip_max

    @property
    def validation_loss_refusal(self) -> str | None:
        return ("Flow-GRPO has no source-batch validation loss; rank checkpoints by a reward metric "
                "with an explicit Best, or pass no validation data")

    def _walk(self, params: Variables, batch: Batch) -> Walk:
        """The walk of the denoiser this batch's conditions select, guided as
        the rollout's walk was: over its `rollout_steps` points, a
        transition's column being its step's index."""
        given = self.encoded_conditions(params, batch)
        # An inpainting model's masked-image inputs, as the rollout encoded them (`FlowRollout`).
        spatial = batch.get("spatial", {})
        denoise = self.denoiser(params, {**given, **spatial},
                                {**self.blank_conditions(params, given), **spatial})
        points = jnp.asarray(batch["rollout_steps"], jnp.int32)
        return Walk.over(denoise, self.guidance, points[0] - 1)

    def _transition(self, predict: Predictor, x: jax.Array,
                    t: jax.Array, following: jax.Array) -> GaussianTransition:
        denoised, eps = predict(x, t)
        return self.sde.transition(x, t, following, denoised, eps, self.process)

    def _window(self, batch: Batch) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Read and validate one batch's recorded transitions.

        Returns the latents, the latents they step to, and both endpoints'
        times, all detached, in float32 or the latents' dtype where it is
        wider: a rollout's record is data, not a path the gradient runs back
        through.
        """
        dtype = at_least_fp32(jnp.result_type(batch["latents"]))
        latents = jnp.asarray(batch["latents"], dtype)
        following = jnp.asarray(batch["next_latents"], dtype)
        times = jnp.asarray(batch["timesteps"], dtype)
        next_times = jnp.asarray(batch["next_timesteps"], dtype)
        if latents.ndim != len(self.latent_shape) + 2 or latents.shape[2:] != self.latent_shape:
            raise ValueError(f"latents must hold [batch, transitions, {self.latent_shape}]")
        if following.shape != latents.shape:
            raise ValueError("latents and next_latents must share one shape")
        if times.shape != latents.shape[:2] or next_times.shape != times.shape:
            raise ValueError("timesteps and next_timesteps must mark every recorded transition")
        if not times.shape[0] or not times.shape[1]:
            raise ValueError("a recorded batch needs rows and transition slots")
        if np.shape(batch["rollout_steps"]) != times.shape[:1]:
            raise ValueError("rollout_steps must give every row its rollout's grid points")
        return jax.tree.map(jax.lax.stop_gradient, (latents, following, times, next_times))

    def log_probs(self, params: Variables, batch: Batch) -> jax.Array:
        """Return each recorded transition's joint log density under `params`, guided as the rollout was."""
        latents, following, times, next_times = self._window(batch)
        walk = self._walk(params, batch)

        def score(_, values):
            x, action, t, s, index = values
            return None, self._transition(walk.at((), index), x, t, s).log_prob(action)

        _, values = jax.lax.scan(score, None, (*(
            jnp.swapaxes(value, 0, 1) for value in (latents, following, times, next_times)),
            jnp.arange(times.shape[1])))
        return values.T

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        """Compute the clipped policy-gradient loss over the recorded transitions.

        The scan carries no state from one transition to the next. Each
        transition contributes its surrogate, its KL to the frozen reference,
        whether it counted and whether its ratio was clipped.
        """
        latents, following, times, next_times = self._window(batch)
        old = jax.lax.stop_gradient(jnp.asarray(batch[OLD_LOG_PROBS_KEY], jnp.float32))
        mask = jax.lax.stop_gradient(jnp.asarray(batch["transition_mask"], jnp.bool_))
        advantages = jax.lax.stop_gradient(jnp.asarray(batch[ADVANTAGES_KEY], jnp.float32))
        if old.shape != times.shape or mask.shape != times.shape:
            raise ValueError("old_log_probs and transition_mask must mark every transition")
        if advantages.shape == times.shape[:1]:
            advantages = jnp.broadcast_to(advantages[:, None], times.shape)
        if advantages.shape != times.shape:
            raise ValueError("advantages must hold one value per trajectory or per transition")
        advantages = jnp.clip(advantages, -self.adv_clip_max, self.adv_clip_max)
        walk = self._walk(variables, batch)
        reference = None
        if self.beta > 0:
            if step.ema is None:
                raise ValueError("FlowGRPO's conditional KL needs its frozen reference in step.ema")
            reference = self._walk(jax.tree.map(jax.lax.stop_gradient, step.ema), batch)
        dimensions = math.prod(self.latent_shape)

        def terms(_, values):
            x, action, t, s, old_log_prob, advantage, selected, index = values
            transition = self._transition(walk.at((), index), x, t, s)
            # NaN variance remains selected and fails the numerical check. Only
            # a zero-variance interval is outside the Gaussian policy's support.
            keep = selected & (transition.variance != 0)
            current = jnp.where(keep, transition.log_prob(action), 0) / dimensions
            previous = jnp.where(keep, old_log_prob, 0) / dimensions
            # Flow-GRPO's clip is PPO's without the dual cap, one term a sample.
            surrogate, _ = clipped_surrogate_terms((current - previous)[:, None], advantage, keep[:, None],
                                                   epsilon_low=self.clip_range, epsilon_high=self.clip_range,
                                                   dual_clip=None)
            pg, mass = weighted_total(surrogate[:, 0], keep)
            ratio = jnp.exp(current - previous)
            kl = jnp.asarray(0.0)
            if reference is not None:
                # The released scripts/train_sd3.py that the module docstring
                # compares this KL with is at 879042cf5707f8b90daa98d147d7deac2317c5da.
                reference_mean = self._transition(reference.at((), index), x, t, s).mean
                kl, _ = weighted_total(transition.kl(reference_mean) / dimensions, keep)
            clipped_count = (keep & (jnp.abs(ratio - 1) > self.clip_range)).sum()
            return None, (pg, kl, mass, clipped_count)

        _, summed = jax.lax.scan(terms, None, (*(jnp.swapaxes(value, 0, 1) for value in (
            latents, following, times, next_times, old, advantages, mask)), jnp.arange(times.shape[1])))
        pg, kl, mass, clipped_count = (value.sum() for value in summed)
        denominator = jnp.where(mass > 0, mass, 1)
        metrics = {"pg": pg / denominator, "actor/clipfrac": clipped_count / denominator}
        if self.beta > 0:
            metrics["transition_kl"] = kl / denominator
        if REWARDS_KEY in batch:
            metrics["reward"] = jnp.asarray(batch[REWARDS_KEY], jnp.float32).mean()
        return Ratio(pg + self.beta * kl, mass), Aux(metrics)

    def _rows(self, batch: Batch) -> int:
        """One sample per source row, which a prompt-only batch has too."""
        return _source(self.inputs, batch).shape[0]


type FlowReward = Callable[[np.ndarray, Batch], np.ndarray | jax.Array | Sequence[float]]
"""A function that scores decoded [-1, 1] samples, given their repeated source rows, one scalar per sample."""


@dataclass(frozen=True)
class FlowRollout:
    """Collects complete prompt groups and the first `train_steps` transitions of each trajectory.

    Each source row is sampled `groups` times. `steps` counts time points, so
    `steps=11` draws ten transitions, and `train_steps` None trains on all of
    them.

    A sample's advantage is its reward minus its group's mean, divided by
    the group's population standard deviation plus 1e-4, as Flow-GRPO's
    PerPromptStatTracker computes it. Each prompt row is its own group, so
    equal prompt text in other rows does not merge groups. Rows with zero
    advantage are masked, as the reference training loop filters them out.

    Rewards stay in float64 from the callback, through the JSON and byte
    transport, to the population statistics, and they stay float64 on the
    host. The training advantages are float32 after normalization. The `reward` metric
    is a float32 diagnostic, and JAX's device transfer also narrows the
    reward column when x64 is off.

    The trainer passes global arrays on every process. Generation runs on
    every process, rewards are computed once on process zero, and the result
    holds only the rows this process owns, for the trainer's `shard_batch`.
    Gathering to the host currently uses `collective_host`, which copies the
    whole trajectory to every process before each selects its local rows.
    """

    objective: FlowGRPOObjective
    reward: FlowReward
    groups: int = 4
    steps: int = 11
    train_steps: int | None = None

    def __post_init__(self) -> None:
        if self.groups < 2:
            raise ValueError("a flow reward group needs at least two samples")
        if self.steps < 2:
            raise ValueError("a flow rollout needs at least two time points")
        if self.train_steps is not None and not 1 <= self.train_steps < self.steps:
            raise ValueError("train_steps must select between one and steps-1 transitions")
        if self.objective.sde.noise_level == 0:
            raise ValueError("online FlowGRPO needs positive noise for stochastic transitions")
        self.objective.sde.validate(self.objective.process, self.objective.process.times(self.steps))
        reserved = {"latents", "next_latents", "timesteps", "next_timesteps", "rollout_steps", "spatial",
                    OLD_LOG_PROBS_KEY, ADVANTAGES_KEY, REWARDS_KEY, "transition_mask"}
        if any(condition.field in reserved for condition in self.objective.inputs.conditions.values()):
            raise ValueError("conditioning fields must not overwrite FlowGRPO transition fields")



    def _generate_impl(self, params: Variables, batch: Batch,
                       key: jax.Array) -> tuple[FlowTrajectory, jax.Array, dict[str, jax.Array]]:
        """Sample one SDE trajectory per row, decode its final samples, and return the masked-image
        inputs an inpainting model read, which its rescoring reads again."""
        objective = self.objective
        given = objective.encoded_conditions(params, batch)
        spatial = objective._spatial(params, batch, key)
        denoise = objective.denoiser(params, {**given, **spatial},
                                     {**objective.blank_conditions(params, given), **spatial})
        noise_key, sample_key = jax.random.split(key)
        count = _source(objective.inputs, batch).shape[0]
        initial = objective.process.noise(noise_key, (count, *objective.latent_shape))
        trajectory = objective.sde.trajectory(denoise, initial, self.steps,
                                              guidance=objective.guidance, key=sample_key)
        samples = trajectory.samples
        if objective.autoencoder is not None:
            samples = objective.autoencoder.decode(params["autoencoder"], samples)
        return trajectory, jnp.clip(samples, -1, 1), spatial

    def _owned_rows(self, source: jax.Array) -> slice | np.ndarray:
        """Select the expanded rows this process owns.

        Each source row becomes `groups` consecutive rows, so a rank's
        shard of the source picks out that many rows of the result.
        """
        if jax.process_count() == 1:
            return slice(None)
        owned: set[int] = set()
        for index in source.sharding.addressable_devices_indices_map(source.shape).values():
            if index is None:
                raise ValueError("source sharding has no addressable row index")
            owned.update(range(*index[0].indices(source.shape[0])))
        rows = np.asarray(sorted(owned), np.int64)
        return (rows[:, None] * self.groups + np.arange(self.groups)).reshape(-1)

    def _grouped(self, value, xp=jnp):
        """`value`'s rows each repeated into its group of samples, in the array namespace `xp`."""
        return xp.repeat(value, self.groups, axis=0)

    def _expanded(self, batch: Batch) -> tuple[Batch, slice | np.ndarray, int]:
        """Repeat every source row into its group of samples.

        Returns the expanded batch, the rows this process owns within it,
        and the source row count each leaf has to agree with.
        """
        source = _source(self.objective.inputs, batch)
        count = source.shape[0]
        owned = self._owned_rows(source)
        pooled = jax.process_count() > 1

        def repeat(leaf):
            value = jnp.asarray(leaf)
            if value.ndim == 0:
                return value
            if value.shape[0] != count:
                raise ValueError("all flow source rows must share one batch dimension")
            if pooled and value.is_fully_addressable:
                # Process 0 scores every sample against the leaves it holds.
                raise ValueError(
                    f"a flow rollout over a pool of {jax.process_count()} processes reads a batch placed "
                    f"across them (shard_batch), and a {value.shape} leaf of this one is held by this "
                    f"process alone")
            return self._grouped(value)

        return jax.tree.map(repeat, batch), owned, count

    def _transitions(self, trajectory: FlowTrajectory, context: Batch, spatial: Mapping[str, np.ndarray],
                     rewards: np.ndarray, owned: slice | np.ndarray) -> Batch:
        """Cut the trajectory into this rank's training rows.

        Rewards are centred within their prompt's group, only the first
        `train_steps` transitions train, and a row whose group gave it no
        advantage is masked out of the loss.
        """
        # The author code uses float64 population statistics on the host, and 1e-4; text GRPO's
        # estimator (`group_advantage`) float32 and ddof=1, which would change this loss.
        advantages = group_normalised(rewards.reshape(-1, self.groups), ddof=0, eps=1e-4, xp=np)
        advantages = advantages.astype(np.float32).reshape(-1)
        if not np.isfinite(advantages).all():
            raise ValueError("flow group advantages must remain finite")
        selected = self.steps - 1 if self.train_steps is None else self.train_steps
        local_advantages = advantages[owned]
        shape = (local_advantages.shape[0], selected)
        prepared = {condition.field: jax.tree.map(lambda leaf: np.asarray(leaf)[owned],
                                                 context[condition.field])
                    for condition in self.objective.inputs.conditions.values()}
        prepared.update({
            "latents": np.asarray(trajectory.states)[owned, :selected],
            "next_latents": np.asarray(trajectory.states)[owned, 1:selected + 1],
            "timesteps": np.broadcast_to(trajectory.times[:selected], shape),
            "next_timesteps": np.broadcast_to(trajectory.times[1:selected + 1], shape),
            "rollout_steps": np.full(shape[:1], self.steps, np.int32),
            OLD_LOG_PROBS_KEY: np.asarray(trajectory.log_probs)[owned, :selected],
            "transition_mask": (np.asarray(trajectory.stochastic)[owned, :selected]
                                & (local_advantages[:, None] != 0)),
            ADVANTAGES_KEY: local_advantages,
            REWARDS_KEY: rewards[owned],
        })
        if spatial:
            prepared["spatial"] = {name: np.asarray(value)[owned] for name, value in spatial.items()}
        return prepared

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch:
        """Collect one batch of trajectories and return this process's training rows.

        It expands each prompt into its group, generates on every process,
        scores on process zero and broadcasts the rewards. Then it centres
        the rewards within each group and cuts each trajectory to its first
        `train_steps` transitions. The processes agree that each phase
        succeeded before the next collective runs.
        """
        # The text a reward reads stays on each host (`split_fields`); the rest is placed. Every process
        # agrees which host fields there are before any of them gathers them.
        batch, host = split_fields(batch)
        fields = agreed_same("flow rollout host fields", lambda: sorted(
            jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(
                host, is_leaf=lambda column: not isinstance(column, Mapping))))
        expanded, owned, count = agreed("flow rollout setup", lambda: self._expanded(batch))
        # The objective owns the program; the rollout fixes only its count of time points.
        generate = self.objective.program(f"flow rollout over {self.steps} time points", self._generate_impl)
        generated = agreed("flow rollout generation", lambda: generate(state.variables, expanded, key))
        source = _source(self.objective.inputs, batch)
        # In a pool the source is placed across processes (`_expanded`), and each source row's origin rides
        # the rollout's own collective (`row_origins`), so its host fields meet in that order.
        placed = source.sharding
        mesh = (placed.mesh if isinstance(placed, jax.sharding.NamedSharding)
                and isinstance(placed.mesh, jax.sharding.Mesh) and not source.is_fully_addressable else None)
        rows = rows_of(host) if fields else 0
        origins = row_origins(mesh, rows) if fields and mesh is not None else None
        (trajectory, images, spatial), context, origins = collective_host(
            (generated, expanded, origins), phase="flow rollout")
        # The reward reads the host fields beside the samples' context, each source row's repeated into its
        # group as the context's are; the training rows read neither.
        scored = context
        if fields:
            held = (host if origins is None or mesh is None
                    else shared_host_fields(mesh, host, np.asarray(origins), rows))
            grouped = jax.tree.map(lambda column: self._grouped(np.asarray(column), np), held,
                                   is_leaf=lambda column: not isinstance(column, Mapping))
            scored = with_host(context, grouped)

        def score() -> np.ndarray | None:
            if jax.process_index() != 0:
                return None
            rewards = np.asarray(self.reward(np.asarray(images), scored), np.float64)
            if rewards.shape != (count * self.groups,) or not np.isfinite(rewards).all():
                raise ValueError("a flow reward must return one finite scalar per generated sample")
            return rewards

        rewards = agreed("flow rollout reward", score)
        rewards = np.asarray(broadcast_from_process_zero(
            None if rewards is None else rewards.tolist()), np.float64)
        return agreed("flow rollout batching",
                      lambda: self._transitions(trajectory, context, spatial, rewards, owned))


__all__ = ["FlowGRPOObjective", "FlowReward", "FlowRollout"]

