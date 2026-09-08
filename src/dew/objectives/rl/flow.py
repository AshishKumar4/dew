"""Flow-GRPO over recorded rectified-flow transitions.

The policy ratio follows the released implementation's mean coordinate log
likelihood, not the product of all coordinate likelihood ratios. The KL term
is the conditional Gaussian KL from arXiv:2505.05470v5 section 4, also averaged
across coordinates. Its variance includes elapsed time. The released
scripts/train_sd3.py at 879042cf5707f8b90daa98d147d7deac2317c5da instead divides
squared mean displacement by the diffusion coefficient squared, a term equal
to elapsed time times this conditional KL. These regularizers have different
step weighting. Callback scores are retained in float64 through host grouping,
unlike the released trainer's earlier float32 score conversion.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


from dew.artifacts import agree_process_phase, broadcast_from_process_zero, collective_host

from dew.diffusion.process import Process
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.objectives.base import Aux, Batch, Mean, Step, Variables
from dew.objectives.diffusion.objective import DiffusionObjective, VALIDATION_SAMPLES
from dew.registry import objectives

from dew.sampling.flow import FlowSDE, FlowTrajectory, GaussianTransition, sample_trajectory
from dew.sampling.guidance import CFG
from dew.sampling.solvers import Euler, Solver

from .rollout import ADVANTAGES_KEY, OLD_LOG_PROBS_KEY, REWARDS_KEY

if TYPE_CHECKING:
    from dew.training.state import TrainState

Predictor = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]



def _source(inputs: InputSpec, batch: Batch) -> jax.Array:
    """The real row-bearing field used by rollout, scoring, and preview."""
    conditions = inputs.conditions
    name = next(iter(conditions.values())).field if conditions else inputs.sample.key
    source = jnp.asarray(jax.tree.leaves(batch[name])[0])
    if source.ndim == 0 or source.shape[0] == 0:
        raise ValueError("flow sampling needs a non-empty leading batch dimension")
    return source



@objectives("flow_grpo")
class FlowGRPOObjective(DiffusionObjective):
    """Clipped, coordinate-normalized policy gradients with conditional transition KL.

    Batches carry latents/next_latents [N, K, ...], timesteps/next_timesteps,
    joint old_log_probs and transition_mask [N, K], and advantages [N] or
    [N, K]. K is the selected transition count. The denominator counts kept
    stochastic transitions. Deterministic intervals contribute no policy loss.

    beta > 0 freezes the initial denoiser in the existing EMA slot. Evaluation
    and previews always use the live policy. sampler and steps configure
    evaluation; sde specifies both rollout and rescoring. pretrained is a
    model variables dict, as returned by model.init; encoders and an optional
    autoencoder are supplied through the existing diffusion input contract.
    """

    def __init__(self, model: nn.Module, process: Process, inputs: InputSpec, *,
                 sde: FlowSDE = FlowSDE(), beta: float = 0.0,
                 clip_range: float = 1e-4, adv_clip_max: float = 5.0,
                 autoencoder: AutoEncoder | None = None,
                 guidance: CFG | None = CFG(3.0), sampler: Solver[object] = Euler(),
                 steps: int = 41, pretrained: Variables | None = None):
        sde.validate(process)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if not math.isfinite(clip_range) or not 0 <= clip_range < 1:
            raise ValueError("clip_range must be finite and in [0, 1)")
        if not math.isfinite(adv_clip_max) or adv_clip_max <= 0:
            raise ValueError("adv_clip_max must be finite and positive")
        if steps < 2:
            raise ValueError("evaluation needs at least two time points")
        if pretrained is not None and "params" not in pretrained:
            raise ValueError("pretrained must be a model variables dict with a params collection")
        super().__init__(model, process, inputs, autoencoder=autoencoder,
                         unconditional_prob=0, ema_decay=1.0, sampler=sampler,
                         guidance=guidance, steps=steps)
        if beta == 0:
            self.ema = None
        self.sde = sde
        self.beta = beta
        self.clip_range = clip_range
        self.adv_clip_max = adv_clip_max
        self.pretrained = pretrained

    def held_variables(self) -> Variables:
        held: dict[str, Any] = dict(super().held_variables())
        if self.pretrained is not None:
            held["pretrained"] = self.pretrained
        return held

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        held = self.held_variables() if variables is None else variables
        if "pretrained" not in held:
            return super().init(key, held)
        state: dict[str, Any] = {**held["pretrained"], "encoders": held["encoders"]}
        if "autoencoder" in held:
            state["autoencoder"] = held["autoencoder"]
        return state

    def _predictor(self, params: Variables, batch: Batch) -> Predictor:
        tokens = {keyword: batch[condition.field]
                  for keyword, condition in self.inputs.conditions.items()}
        given = self.encode(params["encoders"], tokens)
        denoise = self.process.denoiser(
            self.model, self.trainable(params), given,
            None if self.guidance is None else self.unconditional)
        return denoise if self.guidance is None else self.guidance(denoise)

    def _transition(self, predict: Predictor, x: jax.Array,
                    t: jax.Array, following: jax.Array) -> GaussianTransition:
        denoised, eps = predict(x, t)
        return self.sde.transition(x, t, following, denoised, eps, self.process)

    def _window(self, batch: Batch) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        latents = jnp.asarray(batch["latents"], jnp.float32)
        following = jnp.asarray(batch["next_latents"], jnp.float32)
        times = jnp.asarray(batch["timesteps"], jnp.float32)
        next_times = jnp.asarray(batch["next_timesteps"], jnp.float32)
        if latents.ndim != len(self.latent_shape) + 2 or latents.shape[2:] != self.latent_shape:
            raise ValueError(f"latents must hold [batch, transitions, {self.latent_shape}]")
        if following.shape != latents.shape:
            raise ValueError("latents and next_latents must share one shape")
        if times.shape != latents.shape[:2] or next_times.shape != times.shape:
            raise ValueError("timesteps and next_timesteps must mark every recorded transition")
        if not times.shape[0] or not times.shape[1]:
            raise ValueError("a recorded batch needs rows and transition slots")
        return jax.tree.map(jax.lax.stop_gradient, (latents, following, times, next_times))

    def log_probs(self, params: Variables, batch: Batch) -> jax.Array:
        """Rescore joint transition log densities with the rollout's guidance."""
        latents, following, times, next_times = self._window(batch)
        predict = self._predictor(params, batch)

        def score(_, values):
            x, action, t, s = values
            return None, self._transition(predict, x, t, s).log_prob(action)

        _, values = jax.lax.scan(score, None, tuple(
            jnp.swapaxes(value, 0, 1) for value in (latents, following, times, next_times)))
        return values.T

    def loss(self, params: Variables, batch: Batch, step: Step) -> tuple[Mean, Aux]:
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
        predict = self._predictor(params, batch)
        reference = None
        if self.beta > 0:
            if step.ema is None:
                raise ValueError("FlowGRPO's conditional KL needs its frozen reference in step.ema")
            reference = self._predictor(jax.tree.map(jax.lax.stop_gradient, step.ema), batch)
        dimensions = math.prod(self.latent_shape)

        def terms(_, values):
            x, action, t, s, old_log_prob, advantage, selected = values
            transition = self._transition(predict, x, t, s)
            # NaN variance remains selected and fails the numerical check. Only
            # a zero-variance interval is outside the Gaussian policy's support.
            keep = selected & (transition.variance != 0)
            current = jnp.where(keep, transition.log_prob(action), 0) / dimensions
            previous = jnp.where(keep, old_log_prob, 0) / dimensions
            ratio = jnp.exp(current - previous)
            unclipped = -advantage * ratio
            clipped = -advantage * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
            pg = jnp.where(keep, jnp.maximum(unclipped, clipped), 0).sum()
            kl = jnp.asarray(0.0)
            if reference is not None:
                reference_mean = self._transition(reference, x, t, s).mean
                kl = jnp.where(keep, transition.kl(reference_mean) / dimensions, 0).sum()
            clipped_count = (keep & (jnp.abs(ratio - 1) > self.clip_range)).sum()
            return None, (pg, kl, keep.astype(jnp.float32).sum(), clipped_count)

        _, summed = jax.lax.scan(terms, None, tuple(jnp.swapaxes(value, 0, 1) for value in (
            latents, following, times, next_times, old, advantages, mask)))
        pg, kl, mass, clipped_count = (value.sum() for value in summed)
        denominator = jnp.where(mass > 0, mass, 1)
        metrics = {"pg": pg / denominator, "actor/clipfrac": clipped_count / denominator}
        if self.beta > 0:
            metrics["transition_kl"] = kl / denominator
        if REWARDS_KEY in batch:
            metrics["reward"] = jnp.asarray(batch[REWARDS_KEY], jnp.float32).mean()
        return Mean(pg + self.beta * kl, mass), Aux(metrics)

    def _draw(self, params: Variables, batch: Batch, key: jax.Array,
              limit: int | None = None) -> tuple[jax.Array, Batch]:
        error = None
        prepared = None
        try:
            count = _source(self.inputs, batch).shape[0]
            sample_batch = self._sampling_batch(batch)
            if limit is not None:
                count = min(limit, count)
                sample_batch = jax.tree.map(lambda value: value[:count], sample_batch)
            prepared = (count, sample_batch)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="flow sample setup")
        assert prepared is not None
        count, sample_batch = prepared
        error = None
        samples = None
        try:
            samples = self._sample(params, sample_batch, key, count=count)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="flow sample generation")
        assert samples is not None
        return samples, {keyword: sample_batch[condition.field]
                         for keyword, condition in self.inputs.conditions.items()}

    def evaluate(self, params: Variables, batch: Batch, step: Step):
        """Generate one live-policy sample per source row, including prompt-only batches."""
        samples, _ = self._draw(params, batch, step.key)
        assert self.artifact is not None
        return self.artifact(samples)

    def preview(self, params: Variables, batch: Batch, step: Step, *, scored=None):
        """Draw on all ranks; materialize before root-only caption decoding."""
        samples, tokens = self._draw(params, batch, step.key, VALIDATION_SAMPLES)
        samples, tokens = collective_host((samples, tokens), phase="flow preview")
        if jax.process_index() != 0:
            return None
        captions = ()
        for keyword, condition in self.inputs.conditions.items():
            captions = condition.encoder.captions(tokens[keyword])
            if captions:
                break
        assert self.artifact is not None
        return self.artifact(samples, captions)




FlowReward: TypeAlias = Callable[[np.ndarray, Batch], np.ndarray | jax.Array | Sequence[float]]
"""Score decoded [-1, 1] samples and repeated source rows, one scalar per sample."""


@dataclass(frozen=True)
class FlowRollout:
    """Collect complete prompt groups and the first train_steps transitions.

    steps counts time points, so steps=11 draws ten transitions. None selects
    all transitions. Rewards use population group standard deviation and
    epsilon 1e-4, as Flow-GRPO's PerPromptStatTracker does. Each prompt row
    defines a group; equal prompt text in other rows does not merge groups.
    Zero-advantage rows are masked, as the reference training loop filters them.
    Callback collection, JSON/byte transport, and population statistics retain
    float64 values. Host rewards remain float64; training advantages are
    float32 after normalization. The reward metric is a float32 diagnostic,
    and JAX device transfer also narrows the reward column when x64 is off.

    The trainer supplies global arrays on every process. Generation remains
    collective, rewards run once on rank zero, and the result contains only
    this process's owned rows for the trainer's shard_batch boundary. Host
    materialization currently uses collective_host and replicates the complete
    trajectory on every process before selecting local rows.
    """

    objective: FlowGRPOObjective
    reward: FlowReward
    groups: int = 4
    steps: int = 11
    train_steps: int | None = None
    _generate: Callable[[Variables, Batch, jax.Array], tuple[FlowTrajectory, jax.Array]] = field(
        init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.groups < 2:
            raise ValueError("a flow reward group needs at least two samples")
        if self.steps < 2:
            raise ValueError("a flow rollout needs at least two time points")
        if self.train_steps is not None and not 1 <= self.train_steps < self.steps:
            raise ValueError("train_steps must select between one and steps-1 transitions")
        if self.objective.sde.noise_level == 0:
            raise ValueError("online FlowGRPO needs positive noise for stochastic transitions")
        reserved = {"latents", "next_latents", "timesteps", "next_timesteps",
                    OLD_LOG_PROBS_KEY, ADVANTAGES_KEY, REWARDS_KEY, "transition_mask"}
        if any(condition.field in reserved for condition in self.objective.inputs.conditions.values()):
            raise ValueError("conditioning fields must not overwrite FlowGRPO transition fields")
        object.__setattr__(self, "_generate", jax.jit(self._generate_impl))



    def _generate_impl(self, params: Variables, batch: Batch,
                       key: jax.Array) -> tuple[FlowTrajectory, jax.Array]:
        objective = self.objective
        tokens = {keyword: batch[condition.field]
                  for keyword, condition in objective.inputs.conditions.items()}
        given = objective.encode(params["encoders"], tokens)
        denoise = objective.process.denoiser(
            objective.model, objective.trainable(params), given,
            None if objective.guidance is None else objective.unconditional)
        noise_key, sample_key = jax.random.split(key)
        count = _source(objective.inputs, batch).shape[0]
        initial = objective.process.noise(noise_key, (count, *objective.latent_shape))
        trajectory = sample_trajectory(denoise, initial, self.steps, solver=objective.sde,
                                       guidance=objective.guidance, key=sample_key)
        samples = trajectory.samples
        if objective.autoencoder is not None:
            samples = objective.autoencoder.decode(params["autoencoder"], samples)
        return trajectory, jnp.clip(samples, -1, 1)

    def _owned_rows(self, source: jax.Array) -> slice | np.ndarray:
        if jax.process_count() == 1:
            return slice(None)
        owned: set[int] = set()
        for index in source.sharding.addressable_devices_indices_map(source.shape).values():
            if index is None:
                raise ValueError("source sharding has no addressable row index")
            owned.update(range(*index[0].indices(source.shape[0])))
        rows = np.asarray(sorted(owned), np.int64)
        return (rows[:, None] * self.groups + np.arange(self.groups)).reshape(-1)

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch:
        error = None
        expanded = None
        owned = slice(None)
        count = 0
        try:
            source = _source(self.objective.inputs, batch)
            count = source.shape[0]
            owned = self._owned_rows(source)

            def repeat(leaf):
                value = jnp.asarray(leaf)
                if value.ndim == 0:
                    return value
                if value.shape[0] != count:
                    raise ValueError("all flow source rows must share one batch dimension")
                return jnp.repeat(value, self.groups, axis=0)

            expanded = jax.tree.map(repeat, batch)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="flow rollout setup")
        assert expanded is not None
        error = None
        generated = None
        try:
            generated = self._generate(state.params, expanded, key)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="flow rollout generation")
        assert generated is not None
        (trajectory, images), context = collective_host(
            (generated, expanded), phase="flow rollout")
        error = None
        rewards = None
        if jax.process_index() == 0:
            try:
                rewards = np.asarray(self.reward(np.asarray(images), context), np.float64)
                if rewards.shape != (count * self.groups,) or not np.isfinite(rewards).all():
                    raise ValueError("a flow reward must return one finite scalar per generated sample")
            except BaseException as failure:
                error = failure
        agree_process_phase(error, phase="flow rollout reward")
        rewards = np.asarray(broadcast_from_process_zero(
            None if rewards is None else rewards.tolist()), np.float64)
        error = None
        result = None
        try:
            grouped = rewards.reshape(-1, self.groups)
            centered = grouped - grouped.mean(axis=1, keepdims=True)
            # The author code uses float64 population statistics. The shared
            # JAX group estimator fixes float32 and ddof=1, which changes this loss.
            deviation = grouped.std(axis=1, keepdims=True)
            advantages = (centered / (deviation + 1e-4)).astype(np.float32).reshape(-1)
            if not np.isfinite(advantages).all():
                raise ValueError("flow group advantages must remain finite")
            selected = self.steps - 1 if self.train_steps is None else self.train_steps
            local_advantages = advantages[owned]
            shape = (local_advantages.shape[0], selected)
            result = {condition.field: jax.tree.map(lambda leaf: np.asarray(leaf)[owned],
                                                   context[condition.field])
                      for condition in self.objective.inputs.conditions.values()}
            result.update({
                "latents": np.asarray(trajectory.states)[owned, :selected],
                "next_latents": np.asarray(trajectory.states)[owned, 1:selected + 1],
                "timesteps": np.broadcast_to(trajectory.times[:selected], shape),
                "next_timesteps": np.broadcast_to(trajectory.times[1:selected + 1], shape),
                OLD_LOG_PROBS_KEY: np.asarray(trajectory.log_probs)[owned, :selected],
                "transition_mask": (np.asarray(trajectory.stochastic)[owned, :selected]
                                    & (local_advantages[:, None] != 0)),
                ADVANTAGES_KEY: local_advantages,
                REWARDS_KEY: rewards[owned],
            })
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="flow rollout batching")
        assert result is not None
        return result


__all__ = ["FlowGRPOObjective", "FlowReward", "FlowRollout"]

