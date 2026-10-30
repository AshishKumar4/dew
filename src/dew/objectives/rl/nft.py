"""DiffusionNFT: online diffusion reinforcement on the forward process (arXiv 2509.16117).

The policy is trained by flow matching on clean samples its old self drew, each sample's target
split between an implicit positive and negative policy by an optimality probability its reward sets
(the paper's Eq. 5): with v+ = (1 - beta) v_old + beta v_theta and v- = (1 + beta) v_old - beta v_theta,
a sample of optimality r trains r on v+ and 1 - r on v-. The sampler is any solver over the old
policy, no likelihood is computed and no classifier-free guidance is used.

The loss is the released trainer's form (NVlabs/DiffusionNFT at cbb14f84, scripts/train_nft_sd3.py
:862-970), which the paper's Eq. 5 is up to a weighting: each implicit policy's clean-sample estimate,
x0 = x_t - t v, against the sample, its squared error over the sample's coordinates divided by the
detached mean absolute error (floored at 1e-5), the two mixed by r, divided by beta and scaled by the
advantage clip; plus `kl` (the release's train.beta, 1e-4) times the mean squared difference of the
live velocity from the base model's (:962-967). An advantage is the reward less its prompt group's
mean over the global standard deviation of the batch's rewards plus 1e-4 (flow_grpo/stat_tracking.py
:14-30, `global_std`); r is
that advantage clipped to the clip and mapped to [0, 1] (train_nft_sd3.py:910-929). The old policy
is the EMA slot, which follows the trained weights at `old_decay` once an update
(train_nft_sd3.py:1033-1038, the release's decay_type 1), and samples every rollout (:640-660); the
base model is the starting tree's weights, frozen under `TEACHER`. The release also keeps an EMA of
the trained weights at 0.9 for its evaluations and saved checkpoints (:565-567, :220, :342); Dew's one
EMA slot holds the old policy, so evaluation, previews and the published task use the live weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew.batch import Batch, Variables, dicts, merge
from dew.diffusion.presets import Preset
from dew.diffusion.process import Process
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.diffusion.schedules.source import SourceGrid
from dew.diffusion.transforms import FlowMatchPredictionTransform
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.nn.precision import at_least_fp32
from dew.objectives.base import OMITTED, TEACHER, Aux, Omitted, Ratio, Shown, Step
from dew.objectives.diffusion.objective import DiffusionObjective, teacher_weights
from dew.sampling.solvers import Solver
from dew.sampling.solvers.dpm import DPMSolverMultistep
from dew.sampling.walks import walk_and_decode

from .flow import REWARDS_KEY, GroupedRollout, _source
from .sessions import ADVANTAGES_KEY

if TYPE_CHECKING:
    from dew.state import TrainState


def released_old_decay(updates: jax.typing.ArrayLike) -> jax.Array:
    """The released trainer's old-policy decay at the update after `updates` completed ones,
    min(0.001 s, 0.5) with s the gradient steps taken once it is made (train_nft_sd3.py:169-189 and
    1033-1038, decay_type 1, one gradient step an epoch): 0.001 at the first update, a half from the
    500th."""
    updates = jnp.asarray(updates)
    return jnp.minimum(0.001 * (updates.astype(at_least_fp32(updates.dtype)) + 1), 0.5)


_WEIGHT_FLOOR = 1e-5
"""The floor of a sample's detached mean absolute error, the loss's divisor (train_nft_sd3.py:942-953)."""


def released_policy_loss(samples: jax.Array, live: jax.Array, old: jax.Array, optimality: jax.Array, *,
                         beta: float, scale: float) -> jax.Array:
    """Each sample's policy loss `[N]` at one time, from its clean value `samples`, the live and old
    policies' clean-sample estimates there (`live`, `old`, x_t - t v) and its optimality r: the
    implicit positive and negative policies' estimates, beta live + (1 - beta) old and
    (1 + beta) old - beta live, each against the sample, its mean squared error over the sample's
    coordinates divided by its detached mean absolute error floored at 1e-5, mixed r to 1 - r,
    divided by beta and times `scale`, the advantage clip (train_nft_sd3.py:934-960)."""
    rows = samples.shape[0]

    def coordinate_mean(value):
        return jnp.mean(value.reshape(rows, -1), axis=1)

    def matched(estimate):
        error = estimate - samples
        weight = jax.lax.stop_gradient(jnp.maximum(coordinate_mean(jnp.abs(error)), _WEIGHT_FLOOR))
        return coordinate_mean(jnp.square(error)) / weight

    old = jax.lax.stop_gradient(old)
    positive = beta * live + (1 - beta) * old
    negative = (1 + beta) * old - beta * live
    return (optimality * matched(positive) + (1 - optimality) * matched(negative)) / beta * scale


class DiffusionNFTObjective(DiffusionObjective):
    """Trains a rectified-flow velocity model by DiffusionNFT (the module docstring).

    A batch holds `latents`, the clean samples the old policy drew, `[N, ...]`; `timesteps`, the
    times each trains at, `[N, T]` (the sampler's grid); and `advantages`, `[N]`. `beta` is the
    implicit policies' strength, `adv_clip_max` the advantage clip and the loss's scale, `kl` the
    weight of the velocity's squared difference from the base model's, which a positive weight reads
    from the starting tree (`variables`, a pretrained model's), and `old_decay` the old policy's decay
    over updates. The release's single-reward defaults: beta 1, a clip of 5, kl 1e-4. The
    old policy is the EMA slot; evaluation, previews and the published task use the live weights.
    `solver` and `steps` configure evaluation; a rollout samples over its own grid (`NFTRollout`).
    """

    shown: Mapping[str, Shown] = {"loss": Shown(), "reward": Shown(better="higher")}
    # The EMA slot holds the old policy, the sampler's and v_old's, not an average to publish.
    ema_is_reference = True

    def __init__(self, model: nn.Module, process: Process | Preset, inputs: InputSpec, *,
                 beta: float = 1.0, adv_clip_max: float = 5.0, kl: float = 1e-4,
                 old_decay: optax.Schedule = released_old_decay, autoencoder: AutoEncoder | None = None,
                 solver: Solver | None = None, steps: int = 41, variables: Variables | None = None,
                 grid: SourceGrid | None | Omitted = OMITTED):
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        if not math.isfinite(adv_clip_max) or adv_clip_max <= 0:
            raise ValueError("adv_clip_max must be finite and positive")
        if not math.isfinite(kl) or kl < 0:
            raise ValueError("kl must be finite and non-negative")
        if variables is not None and "params" not in variables:
            raise ValueError("variables must be a variables tree with a params collection")
        if kl > 0 and variables is None:
            raise ValueError("a positive kl regularizes toward the base model, the starting tree's "
                             "weights: pass variables")
        super().__init__(model, process, inputs, autoencoder=autoencoder, unconditional_prob=0,
                         ema_decay=old_decay, solver=DPMSolverMultistep() if solver is None else solver,
                         guidance=None, steps=steps, variables=variables, grid=grid)
        if not (isinstance(self.process.schedule, FlowMatchingScheduler) and not self.process.interval
                and isinstance(self.process.prediction, FlowMatchPredictionTransform)):
            raise ValueError("DiffusionNFT trains a velocity model on the linear path; build the process "
                             "with presets.Flow()")
        self.beta, self.adv_clip_max, self.kl = beta, adv_clip_max, kl
        self.base_variables = dicts(teacher_weights(self.variables, None)) if kl > 0 else None

    @property
    def validation_loss_refusal(self) -> str | None:
        return ("DiffusionNFT has no source-batch validation loss; rank checkpoints by a reward metric "
                "with an explicit Best, or pass no validation data")

    def held_variables(self, *, beside: Variables | None = None) -> Variables:
        held = super().held_variables(beside=beside)
        return held if self.base_variables is None else {**held, TEACHER: self.base_variables}

    def optimality(self, advantages: jax.Array) -> jax.Array:
        """Each sample's optimality probability r: its advantage clipped to `adv_clip_max` and mapped
        linearly onto [0, 1], a zero advantage at one half (train_nft_sd3.py:910-929)."""
        clipped = jnp.clip(advantages, -self.adv_clip_max, self.adv_clip_max)
        return jnp.clip(clipped / self.adv_clip_max / 2 + 0.5, 0, 1)

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        dtype = at_least_fp32(jnp.result_type(batch["latents"]))
        samples = jax.lax.stop_gradient(jnp.asarray(batch["latents"], dtype))
        times = jax.lax.stop_gradient(jnp.asarray(batch["timesteps"], dtype))
        advantages = jax.lax.stop_gradient(jnp.asarray(batch[ADVANTAGES_KEY], jnp.float32))
        if samples.ndim != len(self.latent_shape) + 1 or samples.shape[1:] != self.latent_shape:
            raise ValueError(f"latents must hold [batch, {self.latent_shape}]")
        if times.ndim != 2 or times.shape[0] != samples.shape[0] or advantages.shape != samples.shape[:1]:
            raise ValueError("timesteps must give every sample its times and advantages its one value")
        if step.ema is None:
            raise ValueError("DiffusionNFT's old policy is its EMA slot, which this step does not carry")
        rows, trained = times.shape
        r = self.optimality(advantages)
        given = self.encoded_conditions(variables, batch)
        live = self.denoiser(variables, given, None)
        old = self.denoiser(merge(variables, jax.tree.map(jax.lax.stop_gradient, step.ema)), given, None)
        base = (None if self.kl == 0 else
                self.denoiser({**variables, **jax.tree.map(jax.lax.stop_gradient, variables[TEACHER])},
                              given, None))
        noise = step.sample("noise", jax.random.normal, (trained, *self.latent_shape), dtype=jnp.float32,
                            rows=rows).astype(dtype)

        def at(_, values):
            t, eps = values
            # The training process's own forward map at t, a shifted schedule's noise level among them.
            alpha, sigma = self.process.rates(t, like=samples)
            x_t = alpha * samples + sigma * eps
            x0_live, eps_live = live(x_t, t)
            # Each implicit policy's clean-sample estimate, x_t - t v, is the same mix of the two
            # policies' estimates as its velocity is of theirs (train_nft_sd3.py:934-948).
            policy = released_policy_loss(samples, x0_live, old(x_t, t)[0], r, beta=self.beta,
                                          scale=self.adv_clip_max)
            if base is None:
                return None, (policy, jnp.zeros_like(policy))
            x0_base, eps_base = jax.lax.stop_gradient(base(x_t, t))
            drift = jnp.square((eps_live - x0_live) - (eps_base - x0_base))
            drift = jnp.mean(drift.reshape(rows, -1), axis=1)
            return None, (policy, drift)

        _, (policy, drift) = jax.lax.scan(at, None, (times.T, jnp.swapaxes(noise, 0, 1)))
        per_row = jnp.mean(policy, axis=0) + self.kl * jnp.mean(drift, axis=0)
        metrics = {"policy": jnp.mean(policy), "optimality": jnp.mean(r)}
        if self.kl > 0:
            metrics["velocity_kl"] = jnp.mean(drift)
        if REWARDS_KEY in batch:
            metrics["reward"] = jnp.asarray(batch[REWARDS_KEY], jnp.float32).mean()
        return Ratio(per_row.sum(), jnp.asarray(rows, per_row.dtype)), Aux(metrics)

    def _rows(self, batch: Batch) -> int:
        """One sample per source row, which a prompt-only batch has too."""
        return _source(self.inputs, batch).shape[0]


@dataclass(frozen=True)
class NFTRollout(GroupedRollout):
    """Draws `groups` clean samples of each source row with the old policy, the objective's solver
    over `steps` time points and no guidance (the release's deterministic second-order DPM-Solver at
    guidance 1), and trains each at the first `train_steps` times of that grid (the release's
    int(0.99 num_steps), 9 of its 10 steps' times, at 11 time points).

    A sample's advantage is its reward less its group's mean, over the population standard deviation
    of every reward in the batch plus 1e-4 (the module docstring); the rewards stay float64 on the
    host and the advantages are float32.
    """

    objective: DiffusionNFTObjective
    train_steps: int = 9

    def __post_init__(self) -> None:
        if self.groups < 2:
            raise ValueError("a reward group needs at least two samples")
        if not 1 <= self.train_steps < self.steps:
            raise ValueError("train_steps must select between one and steps-1 times of the sampler's grid")
        reserved = {"latents", "timesteps", ADVANTAGES_KEY, REWARDS_KEY}
        if any(condition.field in reserved for condition in self.objective.inputs.conditions.values()):
            raise ValueError("conditioning fields must not overwrite DiffusionNFT's sample fields")

    def _policy(self, state: TrainState) -> Variables:
        return state.averaged

    def _generate_impl(self, params: Variables, batch: Batch, key: jax.Array):
        """One clean sample per row, unguided, over the rollout's grid, and its decoded image."""
        objective = self.objective
        # The training process's grid, in the times its denoiser and forward map read, never a loaded
        # pipeline's table coordinates.
        process = objective.process
        with jax.ensure_compile_time_eval():
            times = process.times(self.steps)
        noise_key, sample_key = jax.random.split(key)
        count = _source(objective.inputs, batch).shape[0]
        x_T = process.noise(noise_key, (count, *objective.latent_shape))
        denoise = objective.denoiser(params, objective.encoded_conditions(params, batch), None, process)
        autoencoder, published = objective.published_autoencoder(params)
        images, latents = walk_and_decode(denoise, autoencoder, published, x_T, solver=objective.solver,
                                          guidance=None, key=sample_key, times=np.asarray(times),
                                          final_denoise=objective.final_denoise)
        return (latents, jnp.asarray(times)), images, {}

    def _training_rows(self, generation, context: Batch, spatial: Mapping[str, np.ndarray],
                       rewards: np.ndarray, owned: slice | np.ndarray) -> Batch:
        """This process's samples, their training times and their advantages."""
        latents, times = generation
        grouped = rewards.reshape(-1, self.groups)
        centred = grouped - grouped.mean(axis=1, keepdims=True)
        advantages = (centred / (rewards.std() + 1e-4)).reshape(-1).astype(np.float32)
        if not np.isfinite(advantages).all():
            raise ValueError("DiffusionNFT's advantages must remain finite")
        local = np.asarray(latents)[owned]
        prepared = {condition.field: jax.tree.map(lambda leaf: np.asarray(leaf)[owned],
                                                 context[condition.field])
                    for condition in self.objective.inputs.conditions.values()}
        prepared.update({
            "latents": local,
            "timesteps": np.broadcast_to(np.asarray(times)[:self.train_steps],
                                         (local.shape[0], self.train_steps)),
            ADVANTAGES_KEY: advantages[owned],
            REWARDS_KEY: rewards[owned],
        })
        return prepared


__all__ = ["DiffusionNFTObjective", "NFTRollout", "released_old_decay", "released_policy_loss"]
