"""REPA-E: the autoencoder trained with the diffusion model through REPA.

Leng et al. 2025, "REPA-E: Unlocking VAE for End-to-End Tuning with Latent
Diffusion Transformers"; the official code is End2End-Diffusion/REPA-E's
`train_repae.py`, `models/sit.py` and `loss/losses.py`. One step there
updates the VAE on its own regularizer plus `vae_align_proj_coeff` times the
REPA loss of its latent, with the diffusion model frozen and its batch norm
reading running statistics; then the diffusion model on the denoising and
REPA losses of the detached latent, its batch norm normalizing with the
batch's statistics and updating the running ones. The two parameter sets
are disjoint, so one loss that stops each gradient where the reference
freezes it is both updates at once, on the same latent, times and noise.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dew.objectives.base import Variables

if TYPE_CHECKING:
    from dew.nn.autoencoders import AutoEncoder

AUTOENCODER = "autoencoder"
"""Where the trained autoencoder's parameters sit in `params`."""

LATENT_STATS = "latent_statistics"
"""The collection of the latent batch norm's running mean and variance."""


@dataclass(frozen=True)
class EndToEnd:
    """REPA-E's end-to-end tuning of the autoencoder.

    `align_weight` is `vae_align_proj_coeff`, the REPA loss's weight in the
    autoencoder's update. The autoencoder's own regularizer is its L1
    reconstruction at `reconstruction_weight` plus its posterior's KL at
    `kl_weight`, summed over each latent and averaged over the batch, as
    REPA-E's `ReconstructionLoss_Single_Stage` weighs them; the reference's
    LPIPS and PatchGAN terms are not part of it. The latents are normalized
    per channel by an affine-free batch norm (`momentum`, `epsilon`), which
    replaces the autoencoder's fixed latent scale and starts from it.
    """

    align_weight: float = 1.5
    reconstruction_weight: float = 1.0
    kl_weight: float = 1e-6
    momentum: float = 0.1
    epsilon: float = 1e-4

    def initial_statistics(self, shift: ArrayLike, scale: ArrayLike, channels: int) -> Variables:
        """The running statistics REPA-E's `init_bn` starts from: the
        autoencoder's latent shift as the mean and 1 / scale^2 as the
        variance, per channel."""
        scale = jnp.asarray(scale, jnp.float32)
        return {"mean": jnp.broadcast_to(jnp.asarray(shift, jnp.float32), (channels,)),
                "var": jnp.broadcast_to(1.0 / scale ** 2, (channels,))}

    def tuned(self, autoencoder: AutoEncoder, variables: Variables) -> tuple[AutoEncoder, Variables]:
        """The autoencoder a task over a tuned run's `variables` decodes with,
        and the variables with its weights under `autoencoder`: the trained
        weights at `params/autoencoder`, its latents shifted by the running
        mean and scaled by the running variance's reciprocal square root,
        without the batch norm's epsilon, as REPA-E's
        `SiT.extract_latents_stats` gives the scale its sampling
        denormalizes with."""
        tuned = copy.copy(autoencoder)
        statistics = variables[LATENT_STATS]
        tuned.latent_shift = statistics["mean"]
        tuned.latent_scale = jax.lax.rsqrt(statistics["var"])
        tuned.params = variables["params"][AUTOENCODER]
        return tuned, {**variables, "autoencoder": tuned.params}

    def normalized(self, latents: jax.Array, statistics: Variables) -> jax.Array:
        """`latents` under the running statistics: the batch norm in eval mode."""
        return (latents - statistics["mean"]) / jnp.sqrt(statistics["var"] + self.epsilon)

    def batch_normalized(self, latents: jax.Array, statistics: Variables) -> tuple[jax.Array, Variables]:
        """`latents` under their own statistics over every axis but the
        channels, and the running statistics after them: torch's
        `BatchNorm2d` in training mode, whose running variance is unbiased."""
        axes = tuple(range(latents.ndim - 1))
        mean = jnp.mean(latents, axis=axes)
        var = jnp.var(latents, axis=axes)
        count = latents.size // latents.shape[-1]
        following = {"mean": (1 - self.momentum) * statistics["mean"] + self.momentum * mean,
                     "var": (1 - self.momentum) * statistics["var"]
                     + self.momentum * var * count / (count - 1)}
        return (latents - mean) / jnp.sqrt(var + self.epsilon), following

    def regularizer(self, images: jax.Array, reconstruction: jax.Array,
                    moments: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """The autoencoder's own loss and its two terms."""
        mean, log_variance = jnp.split(moments.astype(jnp.float32), 2, axis=-1)
        log_variance = jnp.clip(log_variance, -30.0, 20.0)
        axes = tuple(range(1, mean.ndim))
        kl = jnp.mean(0.5 * jnp.sum(jnp.square(mean) + jnp.exp(log_variance) - 1.0 - log_variance, axis=axes))
        reconstruction_error = jnp.mean(jnp.abs(images - reconstruction.astype(jnp.float32)))
        total = self.reconstruction_weight * reconstruction_error + self.kl_weight * kl
        return total, {"reconstruction": reconstruction_error, "kl": kl}


__all__ = ["AUTOENCODER", "LATENT_STATS", "EndToEnd"]
