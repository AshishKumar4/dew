"""REPA-E: the autoencoder trained together with the diffusion model through REPA.

REPA-E is from Leng et al. 2025, "REPA-E: Unlocking VAE for End-to-End Tuning
with Latent Diffusion Transformers"; the official code is
End2End-Diffusion/REPA-E's `train_repae.py`, `models/sit.py` and
`loss/losses.py`. One step there makes three updates:

1. The VAE, on its own loss (L1 reconstruction, LPIPS, KL and the PatchGAN's
   generator term) plus `vae_align_proj_coeff` times the REPA loss of its
   latent. The diffusion model is frozen, and its batch norm reads the
   running statistics.
2. The discriminator, on its hinge loss against the same reconstruction.
3. The diffusion model, on the denoising and REPA losses of the detached
   latent. Its batch norm normalizes with the batch's statistics and updates
   the running ones.

The three parameter sets are disjoint, and each update reads the others as
they were before the step. So one loss that stops each gradient where the
reference freezes it makes all three updates at once, on the same latent,
times and noise.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.typing import ArrayLike

from dew.objectives.base import Batch, Objective, Variables

if TYPE_CHECKING:
    from dew.nn.autoencoders import AutoEncoder

AUTOENCODER = "autoencoder"
"""Where the trained autoencoder's parameters sit in `params`."""

LATENT_STATS = "latent_statistics"
"""The collection of the latent batch norm's running mean and variance."""

PERCEPTUAL = "perceptual"
"""The collection that holds the frozen LPIPS network's weights."""


class _BatchNorm(nn.Module):
    """torch's `BatchNorm2d` in training mode: channels normalized by the
    batch's mean and biased variance (`epsilon` 1e-5), then scaled by
    `weight` (taming's `weights_init` draws it normal at 1, 0.02) and
    shifted by `bias`. REPA-E never puts its discriminator in evaluation
    mode, so no running statistics are kept. The statistics are `batch`'s
    rows' (`Objective.row_mean`)."""

    @nn.compact
    def __call__(self, hidden: jax.Array, batch: Batch) -> jax.Array:
        width = hidden.shape[-1]
        weight = self.param("weight", lambda key, shape: 1 + 0.02 * jax.random.normal(key, shape), (width,))
        bias = self.param("bias", nn.initializers.zeros, (width,))
        mean, var = _moments(hidden, batch)
        return (hidden - mean) / jnp.sqrt(var + 1e-5) * weight + bias


def _moments(values: jax.Array, batch: Batch) -> tuple[jax.Array, jax.Array]:
    """`values`' mean and biased variance over every axis but the channels,
    over `batch`'s rows (`Objective.row_mean`), as `jnp.mean` and `jnp.var` take them."""
    axes = tuple(range(values.ndim - 1))
    mean, _ = Objective.row_mean(values, batch, axes).mean()
    var, _ = Objective.row_mean(jnp.square(values - mean), batch, axes).mean()
    return mean, var


class PatchDiscriminator(nn.Module):
    """pix2pix's PatchGAN as taming-transformers' `NLayerDiscriminator`
    writes it, which REPA-E copies (loss/discriminator.py): a 4x4 stride-2
    convolution to `width` channels, then `layers - 1` more stride-2 ones and
    one stride-1 one, each doubling the width up to 8x, batch normalized,
    every activation a 0.2 leaky ReLU, and a final 4x4 convolution to one
    logit per patch; every convolution pads by one. A fresh one draws its
    kernels normal at 0.02 as taming's `weights_init` does; REPA-E's recipe
    starts from a pretrained one (`variables_from_torch`). Returns the
    logits, `[B, h, w, 1]`. Its batch norms read `batch`'s rows.
    """

    width: int = 64
    layers: int = 3

    @nn.compact
    def __call__(self, images: jax.Array, batch: Batch) -> jax.Array:
        def conv(features: int, stride: int, name: str, *, bias: bool = True) -> nn.Conv:
            return nn.Conv(features, (4, 4), strides=stride, padding=1, use_bias=bias, name=name,
                           kernel_init=nn.initializers.normal(0.02))

        hidden = nn.leaky_relu(conv(self.width, 2, "conv_0")(images), 0.2)
        for index in range(1, self.layers + 1):
            features = self.width * min(2 ** index, 8)
            hidden = conv(features, 2 if index < self.layers else 1, f"conv_{index}", bias=False)(hidden)
            hidden = nn.leaky_relu(_BatchNorm(name=f"norm_{index}")(hidden, batch), 0.2)
        return conv(1, 1, f"conv_{self.layers + 1}")(hidden)

    def variables_from_torch(self, state: Mapping[str, np.ndarray]) -> Variables:
        """The variables of `NLayerDiscriminator(ndf=width, n_layers=layers)`'s
        torch state dict, kernels moved from OIHW to HWIO. Its batch norms'
        running statistics are not read, as no step of REPA-E reads them."""
        params: dict[str, dict[str, jax.Array]] = {}

        def put(key: str, module: str, name: str) -> None:
            value = np.asarray(state[key])
            params.setdefault(module, {})[name] = jnp.asarray(
                np.transpose(value, (2, 3, 1, 0)) if value.ndim == 4 else value)

        put("main.0.weight", "conv_0", "kernel")
        put("main.0.bias", "conv_0", "bias")
        position = 2
        for index in range(1, self.layers + 1):
            put(f"main.{position}.weight", f"conv_{index}", "kernel")
            put(f"main.{position + 1}.weight", f"norm_{index}", "weight")
            put(f"main.{position + 1}.bias", f"norm_{index}", "bias")
            position += 3
        put(f"main.{position}.weight", f"conv_{self.layers + 1}", "kernel")
        put(f"main.{position}.bias", f"conv_{self.layers + 1}", "bias")
        return {"params": params}


def _applied(module: nn.Module, variables: Variables, *inputs: jax.Array | Batch) -> jax.Array:
    """`module` over `inputs`, which returns one array."""
    output = module.apply(variables, *inputs)
    assert isinstance(output, jax.Array)
    return output


@dataclass(frozen=True)
class EndToEnd:
    """REPA-E's end-to-end tuning of the autoencoder.

    `align_weight` is `vae_align_proj_coeff`, the REPA loss's weight in the
    autoencoder's update. The autoencoder's own loss is REPA-E's
    `ReconstructionLoss_Single_Stage` with configs/l1_lpips_kl_gan.yaml:

    - the L1 reconstruction, at `reconstruction_weight`;
    - LPIPS between the images and the reconstruction, at `perceptual_weight`
      (`dew.eval.LPIPSNetwork`, its published weights held frozen under
      `PERCEPTUAL`);
    - the posterior's KL, at `kl_weight`, summed over each latent and averaged
      over the batch;
    - the PatchGAN's generator term, minus the mean logit of the reconstruction,
      at `discriminator_weight` from step `discriminator_start`.

    The PatchGAN (`PatchDiscriminator` at `discriminator_width` and
    `discriminator_layers`) trains on the hinge loss of the images against the
    same step's reconstruction, in `params` as `DISCRIMINATOR`. A weight of 0
    leaves its network out. An affine-free batch norm (`momentum`, `epsilon`)
    normalizes the latents per channel; it replaces the autoencoder's fixed
    latent scale and starts from it.
    """

    align_weight: float = 1.5
    reconstruction_weight: float = 1.0
    perceptual_weight: float = 1.0
    kl_weight: float = 1e-6
    discriminator_weight: float = 0.1
    discriminator_start: int = 0
    discriminator_width: int = 64
    discriminator_layers: int = 3
    momentum: float = 0.1
    epsilon: float = 1e-4

    @property
    def discriminator(self) -> PatchDiscriminator | None:
        """The PatchGAN the recipe trains, or None at `discriminator_weight` 0."""
        if not self.discriminator_weight:
            return None
        return PatchDiscriminator(self.discriminator_width, self.discriminator_layers)

    def initial_statistics(self, shift: ArrayLike, scale: ArrayLike, channels: int) -> Variables:
        """Return the running statistics REPA-E's `init_bn` starts from.

        Per channel, the mean is the autoencoder's latent shift and the variance is
        1 / scale^2.
        """
        scale = jnp.asarray(scale, jnp.float32)
        return {"mean": jnp.broadcast_to(jnp.asarray(shift, jnp.float32), (channels,)),
                "var": jnp.broadcast_to(1.0 / scale ** 2, (channels,))}

    def tuned(self, autoencoder: AutoEncoder, variables: Variables) -> tuple[AutoEncoder, Variables]:
        """Return the autoencoder a task over a tuned run's `variables` decodes with, and its variables.

        The trained weights come from `params/autoencoder` and go under
        `autoencoder`. The latents are shifted by the running mean and scaled by the
        reciprocal square root of the running variance, without the batch norm's
        epsilon, which is the scale REPA-E's `SiT.extract_latents_stats` gives its
        sampling to denormalize with.
        """
        tuned = copy.copy(autoencoder)
        statistics = variables[LATENT_STATS]
        tuned.latent_shift = statistics["mean"]
        tuned.latent_scale = jax.lax.rsqrt(statistics["var"])
        tuned.params = variables["params"][AUTOENCODER]
        return tuned, {**variables, "autoencoder": tuned.params}

    def normalized(self, latents: jax.Array, statistics: Variables) -> jax.Array:
        """`latents` under the running statistics: the batch norm in eval mode."""
        return (latents - statistics["mean"]) / jnp.sqrt(statistics["var"] + self.epsilon)

    def batch_normalized(self, latents: jax.Array, statistics: Variables, batch: Batch
                         ) -> tuple[jax.Array, Variables]:
        """`latents` under their own statistics over every axis but the
        channels, and the running statistics after them: torch's
        `BatchNorm2d` in training mode, whose running variance is unbiased.
        The statistics are `batch`'s rows' (`Objective.row_mean`)."""
        mean, var = _moments(latents, batch)
        count = Objective.row_mean(latents[..., 0], batch).mass
        following = {"mean": (1 - self.momentum) * statistics["mean"] + self.momentum * mean,
                     "var": (1 - self.momentum) * statistics["var"]
                     + self.momentum * var * count / (count - 1)}
        return (latents - mean) / jnp.sqrt(var + self.epsilon), following

    def regularizer(self, images: jax.Array, reconstruction: jax.Array, moments: jax.Array, *,
                    perceptual: Variables | None, discriminator: Variables | None,
                    step: jax.Array, batch: Batch) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        """The autoencoder's own loss, the discriminator's, and their terms.

        `perceptual` is the LPIPS network's variables and `discriminator` the
        PatchGAN's, each None where its weight is 0. The generator term reads
        the discriminator frozen and the discriminator's hinge loss reads the
        images and the reconstruction frozen, so each loss trains only its
        own network, on what the step began with. Every mean is over
        `batch`'s rows (`Objective.row_mean`)."""
        def mean_of(values: jax.Array) -> jax.Array:
            return Objective.row_mean(values, batch).mean()[0]

        # At least float32, and a float64 run stays float64.
        dtype = jnp.promote_types(jnp.result_type(moments, reconstruction), jnp.float32)
        mean, log_variance = jnp.split(moments.astype(dtype), 2, axis=-1)
        log_variance = jnp.clip(log_variance, -30.0, 20.0)
        axes = tuple(range(1, mean.ndim))
        kl = mean_of(0.5 * jnp.sum(jnp.square(mean) + jnp.exp(log_variance) - 1.0 - log_variance, axis=axes))
        reconstruction = reconstruction.astype(dtype)
        reconstruction_error = mean_of(jnp.abs(images - reconstruction))
        total = self.reconstruction_weight * reconstruction_error + self.kl_weight * kl
        terms = {"reconstruction": reconstruction_error, "kl": kl}
        if perceptual is not None:
            from dew.eval.lpips import LPIPSNetwork

            terms["perceptual"] = mean_of(_applied(LPIPSNetwork(), perceptual, images, reconstruction))
            total = total + self.perceptual_weight * terms["perceptual"]
        hinge = jnp.zeros((), dtype)
        network = self.discriminator
        if network is not None and discriminator is not None:
            started = (step >= self.discriminator_start).astype(dtype)
            frozen = jax.lax.stop_gradient(discriminator)
            terms["generator"] = -mean_of(_applied(network, frozen, reconstruction, batch))
            total = total + self.discriminator_weight * started * terms["generator"]
            real, fake = (_applied(network, discriminator, jax.lax.stop_gradient(pixels), batch)
                          for pixels in (images, reconstruction))
            hinge = started * 0.5 * (mean_of(nn.relu(1.0 - real)) + mean_of(nn.relu(1.0 + fake)))
            terms["discriminator"] = hinge
        return total, hinge, terms


__all__ = ["AUTOENCODER", "LATENT_STATS", "PERCEPTUAL", "EndToEnd", "PatchDiscriminator"]
