"""FLUX.2's autoencoder, diffusers' `AutoencoderKLFlux2`, as its pipeline uses it.

The network is a Stable Diffusion `AutoencoderKL` with 32 latent channels
(and, in the small decoder, narrower decoder levels). What differs is the
latent the transformer reads: `Flux2Pipeline` folds each 2x2 block of the
VAE's latent into the channels, 32 to 128 at a sixteenth of the image's side,
and normalizes each of the 128 channels by the running statistics of the
VAE's affine-free batch norm, `(z - mean) / sqrt(var + eps)`. Both belong to
the latent here, so a diffusion run sees what the pipeline's transformer
sees.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from dew.interop.components import bind_component, component_source
from dew.objectives.base import Variables

from ..scan_orders import pixel_shuffle, pixel_unshuffle
from .api import ModuleAutoEncoder
from .kl import AutoencoderKL

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

STATISTICS = ("bn.running_mean", "bn.running_var", "bn.num_batches_tracked")
"""The batch norm's buffers: its statistics become the latent normalization,
and it computes nothing else."""


class Flux2Autoencoder(ModuleAutoEncoder[AutoencoderKL]):
    """Native FLUX.2 VAE weights, the 2x2 fold and the batch-norm statistics
    as the latent normalization, per folded channel."""

    def __init__(self, *, model: AutoencoderKL, params: Variables, mean, variance, epsilon: float):
        super().__init__(model, params)
        # `Flux2Pipeline._patchify_latents` and `_unpatchify_latents`, channels last.
        self.encode_single_frame = jax.jit(lambda params, image, key=None: pixel_unshuffle(jnp.asarray(
            model.apply({"params": params}, image, key, method=model.encode))))
        self.decode_single_frame = jax.jit(lambda params, latent: model.apply(
            {"params": params}, pixel_shuffle(latent), method=model.decode))
        self.latent_shift = np.asarray(mean, np.float32)
        self.latent_scale = 1.0 / np.sqrt(np.asarray(variance, np.float32) + np.float32(epsilon))
        if (
            self.latent_shift.shape != (4 * model.latent_channels,)
            or self.latent_scale.shape != self.latent_shift.shape
        ):
            raise ValueError(
                f"batch-norm statistics {self.latent_shift.shape} do not hold one value per folded "
                f"channel ({4 * model.latent_channels},)"
            )

    @property
    def downscale_factor(self) -> int:
        return 2 * self.model.downscale_factor

    @property
    def latent_channels(self) -> int:
        return 4 * self.model.latent_channels


def load_flux2_vae(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
                   subfolder: str = "vae", param_dtype: str = "float32", params: Variables | None = None,
                   lazy: bool = False
                   ) -> tuple[Flux2Autoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published FLUX.2 VAE, its parameters and their source layouts
    from `subfolder` of a pipeline directory or Hub repo. The batch norm's
    running statistics are read from the weights even where `params` are
    supplied, since they are the latent normalization, not parameters.
    `lazy` leaves the parameters `SourceLeaf`s for a placement to read
    (`dew.interop.weights.record_layouts`)."""
    from dew.interop.safetensors_io import read_weights
    from dew.nn.autoencoders.vae import _vae_path

    directory, config = component_source(name_or_dir, revision, subfolder, weights=True)
    if config.get("_class_name") != "AutoencoderKLFlux2":
        raise ValueError(
            f"{directory} holds a {config.get('_class_name')}, not an AutoencoderKLFlux2"
        )
    if tuple(config.get("patch_size", (2, 2))) != (2, 2):
        raise ValueError(f"FLUX.2's pipeline folds 2x2 latent blocks, not {config.get('patch_size')}")
    if not config.get("mid_block_add_attention", True) or config.get("act_fn", "silu") != "silu":
        raise ValueError("the port computes the published VAE: SiLU and mid-block attention")
    model = AutoencoderKL.from_diffusers(config, compute)
    tensors = read_weights(directory)
    frame = jax.ShapeDtypeStruct((1, model.downscale_factor, model.downscale_factor, model.image_channels),
                                 jnp.float32)
    return bind_component(
        directory, "vae", config, model,
        lambda name, rank: None if name in STATISTICS else _vae_path(name, rank),
        lambda bound: Flux2Autoencoder(model=model, params=bound, mean=tensors["bn.running_mean"],
                                       variance=tensors["bn.running_var"],
                                       epsilon=config.get("batch_norm_eps", 1e-4)),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy,
        inputs=(frame,), tensors=tensors)
