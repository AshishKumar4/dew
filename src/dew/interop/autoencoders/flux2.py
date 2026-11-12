"""FLUX.2's VAE as a release stores it: its batch norm's statistics read beside its KL autoencoder's tensors,
and its loader, for the model `dew.nn.autoencoders.flux2` computes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from dew.batch import Variables
from dew.interop.components import bind_component
from dew.interop.unread import Stored
from dew.nn.autoencoders.flux2 import Flux2Autoencoder

from .common import published_source
from .kl import autoencoder_kl_from_diffusers

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

LATENT_STATISTICS = ("bn.running_mean", "bn.running_var")
"""The batch norm's running mean and variance read as latent-normalization metadata."""

STATISTICS = (*LATENT_STATISTICS, "bn.num_batches_tracked")
"""The batch norm's buffers: its statistics become the latent normalization,
and it computes nothing else."""

PUBLISHED = ("AutoencoderKLFlux2",)
"""The `_class_name` a published AutoencoderKLFlux2's config states."""


def load_flux2_vae(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
                   subfolder: str = "vae", param_dtype: str = "float32", params: Variables | None = None,
                   lazy: bool = False, shapes: Mapping[str, tuple[int, ...]] | None = None,
                   tensors: Mapping[str, Stored] | None = None
                   ) -> tuple[Flux2Autoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published FLUX.2 VAE, its parameters and their source layouts
    from `subfolder` of a pipeline directory or Hub repo. The batch norm's
    running statistics are read from the weights even where `params` are
    supplied, since they are the latent normalization, not parameters.
    `lazy` leaves the parameters `SourceLeaf`s for a placement to read
    (`dew.interop.weights.record_layouts`)."""
    from dew.interop.autoencoders.vae import _vae_path
    from dew.interop.safetensors_io import read_weights

    directory, config = published_source(name_or_dir, revision, subfolder, PUBLISHED,
                                         weights=tensors is None)
    if tuple(config.get("patch_size", (2, 2))) != (2, 2):
        raise ValueError(f"FLUX.2's pipeline folds 2x2 latent blocks, not {config.get('patch_size')}")
    model = autoencoder_kl_from_diffusers(config, compute)
    loaded = read_weights(directory) if tensors is None else tensors
    mean, variance = loaded["bn.running_mean"], loaded["bn.running_var"]
    frame = jax.ShapeDtypeStruct((1, model.downscale_factor, model.downscale_factor, model.image_channels),
                                 jnp.float32)
    return bind_component(
        directory, "vae", config, model,
        lambda name, rank: None if name in STATISTICS else _vae_path(name, rank),
        lambda bound: Flux2Autoencoder(model=model, params=bound, mean=mean,
                                       variance=variance,
                                       epsilon=config.get("batch_norm_eps", 1e-4)),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy, shapes=shapes,
        inputs=(frame,), tensors=loaded)
