"""A published autoencoder, built as its own config names it."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from dew.batch import Variables
from dew.interop.components import bind_component
from dew.registry import DtypeName

from . import dc_ae, flux2, qwen_image, rae, wan
from .api import AutoEncoder, LatentNormalization, published_source
from .dc_ae import DCAutoencoder, load_dc_ae
from .flux2 import Flux2Autoencoder, load_flux2_vae
from .qwen_image import QwenImageAutoencoder, load_qwen_image_vae
from .rae import RAEAutoencoder, load_rae
from .sd_vae import StableDiffusionVAE, published
from .vae import (
    FLAX_REVISIONS,
    FLAX_WEIGHTS,
    PUBLISHED as KL_PUBLISHED,
    TORCH_WEIGHTS,
    _vae_path,
    flax_vae,
    located_vae,
)
from .wan import WanAutoencoder, load_wan_vae

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

_log = logging.getLogger(__name__)

_PLACES = (None, "vae")
"""Where a repo keeps its autoencoder's config: at the root (a DC-AE's), or
under `vae/` (a pipeline's, such as Wan's)."""


class Loader(Protocol):
    """Builds a published autoencoder, its parameters, their source layouts
    and its config from `subfolder` of a directory or Hub repo. Supplied
    `params` are bound without a weight read, their layouts from the stored
    tensors' `shapes` where given, and `lazy` leaves read ones `SourceLeaf`s
    for a placement to read (`dew.interop.weights.record_layouts`)."""

    def __call__(self, name_or_dir: str | Path, compute=..., *, revision: str | None = None,
                 subfolder: str = ..., param_dtype: str = "float32", params: Variables | None = None,
                 lazy: bool = False, shapes: Mapping[str, tuple[int, ...]] | None = None
                 ) -> tuple[AutoEncoder, Variables, tuple[WeightLayout, ...], dict]: ...


def load_kl_vae(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
                subfolder: str | None = "vae", param_dtype: str = "float32", params: Variables | None = None,
                lazy: bool = False, shapes: Mapping[str, tuple[int, ...]] | None = None
                ) -> tuple[StableDiffusionVAE, Variables, tuple[WeightLayout, ...], dict]:
    """Build an AutoencoderKL, its parameters and their source layouts, as `Loader` does: a pipeline's
    from the torch safetensors in `subfolder`, a standalone one's (`subfolder` None) where its repo or
    directory holds it, the SD1-era flax layouts on their own revisions included (`vae.located_vae`,
    `revision` as it reads it, `bf16` by default). Those flax params keep the dtype their branch stores
    (bf16 on `bf16`), the tree this module's is, their layouts written back as the torch names
    (`vae.flax_vae`)."""
    if subfolder is None:
        directory, config, weights = located_vae(str(name_or_dir), revision or "bf16", params=params)
    else:
        directory, config = published_source(name_or_dir, revision, subfolder, KL_PUBLISHED,
                                             weights=params is None, nameless=True)
        weights = TORCH_WEIGHTS
    if weights == FLAX_WEIGHTS and params is None:
        params, shapes = flax_vae(directory / FLAX_WEIGHTS)
    model, shift, scale = published(config, compute)
    image = jax.ShapeDtypeStruct((1, model.downscale_factor, model.downscale_factor, model.image_channels),
                                 jnp.float32)
    return bind_component(
        directory, "vae", config, model, _vae_path,
        lambda bound: StableDiffusionVAE(str(name_or_dir), dtype=compute, params=bound, model=model,
                                         latent_shift=shift, latent_scale=scale),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy, inputs=(image,),
        shapes=shapes)


@dataclasses.dataclass(frozen=True)
class Registration:
    """One built-in autoencoder a source loads: its wrapper, the `_class_name`s a published config states
    (an AutoencoderKL's, where it states none: `nameless`), and the loader that builds it from a pipeline's
    directory or a repo, which a standalone load reads from its config's subfolder, or, where the
    autoencoder `finds` its own (an AutoencoderKL's, whose SD1-era flax layouts live on their own
    revisions: `load_kl_vae` with no subfolder), where it lives."""
    wrapper: Callable[..., AutoEncoder]
    published: tuple[str, ...]
    load: Loader
    nameless: bool = False
    finds: bool = False


REGISTRATIONS: tuple[Registration, ...] = (
    Registration(StableDiffusionVAE, KL_PUBLISHED, load_kl_vae, nameless=True, finds=True),
    Registration(DCAutoencoder, dc_ae.PUBLISHED, load_dc_ae),
    Registration(WanAutoencoder, wan.PUBLISHED, load_wan_vae),
    Registration(RAEAutoencoder, rae.PUBLISHED, load_rae),
    Registration(Flux2Autoencoder, flux2.PUBLISHED, load_flux2_vae),
    Registration(QwenImageAutoencoder, qwen_image.PUBLISHED, load_qwen_image_vae),
)
"""Every built-in autoencoder a source loads: one registration each."""

LOADERS: Mapping[str, Loader] = MappingProxyType({
    name: entry.load for entry in REGISTRATIONS for name in entry.published})
"""Every published autoencoder's loader, by each `_class_name` its config may state: what a pipeline's load
(`dew.interop.pipeline_assembly`) builds from, a config naming none taken as an AutoencoderKL's."""


def registration(class_name: str | None) -> Registration | None:
    """The registration of a published `_class_name` (None, an AutoencoderKL's: `Registration.nameless`),
    None for one no built-in registers."""
    return next((entry for entry in REGISTRATIONS
                 if class_name in entry.published or (class_name is None and entry.nameless)), None)


def _published(modelname: str, revision: str) -> tuple[str | None, str | None]:
    """The `_class_name` of the first config of `_PLACES` in `modelname`, and
    the subfolder it is in; (None, None) where none can name one. A flax
    layout revision is an SD1-era AutoencoderKL branch, and a repo without
    either config (a revision it lacks, or, offline, configs the cache does
    not hold) leaves the AutoencoderKL load to find the checkpoint or to say
    why it cannot."""
    if revision in FLAX_REVISIONS and not os.path.isdir(modelname):
        return None, None
    for subfolder in _PLACES:
        name = "config.json" if subfolder is None else f"{subfolder}/config.json"
        if os.path.isdir(modelname):
            path = Path(modelname) / name
            if not path.is_file():
                continue
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

            try:
                path = Path(hf_hub_download(modelname, name, revision=revision))
            except (EntryNotFoundError, RevisionNotFoundError) as missing:
                _log.debug("%s has no %s at %s (%s)", modelname, name, revision, missing)
                continue
        return json.loads(path.read_text()).get("_class_name"), subfolder
    return None, None


def load_autoencoder(modelname: str, *, revision: str = "bf16", dtype: DTypeLike = jnp.bfloat16,
                     latent_shift: float | None = None, latent_scale: float | None = None,
                     params: Variables | None = None) -> AutoEncoder:
    """The autoencoder `modelname` publishes: the one its config's class registers (`registration`), by
    its registration's loader (where it lives, for one that finds it: an AutoencoderKL's, read where the
    config names no class), and refused by name where it names one no built-in registers.

    `modelname` is a local directory or a Hub repo; `revision` is as
    `vae.located_vae` reads it, so a class of `LOADERS` takes `main` or a
    commit. Supplied `params` are bound unchanged and only metadata is read,
    but for FLUX.2's latent statistics, its batch norm's running statistics
    in the weights (`load_flux2_vae`). `latent_shift` and `latent_scale`
    replace the checkpoint's normalization where given, an autoencoder's
    whose record keeps them: one that normalizes by published per-channel
    statistics records those, so a run rebuilt from its record
    (`AutoEncoder.bound`) would lose the override, and refuses it.
    """
    class_name, subfolder = _published(modelname, revision)
    entry = registration(class_name)
    if entry is None:
        raise ValueError(f"{modelname} holds a {class_name}, which no built-in autoencoder registers")
    if entry.finds:
        # A local pipeline's autoencoder is read from its `vae/` folder; a Hub repo's, where its loader
        # finds it.
        source = str(Path(modelname) / subfolder) if subfolder and os.path.isdir(modelname) else modelname
        autoencoder, *_ = load_kl_vae(source, dtype, revision=revision, subfolder=None, params=params)
    else:
        autoencoder, *_ = entry.load(modelname, dtype, revision=revision, subfolder=subfolder or "",
                                     params=params)
    given = {name for name, value in (("latent_shift", latent_shift), ("latent_scale", latent_scale))
             if value is not None}
    if given - autoencoder.record_arguments().keys():
        raise ValueError(f"{class_name} normalizes by its published statistics, which its record keeps: "
                         f"{sorted(given)} would not survive the run's rebuild from that record")
    if not given:
        return autoencoder
    shift = autoencoder.latent_shift if latent_shift is None else latent_shift
    scale = autoencoder.latent_scale if latent_scale is None else latent_scale
    record = dict(autoencoder.normalization.record)
    record.update(latent_shift=shift, latent_scale=scale)
    return autoencoder.with_normalization(LatentNormalization(record, shift, scale))


@dataclasses.dataclass(frozen=True)
class PretrainedAutoencoder:
    """A published autoencoder for latent diffusion.

    It is the class the checkpoint's config names, one of
    `dew.nn.autoencoders.pretrained.LOADERS`, or a Stable Diffusion
    AutoencoderKL (`dew.nn.autoencoders.pretrained.load_autoencoder`).
    """

    modelname: str = "pcuenq/sd-vae-ft-mse-flax"
    revision: str = "bf16"
    dtype: DtypeName = "bfloat16"
    latent_shift: float | None = None
    latent_scale: float | None = None
    """Per-dataset latent statistics; None keeps the checkpoint's."""

    def build(self, *, params: Variables | None = None) -> AutoEncoder:
        """Build the autoencoder from its config over supplied params."""
        return load_autoencoder(self.modelname, revision=self.revision, dtype=jnp.dtype(self.dtype),
                                latent_shift=self.latent_shift, latent_scale=self.latent_scale,
                                params=params)


__all__ = ["PretrainedAutoencoder"]
