"""A published autoencoder, built as its own config names it."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from dew.interop.components import bind_component, component_source
from dew.objectives.base import Variables

from .api import AutoEncoder, LatentNormalization
from .dc_ae import DCAE, DCAutoencoder, load_dc_ae
from .flux2 import Flux2Autoencoder, load_flux2_vae
from .kl import AutoencoderKL
from .qwen_image import QwenImageAutoencoder, QwenImageVAE, load_qwen_image_vae
from .rae import RAE, RAEAutoencoder, load_rae
from .sd_vae import StableDiffusionVAE, published_normalization
from .vae import FLAX_REVISIONS, _vae_path, kl_config
from .wan import WanAutoencoder, WanVAE, load_wan_vae

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
                subfolder: str = "vae", param_dtype: str = "float32", params: Variables | None = None,
                lazy: bool = False, shapes: Mapping[str, tuple[int, ...]] | None = None
                ) -> tuple[StableDiffusionVAE, Variables, tuple[WeightLayout, ...], dict]:
    """Build a pipeline's AutoencoderKL, its parameters and their source
    layouts from the torch safetensors in `subfolder`, as `Loader` does. A
    standalone one is read by `StableDiffusionVAE` instead, which also reads
    the SD1-era flax layouts on their own revisions (`load_pretrained_vae`)."""
    directory, config = component_source(name_or_dir, revision, subfolder, weights=params is None)
    model = AutoencoderKL.from_diffusers(kl_config(config, directory), compute)
    shift, scale = published_normalization(config)
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
    """One built-in autoencoder: its wrapper, the name its record goes by and the native module its
    record's 'model' rebuilds (`AutoEncoder.to_json`, `from_json`), the `_class_name` a published
    config states, and the loader that builds it from a pipeline's directory or a repo."""
    wrapper: type[AutoEncoder]
    record: str
    module: type
    published: str
    load: Loader


REGISTRATIONS: tuple[Registration, ...] = (
    Registration(StableDiffusionVAE, 'sd_vae', AutoencoderKL, 'AutoencoderKL', load_kl_vae),
    Registration(DCAutoencoder, 'dc_ae', DCAE, 'AutoencoderDC', load_dc_ae),
    Registration(WanAutoencoder, 'wan', WanVAE, 'AutoencoderKLWan', load_wan_vae),
    Registration(RAEAutoencoder, 'rae', RAE, 'AutoencoderRAE', load_rae),
    Registration(Flux2Autoencoder, 'flux2', AutoencoderKL, 'AutoencoderKLFlux2', load_flux2_vae),
    Registration(QwenImageAutoencoder, 'qwen_image', QwenImageVAE, 'AutoencoderKLQwenImage21',
                 load_qwen_image_vae),
)
"""Every built-in autoencoder, which saved runs rebuild and sources load alike: one registration each."""

LOADERS: Mapping[str, Loader] = MappingProxyType({
    registration.published: registration.load for registration in REGISTRATIONS})
"""Every published autoencoder's loader, by the `_class_name` its config states. A pipeline's load
(`dew.interop.pipeline_assembly`) builds from this table; a standalone one (`load_autoencoder`) too,
but for an AutoencoderKL's, which also reads the SD1-era flax layouts (`StableDiffusionVAE`)."""


def registration(of: type[AutoEncoder] | str) -> Registration | None:
    """The registration of a wrapper class or of a record's name, None for one no built-in declares."""
    return next((entry for entry in REGISTRATIONS if of in (entry.wrapper, entry.record)), None)


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
    """The autoencoder `modelname` publishes: the one `LOADERS` builds for the
    class its config names, and otherwise a Stable Diffusion AutoencoderKL.

    `modelname` is a local directory or a Hub repo; `revision` is as
    `load_pretrained_vae` reads it, so a class of `LOADERS` takes `main` or a
    commit. Supplied `params` are bound unchanged and only metadata is read,
    but for FLUX.2's latent statistics, its batch norm's running statistics
    in the weights (`load_flux2_vae`). `latent_shift` and `latent_scale`
    replace the checkpoint's normalization where given, an autoencoder's
    whose record keeps them: one that normalizes by published per-channel
    statistics records those, so a run rebuilt from its record
    (`AutoEncoder.from_json`) would lose the override, and refuses it.
    """
    class_name, subfolder = _published(modelname, revision)
    if class_name is None or class_name not in LOADERS or class_name == "AutoencoderKL":
        return StableDiffusionVAE(modelname, revision=revision, dtype=dtype, latent_shift=latent_shift,
                                  latent_scale=latent_scale, params=params)
    autoencoder, *_ = LOADERS[class_name](modelname, dtype, revision=revision, subfolder=subfolder or "",
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
