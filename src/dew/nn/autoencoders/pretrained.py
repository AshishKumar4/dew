"""A published autoencoder, built as its own config names it."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import jax.numpy as jnp
from jax.typing import DTypeLike

from dew.objectives.base import Variables

from .api import AutoEncoder
from .vae import FLAX_REVISIONS

_log = logging.getLogger(__name__)

_PLACES = (None, "vae")
"""Where a repo keeps its autoencoder's config: at the root (a DC-AE's), or
under `vae/` (a pipeline's, such as Wan's)."""


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
    """The autoencoder `modelname` publishes: a DC-AE where its config names
    `AutoencoderDC`, Wan 2.1's video VAE where it names `AutoencoderKLWan`, a
    representation autoencoder where it names `AutoencoderRAE`, and otherwise
    a Stable Diffusion AutoencoderKL.

    `modelname` is a local directory or a Hub repo; `revision` is as
    `load_pretrained_vae` reads it, so a DC-AE, a Wan VAE or an RAE takes
    `main` or a commit. Supplied `params` are bound unchanged and only metadata is read.
    `latent_shift` and `latent_scale` replace the checkpoint's own
    normalization where given.
    """
    class_name, subfolder = _published(modelname, revision)
    if class_name == "AutoencoderDC":
        from .dc_ae import load_dc_ae

        autoencoder, *_ = load_dc_ae(
            modelname, dtype, revision=revision, subfolder=subfolder or "", params=params
        )
    elif class_name == "AutoencoderKLWan":
        from .wan import load_wan_vae

        autoencoder, *_ = load_wan_vae(
            modelname, dtype, revision=revision, subfolder=subfolder or "", params=params
        )
    elif class_name == "AutoencoderRAE":
        from .rae import load_rae

        autoencoder, *_ = load_rae(
            modelname, dtype, revision=revision, subfolder=subfolder or "", params=params
        )
    else:
        from .sd_vae import StableDiffusionVAE

        return StableDiffusionVAE(modelname, revision=revision, dtype=dtype, latent_shift=latent_shift,
                                  latent_scale=latent_scale, params=params)
    if latent_shift is not None:
        autoencoder.latent_shift = latent_shift
    if latent_scale is not None:
        autoencoder.latent_scale = latent_scale
    return autoencoder
