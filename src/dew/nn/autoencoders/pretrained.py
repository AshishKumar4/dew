"""A published image autoencoder, built as its own config names it."""
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


def _class_name(modelname: str, revision: str) -> str | None:
    """The `_class_name` of the config at the root of `modelname`, which is
    where a DC-AE repo keeps it, or None where that cannot name one: a flax
    layout revision is an SD1-era AutoencoderKL branch, and a missing root
    config (an AutoencoderKL under `vae/`, a revision the repo lacks, or,
    offline, a config the cache does not hold) leaves the AutoencoderKL
    load to find the checkpoint or to say why it cannot."""
    if os.path.isdir(modelname):
        path = Path(modelname) / "config.json"
    elif revision in FLAX_REVISIONS:
        return None
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

        try:
            path = Path(hf_hub_download(modelname, "config.json", revision=revision))
        except (EntryNotFoundError, RevisionNotFoundError) as missing:
            _log.debug("%s names no root config at %s (%s); loading it as an AutoencoderKL",
                       modelname, revision, missing)
            return None
    return json.loads(path.read_text()).get("_class_name") if path.is_file() else None


def load_autoencoder(modelname: str, *, revision: str = "bf16", dtype: DTypeLike = jnp.bfloat16,
                     latent_shift: float | None = None, latent_scale: float | None = None,
                     params: Variables | None = None) -> AutoEncoder:
    """The autoencoder `modelname` publishes: a DC-AE where its config names
    `AutoencoderDC`, and otherwise a Stable Diffusion AutoencoderKL.

    `modelname` is a local directory or a Hub repo; `revision` is as
    `load_pretrained_vae` reads it, so a DC-AE takes `main` or a commit.
    Supplied `params` are bound unchanged and only metadata is read.
    `latent_shift` and `latent_scale` replace the checkpoint's own
    normalization where given.
    """
    if _class_name(modelname, revision) == "AutoencoderDC":
        from .dc_ae import load_dc_ae

        autoencoder, *_ = load_dc_ae(modelname, dtype, revision=revision, params=params)
        if latent_shift is not None:
            autoencoder.latent_shift = latent_shift
        if latent_scale is not None:
            autoencoder.latent_scale = latent_scale
        return autoencoder
    from .sd_vae import StableDiffusionVAE

    return StableDiffusionVAE(modelname, revision=revision, dtype=dtype, latent_shift=latent_shift,
                              latent_scale=latent_scale, params=params)
