"""Wan's VAE as a release stores it: its config's fields and tensors' names, and its loader, for the model
`dew.nn.autoencoders.wan` computes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp

from dew import _records
from dew.batch import Variables
from dew.interop.components import bind_component
from dew.interop.weights import indexed_path
from dew.nn.autoencoders.wan import WanAutoencoder, WanVAE

from .common import published_source

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

PUBLISHED = ("AutoencoderKLWan",)
"""The `_class_name` a published AutoencoderKLWan's config states."""


class WanVAEFields(TypedDict):
    base: int
    latent: int
    multipliers: tuple[int, ...]
    blocks: int
    temporal: tuple[bool, ...]


def wan_vae_fields(config: Mapping[str, object]) -> WanVAEFields:
    """Read an `AutoencoderKLWan` config into the fields `WanVAE` takes,
    refusing what the port does not compute."""
    multipliers = _records.integers(config["dim_mult"], "dim_mult")
    halvings = config["temperal_downsample"]
    if not isinstance(halvings, (list, tuple)):
        raise ValueError(f"temperal_downsample={halvings!r}: this field is one flag per downsampling")
    temporal = tuple(_records.boolean(halves, "temperal_downsample") for halves in halvings)
    if len(temporal) != len(multipliers) - 1:
        raise ValueError(f"temperal_downsample has {len(temporal)} entries for {len(multipliers)} levels")
    if config.get("attn_scales"):
        raise ValueError(
            "attention inside the levels (attn_scales) is not ported; the published VAEs have none"
        )
    for name in ("is_residual", "patch_size"):
        if config.get(name):
            raise ValueError(f"{name} belongs to Wan 2.2's VAE, which this port does not compute")
    return WanVAEFields(base=_records.integer(config["base_dim"], "base_dim"),
                        latent=_records.integer(config["z_dim"], "z_dim"), multipliers=multipliers,
                        blocks=_records.integer(config["num_res_blocks"], "num_res_blocks"),
                        temporal=temporal)


_LEAF_RANK = {"weight": (4, 5), "bias": (1,), "gamma": (3, 4)}


def wan_vae_path(name: str, ndim: int) -> tuple[str, ...]:
    """The parameter path of one `AutoencoderKLWan` tensor. Digits join the
    name before them; a 4-D or 5-D weight is a convolution kernel, and a gamma
    keeps its stored shape. The attention's output `proj` is `attention_out`:
    a bare `proj` names a sharded projection elsewhere in Dew."""
    parts = name.split(".")
    leaf = parts.pop()
    if parts[0] not in ("encoder", "decoder", "quant_conv", "post_quant_conv") or leaf not in _LEAF_RANK:
        raise ValueError(f"{name!r} is not an AutoencoderKLWan tensor")
    if ndim not in _LEAF_RANK[leaf]:
        raise ValueError(f"{name!r} has {ndim} axes, not {' or '.join(map(str, _LEAF_RANK[leaf]))}")
    path = indexed_path("attention_out" if part == "proj" else part for part in parts)
    return (*path, "kernel" if leaf == "weight" else leaf)


def load_wan_vae(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
                 subfolder: str = "vae", param_dtype: str = "float32", params: Variables | None = None,
                 lazy: bool = False, shapes: Mapping[str, tuple[int, ...]] | None = None
                 ) -> tuple[WanAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published Wan 2.1 VAE, its parameters and their source layouts
    from `subfolder` of a pipeline directory or Hub repo, such as
    `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`; only that component downloads. Every
    tensor the module declares must be published, in the shape it declares.

    Supplied `params` are bound unchanged: only the config is read, and no
    source layouts are returned. `lazy` leaves read ones `SourceLeaf`s for a
    placement to read (`dew.interop.weights.record_layouts`)."""
    directory, config = published_source(name_or_dir, revision, subfolder, PUBLISHED, weights=params is None)
    model = WanVAE(**wan_vae_fields(config), dtype=compute)
    video = jax.ShapeDtypeStruct(
        (1, 1 + model.temporal_factor, model.downscale_factor, model.downscale_factor, 3), jnp.float32
    )
    return bind_component(
        directory, "vae", config, model, wan_vae_path,
        lambda bound: WanAutoencoder(model=model, params=bound, latents_mean=config["latents_mean"],
                                     latents_std=config["latents_std"]),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy, inputs=(video,),
        shapes=shapes)

