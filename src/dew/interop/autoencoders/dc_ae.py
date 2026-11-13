"""DC-AE as a release stores it: its config's fields and tensors' names, and its loader, for the model
`dew.nn.autoencoders.dc_ae` computes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp

from dew import _records
from dew.batch import Variables
from dew.interop.components import bind_component
from dew.interop.weights import indexed_path
from dew.nn.autoencoders.dc_ae import ACTIVATIONS, DCAE, DCAutoencoder

from .common import published_source

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

PUBLISHED = ("AutoencoderDC",)
"""The `_class_name` a published AutoencoderDC's config states."""


class DCAEFields(TypedDict):
    image_channels: int
    latent_channels: int
    head_dim: int
    encoder_blocks: tuple[str, ...]
    decoder_blocks: tuple[str, ...]
    encoder_channels: tuple[int, ...]
    decoder_channels: tuple[int, ...]
    encoder_layers: tuple[int, ...]
    decoder_layers: tuple[int, ...]
    encoder_scales: tuple[tuple[int, ...], ...]
    decoder_scales: tuple[tuple[int, ...], ...]
    decoder_norms: tuple[str, ...]
    decoder_activations: tuple[str, ...]
    unshuffle: bool
    interpolate: bool
    encoder_shortcut: bool
    decoder_shortcut: bool
    decoder_out_activation: str


def _per_level(config: Mapping[str, object], name: str, levels: int) -> tuple:
    """A config entry the source takes as one value or one per level."""
    value = config[name]
    if isinstance(value, str):
        values = (value,) * levels
    elif isinstance(value, (list, tuple)):
        values = tuple(value)
    else:
        raise ValueError(f"{name}={value!r}: this field is one name or one entry per level")
    if len(values) != levels:
        raise ValueError(f"{name} has {len(values)} entries for {levels} levels")
    return values


def dc_ae_fields(config: Mapping[str, object]) -> DCAEFields:
    """Read an `AutoencoderDC` config into the fields `DCAE` takes, refusing
    what the port does not compute."""
    encoder_channels = _records.integers(config["encoder_block_out_channels"], "encoder_block_out_channels")
    decoder_channels = _records.integers(config["decoder_block_out_channels"], "decoder_block_out_channels")
    levels = len(encoder_channels)
    if len(decoder_channels) != levels:
        raise ValueError("the encoder and decoder have different numbers of levels")
    fields = DCAEFields(
        image_channels=_records.integer(config["in_channels"], "in_channels"),
        latent_channels=_records.integer(config["latent_channels"], "latent_channels"),
        head_dim=_records.integer(config["attention_head_dim"], "attention_head_dim"),
        encoder_blocks=_per_level(config, "encoder_block_types", levels),
        decoder_blocks=_per_level(config, "decoder_block_types", levels),
        encoder_channels=encoder_channels, decoder_channels=decoder_channels,
        encoder_layers=_per_level(config, "encoder_layers_per_block", levels),
        decoder_layers=_per_level(config, "decoder_layers_per_block", levels),
        encoder_scales=tuple(map(tuple, _per_level(config, "encoder_qkv_multiscales", levels))),
        decoder_scales=tuple(map(tuple, _per_level(config, "decoder_qkv_multiscales", levels))),
        decoder_norms=_per_level(config, "decoder_norm_types", levels),
        decoder_activations=_per_level(config, "decoder_act_fns", levels),
        unshuffle=config["downsample_block_type"] == "pixel_unshuffle",
        interpolate=config["upsample_block_type"] == "interpolate",
        # AutoencoderDC's own defaults where a config states none.
        encoder_shortcut=_records.boolean(config.get("encoder_out_shortcut", True), "encoder_out_shortcut"),
        decoder_shortcut=_records.boolean(config.get("decoder_in_shortcut", True), "decoder_in_shortcut"),
        decoder_out_activation=str(config.get("decoder_conv_act_fn", "relu")))
    checks = [
        (
            "block types",
            {*fields["encoder_blocks"], *fields["decoder_blocks"]},
            {"ResBlock", "EfficientViTBlock"},
        ),
        ("decoder norm types", set(fields["decoder_norms"]), {"rms_norm", "batch_norm"}),
        ("decoder activations", {*fields["decoder_activations"], fields["decoder_out_activation"]},
         set(ACTIVATIONS)),
        ("downsample_block_type", {config["downsample_block_type"]}, {"pixel_unshuffle", "Conv"}),
        ("upsample_block_type", {config["upsample_block_type"]}, {"pixel_shuffle", "interpolate"}),
    ]
    for what, found, known in checks:
        if not found <= known:
            raise ValueError(
                f"{what} {sorted(found - known)} are not ones AutoencoderDC builds: {sorted(known)}"
            )
    return fields


_NAME = re.compile(r"(encoder|decoder)(\.\d+|\.[a-z][a-z0-9_]*)+\.(weight|bias|running_mean|running_var)")


_NORMS = ("norm", "norm_out")


def dc_ae_path(name: str, ndim: int) -> tuple[str, ...] | None:
    """The parameter path of one `AutoencoderDC` tensor, or None for a batch
    norm's step count. Digits join the name before them; a norm's weight is
    its 1-D scale, and every other weight a 2-D or 4-D kernel. The multiscale
    branch's `proj_out` is `per_head`: a bare `proj_out` names a sharded
    projection elsewhere in Dew."""
    if name.endswith(".num_batches_tracked"):
        return None
    if _NAME.fullmatch(name) is None:
        raise ValueError(f"{name!r} is not an AutoencoderDC tensor")
    parts = name.split(".")
    leaf = parts.pop()
    norm = parts[-1] in _NORMS
    ranks = (2, 4) if leaf == "weight" and not norm else (1,)
    if ndim not in ranks:
        raise ValueError(f"{name!r} has {ndim} axes, not {' or '.join(map(str, ranks))}")
    path = indexed_path("per_head" if part == "proj_out" else part for part in parts)
    if leaf == "weight":
        leaf = "scale" if norm else "kernel"
    return (*path, leaf)


def load_dc_ae(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
               subfolder: str = "", param_dtype: str = "float32", params: Variables | None = None,
               lazy: bool = False, shapes: Mapping[str, tuple[int, ...]] | None = None
               ) -> tuple[DCAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published DC-AE, its parameters and their source layouts from
    `subfolder` of a directory or Hub repo holding its config.json and
    weights, such as `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` (the root)
    or a SANA pipeline (`vae`); only that component downloads. Every tensor
    the module declares must be published, in the shape it declares.

    Supplied `params` are bound unchanged: only the config is read, and no
    source layouts are returned. `lazy` leaves read ones `SourceLeaf`s for a
    placement to read (`dew.interop.weights.record_layouts`)."""
    directory, config = published_source(name_or_dir, revision, subfolder, PUBLISHED, weights=params is None)
    model = DCAE(**dc_ae_fields(config), dtype=compute)
    frame = jax.ShapeDtypeStruct((1, model.downscale_factor, model.downscale_factor, model.image_channels),
                                 jnp.float32)
    return bind_component(
        directory, "vae", config, model, dc_ae_path,
        lambda bound: DCAutoencoder(model=model, params=bound,
                                    latent_scale=config.get("scaling_factor", 1.0)),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy, inputs=(frame,),
        shapes=shapes)

