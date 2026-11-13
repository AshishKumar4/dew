"""Qwen-Image's VAE as a release stores it: its config's fields and tensors' names, and its loader, for the
model `dew.nn.autoencoders.qwen_image` computes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from dew import _records
from dew.batch import Variables
from dew.interop.components import bind_component
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.unread import Stored
from dew.interop.weights import indexed_path
from dew.nn.autoencoders.qwen_image import QwenImageAutoencoder, QwenImageVAE

from .common import published_source

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

PUBLISHED = ("AutoencoderKLQwenImage21",)
"""The `_class_name` a published AutoencoderKLQwenImage21's config states."""


def qwen_image_vae_fields(config: Mapping[str, object]) -> NativeFields[QwenImageVAE]:
    """Read an `AutoencoderKLQwenImage21` config into `QwenImageVAE` fields.

    Refused: `is_residual: false`, whose stack is laid out differently (flat
    down blocks, attention at `attn_scales`); a `patch_size`, whose
    patchify this module does not run; different input
    and output channels; a `scale_factor_spatial` the levels do not produce;
    and widths the average and duplicate shortcuts cannot group. The
    residual stack reads no `attn_scales`, so it is accepted as the source
    accepts it.
    """
    if config.get("is_residual", True) is not True:
        raise ValueError("Only the residual Qwen-Image VAE stack (is_residual: true) is implemented")
    if config.get("patch_size") is not None:
        raise ValueError(f"The Qwen-Image VAE's patchified layout is not implemented, got "
                         f"patch_size {config.get('patch_size')!r}")
    base_dim = _records.integer(config.get("base_dim"), "base_dim")
    decoder_base_dim = (base_dim if config.get("decoder_base_dim") is None
                        else _records.integer(config["decoder_base_dim"], "decoder_base_dim"))
    dim_mult = _records.integers(config.get("dim_mult"), "dim_mult")
    halving = config.get("temperal_downsample")
    if not isinstance(halving, (list, tuple)):
        raise ValueError(f"temperal_downsample={halving!r}: this field is a list of flags")
    temporal = tuple(_records.boolean(halves_time, "temperal_downsample") for halves_time in halving)
    if not dim_mult or len(temporal) != len(dim_mult) - 1:
        raise ValueError(f"temperal_downsample needs one flag per downsampling level of dim_mult {dim_mult}, "
                         f"got {temporal}")
    image_channels = _records.integer(config.get("in_channels"), "in_channels")
    if _records.integer(config.get("out_channels"), "out_channels") != image_channels:
        raise ValueError("The Qwen-Image VAE must reconstruct its input channels: in_channels "
                         f"{image_channels} against out_channels {config.get('out_channels')}")
    spatial = config.get("scale_factor_spatial")
    if spatial is not None and spatial != 2 ** (len(dim_mult) - 1):
        raise ValueError(f"scale_factor_spatial {spatial} is not the {2 ** (len(dim_mult) - 1)} "
                         f"the {len(dim_mult)} levels of dim_mult produce")
    encoder = [base_dim * multiple for multiple in (1, *dim_mult)]
    decoder = [decoder_base_dim * multiple for multiple in (dim_mult[-1], *dim_mult[::-1])]
    for index, halves_time in enumerate((*temporal, False)):
        down = index < len(temporal)
        factor = (2 if halves_time else 1) * (4 if down else 1)
        if encoder[index] * factor % encoder[index + 1]:
            raise ValueError(f"Encoder level {index} cannot average {encoder[index]} x {factor} phases "
                             f"into {encoder[index + 1]} channels")
    for index, doubles_time in enumerate(temporal[::-1]):
        if decoder[index + 1] * (2 if doubles_time else 1) * 4 % decoder[index]:
            raise ValueError(f"Decoder level {index} cannot duplicate {decoder[index]} channels into "
                             f"{decoder[index + 1]}")
    return native_fields(QwenImageVAE)(
        base_dim=base_dim,
        decoder_base_dim=decoder_base_dim,
        latent_channels=_records.integer(config.get("z_dim"), "z_dim"),
        dim_mult=dim_mult,
        num_res_blocks=_records.integer(config.get("num_res_blocks"), "num_res_blocks"),
        temporal_downsample=temporal,
        image_channels=image_channels,
    )


_PARAMETER = r"\.(?:weight|bias)"


_RESNET = rf"resnets\.\d+\.(?:norm[12]\.gamma|(?:conv[12]|conv_shortcut){_PARAMETER})"


_ATTENTION = rf"attentions\.\d+\.(?:norm\.gamma|(?:to_qkv|proj){_PARAMETER})"


_RESAMPLE = rf"(?:downsampler|upsampler)\.(?:resample\.1|time_conv){_PARAMETER}"


_NAME = re.compile(
    rf"(?:quant_conv|post_quant_conv){_PARAMETER}"
    rf"|(?:encoder|decoder)\.(?:(?:conv_in|conv_out){_PARAMETER}|norm_out\.gamma"
    rf"|mid_block\.(?:{_RESNET}|{_ATTENTION})"
    rf"|(?:down_blocks|up_blocks)\.\d+\.(?:{_RESNET}|{_RESAMPLE}))")


_INDEXED = {"down_blocks", "up_blocks", "resnets", "attentions"}


_LEAF_RANK = {"weight": (4,), "bias": (1,), "gamma": (3, 4)}


def qwen_image_vae_path(name: str, ndim: int) -> tuple[str, ...]:
    """The `QwenImageVAE` parameter path of a published tensor.

    A weight is a 4-D convolution kernel, which `record_layouts` transposes
    to HWIO; a gamma keeps its stored `[C, 1, 1(, 1)]` shape.
    """
    if _NAME.fullmatch(name) is None:
        raise ValueError(f"{name!r} is not an AutoencoderKLQwenImage21 tensor")
    # The attention's output conv is `proj` in the source; a bare `proj`
    # names the sharded feed-forward projections elsewhere in Dew.
    parts = name.replace("resample.1", "resample.conv").replace(".proj.", ".attention_out.").split(".")
    leaf = parts.pop()
    if ndim not in _LEAF_RANK[leaf]:
        raise ValueError(f"{name!r} has {ndim} axes, not {' or '.join(map(str, _LEAF_RANK[leaf]))}")
    return (*indexed_path(parts, indexed=_INDEXED), "kernel" if leaf == "weight" else leaf)


def load_qwen_image_vae(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
                        subfolder: str = "vae", param_dtype: str = "float32", params: Variables | None = None,
                        lazy: bool = False, shapes: Mapping[str, tuple[int, ...]] | None = None,
                        tensors: Mapping[str, Stored] | None = None,
                        read: Mapping[str, object] | None = None
                        ) -> tuple[QwenImageAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published Qwen-Image 2.1 VAE, its parameters and their source
    layouts from `subfolder` of a pipeline directory or Hub repo; only that
    component downloads. Every tensor the module declares must be published,
    in the shape it declares. Supplied `params` are bound without a weight read;
    `lazy` leaves read ones `SourceLeaf`s for a placement to read
    (`dew.interop.weights.record_layouts`)."""
    directory, config = published_source(name_or_dir, revision, subfolder, PUBLISHED,
                                         weights=params is None and tensors is None, read=read)
    model = qwen_image_vae_fields(config).value.clone(dtype=compute)
    frame = jax.ShapeDtypeStruct((1, model.downscale_factor, model.downscale_factor, model.image_channels),
                                 jnp.float32)
    return bind_component(
        directory, "vae", config, model, qwen_image_vae_path,
        lambda bound: QwenImageAutoencoder(model=model, params=bound, latents_mean=config["latents_mean"],
                                           latents_std=config["latents_std"]),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy, inputs=(frame,),
        shapes=shapes, tensors=tensors)
