"""RAE and its DINOv2 encoder as a release stores them: the config's fields, the tensors' names, the
statistics it keeps unread, and their loaders, for the models `dew.nn.autoencoders.rae` computes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from dew import _records
from dew.batch import Variables
from dew.interop.components import bind_component, component_source
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.unread import Stored
from dew.nn.autoencoders.rae import ENCODERS, HEAD_WIDTH, RAE, RAEAutoencoder, RepresentationEncoder
from dew.nn.blocks import IMAGENET_MEAN, IMAGENET_STD

from .common import published_source

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

PUBLISHED = ("AutoencoderRAE",)
"""The `_class_name` a published AutoencoderRAE's config states."""


RAE_ENCODERS = ("dinov2", "siglip2", "mae")


def rae_fields(config: Mapping[str, object]) -> NativeFields[RAE]:
    """Read an `AutoencoderRAE` config into the fields `RAE` takes, refusing
    what the port does not compute."""
    kind = _records.text(config.get("encoder_type", "dinov2"), "encoder_type")
    if kind not in RAE_ENCODERS:
        raise ValueError(f"encoder_type {kind!r} is not one of {RAE_ENCODERS}")
    if not _records.boolean(config.get("reshape_to_2d", True), "reshape_to_2d"):
        raise ValueError("a latent diffusion run reads an RAE's latent as a grid; reshape_to_2d must be true")
    if _records.integer(config.get("num_channels", 3), "num_channels") != 3:
        raise ValueError("the RAE encoders read three-channel images")

    def integer(name: str, default: int) -> int:
        return _records.integer(config.get(name, default), name)

    def per_channel(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
        value = config.get(name)
        if value is None:
            return default
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name}={value!r}: this field is one number per image channel")
        return tuple(_records.number(entry, name) for entry in value)

    input_size = integer("encoder_input_size", 224)
    encoder_patch = integer("encoder_patch_size", 14)
    patch = integer("patch_size", 16)
    if input_size % encoder_patch:
        raise ValueError(
            f"encoder_input_size {input_size} is not a multiple of encoder_patch_size {encoder_patch}"
        )
    size = config.get("image_size")
    if size is not None and size != patch * input_size // encoder_patch:
        raise ValueError(f"image_size {size} is not patch_size times the encoder's grid")
    return native_fields(RAE)(
        kind=kind,
        encoder_width=integer("encoder_hidden_size", 768),
        encoder_layers=integer("encoder_num_hidden_layers", 12),
        encoder_patch=encoder_patch,
        input_size=input_size,
        decoder_width=integer("decoder_hidden_size", 512),
        decoder_layers=integer("decoder_num_hidden_layers", 8),
        decoder_heads=integer("decoder_num_attention_heads", 16),
        decoder_intermediate=integer("decoder_intermediate_size", 2048),
        patch=patch,
        pixel_mean=per_channel("encoder_norm_mean", IMAGENET_MEAN),
        pixel_std=per_channel("encoder_norm_std", IMAGENET_STD),
    )


_LAYER = {
    "norm1": "input_layernorm",
    "layernorm_before": "input_layernorm",
    "layer_norm1": "input_layernorm",
    "norm2": "post_attention_layernorm",
    "layernorm_after": "post_attention_layernorm",
    "layer_norm2": "post_attention_layernorm",
    "attention.attention.query": "self_attn.q_proj",
    "attention.attention.key": "self_attn.k_proj",
    "attention.attention.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.o_proj",
    "self_attn.q_proj": "self_attn.q_proj",
    "self_attn.k_proj": "self_attn.k_proj",
    "self_attn.v_proj": "self_attn.v_proj",
    "self_attn.out_proj": "self_attn.o_proj",
    "attention.to_q": "self_attn.q_proj",
    "attention.to_k": "self_attn.k_proj",
    "attention.to_v": "self_attn.v_proj",
    "attention.to_out.0": "self_attn.o_proj",
    "mlp.fc1": "mlp.up_proj",
    "intermediate.dense": "mlp.up_proj",
    "mlp.fc2": "mlp.down_proj",
    "output.dense": "mlp.down_proj",
    "layer_scale1": "attention_scale",
    "layer_scale2": "mlp_scale",
}
"""A layer's module, after its index, in each source's naming, and the port's."""


_LAYERS = {"encoder.encoder.layer": "encoder", "encoder.vision_model.encoder.layers": "encoder",
           "decoder.decoder_layers": "decoder"}


_MODULES = {
    "encoder.embeddings.patch_embeddings.projection": ("encoder", "patch_embed"),
    "encoder.vision_model.embeddings.patch_embedding": ("encoder", "patch_embed"),
    "encoder.layernorm": ("encoder", "post_norm"),
    "decoder.decoder_embed": ("decoder", "embed"), "decoder.decoder_norm": ("decoder", "norm"),
    "decoder.decoder_pred": ("decoder", "pred"),
}


_TENSORS = {
    "encoder.embeddings.cls_token": ("encoder", "class_token"),
    "encoder.embeddings.register_tokens": ("encoder", "register_tokens"),
    "encoder.embeddings.position_embeddings": ("encoder", "position_table", "embedding"),
    "encoder.vision_model.embeddings.position_embedding.weight": ("encoder", "position_table", "embedding"),
    "decoder.trainable_cls_token": ("decoder", "cls_token"),
}


UNREAD = ("encoder_mean", "encoder_std", "_latents_mean", "_latents_std", "encoder.embeddings.mask_token",
          "decoder.decoder_pos_embed", "encoder.vision_model.head.")
"""Tensors the source does not compute with: the normalization buffers the
config also holds, DINOv2's mask token (nothing is masked), the decoder
table it recomputes, and SigLIP's pooling head (the latent is the tokens)."""


def rae_path(name: str, ndim: int) -> tuple[str, ...] | None:
    """The parameter path of one `AutoencoderRAE` tensor, or None for one of
    `UNREAD`. A weight of two or more axes is a kernel, of one a norm's
    scale."""
    if name.startswith(UNREAD):
        return None
    if name in _TENSORS:
        return _TENSORS[name]
    module, _, leaf = name.rpartition(".")
    if leaf not in ("weight", "bias", "lambda1"):
        raise ValueError(f"{name!r} is not an AutoencoderRAE tensor")
    if leaf == "weight":
        leaf = "kernel" if ndim >= 2 else "scale"
    if module in _MODULES:
        return (*_MODULES[module], leaf)
    for prefix, owner in _LAYERS.items():
        index, _, inner = module.removeprefix(prefix + ".").partition(".")
        if module.startswith(prefix + ".") and index.isdigit() and inner in _LAYER:
            renamed = tuple(_LAYER[inner].split("."))
            return (owner, f"blocks_{index}", *renamed) if leaf == "lambda1" else (
                owner, f"blocks_{index}", *renamed, leaf)
    raise ValueError(f"{name!r} is not an AutoencoderRAE tensor")


def load_rae(
    name_or_dir: str | Path,
    compute=jnp.float32,
    *,
    revision: str | None = None,
    subfolder: str = "",
    param_dtype: str = "float32",
    params: Variables | None = None,
    lazy: bool = False,
    shapes: Mapping[str, tuple[int, ...]] | None = None,
    tensors: Mapping[str, Stored] | None = None,
) -> tuple[RAEAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published RAE, its parameters and their source layouts from
    `subfolder` of a directory or Hub repo, such as
    `nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08`. Every tensor the module
    declares must be published, in the shape it declares.

    Supplied `params` are bound unchanged: only the config is read, and no
    source layouts are returned. `lazy` leaves read ones `SourceLeaf`s for a
    placement to read (`dew.interop.weights.record_layouts`)."""
    directory, config = published_source(name_or_dir, revision, subfolder, PUBLISHED,
                                         weights=params is None and tensors is None)
    model = rae_fields(config).value.clone(dtype=compute)
    image = jax.ShapeDtypeStruct((1, model.input_size, model.input_size, 3), jnp.float32)
    return bind_component(
        directory, "vae", config, model, rae_path,
        lambda bound: RAEAutoencoder(model=model, params=bound, latents_mean=config.get("latents_mean"),
                                     latents_std=config.get("latents_std"),
                                     scaling_factor=config.get("scaling_factor", 1.0)),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy, inputs=(image,),
        shapes=shapes, tensors=tensors)

def load_dinov2(name_or_dir: str | Path, compute=jnp.float32, *, revision: str | None = None,
                param_dtype: str = "float32", params: Variables | None = None
                ) -> tuple[RepresentationEncoder, Variables, tuple[WeightLayout, ...]]:
    """Build a transformers `Dinov2Model` checkpoint, such as
    `facebook/dinov2-base`, as a `dinov2_plain` `RepresentationEncoder` at its
    518-pixel input, with its parameters and their source layouts under
    `representation`. It reads ImageNet-normalized `[B, S, S, 3]` pixels;
    `module.clone(input_size=224)` resizes the table to the 16x16 grid as
    `Dinov2Model` does. Supplied `params` are bound unchanged, only the config
    is read, and no layouts are returned."""
    directory, config = component_source(name_or_dir, revision, weights=params is None)
    if (
        config.get("model_type") != "dinov2"
        or config.get("use_swiglu_ffn")
        or config.get("hidden_act") != "gelu"
    ):
        raise ValueError(f"{directory} is not a GELU Dinov2Model; the port computes that one")
    geometry = (config.get("image_size"), config.get("mlp_ratio"), config.get("layer_norm_eps"),
                config["hidden_size"] // config["num_attention_heads"])
    if geometry != (ENCODERS["dinov2_plain"].table_size, 4, ENCODERS["dinov2_plain"].epsilon, HEAD_WIDTH):
        raise ValueError(
            f"the port reads a DINOv2 with a 518-pixel position table, a 4x feed-forward, norms at "
            f"1e-6 and {HEAD_WIDTH}-wide heads, not (image_size, mlp_ratio, layer_norm_eps, head "
            f"width) {geometry}"
        )
    module = RepresentationEncoder("dinov2_plain", config["hidden_size"], config["num_hidden_layers"],
                                   config["patch_size"], config["image_size"], compute)

    def path_of(name: str, rank: int) -> tuple[str, ...] | None:
        path = rae_path(f"encoder.{name}", rank)
        return None if path is None else path[1:]

    bound, tree, layouts, _ = bind_component(
        directory, "dinov2", config, module, path_of, lambda bound: module,
        prefix=("representation",), params=params, param_dtype=param_dtype,
        inputs=(jax.ShapeDtypeStruct((1, module.input_size, module.input_size, 3), jnp.float32),))
    return bound, tree, layouts

