"""Representation autoencoders, diffusers' `AutoencoderRAE`.

An independent linen port of diffusers 0.40.0 autoencoder_rae.py
(Apache-2.0) and the frozen encoders it builds from transformers 4.57.1
(DINOv2 with registers, SigLIP, ViT-MAE), channels last. The latent is the
frozen encoder's patch tokens, a `[grid, grid, width]` map whatever the image
size, since the image is first resized (bicubic, as torch computes it) to
the encoder's input; a ViT decoder paints one patch per token.

The encoders are one pre-norm ViT with different tables: DINOv2 prepends a
class token and four registers, scales each residual branch by a layer
scale, and resizes its 37x37 position table with antialiasing; SigLIP has
neither token and a tanh GELU; ViT-MAE prepends a class token at its
224-pixel table, run with masking off. Each ends in an affine-free layer
norm. `dinov2_plain` is transformers' `Dinov2Model`, which REPA aligns to:
one class token, no registers, no antialiasing, its final norm kept
(`load_dinov2`). The decoder's table is diffusers' 2-D sine-cosine one with a
zero class row first.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype

from dew import records
from dew.interop.components import bind_component, component_source
from dew.nn.activations import activation
from dew.nn.attention import LayerNorm, scaled_dot_product_attention
from dew.nn.backbones.sd3 import sincos_position
from dew.nn.blocks import torch_bicubic_resize
from dew.nn.text_encoders import check_tree
from dew.objectives.base import Variables

from ..conv import Conv
from .api import ModuleAutoEncoder

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

HEAD_WIDTH = 64
"""Every encoder's attention head width; the source sets the heads from it."""


class Encoder(NamedTuple):
    """What the source fixes for one `encoder_type` beyond the RAE config."""

    table_size: int
    """The image size the position table was built for."""
    leading: int
    """Tokens before the patches: a class token, then registers."""
    intermediate: int | None
    """The feed-forward width, or None for four times the encoder's."""
    epsilon: float
    activation: str
    layer_scale: bool
    antialias: bool
    """Whether the position table resizes with antialiasing."""
    final_affine: bool = False
    """Whether the final norm keeps its scale and bias; an RAE strips them."""


ENCODERS = {
    "dinov2": Encoder(518, 5, None, 1e-6, "gelu", layer_scale=True, antialias=True),
    "siglip2": Encoder(256, 0, 3072, 1e-6, "gelu_pytorch_tanh", layer_scale=False, antialias=False),
    "mae": Encoder(224, 1, 3072, 1e-12, "gelu", layer_scale=False, antialias=False),
    "dinov2_plain": Encoder(518, 1, None, 1e-6, "gelu", layer_scale=True, antialias=False, final_affine=True),
}
"""The RAE encoders under their `encoder_type`, and transformers' `Dinov2Model`
(no registers, the final norm kept), which REPA aligns to."""

RAE_ENCODERS = ("dinov2", "siglip2", "mae")


class _Attention(nn.Module):
    width: int
    heads: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        batch, length, _ = x.shape
        split = (batch, length, self.heads, self.width // self.heads)
        query, key, value = (nn.Dense(self.width, dtype=self.dtype, name=name)(x).reshape(split)
                             for name in ("query", "key", "value"))
        attended = scaled_dot_product_attention(query, key, value, dtype=self.dtype)
        return nn.Dense(self.width, dtype=self.dtype, name="out")(attended.reshape(batch, length, self.width))


class _Block(nn.Module):
    """Pre-norm attention and feed-forward, each residual branch optionally
    scaled by a learned vector."""

    width: int
    heads: int
    intermediate: int
    epsilon: float
    activation: str
    layer_scale: bool = False
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        def scaled(branch, name):
            if not self.layer_scale:
                return branch
            return branch * self.param(name, nn.initializers.ones, (self.width,))

        norm = LayerNorm(epsilon=self.epsilon, dtype=self.dtype, name="norm1")
        x = x + scaled(
            _Attention(self.width, self.heads, self.dtype, name="attention")(norm(x)), "layer_scale1"
        )
        hidden = nn.Dense(self.intermediate, dtype=self.dtype, name="fc1")(
            LayerNorm(epsilon=self.epsilon, dtype=self.dtype, name="norm2")(x))
        hidden = activation(self.activation)(hidden)
        return x + scaled(nn.Dense(self.width, dtype=self.dtype, name="fc2")(hidden), "layer_scale2")


class RepresentationEncoder(nn.Module):
    """A frozen encoder: normalized `[B, S, S, 3]` pixels at its input size
    to its patch tokens `[B, grid, grid, width]`."""

    kind: str
    width: int
    layers: int
    patch: int
    input_size: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, pixels):
        spec = ENCODERS[self.kind]
        grid, table = self.input_size // self.patch, spec.table_size // self.patch
        tokens = Conv(self.width, (self.patch, self.patch), strides=(self.patch, self.patch), padding="VALID",
                      dtype=self.dtype, name="patch")(pixels)
        batch = tokens.shape[0]
        classes = int(spec.leading > 0)
        positions = self.param(
            "position_embeddings",
            nn.initializers.zeros,
            (1, classes + table * table, self.width) if classes else (table * table, self.width),
        )
        positions = positions.reshape(classes + table * table, self.width)
        patches = positions[classes:].reshape(1, table, table, self.width)
        if table != grid:
            patches = torch_bicubic_resize(patches, grid, grid, antialias=spec.antialias)
        tokens = (tokens + patches).reshape(batch, grid * grid, self.width)
        if classes:
            cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.width)) + positions[:1]
            leading = [jnp.broadcast_to(cls_token, (batch, 1, self.width))]
            if spec.leading > 1:
                registers = self.param(
                    "register_tokens", nn.initializers.zeros, (1, spec.leading - 1, self.width)
                )
                leading.append(jnp.broadcast_to(registers, (batch, spec.leading - 1, self.width)))
            tokens = jnp.concatenate([*leading, tokens], axis=1)
        intermediate = spec.intermediate or 4 * self.width
        for index in range(self.layers):
            tokens = _Block(self.width, self.width // HEAD_WIDTH, intermediate, spec.epsilon, spec.activation,
                            spec.layer_scale, self.dtype, name=f"layers_{index}")(tokens)
        tokens = LayerNorm(epsilon=spec.epsilon, use_scale=spec.final_affine, use_bias=spec.final_affine,
                           dtype=self.dtype, name="norm")(tokens)
        return tokens[:, spec.leading:].reshape(batch, grid, grid, self.width)


class _Decoder(nn.Module):
    """Latent tokens `[B, grid, grid, width]` to pixels `[B, grid * patch,
    grid * patch, channels]` in the encoder's normalization."""

    width: int
    layers: int
    heads: int
    intermediate: int
    grid: int
    patch: int
    channels: int = 3
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, latents):
        batch = latents.shape[0]
        if latents.shape[1:3] != (self.grid, self.grid):
            raise ValueError(f"an RAE decodes a {self.grid}x{self.grid} latent, not "
                             f"{latents.shape[1]}x{latents.shape[2]}")
        tokens = nn.Dense(self.width, dtype=self.dtype, name="embed")(
            latents.reshape(batch, -1, latents.shape[-1])
        )
        cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.width))
        tokens = jnp.concatenate(
            [jnp.broadcast_to(cls_token, (batch, 1, self.width)).astype(tokens.dtype), tokens], axis=1
        )
        positions = sincos_position(self.width, self.grid, base_size=16)
        tokens = tokens + jnp.concatenate([jnp.zeros((1, 1, self.width)), positions], axis=1)
        for index in range(self.layers):
            tokens = _Block(self.width, self.heads, self.intermediate, 1e-12, "gelu", dtype=self.dtype,
                            name=f"layers_{index}")(tokens)
        tokens = LayerNorm(epsilon=1e-12, dtype=self.dtype, name="norm")(tokens)
        patches = nn.Dense(self.patch * self.patch * self.channels, dtype=self.dtype, name="pred")(
            tokens[:, 1:]
        )
        patches = patches.reshape(batch, self.grid, self.grid, self.patch, self.patch, self.channels)
        return patches.transpose(0, 1, 3, 2, 4, 5).reshape(
            batch, self.grid * self.patch, self.grid * self.patch, self.channels)


class RAEFields(TypedDict):
    kind: str
    encoder_width: int
    encoder_layers: int
    encoder_patch: int
    input_size: int
    decoder_width: int
    decoder_layers: int
    decoder_heads: int
    decoder_intermediate: int
    patch: int
    pixel_mean: tuple[float, ...]
    pixel_std: tuple[float, ...]


class RAE(nn.Module):
    """`AutoencoderRAE` on images in [-1, 1]; the latent normalization
    belongs to the AutoEncoder. An image of any size encodes to the
    `[grid, grid, encoder_width]` latent, and a latent decodes to
    `grid * patch` pixels a side."""

    kind: str = "dinov2"
    encoder_width: int = 768
    encoder_layers: int = 12
    encoder_patch: int = 14
    input_size: int = 224
    decoder_width: int = 512
    decoder_layers: int = 8
    decoder_heads: int = 16
    decoder_intermediate: int = 2048
    patch: int = 16
    pixel_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    pixel_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    dtype: Dtype = jnp.float32

    @property
    def grid(self) -> int:
        return self.input_size // self.encoder_patch

    def setup(self):
        self.encoder = RepresentationEncoder(
            self.kind,
            self.encoder_width,
            self.encoder_layers,
            self.encoder_patch,
            self.input_size,
            self.dtype,
        )
        self.decoder = _Decoder(self.decoder_width, self.decoder_layers, self.decoder_heads,
                                self.decoder_intermediate, self.grid, self.patch, dtype=self.dtype)

    def encode(self, images, key=None):
        """The encoder's tokens; an RAE is deterministic, so `key` is unread."""
        del key
        pixels = (images + 1) / 2
        if pixels.shape[1:3] != (self.input_size, self.input_size):
            pixels = torch_bicubic_resize(pixels, self.input_size, self.input_size)
        return self.encoder((pixels - jnp.asarray(self.pixel_mean)) / jnp.asarray(self.pixel_std))

    def decode(self, latents):
        pixels = self.decoder(latents) * jnp.asarray(self.pixel_std) + jnp.asarray(self.pixel_mean)
        return pixels * 2 - 1

    def __call__(self, images):
        return self.decode(self.encode(images))


def rae_fields(config: Mapping[str, object]) -> RAEFields:
    """Read an `AutoencoderRAE` config into the fields `RAE` takes, refusing
    what the port does not compute."""
    kind = records.text(config.get("encoder_type", "dinov2"), "encoder_type")
    if kind not in RAE_ENCODERS:
        raise ValueError(f"encoder_type {kind!r} is not one of {RAE_ENCODERS}")
    if not records.boolean(config.get("reshape_to_2d", True), "reshape_to_2d"):
        raise ValueError("a latent diffusion run reads an RAE's latent as a grid; reshape_to_2d must be true")
    if records.integer(config.get("num_channels", 3), "num_channels") != 3:
        raise ValueError("the RAE encoders read three-channel images")

    def integer(name: str, default: int) -> int:
        return records.integer(config.get(name, default), name)

    def per_channel(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
        value = config.get(name)
        if value is None:
            return default
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name}={value!r}: this field is one number per image channel")
        return tuple(records.number(entry, name) for entry in value)

    fields = RAEFields(
        kind=kind,
        encoder_width=integer("encoder_hidden_size", 768),
        encoder_layers=integer("encoder_num_hidden_layers", 12),
        encoder_patch=integer("encoder_patch_size", 14),
        input_size=integer("encoder_input_size", 224),
        decoder_width=integer("decoder_hidden_size", 512),
        decoder_layers=integer("decoder_num_hidden_layers", 8),
        decoder_heads=integer("decoder_num_attention_heads", 16),
        decoder_intermediate=integer("decoder_intermediate_size", 2048),
        patch=integer("patch_size", 16),
        pixel_mean=per_channel("encoder_norm_mean", (0.485, 0.456, 0.406)),
        pixel_std=per_channel("encoder_norm_std", (0.229, 0.224, 0.225)),
    )
    if fields["input_size"] % fields["encoder_patch"]:
        raise ValueError(f"encoder_input_size {fields['input_size']} is not a multiple of "
                         f"encoder_patch_size {fields['encoder_patch']}")
    size = config.get("image_size")
    if size is not None and size != fields["patch"] * fields["input_size"] // fields["encoder_patch"]:
        raise ValueError(f"image_size {size} is not patch_size times the encoder's grid")
    return fields

_LAYER = {
    "norm1": "norm1",
    "layernorm_before": "norm1",
    "layer_norm1": "norm1",
    "norm2": "norm2",
    "layernorm_after": "norm2",
    "layer_norm2": "norm2",
    "attention.attention.query": "attention.query",
    "attention.attention.key": "attention.key",
    "attention.attention.value": "attention.value",
    "attention.output.dense": "attention.out",
    "self_attn.q_proj": "attention.query",
    "self_attn.k_proj": "attention.key",
    "self_attn.v_proj": "attention.value",
    "self_attn.out_proj": "attention.out",
    "attention.to_q": "attention.query",
    "attention.to_k": "attention.key",
    "attention.to_v": "attention.value",
    "attention.to_out.0": "attention.out",
    "mlp.fc1": "fc1",
    "intermediate.dense": "fc1",
    "mlp.fc2": "fc2",
    "output.dense": "fc2",
    "layer_scale1": "layer_scale1",
    "layer_scale2": "layer_scale2",
}
"""A layer's module, after its index, in each source's naming, and the port's."""

_LAYERS = {"encoder.encoder.layer": "encoder", "encoder.vision_model.encoder.layers": "encoder",
           "decoder.decoder_layers": "decoder"}

_MODULES = {
    "encoder.embeddings.patch_embeddings.projection": ("encoder", "patch"),
    "encoder.vision_model.embeddings.patch_embedding": ("encoder", "patch"),
    "encoder.layernorm": ("encoder", "norm"),
    "decoder.decoder_embed": ("decoder", "embed"), "decoder.decoder_norm": ("decoder", "norm"),
    "decoder.decoder_pred": ("decoder", "pred"),
}

_TENSORS = {
    "encoder.embeddings.cls_token": ("encoder", "cls_token"),
    "encoder.embeddings.register_tokens": ("encoder", "register_tokens"),
    "encoder.embeddings.position_embeddings": ("encoder", "position_embeddings"),
    "encoder.vision_model.embeddings.position_embedding.weight": ("encoder", "position_embeddings"),
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
            return (owner, f"layers_{index}", *renamed) if leaf == "lambda1" else (
                owner, f"layers_{index}", *renamed, leaf)
    raise ValueError(f"{name!r} is not an AutoencoderRAE tensor")


def _statistic(values, grid: int, width: int, default: float) -> np.ndarray | float:
    """A config's `latents_mean` or `latents_std`, channels last: absent, one
    number, or one per latent position `[C, grid, grid]`."""
    if values is None:
        return default
    array = np.asarray(values, np.float32)
    if array.size == 1:
        return float(array.reshape(()))
    if array.shape != (width, grid, grid):
        raise ValueError(f"latent statistics of shape {array.shape} are neither one number nor one per "
                         f"latent position {(width, grid, grid)}")
    return array.transpose(1, 2, 0)


class RAEAutoencoder(ModuleAutoEncoder[RAE]):
    """Native RAE weights and the source's latent normalization,
    `(z - latents_mean) / (latents_std + 1e-5) * scaling_factor` on the way
    out and its inverse on the way in, per latent position where the
    checkpoint gives per-position statistics."""

    def __init__(self, *, model: RAE, params: Variables, latents_mean=None, latents_std=None,
                 scaling_factor: float = 1.0):
        super().__init__(model, params)
        self.latent_shift = _statistic(latents_mean, model.grid, model.encoder_width, 0.0)
        self.latent_scale = scaling_factor / (
            _statistic(latents_std, model.grid, model.encoder_width, 1.0) + 1e-5
        )

    @property
    def downscale_factor(self) -> int:
        return self.model.patch

    @property
    def latent_channels(self) -> int:
        return self.model.encoder_width

    def latent_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        *lead, _, _, _ = shape
        return (*lead, self.model.grid, self.model.grid, self.latent_channels)


def load_rae(
    name_or_dir: str | Path,
    compute=jnp.float32,
    *,
    revision: str | None = None,
    subfolder: str = "",
    param_dtype: str = "float32",
    params: Variables | None = None,
) -> tuple[RAEAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published RAE, its parameters and their source layouts from
    `subfolder` of a directory or Hub repo, such as
    `nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08`. Every tensor the module
    declares must be published, in the shape it declares.

    Supplied `params` are bound unchanged: only the config is read, and no
    source layouts are returned."""
    directory, config = component_source(name_or_dir, revision, subfolder, weights=params is None)
    if config.get("_class_name") != "AutoencoderRAE":
        raise ValueError(
            f"{directory} holds a {config.get('_class_name')}, not an AutoencoderRAE"
        )
    model = RAE(**rae_fields(config), dtype=compute)
    image = jax.ShapeDtypeStruct((1, model.input_size, model.input_size, 3), jnp.float32)
    return bind_component(
        directory, "vae", config, model, rae_path,
        lambda bound: RAEAutoencoder(model=model, params=bound, latents_mean=config.get("latents_mean"),
                                     latents_std=config.get("latents_std"),
                                     scaling_factor=config.get("scaling_factor", 1.0)),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, inputs=(image,))


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
    from dew.interop import diffusion, sources, weights

    directory = sources.snapshot(str(name_or_dir), revision, weights=params is None)
    config = json.loads((directory / "config.json").read_text())
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
    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        tensors = diffusion.component_tensors(directory, "")

        def path_of(name: str) -> tuple[str, ...] | None:
            path = rae_path(f"encoder.{name}", np.ndim(tensors[name]))
            return None if path is None else path[1:]

        params, layouts = weights.record_layouts("dinov2", tensors, path_of, ("representation",),
                                                   param_dtype=param_dtype)
    check_tree({"params": params}, module,
               jax.ShapeDtypeStruct((1, module.input_size, module.input_size, 3), jnp.float32))
    return module, params, layouts
