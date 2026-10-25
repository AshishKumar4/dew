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

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype

from dew import _records
from dew.batch import Variables
from dew.interop.components import bind_component, component_source
from dew.nn.activations import activation
from dew.nn.attention import LayerNorm
from dew.nn.backbones.sd3 import sincos_position
from dew.nn.blocks import IMAGENET_MEAN, IMAGENET_STD, torch_bicubic_resize
from dew.nn.dense import Dense
from dew.nn.precision import float32_dot_general
from dew.nn.text_encoders import clip_layer
from dew.nn.vision.vit import LearnedPositions, SpecialTokens, VisionTransformer, ViTLayout

from .api import LatentNormalization, ModuleAutoEncoder, RecordValue, published_source

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

HEAD_WIDTH = 64
"""The source's divisor for deriving the number of attention heads from its encoder width."""


PUBLISHED = ("AutoencoderRAE",)
"""The `_class_name` a published AutoencoderRAE's config states."""


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


class RepresentationEncoder(nn.Module):
    """A frozen encoder: normalized `[B, S, S, 3]` pixels at its input size
    to its patch tokens `[B, grid, grid, width]`."""

    kind: str
    width: int
    layers: int
    patch: int
    input_size: int
    dtype: Dtype = jnp.float32

    @nn.nowrap
    def layout(self) -> ViTLayout:
        spec = ENCODERS[self.kind]
        heads = self.width // HEAD_WIDTH
        return ViTLayout(
            depth=self.layers, hidden_size=self.width, num_heads=heads,
            num_kv_heads=heads, head_dim=self.width // heads,
            intermediate_size=spec.intermediate or 4 * self.width, activation=activation(spec.activation),
            in_channels=3, patch_size=self.patch, temporal_patch_size=1, spatial_merge_size=1,
            norm_type="fast_layer", norm_eps=spec.epsilon, norm_bias=True, mlp_bias=True,
            packed=False, patch_bias=True, attention=("full",) * self.layers,
            order=("patches",) * self.layers, sinks=(False,) * self.layers,
            positions=LearnedPositions(spec.table_size // self.patch, "bicubic",
                                       singleton=spec.leading > 0, compute_dtype=False,
                                       antialias=spec.antialias, initializer=nn.initializers.zeros),
            special_tokens=SpecialTokens("first" if spec.leading else None, max(0, spec.leading - 1),
                                         singleton=True, compute_dtype=False),
            patch_projection="conv", image_size=self.input_size, rotary=None, post_norm="layer",
            post_norm_affine=spec.final_affine, outer_norm_eps=spec.epsilon,
            output="patch_grid", branch_scale=1.0 if spec.layer_scale else None, merge_blocks=False)

    def setup(self):
        self.trunk = VisionTransformer(self.layout(), dtype=self.dtype)
        nn.share_scope(self, self.trunk)

    def __call__(self, pixels):
        return self.trunk(jnp.asarray(pixels).transpose(0, 3, 1, 2))


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
        tokens = Dense(self.width, dtype=self.dtype, name="embed",
                       dot_general=float32_dot_general)(
            latents.reshape(batch, -1, latents.shape[-1])
        )
        cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.width))
        tokens = jnp.concatenate(
            [jnp.broadcast_to(cls_token, (batch, 1, self.width)).astype(tokens.dtype), tokens], axis=1
        )
        positions = sincos_position(self.width, self.grid, base_size=16)
        tokens = tokens + jnp.concatenate([jnp.zeros((1, 1, self.width)), positions], axis=1)
        for index in range(self.layers):
            tokens = clip_layer(self.width, self.heads, self.intermediate, self.grid ** 2 + 1,
                                eps=1e-12, activation=activation("gelu"), dtype=self.dtype,
                                name=f"blocks_{index}")(tokens)
        tokens = LayerNorm(epsilon=1e-12, dtype=self.dtype, name="norm")(tokens)
        patches = Dense(self.patch * self.patch * self.channels, dtype=self.dtype, name="pred",
                        dot_general=float32_dot_general)(
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
    pixel_mean: tuple[float, ...] = IMAGENET_MEAN
    pixel_std: tuple[float, ...] = IMAGENET_STD
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

    def encode(self, images, key=None, noise=None):
        """The encoder's tokens. An RAE is deterministic: `key` is unread, and a posterior's `noise`
        raises TypeError, since it has none to draw."""
        del key
        if noise is not None:
            raise TypeError(f"{type(self).__name__} encodes deterministically and draws no posterior noise")
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
        pixel_mean=per_channel("encoder_norm_mean", IMAGENET_MEAN),
        pixel_std=per_channel("encoder_norm_std", IMAGENET_STD),
    )
    if fields["input_size"] % fields["encoder_patch"]:
        raise ValueError(f"encoder_input_size {fields['input_size']} is not a multiple of "
                         f"encoder_patch_size {fields['encoder_patch']}")
    size = config.get("image_size")
    if size is not None and size != fields["patch"] * fields["input_size"] // fields["encoder_patch"]:
        raise ValueError(f"image_size {size} is not patch_size times the encoder's grid")
    return fields

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

    @property
    def stochastic(self) -> bool:
        return False

    def __init__(self, *, model: RAE, params: Variables, latents_mean=None, latents_std=None,
                 scaling_factor: float = 1.0):
        super().__init__(model, params)
        # The record keeps the checkpoint's statistics as given, in the source's layout.
        record: dict[str, RecordValue] = {
            name: None if value is None else np.asarray(value, np.float32)
            for name, value in (('latents_mean', latents_mean), ('latents_std', latents_std))}
        record['scaling_factor'] = scaling_factor
        self.normalization = LatentNormalization(
            record, _statistic(latents_mean, model.grid, model.encoder_width, 0.0),
            scaling_factor / (_statistic(latents_std, model.grid, model.encoder_width, 1.0) + 1e-5))

    def record_arguments(self) -> dict:
        return dict(self.normalization.record)

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
    lazy: bool = False,
    shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[RAEAutoencoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build a published RAE, its parameters and their source layouts from
    `subfolder` of a directory or Hub repo, such as
    `nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08`. Every tensor the module
    declares must be published, in the shape it declares.

    Supplied `params` are bound unchanged: only the config is read, and no
    source layouts are returned. `lazy` leaves read ones `SourceLeaf`s for a
    placement to read (`dew.interop.weights.record_layouts`)."""
    directory, config = published_source(name_or_dir, revision, subfolder, PUBLISHED, weights=params is None)
    model = RAE(**rae_fields(config), dtype=compute)
    image = jax.ShapeDtypeStruct((1, model.input_size, model.input_size, 3), jnp.float32)
    return bind_component(
        directory, "vae", config, model, rae_path,
        lambda bound: RAEAutoencoder(model=model, params=bound, latents_mean=config.get("latents_mean"),
                                     latents_std=config.get("latents_std"),
                                     scaling_factor=config.get("scaling_factor", 1.0)),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy, inputs=(image,),
        shapes=shapes)


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
