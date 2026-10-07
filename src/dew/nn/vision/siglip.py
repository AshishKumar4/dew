"""The SigLIP vision trunk and Gemma 3's projector, with their checkpoint maps
(transformers 5.16.1 `models/siglip/modeling_siglip.py`, `models/gemma3/modeling_gemma3.py`).
"""

import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import records
from dew.interop.weights import checkpoint_array
from dew.nn.attention import LayerNorm, RMSNorm
from dew.nn.conv import Conv
from dew.nn.text_encoders import CLIPEncoderLayer, _encoder_layer_path
from dew.objectives.base import Variables
from dew.registry import Record

from .common import _PROJECTOR_PATHS, ProjectorBase, TowerBase, TowerGeometry, _image_size, _vision_section


class SiglipVisionTransformer(nn.Module):
    """The SigLIP vision trunk, param layout of `SiglipVisionConfig`.

    `pixel_values` are what the checkpoint's image processor emits: [B, C, H,
    W] at `image_size`, normalized. The sequence returned is the encoder
    output through the post norm. The attention pooling head some SigLIP
    checkpoints carry is not built; the Gemma path reads the sequence alone.
    """

    config: "SiglipVision"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        cfg = self.config
        if cfg.hidden_size % cfg.num_heads:
            raise ValueError(
                f"hidden_size ({cfg.hidden_size}) must split over num_heads "
                f"({cfg.num_heads})")
        patches = (cfg.image_size // cfg.patch_size) ** 2
        # torch Conv2d carries a bias unless told otherwise; the CLIP tower's
        # convolution is the bias-free exception, not the rule.
        self.patch_embedding = Conv(
            cfg.hidden_size, (cfg.patch_size, cfg.patch_size),
            strides=(cfg.patch_size, cfg.patch_size), padding="VALID",
            use_bias=True, dtype=self.dtype, precision=self.precision,
            name="patch_embedding")
        self.position_embedding = nn.Embed(patches, cfg.hidden_size,
                                           dtype=self.dtype, name="position_embedding")
        self.layers = [
            CLIPEncoderLayer(
                cfg.hidden_size, cfg.num_heads, cfg.intermediate_size, causal=False,
                layer_norm_eps=cfg.layer_norm_eps, dtype=self.dtype, precision=self.precision,
                activation=cfg.hidden_act, name=f"layers_{index}")
            for index in range(cfg.num_layers)]
        self.post_layernorm = LayerNorm(
            epsilon=cfg.layer_norm_eps, dtype=self.dtype, name="post_layernorm")

    def __call__(self, pixel_values) -> jax.Array:
        cfg = self.config
        pixel_values = jnp.asarray(pixel_values)
        batch, channels, height, width = pixel_values.shape
        expected = (cfg.num_channels, cfg.image_size, cfg.image_size)
        if (channels, height, width) != expected:
            raise ValueError(
                f"pixel_values of {channels}x{height}x{width} are not the "
                f"{'x'.join(map(str, expected))} this checkpoint was trained with")
        patches = self.patch_embedding(jnp.transpose(pixel_values, (0, 2, 3, 1)))
        hidden_states = patches.reshape(batch, -1, cfg.hidden_size)
        hidden_states = hidden_states + self.position_embedding(
            jnp.arange(hidden_states.shape[1]))
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)


@dataclasses.dataclass(frozen=True)
class SiglipVision(TowerBase):
    """A SigLIP trunk's geometry, under the reference's field names."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6

    def build(self) -> nn.Module:
        return SiglipVisionTransformer(self)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(image_size=self.image_size, patch_size=self.patch_size,
                             channels=self.num_channels)


class GemmaProjectorModule(nn.Module):
    """Patch features into soft tokens: block average, norm, map to text width.

    The reference reshapes the trunk sequence into its patch grid, averages
    each kernel block, norms and multiplies by its (vision, text) matrix
    (modeling_gemma3.py, Gemma3MultiModalProjector).
    """

    text_width: int
    patches_per_side: int
    tokens_per_side: int
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if self.patches_per_side % self.tokens_per_side:
            raise ValueError(
                f"patches_per_side ({self.patches_per_side}) must split over "
                f"tokens_per_side ({self.tokens_per_side})")
        self.mm_soft_emb_norm = RMSNorm(
            epsilon=self.norm_eps, scale_offset=True, dtype=self.dtype,
            name="mm_soft_emb_norm")
        self.mm_input_projection = nn.Dense(
            self.text_width, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="mm_input_projection")

    def __call__(self, vision_outputs) -> jax.Array:
        batch, length, width = vision_outputs.shape
        patches, tokens = self.patches_per_side, self.tokens_per_side
        if length != patches * patches:
            raise ValueError(
                f"{length} patch features are not the {patches}x{patches} grid "
                "this projector pools")
        kernel = patches // tokens
        grid = vision_outputs.reshape(batch, patches, patches, width)
        pooled = grid.reshape(batch, tokens, kernel, tokens, kernel, width).mean(axis=(2, 4))
        return self.mm_input_projection(
            self.mm_soft_emb_norm(pooled.reshape(batch, tokens * tokens, width)))


@dataclasses.dataclass(frozen=True)
class GemmaProjector(ProjectorBase):
    """Gemma's projector fields: the decoder width and the patch grid pooled
    into the soft-token grid."""

    text_width: int
    patches_per_side: int
    tokens_per_side: int
    norm_eps: float = 1e-6

    def build(self) -> nn.Module:
        return GemmaProjectorModule(
            text_width=self.text_width, patches_per_side=self.patches_per_side,
            tokens_per_side=self.tokens_per_side, norm_eps=self.norm_eps)


_SIGLIP_TENSORS = {
    "embeddings.patch_embedding.weight": ("patch_embedding", "kernel"),
    "embeddings.patch_embedding.bias": ("patch_embedding", "bias"),
    "embeddings.position_embedding.weight": ("position_embedding", "embedding"),
    "post_layernorm.weight": ("post_layernorm", "scale"),
    "post_layernorm.bias": ("post_layernorm", "bias"),
}


def siglip_vision_path(hf_name: str) -> tuple[str, ...] | None:
    """One SigLIP vision tensor name into its path in a trunk tree.

    position_ids is an arange buffer, not a parameter. The attention pooling
    head some checkpoints carry maps to nothing: the Gemma path reads the
    trunk sequence alone. Anything else unknown raises ValueError.
    """
    if hf_name == "embeddings.position_ids" or hf_name.split(".")[0] == "head":
        return None
    path = _SIGLIP_TENSORS.get(hf_name) or _encoder_layer_path(
        hf_name.split("."), "encoder", ("layer_norm1", "layer_norm2"),
        ("q_proj", "k_proj", "v_proj", "out_proj"))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_gemma_projector_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """A Gemma projector's two tensors into its parameter tree.

    The norm's weight becomes its scale; the projection matrix is a plain
    parameter the reference multiplies as is, so unlike a Linear kernel it
    keeps its layout.
    """
    known = set(_PROJECTOR_PATHS["gemma"])
    unknown = sorted(set(hf_tensors) - known)
    if unknown:
        raise ValueError(f"unknown tensor names {unknown}")
    return {module: {leaf: np.ascontiguousarray(checkpoint_array(hf_tensors[name], param_dtype))}
            for name, (module, leaf) in _PROJECTOR_PATHS["gemma"].items()}


def translate_siglip_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A SiglipVisionConfig into a SiglipVision value's fields.

    Reads the vision_config of a multimodal wrapper or a bare vision config.
    Only square images map; anything but tanh or exact gelu refuses with its
    name, and a nonzero dropout refuses as training-only.
    """
    vision = _vision_section(hf_config)
    hidden = records.integer(vision["hidden_size"], "hidden_size")
    patch = vision.get("patch_size", 16)
    if isinstance(patch, (list, tuple)):
        patch = patch[0]
    activation = str(vision.get("hidden_act", vision.get("hidden_activation",
                                                         "gelu_pytorch_tanh")))
    if activation not in ("gelu_pytorch_tanh", "gelu"):
        raise ValueError(
            f"hidden_act {activation!r} is not expressible: this trunk runs tanh or "
            "exact gelu")
    if records.number(vision.get("attention_dropout", 0.0), "attention_dropout"):
        raise ValueError("attention_dropout is training-time; this trunk runs eval")
    return {
        "class": "siglip", "fields": {
        "hidden_size": hidden,
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "num_heads": records.integer(vision["num_attention_heads"], "num_attention_heads"),
        "image_size": _image_size(vision.get("image_size", 224), "image_size"),
        "patch_size": records.integer(patch, "patch_size"),
        "num_channels": records.integer(vision.get("num_channels", 3), "num_channels"),
        "hidden_act": activation,
        "layer_norm_eps": records.number(vision.get("layer_norm_eps", 1e-6), "layer_norm_eps"),
    }}


def translate_gemma_projector_config(vision: Mapping[str, object], text_width: int,
                                     mm_tokens_per_image: object) -> Record:
    """A Gemma wrapper's projector fields: decoder width, grids."""
    if isinstance(mm_tokens_per_image, bool) or not isinstance(mm_tokens_per_image, int):
        raise ValueError(
            f"mm_tokens_per_image is {mm_tokens_per_image!r}, the soft-token count "
            "is an int")
    side = int(mm_tokens_per_image ** 0.5)
    if side * side != mm_tokens_per_image:
        raise ValueError(
            f"mm_tokens_per_image ({mm_tokens_per_image}) is not a square, this "
            "projector pools a grid into a grid")
    patches = records.integer(vision["image_size"], "image_size") // records.integer(
        vision["patch_size"], "patch_size"
    )
    if patches % side:
        raise ValueError(
            f"{patches} patches per side do not split over {side} soft tokens per side")
    return {
        "class": "gemma", "fields": {
        "text_width": int(text_width),
        "patches_per_side": patches,
        "tokens_per_side": side,
        "norm_eps": records.number(vision.get("layer_norm_eps", 1e-6), "layer_norm_eps"),
    }}
