"""The Llama 4 vision trunk and its projector, with their checkpoint maps
(transformers 5.16.1 `models/llama4/modeling_llama4.py`).
"""

import dataclasses
import functools
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import records
from dew.nn.activations import gelu_exact
from dew.nn.attention import LayerNorm
from dew.nn.inputs import AttentionMetadata
from dew.nn.text_encoders import _encoder_layer_path, clip_layer
from dew.registry import Record

from .common import ProjectorBase, TowerBase, TowerGeometry, _image_size, _vision_section


def pixel_shuffle(patches: jax.Array, ratio: float) -> jax.Array:
    """Space to depth: each ratio-by-ratio block becomes one thicker token.

    The inverse of a pixel shuffle at the same ratio (modeling_llama4.py,
    pixel_shuffle): tokens shrink by ratio squared and channels grow by it.
    """
    batch, count, channels = patches.shape
    side = round(count ** 0.5)
    grown = round(channels / ratio ** 2)
    if abs(grown * ratio ** 2 - channels) > 1e-6:
        raise ValueError(
            f"{channels} channels do not split over a shuffle ratio of {ratio}")
    grid = patches.reshape(batch, side, side, channels)
    block = round(1 / ratio)
    shuffled = grid.reshape(batch, side // block, block, side // block, block, channels)
    shuffled = shuffled.transpose(0, 1, 3, 2, 4, 5)
    return shuffled.reshape(batch, (side // block) ** 2, grown)


class Llama4VisionAdapterMLP(nn.Module):
    """The tower's own projector: no-bias maps around an exact GELU."""

    input_dim: int
    output_dim: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, use_bias=False,
                                  dtype=self.dtype, precision=self.precision)
        self.fc1 = dense(self.input_dim, name="fc1")
        self.fc2 = dense(self.output_dim, name="fc2")

    def __call__(self, hidden_states):
        # The reference gels after both maps, including the last one
        # (modeling_llama4.py, Llama4VisionMLP2.forward).
        return gelu_exact(self.fc2(gelu_exact(self.fc1(hidden_states))))


class Llama4VisionAdapter(nn.Module):
    """Pixel shuffle into the adapter MLP, the tower's last stage."""

    ratio: float
    input_dim: int
    output_dim: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.mlp = Llama4VisionAdapterMLP(
            self.input_dim, self.output_dim, dtype=self.dtype,
            precision=self.precision, name="mlp")

    def __call__(self, encoded_patches) -> jax.Array:
        return self.mlp(pixel_shuffle(encoded_patches, self.ratio))


class Llama4VisionTransformer(nn.Module):
    """The Llama 4 vision trunk, param layout of `Llama4VisionConfig`.

    `pixel_values` are what the checkpoint's image processor emits: [B, C, H,
    W] at `image_size`. The patches embed through one bias-free map, the class
    token rides last through the trunk and is dropped before the adapter, and
    what returns is the pixel-shuffled MLP output the outer projector maps to
    text width.
    """

    config: "Llama4Vision"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        cfg = self.config
        if cfg.hidden_size % cfg.num_heads:
            raise ValueError(
                f"hidden_size ({cfg.hidden_size}) must split over num_heads "
                f"({cfg.num_heads})")
        grid = cfg.image_size // cfg.patch_size
        if grid * cfg.patch_size != cfg.image_size:
            raise ValueError(
                f"image_size ({cfg.image_size}) must tile patch_size "
                f"({cfg.patch_size})")
        self.patch_embedding = nn.Dense(
            cfg.hidden_size, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="patch_embedding")
        self.class_embedding = self.param(
            "class_embedding", nn.initializers.normal(cfg.hidden_size ** -0.5),
            (cfg.hidden_size,))
        self.positional_embedding = self.param(
            "positional_embedding", nn.initializers.normal(cfg.hidden_size ** -0.5),
            (grid * grid + 1, cfg.hidden_size))
        norm = functools.partial(LayerNorm, epsilon=cfg.layer_norm_eps,
                                 dtype=self.dtype)
        self.layernorm_pre = norm(name="layernorm_pre")
        # The grid's rotary turns adjacent pairs, the x angles over the first
        # half of each head and the y angles over the second
        # (modeling_llama4.py, Llama4VisionRotaryEmbedding).
        side = cfg.hidden_size // cfg.num_heads // 2
        self.layers = [
            clip_layer(cfg.hidden_size, cfg.num_heads, cfg.intermediate_size, grid * grid + 1,
                       activation="gelu", eps=cfg.layer_norm_eps, rotary_axes=(side, side),
                       rotary_pairs="adjacent", rope_theta=cfg.rope_theta, dtype=self.dtype,
                       precision=self.precision, name=f"layers_{index}")
            for index in range(cfg.num_layers)]
        self.layernorm_post = norm(name="layernorm_post")
        self.vision_adapter = Llama4VisionAdapter(
            cfg.pixel_shuffle_ratio, cfg.projector_input_dim, cfg.projector_output_dim,
            dtype=self.dtype, precision=self.precision, name="vision_adapter")

    def __call__(self, pixel_values) -> jax.Array:
        cfg = self.config
        pixel_values = jnp.asarray(pixel_values)
        batch, channels, height, width = pixel_values.shape
        expected = (cfg.num_channels, cfg.image_size, cfg.image_size)
        if (channels, height, width) != expected:
            raise ValueError(
                f"pixel_values of {channels}x{height}x{width} are not the "
                f"{'x'.join(map(str, expected))} this checkpoint was trained with")
        grid = cfg.image_size // cfg.patch_size
        patches = pixel_values.reshape(
            batch, channels, grid, cfg.patch_size, grid, cfg.patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        hidden_states = self.patch_embedding(
            patches.reshape(batch, grid * grid, channels * cfg.patch_size ** 2))
        class_token = jnp.broadcast_to(
            self.class_embedding.astype(hidden_states.dtype), (batch, 1, cfg.hidden_size))
        hidden_states = jnp.concatenate([hidden_states, class_token], axis=1)
        hidden_states = hidden_states + self.positional_embedding.astype(hidden_states.dtype)
        hidden_states = self.layernorm_pre(hidden_states)
        # Each patch sits at its column and row plus one, row-major, and the
        # class token last at the origin, which turns by nothing
        # (Llama4VisionRotaryEmbedding._compute_freqs_ci).
        index = jnp.arange(grid * grid)
        positions = jnp.concatenate([jnp.stack([index % grid + 1, index // grid + 1], axis=-1),
                                     jnp.zeros((1, 2), index.dtype)])
        metadata = AttentionMetadata(rotary_positions=positions[None])
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_metadata=metadata)
        hidden_states = self.layernorm_post(hidden_states)[:, :-1, :]
        return self.vision_adapter(hidden_states)


@dataclasses.dataclass(frozen=True)
class Llama4Vision(TowerBase):
    """A Llama 4 trunk's geometry, under the reference's field names."""

    hidden_size: int = 1408
    intermediate_size: int = 5632
    num_layers: int = 32
    num_heads: int = 16
    image_size: int = 336
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    pixel_shuffle_ratio: float = 0.5
    projector_input_dim: int = 4096
    projector_output_dim: int = 4096

    def build(self) -> nn.Module:
        return Llama4VisionTransformer(self)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(image_size=self.image_size, patch_size=self.patch_size,
                             channels=self.num_channels)


class Llama4ProjectorModule(nn.Module):
    """Tower output to text width: one bias-free map."""

    text_width: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.linear = nn.Dense(
            self.text_width, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="linear")

    def __call__(self, image_features) -> jax.Array:
        return self.linear(image_features)


@dataclasses.dataclass(frozen=True)
class Llama4Projector(ProjectorBase):
    """Llama 4's projector fields: the text width."""

    text_width: int

    def build(self) -> nn.Module:
        return Llama4ProjectorModule(text_width=self.text_width)


# Llama 4 names its layers' norms and maps as the decoder block does.
_LAYER_NORMS = {name: name for name in ("input_layernorm", "post_attention_layernorm")}
_LAYER_PROJECTIONS = {name: name for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
_LLAMA4_VISION_TENSORS = {
    "patch_embedding.linear.weight": ("patch_embedding", "kernel"),
    "class_embedding": ("class_embedding",),
    "positional_embedding_vlm": ("positional_embedding",),
    "layernorm_pre.weight": ("layernorm_pre", "scale"),
    "layernorm_pre.bias": ("layernorm_pre", "bias"),
    "layernorm_post.weight": ("layernorm_post", "scale"),
    "layernorm_post.bias": ("layernorm_post", "bias"),
    "vision_adapter.mlp.fc1.weight": ("vision_adapter", "mlp", "fc1", "kernel"),
    "vision_adapter.mlp.fc2.weight": ("vision_adapter", "mlp", "fc2", "kernel"),
}


def llama4_vision_path(hf_name: str) -> tuple[str, ...] | None:
    """One Llama 4 vision tensor name into its path in a trunk tree."""
    path = _LLAMA4_VISION_TENSORS.get(hf_name) or _encoder_layer_path(
        hf_name.split("."), "model", _LAYER_NORMS, _LAYER_PROJECTIONS)
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_llama4_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A Llama4VisionConfig into a Llama4Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The feature
    selection defaults to the last layer; the concatenated-layer strategy has
    no counterpart and refuses. Rope reads the nested spelling transformers
    writes or the flat theta the released Scout carries.
    """
    vision = _vision_section(hf_config)
    if vision.get("vision_feature_select_strategy", "default") != "default":
        raise ValueError(
            f"vision_feature_select_strategy "
            f"{vision.get('vision_feature_select_strategy')!r} concatenates layers "
            "this trunk never builds")
    if vision.get("vision_feature_layer", -1) not in (-1, None):
        raise ValueError(
            f"vision_feature_layer {vision.get('vision_feature_layer')!r} reads a "
            "middle layer this trunk never returns")
    rope = records.record(vision.get("rope_parameters") or {}, "rope_parameters")
    theta = records.number(rope.get("rope_theta", vision.get("rope_theta", 10000.0)), "rope_theta")
    if records.number(vision.get("attention_dropout", 0.0), "attention_dropout") or records.number(
        vision.get("projector_dropout", 0.0), "projector_dropout"
    ):
        raise ValueError("attention_dropout/projector_dropout is training-time")
    if vision.get("multi_modal_projector_bias", False):
        raise ValueError("multi_modal_projector_bias=True needs a projector bias this map lacks")
    output_dim = records.integer(
        vision.get("vision_output_dim", vision.get("projector_output_dim", 0)), "vision_output_dim"
    )
    if output_dim != records.integer(vision.get("projector_output_dim", output_dim), "projector_output_dim"):
        raise ValueError(
            f"vision_output_dim ({output_dim}) disagrees with projector_output_dim "
            f"({vision.get('projector_output_dim')}), the adapter's width")
    return {
        "class": "llama4", "fields": {
        "hidden_size": records.integer(vision["hidden_size"], "hidden_size"),
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "num_heads": records.integer(vision["num_attention_heads"], "num_attention_heads"),
        "image_size": _image_size(vision.get("image_size", 336), "image_size"),
        "patch_size": records.integer(vision.get("patch_size", 14), "patch_size"),
        "num_channels": records.integer(vision.get("num_channels", 3), "num_channels"),
        "layer_norm_eps": records.number(
            vision.get("norm_eps", vision.get("layer_norm_eps", 1e-5)), "norm_eps"
        ),
        "rope_theta": theta,
        "pixel_shuffle_ratio": records.number(vision.get("pixel_shuffle_ratio", 0.5), "pixel_shuffle_ratio"),
        "projector_input_dim": records.integer(vision["projector_input_dim"], "projector_input_dim"),
        "projector_output_dim": records.integer(vision["projector_output_dim"], "projector_output_dim"),
    }}


def translate_llama4_projector_config(text_width: int) -> Record:
    """A Llama 4 wrapper's projector fields: the text width."""
    return {"class": "llama4", "fields": {"text_width": int(text_width)}}
