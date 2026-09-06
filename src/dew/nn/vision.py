"""Vision towers, projectors and reference checkpoint maps.

transformers 5 ships no Flax classes, so the towers are vendored the way
`dew/nn/text_encoders.py` vendors CLIP, in the reference layout, with each
weight read from the checkpoint's safetensors under its reference tensor name.
The operation order follows transformers 5.16.1
`models/siglip/modeling_siglip.py`, `models/llama4/modeling_llama4.py`,
`models/gemma4/modeling_gemma4.py` and `models/qwen3_5/modeling_qwen3_5.py`.

The SigLIP trunk is patch convolution with bias, learned position embeddings
with no class token, pre-norm encoder blocks and a post layer norm. Its block
shares both halves with CLIP's: the attention and the feed-forward
(`dew.nn.text_encoders`), the MLP carrying the config's activation. The Llama 4
trunk is MetaCLIP-style: an unfold patch embedding without bias, a class token
appended after the patches, learned positions, a pre norm, full-attention
blocks with a complex rotary over the patch grid, a post norm, the class
token dropped, and the pixel-shuffle MLP inside the tower where the reference
keeps it. The Gemma 4 trunk is patch pixels scaled to [-1, 1] through a
bias-free map, summed 2D position tables, RMS-normed blocks with a 2D rotary
and gated feed-forwards, and a position pooler with standardization. The
Qwen 3.5 trunk is a NaViT-style patchify with the still frame repeated along
time, interpolated learned positions with a 2D rotary, full-attention blocks
and the merger MLP as its projector. Each tower's projector is a registered
value beside it: Gemma 3's averages each patch block, norms and maps to the
decoder width, Llama 4's maps the shuffled output to the decoder width,
Gemma 4's norms without a scale and maps, and Qwen 3.5's is the merger.
Gemma 3n uses the MobileNet-v5 encoder in ``dew.nn.mobilenet`` and the hard/soft
vision embedder defined here.
"""

import dataclasses
import functools
from typing import Any, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm, scaled_dot_product_attention
from dew.nn.text_encoders import CLIPAttention, MLP
from dew.registry import Registry
from .mobilenet import MobileNetV5Encoder

PIXEL_VALUES_KEY = "pixel_values"
"""The batch field carrying images as the checkpoint's processor emitted them."""


projectors: Registry[type] = Registry("projector")
towers: Registry[type] = Registry("tower")


class ProjectorBase:
    """One projector kind's value: its fields, and how it builds its module.

    Each kind is a frozen dataclass of the reference's field names, registered
    under its name (`@projectors("gemma")`). `build` turns the value into the
    Flax module. A record without a kind, or one naming nothing registered,
    raises ValueError.
    """

    def build(self) -> nn.Module:
        """The Flax module for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} names a projector kind but builds no module")


class TowerBase:
    """One tower kind's value: its fields, and how it builds its module."""

    def build(self) -> nn.Module:
        """The Flax module for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} names a tower kind but builds no module")


def projector_from_record(record: Mapping[str, object]) -> ProjectorBase:
    """A `{"kind": ..., ...fields}` record as the kind value it names."""
    fields = dict(record)
    try:
        kind = fields.pop("kind")
    except KeyError:
        raise ValueError(
            f"a projector record names its kind, got {sorted(fields)}; known: "
            f"{', '.join(sorted(projectors))}") from None
    if not isinstance(kind, str):
        raise ValueError(
            f"a projector kind is a registered name, not {kind!r}; known: "
            f"{', '.join(sorted(projectors))}")
    try:
        built = projectors.build(kind, **fields)
    except KeyError:
        raise ValueError(
            f"a projector kind is a registered name, not {kind!r}; known: "
            f"{', '.join(sorted(projectors))}") from None
    if not isinstance(built, ProjectorBase):
        raise ValueError(
            f"projector {kind!r} built {type(built).__name__}, which is not a "
            "projector value")
    return built


def tower_from_record(record: Mapping[str, object]) -> TowerBase:
    """A `{"kind": ..., ...fields}` record as the kind value it names."""
    fields = dict(record)
    try:
        kind = fields.pop("kind")
    except KeyError:
        raise ValueError(
            f"a tower record names its kind, got {sorted(fields)}; known: "
            f"{', '.join(sorted(towers))}") from None
    if not isinstance(kind, str):
        raise ValueError(
            f"a tower kind is a registered name, not {kind!r}; known: "
            f"{', '.join(sorted(towers))}")
    try:
        built = towers.build(kind, **fields)
    except KeyError:
        raise ValueError(
            f"a tower kind is a registered name, not {kind!r}; known: "
            f"{', '.join(sorted(towers))}") from None
    if not isinstance(built, TowerBase):
        raise ValueError(
            f"tower {kind!r} built {type(built).__name__}, which is not a "
            "tower value")
    return built


class SiglipEncoderLayer(nn.Module):
    """Pre-norm full attention over pre-norm MLP, both residual.

    The structure matches a CLIP encoder layer; the attention is shared and
    the MLP carries the SigLIP activation.
    """

    hidden_size: int
    num_heads: int
    intermediate_size: int
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(nn.LayerNorm, epsilon=self.layer_norm_eps,
                                 dtype=self.dtype)
        self.layer_norm1 = norm(name="layer_norm1")
        self.self_attn = CLIPAttention(
            self.hidden_size, self.num_heads, causal=False, dtype=self.dtype,
            precision=self.precision, name="self_attn")
        self.layer_norm2 = norm(name="layer_norm2")
        self.mlp = MLP(self.hidden_size, self.intermediate_size,
                       activation=self.hidden_act, dtype=self.dtype,
                       precision=self.precision, name="mlp")

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states))
        return residual + hidden_states


class SiglipVisionTransformer(nn.Module):
    """The SigLIP vision trunk, param layout of `SiglipVisionConfig`.

    `pixel_values` are what the checkpoint's image processor emits: [B, C, H,
    W] at `image_size`, normalized. The sequence returned is the encoder
    output through the post norm. The attention pooling head some SigLIP
    checkpoints carry is not built; the Gemma path reads the sequence alone.
    """

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.hidden_size % self.num_heads:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must split over num_heads "
                f"({self.num_heads})")
        patches = (self.image_size // self.patch_size) ** 2
        # torch Conv2d carries a bias unless told otherwise; the CLIP tower's
        # convolution is the bias-free exception, not the rule.
        self.patch_embedding = nn.Conv(
            self.hidden_size, (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size), padding="VALID",
            use_bias=True, dtype=self.dtype, precision=self.precision,
            name="patch_embedding")
        self.position_embedding = nn.Embed(patches, self.hidden_size,
                                           dtype=self.dtype, name="position_embedding")
        self.layers = [
            SiglipEncoderLayer(
                self.hidden_size, self.num_heads, self.intermediate_size,
                self.hidden_act, layer_norm_eps=self.layer_norm_eps,
                dtype=self.dtype, precision=self.precision, name=f"layers_{index}")
            for index in range(self.num_layers)]
        self.post_layernorm = nn.LayerNorm(
            epsilon=self.layer_norm_eps, dtype=self.dtype, name="post_layernorm")

    def __call__(self, pixel_values) -> jax.Array:
        pixel_values = jnp.asarray(pixel_values)
        batch, channels, height, width = pixel_values.shape
        expected = (self.num_channels, self.image_size, self.image_size)
        if (channels, height, width) != expected:
            raise ValueError(
                f"pixel_values of {channels}x{height}x{width} are not the "
                f"{'x'.join(map(str, expected))} this checkpoint was trained with")
        patches = self.patch_embedding(jnp.transpose(pixel_values, (0, 2, 3, 1)))
        hidden_states = patches.reshape(batch, -1, self.hidden_size)
        hidden_states = hidden_states + self.position_embedding(
            jnp.arange(hidden_states.shape[1]))
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)


@towers("siglip")
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
        return SiglipVisionTransformer(
            hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
            num_layers=self.num_layers, num_heads=self.num_heads,
            image_size=self.image_size, patch_size=self.patch_size,
            num_channels=self.num_channels, hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps)


def _llama4_vision_rope(values: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """The complex rotation on real pairs, one shared angle per pair."""
    pairs = values.reshape(*values.shape[:-1], -1, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    table, turn = cos[:, None, :], sin[:, None, :]
    rotated = jnp.stack([first * table - second * turn,
                         first * turn + second * table], axis=-1)
    return rotated.reshape(values.shape)


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
    dtype: Optional[Dtype] = None
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


@projectors("gemma")
@dataclasses.dataclass(frozen=True)
class GemmaProjector(ProjectorBase):
    """Gemma's projector fields: the trunk width, the decoder width, and the
    patch grid pooled into the soft-token grid."""

    vision_width: int
    text_width: int
    patches_per_side: int
    tokens_per_side: int
    norm_eps: float = 1e-6

    def build(self) -> nn.Module:
        return GemmaProjectorModule(
            text_width=self.text_width, patches_per_side=self.patches_per_side,
            tokens_per_side=self.tokens_per_side, norm_eps=self.norm_eps)


def _llama4_vision_tables(grid: int, head_dim: int, theta: float) -> Tuple[jax.Array, jax.Array]:
    """The complex rotary tables of the Llama 4 vision attention, as cos/sin.

    Positions are the patch grid in row-major order with the class token last
    (modeling_llama4.py, Llama4VisionRotaryEmbedding._compute_freqs_ci): the
    x angles cover the first half of each head and the y angles the second,
    each pair sharing one angle, and the class row is the identity. Writing
    the doubled values out directly reads the same angles the reference's
    interleave and stride do.
    """
    positions = jnp.arange(grid * grid + 1, dtype=jnp.int32)
    kinds = jnp.where(positions == grid * grid, -2, positions)
    safe = jnp.where(kinds < 0, 0, kinds)
    freq_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, freq_dim, 2, dtype=jnp.float32)
                                / freq_dim))
    angles = jnp.concatenate([(safe % grid + 1)[:, None] * inv_freq[None, :],
                              (safe // grid + 1)[:, None] * inv_freq[None, :]], axis=1)
    angles = jnp.where((kinds < 0)[:, None], 0.0, angles)
    return jnp.cos(angles), jnp.sin(angles)



class Llama4VisionAttention(nn.Module):
    """Biased multi-head attention with the grid rotary on queries and keys."""

    hidden_size: int
    num_heads: int
    grid: int
    rope_theta: float = 10000.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, use_bias=True,
                                  dtype=self.dtype, precision=self.precision)
        self.q_proj = dense(self.hidden_size, name="q_proj")
        self.k_proj = dense(self.hidden_size, name="k_proj")
        self.v_proj = dense(self.hidden_size, name="v_proj")
        self.o_proj = dense(self.hidden_size, name="o_proj")

    def __call__(self, hidden_states) -> jax.Array:
        batch, length, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads
        heads = (batch, length, self.num_heads, head_dim)
        query = self.q_proj(hidden_states).reshape(heads)
        key = self.k_proj(hidden_states).reshape(heads)
        value = self.v_proj(hidden_states).reshape(heads)
        cos, sin = _llama4_vision_tables(self.grid, head_dim, self.rope_theta)
        query = _llama4_vision_rope(query, cos, sin)
        key = _llama4_vision_rope(key, cos, sin)
        attended = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision)
        return self.o_proj(attended.reshape(batch, length, self.hidden_size))


class Llama4VisionEncoderLayer(nn.Module):
    """Pre-norm attention over pre-norm MLP, both residual."""

    hidden_size: int
    num_heads: int
    intermediate_size: int
    grid: int
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(nn.LayerNorm, epsilon=self.layer_norm_eps,
                                 dtype=self.dtype)
        self.input_layernorm = norm(name="input_layernorm")
        self.self_attn = Llama4VisionAttention(
            self.hidden_size, self.num_heads, self.grid, self.rope_theta,
            dtype=self.dtype, precision=self.precision, name="self_attn")
        self.post_attention_layernorm = norm(name="post_attention_layernorm")
        self.mlp = MLP(self.hidden_size, self.intermediate_size,
                       activation="gelu", dtype=self.dtype,
                       precision=self.precision, name="mlp")

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


def pixel_shuffle(patches: jax.Array, ratio: float) -> jax.Array:
    """Space to depth: each ratio-by-ratio block becomes one thicker token.

    The inverse of a pixel shuffle at the same ratio (modeling_llama4.py,
    pixel_shuffle): tokens shrink by ratio squared and channels grow by it.
    """
    batch, count, channels = patches.shape
    side = int(round(count ** 0.5))
    if side * side != count:
        raise ValueError(
            f"{count} patches are not a square grid, so no shuffle ratio tiles them")
    grown = int(round(channels / ratio ** 2))
    if abs(grown * ratio ** 2 - channels) > 1e-6:
        raise ValueError(
            f"{channels} channels do not split over a shuffle ratio of {ratio}")
    grid = patches.reshape(batch, side, side, channels)
    block = int(round(1 / ratio))
    shuffled = grid.reshape(batch, side // block, block, side // block, block, channels)
    shuffled = shuffled.transpose(0, 1, 3, 2, 4, 5)
    return shuffled.reshape(batch, (side // block) ** 2, grown)


class Llama4VisionAdapterMLP(nn.Module):
    """The tower's own projector: no-bias maps around an exact GELU."""

    input_dim: int
    output_dim: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, use_bias=False,
                                  dtype=self.dtype, precision=self.precision)
        self.fc1 = dense(self.input_dim, name="fc1")
        self.fc2 = dense(self.output_dim, name="fc2")

    def __call__(self, hidden_states):
        # The reference gels after both maps, including the last one
        # (modeling_llama4.py, Llama4VisionMLP2.forward).
        gelu = functools.partial(jax.nn.gelu, approximate=False)
        return gelu(self.fc2(gelu(self.fc1(hidden_states))))


class Llama4VisionTransformer(nn.Module):
    """The Llama 4 vision trunk, param layout of `Llama4VisionConfig`.

    `pixel_values` are what the checkpoint's image processor emits: [B, C, H,
    W] at `image_size`. The patches embed through one bias-free map, the class
    token rides last through the trunk and is dropped before the adapter, and
    what returns is the pixel-shuffled MLP output the outer projector maps to
    text width.
    """

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
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.hidden_size % self.num_heads:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must split over num_heads "
                f"({self.num_heads})")
        grid = self.image_size // self.patch_size
        if grid * self.patch_size != self.image_size:
            raise ValueError(
                f"image_size ({self.image_size}) must tile patch_size "
                f"({self.patch_size})")
        self.patch_embedding = nn.Dense(
            self.hidden_size, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="patch_embedding")
        self.class_embedding = self.param(
            "class_embedding", nn.initializers.normal(self.hidden_size ** -0.5),
            (self.hidden_size,))
        self.positional_embedding = self.param(
            "positional_embedding", nn.initializers.normal(self.hidden_size ** -0.5),
            (grid * grid + 1, self.hidden_size))
        norm = functools.partial(nn.LayerNorm, epsilon=self.layer_norm_eps,
                                 dtype=self.dtype)
        self.layernorm_pre = norm(name="layernorm_pre")
        self.layers = [
            Llama4VisionEncoderLayer(
                self.hidden_size, self.num_heads, self.intermediate_size, grid,
                self.rope_theta, layer_norm_eps=self.layer_norm_eps,
                dtype=self.dtype, precision=self.precision, name=f"layers_{index}")
            for index in range(self.num_layers)]
        self.layernorm_post = norm(name="layernorm_post")
        self.vision_adapter = Llama4VisionAdapter(
            self.pixel_shuffle_ratio, self.intermediate_size,
            self.projector_input_dim, self.projector_output_dim,
            dtype=self.dtype, precision=self.precision, name="vision_adapter")

    def __call__(self, pixel_values) -> jax.Array:
        pixel_values = jnp.asarray(pixel_values)
        batch, channels, height, width = pixel_values.shape
        expected = (self.num_channels, self.image_size, self.image_size)
        if (channels, height, width) != expected:
            raise ValueError(
                f"pixel_values of {channels}x{height}x{width} are not the "
                f"{'x'.join(map(str, expected))} this checkpoint was trained with")
        grid = self.image_size // self.patch_size
        patches = pixel_values.reshape(
            batch, channels, grid, self.patch_size, grid, self.patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        hidden_states = self.patch_embedding(
            patches.reshape(batch, grid * grid, channels * self.patch_size ** 2))
        class_token = jnp.broadcast_to(
            self.class_embedding.astype(hidden_states.dtype), (batch, 1, self.hidden_size))
        hidden_states = jnp.concatenate([hidden_states, class_token], axis=1)
        hidden_states = hidden_states + self.positional_embedding.astype(hidden_states.dtype)
        hidden_states = self.layernorm_pre(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.layernorm_post(hidden_states)[:, :-1, :]
        return self.vision_adapter(hidden_states)


class Llama4VisionAdapter(nn.Module):
    """Pixel shuffle into the adapter MLP, the tower's last stage."""

    ratio: float
    intermediate_size: int
    input_dim: int
    output_dim: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        self.mlp = Llama4VisionAdapterMLP(
            self.input_dim, self.output_dim, dtype=self.dtype,
            precision=self.precision, name="mlp")

    def __call__(self, encoded_patches) -> jax.Array:
        return self.mlp(pixel_shuffle(encoded_patches, self.ratio))


@towers("llama4")
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
        return Llama4VisionTransformer(
            hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
            num_layers=self.num_layers, num_heads=self.num_heads,
            image_size=self.image_size, patch_size=self.patch_size,
            num_channels=self.num_channels, layer_norm_eps=self.layer_norm_eps,
            rope_theta=self.rope_theta, pixel_shuffle_ratio=self.pixel_shuffle_ratio,
            projector_input_dim=self.projector_input_dim,
            projector_output_dim=self.projector_output_dim)


class Llama4ProjectorModule(nn.Module):
    """Tower output to text width: one bias-free map."""

    text_width: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        self.linear = nn.Dense(
            self.text_width, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="linear")

    def __call__(self, image_features) -> jax.Array:
        return self.linear(image_features)


@projectors("llama4")
@dataclasses.dataclass(frozen=True)
class Llama4Projector(ProjectorBase):
    """Llama 4's projector fields: the tower output width and the text width."""

    vision_width: int
    text_width: int

    def build(self) -> nn.Module:
        return Llama4ProjectorModule(text_width=self.text_width)

def _gemma4_rope_tables(positions: jax.Array, head_dim: int,
                        theta: float) -> Tuple[jax.Array, jax.Array]:
    """The 2D rotary tables of the Gemma 4 vision attention, as cos/sin.

    Each spatial dim carries its own frequencies over half the head
    (modeling_gemma4.py, Gemma4VisionRotaryEmbedding.
    compute_default_rope_parameters): the angles double up within a dim and
    the dims concatenate, so `positions` [B, P, 2] yields [B, P, head_dim].
    """
    spatial = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, spatial, 2, dtype=jnp.float32)
                                / spatial))
    angles = positions.astype(jnp.float32)[:, :, :, None] * inv_freq
    doubled = jnp.concatenate([angles, angles], axis=-1)
    cos = jnp.concatenate([jnp.cos(doubled[:, :, 0]), jnp.cos(doubled[:, :, 1])],
                          axis=-1)
    sin = jnp.concatenate([jnp.sin(doubled[:, :, 0]), jnp.sin(doubled[:, :, 1])],
                          axis=-1)
    return cos, sin


def _gemma4_rope(values: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """The half rotation on each spatial half, broadcast over the heads.

    Each half turns the NeoX way (modeling_gemma4.py, rotate_half and
    apply_multidimensional_rope): the halves split again and the second
    quarter crosses negated over the first.
    """
    halves = jnp.split(values, 2, axis=-1)
    angles = jnp.split(cos, 2, axis=-1)
    turns = jnp.split(sin, 2, axis=-1)
    rotated = [half * angle + jnp.concatenate(
        [-half[..., half.shape[-1] // 2:], half[..., :half.shape[-1] // 2]],
        axis=-1) * turn for half, angle, turn in zip(halves, angles, turns)]
    return jnp.concatenate(rotated, axis=-1)


class Gemma4VisionAttention(nn.Module):
    """Grouped-query attention with scaled q/k/v norms and the 2D rotary.

    The four maps are bias-free, the queries and keys norm with a scale and
    the values without (modeling_gemma4.py, Gemma4VisionAttention), and the
    scores run unscaled with the softmax in fp32 (its scaling is 1.0, not
    the shared kernel's 1/sqrt(d), so the few lines sit here).
    """

    hidden_size: int
    num_heads: int
    num_key_value_heads: int
    rope_theta: float = 100.0
    rms_norm_eps: float = 1e-6
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, use_bias=False,
                                  dtype=self.dtype, precision=self.precision)
        self.q_proj = dense(self.hidden_size, name="q_proj")
        self.k_proj = dense(self.num_key_value_heads * (self.hidden_size // self.num_heads),
                            name="k_proj")
        self.v_proj = dense(self.num_key_value_heads * (self.hidden_size // self.num_heads),
                            name="v_proj")
        self.o_proj = dense(self.hidden_size, name="o_proj")
        self.q_norm = RMSNorm(epsilon=self.rms_norm_eps, dtype=self.dtype, name="q_norm")
        self.k_norm = RMSNorm(epsilon=self.rms_norm_eps, dtype=self.dtype, name="k_norm")
        self.v_norm = RMSNorm(epsilon=self.rms_norm_eps, with_scale=False,
                              dtype=self.dtype, name="v_norm")

    def __call__(self, hidden_states, cos, sin) -> jax.Array:
        batch, length, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads
        # The norms read the heads (modeling_gemma4.py,
        # Gemma4VisionAttention.forward): project, split, then normalize.
        query = self.q_norm(self.q_proj(hidden_states).reshape(batch, length, -1, head_dim))
        key = self.k_norm(self.k_proj(hidden_states).reshape(batch, length, -1, head_dim))
        value = self.v_norm(self.v_proj(hidden_states).reshape(batch, length, -1, head_dim))
        query = _gemma4_rope(query, cos[:, :, None, :], sin[:, :, None, :])
        key = _gemma4_rope(key, cos[:, :, None, :], sin[:, :, None, :])
        repeats = self.num_heads // self.num_key_value_heads
        key = jnp.repeat(key, repeats, axis=2)
        value = jnp.repeat(value, repeats, axis=2)
        scores = jnp.einsum("bqhd,bkhd->bhqk", query, key)
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", probs, value)
        return self.o_proj(attended.reshape(batch, length, self.hidden_size))


class Gemma4VisionMLP(nn.Module):
    """Gated feed-forward without biases: act(gate) times up, then down.

    The reference reads the activation from the config (modeling_gemma4.py,
    Gemma4VisionMLP); only the two Gaussian forms map.
    """

    hidden_size: int
    intermediate_size: int
    hidden_act: str = "gelu_pytorch_tanh"
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, use_bias=False,
                                  dtype=self.dtype, precision=self.precision)
        self.gate_proj = dense(self.intermediate_size, name="gate_proj")
        self.up_proj = dense(self.intermediate_size, name="up_proj")
        self.down_proj = dense(self.hidden_size, name="down_proj")

    def __call__(self, hidden_states):
        if self.hidden_act == "gelu_pytorch_tanh":
            act = functools.partial(jax.nn.gelu, approximate=True)
        elif self.hidden_act == "gelu":
            act = functools.partial(jax.nn.gelu, approximate=False)
        else:
            raise ValueError(
                f"hidden_act {self.hidden_act!r} is not expressible: this MLP "
                "runs gelu_pytorch_tanh or gelu")
        return self.down_proj(act(self.gate_proj(hidden_states))
                              * self.up_proj(hidden_states))


class Gemma4VisionEncoderLayer(nn.Module):
    """RMS sandwich around attention and the gated MLP, both residual.

    The four norms all carry a scale (modeling_gemma4.py,
    Gemma4VisionEncoderLayer): one before and one after each half.
    """

    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_key_value_heads: int
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(RMSNorm, epsilon=self.rms_norm_eps, dtype=self.dtype)
        self.input_layernorm = norm(name="input_layernorm")
        self.self_attn = Gemma4VisionAttention(
            self.hidden_size, self.num_heads, self.num_key_value_heads,
            rope_theta=self.rope_theta, rms_norm_eps=self.rms_norm_eps,
            dtype=self.dtype, precision=self.precision, name="self_attn")
        self.post_attention_layernorm = norm(name="post_attention_layernorm")
        self.pre_feedforward_layernorm = norm(name="pre_feedforward_layernorm")
        self.mlp = Gemma4VisionMLP(
            self.hidden_size, self.intermediate_size, self.hidden_act,
            dtype=self.dtype, precision=self.precision, name="mlp")
        self.post_feedforward_layernorm = norm(name="post_feedforward_layernorm")

    def __call__(self, hidden_states, cos, sin):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), cos, sin)
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(hidden_states)


class Gemma4VisionTransformer(nn.Module):
    """The Gemma 4 vision trunk, param layout of `Gemma4VisionConfig`.

    `pixel_values` are [B, C, H, W] squares tiling patch_size, one resolution
    per call: the patches unfold in row-major order, the (x, y) ids come from
    the patch grid the way the processor's meshgrid lays them, and the pooler
    averages each pooling_kernel_size block into one soft token. Padding and
    video frames have no form here; the reference pads ragged batches with
    (-1, -1) marks, which this trunk refuses by taking images alone.
    """

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_layers: int = 27
    num_heads: int = 16
    num_key_value_heads: int = 16
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.hidden_size % self.num_heads:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must split over num_heads "
                f"({self.num_heads})")
        if self.hidden_size // self.num_heads % 4:
            raise ValueError(
                f"the head width ({self.hidden_size // self.num_heads}) must split "
                "over the two rotary dims and their halves")
        if self.num_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_heads ({self.num_heads}) must repeat num_key_value_heads "
                f"({self.num_key_value_heads})")
        self.patch_embed = nn.Dense(
            self.hidden_size, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="patch_embed")
        self.position_table = self.param(
            "position_table", nn.initializers.normal(0.02),
            (2, self.position_embedding_size, self.hidden_size))
        self.layers = [
            Gemma4VisionEncoderLayer(
                self.hidden_size, self.intermediate_size, self.num_heads,
                self.num_key_value_heads, self.hidden_act,
                rms_norm_eps=self.rms_norm_eps, rope_theta=self.rope_theta,
                dtype=self.dtype, precision=self.precision, name=f"layers_{index}")
            for index in range(self.num_layers)]
        self.std_bias = self.param("std_bias", nn.initializers.zeros, (self.hidden_size,))
        self.std_scale = self.param("std_scale", nn.initializers.ones, (self.hidden_size,))

    def __call__(self, pixel_values) -> jax.Array:
        pixel_values = jnp.asarray(pixel_values)
        batch, channels, height, width = pixel_values.shape
        if height != width or height % (self.patch_size * self.pooling_kernel_size):
            raise ValueError(
                f"pixel_values of {channels}x{height}x{width} are not a square "
                f"tiled by {self.pooling_kernel_size}x{self.pooling_kernel_size} "
                f"blocks of {self.patch_size}px patches")
        grid = height // self.patch_size
        patches = pixel_values.reshape(
            batch, channels, grid, self.patch_size, grid, self.patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        hidden_states = self.patch_embed(
            2 * (patches.reshape(batch, grid * grid,
                                channels * self.patch_size ** 2) - 0.5))
        rows = jnp.repeat(jnp.arange(grid), grid)
        cols = jnp.tile(jnp.arange(grid), grid)
        table = self.position_table.astype(hidden_states.dtype)
        hidden_states = hidden_states + table[0, cols] + table[1, rows]
        ids = jnp.stack([cols, rows], axis=-1)
        head_dim = self.hidden_size // self.num_heads
        cos, sin = _gemma4_rope_tables(
            jnp.broadcast_to(ids, (batch,) + ids.shape), head_dim, self.rope_theta)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        # The pooler averages each kernel block and scales in fp32, and the
        # trunk standardizes in fp32 (modeling_gemma4.py, Gemma4VisionPooler
        # and Gemma4VisionModel.forward).
        side = grid // self.pooling_kernel_size
        pooled = hidden_states.reshape(
            batch, side, self.pooling_kernel_size, side,
            self.pooling_kernel_size, self.hidden_size).mean(axis=(2, 4))
        pooled = pooled.reshape(batch, side * side, self.hidden_size)
        pooled = pooled.astype(jnp.float32) * self.hidden_size ** 0.5
        standardized = ((pooled - self.std_bias.astype(jnp.float32))
                        * self.std_scale.astype(jnp.float32))
        return standardized.astype(hidden_states.dtype)


@towers("gemma4")
@dataclasses.dataclass(frozen=True)
class Gemma4Vision(TowerBase):
    """A Gemma 4 trunk's geometry, under the reference's field names."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_layers: int = 27
    num_heads: int = 16
    num_key_value_heads: int = 16
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0

    def build(self) -> nn.Module:
        return Gemma4VisionTransformer(
            hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
            num_layers=self.num_layers, num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads, patch_size=self.patch_size,
            pooling_kernel_size=self.pooling_kernel_size,
            position_embedding_size=self.position_embedding_size,
            hidden_act=self.hidden_act, rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta)


class Gemma4ProjectorModule(nn.Module):
    """Soft tokens to text width: a scale-free RMS norm, then the map.

    The reference keeps this beside the tower (modeling_gemma4.py,
    Gemma4MultimodalEmbedder): the pre-projection norm carries no weight and
    the projection no bias.
    """

    text_width: int
    norm_eps: float = 1e-6
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        self.pre_norm = RMSNorm(epsilon=self.norm_eps, with_scale=False,
                                dtype=self.dtype, name="pre_norm")
        self.projection = nn.Dense(
            self.text_width, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="projection")

    def __call__(self, image_features) -> jax.Array:
        return self.projection(self.pre_norm(image_features))


@projectors("gemma4")
@dataclasses.dataclass(frozen=True)
class Gemma4Projector(ProjectorBase):
    """Gemma 4's projector fields: the tower width, the decoder width."""

    vision_width: int
    text_width: int
    norm_eps: float = 1e-6

    def build(self) -> nn.Module:
        return Gemma4ProjectorModule(text_width=self.text_width, norm_eps=self.norm_eps)

def _qwen35_interp_taps(index: jax.Array, size: int, side: int) -> Tuple[jax.Array, jax.Array]:
    """Bilinear taps into a `side`-long table for positions along one axis.

    The closed form of the align-corners linspace the reference resamples
    with (vision_utils.py, _interpolation_axis_taps_weights): each position
    reads its floor and ceiling with the linear hat weights.
    """
    src = index.astype(jnp.float32) * (side - 1) / max(size - 1, 1)
    floor = jnp.floor(src).astype(jnp.int32)
    distance = (src - floor.astype(jnp.float32))[:, None]
    taps = jnp.stack([floor, floor + 1], axis=-1).clip(0, side - 1)
    weights = jnp.concatenate([1 - distance, distance], axis=-1).clip(min=0)
    return taps, weights


def _qwen35_patch_positions(grid_height: int, grid_width: int,
                            merge_size: int) -> Tuple[jax.Array, jax.Array]:
    """Patch rows and columns in spatial-merge-block order.

    The tokens run block by block (vision_utils.py,
    get_vision_interpolation_indices_and_weights): the block row and column
    vary slowly, the within-block row and column fast.
    """
    within = jnp.arange(grid_height * grid_width)
    blocks_wide = grid_width // merge_size
    in_col = within % merge_size
    in_row = (within // merge_size) % merge_size
    block_col = (within // (merge_size * merge_size)) % blocks_wide
    block_row = within // (merge_size * merge_size * blocks_wide)
    return block_row * merge_size + in_row, block_col * merge_size + in_col


def _qwen35_pos_embeds(table: jax.Array, grid_height: int, grid_width: int,
                       merge_size: int) -> jax.Array:
    """The learned table resampled to one image's patch grid.

    Bilinear over the square table's rows and columns separately, outer
    product of the taps (vision_utils.py,
    get_vision_interpolation_indices_and_weights).
    """
    side = int(round(len(table) ** 0.5))
    rows, cols = _qwen35_patch_positions(grid_height, grid_width, merge_size)
    row_taps, row_weights = _qwen35_interp_taps(rows, grid_height, side)
    col_taps, col_weights = _qwen35_interp_taps(cols, grid_width, side)
    indices = (row_taps[:, :, None] * side + col_taps[:, None, :]).reshape(-1, 4)
    weights = (row_weights[:, :, None] * col_weights[:, None, :]).reshape(-1, 4)
    return (table[indices] * weights[..., None]).sum(axis=1)


def _qwen35_rope_tables(positions: jax.Array, head_dim: int) -> Tuple[jax.Array, jax.Array]:
    """The 2D rotary tables of the Qwen 3.5 vision attention, as cos/sin.

    Heights then widths share one frequency table over half the head
    (modeling_qwen3_5.py, Qwen3_5VisionRotaryEmbedding.forward), doubled the
    way the text rope doubles its pairs.
    """
    dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    flat = (positions.astype(jnp.float32)[:, :, None] * inv_freq).reshape(
        positions.shape[0], -1)
    doubled = jnp.concatenate([flat, flat], axis=-1)
    return jnp.cos(doubled), jnp.sin(doubled)


def _qwen35_rope(values: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """The half rotation, broadcast over the heads (modeling_qwen3_5.py,
    apply_rotary_pos_emb_vision)."""
    half = values.shape[-1] // 2
    first, second = values[..., :half], values[..., half:]
    return values * cos + jnp.concatenate([-second, first], axis=-1) * sin


class Qwen35VisionAttention(nn.Module):
    """Full attention over one image's patches with the 2D rotary.

    One fused map carries queries, keys and values together
    (modeling_qwen3_5.py, Qwen3_5VisionAttention); the scores scale with the
    head width and the softmax runs in fp32 through the shared kernel.
    """

    hidden_size: int
    num_heads: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        self.qkv = nn.Dense(3 * self.hidden_size, use_bias=True, dtype=self.dtype,
                            precision=self.precision, name="qkv")
        self.proj = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype,
                             precision=self.precision, name="proj")

    def __call__(self, hidden_states, cos, sin) -> jax.Array:
        batch, length, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads
        fused = self.qkv(hidden_states).reshape(batch, length, 3, self.num_heads, head_dim)
        query, key, value = (fused[:, :, 0], fused[:, :, 1], fused[:, :, 2])
        query = _qwen35_rope(query, cos[:, :, None, :], sin[:, :, None, :])
        key = _qwen35_rope(key, cos[:, :, None, :], sin[:, :, None, :])
        attended = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision)
        return self.proj(attended.reshape(batch, length, self.hidden_size))


class Qwen35VisionBlock(nn.Module):
    """Pre-norm attention over pre-norm shared MLP, both residual."""

    hidden_size: int
    intermediate_size: int
    num_heads: int
    hidden_act: str = "gelu_pytorch_tanh"
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(nn.LayerNorm, epsilon=1e-6, dtype=self.dtype)
        self.norm1 = norm(name="norm1")
        self.attn = Qwen35VisionAttention(
            self.hidden_size, self.num_heads, dtype=self.dtype,
            precision=self.precision, name="attn")
        self.norm2 = norm(name="norm2")
        self.mlp = MLP(self.hidden_size, self.intermediate_size,
                       activation=self.hidden_act, dtype=self.dtype,
                       precision=self.precision, name="mlp")

    def __call__(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen35VisionTransformer(nn.Module):
    """The Qwen 3.5 vision trunk, param layout of `Qwen3_5VisionConfig`.

    `pixel_values` are [B, C, H, W] stills, one resolution per call: each
    frame repeats along time the way the processor fills the temporal patch
    (image_processing_qwen2_vl.py, patchify), the learned positions resample
    to the patch grid, and what returns is the pre-merge sequence the merger
    reads. Packed batches of mixed resolutions ride cumulative lengths the
    pixel Field cannot carry, so only stills of one resolution map here.
    """

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.hidden_size % self.num_heads:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must split over num_heads "
                f"({self.num_heads})")
        self.patch_embed = nn.Dense(
            self.hidden_size, use_bias=True, dtype=self.dtype,
            precision=self.precision, name="patch_embed")
        self.position_table = nn.Embed(self.num_position_embeddings, self.hidden_size,
                                       dtype=self.dtype, name="position_table")
        self.blocks = [
            Qwen35VisionBlock(
                self.hidden_size, self.intermediate_size, self.num_heads,
                self.hidden_act, dtype=self.dtype, precision=self.precision,
                name=f"blocks_{index}")
            for index in range(self.depth)]

    def __call__(self, pixel_values) -> jax.Array:
        pixel_values = jnp.asarray(pixel_values)
        batch, channels, height, width = pixel_values.shape
        stride = self.patch_size * self.spatial_merge_size
        if channels != self.in_channels or height % stride or width % stride:
            raise ValueError(
                f"pixel_values of {channels}x{height}x{width} are not "
                f"{self.in_channels}-channel images tiled by {stride}px merge "
                "blocks")
        grid_height, grid_width = height // self.patch_size, width // self.patch_size
        patches = pixel_values.reshape(
            batch, channels, grid_height, self.patch_size, grid_width, self.patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        order = _qwen35_patch_positions(grid_height, grid_width, self.spatial_merge_size)
        flat = patches.reshape(batch, grid_height * grid_width, -1)[
            :, order[0] * grid_width + order[1]]
        # The still frame repeats along time (image_processing_qwen2_vl.py,
        # patchify), and the trunk views the flat buffer as [channel, time]:
        # slot (c, t) reads pixel channel (c*tp + t) % in_channels, which is
        # the flat tile below, while the weight map stays a plain reshape.
        patch = flat.reshape(batch, -1, channels, self.patch_size, self.patch_size)
        frames = jnp.tile(patch, (1, 1, self.temporal_patch_size, 1, 1))
        hidden_states = self.patch_embed(
            frames.reshape(batch, -1, channels * self.temporal_patch_size
                           * self.patch_size ** 2))
        table = self.position_table.embedding.astype(hidden_states.dtype)
        hidden_states = hidden_states + _qwen35_pos_embeds(
            table, grid_height, grid_width, self.spatial_merge_size)
        head_dim = self.hidden_size // self.num_heads
        cos, sin = _qwen35_rope_tables(jnp.stack(order, axis=-1), head_dim)
        cos = jnp.broadcast_to(cos, (batch,) + cos.shape)
        sin = jnp.broadcast_to(sin, (batch,) + sin.shape)
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)
        return hidden_states


@towers("qwen3_5")
@dataclasses.dataclass(frozen=True)
class Qwen35Vision(TowerBase):
    """A Qwen 3.5 trunk's geometry, under the reference's field names."""

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304

    def build(self) -> nn.Module:
        return Qwen35VisionTransformer(
            depth=self.depth, hidden_size=self.hidden_size, hidden_act=self.hidden_act,
            intermediate_size=self.intermediate_size, num_heads=self.num_heads,
            in_channels=self.in_channels, patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
            temporal_patch_size=self.temporal_patch_size,
            out_hidden_size=self.out_hidden_size,
            num_position_embeddings=self.num_position_embeddings)


class Qwen35ProjectorModule(nn.Module):
    """The merger: norm, group each merge block, map through an exact GELU.

    The reference keeps this inside the vision model (modeling_qwen3_5.py,
    Qwen3_5VisionPatchMerger); it reads the trunk sequence here so the tower
    and projector parities stay separate.
    """

    hidden_size: int
    spatial_merge_size: int
    out_hidden_size: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        self.norm = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm")
        grown = self.hidden_size * self.spatial_merge_size ** 2
        self.fc1 = nn.Dense(grown, use_bias=True, dtype=self.dtype,
                            precision=self.precision, name="fc1")
        self.fc2 = nn.Dense(self.out_hidden_size, use_bias=True, dtype=self.dtype,
                            precision=self.precision, name="fc2")

    def __call__(self, image_features) -> jax.Array:
        batch, count, _ = image_features.shape
        block = self.spatial_merge_size ** 2
        grown = self.hidden_size * block
        if count % block:
            raise ValueError(
                f"{count} patch features do not group into "
                f"{self.spatial_merge_size}x{self.spatial_merge_size} merge blocks")
        grouped = self.norm(image_features).reshape(batch, count // block, grown)
        return self.fc2(jax.nn.gelu(self.fc1(grouped), approximate=False))


@projectors("qwen3_5")
@dataclasses.dataclass(frozen=True)
class Qwen35Projector(ProjectorBase):
    """Qwen 3.5's projector fields: the trunk width, the merge size, the
    merged width the text embeddings read directly."""

    vision_width: int
    merge_size: int
    out_width: int

    def build(self) -> nn.Module:
        return Qwen35ProjectorModule(hidden_size=self.vision_width,
                                     spatial_merge_size=self.merge_size,
                                     out_hidden_size=self.out_width)


def merge_soft_tokens(token_embeds: jax.typing.ArrayLike, soft_tokens: jax.typing.ArrayLike,
                      image_mask: jax.typing.ArrayLike) -> jax.Array:
    """Token embeddings with one image's soft tokens at its image positions.

    `token_embeds` is [B, S, H], `soft_tokens` [B, T, H], and `image_mask`
    [B, S] marks the T positions of each row that carry image features. Every
    row must mark exactly T positions; the counts the reference checks in
    get_placeholder_mask are what is enforced here.
    """
    token_embeds = jnp.asarray(token_embeds)
    soft_tokens = jnp.asarray(soft_tokens)
    mask = jnp.asarray(image_mask)
    batch, length, _ = token_embeds.shape
    tokens = soft_tokens.shape[1]
    if (soft_tokens.shape[0], mask.shape) != (batch, (batch, length)):
        raise ValueError(
            f"soft tokens of {soft_tokens.shape} and a mask of {mask.shape} do not "
            f"cover token embeddings of {token_embeds.shape}")
    counts = mask.sum(axis=1)
    if bool((counts != tokens).any()):
        raise ValueError(
            f"each row must mark {tokens} image positions, got {counts.tolist()}")
    order = jnp.where(mask, jnp.cumsum(mask, axis=1, dtype=jnp.int32) - 1, 0)
    chosen = soft_tokens[jnp.arange(batch)[:, None], order]
    return jnp.where(mask[..., None], chosen, token_embeds)


def _leaf(path: Tuple[str, ...], tensor: np.ndarray) -> np.ndarray:
    """The tensor as the fp32 leaf at `path`, in linen's layout.

    torch Linear holds [out, in] and `nn.Dense` keeps [in, out]; torch Conv2d
    holds [out, in, kh, kw] and `nn.Conv` [kh, kw, in, out]. A norm's `weight`
    becomes `scale`, an embedding's `weight` becomes `embedding`, and a plain
    parameter matrix (a projector map, a class token) keeps its layout.
    """
    leaf = np.asarray(tensor, np.float32)
    if path[-1] == "kernel":
        leaf = np.ascontiguousarray(leaf.T if leaf.ndim == 2 else leaf.transpose(2, 3, 1, 0))
    return leaf


def _translate(hf_tensors: Mapping[str, np.ndarray], path_of) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, tensor in hf_tensors.items():
        path = path_of(name)
        if path is None:
            continue
        node = params
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = _leaf(path, tensor)
    return params


_SIGLIP_TENSORS = {
    "embeddings.patch_embedding.weight": ("patch_embedding", "kernel"),
    "embeddings.patch_embedding.bias": ("patch_embedding", "bias"),
    "embeddings.position_embedding.weight": ("position_embedding", "embedding"),
    "post_layernorm.weight": ("post_layernorm", "scale"),
    "post_layernorm.bias": ("post_layernorm", "bias"),
}
_SIGLIP_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")
_SIGLIP_NORMS = ("layer_norm1", "layer_norm2")


def _siglip_layer_path(parts) -> Optional[Tuple[str, ...]]:
    """`encoder.layers.N...` into the layer's path."""
    if len(parts) < 5 or parts[:2] != ["encoder", "layers"] or not parts[2].isdigit():
        return None
    layer, module, leaf = f"layers_{parts[2]}", parts[3], parts[-1]
    if len(parts) == 5 and module in _SIGLIP_NORMS and leaf in ("weight", "bias"):
        return (layer, module, "scale" if leaf == "weight" else "bias")
    if len(parts) == 6 and leaf in ("weight", "bias"):
        sublayer = parts[4]
        if module == "self_attn" and sublayer in _SIGLIP_PROJECTIONS:
            return (layer, module, sublayer, "kernel" if leaf == "weight" else "bias")
        if module == "mlp" and sublayer in ("fc1", "fc2"):
            return (layer, module, sublayer, "kernel" if leaf == "weight" else "bias")
    return None


def siglip_vision_path(hf_name: str) -> Optional[Tuple[str, ...]]:
    """One SigLIP vision tensor name into its path in a trunk tree.

    position_ids is an arange buffer, not a parameter. The attention pooling
    head some checkpoints carry maps to nothing: the Gemma path reads the
    trunk sequence alone. Anything else unknown raises ValueError.
    """
    if hf_name == "embeddings.position_ids":
        return None
    path = _SIGLIP_TENSORS.get(hf_name) or _siglip_layer_path(hf_name.split("."))
    if path is not None:
        return path
    if hf_name.split(".")[0] == "head":
        # The attention pooling head some checkpoints carry. The Gemma path
        # reads the trunk sequence alone, so its tensors map to nothing.
        return None
    raise ValueError(f"unknown tensor name {hf_name!r}")


def translate_siglip_vision_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """SigLIP vision tensors into a trunk params tree, in fp32."""
    return _translate(hf_tensors, siglip_vision_path)


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
_LLAMA4_VISION_LAYERS = {
    "input_layernorm": "input_layernorm",
    "post_attention_layernorm": "post_attention_layernorm",
}


def _llama4_vision_layer_path(parts) -> Optional[Tuple[str, ...]]:
    """`model.layers.N...` into the layer's path."""
    if len(parts) < 5 or parts[:2] != ["model", "layers"] or not parts[2].isdigit():
        return None
    layer, module, leaf = f"layers_{parts[2]}", parts[3], parts[-1]
    if len(parts) == 5 and module in _LLAMA4_VISION_LAYERS and leaf in ("weight", "bias"):
        return (layer, module, "scale" if leaf == "weight" else "bias")
    if len(parts) == 6 and leaf == "weight":
        sublayer = parts[4]
        if module == "self_attn" and sublayer in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return (layer, module, sublayer, "kernel")
        if module == "mlp" and sublayer in ("fc1", "fc2"):
            return (layer, module, sublayer, "kernel")
    if len(parts) == 6 and leaf == "bias":
        sublayer = parts[4]
        if module == "self_attn" and sublayer in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return (layer, module, sublayer, "bias")
        if module == "mlp" and sublayer in ("fc1", "fc2"):
            return (layer, module, sublayer, "bias")
    return None


def llama4_vision_path(hf_name: str) -> Optional[Tuple[str, ...]]:
    """One Llama 4 vision tensor name into its path in a trunk tree."""
    path = _LLAMA4_VISION_TENSORS.get(hf_name) or _llama4_vision_layer_path(
        hf_name.split("."))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_llama4_vision_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Llama 4 vision tensors into a trunk params tree, in fp32."""
    return _translate(hf_tensors, llama4_vision_path)


def translate_gemma_projector_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """A Gemma projector's two tensors into its params tree, in fp32.

    The norm's weight becomes its scale; the projection matrix is a plain
    parameter the reference multiplies as is, so unlike a Linear kernel it
    keeps its layout.
    """
    known = {"mm_soft_emb_norm.weight", "mm_input_projection_weight"}
    unknown = sorted(set(hf_tensors) - known)
    if unknown:
        raise ValueError(f"unknown tensor names {unknown}")
    return {
        "mm_soft_emb_norm": {
            "scale": np.asarray(hf_tensors["mm_soft_emb_norm.weight"], np.float32)},
        "mm_input_projection": {
            "kernel": np.ascontiguousarray(np.asarray(
                hf_tensors["mm_input_projection_weight"], np.float32))},
    }


def translate_llama4_projector_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Llama 4's outer projector map into its params tree, in fp32."""

    def path_of(hf_name: str) -> Optional[Tuple[str, ...]]:
        if hf_name != "linear_1.weight":
            raise ValueError(f"unknown tensor name {hf_name!r}")
        return ("linear", "kernel")

    return _translate(hf_tensors, path_of)


def _image_size(value: object, field: str) -> int:
    """A square image size as one side: an int, or a pair with equal sides."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(
                f"{field} {list(value)!r} is not square, this trunk tiles squares")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} is {value!r}, an image side is an int")
    return value


def translate_siglip_vision_config(hf_config: Mapping[str, Any]) -> Dict[str, object]:
    """A SiglipVisionConfig into a SiglipVision value's fields.

    Reads the vision_config of a multimodal wrapper or a bare vision config.
    Only square images map; anything but tanh or exact gelu refuses with its
    name, and a nonzero dropout refuses as training-only.
    """
    vision = hf_config.get("vision_config", hf_config)
    if not isinstance(vision, Mapping):
        raise ValueError(f"vision_config is {vision!r}, not a config")
    hidden = int(vision["hidden_size"])
    image = vision.get("image_size", 224)
    patch = vision.get("patch_size", 16)
    if isinstance(image, (list, tuple)):
        if len(image) != 2 or image[0] != image[1]:
            raise ValueError(
                f"image_size {list(image)!r} is not square, this trunk tiles squares")
        image = image[0]
    if isinstance(patch, (list, tuple)):
        patch = patch[0]
    activation = str(vision.get("hidden_act", vision.get("hidden_activation",
                                                         "gelu_pytorch_tanh")))
    if activation not in ("gelu_pytorch_tanh", "gelu"):
        raise ValueError(
            f"hidden_act {activation!r} is not expressible: this trunk runs tanh or "
            "exact gelu")
    if float(vision.get("attention_dropout", 0.0)):
        raise ValueError("attention_dropout is training-time; this trunk runs eval")
    return {
        "kind": "siglip",
        "hidden_size": hidden,
        "intermediate_size": int(vision["intermediate_size"]),
        "num_layers": int(vision["num_hidden_layers"]),
        "num_heads": int(vision["num_attention_heads"]),
        "image_size": int(image),
        "patch_size": int(patch),
        "num_channels": int(vision.get("num_channels", 3)),
        "hidden_act": activation,
        "layer_norm_eps": float(vision.get("layer_norm_eps", 1e-6)),
    }


def translate_llama4_vision_config(hf_config: Mapping[str, Any]) -> Dict[str, object]:
    """A Llama4VisionConfig into a Llama4Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The feature
    selection defaults to the last layer; the concatenated-layer strategy has
    no counterpart and refuses. Rope reads the nested spelling transformers
    writes or the flat theta the released Scout carries.
    """
    vision = hf_config.get("vision_config", hf_config)
    if not isinstance(vision, Mapping):
        raise ValueError(f"vision_config is {vision!r}, not a config")
    if vision.get("vision_feature_select_strategy", "default") != "default":
        raise ValueError(
            f"vision_feature_select_strategy "
            f"{vision.get('vision_feature_select_strategy')!r} concatenates layers "
            "this trunk never builds")
    if vision.get("vision_feature_layer", -1) not in (-1, None):
        raise ValueError(
            f"vision_feature_layer {vision.get('vision_feature_layer')!r} reads a "
            "middle layer this trunk never returns")
    rope = vision.get("rope_parameters") or {}
    if not isinstance(rope, Mapping):
        raise ValueError(f"rope_parameters is {rope!r}, not a config")
    theta = rope.get("rope_theta", vision.get("rope_theta", 10000.0))
    if theta is None or isinstance(theta, (Mapping, bool)):
        raise ValueError(f"rope_theta is {theta!r}, not a frequency")
    if float(vision.get("attention_dropout", 0.0)) or float(vision.get("projector_dropout", 0.0)):
        raise ValueError("attention_dropout/projector_dropout is training-time")
    if vision.get("multi_modal_projector_bias", False):
        raise ValueError("multi_modal_projector_bias=True needs a projector bias this map lacks")
    output_dim = int(vision.get("vision_output_dim", vision.get("projector_output_dim", 0)))
    if output_dim != int(vision.get("projector_output_dim", output_dim)):
        raise ValueError(
            f"vision_output_dim ({output_dim}) disagrees with projector_output_dim "
            f"({vision.get('projector_output_dim')}), the adapter's width")
    return {
        "kind": "llama4",
        "hidden_size": int(vision["hidden_size"]),
        "intermediate_size": int(vision["intermediate_size"]),
        "num_layers": int(vision["num_hidden_layers"]),
        "num_heads": int(vision["num_attention_heads"]),
        "image_size": _image_size(vision.get("image_size", 336), "image_size"),
        "patch_size": int(vision.get("patch_size", 14)),
        "num_channels": int(vision.get("num_channels", 3)),
        "layer_norm_eps": float(vision.get("norm_eps", vision.get("layer_norm_eps", 1e-5))),
        "rope_theta": float(theta),
        "pixel_shuffle_ratio": float(vision.get("pixel_shuffle_ratio", 0.5)),
        "projector_input_dim": int(vision["projector_input_dim"]),
        "projector_output_dim": int(vision["projector_output_dim"]),
    }


def translate_gemma_projector_config(vision: Mapping[str, Any], text_width: int,
                                     mm_tokens_per_image: object) -> Dict[str, object]:
    """A Gemma wrapper's projector fields: trunk width, decoder width, grids."""
    if isinstance(mm_tokens_per_image, bool) or not isinstance(mm_tokens_per_image, int):
        raise ValueError(
            f"mm_tokens_per_image is {mm_tokens_per_image!r}, the soft-token count "
            "is an int")
    side = int(mm_tokens_per_image ** 0.5)
    if side * side != mm_tokens_per_image:
        raise ValueError(
            f"mm_tokens_per_image ({mm_tokens_per_image}) is not a square, this "
            "projector pools a grid into a grid")
    patches = int(vision["image_size"]) // int(vision["patch_size"])
    if patches % side:
        raise ValueError(
            f"{patches} patches per side do not split over {side} soft tokens per side")
    return {
        "kind": "gemma",
        "vision_width": int(vision["hidden_size"]),
        "text_width": int(text_width),
        "patches_per_side": patches,
        "tokens_per_side": side,
        "norm_eps": float(vision.get("layer_norm_eps", 1e-6)),
    }


def translate_llama4_projector_config(vision: Mapping[str, Any],
                                      text_width: int) -> Dict[str, object]:
    """A Llama 4 wrapper's projector fields: tower output width, text width."""
    return {
        "kind": "llama4",
        "vision_width": int(vision["projector_output_dim"]),
        "text_width": int(text_width),
    }

_GEMMA4_VISION_TENSORS = {
    "patch_embedder.input_proj.weight": ("patch_embed", "kernel"),
    "patch_embedder.position_embedding_table": ("position_table",),
    "std_bias": ("std_bias",),
    "std_scale": ("std_scale",),
}
_GEMMA4_VISION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_GEMMA4_VISION_MLP = ("gate_proj", "up_proj", "down_proj")
_GEMMA4_VISION_NORMS = ("input_layernorm", "post_attention_layernorm",
                        "pre_feedforward_layernorm", "post_feedforward_layernorm")


def _gemma4_vision_layer_path(parts) -> Optional[Tuple[str, ...]]:
    """`encoder.layers.N...` into the layer's path."""
    if len(parts) < 5 or parts[:2] != ["encoder", "layers"] or not parts[2].isdigit():
        return None
    layer = f"layers_{parts[2]}"
    if len(parts) == 5 and parts[3] in _GEMMA4_VISION_NORMS and parts[4] == "weight":
        return (layer, parts[3], "scale")
    if len(parts) == 7 and parts[5] == "linear" and parts[6] == "weight":
        if parts[3] == "self_attn" and parts[4] in _GEMMA4_VISION_PROJECTIONS:
            return (layer, "self_attn", parts[4], "kernel")
        if parts[3] == "mlp" and parts[4] in _GEMMA4_VISION_MLP:
            return (layer, "mlp", parts[4], "kernel")
    if len(parts) == 6 and parts[3] == "self_attn" and parts[5] == "weight":
        if parts[4] in ("q_norm", "k_norm"):
            # The value norm carries no scale (modeling_gemma4.py,
            # Gemma4VisionAttention), so a weight under its name is unknown.
            return (layer, "self_attn", parts[4], "scale")
    return None

def gemma4_vision_path(hf_name: str) -> Optional[Tuple[str, ...]]:
    """One Gemma 4 vision tensor name into its path in a trunk tree.

    The rotary tables are buffers the checkpoint leaves out, recomputed from
    the grid at call time. Anything else unknown raises ValueError.
    """
    path = _GEMMA4_VISION_TENSORS.get(hf_name) or _gemma4_vision_layer_path(
        hf_name.split("."))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_gemma4_vision_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Gemma 4 vision tensors into a trunk params tree, in fp32."""
    return _translate(hf_tensors, gemma4_vision_path)


def translate_gemma4_projector_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """A Gemma 4 embedder's map into its params tree, in fp32.

    The pre-projection norm carries no scale, so the projection weight is
    the only tensor.
    """
    if set(hf_tensors) != {"embedding_projection.weight"}:
        raise ValueError(f"unknown tensor names {sorted(hf_tensors)}")
    return {
        "projection": {
            "kernel": _leaf(("projection", "kernel"),
                            hf_tensors["embedding_projection.weight"])},
    }


def _gemma4_rope_theta(vision: Mapping[str, Any]) -> float:
    """The vision rope theta, defaulting the way the config class does."""
    rope = vision.get("rope_parameters") or {}
    if not isinstance(rope, Mapping):
        raise ValueError(f"rope_parameters is {rope!r}, not a config")
    if rope.get("rope_type", "default") != "default":
        raise ValueError(
            f"rope_type {rope.get('rope_type')!r} is not expressible: this trunk "
            "runs the default 2D rotary")
    theta = rope.get("rope_theta", vision.get("rope_theta", 100.0))
    if theta is None or isinstance(theta, (Mapping, bool)):
        raise ValueError(f"rope_theta is {theta!r}, not a frequency")
    return float(theta)


def translate_gemma4_vision_config(hf_config: Mapping[str, Any]) -> Dict[str, object]:
    """A Gemma4VisionConfig into a Gemma4Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The head
    width derives from the hidden size and the head count; a config carrying
    any other width refuses. Biases, dropout, clipped linears and an
    unstandardized trunk refuse, each naming its field.
    """
    vision = hf_config.get("vision_config", hf_config)
    if not isinstance(vision, Mapping):
        raise ValueError(f"vision_config is {vision!r}, not a config")
    hidden = int(vision["hidden_size"])
    heads = int(vision["num_attention_heads"])
    head_dim = vision.get("head_dim", hidden // heads)
    if int(head_dim) != hidden // heads or hidden % heads:
        raise ValueError(
            f"head_dim ({head_dim}) is not hidden_size ({hidden}) over "
            f"num_attention_heads ({heads}), the width this trunk derives")
    activation = str(vision.get("hidden_activation", vision.get("hidden_act",
                                                                "gelu_pytorch_tanh")))
    if activation not in ("gelu_pytorch_tanh", "gelu"):
        raise ValueError(
            f"hidden_activation {activation!r} is not expressible: this trunk "
            "runs gelu_pytorch_tanh or gelu")
    if vision.get("attention_bias", False):
        raise ValueError("attention_bias=True needs biased maps this trunk lacks")
    if float(vision.get("attention_dropout", 0.0)):
        raise ValueError("attention_dropout is training-time; this trunk runs eval")
    if vision.get("use_clipped_linears", False):
        raise ValueError(
            "use_clipped_linears=True needs clamped maps this trunk lacks")
    if not vision.get("standardize", False):
        raise ValueError(
            "standardize=False skips the bias and scale this trunk applies")
    if "output_proj_dims" in vision:
        raise ValueError(
            f"output_proj_dims ({vision['output_proj_dims']!r}) changes the "
            "projector width this record leaves to the text width")
    return {
        "kind": "gemma4",
        "hidden_size": hidden,
        "intermediate_size": int(vision["intermediate_size"]),
        "num_layers": int(vision["num_hidden_layers"]),
        "num_heads": heads,
        "num_key_value_heads": int(vision.get("num_key_value_heads", heads)),
        "patch_size": int(vision.get("patch_size", 16)),
        "pooling_kernel_size": int(vision.get("pooling_kernel_size", 3)),
        "position_embedding_size": int(vision.get("position_embedding_size", 10240)),
        "hidden_act": activation,
        "rms_norm_eps": float(vision.get("rms_norm_eps", 1e-6)),
        "rope_theta": _gemma4_rope_theta(vision),
    }


def translate_gemma4_projector_config(vision: Mapping[str, Any],
                                      text_width: int) -> Dict[str, object]:
    """A Gemma 4 wrapper's projector fields: tower width, decoder width."""
    return {
        "kind": "gemma4",
        "vision_width": int(vision["hidden_size"]),
        "text_width": int(text_width),
        "norm_eps": float(vision.get("rms_norm_eps", 1e-6)),
    }


_QWEN35_VISION_TENSORS = {
    "pos_embed.weight": ("position_table", "embedding"),
}
_QWEN35_VISION_BLOCK_NORMS = ("norm1", "norm2")
_QWEN35_VISION_MLP = ("linear_fc1", "linear_fc2")


def _qwen35_vision_block_path(parts) -> Optional[Tuple[str, ...]]:
    """`blocks.N...` into the block's path."""
    if len(parts) < 4 or parts[0] != "blocks" or not parts[1].isdigit():
        return None
    block = f"blocks_{parts[1]}"
    if len(parts) == 4 and parts[2] in _QWEN35_VISION_BLOCK_NORMS and parts[3] in (
            "weight", "bias"):
        return (block, parts[2], "scale" if parts[3] == "weight" else "bias")
    if len(parts) == 5 and parts[4] in ("weight", "bias"):
        leaf = "kernel" if parts[4] == "weight" else "bias"
        if parts[2] == "attn" and parts[3] in ("qkv", "proj"):
            return (block, "attn", parts[3], leaf)
        if parts[2] == "mlp" and parts[3] in _QWEN35_VISION_MLP:
            return (block, "mlp", {"linear_fc1": "fc1", "linear_fc2": "fc2"}[parts[3]],
                    leaf)
    return None


def qwen35_vision_path(hf_name: str) -> Optional[Tuple[str, ...]]:
    """One Qwen 3.5 vision tensor name into its path in a trunk tree.

    The merger lives under its own prefix and maps with the projector; the
    rotary table is a buffer the checkpoint leaves out. Anything else unknown
    raises ValueError.
    """
    if hf_name.split(".")[0] == "merger":
        return None
    if hf_name == "patch_embed.proj.weight":
        return ("patch_embed", "kernel")
    if hf_name == "patch_embed.proj.bias":
        return ("patch_embed", "bias")
    path = _QWEN35_VISION_TENSORS.get(hf_name) or _qwen35_vision_block_path(
        hf_name.split("."))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_qwen35_vision_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Qwen 3.5 vision tensors into a trunk params tree, in fp32.

    The patch convolution carries [out, in, time, h, w] and lands as one map;
    the trunk's buffer already holds the viewed channel order, so the rows
    are a plain reshape and every other leaf rides the shared transpose.
    """
    rest = {name: tensor for name, tensor in hf_tensors.items()
            if name != "patch_embed.proj.weight"}
    params = _translate(rest, qwen35_vision_path)
    if "patch_embed.proj.weight" not in hf_tensors:
        raise ValueError("patch_embed.proj.weight is missing, the trunk reads it")
    conv = np.asarray(hf_tensors["patch_embed.proj.weight"], np.float32)
    params.setdefault("patch_embed", {})["kernel"] = np.ascontiguousarray(
        conv.transpose(1, 2, 3, 4, 0).reshape(-1, conv.shape[0]))
    return params


def translate_qwen35_projector_weights(
        hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """A Qwen 3.5 merger's tensors into its params tree, in fp32."""

    def path_of(hf_name: str) -> Optional[Tuple[str, ...]]:
        # The plain fixture keeps the reference's merger prefix while the
        # wrapper routing strips it; both name the same leaves.
        bare = hf_name[7:] if hf_name.startswith("merger.") else hf_name
        parts = bare.split(".")
        if len(parts) == 2 and parts[1] in ("weight", "bias"):
            leaf = "kernel" if parts[1] == "weight" else "bias"
            if parts[0] == "norm":
                return ("norm", "scale" if parts[1] == "weight" else "bias")
            if parts[0] in ("linear_fc1", "linear_fc2"):
                return ({"linear_fc1": "fc1", "linear_fc2": "fc2"}[parts[0]], leaf)
        raise ValueError(f"unknown tensor name {hf_name!r}")

    return _translate(hf_tensors, path_of)


def _qwen35_patch_field(vision: Mapping[str, Any], field: str) -> int:
    """A patch-size field as an int; a pair has no square form here."""
    value = vision[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field} is {value!r}, the trunk tiles square patches of one size")
    return value


def translate_qwen35_vision_config(hf_config: Mapping[str, Any]) -> Dict[str, object]:
    """A Qwen3_5VisionConfig into a Qwen35Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The
    released 0.8B checkpoint spells the tower's model_type qwen3_5, which the
    config class rewrites on load; both spellings map. The position table
    must be square, and the activation one the shared MLP runs.
    """
    vision = hf_config.get("vision_config", hf_config)
    if not isinstance(vision, Mapping):
        raise ValueError(f"vision_config is {vision!r}, not a config")
    if vision.get("model_type", "qwen3_5_vision") not in ("qwen3_5_vision", "qwen3_5"):
        raise ValueError(
            f"vision model_type {vision.get('model_type')!r} is not the Qwen 3.5 tower")
    table = int(vision["num_position_embeddings"])
    if int(table ** 0.5) ** 2 != table:
        raise ValueError(
            f"num_position_embeddings ({table}) is not a square, this trunk "
            "resamples a square table")
    activation = str(vision.get("hidden_act", "gelu_pytorch_tanh"))
    if activation not in ("quick_gelu", "gelu_pytorch_tanh", "gelu"):
        raise ValueError(
            f"hidden_act {activation!r} is not expressible: this trunk runs the "
            "shared MLP's activations")
    return {
        "kind": "qwen3_5",
        "depth": int(vision["depth"]),
        "hidden_size": int(vision["hidden_size"]),
        "hidden_act": activation,
        "intermediate_size": int(vision["intermediate_size"]),
        "num_heads": int(vision["num_heads"]),
        "in_channels": int(vision.get("in_channels", 3)),
        "patch_size": _qwen35_patch_field(vision, "patch_size"),
        "spatial_merge_size": int(vision.get("spatial_merge_size", 2)),
        "temporal_patch_size": _qwen35_patch_field(vision, "temporal_patch_size"),
        "out_hidden_size": int(vision["out_hidden_size"]),
        "num_position_embeddings": table,
    }


def translate_qwen35_projector_config(vision: Mapping[str, Any],
                                      text_width: int) -> Dict[str, object]:
    """A Qwen 3.5 wrapper's projector fields: trunk width, merge, output.

    The merged features enter the text embeddings directly, so a merger width
    beside the decoder width refuses.
    """
    merged = int(vision["out_hidden_size"])
    if merged != int(text_width):
        raise ValueError(
            f"out_hidden_size ({merged}) is not the decoder width ({text_width}), "
            "the merger output enters the text embeddings as it is")
    return {
        "kind": "qwen3_5",
        "vision_width": int(vision["hidden_size"]),
        "merge_size": int(vision["spatial_merge_size"]),
        "out_width": merged,
    }


@towers("gemma3n")
@dataclasses.dataclass(frozen=True)
class Gemma3nVision(TowerBase):
    """MobileNet-v5's encoder construction fields, as timm model_args names them."""

    channel_multiplier: float = 1.0
    stem_size: int = 64
    stem_bias: bool = True
    fix_stem: bool | None = None
    in_chans: int = 3
    pad_type: str = "same"
    group_size: int | None = None
    msfa_indices: tuple[int, ...] = (-2, -1)
    msfa_output_resolution: int = 16
    layer_scale_init_value: float | None = 1e-5
    drop_path_rate: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "msfa_indices", tuple(self.msfa_indices))

    def build(self) -> nn.Module:
        return MobileNetV5Encoder(**dataclasses.asdict(self))


class Gemma3nProjectorModule(nn.Module):
    """Gemma3nMultimodalEmbedder's vision hard and soft token paths."""

    vision_width: int
    text_width: int
    vocab_size: int = 128
    vocab_offset: int = 262144
    norm_eps: float = 1e-6
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if min(self.vision_width, self.text_width, self.vocab_size) < 1 or self.vocab_offset < 0:
            raise ValueError("vision/text widths and vocab_size must be positive; vocab_offset is nonnegative")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        self.embedding = nn.Embed(self.vocab_size, self.vision_width, dtype=self.dtype,
                                  name="embedding")
        norm = functools.partial(RMSNorm, epsilon=self.norm_eps, dtype=self.dtype)
        self.hard_embedding_norm = norm(name="hard_embedding_norm")
        self.soft_embedding_norm = norm(name="soft_embedding_norm")
        self.embedding_projection = nn.Dense(self.text_width, use_bias=False,
                                              dtype=self.dtype, precision=self.precision,
                                              name="embedding_projection")
        self.embedding_post_projection_norm = norm(with_scale=False,
                                                   name="embedding_post_projection_norm")

    def __call__(self, features):
        features = jnp.asarray(features)
        if features.ndim != 3 or features.shape[-1] != self.vision_width:
            raise ValueError(f"vision features must be [B, N, {self.vision_width}], got {features.shape}")
        if not jnp.issubdtype(features.dtype, jnp.floating):
            raise ValueError("vision features must be floating point")
        if self.is_initializing():
            # Both paths belong to one checkpoint even when init starts with
            # image features. Linen creates an embedding only when called.
            self.hard_embedding_norm(self.embedding(jnp.zeros((1, 1), jnp.int32)))
        features = features * jnp.asarray(self.vision_width ** 0.5, features.dtype)
        return self.embedding_post_projection_norm(
            self.embedding_projection(self.soft_embedding_norm(features)))

    def hard_embeddings(self, input_ids):
        """Embed IDs in [vocab_offset, vocab_offset + vocab_size)."""
        ids = jnp.asarray(input_ids)
        if ids.ndim != 2 or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError("vision token IDs must be an integer [B, S] array")
        embedded = self.embedding(ids - self.vocab_offset)
        if self.is_initializing():
            self.soft_embedding_norm(jnp.zeros_like(embedded))
        return self.embedding_post_projection_norm(
            self.embedding_projection(self.hard_embedding_norm(embedded)))

    def merge_hard_embeddings(self, token_embeddings, input_ids):
        """Replace vision-vocabulary IDs before image soft tokens are inserted."""
        ids = jnp.asarray(input_ids)
        if token_embeddings.shape != ids.shape + (self.text_width,):
            raise ValueError("token embeddings must align with input_ids and text_width")
        mask = (ids >= self.vocab_offset) & (ids < self.vocab_offset + self.vocab_size)
        chosen = jnp.where(mask, ids, self.vocab_offset + self.vocab_size - 1)
        hard = self.hard_embeddings(chosen).astype(token_embeddings.dtype)
        return jnp.where(mask[..., None], hard, token_embeddings)

    def model_inputs(self, token_embeddings, input_ids, soft_tokens=None,
                     image_positions=None, *, per_layer_input_vocab: int) -> dict[str, jax.Array]:
        """CausalTransformer inputs after vision fusion and the PLE vocabulary mask.

        token_embeddings are the decoder's scaled embeddings. Image positions
        are a precomputed integer [B, N] array, matching the [B, N, D] soft
        tokens. Keeping positions explicit makes this path differentiable and
        jittable without a host-side token-count check.
        """
        ids = jnp.asarray(input_ids)
        embeddings = jnp.asarray(token_embeddings)
        if not jnp.issubdtype(embeddings.dtype, jnp.floating):
            raise ValueError("token_embeddings must be floating point")
        if per_layer_input_vocab < 1:
            raise ValueError("per_layer_input_vocab must be positive")
        if (soft_tokens is None) != (image_positions is None):
            raise ValueError("soft_tokens and image_positions arrive together")
        merged = self.merge_hard_embeddings(embeddings, ids)
        if soft_tokens is not None:
            soft, positions = jnp.asarray(soft_tokens), jnp.asarray(image_positions)
            if (soft.ndim != 3 or soft.shape[0] != ids.shape[0]
                    or soft.shape[-1] != self.text_width or positions.shape != soft.shape[:2]):
                raise ValueError("soft_tokens and image_positions must be [B, N, D] and [B, N]")
            if not jnp.issubdtype(positions.dtype, jnp.integer):
                raise ValueError("image_positions must be integers")
            merged = merged.at[jnp.arange(ids.shape[0])[:, None], positions].set(
                soft.astype(merged.dtype))
        tokens = jnp.where((ids >= 0) & (ids < per_layer_input_vocab), ids, 0)
        return {"tokens": tokens, "input_embeddings": merged,
                "embedding_positions": jnp.broadcast_to(
                    jnp.arange(ids.shape[1], dtype=jnp.int32), ids.shape)}

@projectors("gemma3n")
@dataclasses.dataclass(frozen=True)
class Gemma3nProjector(ProjectorBase):
    vision_width: int
    text_width: int
    vocab_size: int = 128
    vocab_offset: int = 262144
    norm_eps: float = 1e-6

    def build(self) -> nn.Module:
        return Gemma3nProjectorModule(**dataclasses.asdict(self))


def gemma3n_vision_path(hf_name: str) -> Tuple[str, ...]:
    """A timm MobileNet-v5 weight into the corresponding Linen module."""
    from .mobilenet import _ARCHITECTURE

    bare = hf_name.removeprefix("timm_model.")
    parts = tuple(bare.split("."))
    prefix: tuple[str, ...] = ()
    tails: set[tuple[str, ...]]
    if parts[:1] == ("blocks",) and len(parts) >= 5 and parts[1].isdigit() and parts[2].isdigit():
        stage, index = int(parts[1]), int(parts[2])
        if stage >= len(_ARCHITECTURE) or index >= len(_ARCHITECTURE[stage]):
            raise ValueError(f"unknown tensor name {hf_name!r}")
        spec = _ARCHITECTURE[stage][index]
        prefix = (f"stages_{stage}", f"blocks_{index}")
        parts = parts[3:]
        if spec.kind == "edge":
            tails = {(name, "weight") for name in ("conv_exp", "conv_pwl", "bn1", "bn2")}
        elif spec.kind == "inverted":
            modules = ["pw_exp", "pw_proj"]
            if spec.start_kernel:
                modules.append("dw_start")
            if spec.middle_kernel:
                modules.append("dw_mid")
            tails = {(name, child, "weight") for name in modules for child in ("conv", "bn")}
            tails.add(("layer_scale", "gamma"))
        else:
            tails = {("norm", "weight"), ("layer_scale", "gamma")}
            tails.update(("attn", name, "proj", "weight") for name in ("query", "key", "value", "output"))
            if spec.kv_stride > 1:
                tails.update(("attn", name, child, "weight") for name in ("key", "value")
                             for child in ("down_conv", "norm"))
    elif parts[:1] == ("conv_stem",):
        tails = {("conv", "weight"), ("conv", "bias"), ("bn", "weight")}
        prefix, parts = ("conv_stem",), parts[1:]
    elif parts[:1] == ("msfa",):
        tails = {("norm", "weight")}
        tails.update(("ffn", name, child, "weight") for name in ("pw_exp", "pw_proj")
                     for child in ("conv", "bn"))
        prefix, parts = ("msfa",), parts[1:]
    else:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    if len(parts) < 2 or parts not in tails:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    if parts[-1] != "weight":
        return prefix + parts
    norm = parts[-2] in ("bn", "bn1", "bn2", "norm")
    return prefix + parts[:-1] + ("scale" if norm else "kernel",)


def translate_gemma3n_vision_weights(hf_tensors: Mapping[str, np.ndarray]) -> dict[str, object]:
    return _translate(hf_tensors, gemma3n_vision_path)


def translate_gemma3n_projector_weights(hf_tensors: Mapping[str, np.ndarray]) -> dict[str, object]:
    paths = {
        "embedding.weight": ("embedding", "embedding"),
        "hard_embedding_norm.weight": ("hard_embedding_norm", "scale"),
        "soft_embedding_norm.weight": ("soft_embedding_norm", "scale"),
        "embedding_projection.weight": ("embedding_projection", "kernel"),
    }
    if set(hf_tensors) != set(paths):
        raise ValueError(f"vision embedder tensors differ: missing {sorted(set(paths) - set(hf_tensors))}, "
                         f"unknown {sorted(set(hf_tensors) - set(paths))}")
    return _translate(hf_tensors, paths.__getitem__)


def translate_gemma3n_vision_config(hf_config: Mapping[str, object]) -> dict[str, object]:
    vision = hf_config.get("vision_config", hf_config)
    if not isinstance(vision, Mapping):
        raise ValueError("vision_config must be a mapping")
    if vision.get("architecture", "mobilenetv5_300m_enc") != "mobilenetv5_300m_enc":
        raise ValueError(f"architecture {vision.get('architecture')!r} is not the MobileNet-v5 encoder")
    if int(vision.get("hidden_size", 2048)) != 2048:
        raise ValueError("hidden_size must be 2048; timm's MobileNet-v5 encoder fixes its adapter width")
    if vision.get("do_pooling", False):
        raise ValueError("do_pooling=True requests a classifier head the encoder does not have")
    options = vision.get("model_args") or {}
    if not isinstance(options, Mapping):
        raise ValueError("model_args must be a mapping")
    allowed = {field.name for field in dataclasses.fields(Gemma3nVision)}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"MobileNet-v5 model_args {sorted(unknown)} are not supported")
    value = Gemma3nVision(**options)
    return {"kind": "gemma3n", **dataclasses.asdict(value)}


def translate_gemma3n_projector_config(hf_config: Mapping[str, object],
                                       text_width: int) -> dict[str, object]:
    vision = hf_config.get("vision_config", hf_config)
    if not isinstance(vision, Mapping):
        raise ValueError("vision_config must be a mapping")
    return {"kind": "gemma3n", "vision_width": int(vision.get("hidden_size", 2048)),
            "text_width": text_width, "vocab_size": int(vision.get("vocab_size", 128)),
            "vocab_offset": int(vision.get("vocab_offset", 262144)),
            "norm_eps": float(vision.get("rms_norm_eps", 1e-6))}
