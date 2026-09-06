"""SigLIP and Llama 4 vision towers with their multimodal projectors, and weights.

transformers 5 ships no Flax classes, so the towers are vendored the way
`dew/nn/text_encoders.py` vendors CLIP, in the reference layout, with each
weight read from the checkpoint's safetensors under its reference tensor name.
The operation order follows transformers 5.16.1
`models/siglip/modeling_siglip.py` and `models/llama4/modeling_llama4.py`.

The SigLIP trunk is patch convolution with bias, learned position embeddings
with no class token, pre-norm encoder blocks and a post layer norm. Its block
shares both halves with CLIP's: the attention and the feed-forward
(`dew.nn.text_encoders`), the MLP carrying the config's activation. The Llama 4
trunk is MetaCLIP-style: an unfold patch embedding without bias, a class token
appended after the patches, learned positions, a pre norm, full-attention
blocks with a complex rotary over the patch grid, a post norm, the class
token dropped, and the pixel-shuffle MLP inside the tower where the reference
keeps it. Each tower's projector is a registered value beside it: Gemma's
averages each patch block, norms and maps to the decoder width, and Llama 4's
maps the tower output to the decoder width.
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
