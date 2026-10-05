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
vision embedder defined here. DeepSeek-V4.1's trunk (the release's
`inference/vision.py`) is a biased patch map, RMS-normed blocks with the
same 2D rotary as Qwen 3.5's and a SwiGLU feed-forward, and a final norm; its
projector is the aligner, which groups squares of the patch grid through an
exact-GELU MLP and lays the image out as its decoder reads it.
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.typing import DTypeLike

from dew import records
from dew._model_types import _QWEN35_VISION_TYPES
from dew.interop.weights import checkpoint_array, translate_parameters
from dew.nn.attention import LayerNorm, RMSNorm, scaled_dot_product_attention
from dew.nn.conv import Conv
from dew.nn.precision import at_least_fp32
from dew.nn.rope import inverse_frequencies
from dew.nn.text_encoders import MLP, CLIPEncoderLayer
from dew.objectives.base import Variables
from dew.registry import from_record, projectors, towers

from .mobilenet import _ARCHITECTURE, MobileNetV5Encoder

PIXEL_VALUES_KEY = "pixel_values"
"""The batch field carrying images as the checkpoint's processor emitted them."""


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


@dataclasses.dataclass(frozen=True)
class TowerGeometry:
    """The shapes a conditioner has to invent to create a tower's media leaves.

    A token-only init has no batch, so `VisionConditioner` and
    `AudioConditioner` build one smallest input the tower accepts. What that
    is differs by kind: a fixed-resolution tower states its `image_size`, a
    patch-and-pool tower states the patch and block that make one, and an
    audio tower states how many mel bins a frame carries. None means this
    tower has no such field, and the conditioner uses its own default.
    """

    image_size: int | None = None
    patch_size: int | None = None
    block_size: int | None = None
    channels: int | None = None
    mel_features: int | None = None


class TowerBase:
    """One tower kind's value: its fields, and how it builds its module."""

    def build(self) -> nn.Module:
        """The Flax module for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} names a tower kind but builds no module")

    def geometry(self) -> TowerGeometry:
        """What an initialising input has to look like for this tower.

        Nothing by default: a tower states the fields it has, and a
        conditioner reads them here instead of asking the value at runtime
        whether it carries each one.
        """
        return TowerGeometry()


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


@projectors("gemma")
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


def _llama4_vision_tables(grid: int, head_dim: int, theta: float, *,
                          dtype: DTypeLike) -> tuple[jax.Array, jax.Array]:
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
    inv_freq = inverse_frequencies(theta, freq_dim, dtype=dtype)
    angles = jnp.concatenate([(safe % grid + 1)[:, None] * inv_freq[None, :],
                              (safe // grid + 1)[:, None] * inv_freq[None, :]], axis=1)
    angles = jnp.where((kinds < 0)[:, None], 0.0, angles)
    return jnp.cos(angles), jnp.sin(angles)


def _llama4_vision_rope(values: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """The complex rotation on real pairs, one shared angle per pair."""
    pairs = values.reshape(*values.shape[:-1], -1, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    table, turn = cos[:, None, :], sin[:, None, :]
    rotated = jnp.stack([first * table - second * turn,
                         first * turn + second * table], axis=-1)
    return rotated.reshape(values.shape)


class Llama4VisionAttention(nn.Module):
    """Biased multi-head attention with the grid rotary on queries and keys."""

    hidden_size: int
    num_heads: int
    grid: int
    rope_theta: float = 10000.0
    dtype: Dtype | None = None
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
        cos, sin = _llama4_vision_tables(self.grid, head_dim, self.rope_theta,
                                         dtype=at_least_fp32(query.dtype))
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
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(LayerNorm, epsilon=self.layer_norm_eps,
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
        gelu = functools.partial(jax.nn.gelu, approximate=False)
        return gelu(self.fc2(gelu(self.fc1(hidden_states))))


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
        self.layers = [
            Llama4VisionEncoderLayer(
                cfg.hidden_size, cfg.num_heads, cfg.intermediate_size, grid,
                cfg.rope_theta, layer_norm_eps=cfg.layer_norm_eps,
                dtype=self.dtype, precision=self.precision, name=f"layers_{index}")
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
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.layernorm_post(hidden_states)[:, :-1, :]
        return self.vision_adapter(hidden_states)


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


@projectors("llama4")
@dataclasses.dataclass(frozen=True)
class Llama4Projector(ProjectorBase):
    """Llama 4's projector fields: the text width."""

    text_width: int

    def build(self) -> nn.Module:
        return Llama4ProjectorModule(text_width=self.text_width)


def _gemma4_rope_tables(positions: jax.Array, head_dim: int,
                        theta: float, *, dtype: DTypeLike) -> tuple[jax.Array, jax.Array]:
    """The 2D rotary tables of the Gemma 4 vision attention, as cos/sin.

    Each spatial dim carries its own frequencies over half the head
    (modeling_gemma4.py, Gemma4VisionRotaryEmbedding.
    compute_default_rope_parameters): the angles double up within a dim and
    the dims concatenate, so `positions` [B, P, 2] yields [B, P, head_dim].
    """
    spatial = head_dim // 2
    inv_freq = inverse_frequencies(theta, spatial, dtype=dtype)
    angles = positions.astype(dtype)[:, :, :, None] * inv_freq
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
        axis=-1) * turn for half, angle, turn in zip(halves, angles, turns, strict=True)]
    return jnp.concatenate(rotated, axis=-1)


class Gemma4ClippableLinear(nn.Dense):
    """The reference linear map with optional nontrainable activation bounds.

    Subclassing Dense keeps the existing kernel path; HF's extra linear
    name belongs to checkpoint conversion, not another numerical operation.
    """

    use_clipped_linears: bool = False

    @nn.compact
    def __call__(self, inputs):
        if self.use_clipped_linears:
            low = self.variable("constants", "input_min", lambda: jnp.array(-jnp.inf, jnp.float32))
            high = self.variable("constants", "input_max", lambda: jnp.array(jnp.inf, jnp.float32))
            inputs = jnp.clip(inputs, low.value.astype(inputs.dtype), high.value.astype(inputs.dtype))
        output = super().__call__(inputs)
        if self.use_clipped_linears:
            low = self.variable("constants", "output_min", lambda: jnp.array(-jnp.inf, jnp.float32))
            high = self.variable("constants", "output_max", lambda: jnp.array(jnp.inf, jnp.float32))
            output = jnp.clip(output, low.value.astype(output.dtype), high.value.astype(output.dtype))
        return output


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
    rms_norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_clipped_linears: bool = False
    head_dim: int | None = None

    @property
    def features_per_head(self) -> int:
        return self.hidden_size // self.num_heads if self.head_dim is None else self.head_dim

    def setup(self):
        dense = functools.partial(Gemma4ClippableLinear, use_bias=False,
                                  use_clipped_linears=self.use_clipped_linears,
                                  dtype=self.dtype, precision=self.precision)
        head_dim = self.features_per_head
        self.q_proj = dense(self.num_heads * head_dim, name="q_proj")
        self.k_proj = dense(self.num_key_value_heads * head_dim,
                            name="k_proj")
        self.v_proj = dense(self.num_key_value_heads * head_dim,
                            name="v_proj")
        self.o_proj = dense(self.hidden_size, name="o_proj")
        self.q_norm = RMSNorm(epsilon=self.rms_norm_eps, dtype=self.dtype, name="q_norm")
        self.k_norm = RMSNorm(epsilon=self.rms_norm_eps, dtype=self.dtype, name="k_norm")
        self.v_norm = RMSNorm(epsilon=self.rms_norm_eps, with_scale=False,
                              dtype=self.dtype, name="v_norm")

    def __call__(self, hidden_states, cos, sin, valid=None) -> jax.Array:
        batch, length, _ = hidden_states.shape
        head_dim = self.features_per_head
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
        scores = jnp.einsum("bqhd,bkhd->bhqk", query, key, precision=self.precision)
        if valid is not None:
            scores = jnp.where(valid[:, None, None, :], scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", probs, value, precision=self.precision)
        return self.o_proj(attended.reshape(batch, length, self.num_heads * head_dim))


class Gemma4VisionMLP(nn.Module):
    """Gated feed-forward without biases: act(gate) times up, then down.

    The reference reads the activation from the config (modeling_gemma4.py,
    Gemma4VisionMLP); only the two Gaussian forms map.
    """

    hidden_size: int
    intermediate_size: int
    hidden_act: str = "gelu_pytorch_tanh"
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_clipped_linears: bool = False

    def setup(self):
        dense = functools.partial(Gemma4ClippableLinear, use_bias=False,
                                  use_clipped_linears=self.use_clipped_linears,
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
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_clipped_linears: bool = False
    head_dim: int | None = None

    def setup(self):
        norm = functools.partial(RMSNorm, epsilon=self.rms_norm_eps, dtype=self.dtype)
        self.input_layernorm = norm(name="input_layernorm")
        self.self_attn = Gemma4VisionAttention(
            self.hidden_size, self.num_heads, self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            dtype=self.dtype, precision=self.precision,
            use_clipped_linears=self.use_clipped_linears, name="self_attn")
        self.post_attention_layernorm = norm(name="post_attention_layernorm")
        self.pre_feedforward_layernorm = norm(name="pre_feedforward_layernorm")
        self.mlp = Gemma4VisionMLP(
            self.hidden_size, self.intermediate_size, self.hidden_act,
            dtype=self.dtype, precision=self.precision,
            use_clipped_linears=self.use_clipped_linears, name="mlp")
        self.post_feedforward_layernorm = norm(name="post_feedforward_layernorm")

    def __call__(self, hidden_states, cos, sin, valid=None):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), cos, sin, valid)
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(hidden_states)


class Gemma4VisionTransformer(nn.Module):
    """The Gemma 4 vision trunk, param layout of `Gemma4VisionConfig`.

    The processor supplies [B, patches, patch_pixels] and (x, y) position IDs
    with (-1, -1) for padding. Fixed NCHW images are patchified in the same
    HWC order. Outputs retain padded soft-token slots so their shape remains
    static under JIT; callers select valid features using processor lengths.
    """

    config: "Gemma4Vision"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        cfg = self.config
        if cfg.head_dim is None and cfg.hidden_size % cfg.num_heads:
            raise ValueError(
                f"hidden_size ({cfg.hidden_size}) must split over num_heads "
                f"({cfg.num_heads})")
        head_dim = cfg.hidden_size // cfg.num_heads if cfg.head_dim is None else cfg.head_dim
        if head_dim <= 0 or head_dim % 4:
            raise ValueError(
                f"the head width ({head_dim}) must split "
                "over the two rotary dims and their halves")
        if cfg.num_heads % cfg.num_key_value_heads:
            raise ValueError(
                f"num_heads ({cfg.num_heads}) must repeat num_key_value_heads "
                f"({cfg.num_key_value_heads})")
        self.patch_embed = nn.Dense(
            cfg.hidden_size, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="patch_embed")
        self.position_table = self.param(
            "position_table", nn.initializers.normal(0.02),
            (2, cfg.position_embedding_size, cfg.hidden_size))
        self.layers = [
            Gemma4VisionEncoderLayer(
                cfg.hidden_size, cfg.intermediate_size, cfg.num_heads,
                cfg.num_key_value_heads, cfg.hidden_act,
                head_dim=cfg.head_dim,
                rms_norm_eps=cfg.rms_norm_eps,
                dtype=self.dtype, precision=self.precision,
                use_clipped_linears=cfg.use_clipped_linears, name=f"layers_{index}")
            for index in range(cfg.num_layers)]
        if cfg.standardize:
            self.std_bias = self.variable(
                "constants", "std_bias", jnp.zeros, (cfg.hidden_size,), jnp.float32
            )
            self.std_scale = self.variable(
                "constants", "std_scale", jnp.ones, (cfg.hidden_size,), jnp.float32
            )

    def __call__(self, pixel_values, pixel_position_ids=None) -> jax.Array:
        cfg = self.config
        pixels = jnp.asarray(pixel_values)
        kernel = cfg.pooling_kernel_size
        if pixels.ndim == 4:
            if pixel_position_ids is not None:
                raise ValueError("position IDs accompany patch pixels, not NCHW images")
            batch, channels, height, width = pixels.shape
            stride = cfg.patch_size * kernel
            if channels != 3 or height % stride or width % stride:
                raise ValueError(f"Gemma4 images must have three channels and sides divisible by {stride}")
            rows, columns = height // cfg.patch_size, width // cfg.patch_size
            patches = pixels.reshape(batch, channels, rows, cfg.patch_size, columns, cfg.patch_size)
            pixels = patches.transpose(0, 2, 4, 3, 5, 1).reshape(batch, rows * columns, -1)
            x = jnp.tile(jnp.arange(columns), rows)
            y = jnp.repeat(jnp.arange(rows), columns)
            pixel_position_ids = jnp.broadcast_to(jnp.stack([x, y], axis=-1), (batch, rows * columns, 2))
        if pixels.ndim != 3 or pixels.shape[-1] != 3 * cfg.patch_size ** 2:
            raise ValueError("Gemma4 patch pixels must be [B, patches, 3 * patch_size**2]")
        if pixel_position_ids is None or pixel_position_ids.shape != (*pixels.shape[:2], 2):
            raise ValueError("Gemma4 patch pixels require aligned [B, patches, 2] position IDs")
        if not jnp.issubdtype(pixel_position_ids.dtype, jnp.integer):
            raise ValueError("pixel_position_ids must be integers")
        if pixels.shape[1] % kernel ** 2:
            raise ValueError("patch count must divide into whole pooling blocks")
        valid = (pixel_position_ids >= 0).all(axis=-1)
        safe = jnp.maximum(pixel_position_ids, 0)
        hidden_states = self.patch_embed(2 * (pixels - 0.5))
        table = jnp.asarray(self.position_table, hidden_states.dtype)
        positional = table[0, safe[..., 0]] + table[1, safe[..., 1]]
        hidden_states = hidden_states + jnp.where(valid[..., None], positional, 0)
        head_dim = cfg.head_dim or cfg.hidden_size // cfg.num_heads
        cos, sin = _gemma4_rope_tables(pixel_position_ids, head_dim, cfg.rope_theta,
                                       dtype=at_least_fp32(hidden_states.dtype))
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, valid)
        output_length = pixels.shape[1] // kernel ** 2
        width = safe[..., 0].max(axis=-1, keepdims=True) + 1
        indices = safe[..., 0] // kernel + (width // kernel) * (safe[..., 1] // kernel)
        values = jnp.where(valid[..., None], hidden_states, 0).astype(jnp.float32) / kernel ** 2
        # Segment sums avoid the reference pooler's [patches, soft_tokens]
        # one-hot matrix while preserving its position-indexed block average.
        pooled = jax.vmap(lambda value, index: jax.ops.segment_sum(value, index, output_length))(
            values, indices
        )
        pooled = pooled * cfg.hidden_size ** 0.5
        if cfg.standardize:
            pooled = (pooled - self.std_bias.value) * self.std_scale.value
        return pooled.astype(hidden_states.dtype)


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
    standardize: bool = False
    use_clipped_linears: bool = False
    head_dim: int | None = None
    """None retains hidden_size // num_heads; checkpoints may project a wider head."""

    def build(self) -> nn.Module:
        return Gemma4VisionTransformer(self)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.pooling_kernel_size)


class Gemma4ProjectorModule(nn.Module):
    """Soft tokens to text width: a scale-free RMS norm, then the map.

    The reference keeps this beside the tower (modeling_gemma4.py,
    Gemma4MultimodalEmbedder): the pre-projection norm carries no weight and
    the projection no bias.
    """

    text_width: int
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
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
    """Gemma 4's projector fields: the decoder width and the norm epsilon."""

    text_width: int
    norm_eps: float = 1e-6

    def build(self) -> nn.Module:
        return Gemma4ProjectorModule(text_width=self.text_width, norm_eps=self.norm_eps)


def _qwen35_interp_taps(index: jax.Array, size: int | jax.Array, side: int) -> tuple[jax.Array, jax.Array]:
    """Bilinear taps into a `side`-long table for positions along one axis.

    The closed form of the align-corners linspace the reference resamples
    with (vision_utils.py, _interpolation_axis_taps_weights): each position
    reads its floor and ceiling with the linear hat weights.
    """
    src = index.astype(jnp.float32) * (side - 1) / jnp.maximum(size - 1, 1)
    floor = jnp.floor(src).astype(jnp.int32)
    distance = (src - floor.astype(jnp.float32))[..., None]
    taps = jnp.stack([floor, floor + 1], axis=-1).clip(0, side - 1)
    weights = jnp.concatenate([1 - distance, distance], axis=-1).clip(min=0)
    return taps, weights


def _qwen35_pos_embeds(table: jax.Array, rows: jax.Array, cols: jax.Array,
                       grid_height: int | jax.Array, grid_width: int | jax.Array) -> jax.Array:
    """The learned table resampled to one image's patch grid.

    Bilinear over the square table's rows and columns separately, outer
    product of the taps (vision_utils.py,
    get_vision_interpolation_indices_and_weights).
    """
    side = round(len(table) ** 0.5)
    row_taps, row_weights = _qwen35_interp_taps(rows, grid_height, side)
    col_taps, col_weights = _qwen35_interp_taps(cols, grid_width, side)
    indices = (row_taps[..., :, None] * side + col_taps[..., None, :]).reshape(*rows.shape, 4)
    weights = (row_weights[..., :, None] * col_weights[..., None, :]).reshape(*rows.shape, 4)
    return (table[indices] * weights[..., None]).sum(axis=-2)


def _grid_rope_tables(positions: jax.Array, head_dim: int, theta: float = 10000.0, *,
                      dtype: DTypeLike) -> tuple[jax.Array, jax.Array]:
    """The 2D rotary tables of a patch grid's attention, as cos/sin.

    Heights then widths share one frequency table over half the head
    (modeling_qwen3_5.py, Qwen3_5VisionRotaryEmbedding.forward; DeepSeek-V4.1
    vision.py:8-15), doubled the way the text rope doubles its pairs.
    """
    dim = head_dim // 2
    inv_freq = inverse_frequencies(theta, dim, dtype=dtype)
    flat = (positions.astype(dtype)[..., None] * inv_freq).reshape(
        *positions.shape[:-1], -1)
    doubled = jnp.concatenate([flat, flat], axis=-1)
    return jnp.cos(doubled), jnp.sin(doubled)


def _grid_rope(values: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """The half rotation, broadcast over the heads (modeling_qwen3_5.py,
    apply_rotary_pos_emb_vision; DeepSeek-V4.1 vision.py:18-21)."""
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
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.qkv = nn.Dense(3 * self.hidden_size, use_bias=True, dtype=self.dtype,
                            precision=self.precision, name="qkv")
        self.proj = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype,
                             precision=self.precision, name="proj")

    def __call__(self, hidden_states, cos, sin, mask=None) -> jax.Array:
        batch, length, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads
        fused = self.qkv(hidden_states).reshape(batch, length, 3, self.num_heads, head_dim)
        query, key, value = (fused[:, :, 0], fused[:, :, 1], fused[:, :, 2])
        query = _grid_rope(query, cos[:, :, None, :], sin[:, :, None, :])
        key = _grid_rope(key, cos[:, :, None, :], sin[:, :, None, :])
        attended = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision, mask=mask)
        return self.proj(attended.reshape(batch, length, self.hidden_size))


class Qwen35VisionBlock(nn.Module):
    """Pre-norm attention over pre-norm shared MLP, both residual."""

    hidden_size: int
    intermediate_size: int
    num_heads: int
    hidden_act: str = "gelu_pytorch_tanh"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(LayerNorm, epsilon=1e-6, dtype=self.dtype)
        self.norm1 = norm(name="norm1")
        self.attn = Qwen35VisionAttention(
            self.hidden_size, self.num_heads, dtype=self.dtype,
            precision=self.precision, name="attn")
        self.norm2 = norm(name="norm2")
        self.mlp = MLP(self.hidden_size, self.intermediate_size,
                       activation=self.hidden_act, dtype=self.dtype,
                       precision=self.precision, name="mlp")

    def __call__(self, hidden_states, cos, sin, mask=None):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin, mask)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen35VisionTransformer(nn.Module):
    """The Qwen 3.5 vision trunk, param layout of `Qwen3_5VisionConfig`.

    Packed processor pixels are padded to [B, patches, patch_pixels] and
    accompanied by each row's [time, height, width] patch grid. Attention stays
    inside each frame and excludes padded keys. Fixed NCHW images use the
    processor's channel-then-time patch order. The return value is the
    pre-merge sequence; the projector owns the merger.
    """

    config: "Qwen35Vision"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        cfg = self.config
        if cfg.hidden_size % cfg.num_heads:
            raise ValueError(
                f"hidden_size ({cfg.hidden_size}) must split over num_heads "
                f"({cfg.num_heads})")
        self.patch_embed = nn.Dense(
            cfg.hidden_size, use_bias=True, dtype=self.dtype,
            precision=self.precision, name="patch_embed")
        self.position_table = nn.Embed(cfg.num_position_embeddings, cfg.hidden_size,
                                       dtype=self.dtype, name="position_table")
        self.blocks = [
            Qwen35VisionBlock(
                cfg.hidden_size, cfg.intermediate_size, cfg.num_heads,
                cfg.hidden_act, dtype=self.dtype, precision=self.precision,
                name=f"blocks_{index}")
            for index in range(cfg.depth)]

    def __call__(self, pixel_values, grid_thw=None) -> jax.Array:
        cfg = self.config
        pixels = jnp.asarray(pixel_values)
        merge = cfg.spatial_merge_size
        patch = cfg.patch_size
        if pixels.ndim == 4:
            if grid_thw is not None:
                raise ValueError("grid_thw accompanies packed pixels, not NCHW images")
            batch, channels, height, width = pixels.shape
            stride = patch * merge
            if channels != cfg.in_channels or height % stride or width % stride:
                raise ValueError(
                    f"pixel_values must be {cfg.in_channels}-channel images tiled by {stride}px blocks"
                )
            rows, columns = height // patch, width // patch
            blocks = pixels.reshape(batch, channels, rows // merge, merge, patch,
                                    columns // merge, merge, patch)
            blocks = blocks.transpose(0, 2, 5, 3, 6, 1, 4, 7)
            flat = blocks.reshape(batch, rows * columns, channels, 1, patch, patch)
            frames = jnp.broadcast_to(flat, (batch, rows * columns, channels,
                                            cfg.temporal_patch_size, patch, patch))
            pixels = frames.reshape(batch, rows * columns, -1)
            grid_thw = jnp.broadcast_to(jnp.array([1, rows, columns], jnp.int32), (batch, 3))
        if pixels.ndim != 3 or pixels.shape[-1] != cfg.in_channels * cfg.temporal_patch_size * patch ** 2:
            raise ValueError(
                "packed Qwen pixels must be [B, patches, C * temporal_patch_size * patch_size**2]"
            )
        batch, length, _ = pixels.shape
        if grid_thw is None or grid_thw.shape != (batch, 3):
            raise ValueError("packed Qwen pixels require aligned [B, 3] grid_thw")
        grid = jnp.asarray(grid_thw)
        heights, widths = grid[:, 1:2], grid[:, 2:3]
        area = heights * widths
        within = jnp.arange(length)[None, :] % area
        blocks_wide = widths // merge
        rows = (within // (merge * merge * blocks_wide)) * merge + (within // merge) % merge
        columns = ((within // (merge * merge)) % blocks_wide) * merge + within % merge
        hidden_states = self.patch_embed(pixels)
        table = jnp.asarray(self.position_table.embedding, hidden_states.dtype)
        hidden_states = hidden_states + _qwen35_pos_embeds(table, rows, columns, heights, widths)
        cos, sin = _grid_rope_tables(jnp.stack([rows, columns], axis=-1), cfg.hidden_size // cfg.num_heads,
                                     dtype=at_least_fp32(hidden_states.dtype))
        valid = jnp.arange(length)[None, :] < jnp.prod(grid, axis=1, keepdims=True)
        frames = jnp.arange(length)[None, :] // area
        keep = (frames[:, :, None] == frames[:, None, :]) & valid[:, None, :]
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin, keep[:, None])
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
    num_position_embeddings: int = 2304

    def build(self) -> nn.Module:
        return Qwen35VisionTransformer(self)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.spatial_merge_size,
                             channels=self.in_channels)


class Qwen35ProjectorModule(nn.Module):
    """The merger: norm, group each merge block, map through an exact GELU.

    The reference keeps this inside the vision model (modeling_qwen3_5.py,
    Qwen3_5VisionPatchMerger); it reads the trunk sequence here so the tower
    and projector parities stay separate.
    """

    hidden_size: int
    spatial_merge_size: int
    out_hidden_size: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.norm = LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm")
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


class DeepseekV41VisionBlock(nn.Module):
    """Pre-norm full attention under the grid rotary, then a pre-norm SwiGLU
    feed-forward, both residual (DeepSeek-V4.1 vision.py:46-84).

    One biased map carries queries, keys and values; the rotary runs in fp32
    and casts back, as the reference's does.
    """

    hidden_size: int
    num_heads: int
    intermediate_size: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, hidden_states, cos, sin):
        dense = functools.partial(nn.Dense, dtype=self.dtype, precision=self.precision)
        norm = functools.partial(RMSNorm, epsilon=1e-6, dtype=self.dtype)
        batch, length, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads
        fused = dense(3 * self.hidden_size, name="wqkv")(norm(name="norm1")(hidden_states)).reshape(
            batch, length, 3, self.num_heads, head_dim)
        query, key, value = (fused[:, :, 0], fused[:, :, 1], fused[:, :, 2])
        query, key = (_grid_rope(part, cos[:, :, None, :], sin[:, :, None, :]).astype(part.dtype)
                      for part in (query, key))
        attended = scaled_dot_product_attention(query, key, value, dtype=self.dtype, precision=self.precision)
        hidden_states = hidden_states + dense(self.hidden_size, name="wo")(
            attended.reshape(batch, length, self.hidden_size))
        gate, up = jnp.split(dense(2 * self.intermediate_size, use_bias=False, name="w1")(
            norm(name="norm2")(hidden_states)), 2, axis=-1)
        return hidden_states + dense(self.hidden_size, use_bias=False, name="w2")(jax.nn.silu(gate) * up)


class DeepseekV41VisionTransformer(nn.Module):
    """DeepSeek-V4.1's ViT over NCHW images (vision.py:87-103).

    Each `patch_size` square's pixels, channel-major, map through one biased
    projection; the blocks attend over the whole grid with its 2D rotary, and
    a final RMS norm closes. The return value keeps the grid,
    `[images, rows, columns, hidden_size]`, which the aligner groups.
    """

    config: "DeepseekV41Vision"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, pixel_values) -> jax.Array:
        cfg = self.config
        pixels = jnp.asarray(pixel_values)
        patch = cfg.patch_size
        images, channels, height, width = pixels.shape
        if height % patch or width % patch:
            raise ValueError(f"pixel_values must tile into {patch}px patches, got {height}x{width}")
        rows, columns = height // patch, width // patch
        patches = pixels.reshape(images, channels, rows, patch, columns, patch).transpose(0, 2, 4, 1, 3, 5)
        hidden_states = nn.Dense(cfg.hidden_size, dtype=self.dtype, precision=self.precision,
                                 name="patch_embed")(patches.reshape(images, rows * columns, -1))
        grid = jnp.stack(jnp.meshgrid(jnp.arange(rows), jnp.arange(columns), indexing="ij"), axis=-1)
        cos, sin = _grid_rope_tables(grid.reshape(1, rows * columns, 2),
                                     cfg.hidden_size // cfg.num_attention_heads, cfg.rope_theta,
                                     dtype=at_least_fp32(hidden_states.dtype))
        for index in range(cfg.num_hidden_layers):
            hidden_states = DeepseekV41VisionBlock(
                cfg.hidden_size, cfg.num_attention_heads, cfg.intermediate_size,
                dtype=self.dtype, precision=self.precision, name=f"blocks_{index}")(hidden_states, cos, sin)
        hidden_states = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(hidden_states)
        return hidden_states.reshape(images, rows, columns, cfg.hidden_size)


@towers("deepseek_v41")
@dataclasses.dataclass(frozen=True)
class DeepseekV41Vision(TowerBase):
    """DeepSeek-V4.1's ViT geometry, under its vision_config's names."""

    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    rope_theta: float = 10000.0

    def build(self) -> nn.Module:
        return DeepseekV41VisionTransformer(self)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, channels=3)


class DeepseekV41ProjectorModule(nn.Module):
    """DeepSeek-V4.1's aligner and image span (vision.py:106-119,
    model.py:1228-1239).

    The patch grid is zero-padded at its bottom and right to whole
    `downsample_ratio` squares, each square's features concatenate
    channel-major (unfold's order), and a two-layer exact-GELU map takes
    them to the decoder width. The span the decoder reads is a learned start
    vector, each row of aligned features followed by a learned newline
    vector, and a learned end vector: `rows * (columns + 1) + 2` positions.
    """

    vision_width: int
    downsample_ratio: int
    out_width: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, image_features) -> jax.Array:
        ratio = self.downsample_ratio
        images, rows, columns, _ = image_features.shape
        padded = jnp.pad(image_features, ((0, 0), (0, -rows % ratio), (0, -columns % ratio), (0, 0)))
        high, wide = padded.shape[1] // ratio, padded.shape[2] // ratio
        squares = padded.reshape(images, high, ratio, wide, ratio, self.vision_width).transpose(
            0, 1, 3, 5, 2, 4
        )
        dense = functools.partial(nn.Dense, self.out_width, dtype=self.dtype, precision=self.precision)
        aligned = dense(name="w2")(jax.nn.gelu(dense(name="w1")(
            squares.reshape(images, high, wide, -1)), approximate=False))
        start, newline, end = (
            self.param(name, nn.initializers.normal(1.0), (self.out_width,), jnp.float32).astype(
                aligned.dtype
            )
            for name in ("image_start", "image_newline", "image_end")
        )
        lines = jnp.concatenate(
            [aligned, jnp.broadcast_to(newline, (images, high, 1, self.out_width))], axis=2
        )
        return jnp.concatenate([jnp.broadcast_to(start, (images, 1, self.out_width)),
                                lines.reshape(images, high * (wide + 1), self.out_width),
                                jnp.broadcast_to(end, (images, 1, self.out_width))], axis=1)


@projectors("deepseek_v41")
@dataclasses.dataclass(frozen=True)
class DeepseekV41Projector(ProjectorBase):
    """DeepSeek-V4.1's aligner fields: the ViT width, the side of the squares
    it groups, and the decoder width."""

    vision_width: int
    downsample_ratio: int
    out_width: int

    def build(self) -> nn.Module:
        return DeepseekV41ProjectorModule(vision_width=self.vision_width,
                                          downsample_ratio=self.downsample_ratio,
                                          out_width=self.out_width)


_SIGLIP_TENSORS = {
    "embeddings.patch_embedding.weight": ("patch_embedding", "kernel"),
    "embeddings.patch_embedding.bias": ("patch_embedding", "bias"),
    "embeddings.position_embedding.weight": ("position_embedding", "embedding"),
    "post_layernorm.weight": ("post_layernorm", "scale"),
    "post_layernorm.bias": ("post_layernorm", "bias"),
}


def _encoder_layer_path(parts, root: str, norms: tuple[str, ...],
                        projections: tuple[str, ...]) -> tuple[str, ...] | None:
    """`<root>.layers.N...` into the path of a CLIP-style encoder layer: two
    layer norms, biased attention maps and a biased fc1/fc2 MLP."""
    if (len(parts) < 5 or parts[:2] != [root, "layers"] or not parts[2].isdigit()
            or parts[-1] not in ("weight", "bias")):
        return None
    layer, module, leaf = f"layers_{parts[2]}", parts[3], parts[-1]
    if len(parts) == 5 and module in norms:
        return (layer, module, "scale" if leaf == "weight" else "bias")
    if len(parts) == 6 and ((module == "self_attn" and parts[4] in projections)
                            or (module == "mlp" and parts[4] in ("fc1", "fc2"))):
        return (layer, module, parts[4], "kernel" if leaf == "weight" else "bias")
    return None


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


def translate_siglip_vision_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """SigLIP vision parameters at the requested storage precision."""
    return translate_parameters(hf_tensors, siglip_vision_path, param_dtype)


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
        hf_name.split("."), "model", ("input_layernorm", "post_attention_layernorm"),
        ("q_proj", "k_proj", "v_proj", "o_proj"))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_llama4_vision_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """Llama 4 vision parameters at the requested storage precision."""
    return translate_parameters(hf_tensors, llama4_vision_path, param_dtype)


_PROJECTOR_PATHS: dict[str, dict[str, tuple[str, ...]]] = {
    "gemma": {
        "mm_soft_emb_norm.weight": ("mm_soft_emb_norm", "scale"),
        "mm_input_projection_weight": ("mm_input_projection", "kernel"),
    },
    "llama4": {"linear_1.weight": ("linear", "kernel")},
    "gemma4": {"embedding_projection.weight": ("projection", "kernel")},
    "qwen3_5": {
        "norm.weight": ("norm", "scale"), "norm.bias": ("norm", "bias"),
        "linear_fc1.weight": ("fc1", "kernel"), "linear_fc1.bias": ("fc1", "bias"),
        "linear_fc2.weight": ("fc2", "kernel"), "linear_fc2.bias": ("fc2", "bias"),
    },
    "gemma3n": {
        "embedding.weight": ("embedding", "embedding"),
        "hard_embedding_norm.weight": ("hard_embedding_norm", "scale"),
        "soft_embedding_norm.weight": ("soft_embedding_norm", "scale"),
        "embedding_projection.weight": ("embedding_projection", "kernel"),
    },
    # The aligner's maps under its `aligner.` prefix, and the span's learned
    # vectors, which the release keeps at its top level (model.py:1215-1222).
    "deepseek_v41": {
        "w1.weight": ("w1", "kernel"), "w1.bias": ("w1", "bias"),
        "w2.weight": ("w2", "kernel"), "w2.bias": ("w2", "bias"),
        "image_start": ("image_start",), "image_newline": ("image_newline",),
        "image_end": ("image_end",),
    },
}


def projector_weight_path(kind: str, name: str) -> tuple[str, ...]:
    """The projector's canonical tensor path, shared by load and source export."""
    if kind == "qwen3_5":
        name = name.removeprefix("merger.")
    if kind not in _PROJECTOR_PATHS or name not in _PROJECTOR_PATHS[kind]:
        raise ValueError(f"unknown {kind} projector tensor {name!r}")
    return _PROJECTOR_PATHS[kind][name]


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


def translate_llama4_projector_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """Llama 4's outer projector map at the requested storage precision."""
    return translate_parameters(hf_tensors, lambda name: projector_weight_path("llama4", name), param_dtype)


def _vision_section(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """The vision half of a wrapper config, or a bare vision config."""
    return records.record(hf_config.get("vision_config", hf_config), "vision_config")


def _image_size(value: object, field: str) -> int:
    """A square image size as one side: an int, or a pair with equal sides."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(
                f"{field} {list(value)!r} is not square, this trunk tiles squares")
        value = value[0]
    return records.integer(value, field)


def translate_siglip_vision_config(hf_config: Mapping[str, object]) -> Mapping[str, object]:
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
        "kind": "siglip",
        "hidden_size": hidden,
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "num_heads": records.integer(vision["num_attention_heads"], "num_attention_heads"),
        "image_size": _image_size(vision.get("image_size", 224), "image_size"),
        "patch_size": records.integer(patch, "patch_size"),
        "num_channels": records.integer(vision.get("num_channels", 3), "num_channels"),
        "hidden_act": activation,
        "layer_norm_eps": records.number(vision.get("layer_norm_eps", 1e-6), "layer_norm_eps"),
    }


def translate_llama4_vision_config(hf_config: Mapping[str, object]) -> Mapping[str, object]:
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
        "kind": "llama4",
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
    }


def translate_gemma_projector_config(vision: Mapping[str, object], text_width: int,
                                     mm_tokens_per_image: object) -> Mapping[str, object]:
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
        "kind": "gemma",
        "text_width": int(text_width),
        "patches_per_side": patches,
        "tokens_per_side": side,
        "norm_eps": records.number(vision.get("layer_norm_eps", 1e-6), "layer_norm_eps"),
    }


def translate_llama4_projector_config(text_width: int) -> Mapping[str, object]:
    """A Llama 4 wrapper's projector fields: the text width."""
    return {"kind": "llama4", "text_width": int(text_width)}


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


def _gemma4_vision_layer_path(parts) -> tuple[str, ...] | None:
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
    if (len(parts) == 6 and parts[5] in ("input_min", "input_max", "output_min", "output_max")
            and ((parts[3] == "self_attn" and parts[4] in _GEMMA4_VISION_PROJECTIONS)
                 or (parts[3] == "mlp" and parts[4] in _GEMMA4_VISION_MLP))):
        return (layer, parts[3], parts[4], parts[5])
    if (len(parts) == 6 and parts[3] == "self_attn" and parts[5] == "weight"
            and parts[4] in ("q_norm", "k_norm")):
        # The value norm carries no scale (modeling_gemma4.py,
        # Gemma4VisionAttention), so a weight under its name is unknown.
        return (layer, "self_attn", parts[4], "scale")
    return None


def gemma4_vision_path(hf_name: str) -> tuple[str, ...] | None:
    """One Gemma 4 vision tensor name into its collection and trunk path.

    The rotary tables are buffers the checkpoint leaves out, recomputed from
    the grid at call time. Anything else unknown raises ValueError.
    """
    path = _GEMMA4_VISION_TENSORS.get(hf_name) or _gemma4_vision_layer_path(
        hf_name.split("."))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    collection = (
        "constants"
        if path[-1] in ("std_bias", "std_scale", "input_min", "input_max", "output_min", "output_max")
        else "params"
    )
    return (collection, *path)


def translate_gemma4_vision_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """Gemma 4 parameters plus native FP32 frozen and clipping buffers."""
    return translate_parameters(hf_tensors, gemma4_vision_path, param_dtype)


def translate_gemma4_projector_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """A Gemma 4 embedder's map into its parameter tree.

    The pre-projection norm carries no scale, so the projection weight is
    the only tensor.
    """
    if set(hf_tensors) != set(_PROJECTOR_PATHS["gemma4"]):
        raise ValueError(f"unknown tensor names {sorted(hf_tensors)}")
    return translate_parameters(hf_tensors, lambda name: projector_weight_path("gemma4", name), param_dtype)


def translate_gemma4_vision_config(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """A Gemma4VisionConfig into a Gemma4Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The head
    width comes from head_dim or the hidden size divided by the head count.
    Standardization and activation clipping retain
    their reference buffers outside the trainable parameter collection.
    """
    vision = _vision_section(hf_config)
    hidden = records.integer(vision["hidden_size"], "hidden_size")
    heads = records.integer(vision["num_attention_heads"], "num_attention_heads")
    head_dim = vision.get("head_dim", hidden // heads)
    if records.integer(head_dim, "head_dim") <= 0 or records.integer(head_dim, "head_dim") % 4:
        raise ValueError(
            f"head_dim ({head_dim}) must split into the two rotary dimensions and their halves")
    activation = str(vision.get("hidden_activation", vision.get("hidden_act",
                                                                "gelu_pytorch_tanh")))
    if activation not in ("gelu_pytorch_tanh", "gelu"):
        raise ValueError(
            f"hidden_activation {activation!r} is not expressible: this trunk "
            "runs gelu_pytorch_tanh or gelu")
    if vision.get("attention_bias", False):
        raise ValueError("attention_bias=True needs biased maps this trunk lacks")
    if records.number(vision.get("attention_dropout", 0.0), "attention_dropout"):
        raise ValueError("attention_dropout is training-time; this trunk runs eval")
    if "output_proj_dims" in vision:
        raise ValueError(
            f"output_proj_dims ({vision['output_proj_dims']!r}) changes the "
            "projector width this record leaves to the text width")
    rope = records.record(vision.get("rope_parameters") or {}, "rope_parameters")
    if rope.get("rope_type", "default") != "default":
        raise ValueError(
            f"rope_type {rope.get('rope_type')!r} is not expressible: this trunk "
            "runs the default 2D rotary")
    return {
        "kind": "gemma4",
        "hidden_size": hidden,
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "num_heads": heads,
        "num_key_value_heads": records.integer(
            vision.get("num_key_value_heads", heads), "num_key_value_heads"
        ),
        "head_dim": records.integer(head_dim, "head_dim"),
        "patch_size": records.integer(vision.get("patch_size", 16), "patch_size"),
        "pooling_kernel_size": records.integer(vision.get("pooling_kernel_size", 3), "pooling_kernel_size"),
        "position_embedding_size": records.integer(
            vision.get("position_embedding_size", 10240), "position_embedding_size"
        ),
        "hidden_act": activation,
        "rms_norm_eps": records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
        "rope_theta": records.number(rope.get("rope_theta", vision.get("rope_theta", 100.0)), "rope_theta"),
        "standardize": bool(vision.get("standardize", False)),
        "use_clipped_linears": bool(vision.get("use_clipped_linears", False)),
    }


def export_gemma4_vision_config(tower: Gemma4Vision) -> Mapping[str, object]:
    """A Gemma4Vision value as the Gemma4VisionConfig that
    `translate_gemma4_vision_config` reads back to it. An unset head_dim is
    written as the width it stands for."""
    return {
        "model_type": "gemma4_vision",
        "hidden_size": tower.hidden_size,
        "intermediate_size": tower.intermediate_size,
        "num_hidden_layers": tower.num_layers,
        "num_attention_heads": tower.num_heads,
        "num_key_value_heads": tower.num_key_value_heads,
        "head_dim": tower.hidden_size // tower.num_heads if tower.head_dim is None else tower.head_dim,
        "patch_size": tower.patch_size,
        "pooling_kernel_size": tower.pooling_kernel_size,
        "position_embedding_size": tower.position_embedding_size,
        "hidden_activation": tower.hidden_act,
        "rms_norm_eps": tower.rms_norm_eps,
        "rope_parameters": {"rope_type": "default", "rope_theta": tower.rope_theta},
        "standardize": tower.standardize,
        "use_clipped_linears": tower.use_clipped_linears,
    }


def translate_gemma4_projector_config(vision: Mapping[str, object],
                                      text_width: int) -> Mapping[str, object]:
    """A Gemma 4 wrapper's projector fields: decoder width, norm epsilon."""
    return {
        "kind": "gemma4",
        "text_width": int(text_width),
        "norm_eps": records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
    }


_QWEN35_VISION_TENSORS = {
    "patch_embed.proj.weight": ("patch_embed", "kernel"),
    "patch_embed.proj.bias": ("patch_embed", "bias"),
    "pos_embed.weight": ("position_table", "embedding"),
}


def _qwen35_vision_block_path(parts) -> tuple[str, ...] | None:
    """`blocks.N...` into the block's path."""
    if (
        len(parts) < 4
        or parts[0] != "blocks"
        or not parts[1].isdigit()
        or parts[-1] not in ("weight", "bias")
    ):
        return None
    block, leaf = f"blocks_{parts[1]}", parts[-1]
    if len(parts) == 4 and parts[2] in ("norm1", "norm2"):
        return (block, parts[2], "scale" if leaf == "weight" else "bias")
    if len(parts) == 5 and (parts[2], parts[3]) in (("attn", "qkv"), ("attn", "proj"),
                                                     ("mlp", "linear_fc1"), ("mlp", "linear_fc2")):
        return (block, parts[2], parts[3].removeprefix("linear_"), "kernel" if leaf == "weight" else "bias")
    return None


def qwen35_vision_path(hf_name: str) -> tuple[str, ...] | None:
    """One Qwen 3.5 vision tensor name into its path in a trunk tree.

    The merger lives under its own prefix and maps with the projector; the
    rotary table is a buffer the checkpoint leaves out. Anything else unknown
    raises ValueError.
    """
    if hf_name.split(".")[0] == "merger":
        return None
    path = _QWEN35_VISION_TENSORS.get(hf_name) or _qwen35_vision_block_path(
        hf_name.split("."))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_qwen35_vision_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """Qwen 3.5 vision tensors into a trunk parameter tree.

    The patch convolution carries [out, in, time, h, w] and lands as one map;
    the trunk's buffer already holds the viewed channel order, so the rows
    are a plain reshape and every other leaf rides the shared transpose.
    """
    rest = {name: tensor for name, tensor in hf_tensors.items()
            if name != "patch_embed.proj.weight"}
    params = translate_parameters(rest, qwen35_vision_path, param_dtype)
    conv = checkpoint_array(hf_tensors["patch_embed.proj.weight"], param_dtype)
    params.setdefault("patch_embed", {})["kernel"] = np.ascontiguousarray(
        conv.transpose(1, 2, 3, 4, 0).reshape(-1, conv.shape[0]))
    return params


def translate_qwen35_projector_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """A Qwen 3.5 merger's tensors at the requested storage precision."""
    return translate_parameters(hf_tensors, lambda name: projector_weight_path("qwen3_5", name), param_dtype)


def translate_qwen35_vision_config(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """A Qwen3_5VisionConfig into a Qwen35Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The
    released 0.8B checkpoint spells the tower's model_type qwen3_5, which the
    config class rewrites on load; both spellings map. The position table
    must be square, and the activation one the shared MLP runs.
    """
    vision = _vision_section(hf_config)
    if vision.get("model_type", "qwen3_5_vision") not in _QWEN35_VISION_TYPES:
        raise ValueError(
            f"vision model_type {vision.get('model_type')!r} is not the Qwen 3.5 tower")
    table = records.integer(vision["num_position_embeddings"], "num_position_embeddings")
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
        "depth": records.integer(vision["depth"], "depth"),
        "hidden_size": records.integer(vision["hidden_size"], "hidden_size"),
        "hidden_act": activation,
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_heads": records.integer(vision["num_heads"], "num_heads"),
        "in_channels": records.integer(vision.get("in_channels", 3), "in_channels"),
        "patch_size": records.integer(vision["patch_size"], "patch_size"),
        "spatial_merge_size": records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size"),
        "temporal_patch_size": records.integer(vision["temporal_patch_size"], "temporal_patch_size"),
        "num_position_embeddings": table,
    }


def translate_qwen35_projector_config(hf_config: Mapping[str, object],
                                      text_width: int) -> Mapping[str, object]:
    """A Qwen 3.5 wrapper's projector fields: trunk width, merge, output.

    The merged features enter the text embeddings directly, so a merger width
    beside the decoder width refuses.
    """
    vision = _vision_section(hf_config)
    merged = records.integer(vision["out_hidden_size"], "out_hidden_size")
    if merged != int(text_width):
        raise ValueError(
            f"out_hidden_size ({merged}) is not the decoder width ({text_width}), "
            "the merger output enters the text embeddings as it is")
    return {
        "kind": "qwen3_5",
        "vision_width": records.integer(vision["hidden_size"], "hidden_size"),
        "merge_size": records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size"),
        "out_width": merged,
    }


_DEEPSEEK_V41_VISION_TENSORS = {
    "patch_embed.proj.weight": ("patch_embed", "kernel"),
    "patch_embed.proj.bias": ("patch_embed", "bias"),
    "norm.weight": ("norm", "scale"),
}


def deepseek_v41_vision_path(hf_name: str) -> tuple[str, ...]:
    """One DeepSeek-V4.1 ViT tensor name, its `vision.` prefix off, into its
    path in the trunk tree; anything else raises ValueError."""
    path = _DEEPSEEK_V41_VISION_TENSORS.get(hf_name)
    parts = hf_name.split(".")
    if path is None and len(parts) > 3 and parts[0] == "blocks" and parts[1].isdigit():
        block, leaf = f"blocks_{parts[1]}", parts[-1]
        if len(parts) == 4 and parts[2] in ("norm1", "norm2") and leaf == "weight":
            path = (block, parts[2], "scale")
        elif (len(parts) == 5 and (parts[2], parts[3]) in (("attn", "wqkv"), ("attn", "wo"), ("mlp", "w1"),
                                                           ("mlp", "w2")) and leaf in ("weight", "bias")):
            path = (block, parts[3], "kernel" if leaf == "weight" else "bias")
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_deepseek_v41_vision_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """DeepSeek-V4.1 ViT parameters at the requested storage precision."""
    return translate_parameters(hf_tensors, deepseek_v41_vision_path, param_dtype)


def translate_deepseek_v41_projector_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Variables:
    """DeepSeek-V4.1's aligner and span vectors at the requested storage precision."""
    return translate_parameters(hf_tensors, lambda name: projector_weight_path("deepseek_v41", name),
                                 param_dtype)


def translate_deepseek_v41_vision_config(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """A DeepSeek-V4.1 vision_config into a DeepseekV41Vision value's fields.

    The image-size fields (max_image_tokens, min_pixels, max_wh_ratio) plan
    the processor's resize and the downsample ratio is the aligner's, so the
    trunk reads neither.
    """
    vision = _vision_section(hf_config)
    if vision.get("model_type", "deepseek_v41_vision") != "deepseek_v41_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not DeepSeek-V4.1's ViT")
    width = records.integer(vision["hidden_size"], "hidden_size")
    heads = records.integer(vision["num_attention_heads"], "num_attention_heads")
    if width % heads or (width // heads) % 4:
        raise ValueError(f"hidden_size {width} over {heads} heads leaves no head width the 2D rotary "
                         "splits into height and width pairs")
    return {
        "kind": "deepseek_v41",
        "num_hidden_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "hidden_size": width,
        "num_attention_heads": heads,
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "patch_size": records.integer(vision["patch_size"], "patch_size"),
        "rope_theta": records.number(vision.get("rope_theta", 10000.0), "rope_theta"),
    }


def translate_deepseek_v41_projector_config(hf_config: Mapping[str, object],
                                            text_width: int) -> Mapping[str, object]:
    """DeepSeek-V4.1's aligner fields: the ViT width, its downsample ratio and
    the decoder width its rows and span vectors enter."""
    vision = _vision_section(hf_config)
    return {
        "kind": "deepseek_v41",
        "vision_width": records.integer(vision["hidden_size"], "hidden_size"),
        "downsample_ratio": records.integer(vision.get("downsample_ratio", 3), "downsample_ratio"),
        "out_width": int(text_width),
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

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(channels=self.in_chans)


class Gemma3nProjectorModule(nn.Module):
    """Gemma3nMultimodalEmbedder's vision hard and soft token paths."""

    vision_width: int
    text_width: int
    vocab_size: int = 128
    vocab_offset: int = 262144
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if min(self.vision_width, self.text_width, self.vocab_size) < 1 or self.vocab_offset < 0:
            raise ValueError(
                "vision/text widths and vocab_size must be positive; vocab_offset is nonnegative"
            )
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
        return self.soft_embeddings(features * jnp.asarray(self.vision_width ** 0.5, features.dtype))

    def soft_embeddings(self, features):
        """The reference embedder over soft features, without vision-only scaling."""
        features = jnp.asarray(features)
        if features.ndim != 3 or features.shape[-1] != self.vision_width:
            raise ValueError(f"soft features must be [B, N, {self.vision_width}], got {features.shape}")
        if not jnp.issubdtype(features.dtype, jnp.floating):
            raise ValueError("soft features must be floating point")
        if self.is_initializing():
            # Both paths belong to one checkpoint even when init starts with
            # image features. Linen creates an embedding only when called.
            self.hard_embedding_norm(self.embedding(jnp.zeros((1, 1), jnp.int32)))
        return self.embedding_post_projection_norm(
            self.embedding_projection(self.soft_embedding_norm(features)))

    def embed_hard(self, ids):
        """Embed vocabulary ids in [vocab_offset, vocab_offset + vocab_size).

        Pure: the host processor validates ids before device work.
        """
        embedded = self.embedding(ids - self.vocab_offset)
        if self.is_initializing():
            self.soft_embedding_norm(jnp.zeros_like(embedded))
        return self.embedding_post_projection_norm(
            self.embedding_projection(self.hard_embedding_norm(embedded)))

    def merge_hard_embeddings(self, token_embeddings, ids):
        """Replace this vocabulary range's slots with hard embeddings, through the
        reference's dummy id for every other slot."""
        mask = (ids >= self.vocab_offset) & (ids < self.vocab_offset + self.vocab_size)
        chosen = jnp.where(mask, ids, self.vocab_offset + self.vocab_size - 1)
        hard = self.embed_hard(chosen).astype(token_embeddings.dtype)
        return jnp.where(mask[..., None], hard, token_embeddings)


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


def gemma3n_vision_path(hf_name: str) -> tuple[str, ...]:
    """A timm MobileNet-v5 weight into the corresponding Linen module."""
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


def translate_gemma3n_vision_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Mapping[str, object]:
    return translate_parameters(hf_tensors, gemma3n_vision_path, param_dtype)


def translate_gemma3n_projector_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> Mapping[str, object]:
    paths = _PROJECTOR_PATHS["gemma3n"]
    if set(hf_tensors) != set(paths):
        raise ValueError(f"vision embedder tensors differ: missing {sorted(set(paths) - set(hf_tensors))}, "
                         f"unknown {sorted(set(hf_tensors) - set(paths))}")
    return translate_parameters(hf_tensors, paths.__getitem__, param_dtype)


def _gemma3n_vision_record(
        hf_config: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """The vision embedder's fields and the encoder's model_args, the whole
    record validated before either component consumes it."""
    vision = _vision_section(hf_config)
    if vision.get("model_type", "gemma3n_vision") != "gemma3n_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not gemma3n_vision")
    used = {"model_type", "architecture", "hidden_size", "do_pooling", "model_args",
            "vocab_size", "vocab_offset", "rms_norm_eps"}
    # These are serialized HF metadata. Vocabulary and RMS fields above feed
    # the vision embedder; construction fields feed the MobileNet encoder.
    metadata = {"architectures", "transformers_version", "torch_dtype", "dtype",
                "initializer_range", "label_names", "num_classes", "id2label",
                "label2id", "output_hidden_states", "output_attentions", "return_dict",
                "is_encoder_decoder", "problem_type", "chunk_size_feed_forward"}
    unknown = (set(vision) - used - metadata
               - {key for key in vision if str(key).startswith("_")})
    if unknown:
        raise ValueError(f"vision_config fields {sorted(unknown)} have no counterpart")
    if vision.get("architecture", "mobilenetv5_300m_enc") != "mobilenetv5_300m_enc":
        raise ValueError(f"architecture {vision.get('architecture')!r} is not the MobileNet-v5 encoder")
    embedder = {
        "vision_width": records.integer(vision.get("hidden_size", 2048), "hidden_size"),
        "vocab_size": records.integer(vision.get("vocab_size", 128), "vocab_size"),
        "vocab_offset": records.integer(vision.get("vocab_offset", 262144), "vocab_offset"),
        "norm_eps": records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
    }
    if embedder["vision_width"] != 2048:
        raise ValueError("hidden_size must be 2048; timm's MobileNet-v5 encoder fixes its adapter width")
    if vision.get("do_pooling", False):
        raise ValueError("do_pooling=True requests a classifier head the encoder does not have")
    options = vision.get("model_args")
    options = records.record({} if options is None else options, "model_args")
    allowed = {field.name for field in dataclasses.fields(Gemma3nVision)}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"MobileNet-v5 model_args {sorted(unknown)} are not supported")
    return embedder, options


def translate_gemma3n_vision_config(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    _, options = _gemma3n_vision_record(hf_config)
    value: Gemma3nVision = from_record(Gemma3nVision, options)
    return {"kind": "gemma3n", **dataclasses.asdict(value)}


def translate_gemma3n_projector_config(hf_config: Mapping[str, object],
                                       text_width: int) -> Mapping[str, object]:
    embedder, _ = _gemma3n_vision_record(hf_config)
    value: Gemma3nProjector = from_record(Gemma3nProjector, {**embedder, "text_width": text_width})
    return {"kind": "gemma3n", **dataclasses.asdict(value)}


# Where each tower and projector kind's tensors sit in a media checkpoint, and
# the translators that read them.
TOWER_PREFIX = {"siglip": "vision_tower.", "llama4": "vision_model.", "gemma4": "vision_tower.",
                "qwen3_5": "visual.", "gemma3n": "vision_tower.", "deepseek_v41": "vision."}
PROJECTOR_PREFIX = {"gemma": "multi_modal_projector.", "llama4": "multi_modal_projector.",
                    "gemma4": "embed_vision.", "qwen3_5": "visual.merger.", "gemma3n": "embed_vision.",
                    "deepseek_v41": "aligner."}
TOWER_PATHS: dict[str, Callable[[str], tuple[str, ...] | None]] = {
    "siglip": siglip_vision_path, "llama4": llama4_vision_path, "gemma4": gemma4_vision_path,
    "qwen3_5": qwen35_vision_path, "gemma3n": gemma3n_vision_path, "deepseek_v41": deepseek_v41_vision_path}
# Gemma 4 is absent: its tower's map returns whole collections, not one params tree.
_TOWER_WEIGHTS = {"siglip": translate_siglip_vision_weights, "llama4": translate_llama4_vision_weights,
                  "qwen3_5": translate_qwen35_vision_weights, "gemma3n": translate_gemma3n_vision_weights,
                  "deepseek_v41": translate_deepseek_v41_vision_weights}
_PROJECTOR_WEIGHTS = {
    "gemma": translate_gemma_projector_weights,
    "llama4": translate_llama4_projector_weights,
    "gemma4": translate_gemma4_projector_weights,
    "qwen3_5": translate_qwen35_projector_weights,
    "gemma3n": translate_gemma3n_projector_weights,
    "deepseek_v41": translate_deepseek_v41_projector_weights,
}


def tower_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str) -> Variables:
    """One vision tower's variables, in the requested storage."""
    if kind == "gemma4":
        return translate_gemma4_vision_weights(hf_tensors, param_dtype=param_dtype)
    if kind not in _TOWER_WEIGHTS:
        raise ValueError(f"tower kind {kind!r} has no weight map here")
    return {"params": _TOWER_WEIGHTS[kind](hf_tensors, param_dtype=param_dtype)}


def projector_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str) -> Variables:
    """One projector kind's tensors, in the requested storage."""
    if kind not in _PROJECTOR_WEIGHTS:
        raise ValueError(f"projector kind {kind!r} has no weight map here")
    return _PROJECTOR_WEIGHTS[kind](hf_tensors, param_dtype=param_dtype)
