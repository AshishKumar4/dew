"""DeepSeek-V4.1's vision trunk and aligner, with their checkpoint maps
(the release's `inference/vision.py`).
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
from dew.nn.attention import RMSNorm, scaled_dot_product_attention
from dew.nn.precision import at_least_fp32
from dew.registry import Record

from .common import ProjectorBase, TowerBase, TowerGeometry, _grid_rope, _grid_rope_tables, _vision_section


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
        aligned = dense(name="w2")(gelu_exact(dense(name="w1")(squares.reshape(images, high, wide, -1))))
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


def translate_deepseek_v41_vision_config(hf_config: Mapping[str, object]) -> Record:
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
        "class": "deepseek_v41", "fields": {
        "num_hidden_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "hidden_size": width,
        "num_attention_heads": heads,
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "patch_size": records.integer(vision["patch_size"], "patch_size"),
        "rope_theta": records.number(vision.get("rope_theta", 10000.0), "rope_theta"),
    }}


def translate_deepseek_v41_projector_config(hf_config: Mapping[str, object],
                                            text_width: int) -> Record:
    """DeepSeek-V4.1's aligner fields: the ViT width, its downsample ratio and
    the decoder width its rows and span vectors enter."""
    vision = _vision_section(hf_config)
    return {
        "class": "deepseek_v41", "fields": {
        "vision_width": records.integer(vision["hidden_size"], "hidden_size"),
        "downsample_ratio": records.integer(vision.get("downsample_ratio", 3), "downsample_ratio"),
        "out_width": int(text_width),
    }}
