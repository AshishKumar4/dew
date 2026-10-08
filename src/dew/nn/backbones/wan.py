"""Wan 2.1's video transformer, as Diffusers 0.34.0's `WanTransformer3DModel` runs it for text-to-video.

A strided 3-D convolution cuts the clip's latent into 1x2x2 patches, one
token each, in frame, row, column order. Every block normalizes its residual
stream and modulates it by the time, self-attends with a three-axis rotary
table over (frame, row, column), cross-attends to the text, and runs the
tanh-GELU feed-forward. A block's six modulation pieces are its own learned
table plus one projection of the time embedding that all blocks share.
Queries and keys are RMS-normalized across all heads, before they are split
into heads.

The source keeps its norms, its modulation tables and its time embedder in
fp32 (`_keep_in_fp32_modules`) and adds the gated residuals in fp32. A
bfloat16 model here does the same at `at_least_fp32` of its dtype, and
rounds the stream back after each sum. The source rotates queries and keys
in float64; Dew rotates them at `at_least_fp32` of their dtype, by
cosines and sines computed in float64 (`dew.nn.rope.axis_tables`) and
rounded to float32.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import einops
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import LayerNorm, RMSNorm, scaled_dot_product_attention
from dew.nn.blocks import sinusoidal_time
from dew.nn.conv import Conv
from dew.nn.precision import at_least_fp32
from dew.nn.rope import axis_tables, rotate
from dew.nn.sharding import logical_axes

from .decoder_block import GatedMLP
from .joint import Modulation, embedding, layer_norm, modulate

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition


def wan_rotation(frames: int, rows: int, columns: int, head_dim: int, *,
                 theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """`WanRotaryPosEmbed` over a patch grid: the head's channel pairs split
    between frame, row and column, `head_dim // 6` pairs each for the rows
    and the columns and the rest for the frames, each axis's angles at its
    own width, in float64, `[frames * rows * columns, head_dim // 2]`."""
    side = 2 * (head_dim // 6)
    positions = np.indices((frames, rows, columns)).reshape(3, -1).T
    return axis_tables(positions, (head_dim - 2 * side, side, side), theta, dtype=np.float64)


# The source stores one flat [inner, inner] tensor per projection; the
# suffixes carry the attention's own name, as JointAttention's do.
@logical_axes({(module, name): axes for module in ("attn1", "attn2")
               for name, axes in (*(((name), ("embed", None)) for name in ("to_q", "to_k", "to_v")),
                                  ("to_out_0", (None, "embed")))})
class WanAttention(nn.Module):
    """Runs the source's `Attention` under `WanAttnProcessor2_0`, for text-to-video.

    Queries come from the video tokens. Keys and values come from `context`
    (the text, for the cross-attention) or from the tokens themselves. It
    differs from `JointAttention` in two ways. Wan's `rms_norm_across_heads`
    normalizes each token's whole projection before it is split into heads,
    where the MM-DiTs normalize each head. And its cross-attention reads keys
    and values from the text alone, where theirs join the two streams.
    `rotation`, `[1, S, D // 2]`, rotates the self-attention's queries and
    keys.
    """

    heads: int
    head_dim: int
    qk_norm: bool = True
    epsilon: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    def _projected(self, name: str, x, *, norm: str | None = None):
        projected = nn.Dense(self.heads * self.head_dim, dtype=self.dtype, precision=self.precision,
                             name=name)(x)
        if norm is not None and self.qk_norm:
            # Diffusers' `RMSNorm` rounds the normalized row to the weight's
            # dtype before it scales.
            projected = RMSNorm(epsilon=self.epsilon, scale_after_cast=True, dtype=self.dtype,
                                name=norm)(projected)
        return projected.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)

    @nn.compact
    def __call__(self, x, context=None, rotation=None):
        source = x if context is None else context
        query = self._projected("to_q", x, norm="norm_q")
        key = self._projected("to_k", source, norm="norm_k")
        value = self._projected("to_v", source)
        if rotation is not None:
            # The source rotates in float64 and rounds once to its dtype.
            query, key = (rotate(part, *rotation, pairs="adjacent") for part in (query, key))
        attended = scaled_dot_product_attention(query, key, value, implementation=self.attention_impl,
                                                precision=self.precision)
        attended = attended.reshape(x.shape[0], x.shape[1], self.heads * self.head_dim)
        return nn.Dense(self.heads * self.head_dim, dtype=self.dtype, precision=self.precision,
                        name="to_out_0")(attended)


class WanBlock(nn.Module):
    """Runs `WanTransformerBlock`: modulated self-attention, cross-attention to the text, and a feed-forward.

    The cross-attention reads the text through its own affine norm
    (`cross_attn_norm`), and the feed-forward is modulated like the
    self-attention. `modulation` is the model's shared projection of the
    time, `[B, 6, features]`, and the block adds its `scale_shift_table` to
    it.
    """

    features: int
    heads: int
    ffn_dim: int
    qk_norm: bool = True
    cross_attn_norm: bool = True
    epsilon: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.compact
    def __call__(self, x, context, modulation, rotation):
        wide = at_least_fp32(self.dtype)
        table = self.param("scale_shift_table", nn.initializers.normal(self.features ** -0.5),
                           (1, 6, self.features), jnp.float32)
        pieces = table.astype(wide) + modulation.astype(wide)
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = (pieces[:, index] for index in range(6))
        attention = {"qk_norm": self.qk_norm, "epsilon": self.epsilon, "dtype": self.dtype,
                     "precision": self.precision, "attention_impl": self.attention_impl}
        head_dim = self.features // self.heads
        norm = layer_norm(wide, self.epsilon)

        attended = WanAttention(self.heads, head_dim, name="attn1", **attention)(
            modulate(norm(x), shift, scale).astype(x.dtype), rotation=rotation)
        x = (x.astype(wide) + attended * gate[:, None]).astype(x.dtype)
        normalized = (LayerNorm(epsilon=self.epsilon, dtype=x.dtype, name="norm2")(x)
                      if self.cross_attn_norm else x)
        x = x + WanAttention(self.heads, head_dim, name="attn2", **attention)(normalized, context)
        fed = GatedMLP(self.ffn_dim, self.features, activation="gelu", use_bias=True, dtype=self.dtype,
                       precision=self.precision, name="ffn")(
            modulate(norm(x), shift_mlp, scale_mlp).astype(x.dtype))
        return (x.astype(wide) + fed.astype(wide) * gate_mlp[:, None]).astype(x.dtype)


@logical_axes({("text_embedder_linear_1",): (None, "embed"), ("text_embedder_linear_2",): (None, "embed"),
               ("time_embedder_linear_1",): (None, "embed"), ("time_embedder_linear_2",): (None, "embed"),
               ("patch_embedding_3d",): (None, None, None, None, "embed"), ("proj_out",): ("embed", None)})
class WanTransformer(nn.Module):
    """Runs Diffusers 0.34.0's `WanTransformer3DModel` for text-to-video, behind Dew's model interface.

    `__call__` takes a clip's latent `[B, F, H, W, C]`, the model time the
    schedule supplies, and a `DenoisingCondition`, and returns the flow
    `[B, F, H, W, out_channels]`. The model time is the sigma times the
    training count, which is the timestep the source's pipeline passes. The
    condition's `context` is the UMT5 states `[B, L, text_dim]`, and the
    cross-attention reads every one of them, as the source's does. The
    fields are the config's; `qk_norm` takes the value the source names,
    `rms_norm_across_heads`, or None.
    """

    patch_size: Sequence[int] = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str | None = "rms_norm_across_heads"
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @property
    def features(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def text_keyword(self) -> str:
        """Every call takes the text as `conditioning`, which every block cross-attends to."""
        return "conditioning"

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """Return the flow for the latent `x` at `time` under `conditioning`.

        `train` is part of the objective's standard call; the published
        transformer has no dropout, so it changes nothing here. Raises
        `ValueError` for latents that are not whole patches, for a patch grid
        longer than `rope_max_seq_len` on any axis, and for an unknown
        `qk_norm`.
        """
        if self.qk_norm not in (None, "rms_norm_across_heads"):
            raise ValueError(f"Native Wan implements qk_norm 'rms_norm_across_heads', not "
                             f"{self.qk_norm!r}")
        patch = tuple(self.patch_size)
        if x.ndim != 5 or any(size % step for size, step in zip(x.shape[1:4], patch, strict=True)):
            raise ValueError(f"Wan takes [B, F, H, W, C] latents in whole {patch} patches, "
                             f"got shape {x.shape}")
        grid = tuple(size // step for size, step in zip(x.shape[1:4], patch, strict=True))
        frames, rows, columns = grid
        if max(grid) > self.rope_max_seq_len:
            raise ValueError(f"A {grid} patch grid runs past the rotary table's "
                             f"{self.rope_max_seq_len} positions")
        dense = {"dtype": self.dtype, "precision": self.precision}
        wide = at_least_fp32(self.dtype)

        # The source's `patch_embedding`; CLIP's vision tower holds a 2-D
        # kernel at that path, and one path holds one set of axes.
        tokens = Conv(self.features, patch, strides=patch, padding="VALID", name="patch_embedding_3d",
                      **dense)(x)
        tokens = tokens.reshape(x.shape[0], -1, self.features)
        # `WanTimeTextImageEmbedding`: the time embedder in fp32, its output
        # in the model's dtype, and one projection of it into six pieces.
        sinusoids = sinusoidal_time(time, self.freq_dim, dtype=wide)
        embedded = embedding(sinusoids, self.features, "time_embedder", dtype=wide,
                             precision=self.precision).astype(tokens.dtype)
        modulation = jnp.stack(Modulation(self.features, 6, name="time_proj", **dense)(embedded), axis=1)
        context = embedding(conditioning.context, self.features, "text_embedder",
                            activation=lambda hidden: nn.gelu(hidden, approximate=True), **dense)
        rotation = tuple(jnp.asarray(table[None], wide)
                         for table in wan_rotation(frames, rows, columns, self.attention_head_dim))
        for index in range(self.num_layers):
            tokens = WanBlock(self.features, self.num_attention_heads, self.ffn_dim,
                              qk_norm=self.qk_norm is not None, cross_attn_norm=self.cross_attn_norm,
                              epsilon=self.eps, attention_impl=self.attention_impl, name=f"blocks_{index}",
                              **dense)(tokens, context, modulation, rotation)

        table = self.param("scale_shift_table", nn.initializers.normal(self.features ** -0.5),
                           (1, 2, self.features), jnp.float32)
        shift, scale = (table.astype(wide)[:, index] + embedded.astype(wide) for index in range(2))
        tokens = modulate(layer_norm(wide, self.eps)(tokens), shift, scale).astype(tokens.dtype)
        out = nn.Dense(int(np.prod(patch)) * self.out_channels, name="proj_out", **dense)(tokens)
        return einops.rearrange(out.reshape(x.shape[0], *grid, *patch, self.out_channels),
                                "b f h w pf ph pw c -> b (f pf) (h ph) (w pw) c")


__all__ = ["WanAttention", "WanBlock", "WanTransformer"]
