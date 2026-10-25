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

from dew.nn.activations import gelu_tanh
from dew.nn.attention import LayerNorm, NormalAttention, QKNorm
from dew.nn.blocks import Modulation, sinusoidal_time
from dew.nn.conv import Conv
from dew.nn.dense import Dense
from dew.nn.flops import MissingFlopsCount, matching_lengths, patch_lengths, projection_flops
from dew.nn.precision import at_least_fp32, float32_dot_general
from dew.nn.rope import axis_tables
from dew.nn.sharding import logical_axes

from ..mlp import GatedMLP
from .joint import embedding, embedding_flops, layer_norm, modulate

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

    @nn.nowrap
    def _attention(self, name=None):
        """The source's `Attention` under `WanAttnProcessor2_0`: flat projections, Wan's
        `rms_norm_across_heads` over each token's whole query and key, rounded to the weight's dtype
        before they scale, and adjacent rotary pairs. Its cross-attention reads keys and values from
        the text alone, where the MM-DiTs' join the two streams (`JointAttention`)."""
        return NormalAttention(
            self.features, self.heads, self.features // self.heads, dtype=self.dtype,
            precision=self.precision,
            qk_norm=QKNorm(("norm_q", "norm_k"), self.epsilon, across_heads=True, scale_after_cast=True)
            if self.qk_norm else None,
            attention_impl=self.attention_impl, flat=True, rotary_pairs="adjacent", name=name)

    @nn.nowrap
    def _feed_forward(self, name=None):
        return GatedMLP(self.ffn_dim, self.features, activation="gelu", use_bias=True, dtype=self.dtype,
                        precision=self.precision, name=name)

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("WanBlock requires multisequence_flops(video_lengths, text_lengths)")

    @nn.nowrap
    def multisequence_flops(self, lengths, context_lengths) -> float:
        """Self-attention, cross-attention maps/products and the ungated FFN."""
        matching_lengths(lengths, context_lengths)
        attention = self._attention()
        return (attention.model_flops(lengths)
                + attention.cross_flops(lengths, context_lengths, self.features)
                + self._feed_forward().model_flops(lengths))

    @nn.compact
    def __call__(self, x, context, modulation, rotation):
        wide = at_least_fp32(self.dtype)
        table = self.param("scale_shift_table", nn.initializers.normal(self.features ** -0.5),
                           (1, 6, self.features), jnp.float32)
        pieces = table.astype(wide) + modulation.astype(wide)
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = (pieces[:, index] for index in range(6))
        norm = layer_norm(wide, self.epsilon)

        attended = self._attention("attn1")(
            modulate(norm(x), shift, scale).astype(x.dtype), freqs_cis=rotation)
        x = (x.astype(wide) + attended * gate[:, None]).astype(x.dtype)
        normalized = (LayerNorm(epsilon=self.epsilon, dtype=x.dtype, name="norm2")(x)
                      if self.cross_attn_norm else x)
        x = x + self._attention("attn2")(normalized, context)
        fed = self._feed_forward("ffn")(
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

    @nn.nowrap
    def condition_lengths(self, lengths: np.ndarray, spans: np.ndarray) -> np.ndarray:
        return spans

    @property
    def text_keyword(self) -> str:
        """Every call takes the text as `conditioning`, which every block cross-attends to."""
        return "conditioning"

    @nn.nowrap
    def _block(self, **where) -> WanBlock:
        """A block as the model runs it, named by `where` (or unbound, `parent=None`, where a FLOP count
        reads it): one factory, so the count and the execution build the same block."""
        return WanBlock(self.features, self.num_attention_heads, self.ffn_dim,
                        qk_norm=self.qk_norm is not None, cross_attn_norm=self.cross_attn_norm,
                        epsilon=self.eps, attention_impl=self.attention_impl, dtype=self.dtype,
                        precision=self.precision, **where)

    @nn.nowrap
    def _modulation(self):
        return Modulation(self.features, 6, dtype=self.dtype, precision=self.precision, name="time_proj")

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("WanTransformer requires multisequence_flops(video_lengths, text_lengths)")

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        return patch_lengths(sample_shape, tuple(self.patch_size), rows)

    @nn.nowrap
    def multisequence_flops(self, lengths, context_lengths, *, input_channels: int,
                            text_features: int, pooled_features: int | None = None) -> float:
        """Every video/text/time projection and the shared sixfold time modulation."""
        image, _ = matching_lengths(lengths, context_lengths)
        rows = np.ones(image.shape, np.int64)
        patch = int(np.prod(self.patch_size))
        count = (projection_flops(lengths, patch * input_channels, self.features)
                 + embedding_flops(context_lengths, text_features, self.features)
                 + embedding_flops(rows, self.freq_dim, self.features)
                 + self._modulation().model_flops(rows))
        block = self._block(parent=None)
        return (count + self.num_layers * block.multisequence_flops(lengths, context_lengths)
                + projection_flops(lengths, self.features, patch * self.out_channels))

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
        embedded = embedding(self, sinusoids, self.features, "time_embedder", dtype=wide,
                             precision=self.precision).astype(tokens.dtype)
        modulation = jnp.stack(self._modulation()(embedded), axis=1)
        context = embedding(self, conditioning.context, self.features, "text_embedder",
                            activation=gelu_tanh, **dense)
        rotation = tuple(jnp.asarray(table[None], wide)
                         for table in wan_rotation(frames, rows, columns, self.attention_head_dim))
        for index in range(self.num_layers):
            tokens = self._block(name=f"blocks_{index}")(tokens, context, modulation, rotation)

        table = self.param("scale_shift_table", nn.initializers.normal(self.features ** -0.5),
                           (1, 2, self.features), jnp.float32)
        shift, scale = (table.astype(wide)[:, index] + embedded.astype(wide) for index in range(2))
        tokens = modulate(layer_norm(wide, self.eps)(tokens), shift, scale).astype(tokens.dtype)
        out = Dense(int(np.prod(patch)) * self.out_channels, name="proj_out", **dense,
                    dot_general=float32_dot_general)(tokens)
        return einops.rearrange(out.reshape(x.shape[0], *grid, *patch, self.out_channels),
                                "b f h w pf ph pw c -> b (f pf) (h ph) (w pw) c")


__all__ = ["WanBlock", "WanTransformer"]
