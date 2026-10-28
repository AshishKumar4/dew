"""Z-Image's single-stream transformer (S3-DiT), as Diffusers 0.40.0's
`ZImageTransformer2DModel` runs it for text-to-image.

Each 2x2 latent patch is one token. Each stream is padded to a multiple of 32
tokens with a learned pad token, and the image's pad tokens sit at position
(0, 0, 0). Two time-modulated refiner blocks run over the image and two
unmodulated ones over the prompt, then the main blocks run over both, image
first. Every block RMS-norms before and after the attention and the SwiGLU,
and the tanh of the time modulation gates its residuals. The three-axis
rotary places prompt token i at (1 + i, 0, 0) and patch (h, w) at
(1 + P, h, w), where P is the padded prompt length. The angles are looked up
in a float64 table rounded to float32, as the source does. The source pads a
batch to its longest row and masks; here the prompt region is the
conditioner's budget rounded up to 32, masked the same way. The source takes
time 1 - sigma and returns the negated flow, while this module takes Dew's
model time and returns Dew's flow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.blocks import modulation, sinusoidal_time
from dew.nn.dense import Dense
from dew.nn.flops import MissingFlopsCount, matching_lengths, patch_lengths, projection_flops
from dew.nn.norms import RMSNorm
from dew.nn.precision import at_least_fp32, float32_dot_general
from dew.nn.rope import axis_tables
from dew.nn.scan_orders import patchify, unpatchify
from dew.nn.sharding import logical_axes

from ..mlp import GatedMLP
from .joint import JointAttention, layer_norm

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition

MULTIPLE = 32
"""Each stream's padded length is a multiple of this."""
FREQUENCIES = 256
"""The time's sinusoid channels, and the widest time embedding the blocks
modulate by: `min(dim, 256)`."""


def _rotation(positions, axes: Sequence[int], lengths: Sequence[int], theta: float, dtype):
    """The rotary cosines and sines for integer `positions` `[B, S, 3]` in
    `dtype`, one per channel pair, `[B, S, sum(axes) // 2]`, looked up in
    each axis's table of `RopeEmbedder.precompute_freqs_cis`: `[length, dim // 2]`
    over the integer ids below `length`, float32 angles of float64 products
    and their float32 cosines and sines."""
    cosines, sines = [], []
    for index, (dim, length) in enumerate(zip(axes, lengths, strict=True)):
        cos_table, sin_table = axis_tables(np.arange(length)[:, None], (dim,), theta,
                                           dtype=np.float64, stored=np.float32)
        cosines.append(jnp.asarray(cos_table, dtype)[positions[..., index]])
        sines.append(jnp.asarray(sin_table, dtype)[positions[..., index]])
    return jnp.concatenate(cosines, axis=-1), jnp.concatenate(sines, axis=-1)


class ZImageBlock(nn.Module):
    """Runs one `ZImageTransformerBlock`.

    With `modulation`, one projection of the time embedding gives the input
    scales (one plus the projection) and the output gates (its tanh) of the
    attention and the feed-forward. Without it, as in the prompt's refiner
    blocks, the block has no time modulation.
    """

    features: int
    heads: int
    epsilon: float = 1e-5
    modulation: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.nowrap
    def _feed_forward(self, name=None):
        return GatedMLP(int(self.features / 3 * 8), self.features, dtype=self.dtype,
                        precision=self.precision, name=name)

    @nn.nowrap
    def _attention(self):
        return JointAttention(self.heads, self.features // self.heads, bias=False, epsilon=1e-5,
                              dtype=self.dtype, precision=self.precision,
                              attention_impl=self.attention_impl, name="attention")

    @property
    def modulation_features(self) -> int:
        return 4 * self.features

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Real-token QKV/output and gated MLP; pad rows stay outside MFU."""
        length, = matching_lengths(lengths)
        count = (self._attention().stream_flops(lengths, None, self.features)
                 + self._feed_forward().model_flops(lengths))
        if self.modulation:
            count += projection_flops(np.ones(length.shape, np.int64), min(self.features, FREQUENCIES),
                                       self.modulation_features)
        return count

    @nn.compact
    def __call__(self, x, rotation, lengths, embedded=None):
        def norm(name):
            return RMSNorm(epsilon=self.epsilon, dtype=self.dtype, name=name)

        # `ZSingleStreamAttnProcessor`: bias-free, its query and key norms at
        # 1e-5 whatever `norm_eps`.
        joint = self._attention()

        def attention(x):
            return joint(x, rotation=rotation, lengths=lengths)[0]

        feed_forward = self._feed_forward("feed_forward")
        if not self.modulation:
            x = x + norm("attention_norm2")(attention(norm("attention_norm1")(x)))
            return x + norm("ffn_norm2")(feed_forward(norm("ffn_norm1")(x)))
        modulated = modulation(embedded, self.features, 4, name="modulation", silu=False, dtype=self.dtype,
                               precision=self.precision)[:, None]
        scale, gate, scale_mlp, gate_mlp = jnp.split(modulated, 4, axis=-1)
        attended = attention(norm("attention_norm1")(x) * (1 + scale))
        x = x + jnp.tanh(gate) * norm("attention_norm2")(attended)
        return x + jnp.tanh(gate_mlp) * norm("ffn_norm2")(
            feed_forward(norm("ffn_norm1")(x) * (1 + scale_mlp))
        )


@logical_axes({("x_embedder",): (None, "embed"), ("cap_embedder",): (None, "embed"),
               ("final_linear",): ("embed", None), ("final_modulation",): (None, "embed"),
               ("t_embedder_1",): (None, "mlp"), ("t_embedder_2",): ("mlp", None)})
class ZImageTransformer(nn.Module):
    """Runs Diffusers 0.40.0's `ZImageTransformer2DModel` behind Dew's model interface.

    `__call__` takes NHWC latents, the model time the schedule supplies (the
    sigma times the training count) and a `DenoisingCondition`, and returns
    the flow. The condition's `context` is the prompt states, padded on the
    right to a fixed budget, and its `mask` marks the real tokens.
    """

    in_channels: int = 16
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    norm_eps: float = 1e-5
    cap_feat_dim: int = 2560
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: Sequence[int] = (32, 48, 48)
    axes_lens: Sequence[int] = (1536, 512, 512)
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.nowrap
    def projections(self, input_channels: int, text_features: int) -> dict[str, tuple[int, int]]:
        time = min(self.dim, FREQUENCIES)
        return {'x_embedder': (4 * input_channels, self.dim), 'cap_embedder': (text_features, self.dim),
                't_embedder_1': (FREQUENCIES, 1024), 't_embedder_2': (1024, time),
                'final_modulation': (time, self.dim), 'final_linear': (self.dim, 4 * input_channels)}

    @nn.nowrap
    def condition_lengths(self, lengths: np.ndarray, spans: np.ndarray) -> np.ndarray:
        return lengths

    @property
    def text_keyword(self) -> str:
        """Every call takes the caption as `conditioning`, which joins the image's single stream."""
        return "conditioning"

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("ZImageTransformer requires multimodal_flops with actual image/text widths")

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        return patch_lengths(sample_shape, (2, 2), rows)

    @nn.nowrap
    def multimodal_flops(self, lengths, context_lengths, input_channels: int, text_features: int) -> float:
        """Static patch/text maps, both refiner stacks and the joined stack."""
        image, context = matching_lengths(lengths, context_lengths)
        rows = np.ones(image.shape, np.int64)
        maps = self.projections(input_channels, text_features)
        count = sum(projection_flops(lengths if name in ('x_embedder', 'final_linear') else
                                     context_lengths if name == 'cap_embedder' else rows, *shape)
                    for name, shape in maps.items())
        image_block = self._block(parent=None)
        text_block = self._block(modulation=False, parent=None)
        # Learned pad slots are visible architectural tokens, not masked batch padding.
        image_span = (np.ceil(image / MULTIPLE) * MULTIPLE).astype(np.int64)
        context_span = (np.ceil(context / MULTIPLE) * MULTIPLE).astype(np.int64)
        count += self.n_refiner_layers * (image_block.model_flops(image_span)
                                          + text_block.model_flops(context_span))
        return count + self.n_layers * image_block.model_flops(image_span + context_span)

    @nn.nowrap
    def _block(self, **where) -> ZImageBlock:
        """A block as the model runs it, an image or joint one or, `modulation=False`, a caption one,
        named by `where` (or unbound, `parent=None`, where a FLOP count reads it): one factory, so the
        count and the execution build the same block."""
        return ZImageBlock(self.dim, self.n_heads, epsilon=self.norm_eps, dtype=self.dtype,
                           precision=self.precision, attention_impl=self.attention_impl, **where)

    def _embedded_time(self, time, dtype):
        """`TimestepEmbedder` of the source's time, (1000 - time) / 1000 times
        `t_scale`: 256 sinusoids, cosines first, 1024 wide inside, and
        `min(dim, 256)` out."""
        width = at_least_fp32(self.dtype)
        # The scale folded on the host and rounded once (exactly 1 at the published 1000): a division
        # by a constant left in the program is any XLA backend's to rewrite as a product with its
        # inexact reciprocal, which moves the time's sinusoids, and every modulation they drive, off
        # the source's value (measured twice as far from float64 on JAX 0.12's CPU as on 0.11's).
        source_time = (1000.0 - jnp.asarray(time, width)) * jnp.asarray(self.t_scale / 1000.0, width)
        features = sinusoidal_time(source_time, FREQUENCIES, dtype=width).astype(dtype)
        maps = self.projections(0, 0)
        mlp = GatedMLP(maps['t_embedder_1'][1], maps['t_embedder_2'][1], in_features=features.shape[-1],
                        activation=nn.silu, use_bias=True, names=("t_embedder_1", "t_embedder_2"),
                        dtype=self.dtype, precision=self.precision)
        nn.share_scope(self, mlp)
        return mlp(features)

    @nn.compact
    def __call__(self, x, time, conditioning: DenoisingCondition, train: bool = False):
        """Return the flow for the latents `x` at `time` under `conditioning`.

        `train` is part of the objective's standard call; the published
        transformer has no dropout, so it changes nothing here. Raises
        `ValueError` for latents of odd height or width, and for a condition
        without a `mask`.
        """
        if x.ndim != 4 or x.shape[1] % 2 or x.shape[2] % 2:
            raise ValueError(f"Z-Image takes NHWC latents of even height and width, got shape {x.shape}")
        if conditioning.mask is None:
            raise ValueError("Z-Image reads the prompt's real tokens; its conditioning needs a mask")
        batch, height, width, _ = x.shape
        rows, columns = height // 2, width // 2
        count = rows * columns
        image_span = -(-count // MULTIPLE) * MULTIPLE
        context, mask = conditioning.context, conditioning.mask
        # The prompt region holds every row's real tokens rounded up to 32.
        extra = -context.shape[1] % MULTIPLE
        context = jnp.pad(context, ((0, 0), (0, extra), (0, 0)))
        mask = jnp.pad(mask, ((0, 0), (0, extra)))
        budget = context.shape[1]
        spans = -(-jnp.sum(mask, axis=1, dtype=jnp.int32) // MULTIPLE) * MULTIPLE
        slots = jnp.arange(budget)

        embedded = self._embedded_time(time, x.dtype)
        pad_image = self.param("x_pad_token", nn.initializers.zeros, (1, self.dim))
        pad_caption = self.param("cap_pad_token", nn.initializers.zeros, (1, self.dim))
        maps = self.projections(x.shape[-1], conditioning.context.shape[-1])
        image = Dense(maps['x_embedder'][1], dtype=self.dtype, precision=self.precision,
                      name="x_embedder", dot_general=float32_dot_general)(patchify(x, 2))
        image = jnp.concatenate(
            [image, jnp.broadcast_to(pad_image.astype(image.dtype), (batch, image_span - count, self.dim))],
            axis=1,
        )
        caption = Dense(maps['cap_embedder'][1], dtype=self.dtype, precision=self.precision,
                        name="cap_embedder", dot_general=float32_dot_general)(
            RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name="cap_norm")(context))
        caption = jnp.where(mask[..., None], caption, pad_caption.astype(caption.dtype))

        grid = np.indices((rows, columns)).reshape(2, -1).T
        image_positions = jnp.concatenate([
            jnp.concatenate([jnp.broadcast_to(spans[:, None, None] + 1, (batch, count, 1)),
                             jnp.broadcast_to(jnp.asarray(grid)[None], (batch, count, 2))], axis=-1),
            jnp.zeros((batch, image_span - count, 3), jnp.int32)], axis=1)
        # Slots past a row's padded prompt are masked; they sit at the origin.
        leading = jnp.where(slots[None] < spans[:, None], slots[None] + 1, 0)
        caption_positions = jnp.stack([leading, jnp.zeros_like(leading), jnp.zeros_like(leading)], axis=-1)
        # Each axis's table reaches every position the call takes, `axes_lens` at least: a prompt's
        # image sits one past its padded span, the grid's rows and columns at their own indices, and
        # a row's rotation is its own position's however long the table, so a call past the published
        # lengths rotates at its own places rather than at the last row's.
        reach = tuple(map(max, self.axes_lens, (budget + 2, rows, columns)))
        table = (self.axes_dims, reach, self.rope_theta)
        image_rotation = _rotation(image_positions, *table, at_least_fp32(image.dtype))
        caption_rotation = _rotation(caption_positions, *table, at_least_fp32(image.dtype))
        whole_image = jnp.full((batch,), image_span, jnp.int32)
        for index in range(self.n_refiner_layers):
            image = self._block(name=f"noise_refiner_{index}")(image, image_rotation, whole_image, embedded)
        for index in range(self.n_refiner_layers):
            caption = self._block(modulation=False, name=f"context_refiner_{index}")(
                caption, caption_rotation, spans)
        joined = jnp.concatenate([image, caption], axis=1)
        rotation = tuple(jnp.concatenate(pair, axis=1)
                         for pair in zip(image_rotation, caption_rotation, strict=True))
        for index in range(self.n_layers):
            joined = self._block(name=f"layers_{index}")(joined, rotation, image_span + spans, embedded)

        scale = 1 + modulation(embedded, self.dim, 1, name="final_modulation", dtype=self.dtype,
                               precision=self.precision)[:, None]
        out = Dense(maps['final_linear'][1], dtype=self.dtype, precision=self.precision,
                    name="final_linear", dot_general=float32_dot_general)(
            layer_norm(self.dtype)(joined[:, :count]) * scale)
        return -unpatchify(out, 2, *x.shape[1:])


__all__ = ["ZImageBlock", "ZImageTransformer"]
