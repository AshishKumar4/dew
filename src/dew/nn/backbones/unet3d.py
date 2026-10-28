"""
UNet3D: the 2D UNet inflated for video, AnimateDiff-style.

Every frame goes through the 2D UNet's body, with a zero-initialized
temporal attention block after the residual blocks of each resolution level.
Zero init means a freshly inflated model reproduces the 2D UNet frame by
frame, so a pretrained image checkpoint (inflate_unet_variables) is the starting
point and training only has to learn motion.
"""


import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.batch import merge
from dew.nn.dense import Dense

from ..attention import NormalAttention, RMSNorm
from ..blocks import TimeProjection
from ..dit import ROPE_THETA
from ..flops import MissingFlopsCount, patch_lengths, projection_flops
from ..precision import at_least_fp32, float32_dot_general
from ..rope import rotary_freqs
from .spatial import SpatialCount, SpatialShape
from .unet import Unet, unet_body


class TemporalBlock(nn.Module):
    """Temporal self-attention over the frame axis at every spatial position.

    The output projection is zero-initialized, so the block is an exact
    identity at init and an inflated model starts as the per-frame 2D model.
    """
    heads: int = 8
    norm_epsilon: float = 1e-5
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def model_flops(self, lengths, input_features: int) -> float:
        attention = NormalAttention(input_features, self.heads, input_features // self.heads, parent=None)
        return attention.model_flops(lengths) + projection_flops(lengths, input_features, input_features)

    @nn.compact
    def __call__(self, x, frames: int):
        # x: [B*T, H, W, C] -> tokens over T per spatial position
        BT, H, W, C = x.shape
        B = BT // frames
        h = x.reshape(B, frames, H * W, C)
        h = h.transpose(0, 2, 1, 3).reshape(B * H * W, frames, C)

        h = RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(h)
        freqs_cis = rotary_freqs(
            jnp.arange(frames), C // self.heads, ROPE_THETA, dtype=at_least_fp32(h.dtype)
        )
        h = NormalAttention(
            query_dim=C,
            heads=self.heads,
            dim_head=C // self.heads,
            dtype=self.dtype,
            precision=self.precision,
            use_bias=True,
            name="temporal_attention",
        )(h, freqs_cis=freqs_cis)
        # zero-init gate: identity at init, so inflation preserves the 2D model
        h = Dense(
            features=C, dtype=self.dtype, precision=self.precision,
            kernel_init=nn.initializers.zeros, name="temporal_out", dot_general=float32_dot_general)(h)

        h = h.reshape(B, H * W, frames, C).transpose(0, 2, 1, 3).reshape(BT, H, W, C)
        return x + h


class UNet3D(Unet):
    """Denoises video (B, T, H, W, C) with the 2D `Unet` body per frame and a `TemporalBlock` per level.

    There is a `TemporalBlock` at every resolution level. The spatial
    parameter paths are identical to `Unet`'s, so 2D checkpoints inflate
    directly.
    """
    temporal_heads: int = 8

    @nn.nowrap
    def spatial_flops(self, sample_shape: tuple[int, ...], rows: int, context_lengths=None,
                       text_features: int | None = None) -> float:
        """The frame body and its temporal hook share the forward's ordered walk."""
        if len(sample_shape) != 4:
            raise MissingFlopsCount("UNet3D needs [frames, height, width, channels] geometry")
        frames, height, width, channels = sample_shape
        divisor = 2**(len(self.feature_depths) - 1)
        patch_lengths(sample_shape[1:], (divisor, divisor), rows)
        text = None if context_lengths is None else np.repeat(context_lengths, frames)
        operations = SpatialCount(text, text_features)
        operations.flops = TimeProjection(self.emb_features, parent=None).model_flops(
            np.ones((rows * frames,), np.int64), self.emb_features)

        def temporal(x, name):
            batch_frames, h, w, features = x.shape
            lengths = np.full((batch_frames // frames * h * w,), frames, np.int64)
            operations.flops += self.temporal_block(name, parent=None).model_flops(lengths, features)
            return x

        unet_body(self, SpatialShape((rows * frames, height, width, channels)),
                  SpatialShape((rows * frames, self.emb_features)), None,
                  temporal=temporal, operations=operations)
        return operations.flops

    @nn.nowrap
    def temporal_block(self, name, **scope):
        return TemporalBlock(heads=self.temporal_heads, dtype=self.dtype, precision=self.precision,
                             name=name, **scope)

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train: bool = False):
        B, T, H, W, C = x.shape
        text = None if textcontext is None else jnp.repeat(textcontext.hidden, T, axis=0)

        def temporal(x, name):
            return self.temporal_block(name)(x, T)

        temb = self.time_embedding(jnp.repeat(temb, T, axis=0))
        out = unet_body(self, x.reshape(B * T, H, W, C), temb, text,
                        temporal=temporal)
        return out.reshape(B, T, H, W, self.output_channels)


def inflate_unet_variables(variables_2d, variables_3d):
    """Copy a trained 2D Unet's variables into a UNet3D init.

    Spatial module names are identical between the two models, so every 2D
    leaf lands on its 3D counterpart, in every collection: the weights and
    the Fourier table the 2D model's time embedding was trained against. The
    temporal blocks keep their (zero-init) fresh params. The result
    reproduces the 2D model frame by frame until training moves the temporal
    weights.
    """
    return merge(variables_3d, variables_2d)
