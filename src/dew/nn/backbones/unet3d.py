"""
UNet3D: the 2D UNet inflated for video, AnimateDiff-style.

Every frame goes through the 2D UNet's body, with a zero-initialized
temporal attention block after the residual blocks of each resolution level.
Zero init means a freshly inflated model reproduces the 2D UNet frame by
frame, so a pretrained image checkpoint (inflate_unet_params) is the starting
point and training only has to learn motion.
"""

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from typing import Optional

from ..attention import NormalAttention, rotary_freqs
from ..dit import ROPE_THETA
from .unet import Unet, unet_body
from dew.registry import models
from ..sharding import logical_axes


@logical_axes({}, heuristic=(("temporal_out",),))
class TemporalBlock(nn.Module):
    """Temporal self-attention over the frame axis at every spatial position.

    The output projection is zero-initialized, so the block is an exact
    identity at init and an inflated model starts as the per-frame 2D model.
    """
    features: int
    heads: int = 8
    norm_epsilon: float = 1e-5
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, frames: int):
        # x: [B*T, H, W, C] -> tokens over T per spatial position
        BT, H, W, C = x.shape
        B = BT // frames
        h = x.reshape(B, frames, H * W, C)
        h = h.transpose(0, 2, 1, 3).reshape(B * H * W, frames, C)

        h = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(h)
        h = NormalAttention(
            query_dim=C,
            heads=self.heads,
            dim_head=C // self.heads,
            dtype=self.dtype,
            precision=self.precision,
            use_bias=True,
            name="temporal_attention",
        )(h, freqs_cis=rotary_freqs(jnp.arange(frames), C // self.heads, ROPE_THETA))
        # zero-init gate: identity at init, so inflation preserves the 2D model
        h = nn.Dense(
            features=C, dtype=self.dtype, precision=self.precision,
            kernel_init=nn.initializers.zeros, name="temporal_out")(h)

        h = h.reshape(B, H * W, frames, C).transpose(0, 2, 1, 3).reshape(BT, H, W, C)
        return x + h


@models("unet_3d")
class UNet3D(Unet):
    """Video UNet over (B, T, H, W, C): the 2D Unet body per frame, with a
    TemporalBlock at every resolution level. Spatial param paths are
    identical to Unet, so 2D checkpoints inflate directly."""
    temporal_heads: int = 8

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train: bool = False):
        B, T, H, W, C = x.shape
        text = None if textcontext is None else jnp.repeat(textcontext.hidden, T, axis=0)

        def temporal(x, name):
            return TemporalBlock(features=x.shape[-1], heads=self.temporal_heads,
                                 dtype=self.dtype, precision=self.precision, name=name)(x, T)

        out = unet_body(self, x.reshape(B * T, H, W, C), jnp.repeat(temb, T, axis=0), text,
                        temporal=temporal)
        return out.reshape(B, T, H, W, self.output_channels)


def inflate_unet_params(params_2d, params_3d):
    """Copy a trained 2D Unet param tree into a UNet3D init.

    Spatial module names are identical between the two models, so every 2D
    leaf lands on its 3D counterpart; the temporal blocks keep their
    (zero-init) fresh params. The result reproduces the 2D model frame by
    frame until training moves the temporal weights.
    """
    def merge(dst, src):
        out = dict(dst)
        for key, value in src.items():
            if isinstance(value, dict):
                assert key in dst, f"2D module '{key}' has no counterpart in the 3D tree"
                out[key] = merge(dst[key], value)
            else:
                out[key] = value
        return out
    return merge(params_3d, params_2d)
