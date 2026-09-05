"""The convolutional UNet, and the body it shares with the video UNet."""

import dataclasses
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from functools import partial

from ..blocks import Downsample, Upsample, FourierEmbedding, TimeProjection, ResidualBlock
from ..attention import Stage, stage_attention
from ..sharding import logical_axes
from dew.registry import models


def unet_body(model: "Unet", x, temb, text, temporal=None):
    """The UNet forward over frames `[N, H, W, C]`, run inside the model's
    compact `__call__` so every block here is the model's own submodule.

    `temporal(x, name)`, when given, runs after the residual blocks of every
    level, which is where the video UNet mixes across frames; the spatial
    blocks and their names are the same either way, which is what lets a 2D
    checkpoint inflate into the 3D model.
    """
    temb = FourierEmbedding(features=model.emb_features)(temb)
    temb = TimeProjection(features=model.emb_features)(temb)

    feature_depths = model.feature_depths
    attention_configs = model.attention_configs
    conv = partial(nn.Conv, kernel_size=(3, 3), strides=(1, 1),
                   dtype=model.dtype, precision=model.precision)
    residual = partial(ResidualBlock, kernel_size=(3, 3), activation=model.activation,
                       norm_groups=model.norm_groups, dtype=model.dtype,
                       precision=model.precision)
    attention = partial(stage_attention, attention_impl=model.attention_impl,
                        precision=model.precision)

    x = conv(features=feature_depths[0])(x)
    downs = [x]

    for i, (dim_out, stage) in enumerate(zip(feature_depths, attention_configs)):
        dim_in = x.shape[-1]
        for j in range(model.num_res_blocks):
            x = residual(features=dim_in, name=f"down_{i}_residual_{j}")(x, temb)
            if stage is not None and j == model.num_res_blocks - 1:
                x = attention(stage, dim_in, name=f"down_{i}_attention_{j}")(x, text)
            downs.append(x)
        if temporal is not None:
            x = temporal(x, f"down_{i}_temporal")
        if i != len(feature_depths) - 1:
            x = Downsample(features=dim_out, name=f"down_{i}_downsample",
                           dtype=model.dtype, precision=model.precision)(x)

    middle_dim_out = feature_depths[-1]
    middle_stage = attention_configs[-1]
    for j in range(model.num_middle_res_blocks):
        x = residual(features=middle_dim_out, name=f"middle_res1_{j}")(x, temb)
        if middle_stage is not None and j == model.num_middle_res_blocks - 1:
            # The middle stage attends over the text alone.
            middle = dataclasses.replace(middle_stage, use_self_and_cross=False)
            x = attention(middle, middle_dim_out, name=f"middle_attention_{j}")(x, text)
        if temporal is not None:
            x = temporal(x, f"middle_temporal_{j}")
        x = residual(features=middle_dim_out, name=f"middle_res2_{j}")(x, temb)

    for i, (dim_out, stage) in enumerate(zip(reversed(feature_depths), reversed(attention_configs))):
        for j in range(model.num_res_blocks):
            x = jnp.concatenate([x, downs.pop()], axis=-1)
            x = residual(features=dim_out, name=f"up_{i}_residual_{j}")(x, temb)
            if stage is not None and j == model.num_res_blocks - 1:
                x = attention(stage, dim_out, name=f"up_{i}_attention_{j}")(x, text)
        if temporal is not None:
            x = temporal(x, f"up_{i}_temporal")
        if i != len(feature_depths) - 1:
            x = Upsample(features=feature_depths[-i], scale=2, name=f"up_{i}_upsample",
                         dtype=model.dtype, precision=model.precision)(x)

    x = conv(features=feature_depths[0])(x)
    x = jnp.concatenate([x, downs.pop()], axis=-1)
    x = residual(features=feature_depths[0], name="final_residual")(x, temb)

    x = model.activation(model.conv_out_norm(x))
    return conv(features=model.output_channels)(x)


@models("unet")
@logical_axes({}, heuristic=(("Conv_*",),))
class Unet(nn.Module):
    """A convolutional UNet with residual blocks and cross-attention stages.

    Without text the attention stages self-attend (`TransformerBlock`'s
    context defaults to its input). Cross-attention reads the whole text
    sequence; the mask the DiT family pools with has no place here.
    """
    output_channels:int=3
    emb_features:int=64*4
    feature_depths: Sequence[int] = (64, 128, 256, 512)
    attention_configs: Sequence[Optional[Stage]] = (
        Stage(heads=8), Stage(heads=8), Stage(heads=8), Stage(heads=8))
    """Attention per resolution stage, one entry per feature depth; None is a
    stage with no attention."""
    num_res_blocks:int=2
    num_middle_res_blocks:int=1
    activation:Callable = jax.nn.swish
    norm_groups:int=8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None

    def setup(self):
        if self.norm_groups > 0:
            self.conv_out_norm = nn.GroupNorm(self.norm_groups)
        else:
            self.conv_out_norm = nn.RMSNorm(1e-5)

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train: bool = False):
        text = None if textcontext is None else textcontext.hidden
        return unet_body(self, x, temb, text)
