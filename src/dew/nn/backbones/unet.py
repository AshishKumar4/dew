"""The convolutional UNet, and the body it shares with the video UNet."""

import dataclasses
from collections.abc import Callable, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from ..attention import RMSNorm, Stage, stage_attention
from ..blocks import Downsample, FourierEmbedding, ResidualBlock, TimeProjection, Upsample
from ..conv import Conv
from ..flops import MissingFlopsCount, patch_lengths


def unet_body(model: "Unet", x, temb, text, temporal=None):
    """The UNet forward over frames `[N, H, W, C]`, run inside the model's
    compact `__call__` so every block here is the model's own submodule.

    `temporal(x, name)`, when given, runs after the residual blocks of every
    level; the video UNet mixes across frames there. The spatial blocks and
    their names are the same either way, which lets a 2D checkpoint inflate
    into the 3D model.
    """
    temb = FourierEmbedding(features=model.emb_features, dtype=model.dtype)(temb)
    temb = TimeProjection(features=model.emb_features, dtype=model.dtype,
                          precision=model.precision)(temb)

    feature_depths = model.feature_depths
    attention_configs = model.attention_configs
    if len(attention_configs) != len(feature_depths):
        # The decoder walks the two reversed, so a disagreement does not drop
        # the odd stage: it offsets the levels the decoder attends in from the
        # ones the encoder does, and the halves stop mirroring silently.
        raise ValueError(
            "attention_configs names one stage per feature depth; got "
            f"{len(attention_configs)} stages for {len(feature_depths)} depths")

    conv = model._convolution
    residual = partial(ResidualBlock, activation=model.activation,
                       norm_groups=model.norm_groups, dtype=model.dtype,
                       precision=model.precision)
    attention = partial(stage_attention, attention_impl=model.attention_impl, dtype=model.dtype,
                        precision=model.precision)

    x = conv(features=feature_depths[0])(x)
    downs = [x]

    for i, (dim_out, stage) in enumerate(zip(feature_depths, attention_configs, strict=True)):
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

    for i, (dim_out, stage) in enumerate(
            zip(reversed(feature_depths), reversed(attention_configs), strict=True)):
        for j in range(model.num_res_blocks):
            x = jnp.concatenate([x, downs.pop()], axis=-1)
            x = residual(features=dim_out, name=f"up_{i}_residual_{j}")(x, temb)
            if stage is not None and j == model.num_res_blocks - 1:
                x = attention(stage, dim_out, name=f"up_{i}_attention_{j}")(x, text)
        if temporal is not None:
            x = temporal(x, f"up_{i}_temporal")
        if i != len(feature_depths) - 1:
            x = Upsample(features=feature_depths[-i - 2], scale=2, name=f"up_{i}_upsample",
                         dtype=model.dtype, precision=model.precision)(x)

    x = conv(features=feature_depths[0])(x)
    x = jnp.concatenate([x, downs.pop()], axis=-1)
    x = residual(features=feature_depths[0], name="final_residual")(x, temb)

    x = model.activation(model.conv_out_norm(x))
    return conv(features=model.output_channels)(x)


class Unet(nn.Module):
    """Denoises images with a convolutional UNet of residual blocks and cross-attention stages.

    Without text, the attention stages self-attend (`TransformerBlock`'s
    context defaults to its input). Cross-attention reads the whole text
    sequence, so the mask that the DiT family pools with is not used here.

    It is FlaxDiff's `Unet` except for one width. The decoder upsamples level
    i into `feature_depths[-i - 2]`, the width of the level it enters, where
    FlaxDiff's upsamples into `feature_depths[-i]`. The two agree for two
    levels or for equal depths.
    """
    # tests/test_flaxdiff_family_source.py holds this model to FlaxDiff's in
    # the configurations where the two agree.
    output_channels:int=3
    emb_features:int=64*4
    feature_depths: Sequence[int] = (64, 128, 256, 512)
    attention_configs: Sequence[Stage | None] = (
        Stage(heads=8), Stage(heads=8), Stage(heads=8), Stage(heads=8))
    """The attention of each resolution stage, one entry per feature depth.
    None is a stage with no attention."""
    num_res_blocks:int=2
    num_middle_res_blocks:int=1
    activation:Callable = jax.nn.swish
    norm_groups:int=8
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.nowrap
    def _convolution(self, features: int, **scope):
        return Conv(features, (3, 3), dtype=self.dtype, precision=self.precision, **scope)

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount(f"{type(self).__name__} needs spatial_flops with image and text geometry")

    @nn.nowrap
    def spatial_flops(self, sample_shape: tuple[int, ...], rows: int, context_lengths=None,
                       text_features: int | None = None) -> float:
        """The same stage walk as unet_body, with convolution output areas explicit."""
        divisor = 2**(len(self.feature_depths) - 1)
        patch_lengths(sample_shape, (divisor, divisor), rows)
        height, width, channels = sample_shape
        area = np.full((rows,), height * width, np.int64)
        row = np.ones((rows,), np.int64)
        first = self.feature_depths[0]
        count = TimeProjection(self.emb_features, parent=None).model_flops(row, self.emb_features)
        count += self._convolution(first, parent=None).model_flops(area, channels)
        channels, skips = first, [first]

        def attention(stage, features, positions):
            if stage is None:
                return 0.0
            block = stage_attention(stage, features, self.attention_impl, self.dtype,
                                    self.precision, name="counted_attention")
            return block.cross_flops(positions, features, context_lengths, text_features)

        for level, (features, stage) in enumerate(zip(
                self.feature_depths, self.attention_configs, strict=True)):
            for index in range(self.num_res_blocks):
                count += ResidualBlock(channels, parent=None).model_flops(area, channels, self.emb_features)
                skips.append(channels)
                if index == self.num_res_blocks - 1:
                    count += attention(stage, channels, area)
            if level < len(self.feature_depths) - 1:
                area = area // 4
                count += Downsample(features, parent=None).model_flops(area, channels)
                channels = features
        for _ in range(self.num_middle_res_blocks):
            features = self.feature_depths[-1]
            count += ResidualBlock(features, parent=None).model_flops(area, channels, self.emb_features)
            channels = features
            count += ResidualBlock(channels, parent=None).model_flops(area, channels, self.emb_features)
        if self.num_middle_res_blocks and self.attention_configs[-1] is not None:
            stage = dataclasses.replace(self.attention_configs[-1], use_self_and_cross=False)
            count += attention(stage, channels, area)
        for level in range(len(self.feature_depths) - 1, -1, -1):
            features = self.feature_depths[level]
            for index in range(self.num_res_blocks):
                count += ResidualBlock(features, parent=None).model_flops(
                    area, channels + skips.pop(), self.emb_features)
                channels = features
                if index == self.num_res_blocks - 1:
                    count += attention(self.attention_configs[level], channels, area)
            if level:
                area = area * 4
                count += Upsample(self.feature_depths[level - 1], 2, parent=None).model_flops(area, channels)
                channels = self.feature_depths[level - 1]
        count += self._convolution(first, parent=None).model_flops(area, channels)
        count += ResidualBlock(first, parent=None).model_flops(area, first + skips.pop(), self.emb_features)
        return count + self._convolution(self.output_channels, parent=None).model_flops(area, first)

    def setup(self):
        if self.norm_groups > 0:
            self.conv_out_norm = nn.GroupNorm(self.norm_groups, dtype=self.dtype)
        else:
            self.conv_out_norm = RMSNorm(epsilon=1e-5, dtype=self.dtype)

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train: bool = False):
        text = None if textcontext is None else textcontext.hidden
        return unet_body(self, x, temb, text)
