"""Conditional convolutional UNet for SD1/2 and SDXL latent denoising."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import BasicTransformerBlock, GroupNorm
from dew.nn.blocks import ResidualBlock, sinusoidal_time
from dew.nn.conv import Conv
from dew.nn.dense import Dense
from dew.nn.flops import MissingFlopsCount, projection_flops
from dew.nn.precision import at_least_fp32, float32_dot_general

from .decoder_stack import pipelined_blocks, pipelines
from .joint import embedding, embedding_flops
from .spatial import SpatialCount, SpatialForward, SpatialShape

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition
    from dew.inputs.encoders import ConditionWork


@dataclass(frozen=True)
class UNetStage:
    """Describes one resolution level: its width, heads and transformer depth.

    With `cross_attention`, the level has a spatial attention block, and
    without it the level has none. `cross_only` drops the self-attention
    inside that block.
    """

    features: int
    heads: int
    depth: int = 1
    cross_attention: bool = True
    cross_only: bool = False


class _SpatialAttention(nn.Module):
    """Flatten an image to tokens, run `stage.depth` transformers, fold it back.

    A group norm and a residual wrap the whole block. `linear_projection`
    picks the projection into and out of the token width, a dense layer or a
    1x1 convolution; SD1/2 store the convolution and SDXL the dense layer.
    Nothing in it drops: diffusers' conditional UNet builds its transformers
    without the model's dropout, which only its residual blocks take.
    """

    stage: UNetStage
    linear_projection: bool
    dtype: Dtype
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    norm_groups: int = 32
    norm_epsilon: float = 1e-5
    approximate_gelu: bool = True

    @nn.nowrap
    def cross_flops(self, lengths, context_lengths, text_features: int) -> float:
        """Spatial input/output maps and every configured transformer sublayer."""
        width = self.stage.features
        first = "context" if self.stage.cross_only else "self"
        heads = self.stage.heads
        block = BasicTransformerBlock(width, heads, width // heads, first_attention=first, parent=None)
        return (2 * projection_flops(lengths, width, width)
                + self.stage.depth * block.cross_flops(lengths, context_lengths, text_features))

    @nn.compact
    def __call__(self, x, context, *, train=False):
        residual = x
        x = GroupNorm(self.norm_groups, epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")(x)
        batch, height, width, channels = x.shape
        if self.linear_projection:
            x = x.reshape(batch, height * width, channels)
            x = Dense(channels, dtype=self.dtype, precision=self.precision, name="input",
                      dot_general=float32_dot_general)(x)
        else:
            x = Conv(channels, (1, 1), padding="VALID", dtype=self.dtype,
                     precision=self.precision, name="input")(x).reshape(batch, height * width, channels)
        def layer(index: int) -> BasicTransformerBlock:
            # Affine layer norms at 1e-5, no query, key or value bias, and under `stage.cross_only`
            # a first attention that reads the text too: what the SD checkpoints hold.
            return BasicTransformerBlock(
                self.stage.features, self.stage.heads, self.stage.features // self.stage.heads, self.dtype,
                self.precision, qkv_bias=False,
                first_attention="context" if self.stage.cross_only else "self", norm="layer",
                norm_epsilon=1e-5, approximate_gelu=self.approximate_gelu,
                attention_impl=self.attention_impl, name=f"layer_{index}")

        def run(blocks, x, context):
            for block in blocks:
                x = block(x, context, train=train)
            return x

        # The level's transformers are like blocks, which a stage axis that
        # splits them runs as a pipeline; SDXL's levels of two and ten do.
        if pipelines(self, ["transformer"] * self.stage.depth):
            x = pipelined_blocks(self, ("layer_",), self.stage.depth, x, layer, run, rows=context)
        else:
            x = run([layer(index) for index in range(self.stage.depth)], x, context)
        if self.linear_projection:
            x = Dense(channels, dtype=self.dtype, precision=self.precision, name="output",
                      dot_general=float32_dot_general)(x)
            x = x.reshape(batch, height, width, channels)
        else:
            x = Conv(channels, (1, 1), padding="VALID", dtype=self.dtype, precision=self.precision,
                     name="output")(x.reshape(batch, height, width, channels))
        return residual + x


class _Level(nn.Module):
    """Run one resolution level of the encoder or the decoder.

    Each of `blocks` steps is a residual block on the time embedding, then
    the level's spatial attention when the stage has one. A 'up' level
    concatenates a skip onto its input first. `resize` adds the level's
    final convolution, strided in the encoder and preceded by a
    nearest-neighbour upsample in the decoder. Returns the level's output
    and every output the decoder consumes as a skip.
    """

    stage: UNetStage
    blocks: int
    direction: str
    resize: bool
    linear_projection: bool
    dropout: float
    dtype: Dtype
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    norm_groups: int = 32
    norm_epsilon: float = 1e-5
    attention_norm_epsilon: float = 1e-5
    approximate_gelu: bool = True

    @nn.nowrap
    def _resize_conv(self, **scope):
        stride = (2, 2) if self.direction == "down" else (1, 1)
        return Conv(self.stage.features, (3, 3), strides=stride, padding=((1, 1), (1, 1)),
                    dtype=self.dtype, precision=self.precision, **scope)

    @nn.compact
    def __call__(self, x, time, context, skips=(), *, upsample_shape=None, train=False):
        return self.walk(x, time, context, skips, upsample_shape=upsample_shape, train=train,
                         operations=SpatialForward())

    @nn.nowrap
    def walk(self, x, time, context, skips=(), *, upsample_shape=None, train=False, operations):
        outputs = []
        for index in range(self.blocks):
            if self.direction == "up":
                x = operations.concatenate(x, skips[-(index + 1)])
            x = operations.residual(ResidualBlock(
                self.stage.features,
                norm_groups=self.norm_groups,
                norm_epsilon=self.norm_epsilon,
                dropout=self.dropout,
                dtype=self.dtype,
                precision=self.precision,
                name=f"residual_{index}",
            ), x, operations.activated(time, nn.silu), train=train)
            if self.stage.cross_attention:
                x = operations.attention(_SpatialAttention(
                    self.stage,
                    self.linear_projection,
                    self.dtype,
                    self.precision,
                    self.attention_impl,
                    norm_groups=self.norm_groups,
                    norm_epsilon=self.attention_norm_epsilon,
                    approximate_gelu=self.approximate_gelu,
                    name=f"attention_{index}",
                ), x, context, train=train)
            outputs.append(x)
        if self.resize:
            if self.direction == "up":
                if upsample_shape is None:
                    raise ValueError("Upsampling requires the next skip spatial shape")
                height, width = upsample_shape
                x = operations.resize(x, height, width)
            x = operations.project(self._resize_conv(name="resize"), x)
            outputs.append(x)
        return x, tuple(outputs)


class UNet2DCondition(nn.Module):
    """Denoises NHWC latents conditioned on text, pooled and size features, and optional inpainting inputs.

    The stages set each level's spatial width, attention heads and
    transformer depth. The encoder saves each residual output and each
    downsample, and the decoder uses those skips in reverse order, with one
    extra residual block per level.
    """
    stages: tuple[UNetStage, ...]
    in_channels: int = 4
    out_channels: int = 4
    blocks_per_level: int = 2
    linear_projection: bool = False
    additional_time_features: int = 0
    middle_attention: bool = True
    frequency_shift: float = 0
    cosine_first: bool = True
    dropout: float = 0.0
    dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    norm_groups: int = 32
    norm_epsilon: float = 1e-5
    attention_norm_epsilon: float = 1e-5
    approximate_gelu: bool = True

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("UNet2DCondition needs spatial_flops with image/text/pooled geometry")

    @property
    def time_features(self) -> int:
        return self.stages[0].features * 4

    @nn.nowrap
    def condition_lengths(self, lengths: np.ndarray, spans: np.ndarray) -> np.ndarray:
        return spans

    @nn.nowrap
    def _convolution(self, features: int, **scope):
        return Conv(features, (3, 3), dtype=self.dtype, precision=self.precision, **scope)

    @nn.nowrap
    def conditioned_spatial_flops(self, sample_shape: tuple[int, ...], rows: int,
                                   condition: ConditionWork) -> float:
        additional = None
        if self.additional_time_features:
            if condition.pooled_features is None or condition.time_ids is None:
                raise MissingFlopsCount("UNet2DCondition needs pooled width and the number of time ids")
            additional = condition.pooled_features + condition.time_ids * self.additional_time_features
        lengths = self.condition_lengths(condition.lengths, condition.spans)
        return self.spatial_flops(sample_shape, rows, lengths, condition.features,
                                  additional_features=additional)

    @nn.nowrap
    def spatial_flops(self, sample_shape: tuple[int, ...], rows: int, context_lengths=None,
                       text_features: int | None = None, *, additional_features: int | None = None) -> float:
        """The residual/attention stage walk, including every resizer and skip width."""
        if len(sample_shape) != 3 or context_lengths is None or text_features is None:
            raise MissingFlopsCount("UNet2DCondition needs [height,width,channels] and text geometry")
        height, width, _ = sample_shape
        first = self.stages[0].features
        row = np.ones((rows,), np.int64)
        count = embedding_flops(row, first, self.time_features)
        if self.additional_time_features:
            if additional_features is None:
                raise MissingFlopsCount("UNet2DCondition needs pooled width plus sinusoidal time-id width")
            count += embedding_flops(row, additional_features, self.time_features)
        operations = SpatialCount(context_lengths, text_features)
        operations.flops = count
        self.spatial_walk(SpatialShape((rows, height, width, self.in_channels)),
                          SpatialShape((rows, self.time_features)), None, operations=operations)
        return operations.flops

    @property
    def text_keyword(self) -> str:
        """Every call takes the text as `conditioning`, which every attention stage cross-attends to."""
        return "conditioning"

    @nn.compact
    def __call__(
        self, x, time, *, conditioning: DenoisingCondition, mask=None, masked_image=None, train=False
    ):
        if mask is not None:
            if masked_image is None:
                raise ValueError("An inpaint mask needs its masked-image latents")
            x = jnp.concatenate([x, mask, masked_image], axis=-1)
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"UNet input has {x.shape[-1]} channels; expected {self.in_channels}")
        first = self.stages[0].features
        time = sinusoidal_time(time, first, dtype=at_least_fp32(self.dtype), shift=self.frequency_shift,
                               cosine_first=self.cosine_first)
        time = embedding(self, time, self.time_features, "time", dtype=self.dtype, precision=self.precision)
        if self.additional_time_features:
            if conditioning.pooled is None or conditioning.time_ids is None:
                raise ValueError("This UNet needs pooled text and size/aesthetic conditioning")
            ids = sinusoidal_time(conditioning.time_ids.reshape(-1), self.additional_time_features,
                                  dtype=at_least_fp32(self.dtype), shift=self.frequency_shift,
                                  cosine_first=self.cosine_first)
            extra = jnp.concatenate([conditioning.pooled, ids.reshape(x.shape[0], -1)], axis=-1)
            time = time + embedding(self, extra, self.time_features, "additional_time", dtype=self.dtype,
                                    precision=self.precision)
        return self.spatial_walk(x, time, conditioning.context, train=train, operations=SpatialForward())

    @nn.nowrap
    def spatial_walk(self, x, time, context, *, train=False, operations):
        first = self.stages[0].features
        x = operations.project(self._convolution(first, name="input"), x)
        skips = [x]
        for index, stage in enumerate(self.stages):
            x, outputs = operations.level(_Level(
                stage, self.blocks_per_level, "down", index + 1 < len(self.stages),
                self.linear_projection, self.dropout, self.dtype, self.precision, self.attention_impl,
                norm_groups=self.norm_groups, norm_epsilon=self.norm_epsilon,
                attention_norm_epsilon=self.attention_norm_epsilon, approximate_gelu=self.approximate_gelu,
                name=f"down_{index}"), x, time, context, train=train)
            skips.extend(outputs)
        if self.middle_attention:
            last = self.stages[-1]
            stage = UNetStage(last.features, last.heads, last.depth)
            x = operations.residual(ResidualBlock(
                stage.features,
                norm_groups=self.norm_groups,
                norm_epsilon=self.norm_epsilon,
                dropout=self.dropout,
                dtype=self.dtype,
                precision=self.precision,
                name="middle_in",
            ), x, operations.activated(time, nn.silu), train=train)
            x = operations.attention(_SpatialAttention(
                stage,
                self.linear_projection,
                self.dtype,
                self.precision,
                self.attention_impl,
                norm_groups=self.norm_groups,
                norm_epsilon=self.attention_norm_epsilon,
                approximate_gelu=self.approximate_gelu,
                name="middle_attention",
            ), x, context, train=train)
            x = operations.residual(ResidualBlock(
                stage.features,
                norm_groups=self.norm_groups,
                norm_epsilon=self.norm_epsilon,
                dropout=self.dropout,
                dtype=self.dtype,
                precision=self.precision,
                name="middle_out",
            ), x, operations.activated(time, nn.silu), train=train)
        for index, stage in enumerate(reversed(self.stages)):
            count = self.blocks_per_level + 1
            inputs, skips = tuple(skips[-count:]), skips[:-count]
            target = skips[-1].shape[1:3] if skips else None
            x, _ = operations.level(_Level(
                stage,
                count,
                "up",
                index + 1 < len(self.stages),
                self.linear_projection,
                self.dropout,
                self.dtype,
                self.precision,
                self.attention_impl,
                norm_groups=self.norm_groups,
                norm_epsilon=self.norm_epsilon,
                attention_norm_epsilon=self.attention_norm_epsilon,
                approximate_gelu=self.approximate_gelu,
                name=f"up_{index}",
            ), x, time, context, inputs, upsample_shape=target, train=train)
        norm = GroupNorm(self.norm_groups, epsilon=self.norm_epsilon, activation=nn.silu, dtype=self.dtype,
                         name="output_norm")
        x = operations.normalized(x, norm)
        return operations.project(self._convolution(self.out_channels, name="output"), x)
