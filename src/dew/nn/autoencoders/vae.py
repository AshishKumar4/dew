"""Build the convolutional encoder and decoder of a diffusion VAE.

The modules are independent linen ports of huggingface/diffusers v0.29.2
src/diffusers/models/vae_flax.py (Apache-2.0). The loader below reads source
configuration and checkpoint files; no external model implementation runs.
"""

import logging
from collections.abc import Sequence
from functools import partial
from typing import Literal

import flax.linen as nn
import jax.numpy as jnp
from flax.typing import Dtype

from dew.nn.attention_blocks import NormalAttention
from dew.nn.blocks import ResidualBlock, Upsample
from dew.nn.conv import Conv
from dew.nn.norms import GroupNorm

_log = logging.getLogger(__name__)


def _conv(features: int, dtype, name: str) -> Conv:
    """A 3x3 convolution padded to keep the size."""
    return Conv(features, (3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), dtype=dtype, name=name)


class FlaxDownsample2D(nn.Module):
    """Halve each spatial axis with a stride-2 3x3 conv, padded on the far side."""

    in_channels: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        return Conv(self.in_channels, kernel_size=(3, 3), strides=(2, 2), padding="VALID", dtype=self.dtype,
                    name="conv")(jnp.pad(hidden_states, pad_width=pad))


def _resnet(features: int, groups: int, dtype, name: str | None = None) -> ResidualBlock:
    """An AutoencoderKL resnet: the shared residual block called without a time
    embedding, with diffusers' norms (1e-6, then SiLU) and its 1x1 shortcut
    under the checkpoint's `conv_shortcut`."""
    return ResidualBlock(features, norm_groups=groups, norm_epsilon=1e-6, dtype=dtype,
                         shortcut="conv_shortcut", name=name)


class FlaxAttentionBlock(nn.Module):
    """Attend over an image's pixels as a sequence, with a residual.

    One head as wide as the channels, which is what every AutoencoderKL
    mid-block runs, as an ordinary attention (`NormalAttention`) whose
    projections keep the block's own flat `query/key/value/proj_attn` and
    share its scope. Its logits are scaled by 1/sqrt(channels), the scale
    diffusers splits over the query and the key: the same function, rounded
    in another order.
    """

    channels: int
    num_groups: int = 32
    dtype: Dtype = jnp.float32

    def setup(self):
        self.group_norm = GroupNorm(self.num_groups, dtype=self.dtype)
        self.attention = NormalAttention(self.channels, 1, self.channels, dtype=self.dtype, flat=True,
                                         names=("query", "key", "value", "proj_attn"))
        nn.share_scope(self, self.attention)

    def __call__(self, hidden_states):
        return self.attention(self.group_norm(hidden_states)) + hidden_states


class _Level(nn.Module):
    """One encoder or decoder level: `num_layers` resnets to `out_channels`,
    only the first changing the channel count, then the level's resampling,
    `downsamplers_0` or `upsamplers_0`, which the last level does without."""

    out_channels: int
    num_layers: int
    groups: int
    resample: Literal["down", "up"] | None
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        for index in range(self.num_layers):
            hidden_states = _resnet(self.out_channels, self.groups, self.dtype,
                                    f"resnets_{index}")(hidden_states)
        if self.resample == "down":
            return FlaxDownsample2D(self.out_channels, dtype=self.dtype, name="downsamplers_0")(hidden_states)
        if self.resample == "up":
            return Upsample(self.out_channels, 2, dtype=self.dtype, conv="conv",
                            name="upsamplers_0")(hidden_states)
        return hidden_states


class FlaxUNetMidBlock2D(nn.Module):
    """A resnet, an attention block and a second resnet at the bottleneck
    resolution. The channel count does not change."""

    in_channels: int
    resnet_groups: int = 32
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        resnet = partial(_resnet, self.in_channels, self.resnet_groups, self.dtype)
        hidden_states = resnet("resnets_0")(hidden_states)
        hidden_states = FlaxAttentionBlock(self.in_channels, self.resnet_groups, self.dtype,
                                           name="attentions_0")(hidden_states)
        return resnet("resnets_1")(hidden_states)


class FlaxEncoder(nn.Module):
    """Encode images to latent moments with a conv stack that halves each axis.

    `block_out_channels` gives one level per entry, each `layers_per_block`
    resnets; the last level does not downsample. The output carries twice
    `out_channels`, which the caller splits into mean and log-variance.
    """

    out_channels: int = 3
    block_out_channels: Sequence[int] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, sample):
        channels = self.block_out_channels
        sample = _conv(channels[0], self.dtype, "conv_in")(sample)
        for index, width in enumerate(channels):
            sample = _Level(width, self.layers_per_block, self.norm_num_groups,
                            "down" if index != len(channels) - 1 else None, self.dtype,
                            name=f"down_blocks_{index}")(sample)
        sample = FlaxUNetMidBlock2D(channels[-1], self.norm_num_groups, self.dtype, name="mid_block")(sample)
        sample = GroupNorm(self.norm_num_groups, activation=nn.swish, dtype=self.dtype,
                           name="conv_norm_out")(sample)
        return _conv(2 * self.out_channels, self.dtype, "conv_out")(sample)


class FlaxDecoder(nn.Module):
    """Decode latents to images with a conv stack that doubles each axis.

    `block_out_channels` is read in reverse, one level per entry, each
    `layers_per_block + 1` resnets; the last level does not upsample.
    """

    out_channels: int = 3
    block_out_channels: Sequence[int] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, sample):
        channels = list(reversed(self.block_out_channels))
        sample = _conv(channels[0], self.dtype, "conv_in")(sample)
        sample = FlaxUNetMidBlock2D(channels[0], self.norm_num_groups, self.dtype, name="mid_block")(sample)
        for index, width in enumerate(channels):
            sample = _Level(width, self.layers_per_block + 1, self.norm_num_groups,
                            "up" if index != len(channels) - 1 else None, self.dtype,
                            name=f"up_blocks_{index}")(sample)
        sample = GroupNorm(self.norm_num_groups, activation=nn.swish, dtype=self.dtype,
                           name="conv_norm_out")(sample)
        return _conv(self.out_channels, self.dtype, "conv_out")(sample)


# The SD1-era flax weights live on their own branches, so these name a layout,
# not a pinned revision.


