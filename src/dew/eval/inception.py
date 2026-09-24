"""The FID feature extractor: InceptionV3 up to pool3, as pytorch-fid runs it.

Ported from matthias-wright/jax-fid, itself a port of pytorch-fid's
`FIDInceptionA/C/E` blocks over torchvision's InceptionV3. What FID runs is
inference to the 2048-wide pool3 features, so that is all this builds: no
classifier head, no auxiliary branch, no training mode. The norms are
`flax.linen.BatchNorm` over their running statistics, and the average pools
are `flax.linen.avg_pool(count_include_pad=False)`, pytorch-fid's
`F.avg_pool2d(..., count_include_pad=False)`.

The trained weights are a variables tree like any other Flax module's:
`dew.interop.inception_fid` converts the published jax-fid checkpoint into
one, and `apply` takes it.
"""

import functools
from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def _avg_pool(x: jax.Array, window_shape: tuple[int, int], strides: tuple[int, int],
              padding: str) -> jax.Array:
    """pytorch-fid's FIDInceptionA, FIDInceptionC and FIDInceptionE_1 average
    without the zero padding; torchvision's blocks count it. Every pool here
    is 3x3 at stride 1, where "SAME" is one pixel of padding on each side,
    pytorch-fid's `padding=1`."""
    return nn.avg_pool(x, window_shape, strides=strides, padding=padding, count_include_pad=False)


class InceptionV3(nn.Module):
    """InceptionV3 (https://arxiv.org/abs/1512.00567) to its pool3 features.

    `apply(variables, x)` takes [B, 299, 299, 3] pixels in [-1, 1] and returns
    [B, 1, 1, 2048]. `channel_divisor` divides every channel width, which is
    how the tiny test extractor is built.
    """
    channel_divisor: int = 1

    @nn.compact
    def __call__(self, x):
        conv = functools.partial(BasicConv2d, channel_divisor=self.channel_divisor)
        d = self.channel_divisor
        x = conv(out_channels=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = conv(out_channels=32, kernel_size=(3, 3))(x)
        x = conv(out_channels=64, kernel_size=(3, 3), padding=((1, 1), (1, 1)))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = conv(out_channels=80, kernel_size=(1, 1))(x)
        x = conv(out_channels=192, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = InceptionA(pool_features=32, channel_divisor=d)(x)
        x = InceptionA(pool_features=64, channel_divisor=d)(x)
        x = InceptionA(pool_features=64, channel_divisor=d)(x)
        x = InceptionB(channel_divisor=d)(x)
        x = InceptionC(channels_7x7=128, channel_divisor=d)(x)
        x = InceptionC(channels_7x7=160, channel_divisor=d)(x)
        x = InceptionC(channels_7x7=160, channel_divisor=d)(x)
        x = InceptionC(channels_7x7=192, channel_divisor=d)(x)
        x = InceptionD(channel_divisor=d)(x)
        x = InceptionE(_avg_pool, channel_divisor=d)(x)
        # pytorch-fid's FIDInceptionE_2 pools its last block with a max pool,
        # following the TensorFlow graph the published weights came from.
        x = InceptionE(nn.max_pool, channel_divisor=d)(x)
        return jnp.mean(x, axis=(1, 2), keepdims=True)


class BasicConv2d(nn.Module):
    out_channels: int
    channel_divisor: int
    kernel_size: Sequence[int] = (3, 3)
    strides: Sequence[int] = (1, 1)
    padding: str | Sequence[tuple[int, int]] = 'valid'

    @nn.compact
    def __call__(self, x):
        # Every width in the network reaches a convolution through here, so
        # the divisor is applied once, at the only place a filter count is
        # declared, and so is the precision: pytorch-fid's features are fp32
        # convolutions, and a TPU's DEFAULT is one bf16 pass. The norm below
        # takes its shape from the input.
        x = nn.Conv(features=max(self.out_channels // self.channel_divisor, 1),
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    use_bias=False,
                    precision=jax.lax.Precision.HIGHEST)(x)
        x = nn.BatchNorm(use_running_average=True, epsilon=0.001)(x)
        return jax.nn.relu(x)


class InceptionA(nn.Module):
    pool_features: int
    channel_divisor: int

    @nn.compact
    def __call__(self, x):
        conv = functools.partial(BasicConv2d, channel_divisor=self.channel_divisor)
        branch1x1 = conv(out_channels=64, kernel_size=(1, 1))(x)
        branch5x5 = conv(out_channels=48, kernel_size=(1, 1))(x)
        branch5x5 = conv(out_channels=64, kernel_size=(5, 5), padding=((2, 2), (2, 2)))(branch5x5)

        branch3x3dbl = conv(out_channels=64, kernel_size=(1, 1))(x)
        branch3x3dbl = conv(out_channels=96, kernel_size=(3, 3),
                            padding=((1, 1), (1, 1)))(branch3x3dbl)
        branch3x3dbl = conv(out_channels=96, kernel_size=(3, 3),
                            padding=((1, 1), (1, 1)))(branch3x3dbl)

        branch_pool = _avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        branch_pool = conv(out_channels=self.pool_features, kernel_size=(1, 1))(branch_pool)
        return jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=-1)


class InceptionB(nn.Module):
    channel_divisor: int

    @nn.compact
    def __call__(self, x):
        conv = functools.partial(BasicConv2d, channel_divisor=self.channel_divisor)
        branch3x3 = conv(out_channels=384, kernel_size=(3, 3), strides=(2, 2))(x)

        branch3x3dbl = conv(out_channels=64, kernel_size=(1, 1))(x)
        branch3x3dbl = conv(out_channels=96, kernel_size=(3, 3),
                            padding=((1, 1), (1, 1)))(branch3x3dbl)
        branch3x3dbl = conv(out_channels=96, kernel_size=(3, 3), strides=(2, 2))(branch3x3dbl)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        return jnp.concatenate((branch3x3, branch3x3dbl, branch_pool), axis=-1)


class InceptionC(nn.Module):
    channels_7x7: int
    channel_divisor: int

    @nn.compact
    def __call__(self, x):
        conv = functools.partial(BasicConv2d, channel_divisor=self.channel_divisor)
        wide, tall = ((0, 0), (3, 3)), ((3, 3), (0, 0))
        branch1x1 = conv(out_channels=192, kernel_size=(1, 1))(x)

        branch7x7 = conv(out_channels=self.channels_7x7, kernel_size=(1, 1))(x)
        branch7x7 = conv(out_channels=self.channels_7x7, kernel_size=(1, 7), padding=wide)(branch7x7)
        branch7x7 = conv(out_channels=192, kernel_size=(7, 1), padding=tall)(branch7x7)

        branch7x7dbl = conv(out_channels=self.channels_7x7, kernel_size=(1, 1))(x)
        branch7x7dbl = conv(out_channels=self.channels_7x7, kernel_size=(7, 1),
                            padding=tall)(branch7x7dbl)
        branch7x7dbl = conv(out_channels=self.channels_7x7, kernel_size=(1, 7),
                            padding=wide)(branch7x7dbl)
        branch7x7dbl = conv(out_channels=self.channels_7x7, kernel_size=(7, 1),
                            padding=tall)(branch7x7dbl)
        # The last of the seven-by-seven pair widens to 192, as torchvision and
        # the published weights do.
        branch7x7dbl = conv(out_channels=192, kernel_size=(1, 7), padding=wide)(branch7x7dbl)

        branch_pool = _avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        branch_pool = conv(out_channels=192, kernel_size=(1, 1))(branch_pool)
        return jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=-1)


class InceptionD(nn.Module):
    channel_divisor: int

    @nn.compact
    def __call__(self, x):
        conv = functools.partial(BasicConv2d, channel_divisor=self.channel_divisor)
        branch3x3 = conv(out_channels=192, kernel_size=(1, 1))(x)
        branch3x3 = conv(out_channels=320, kernel_size=(3, 3), strides=(2, 2))(branch3x3)

        branch7x7x3 = conv(out_channels=192, kernel_size=(1, 1))(x)
        branch7x7x3 = conv(out_channels=192, kernel_size=(1, 7), padding=((0, 0), (3, 3)))(branch7x7x3)
        branch7x7x3 = conv(out_channels=192, kernel_size=(7, 1), padding=((3, 3), (0, 0)))(branch7x7x3)
        branch7x7x3 = conv(out_channels=192, kernel_size=(3, 3), strides=(2, 2))(branch7x7x3)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        return jnp.concatenate((branch3x3, branch7x7x3, branch_pool), axis=-1)


class InceptionE(nn.Module):
    pooling: Callable
    channel_divisor: int

    @nn.compact
    def __call__(self, x):
        conv = functools.partial(BasicConv2d, channel_divisor=self.channel_divisor)
        branch1x1 = conv(out_channels=320, kernel_size=(1, 1))(x)

        branch3x3 = conv(out_channels=384, kernel_size=(1, 1))(x)
        branch3x3_a = conv(out_channels=384, kernel_size=(1, 3), padding=((0, 0), (1, 1)))(branch3x3)
        branch3x3_b = conv(out_channels=384, kernel_size=(3, 1), padding=((1, 1), (0, 0)))(branch3x3)
        branch3x3 = jnp.concatenate((branch3x3_a, branch3x3_b), axis=-1)

        branch3x3dbl = conv(out_channels=448, kernel_size=(1, 1))(x)
        branch3x3dbl = conv(out_channels=384, kernel_size=(3, 3),
                            padding=((1, 1), (1, 1)))(branch3x3dbl)
        branch3x3dbl_a = conv(out_channels=384, kernel_size=(1, 3),
                              padding=((0, 0), (1, 1)))(branch3x3dbl)
        branch3x3dbl_b = conv(out_channels=384, kernel_size=(3, 1),
                              padding=((1, 1), (0, 0)))(branch3x3dbl)
        branch3x3dbl = jnp.concatenate((branch3x3dbl_a, branch3x3dbl_b), axis=-1)

        branch_pool = self.pooling(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        branch_pool = conv(out_channels=192, kernel_size=(1, 1))(branch_pool)
        return jnp.concatenate((branch1x1, branch3x3, branch3x3dbl, branch_pool), axis=-1)
