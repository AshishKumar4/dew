"""EDM2's magnitude-preserving U-Net.

The architecture is Figure 21 of Karras et al. 2024. The code follows
NVlabs/edm2's `training/networks_edm2.py` `UNet`, channels last, built from
`dew.nn.mp`: every convolution and dense layer is an `MPConv`, the time
features are `MPFourier`, branches join with `mp_sum` and skips with
`mp_cat`, and there are no biases or gains. Train it under the `edm` preset
with `OptimConfig.forced_weight_normalization`. In place of the one-hot
class label, it embeds the mask-weighted mean of the `textcontext` states at
unit magnitude through the same dense layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.dit import TextContext, masked_mean
from dew.nn.flops import MissingFlopsCount, attention_flops, patch_lengths, projection_flops
from dew.nn.mp import MPConv, MPFourier, mp_cat, mp_silu, mp_sum, normalize


def _resample(x: jax.Array, mode: str) -> jax.Array:
    """The reference's resampling with its default [1, 1] filter: a 2x2
    average down, a nearest-neighbour doubling up."""
    if mode == "down":
        batch, height, width, channels = x.shape
        return x.reshape(batch, height // 2, 2, width // 2, 2, channels).mean(axis=(2, 4))
    if mode == "up":
        return jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
    return x


class Block(nn.Module):
    """Runs one encoder or decoder block, with optional self-attention."""

    features: int
    flavor: str = "enc"
    resample: str = "keep"
    attention: bool = False
    channels_per_head: int = 64
    dropout: float = 0.0
    res_balance: float = 0.3
    attn_balance: float = 0.3
    clip_act: float | None = 256.0
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.nowrap
    def _residual_conv(self, **scope):
        return MPConv(self.features, (3, 3), **scope)

    @nn.nowrap
    def spatial_flops(self, lengths, input_features: int, emb_features: int) -> float:
        """The block's residual, time and skip maps and optional spatial attention."""
        residual_in = self.features if self.flavor == "enc" else input_features
        conv = self._residual_conv(parent=None)
        count = conv.model_flops(lengths, residual_in) + conv.model_flops(lengths, self.features)
        count += MPConv(self.features, parent=None).model_flops(
            np.ones(np.shape(lengths), np.int64), emb_features)
        if input_features != self.features:
            count += MPConv(self.features, (1, 1), parent=None).model_flops(lengths, input_features)
        if self.attention:
            heads = self.features // self.channels_per_head
            count += MPConv(3 * self.features, (1, 1), parent=None).model_flops(lengths, self.features)
            count += MPConv(self.features, (1, 1), parent=None).model_flops(lengths, self.features)
            count += attention_flops(lengths, heads, self.features // heads,
                                     self.features // heads, causal=False)
        return count

    @nn.compact
    def __call__(self, x: jax.Array, emb: jax.Array, train: bool) -> jax.Array:
        skip = x.shape[-1] != self.features
        x = _resample(x, self.resample)
        if self.flavor == "enc":
            if skip:
                x = MPConv(self.features, (1, 1), name="conv_skip")(x)
            x = normalize(x, axes=(-1,))
        y = self._residual_conv(name="conv_res0")(mp_silu(x))
        gain = self.param("emb_gain", nn.initializers.zeros, ())
        scale = MPConv(self.features, name="emb_linear")(emb, gain=gain) + 1
        y = mp_silu(y * scale[:, None, None, :].astype(y.dtype))
        y = nn.Dropout(self.dropout)(y, deterministic=not train)
        y = self._residual_conv(name="conv_res1")(y)
        if self.flavor == "dec" and skip:
            x = MPConv(self.features, (1, 1), name="conv_skip")(x)
        x = mp_sum(x, y, t=self.res_balance)
        if self.attention:
            heads = self.features // self.channels_per_head
            batch, height, width, _ = x.shape
            qkv = MPConv(self.features * 3, (1, 1), name="attn_qkv")(x)
            # The reference's channel order: heads, then per-head channels,
            # then query, key and value interleaved.
            qkv = qkv.reshape(batch, height * width, heads, self.features // heads, 3)
            q, k, v = (part[..., 0] for part in jnp.split(normalize(qkv, axes=(3,)), 3, axis=-1))
            y = scaled_dot_product_attention(q, k, v, implementation=self.attention_impl)
            y = MPConv(self.features, (1, 1), name="attn_proj")(y.reshape(x.shape))
            x = mp_sum(x, y, t=self.attn_balance)
        if self.clip_act is not None:
            x = jnp.clip(x, -self.clip_act, self.clip_act)
        return x


class EDM2UNet(nn.Module):
    """Runs EDM2's U-Net.

    The fields are the reference's, and `output_channels` is its
    `img_channels`. `attn_resolutions` are feature-map sizes, compared with
    the input's height halved once per level.
    """

    output_channels: int = 3
    model_channels: int = 192
    channel_mult: Sequence[int] = (1, 2, 3, 4)
    channel_mult_noise: int | None = None
    channel_mult_emb: int | None = None
    num_blocks: int = 3
    attn_resolutions: Sequence[int] = (16, 8)
    label_balance: float = 0.5
    concat_balance: float = 0.5
    channels_per_head: int = 64
    dropout: float = 0.0
    res_balance: float = 0.3
    attn_balance: float = 0.3
    clip_act: float | None = 256.0
    dtype: Dtype | None = None
    attention_impl: str = "auto"  # an AttentionImpl

    @property
    def channels(self) -> list[int]:
        return [self.model_channels * mult for mult in self.channel_mult]

    @property
    def noise_features(self) -> int:
        return self.model_channels * (self.channel_mult_noise or self.channel_mult[0])

    @property
    def embedding_features(self) -> int:
        return self.model_channels * (self.channel_mult_emb or max(self.channel_mult))

    @nn.nowrap
    def _convolution(self, features: int, **scope):
        return MPConv(features, (3, 3), **scope)

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        raise MissingFlopsCount("EDM2UNet requires spatial_flops with image shape and optional label width")

    @nn.nowrap
    def spatial_flops(self, sample_shape: tuple[int, ...], rows: int, context_lengths=None,
                       text_features: int | None = None) -> float:
        """Magnitude-preserving convolutions, skip projections and resolution-specific attention."""
        del context_lengths  # The text is mean-pooled before its one label projection.
        divisor = 2**(len(self.channel_mult) - 1)
        patch_lengths(sample_shape, (divisor, divisor), rows)
        height, width, input_channels = sample_shape
        depths, noise, emb = self.channels, self.noise_features, self.embedding_features
        row = np.ones((rows,), np.int64)
        count = projection_flops(row, noise, emb)
        if text_features is not None:
            count += projection_flops(row, text_features, emb)

        def block(area, inputs, outputs, *, encoder: bool, attend: bool):
            counted = Block(outputs, "enc" if encoder else "dec", attention=attend,
                            channels_per_head=self.channels_per_head, parent=None)
            return counted.spatial_flops(area, inputs, emb)

        area = np.full((rows,), height * width, np.int64)
        channels, skips = input_channels + 1, []
        for level, features in enumerate(depths):
            if level == 0:
                count += self._convolution(features, parent=None).model_flops(area, channels)
                channels = features
            else:
                area = area // 4
                count += block(area, channels, channels, encoder=True, attend=False)
            skips.append(channels)
            for _ in range(self.num_blocks):
                count += block(area, channels, features, encoder=True,
                                attend=height // 2**level in self.attn_resolutions)
                channels = features
                skips.append(channels)
        for level in range(len(depths) - 1, -1, -1):
            if level == len(depths) - 1:
                count += block(area, channels, channels, encoder=False, attend=True)
                count += block(area, channels, channels, encoder=False, attend=False)
            else:
                area = area * 4
                count += block(area, channels, channels, encoder=False, attend=False)
            for _ in range(self.num_blocks + 1):
                count += block(area, channels + skips.pop(), depths[level], encoder=False,
                                attend=height // 2**level in self.attn_resolutions)
                channels = depths[level]
        return count + self._convolution(self.output_channels, parent=None).model_flops(area, channels)

    @nn.compact
    def __call__(self, x: jax.Array, time: jax.Array, textcontext: TextContext | None = None,
                 train: bool = False) -> jax.Array:
        channels, noise, width = self.channels, self.noise_features, self.embedding_features
        block = {"channels_per_head": self.channels_per_head, "dropout": self.dropout,
                 "res_balance": self.res_balance, "attn_balance": self.attn_balance,
                 "clip_act": self.clip_act, "attention_impl": self.attention_impl}
        if self.dtype is not None:
            x = x.astype(self.dtype)

        emb = MPConv(width, name="emb_noise")(MPFourier(noise, name="emb_fourier")(time))
        if textcontext is not None:
            label = normalize(masked_mean(textcontext.hidden, textcontext.mask).astype(emb.dtype),
                              axes=(-1,))
            emb = mp_sum(emb, MPConv(width, name="emb_label")(label), t=self.label_balance)
        emb = mp_silu(emb).astype(x.dtype)

        x = jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)
        skips = []
        for level, features in enumerate(channels):
            size = x.shape[1] if level == 0 else skips[-1].shape[1] // 2
            if level == 0:
                x = self._convolution(features, name=f"enc_{size}x{size}_conv")(x)
            else:
                x = Block(x.shape[-1], "enc", "down", name=f"enc_{size}x{size}_down",
                          **block)(x, emb, train)
            skips.append(x)
            for index in range(self.num_blocks):
                x = Block(features, "enc", attention=size in self.attn_resolutions,
                          name=f"enc_{size}x{size}_block{index}", **block)(x, emb, train)
                skips.append(x)

        for level, features in reversed(list(enumerate(channels))):
            size = x.shape[1]
            if level == len(channels) - 1:
                x = Block(x.shape[-1], "dec", attention=True, name=f"dec_{size}x{size}_in0",
                          **block)(x, emb, train)
                x = Block(x.shape[-1], "dec", name=f"dec_{size}x{size}_in1", **block)(x, emb, train)
            else:
                size *= 2
                x = Block(x.shape[-1], "dec", "up", name=f"dec_{size}x{size}_up",
                          **block)(x, emb, train)
            for index in range(self.num_blocks + 1):
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
                x = Block(features, "dec", attention=size in self.attn_resolutions,
                          name=f"dec_{size}x{size}_block{index}", **block)(x, emb, train)
        gain = self.param("out_gain", nn.initializers.zeros, ())
        return self._convolution(self.output_channels, name="out_conv")(x, gain=gain)


__all__ = ["Block", "EDM2UNet"]
