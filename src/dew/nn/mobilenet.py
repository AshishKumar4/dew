"""Gemma 3n's MobileNet-v5 encoder, from timm 1.0.29.

The reference is timm/models/mobilenetv5.py, _efficientnet_blocks.py and
layers/attention2d.py (Apache 2.0). The encoder combines convolutional
residual blocks, spatial multi-query attention and a multiscale adapter.
Timm has no Flax implementation. Convolutions use Linen; attention uses
Dew's kernel seam. Inputs are processor-ready NCHW pixels and outputs are
row-major spatial tokens, [batch, resolution**2, 2048].
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .attention import scaled_dot_product_attention


def _divisible(channels: float) -> int:
    rounded = max(8, int(channels + 4) // 8 * 8)
    return rounded + 8 if rounded < 0.9 * channels else rounded


def _padding(kind: str, kernel: int):
    if kind == "same":
        return "SAME"
    if kind == "valid":
        return "VALID"
    if kind == "":
        return ((kernel // 2, kernel // 2),) * 2
    raise ValueError(f"pad_type must be 'same', 'valid' or '', got {kind!r}")


def _conv(features: int, kernel: int, *, stride: int = 1, groups: int = 1,
          padding: str = "same", bias: bool = False, dtype: Dtype | None = None,
          precision: PrecisionLike = None, name: str) -> nn.Conv:
    return nn.Conv(
        features, (kernel, kernel), strides=(stride, stride),
        padding=_padding(padding, kernel), feature_group_count=groups,
        use_bias=bias, dtype=dtype, precision=precision,
        kernel_init=nn.initializers.variance_scaling(2.0 * groups, "fan_out", "normal"),
        name=name)


def _gelu(x):
    # Torch's fused GELU evaluates the polynomial in fp32 for bf16/fp16
    # inputs. A bf16 polynomial changes its negative tail before rounding.
    compute = jnp.promote_types(x.dtype, jnp.float32)
    return jax.nn.gelu(x.astype(compute), approximate=True).astype(x.dtype)


class MobileConvNormAct(nn.Module):
    features: int
    kernel: int = 1
    stride: int = 1
    groups: int = 1
    padding: str = "same"
    bias: bool = False
    activate: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        x = _conv(self.features, self.kernel, stride=self.stride, groups=self.groups,
                  padding=self.padding, bias=self.bias, dtype=self.dtype,
                  precision=self.precision, name="conv")(x)
        # timm RmsNorm2d computes statistics in the input dtype. Gemma's text
        # RMSNorm has a different fp32-reduction contract.
        x = nn.RMSNorm(epsilon=1e-6, force_float32_reductions=False,
                       dtype=self.dtype, name="bn")(x)
        return _gelu(x) if self.activate else x


class MobileLayerScale(nn.Module):
    initial_value: float

    @nn.compact
    def __call__(self, x):
        gamma = self.param("gamma", nn.initializers.constant(self.initial_value),
                           (x.shape[-1],), jnp.float32)
        return x * gamma.astype(x.dtype)


@dataclasses.dataclass(frozen=True)
class _Block:
    kind: Literal["edge", "inverted", "attention"]
    features: int
    stride: int = 1
    expansion: int = 1
    start_kernel: int = 0
    middle_kernel: int = 0
    heads: int = 1
    head_dim: int = 64
    kv_stride: int = 1


# _gen_mobilenet_v5's mobilenetv5_300m_enc architecture. Repeated blocks own
# distinct parameters; the tuples only share their immutable descriptions.
_ARCHITECTURE = (
    (_Block("edge", 128, stride=2, expansion=4, middle_kernel=3),)
    + (_Block("edge", 128, expansion=4, middle_kernel=3),) * 2,
    (_Block("inverted", 256, stride=2, expansion=6, start_kernel=3, middle_kernel=5),)
    + tuple(_Block("inverted", 256, expansion=4, start_kernel=k) for k in (5, 3, 5, 3)),
    (_Block("inverted", 640, stride=2, expansion=6, start_kernel=5, middle_kernel=5),)
    + (_Block("inverted", 640, expansion=4, start_kernel=5),) * 7
    + (_Block("inverted", 640),)
    + (_Block("attention", 640, heads=12, head_dim=64, kv_stride=2),
       _Block("inverted", 640, expansion=2)) * 14,
    (_Block("inverted", 1280, stride=2, expansion=6, start_kernel=5, middle_kernel=5),)
    + (_Block("attention", 1280, heads=16, head_dim=96),
       _Block("inverted", 1280, expansion=2)) * 19,
)


class MobileResidual(nn.Module):
    spec: _Block
    features: int
    padding: str = "same"
    group_size: int | None = None
    layer_scale: float | None = 1e-5
    drop_path: float = 0.0
    noskip: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, train: bool = False):
        spec, shortcut = self.spec, x
        incoming = x.shape[-1]
        middle = _divisible(incoming * spec.expansion)

        def groups(channels: int) -> int:
            size = (0 if spec.kind == "edge" else 1) if self.group_size is None else self.group_size
            if not size:
                return 1
            if channels % size:
                raise ValueError(f"group_size {size} does not divide {channels} channels")
            return channels // size

        def cna(value, features, kernel, name, *, stride=1, grouped=False, activate=True):
            return MobileConvNormAct(
                features, kernel, stride=stride, groups=groups(features) if grouped else 1,
                padding=self.padding, activate=activate, dtype=self.dtype,
                precision=self.precision, name=name)(value)

        if spec.kind == "edge":
            x = _conv(middle, spec.middle_kernel, stride=spec.stride,
                      groups=groups(middle), padding=self.padding, dtype=self.dtype,
                      precision=self.precision, name="conv_exp")(x)
            x = nn.RMSNorm(epsilon=1e-6, force_float32_reductions=False,
                           dtype=self.dtype, name="bn1")(x)
            x = _gelu(x)
            x = _conv(self.features, 1, padding=self.padding, dtype=self.dtype,
                      precision=self.precision, name="conv_pwl")(x)
            x = nn.RMSNorm(epsilon=1e-6, force_float32_reductions=False,
                           dtype=self.dtype, name="bn2")(x)
        else:
            if spec.start_kernel:
                stride = 1 if spec.middle_kernel else spec.stride
                x = cna(x, incoming, spec.start_kernel, "dw_start", stride=stride,
                        grouped=True, activate=False)
            x = cna(x, middle, 1, "pw_exp")
            if spec.middle_kernel:
                x = cna(x, middle, spec.middle_kernel, "dw_mid", stride=spec.stride,
                        grouped=True)
            x = cna(x, self.features, 1, "pw_proj", activate=False)
            if self.layer_scale is not None:
                x = MobileLayerScale(self.layer_scale, name="layer_scale")(x)
        if incoming == self.features and spec.stride == 1 and not self.noskip:
            if self.drop_path:
                x = nn.Dropout(self.drop_path, broadcast_dims=(1, 2, 3),
                               name="drop_path")(x, deterministic=not train)
            x = x + shortcut
        return x


class MobileKVProjection(nn.Module):
    features: int
    stride: int = 1
    padding: str = "same"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        if self.stride > 1:
            x = _conv(x.shape[-1], 3, stride=self.stride, groups=x.shape[-1],
                      padding=self.padding, dtype=self.dtype,
                      precision=self.precision, name="down_conv")(x)
            x = nn.RMSNorm(epsilon=1e-6, force_float32_reductions=False,
                           dtype=self.dtype, name="norm")(x)
        return _conv(self.features, 1, padding=self.padding, dtype=self.dtype,
                     precision=self.precision, name="proj")(x)


class MobileMultiQueryAttention(nn.Module):
    spec: _Block
    features: int
    padding: str = "same"
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = None

    @nn.compact
    def __call__(self, x):
        batch, height, width, _ = x.shape
        spec = self.spec
        query = MobileKVProjection(spec.heads * spec.head_dim, dtype=self.dtype,
                                    precision=self.precision, name="query")(x)
        key = MobileKVProjection(spec.head_dim, stride=spec.kv_stride,
                                  padding=self.padding, dtype=self.dtype,
                                  precision=self.precision, name="key")(x)
        value = MobileKVProjection(spec.head_dim, stride=spec.kv_stride,
                                    padding=self.padding, dtype=self.dtype,
                                    precision=self.precision, name="value")(x)
        query = query.reshape(batch, height * width, spec.heads, spec.head_dim)
        key, value = (a.reshape(batch, -1, 1, spec.head_dim) for a in (key, value))
        out = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            implementation=self.attention_impl)
        out = out.reshape(batch, height, width, spec.heads * spec.head_dim)
        return MobileKVProjection(self.features, dtype=self.dtype,
                                   precision=self.precision, name="output")(out)


class MobileAttention(nn.Module):
    spec: _Block
    features: int
    padding: str = "same"
    layer_scale: float | None = 1e-5
    drop_path: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = None

    @nn.compact
    def __call__(self, x, train: bool = False):
        shortcut = x
        x = nn.RMSNorm(epsilon=1e-6, force_float32_reductions=False,
                       dtype=self.dtype, name="norm")(x)
        x = MobileMultiQueryAttention(
            self.spec, self.features, padding=self.padding, dtype=self.dtype,
            precision=self.precision, attention_impl=self.attention_impl,
            name="attn")(x)
        if self.layer_scale is not None:
            x = MobileLayerScale(self.layer_scale, name="layer_scale")(x)
        if self.drop_path:
            x = nn.Dropout(self.drop_path, broadcast_dims=(1, 2, 3),
                           name="drop_path")(x, deterministic=not train)
        return x + shortcut


class MobileStage(nn.Module):
    index: int
    multiplier: float
    padding: str
    group_size: int | None
    layer_scale: float | None
    drop_path_rate: float
    dtype: Dtype | None
    precision: PrecisionLike
    attention_impl: str | None

    @nn.compact
    def __call__(self, x, train: bool = False):
        offset = sum(len(stage) for stage in _ARCHITECTURE[:self.index])
        count = sum(len(stage) for stage in _ARCHITECTURE)
        for i, spec in enumerate(_ARCHITECTURE[self.index]):
            features = _divisible(spec.features * self.multiplier)
            drop_path = self.drop_path_rate * (offset + i) / count
            if spec.kind == "attention":
                block = MobileAttention(
                    spec, features, padding=self.padding, layer_scale=self.layer_scale,
                    drop_path=drop_path, dtype=self.dtype, precision=self.precision,
                    attention_impl=self.attention_impl, name=f"blocks_{i}")
            else:
                block = MobileResidual(
                    spec, features, group_size=self.group_size, padding=self.padding,
                    layer_scale=self.layer_scale, drop_path=drop_path, dtype=self.dtype,
                    precision=self.precision, name=f"blocks_{i}")
            x = block(x, train=train)
        return x


def _nearest(x, height: int, width: int):
    # PyTorch's 'nearest' uses floor(dst * src / dst_size), not the
    # half-pixel coordinates of JAX's nearest resize.
    rows = jnp.arange(height) * x.shape[1] // height
    columns = jnp.arange(width) * x.shape[2] // width
    return jnp.take(jnp.take(x, rows, axis=1), columns, axis=2)


class MobileMultiScaleFusion(nn.Module):
    resolution: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, inputs: tuple[jax.Array, ...]):
        height, width = inputs[0].shape[1:3]
        resized = tuple(_nearest(x, height, width) if x.shape[1:3] != (height, width)
                        else x for x in inputs)
        x = jnp.concatenate(resized, axis=-1)
        x = MobileResidual(_Block("inverted", 2048, expansion=2), 2048,
                           layer_scale=None, noskip=True, dtype=self.dtype,
                           precision=self.precision, name="ffn")(x)
        size = self.resolution
        if (height, width) != (size, size):
            if height % size or width % size:
                x = jax.image.resize(x, (x.shape[0], size, size, x.shape[-1]),
                                     method="linear", antialias=False)
            else:
                strides = (height // size, width // size)
                x = nn.avg_pool(x, strides, strides=strides, padding="VALID")
        return nn.RMSNorm(epsilon=1e-6, force_float32_reductions=False,
                          dtype=self.dtype, name="norm")(x)


class MobileNetV5Encoder(nn.Module):
    """The mobilenetv5_300m_enc graph, with timm's construction controls.

    Strict GPU fp32 reference comparisons use precision=HIGHEST. The bf16
    path executes with its own rounding; it is not qualified at the fp32
    reference tolerance.
    """

    channel_multiplier: float = 1.0
    stem_size: int = 64
    stem_bias: bool = True
    fix_stem: bool | None = None
    in_chans: int = 3
    pad_type: str = "same"
    group_size: int | None = None
    msfa_indices: tuple[int, ...] = (-2, -1)
    msfa_output_resolution: int = 16
    layer_scale_init_value: float | None = 1e-5
    drop_path_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str | None = None

    def setup(self):
        if self.channel_multiplier <= 0 or self.stem_size < 1 or self.in_chans < 1:
            raise ValueError("channel_multiplier, stem_size and in_chans must be positive")
        if self.group_size is not None and self.group_size < 0:
            raise ValueError("group_size is nonnegative; 0 selects an ungrouped convolution")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if self.msfa_output_resolution < 1:
            raise ValueError("msfa_output_resolution must be positive")
        indices = tuple(index + 5 if index < 0 else index for index in self.msfa_indices)
        if not indices or len(indices) != len(set(indices)) or any(not 0 <= i < 5 for i in indices):
            raise ValueError("msfa_indices must select distinct features among stem and four stages")
        self.selected = frozenset(indices)
        _padding(self.pad_type, 3)

    @nn.compact
    def __call__(self, pixel_values, train: bool = False):
        pixels = jnp.asarray(pixel_values)
        if (pixels.ndim != 4 or pixels.shape[1] != self.in_chans
                or min(pixels.shape) < 1):
            raise ValueError(f"pixel_values must be nonempty [B, {self.in_chans}, H, W], got {pixels.shape}")
        if not jnp.issubdtype(pixels.dtype, jnp.floating):
            raise ValueError(f"pixel_values must be floating processor output, got {pixels.dtype}")
        x = jnp.moveaxis(pixels, 1, -1)
        fixed = self.channel_multiplier < 1.0 if self.fix_stem is None else self.fix_stem
        stem = self.stem_size if fixed else _divisible(self.stem_size * self.channel_multiplier)
        x = MobileConvNormAct(stem, 3, stride=2, padding=self.pad_type,
                              bias=self.stem_bias, dtype=self.dtype,
                              precision=self.precision, name="conv_stem")(x)
        features = [x] if 0 in self.selected else []
        for stage in range(4):
            x = MobileStage(stage, self.channel_multiplier, self.pad_type,
                            self.group_size, self.layer_scale_init_value,
                            self.drop_path_rate, self.dtype, self.precision,
                            self.attention_impl, name=f"stages_{stage}")(x, train=train)
            if stage + 1 in self.selected:
                features.append(x)
        x = MobileMultiScaleFusion(self.msfa_output_resolution, dtype=self.dtype,
                                   precision=self.precision, name="msfa")(tuple(features))
        return x.reshape(x.shape[0], -1, x.shape[-1])
