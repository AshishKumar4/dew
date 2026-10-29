"""The convolutional and embedding pieces the UNets and the DiT sandwich share."""

import functools
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.typing import DTypeLike

from dew.nn.dense import Dense, DenseGeneral
from dew.nn.dropout import Dropout

from . import activations
from .conv import Conv
from .flops import projection_flops
from .kernels.generation import linear_tangent
from .norms import GroupNorm, RMSNorm
from .precision import at_least_fp32, float32_dot_general, float32_einsum
from .rope import sinusoid_frequencies, sinusoids
from .sharding import constrain, explicit_spec, rows_taken, whole_spec


def normal_kernel(std: float | None, default: Callable | None = None) -> dict:
    """The `kernel_init` keyword for a normal draw of `std`, as lm-engine
    initialises its linears (`nn.init.normal_`, init_utils.py at 45b6b57b).

    None keeps `default`, or the module's own initializer when that is None
    too, so a model that states no std builds exactly what it built before.
    """
    if std is not None:
        return {"kernel_init": nn.initializers.normal(std)}
    return {} if default is None else {"kernel_init": default}


def _table_rows(table: jax.Array, ids: jax.Array, dtype: Dtype) -> jax.Array:
    """`table[ids]` in `dtype`, the table's rows gathered before the cast so
    the gradient accumulates in the table's dtype: casting the table first
    would scatter-add repeated tokens' cotangents in bf16. Under an Explicit
    axis splitting the ids, the rows are placed as the ids are (`rows_taken`)."""
    return rows_taken(table, ids).astype(dtype)


def table_rows(table: jax.Array, ids: jax.Array, dtype: Dtype) -> jax.Array:
    """`table[ids]` in `dtype` (`_table_rows`) differentiated at every order (`generation.linear_tangent`):
    reverse mode adds each token's cotangent to its row in the table's dtype (`_table_rows_backward`), forward
    mode (`jax.jvp`, `jax.linearize`) takes the gather's own tangent, which a custom VJP alone refused (row
    1869, audit NN-06); the ids are not differentiable."""
    return _rows(jnp.dtype(dtype))(table, ids)


@functools.cache
def _rows(dtype: jnp.dtype) -> Callable:
    """The lookup in `dtype` as one function of the table and the ids (`table_rows`)."""
    def forward(table, ids):
        return _table_rows(table, ids, dtype), (table, ids)

    return linear_tangent(forward, _table_rows_backward, lambda table, ids: _table_rows(table, ids, dtype))


def _table_rows_backward(saved: tuple[jax.Array, jax.Array], operands: tuple, cotangent: jax.Array
                         ) -> tuple:
    del operands
    table, ids = saved
    # The gradient adds each token's cotangent to its row. Under a batch
    # split GSPMD scattered each device's tokens into a table-sized buffer
    # and summed the buffers across devices: a table's worth of traffic
    # decided by the rows. Where the rows are the smaller, every device
    # gathers all of them, in the compute dtype, and scatters them itself,
    # into its copy of the table or its shard of the vocabulary, and the
    # gradient needs no sum. An Explicit axis splitting the rows is in their
    # type, which a constraint leaves alone, so they are resharded whole.
    if cotangent.size * cotangent.dtype.itemsize < table.size * table.dtype.itemsize:
        cotangent, ids = _whole(cotangent), _whole(ids)
    rows = cotangent.astype(table.dtype)
    placed = explicit_spec(ids)
    # The ids are integers: their cotangent is float0's zero, as the lookup's own pullback gives it.
    untouched = np.zeros(ids.shape, jax.dtypes.float0)
    if placed is None:
        return jnp.zeros_like(table).at[ids].add(rows), untouched
    # Each device scatters its own rows and the sums are added across the axis.
    summed = jax.sharding.NamedSharding(placed.mesh, jax.typeof(table).sharding.spec)
    return jnp.zeros_like(table).at[ids].add(rows, out_sharding=summed), untouched


def _whole(x: jax.Array) -> jax.Array:
    """`x` on every device: resharded where an Explicit axis splits it, and
    constrained over the Auto axes."""
    placed = whole_spec(x, x.ndim)
    return constrain(x if placed is None else jax.sharding.reshard(x, placed), (None,) * x.ndim)



def modulation(conditioning, features: int, pieces: int, *, name: str, silu: bool = True,
               bias: bool = True, zero_init: bool = False, dtype: Dtype | None = None,
               precision: PrecisionLike = None) -> jax.Array:
    """The adaLN projection of `conditioning`, `[..., pieces * features]`: SiLU, then one linear map,
    named `name` in the calling module so each family keeps its checkpoint's path. `silu` False
    projects the conditioning as it is (FlaxDiff 0.2's DiT); `zero_init` starts it at zero,
    adaLN-Zero's identity block at the start of training."""
    return Dense(pieces * features, use_bias=bias, dtype=dtype, precision=precision, name=name,
                 kernel_init=nn.initializers.zeros if zero_init else nn.initializers.lecun_normal(),
                 dot_general=float32_dot_general)(
        activations.silu(conditioning) if silu else conditioning)


class Modulation(nn.Module):
    """Projects the conditioning as the source's `AdaLayerNormZero` family does: SiLU, then one linear
    layer (`modulation`, named `linear`).

    The output is `pieces` chunks of the width, in the source's order.
    """

    features: int
    pieces: int
    bias: bool = True
    zero_init: bool = False
    """Whether the projection starts at zero, adaLN-Zero's identity block, which
    Dew's own MM-DiT trains from; the published families load their weights."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @property
    def output_features(self) -> int:
        return self.pieces * self.features

    @nn.nowrap
    def model_flops(self, lengths, input_features: int | None = None) -> float:
        return projection_flops(lengths, self.features if input_features is None else input_features,
                                self.output_features)

    @nn.compact
    def __call__(self, conditioning):
        projected = modulation(conditioning, self.features, self.pieces, name="linear", bias=self.bias,
                               zero_init=self.zero_init, dtype=self.dtype, precision=self.precision)
        return jnp.split(projected, self.pieces, axis=-1)


def padded_row(padding_idx: int | None, rows: int) -> int | None:
    """The row of a table of `rows` that `padding_idx` names, as torch's `nn.Embedding` reads it: a
    negative id counts from the table's end, one outside the table is refused, and None names none."""
    if padding_idx is None:
        return None
    if not -rows <= padding_idx < rows:
        raise ValueError(f"padding_idx {padding_idx} is outside the table of {rows} rows")
    return padding_idx % rows


class TokenEmbedding(nn.Embed):
    """Token lookup with cotangent accumulation in the parameter dtype."""

    padding_idx: int | None = None
    """The id whose row this lookup reads but never trains, as torch's `nn.Embedding(padding_idx=...)`
    leaves it: the row's gradient takes nothing from the places that read that id. A negative id
    counts from the table's end, as torch counts it; one outside the table is refused. None trains
    every row it reads."""

    def __call__(self, inputs: jax.Array) -> jax.Array:
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        if self.num_embeddings == 1:
            values = jnp.broadcast_to(self.embedding, (*inputs.shape, self.features))
            rows, = self.promote_dtype(values, dtype=self.dtype, inexact=False)
            if rows is None:
                raise ValueError("Embedding dtype promotion must return an array")
        else:
            rows = table_rows(self.embedding, inputs, self.dtype or self.embedding.dtype)
        padding = padded_row(self.padding_idx, self.num_embeddings)
        if padding is None:
            return rows
        return jnp.where((inputs == padding)[..., None], jax.lax.stop_gradient(rows), rows)


class FourierEmbedding(nn.Module):
    """Random Fourier features of a scalar per example: `[B]` to `[B, features]`,
    sines then cosines of the input against fixed Gaussian frequencies.

    The frequencies, already multiplied by `scale`, are the `constants`
    variable `frequencies`, so a checkpoint carries the table its weights
    learned against. `init` draws it from numpy's RandomState(42), which gives
    the same table on every jax version. A checkpoint converted from
    elsewhere brings its own: FlaxDiff 0.2 (commit 3e3497e, the code flaxdiff
    0.2.8 shipped) drew its with `jax.random.normal`, whose stream changed in
    jax 0.5.0, and FlaxDiff's main branch draws numpy's, as Dew does (commit
    63f2427).
    """
    features: int
    scale: float = 16
    dtype: Dtype | None = None
    """The model's compute dtype; the features are computed in
    `at_least_fp32` of it."""

    def setup(self):
        self.frequencies = self.variable(
            "constants", "frequencies",
            lambda: jnp.asarray(np.random.RandomState(42).normal(size=(self.features // 2,)),
                                dtype=jnp.float32) * self.scale)

    def __call__(self, x):
        x = jax.lax.convert_element_type(x, at_least_fp32(self.dtype))
        # 2 pi times the table is one rounded vector, as it was when the table
        # was a trace-time constant. Without the barrier XLA folds the 2 pi
        # into whatever scalar the caller scaled `x` by (EDM's 1/4) and the
        # sin/cos arguments round differently.
        angular = jax.lax.optimization_barrier(2 * jnp.pi * self.frequencies.value)
        return sinusoids(x[:, None] * angular[None, :])


def sinusoidal_time(time, features: int, *, dtype: DTypeLike, shift: float = 0, cosine_first: bool = True):
    """Embed a scalar timestep as `features` sinusoids: `[B]` to `[B, features]`,
    computed in `dtype` (`at_least_fp32` of the model's).

    `cosine_first` puts the cosines in the leading half, as SD1/2 and SDXL
    store them. `shift` moves the lowest frequency, the reference's
    `freq_shift`.
    """
    half = features // 2
    if features % 2 or half <= shift:
        raise ValueError("Time embedding width must be even and exceed twice the frequency shift")
    frequencies = jnp.asarray(sinusoid_frequencies(half, half - shift, dtype=dtype, exponential=True))
    return sinusoids(jnp.asarray(time, dtype).reshape(-1, 1) * frequencies[None], cosine_first=cosine_first)


class TimeProjection(nn.Module):
    """Two dense layers with the activation after each."""
    features: int
    activation: Callable = activations.gelu_tanh
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def _mlp(self, input_features: int):
        from .mlp import GatedMLP  # mlp reads this module's initializers

        return GatedMLP(self.features, self.features, in_features=input_features,
                        activation=self.activation, activation_after=True, use_bias=True,
                        names=("DenseGeneral_0", "DenseGeneral_1"), dtype=self.dtype,
                        precision=self.precision)

    @nn.nowrap
    def model_flops(self, lengths, input_features: int) -> float:
        return self._mlp(input_features).model_flops(lengths)

    @nn.compact
    def __call__(self, x):
        mlp = self._mlp(x.shape[-1])
        nn.share_scope(self, mlp)
        return mlp(x)


def torch_nearest_resize(x, height: int, width: int):
    """Resample `[B, H, W, C]` to `height` by `width`, torch's nearest rule.

    PyTorch's 'nearest' reads source index `floor(dst * src / dst_size)`,
    not the half-pixel coordinates of `jax.image.resize`, so a converted
    checkpoint only reproduces its reference through this arithmetic.
    """
    rows = jnp.arange(height) * x.shape[1] // height
    columns = jnp.arange(width) * x.shape[2] // width
    return jnp.take(jnp.take(x, rows, axis=1), columns, axis=2)


def _torch_bicubic_weights(size_in: int, size_out: int, antialias: bool) -> np.ndarray:
    """The `[size_out, size_in]` matrix of torch's bicubic interpolation along
    one axis, `align_corners=False`.

    Without antialiasing each output reads four taps at half-pixel source
    coordinates, clamped at the border, with the cubic's a = -0.75; the
    coordinate is computed in float32 as torch's CPU kernel computes it, since
    at 256 pixels its rounding moves a weight by 1e-5. With antialiasing the
    kernel is a = -0.5, widened by the scale when shrinking, and normalized."""
    scale = np.float32(size_in) / np.float32(size_out)
    matrix = np.zeros((size_out, size_in), np.float64)
    if antialias:
        stretch = max(float(scale), 1.0)
        for i in range(size_out):
            center = float(scale) * (i + 0.5)
            low, high = max(int(center - 2 * stretch + 0.5), 0), min(int(center + 2 * stretch + 0.5), size_in)
            x = np.abs((np.arange(low, high) - center + 0.5) / stretch)
            weights = np.where(
                x < 1, (1.5 * x - 2.5) * x * x + 1, np.where(x < 2, ((-0.5 * x + 2.5) * x - 4) * x + 2, 0)
            )
            matrix[i, low:high] = weights / weights.sum()
        return matrix.astype(np.float32)
    a, one = np.float32(-0.75), np.float32(1)

    def near(x):
        return ((a + 2) * x - (a + 3)) * x * x + one

    def far(x):
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

    for i in range(size_out):
        source = scale * (np.float32(i) + np.float32(0.5)) - np.float32(0.5)
        index = int(np.floor(source))
        t = np.float32(source - np.float32(index))
        for tap, weight in zip(
            range(index - 1, index + 3), (far(t + one), near(t), near(one - t), far(2 * one - t)), strict=True
        ):
            matrix[i, min(max(tap, 0), size_in - 1)] += weight
    return matrix.astype(np.float32)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
"""The ImageNet pixel statistics, per RGB channel, that encoders trained on ImageNet read their
inputs under: transformers' `IMAGENET_DEFAULT_MEAN` and `IMAGENET_DEFAULT_STD`, written out since
importing those runs `transformers.utils`, which imports torch where it is installed (transformers
5.16.1, utils/generic.py), and image sampling imports this module with neither
(tests/test_imports.py)."""


def torch_bicubic_resize(x, height: int, width: int, *, antialias: bool = False):
    """Resample `[B, H, W, C]` to `height` by `width` as torch's
    `F.interpolate(mode="bicubic", align_corners=False)` does.

    `jax.image.resize`'s cubic is Keys' a = -0.5 and always antialiases a
    shrink; torch's is a = -0.75 and antialiases only when asked, so a vision
    encoder that resizes its input or its position table reproduces its
    reference only through this arithmetic."""
    rows = _torch_bicubic_weights(x.shape[1], height, antialias)
    columns = _torch_bicubic_weights(x.shape[2], width, antialias)
    return float32_einsum("oh,bhwc,pw->bopc", rows, x, columns, precision=jax.lax.Precision.HIGHEST)


class Upsample(nn.Module):
    """Nearest-neighbour upsampling by `scale`, then a 3x3 convolution to `features`, a module named `conv`
    (its parameters' path, as the model owning them publishes it: a UNet's `Conv_0`, a KL autoencoder's
    `conv`)."""
    features: int
    scale: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    conv: str = "Conv_0"

    @nn.nowrap
    def _convolution(self, **scope):
        return Conv(self.features, (3, 3), dtype=self.dtype, precision=self.precision,
                    **({"name": self.conv} if "parent" not in scope else {}), **scope)

    @nn.nowrap
    def model_flops(self, lengths, input_features: int) -> float:
        return self._convolution(parent=None).model_flops(lengths, input_features)

    @nn.nowrap
    def spatial_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        enlarged = (*tuple(size * self.scale for size in shape[:-1]), shape[-1])
        return self._convolution(parent=None).spatial_shape(enlarged)

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        out = jax.image.resize(x, (B, H * self.scale, W * self.scale, C), method="nearest")
        return self._convolution()(out)


class Downsample(nn.Module):
    """A stride-2 3x3 convolution to `features`."""
    features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def _convolution(self, **scope):
        return Conv(self.features, (3, 3), strides=(2, 2), dtype=self.dtype,
                    precision=self.precision, **scope)

    @nn.nowrap
    def model_flops(self, lengths, input_features: int) -> float:
        return self._convolution(parent=None).model_flops(lengths, input_features)

    @nn.nowrap
    def spatial_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return self._convolution(parent=None).spatial_shape(shape)

    @nn.compact
    def __call__(self, x):
        return self._convolution()(x)


class ResidualBlock(nn.Module):
    """Norm, activation, convolution, the projected time embedding added
    where the call gives one (an AutoencoderKL's blocks take none), norm,
    activation, convolution, plus the input (through a 1x1 convolution when
    the width changes). `norm_groups` of 0 swaps the group norms for RMS
    norms."""
    features: int
    kernel_size: tuple = (3, 3)
    activation: Callable = jax.nn.swish
    norm_groups: int = 8
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-4
    dropout: float = 0.0
    shortcut: str = "residual_conv"
    """The 1x1 convolution's name: Dew's UNets' `residual_conv`, or an AutoencoderKL's
    `conv_shortcut`, under which diffusers' Flax checkpoints load and are written as they are."""

    @nn.nowrap
    def spatial_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return Conv(self.features, self.kernel_size, parent=None).spatial_shape(shape)

    @nn.nowrap
    def model_flops(self, lengths, input_features: int, time_features: int) -> float:
        """The declared convolutions, one time map per row and any learned skip."""
        conv = Conv(self.features, self.kernel_size, parent=None)
        count = conv.model_flops(lengths, input_features) + conv.model_flops(lengths, self.features)
        count += projection_flops(np.ones(np.shape(lengths), np.int64), time_features, self.features)
        if input_features != self.features:
            count += Conv(self.features, (1, 1), parent=None).model_flops(lengths, input_features)
        return count

    def setup(self):
        if self.norm_groups > 0:
            norm = partial(GroupNorm, self.norm_groups, epsilon=self.norm_epsilon, activation=self.activation,
                           dtype=self.dtype)
        else:
            norm = partial(RMSNorm, epsilon=self.norm_epsilon, activation=self.activation, dtype=self.dtype)
        self.norm1 = norm()
        self.norm2 = norm()

    @nn.compact
    def __call__(self, x: jax.Array, temb: jax.Array | None = None, *, train: bool = False):
        conv = partial(Conv, features=self.features, kernel_size=self.kernel_size,
                       strides=(1, 1), dtype=self.dtype, precision=self.precision)
        out = conv(name="conv1")(self.norm1(x))

        if temb is not None:
            temb = DenseGeneral(features=self.features, name="temb_projection",
                                dtype=self.dtype, precision=self.precision,
                                dot_general=float32_dot_general)(temb)
            out = out + temb[:, None, None, :]

        out = self.norm2(out)
        if self.dropout:
            out = Dropout(self.dropout)(out, deterministic=not train)
        out = conv(name="conv2")(out)

        residual = x
        if residual.shape != out.shape:
            residual = conv(kernel_size=(1, 1), name=self.shortcut)(residual)
        return out + residual
