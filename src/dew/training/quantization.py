"""Quantized training through Qwix, applied to the model before it trains.

Qwix (google/qwix, Apache 2.0) describes quantization as rules over module
paths and applies them without editing the model. One call wraps the module,
and the matmuls that run inside the wrapped methods run quantized.

In Dew that call is `Quantization.apply`. A caller builds its model as
usual, then wraps it before the objective ever sees it. A run that sets
`--trainer.quantization` passes the objective to `RunConfig.train` instead,
and the model the objective holds is wrapped there, before anything
initialises it.

Training is fake-quantized. The parameter tree keeps fp32 master weights
with the same structure, so the checkpoint layout, the sharding derivation,
the Muon parameter split and Hugging Face loading are unchanged. Only the
forward and backward matmuls are quantized, with a straight-through
estimator on the backward pass.

Serving stores the weights quantized. `quantize_for_serving` runs Qwix's
post-training quantization: the returned variables hold each matched kernel
as int8 or fp8 values with their scales, and the returned module's matmuls
read them. `TextGeneration.quantized` applies it to a language model and
`TextToImage.quantized` to an image task's denoiser.

The vocabulary head stays fp32, like the rest of Dew's fp32 zones. Its
einsum runs in the objective's chunked cross entropy, outside any model
method Qwix wraps.

The `Quantization` fields follow MaxText's quantization settings where Qwix
has an equivalent. `dtype` is MaxText's `quantization` for the dynamic-range
forms, and `patterns` is its `quant_cfg_path`, written inline as the regexes
Qwix matches. The backward fields are Qwix's finer-grained version of the
same idea.

Three MaxText settings have no equivalent, and `dtype` accepts neither of
the first two. Static activation scaling (`fp8_full`) needs a calibration
pass that Dew has no hook for, `nanoo_fp8` uses AMD-only kernels, and
KV-cache quantization has nothing to read it here because the cache holds
the compute dtype.

Qwix comes with the `quantization` extra (`pip install "dewml[quantization]"`).
It is imported inside the calls that need it, and without the package those
calls raise an error that names the extra.
"""
from __future__ import annotations

import dataclasses
import functools
import importlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import core, linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

if TYPE_CHECKING:
    from dew.diffusion.process import Conditioning
    from dew.objectives.base import Objective, Variables

QuantizedDtype = Literal["int8", "fp8"]
"""The gemm dtypes a run trains with: int8 on any backend, fp8 where the
backend lowers it (measured in docs/performance.md)."""

Rounding = Literal["uniform", "low_bit_uniform"]
"""How a quantized gradient rounds: Qwix's two stochastic modes."""

CALIBRATIONS = ("absmax", "minmax", "rms", "fixed")
"""The weight calibration methods Qwix parses, before an optional `,args`
suffix (`qwix/_src/qconfig.py`, `QuantizationRule`)."""

# The entry methods a run's matmuls travel through, wrapped where the model
# defines them. `__call__` covers sampling and scoring; the language-model
# objective trains through `hidden_states` and its prediction depths through
# `mtp_hidden_states`; text generation (`dew.sampling.text`) enters through
# the rest. Qwix's interception is non-recursive over dynamic extent, so
# `__call__` reaching `hidden_states` quantizes once.
METHODS = ("__call__", "hidden_states", "mtp_hidden_states", "states_and_logits", "states_and_logits_at",
           "init_cache", "init_mtp_cache", "init_draft_cache", "mtp_step", "token_embeddings", "draft",
           "draft_context")


# MaxText's quantization settings, which these fields follow, are in its
# configs/base.yml:128-167.
@dataclasses.dataclass(frozen=True)
class Quantization:
    """Says how a run quantizes its trunk matmuls, for Qwix's provider."""

    dtype: QuantizedDtype = "int8"
    """The dtype weights and activations quantize to, in the forward pass."""
    weight_only: bool = False
    """Whether to quantize only the weights and keep activations in the compute dtype.

    Convolutions stay unquantized when this is set, because Qwix's serving
    provider refuses a convolution whose activations stay in float."""
    patterns: tuple[str, ...] = (".*",)
    """Regexes over module paths that select what is quantized, in Qwix's precedence order.

    The first rule whose regex fully matches a module's `/`-joined scope path
    wins. `'.*mlp.*'` quantizes the feed-forward blocks and leaves attention
    in fp32; the default quantizes every matmul of the wrapped methods."""
    calibration: str = "absmax"
    """The weight calibration method, in the form Qwix parses.

    It is a method name, one of `absmax`, `minmax`, `rms` or `fixed`, with an
    optional `,args` suffix, for example `absmax,0.8`."""
    tile_size: int | None = None
    """The number of elements per tile for sub-channel scales along the contraction axis.

    Unset keeps per-channel scales, which are coarser and cheaper."""
    bwd_qtype: QuantizedDtype | None = None
    """The dtype gradients quantize to in the backward pass; unset keeps
    them in the compute dtype."""
    # Qwix draws the stream in qwix/_src/providers/qt.py:361.
    bwd_stochastic_rounding: Rounding | None = None
    """The stochastic rounding mode for the quantized gradients; unset rounds deterministically.

    A run that sets it passes a `stochastic_rounding` RNG stream at apply
    time, and Qwix draws from that stream."""

    def __post_init__(self) -> None:
        if self.dtype not in ("int8", "fp8"):
            raise ValueError(
                f"quantization trains in int8 or fp8, got {self.dtype!r}")
        if not self.patterns:
            raise ValueError(
                "quantization with no patterns quantizes nothing; pass "
                "('.*',) for the whole trunk")
        for pattern in self.patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"quantization pattern {pattern!r} does not compile: "
                    f"{error}") from error
        method = self.calibration.split(",", 1)[0]
        if method not in CALIBRATIONS:
            raise ValueError(
                f"calibration is one of {list(CALIBRATIONS)}, with an optional "
                f"`,args` suffix, got {self.calibration!r}")
        if self.tile_size is not None and self.tile_size < 1:
            raise ValueError(
                f"tile_size counts elements per tile, got {self.tile_size}")
        if self.bwd_qtype is not None and self.bwd_qtype not in ("int8", "fp8"):
            raise ValueError(
                f"bwd_qtype is int8, fp8 or unset, got {self.bwd_qtype!r}")
        if (self.bwd_stochastic_rounding is not None
                and self.bwd_stochastic_rounding not in ("uniform", "low_bit_uniform")):
            raise ValueError(
                "bwd_stochastic_rounding is 'uniform', 'low_bit_uniform' or "
                f"unset, got {self.bwd_stochastic_rounding!r}")

    def apply(self, model: nn.Module) -> nn.Module:
        """Return a copy of `model` whose trunk matmuls train in this spec's dtype.

        The copy is an instance of Qwix's subclass of the model's class, with
        each method of `METHODS` that the model defines wrapped. Everything
        the registry, the objective and the checkpoint code read from the
        model therefore still works. Invalid settings were already refused
        when the spec was constructed. Without Qwix installed, the call raises
        an error that names the extra.
        """
        rules = _rules(self, training=True)
        methods = tuple(method for method in METHODS if hasattr(model, method))
        wrapped = _qwix().quantize_model(model, _providers()[0](rules), methods=methods)
        # Qwix makes a class per call; it carries the spec, which a model record writes.
        type(wrapped)._dew_quantization = self
        return wrapped


def _qwix(module: str = "qwix") -> ModuleType:
    """Qwix's `module`, imported when a call needs it; without Qwix, an error
    that names the extra which installs it."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name != "qwix":
            raise
        raise ModuleNotFoundError(
            'quantization runs on Qwix; install it with pip install "dewml[quantization]"', name="qwix"
        ) from error


@dataclasses.dataclass(frozen=True)
class NVFP4Input:
    """One checkpoint Linear's stored global scale and published local-scale arithmetic."""

    global_scale: float
    e4m3_scale: bool
    format: Literal['compressed-tensors', 'modelopt'] = 'compressed-tensors'

    def __post_init__(self) -> None:
        if (not math.isfinite(self.global_scale) or self.global_scale < 0
                or (self.global_scale == 0 and self.format == 'compressed-tensors')):
            raise ValueError(f"NVFP4 input_global_scale must be positive and finite, got {self.global_scale}")
        if self.format == 'modelopt' and not self.e4m3_scale:
            raise ValueError("ModelOpt NVFP4 always rounds its local scales to E4M3")


@dataclasses.dataclass(frozen=True)
class FP8Input:
    """One ModelOpt static E4M3 input multiplier, amax/448 in its exported files."""

    scale: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"ModelOpt FP8 input_scale must be positive and finite, got {self.scale}")


def fp8_input_qdq(x: jax.Array, spec: FP8Input) -> jax.Array:
    """ModelOpt's static FP8 input QDQ using Qwix's E4M3 codes and stored scale.

    tensor_quant.py:_fp8_eager widens before scaling and returns the input
    dtype. The reference reconstructs amax from the file's amax/448,
    computes 448/amax and its reciprocal in fp32, casts directly to E4M3
    RN, and dequantizes by that reciprocal. Keeping both divisions holds
    the scale's last bits to the author's fake quantizer. Backward is Dew's STE.
    """
    from dew.nn.fake_quant import straight_through

    qarray = _qwix("qwix._src.core.qarray")
    values = jax.lax.stop_gradient(x).astype(jnp.float32)
    amax = jnp.float32(spec.scale) * jnp.float32(448)
    safe = jnp.where(amax <= jnp.float32(2 ** -24), 1, amax)
    # torch's scalar/tensor reverse divide is reciprocal then multiply.
    multiplier = _nvfp4_divide(jnp.float32(1), safe) * jnp.float32(448)
    scale = _nvfp4_divide(jnp.float32(1), multiplier)
    # The composed GPU forward must round this product before the FP8 cast.
    quotients = jax.lax.optimization_barrier(values * multiplier)
    broadcast_shape = (1,) * values.ndim
    quantized = qarray.quantize_with_scale_zero_point(
        quotients, jnp.float8_e4m3fn, jnp.ones(broadcast_shape, jnp.float32), None)
    quantized = quantized.replace(scale=scale.reshape(broadcast_shape))
    rounded = qarray.dequantize(quantized)
    return straight_through(x, rounded)


def _nvfp4_divide(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """CT's correctly rounded fp32 quotient, including on XLA's approximate GPU divider.

    A Dekker TwoProduct gives the exact residual of the approximate
    quotient, split into two fp32 values. One residual correction puts the
    quotient within one ulp; comparing its exact residual with each
    neighbour's half-spacing times the divisor chooses nearest-even.
    That threshold product is exact because the half-spacing is a power
    of two. The quantizer's observed scales and quotients keep these
    intermediates normal; dense and midpoint checks hold the result to
    NumPy's IEEE divide without enabling x64. Dense keeps its own precision.
    """
    def product_error(a, b, product):
        ah = jax.lax.bitcast_convert_type(
            jax.lax.bitcast_convert_type(a, jnp.uint32) & jnp.uint32(0xfffff000), jnp.float32)
        bh = jax.lax.bitcast_convert_type(
            jax.lax.bitcast_convert_type(b, jnp.uint32) & jnp.uint32(0xfffff000), jnp.float32)
        al, bl = a - ah, b - bh
        remainder = product - ah * bh
        remainder = remainder - al * bh
        remainder = remainder - ah * bl
        return al * bl - remainder

    quotient = numerator / denominator
    product = quotient * denominator
    error = product_error(quotient, denominator, product)
    residual = (numerator - product) - error
    quotient = quotient + residual / denominator
    product = quotient * denominator
    error = product_error(quotient, denominator, product)
    difference = numerator - product
    high = difference - error
    recovered = high - difference
    low = (difference - (high - recovered)) - (error + recovered)
    above, below = jnp.nextafter(quotient, jnp.inf), jnp.nextafter(quotient, -jnp.inf)
    upper = (above - quotient) * 0.5 * denominator
    lower = (quotient - below) * 0.5 * denominator
    odd = (jax.lax.bitcast_convert_type(quotient, jnp.uint32) & 1) != 0
    up = (high > upper) | ((high == upper) & ((low > 0) | ((low == 0) & odd)))
    down = (high < -lower) | ((high == -lower) & ((low < 0) | ((low == 0) & odd)))
    return jnp.where(up, above, jnp.where(down, below, quotient))


def nvfp4_input_qdq(x: jax.Array, spec: NVFP4Input) -> jax.Array:
    """CT 0.17.1's local input QDQ with Qwix's E2M1 QArray representation.

    A local amax/6 is computed in the input dtype, multiplied by the stored
    fp32 inverse global scale, and rounded to E4M3 only when declared. CT
    replaces zero scales by that scale dtype's eps, divides by the global
    scale in fp32, then QDQs the fp32 quotient and returns the input dtype.
    Qwix's automatic NVFP4 calibration has no checkpoint global scale and
    always rounds local scales; quantize_with_scale_zero_point takes CT's
    effective scales instead. The original Dense/dot retains its precision
    policy. RedHatAI/Qwen3-32B-NVFP4 declares unrounded local scales;
    sakamakismile/Qwen3.8-27B-MTP-NVFP4 at a0b936f0bbcb362c38d39840602c8d7b2476a9fc
    declares torch.float8_e4m3fn scales.

    CT 0.17.1's bare fake_quantize runs under torch.no_grad, so its input
    QDQ detaches the input and torch fine-tuning passes no gradient through
    it unless another QAT wrapper (such as llm-compressor's) supplies one.
    Dew deliberately uses the identity's straight-through gradient; the
    reference parity claim here covers the forward, not that backward.
    """
    from dew.nn.fake_quant import straight_through
    from dew.nn.precision import rounded_operand

    qarray = _qwix("qwix._src.core.qarray")
    if x.shape[-1] % 16:
        raise ValueError(f"NVFP4 input width must be divisible by 16, got {x.shape}")
    values = jax.lax.stop_gradient(x)
    groups = values.reshape(*x.shape[:-1], -1, 16)
    if spec.format == 'modelopt':
        return _modelopt_input_qdq(x, spec, qarray)
    largest = jnp.max(jnp.abs(groups), -1)
    divisor = jnp.full(largest.shape, 6, jnp.float32)
    # CT's local division must round before global scaling, including its bf16 cast.
    scale = jnp.asarray(
        rounded_operand(_nvfp4_divide(largest.astype(jnp.float32), divisor), x.dtype), jnp.float32)
    overall = jnp.full(scale.shape, spec.global_scale, jnp.float32)
    scale = scale * overall
    if spec.e4m3_scale:
        scale = jnp.clip(scale, 0, 448).astype(jnp.float8_e4m3fn).astype(jnp.float32)
    eps = jnp.finfo(jnp.float8_e4m3fn if spec.e4m3_scale else jnp.float32).eps
    scale = jnp.where(scale == 0, jnp.asarray(eps, jnp.float32), scale)
    scale = _nvfp4_divide(scale, overall)
    expanded = jnp.broadcast_to(scale[..., None], groups.shape).reshape(x.shape)
    quotients = _nvfp4_divide(values.astype(jnp.float32), expanded)
    quantized = qarray.quantize_with_scale_zero_point(quotients, 'nvfp4', jnp.ones_like(expanded), None)
    quantized = quantized.replace(scale=expanded)
    rounded = qarray.dequantize(quantized)
    # CT adds its symmetric zero point before casting, so exact -0 is +0.
    rounded = jnp.where(values == 0, jnp.zeros_like(rounded), rounded)
    return straight_through(x, rounded)


def _modelopt_input_qdq(x: jax.Array, spec: NVFP4Input, qarray: ModuleType) -> jax.Array:
    """ModelOpt's direct-RN E4M3 scale rule, with Qwix's E2M1 representation.

    fp4_kernel_hopper.py:76-99 widens before amax, rounds amax/(6*g) to
    E4M3, multiplies by stored g, and replaces an effective scale below
    1e-5 by 1. NVFP4QTensor's torch scale cast and TensorRT-LLM's
    quantization.cuh:501 use direct RN E4M3. The reference corrects the
    sm89 Triton backend's fp16 truncation before that cast and attributes
    its differences against the uncorrected kernel separately.
    Its div.full.f32 implements a reciprocal multiply: an input
    0.044189453125 over 0.0589192733168602 becomes 0.75, which rounds to
    1, while IEEE divide gives 0.74999994 and rounds to 0.5. The compiled
    negative-zero FMA yields +0 for a zero code. The 32 pinned real q_proj
    rows and the author's exporter fixture hold this order bit for bit.
    """
    from dew.nn.fake_quant import straight_through

    values = jax.lax.stop_gradient(x).astype(jnp.float32)
    groups = values.reshape(*x.shape[:-1], -1, 16)
    largest = jnp.max(jnp.abs(groups), -1)
    # At g=0 every effective scale falls below the 1e-5 guard and becomes 1;
    # a tiny positive stand-in keeps that result while avoiding 0/0 NaNs.
    global_scale = jnp.float32(spec.global_scale if spec.global_scale > 0 else 1e-12)
    denominator = jnp.full(largest.shape, 6, jnp.float32) * global_scale
    normalized = largest * _nvfp4_divide(jnp.ones_like(largest), denominator)
    saturated = jnp.minimum(normalized, 448)
    scales = saturated.astype(jnp.float8_e4m3fn).astype(jnp.float32) * global_scale
    scales = jnp.where(scales >= 1e-5, scales, 1)
    spread = jnp.broadcast_to(scales[..., None], groups.shape).reshape(x.shape)
    quotients = values * _nvfp4_divide(jnp.ones_like(values), spread)
    quantized = qarray.quantize_with_scale_zero_point(quotients, 'nvfp4', jnp.ones_like(spread), None)
    rounded = qarray.dequantize(quantized.replace(scale=spread))
    return straight_through(x, jnp.where(rounded == 0, jnp.zeros_like(rounded), rounded))


def checkpoint_input_quantization(model: nn.Module, inputs: Mapping[str, NVFP4Input | FP8Input]) -> nn.Module:
    """Return `model` with its checkpoint's input quantization on the Linear layers.

    `inputs` maps each Linear's module path to its NVFP4 or FP8 input scales. The
    wrapper goes through Qwix, as the rest of this module does: before each
    matched layer's dot, its input is quantized to its declared format and back with those
    scales. The weights already hold the values the source reader decoded, and
    the dot itself is unchanged, so its dtype, accumulation precision and the
    bias placement stay as the model defines them.
    """
    qwix = _qwix()
    by_pattern = {re.escape(path): spec for path, spec in inputs.items()}

    class CheckpointInputs(qwix.QtProvider):
        def dot_general(self, lhs, rhs, dimension_numbers, precision=None,
                        preferred_element_type=None, *, out_sharding=None):
            # Qwix 0.1.8's private lookup preserves the native scope these checkpoint scales bind.
            rule, _ = self._get_current_rule_and_op_id('dot_general', only_rule=True)
            if rule is not None:
                if dimension_numbers[0][0] != (lhs.ndim - 1,):
                    raise ValueError("checkpoint NVFP4 input QDQ requires the Linear's trailing input axis")
                spec = by_pattern[rule.module_path]
                lhs = fp8_input_qdq(lhs, spec) if isinstance(spec, FP8Input) else nvfp4_input_qdq(lhs, spec)
            return jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=precision,
                                       preferred_element_type=preferred_element_type,
                                       out_sharding=out_sharding)

    rules = [qwix.QtRule(module_path=pattern, op_names=('dot_general',)) for pattern in by_pattern]
    return qwix.quantize_model(model, CheckpointInputs(rules),
                               methods=tuple(method for method in METHODS if hasattr(model, method)))


def _qtype(dtype: QuantizedDtype) -> jax.typing.DTypeLike:
    """Return `dtype` as the JAX dtype Qwix quantizes to."""
    return jnp.int8 if dtype == "int8" else jnp.float8_e4m3fn


def _group_scales(lhs: jax.Array, dimension_numbers: jax.lax.ConvDimensionNumbers,
                  groups: int) -> jax.Array:
    """Each example's absolute maximum in each feature group of a
    convolution's input, `[batch, groups]`, with 1 standing in for an
    all-zero group."""
    batch, feature = dimension_numbers.lhs_spec[:2]
    moved = jnp.moveaxis(lhs, (batch, feature), (0, -1))
    grouped = moved.reshape(*moved.shape[:-1], groups, moved.shape[-1] // groups)
    peak = jnp.max(jnp.abs(grouped), axis=(*range(1, moved.ndim - 1), moved.ndim))
    return jax.lax.stop_gradient(jnp.where(peak > 0, peak, 1).astype(lhs.dtype))


def _per_feature(scales: jax.Array, ndim: int, batch: int, feature: int, features: int) -> jax.Array:
    """`[batch, groups]` scales repeated over each group's `features // groups`
    contiguous features, laid out to broadcast against an array of rank
    `ndim` with its batch and feature axes where the arguments say."""
    repeated = jnp.repeat(scales, features // scales.shape[1], axis=1)
    shape = [1] * ndim
    shape[batch], shape[feature] = repeated.shape
    return (repeated if batch < feature else repeated.T).reshape(shape)


def _real(*dtypes: jax.typing.DTypeLike) -> bool:
    """Whether no dtype is complex; Qwix quantizes real values only."""
    return not any(jnp.issubdtype(dtype, jnp.complexfloating) for dtype in dtypes)


def _refuse_grouped_on_gpu() -> None:
    """Refuse a grouped convolution with quantized activations on a GPU,
    where jax 0.11.2 computed it wrong or failed to compile it (the error
    says where; in int8 on the A100 the 176M text-to-image model's CLIP score
    fell from 0.247 to 0.137). The refusal covers every GPU, dtype and group
    width until an H100 is measured."""
    if jax.default_backend() == "gpu":
        raise ValueError(
            "Dew refuses to quantize a grouped convolution's activations on a GPU. Measured with jax "
            "0.11.2: in int8, one input channel per group gives wrong values without an error on the "
            "RTX 4080 and the A100, and two per group on the RTX 4080; in fp8, one or two per group "
            "fail to compile on sm_89 (the RTX 4080), the A100 computes it emulated, with no fp8 "
            "units to gain from, and sm_90 is untested. Leave the convolution out of "
            "Quantization.patterns, as patterns=('^(?!.*spatial_fusion).*',) does for the hybrid "
            "DiT's depthwise convolutions, or quantize weights only (weight_only=True)")


def _scaled_in_float32[**P](op: Callable[P, jax.Array]) -> Callable[P, jax.Array]:
    """Qwix's quantized `op` with the product of two quantized operands
    scaled in float32, then rounded once to the dtype Qwix returns it in.

    Qwix 0.1.8 scales it in the scales' dtype: a bf16 model's int32
    accumulators became bf16 before either scale multiplied in, and XLA:TPU
    then emitted its int8 and fp8 matmuls and convolutions with bf16
    results. On a v6e an int8 depthwise convolution scaled that way came out
    NaN in all but a few outputs in plain JAX, while the dense and attention
    forms of the 176M text-to-image model stayed finite on their own. Served
    in bf16 that model sampled NaN images, from its depthwise convolutions in
    int8 and from an attention block in fp8, and scaled in float32 it samples
    as well as unquantized (docs/performance.md). A weight-only product, one
    quantized operand, dequantizes before a float matmul and passes through.

    Quantized training runs Qwix's own operations, with no hook for this.
    Its grouped convolutions quantize from float32 instead
    (`GroupScaledConvolution`); its matmuls keep Qwix's scaling, and whether
    a bf16 fp8 run on a TPU turns NaN the way the served fp8 model did is not
    measured."""
    qarray = _qwix("qwix._src.core.qarray")

    def scaled(*args: P.args, **kwargs: P.kwargs) -> jax.Array:
        operands = [arg for arg in args if isinstance(arg, qarray.QArray)]
        if len(operands) < 2:
            return op(*args, **kwargs)
        _, dtype = qarray.get_accumulator_and_result_type(*operands, preferred_element_type=None)
        rescaled = jax.tree.map(
            lambda arg: arg.astype(jnp.float32) if isinstance(arg, qarray.QArray) else arg,
            args,
            is_leaf=lambda arg: isinstance(arg, qarray.QArray),
        )
        return op(*rescaled, **kwargs).astype(dtype)

    return scaled


@functools.cache
def _grouped_convolution_gradient() -> type:
    """Qwix's quantized-training provider with Dew's gradient for a grouped
    convolution; importing Qwix here keeps it optional."""
    qwix = _qwix()
    conv_general_qt = _qwix("qwix._src.core.conv_general_qt")
    qarray = _qwix("qwix._src.core.qarray")

    @functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(2, 11)))
    def grouped_conv_qt(lhs, rhs, config, window_strides, padding, lhs_dilation, rhs_dilation,
                        dimension_numbers, feature_group_count, batch_group_count, out_sharding):
        return conv_general_qt.conv_general_qt_fwd(
            lhs, rhs, config, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers,
            feature_group_count, batch_group_count, out_sharding)[0]

    def straight_through(config, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers,
                         feature_group_count, batch_group_count, out_sharding, residuals, g):
        operands = [qarray.dequantize(operand) if isinstance(operand, qarray.QArray) else operand
                    for operand in residuals]

        def convolve(lhs, rhs):
            # A bf16 module's kernel meets Dew's float32 grouped input
            # (`GroupScaledConvolution`); its gradient returns as bf16.
            dtype = jnp.result_type(lhs, rhs)
            return jax.lax.conv_general_dilated(
                lhs.astype(dtype), rhs.astype(dtype), window_strides, padding, lhs_dilation, rhs_dilation,
                dimension_numbers, feature_group_count, batch_group_count)

        _, transpose = jax.vjp(convolve, *operands)
        return transpose(g)

    grouped_conv_qt.defvjp(conv_general_qt.conv_general_qt_fwd, straight_through)

    class GroupedConvolutionGradient(qwix.QtProvider):
        """Differentiates a quantized grouped convolution, which Qwix's
        quantized training (0.1.8 `conv_general_qt_bwd`) cannot: its backward
        convolves the gradient with the forward's `feature_group_count` and
        the kernel's grouped shape, which only fits one group, so `jax.grad`
        raised for the hybrid DiT's depthwise convolutions.

        The forward is Qwix's. The backward is the one Qwix computes for an
        ungrouped convolution with float gradients: the float convolution's
        transpose at the dequantized operands the forward computed with,
        here JAX's own, which handles groups. Qwix's quantized gradients
        (`bwd_qtype`) for a grouped convolution are refused.

        An ungrouped convolution with quantized activations quantizes its
        float32 operands and returns the input's dtype, so Qwix scales its
        8-bit product in float32 as `GroupScaledConvolution` does a grouped
        one's. From bf16 operands Qwix scaled it in bf16, and XLA:GPU, which
        lowers an int8 convolution only with a float32 result, failed to
        compile a bf16 int8 convolution (`UNIMPLEMENTED: Can't lower one or
        more integer convolutions`, the RTX 4080)."""

        def conv_general_dilated(self, lhs: jax.Array, rhs: jax.Array, window_strides: Sequence[int],
                                 padding: str | Sequence[tuple[int, int]],
                                 lhs_dilation: Sequence[int] | None = None,
                                 rhs_dilation: Sequence[int] | None = None,
                                 dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
                                 feature_group_count: int = 1, batch_group_count: int = 1,
                                 precision: jax.lax.PrecisionLike = None,
                                 preferred_element_type: jax.typing.DTypeLike | None = None,
                                 out_sharding: jax.sharding.NamedSharding | None = None) -> jax.Array:
            rule, _ = self._get_current_rule_and_op_id("conv_general_dilated", only_rule=True)
            if (
                rule is None
                or rule.weight_qtype is None
                or (feature_group_count == 1 and rule.act_qtype is None)
            ):
                return super().conv_general_dilated(
                    lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers,
                    feature_group_count, batch_group_count, precision, preferred_element_type, out_sharding)
            if feature_group_count == 1:
                return super().conv_general_dilated(
                    lhs.astype(jnp.float32), rhs.astype(jnp.float32), window_strides, padding, lhs_dilation,
                    rhs_dilation, dimension_numbers, feature_group_count, batch_group_count, precision,
                    preferred_element_type, out_sharding).astype(lhs.dtype)
            if rule.bwd_qtype is not None:
                raise ValueError(
                    "Qwix 0.1.8 cannot compute the quantized gradients of a grouped convolution, so Dew "
                    "refuses bwd_qtype for one; leave bwd_qtype unset, or leave the convolution out of "
                    "Quantization.patterns, as patterns=('^(?!.*spatial_fusion).*',) does for the hybrid "
                    "DiT's depthwise convolutions")
            if rule.tile_size:
                raise ValueError("subchannel is not supported for conv_general_dilated.")
            rule, op_id = self._get_current_rule_and_op_id("conv_general_dilated")
            return grouped_conv_qt(lhs, rhs, self._create_conv_general_qt_config(rule, op_id, lhs, rhs),
                                   window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers,
                                   feature_group_count, batch_group_count, out_sharding)

    return GroupedConvolutionGradient


@functools.cache
def _group_scaled_convolution() -> type:
    """Qwix's provider base with Dew's grouped convolution, the base of both
    of Dew's providers; importing Qwix here keeps it optional."""
    qwix = _qwix()
    dew_conv = importlib.import_module("dew.nn.conv")

    class GroupScaledConvolution(qwix.QuantizationProvider):
        """Quantizes a grouped convolution's input with one scale per feature
        group, where Qwix (0.1.8 `conv_general.get_how_to_quantize`) takes one
        per example across every feature.

        A grouped convolution contracts each group's features on their own, so
        one group's range has no bearing on another's rounding. With one scale
        across all of them a depthwise convolution rounds each channel against
        the loudest. The hybrid DiT's spatial fusion, whose channel peaks span
        36x, came out 4.1% off in int8 against 1.0% with a scale per group, and
        quantizing it alone dropped the 176M text-to-image model's CLIP score
        from 0.248 to 0.124.
        Dividing each group by its own peak before Qwix quantizes, and
        multiplying each output group by it after, gives the per-group scales
        exactly: the convolution is linear and keeps groups apart, and every
        group then peaks at 1, so Qwix's single scale is each group's own.

        The divided input is float32, so Qwix scales the convolution's 8-bit
        product in float32 in training as well as in serving
        (`_scaled_in_float32`), and the output returns to the input's dtype.
        From a bf16 input Qwix scaled it in bf16, a form XLA:TPU computes as
        NaN for an int8 depthwise convolution.
        """

        def get_intercept_map(self):
            # Dew's Conv convolves through dew.nn.conv._conv_general_dilated,
            # which on CUDA computes a dilated depthwise convolution as shifted
            # products and so reaches no lax convolution Qwix intercepts.
            return {**super().get_intercept_map(),
                    "dew.nn.conv._conv_general_dilated": self.dew_conv_general_dilated}

        def dew_conv_general_dilated(self, lhs: jax.Array, rhs: jax.Array, *args, **kwargs) -> jax.Array:
            """Dew's convolution, through this provider when a rule
            quantizes it, so that it keeps the per-group scales and the GPU
            refusal, and otherwise Dew's own, as the unwrapped model computes
            it (inside this handler Qwix calls the original). Sent through
            the provider unquantized, an excluded or weight-only dilated
            depthwise convolution took lax's path on CUDA, and quantized
            training of the hybrid DiT parted from Qwix's own and from the
            unwrapped model after its zero-initialized fusion kernels'
            first update (ColabPlan-2, the RTX 4080)."""
            rule, _ = self._get_current_rule_and_op_id("conv_general_dilated", only_rule=True)
            if rule is None or rule.weight_qtype is None:
                return dew_conv._conv_general_dilated(lhs, rhs, *args, **kwargs)
            return self.conv_general_dilated(lhs, rhs, *args, **kwargs)

        def conv_general_dilated(self, lhs: jax.Array, rhs: jax.Array, window_strides: Sequence[int],
                                 padding: str | Sequence[tuple[int, int]],
                                 lhs_dilation: Sequence[int] | None = None,
                                 rhs_dilation: Sequence[int] | None = None,
                                 dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
                                 feature_group_count: int = 1, batch_group_count: int = 1,
                                 precision: jax.lax.PrecisionLike = None,
                                 preferred_element_type: jax.typing.DTypeLike | None = None,
                                 out_sharding: jax.sharding.NamedSharding | None = None) -> jax.Array:
            convolve = functools.partial(
                super().conv_general_dilated,
                rhs=rhs, window_strides=window_strides, padding=padding, lhs_dilation=lhs_dilation,
                rhs_dilation=rhs_dilation, dimension_numbers=dimension_numbers,
                feature_group_count=feature_group_count, batch_group_count=batch_group_count,
                precision=precision, preferred_element_type=preferred_element_type, out_sharding=out_sharding)
            rule, _ = self._get_current_rule_and_op_id("conv_general_dilated", only_rule=True)
            if feature_group_count == 1 or rule is None or rule.act_qtype is None:
                return convolve(lhs=lhs)
            _refuse_grouped_on_gpu()
            numbers = jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
            scales = _group_scales(lhs, numbers, feature_group_count).astype(jnp.float32)
            batch, feature = numbers.lhs_spec[:2]
            out = convolve(lhs=lhs.astype(jnp.float32)
                           / _per_feature(scales, lhs.ndim, batch, feature, lhs.shape[feature]))
            batch, feature = numbers.out_spec[:2]
            return (out * _per_feature(scales, out.ndim, batch, feature, out.shape[feature])).astype(
                lhs.dtype
            )

    return GroupScaledConvolution


@functools.cache
def _providers() -> tuple[type, type]:
    """Qwix's quantized-training and serving providers with Dew's grouped
    convolution; importing Qwix here keeps it optional.

    Both pass a matmul with a complex operand through unquantized, as the
    hybrid DiT's S5 scan needs: Qwix 0.1.8's serving raises on one, and its
    quantized training quantized the real operand of the scan's
    real-by-complex input projection, after which `jax.grad` failed on the
    complex gradient Qwix returned for it. The training provider
    differentiates a grouped convolution (`_grouped_convolution_gradient`).

    The serving provider casts a quantized kernel's scales to the dtype a
    module promotes its kernel to, where Qwix 0.1.8 passes the kernel
    through: a bf16 module then multiplied bf16 activations by fp32
    dequantized kernels, so its matmuls ran in fp32 (50.4 ms against 34.8 for
    the plain bf16 176M DiT forward at batch 24 on the RTX 4080). And it
    scales two quantized operands' product in float32
    (`_scaled_in_float32`)."""
    qwix = _qwix()
    conv_general = _qwix("qwix._src.core.conv_general")
    dot_general = _qwix("qwix._src.core.dot_general")
    einsum = _qwix("qwix._src.core.einsum")
    quantized = _qwix("qwix._src.providers.ptq").WithAux
    group_scaled = _group_scaled_convolution()

    class ComplexInFloat(qwix.QuantizationProvider):
        """Passes a matmul with a complex operand through unquantized."""

        def dot_general(self, lhs, rhs, dimension_numbers, precision=None,
                        preferred_element_type=None, *, out_sharding=None):
            if _real(lhs.dtype, rhs.dtype):
                return super().dot_general(lhs, rhs, dimension_numbers, precision,
                                           preferred_element_type, out_sharding=out_sharding)
            return jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=precision,
                                       preferred_element_type=preferred_element_type,
                                       out_sharding=out_sharding)

        def einsum(self, einsum_str, *operands, **kwargs):
            if _real(*(operand.dtype for operand in operands)):
                return super().einsum(einsum_str, *operands, **kwargs)
            return jnp.einsum(einsum_str, *operands, **kwargs)

    class QtProvider(group_scaled, ComplexInFloat, _grouped_convolution_gradient()):
        pass

    class PtqProvider(group_scaled, ComplexInFloat, qwix.PtqProvider):
        def __init__(self, rules: Sequence[object]) -> None:
            super().__init__(
                rules, _dot_general_fn=_scaled_in_float32(dot_general.dot_general),
                _einsum_fn=_scaled_in_float32(einsum.einsum),
                _conv_general_dilated_fn=_scaled_in_float32(conv_general.conv_general_dilated))

        def promote_dtype(self, *args, **kwargs):
            # `WithAux.astype` itself raises on a QArray in Qwix 0.1.8
            # (`flax_util.update_boxed` accepts boxes and arrays only).
            promoted = super().promote_dtype(*args, **kwargs)
            dtype = kwargs.get("dtype")
            if dtype is None:
                return promoted
            return [value.replace(array=nn.unbox(value.array).astype(dtype))
                    if isinstance(value, quantized) else value for value in promoted]

    return QtProvider, PtqProvider


def _rules(spec: Quantization, training: bool) -> list:
    """One Qwix rule per pattern of `spec`, in its order: quantized-training
    rules with the backward fields, or serving rules. A weight-only rule
    names the matmul ops alone (`Quantization.weight_only`)."""
    qwix = _qwix()
    rule, fields = qwix.QuantizationRule, {}
    if training:
        rule, fields = qwix.QtRule, {
            "bwd_qtype": None if spec.bwd_qtype is None else _qtype(spec.bwd_qtype),
            "bwd_stochastic_rounding": spec.bwd_stochastic_rounding}
    if spec.weight_only:
        fields["op_names"] = ("dot_general", "einsum", "dot")
    return [rule(module_path=pattern, weight_qtype=_qtype(spec.dtype),
                 act_qtype=None if spec.weight_only else _qtype(spec.dtype),
                 tile_size=spec.tile_size, weight_calibration_method=spec.calibration, **fields)
            for pattern in spec.patterns]


def _serving_parameters(parameters: Variables, abstract: Variables) -> Variables:
    """Quantize one kernel at a time, retaining its host or device placement."""
    qwix = _qwix()
    shapes = flatten_dict(abstract)
    quantized = {}
    for path, parameter in flatten_dict(parameters).items():
        host = isinstance(parameter, np.ndarray)
        converted = qwix.quantize_params(
            {"weight": jnp.asarray(parameter) if host else parameter}, {"weight": shapes[path]})["weight"]
        quantized[path] = jax.device_get(converted) if host else converted
        # Release a host kernel's device buffers before converting the next.
        del converted
    return unflatten_dict(quantized)


def quantize_for_serving(model: nn.Module, variables: Variables, spec: Quantization,
                         *args: Conditioning, **kwargs: Conditioning | Mapping[str, jax.Array]
                         ) -> tuple[nn.Module, Variables]:
    """`model` and `variables` with the weights `spec` names stored quantized.

    This is Qwix's post-training quantization. The returned variables hold
    each matched kernel as int8 or fp8 values with their scales, in place of
    the float kernel, so the weights take about a quarter of fp32's memory, and the
    returned module computes with them. Unless `spec.weight_only`, its
    activations quantize at each matmul from their own range, as training
    under `Quantization.apply` does, and a matmul of two quantized operands
    runs in the quantized dtype. `args` and `kwargs` are one example call of
    the model, which Qwix traces abstractly to find the kernels its
    matmuls read. `spec`'s backward fields have nothing to do here.

    A weight-only spec leaves convolutions in float: Qwix 0.1.8's serving
    provider quantizes a convolution's weights only together with its
    activations.

    Host NumPy kernels quantize one at a time on the default device and
    return to host storage, ready for placement. Temporary device storage is
    bounded by one kernel, not the whole model. If one kernel will not fit,
    place the weights with a mesh and layout before quantizing. Resident
    JAX parameters keep their placement through Qwix's operations.
    """
    rules = _rules(spec, training=False)
    qwix = _qwix()
    methods = tuple(method for method in METHODS if hasattr(model, method))
    served = qwix.quantize_model(model, _providers()[1](rules), methods=methods)
    def initialized(scope: core.Scope, supplied: Variables):
        # Initialization lets Qwix annotate each weight with its quantization
        # recipe. Seed the scope first so data-dependent module layouts, such
        # as packed inference projections, trace the weights they will read.
        for collection, values in core.unfreeze(dict(supplied)).items():
            for name, value in values.items():
                scope.put_variable(collection, name, value)
        return served.clone(parent=scope)(*args, **kwargs)

    _, abstract = jax.eval_shape(core.init(initialized), jax.random.key(0), variables)
    parameters = _serving_parameters(variables["params"], abstract["params"])
    return served, {**variables, "params": parameters}


@runtime_checkable
class _Quantized(Protocol):
    """Marks a module class `Quantization.apply` wrapped: Qwix's subclass of
    the model's own class, and the spec that wrapped it."""

    _unquantized_type: ClassVar[type[nn.Module]]
    _dew_quantization: ClassVar[Quantization]


def _quantize[LossT, EffectsT](objective: Objective[LossT, EffectsT], spec: Quantization) -> None:
    """Quantize the trunk matmuls of the modules `objective` trains, in place.

    This is `RunConfig.train`'s step, not a user's. `Quantization.apply`
    wraps a module before an objective is built, which is what a recipe or
    a script that builds its own model does. A run that names
    `--trainer.quantization` has handed `RunConfig.train` the objective
    already, so the wrap lands on the modules the objective runs instead
    (`Objective.substitute`), before anything has initialised or traced
    them; each wrapped module is a copy of the same class, so what the
    objective read off it at construction still holds. A frozen teacher or
    reference keeps its own numerics.

    An objective that trains no module has nothing to wrap and is refused by name.
    """
    entries = objective.program_key()
    if not any(entry.trained for entry in entries):
        raise ValueError(
            f"--trainer.quantization quantizes the modules an objective trains, and "
            f"{type(objective).__name__} trains none; train an objective that holds a model, "
            f"or leave the quantization unset")
    objective.substitute([spec.apply(entry.module) if entry.trained else entry.module for entry in entries])


__all__ = ["Quantization"]
