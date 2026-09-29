"""Quantized training through Qwix, applied to the model before it trains.

Qwix (google/qwix, Apache 2.0) expresses quantization as rules over module
paths and applies them without editing the model. One call wraps the module,
and the matmuls in the wrapped methods' extent run quantized.

Dew's version of that call is `apply_quantization`. A caller builds its model
from the registry as always, then wraps it before the objective ever sees it.
A run that names `--trainer.quantization` instead hands `RunConfig.train` the
objective, and `quantize` wraps the model it holds before anything
initialises it.

What trains is fake-quantized. The parameter tree keeps fp32 master weights
with the same structure, so the checkpoint layout, the sharding derivation,
the Muon parameter split and Hugging Face loading are unchanged. The
quantization lives in the forward and backward matmuls, with a
straight-through estimator on the backward pass.

Serving is not fake-quantized. `quantize_for_serving` runs Qwix's
post-training quantization: the returned variables hold each matched kernel
as int8 or fp8 values with their scales, and the returned module's matmuls
read them. `TextToImage.quantized` applies it to a text-to-image task's
denoiser.

The vocabulary head stays fp32 with the rest of Dew's fp32 zones. Its einsum
lives in the objective's chunked cross entropy, outside any model method
Qwix wraps.

The value mirrors MaxText's knob set (configs/base.yml:128-167) where Qwix
has an equivalent. `dtype` is its `quantization` for the dynamic-range forms
and `patterns` its `quant_cfg_path`, written inline as the regexes Qwix
matches; the backward fields are Qwix's finer-grained version of the same
idea.

Three of its knobs have no equivalent, and `dtype` takes neither of the
first two. Static activation scaling (`fp8_full`) needs a calibration pass
Dew has no seam for, `nanoo_fp8` is AMD-only kernels, and KV-cache
quantization has no reader here since the cache holds the compute dtype.

Qwix is not a dependency. The import sits inside the calls that need it,
and without the package they raise naming it, the way the tokamax branch of
`dew.nn.moe` behaves.
"""
from __future__ import annotations

import dataclasses
import functools
import importlib
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import linen as nn

if TYPE_CHECKING:
    from dew.diffusion.process import Conditioning
    from dew.objectives.base import Variables

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
# `mtp_hidden_states`. Qwix's interception is non-recursive over dynamic
# extent, so `__call__` reaching `hidden_states` quantizes once.
METHODS = ("__call__", "hidden_states", "mtp_hidden_states")


@dataclasses.dataclass(frozen=True)
class Quantization:
    """Says how a run quantizes its trunk matmuls, for Qwix's provider."""

    dtype: QuantizedDtype = "int8"
    """The dtype weights and activations quantize to, in the forward pass."""
    weight_only: bool = False
    """Quantize the weights alone and keep activations in the compute dtype.
    Convolutions stay unquantized under it, since Qwix's serving provider
    refuses a convolution whose activations stay in float."""
    patterns: tuple[str, ...] = (".*",)
    """Module-path regexes the rules apply to, in Qwix precedence order: the
    first rule whose regex full-matches a module's `/`-joined scope path wins.
    `'.*mlp.*'` quantizes the feed-forward blocks and leaves attention in
    fp32; the default quantizes every matmul of the wrapped methods."""
    calibration: str = "absmax"
    """How weights calibrate, as Qwix parses it: a method with an optional
    `,args` suffix, for example `absmax,0.8`."""
    tile_size: int | None = None
    """Sub-channel tiling of the contraction axis; unset keeps per-channel
    scales, the coarser and cheaper form."""
    bwd_qtype: QuantizedDtype | None = None
    """The dtype gradients quantize to in the backward pass; unset keeps
    them in the compute dtype."""
    bwd_stochastic_rounding: Rounding | None = None
    """Stochastic rounding on the quantized gradients. A run that sets this
    passes a `stochastic_rounding` RNG stream at apply time, which Qwix draws
    (`qwix/_src/providers/qt.py:361`); unset rounds deterministically."""

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
    where XLA:GPU (jax 0.11.2) computes one wrongly or not at all.

    In int8, with one or two input channels per group and the int32 result
    scaled in float, plain JAX returns wrong values without an error on the
    RTX 4080 (75% and 50% of the outputs), and on the A100 quantizing the
    176M text-to-image model's depthwise convolutions dropped its CLIP score
    from 0.247 to 0.137. With four per group Dew's int8 convolution failed
    to compile on the RTX 4080, as a plain int8 convolution with int32
    results does there at every group width. In fp8, one or two input
    channels per group fail to compile on the RTX 4080, whose fp8 is
    native."""
    if jax.default_backend() == "gpu":
        raise ValueError(
            "XLA:GPU computes a grouped convolution with int8 or fp8 activations wrongly or not at "
            "all, depending on the GPU, the dtype and the channels per group, so Dew refuses to "
            "quantize one; leave it out of Quantization.patterns, as "
            "patterns=('^(?!.*spatial_fusion).*',) does for the hybrid DiT's depthwise "
            "convolutions, or quantize weights only (weight_only=True)")


def _scaled_in_float32[**P](op: Callable[P, jax.Array]) -> Callable[P, jax.Array]:
    """Qwix's quantized `op` with the product of two quantized operands
    scaled in float32, then rounded once to the dtype Qwix returns it in.

    Qwix 0.1.8 scales it in the scales' dtype: a bf16 model's int32
    accumulators became bf16 before either scale multiplied in, and XLA:TPU
    then emitted its int8 and fp8 matmuls and convolutions with bf16
    results. The bf16 176M text-to-image model served that way sampled NaN
    images on a v6e, while the float32 one, whose 8-bit products come out
    in int32 or float32, sampled as well as unquantized. A weight-only
    product, one quantized operand, dequantizes before a float matmul and
    passes through."""
    qarray = importlib.import_module("qwix._src.core.qarray")

    def scaled(*args: P.args, **kwargs: P.kwargs) -> jax.Array:
        operands = [arg for arg in args if isinstance(arg, qarray.QArray)]
        if len(operands) < 2:
            return op(*args, **kwargs)
        _, dtype = qarray.get_accumulator_and_result_type(*operands, preferred_element_type=None)
        rescaled = jax.tree.map(lambda arg: arg.astype(jnp.float32) if isinstance(arg, qarray.QArray) else arg,
                                args, is_leaf=lambda arg: isinstance(arg, qarray.QArray))
        return op(*rescaled, **kwargs).astype(dtype)

    return scaled


@functools.cache
def _providers() -> tuple[type, type]:
    """Qwix's quantized-training and serving providers with Dew's grouped
    convolution; importing Qwix here keeps it optional.

    The serving provider also passes complex matmuls through unquantized,
    as the training provider does (`numerics.should_quantize`); Qwix 0.1.8's
    PTQ raises on them instead, and the hybrid DiT's S5 scan multiplies
    complex states. It casts a quantized kernel's scales to the dtype a
    module promotes its kernel to, where Qwix 0.1.8 passes the kernel
    through: a bf16 module then multiplied bf16 activations by fp32
    dequantized kernels, so its matmuls ran in fp32 (50.4 ms against 34.8 for
    the plain bf16 176M DiT forward at batch 24 on the RTX 4080). And it
    scales two quantized operands' product in float32
    (`_scaled_in_float32`)."""
    qwix = importlib.import_module("qwix")
    core = {name: importlib.import_module(f"qwix._src.core.{name}")
            for name in ("conv_general", "dot_general", "einsum")}
    quantized = importlib.import_module("qwix._src.providers.ptq").WithAux

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
        """

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
            scales = _group_scales(lhs, numbers, feature_group_count)
            batch, feature = numbers.lhs_spec[:2]
            out = convolve(lhs=lhs / _per_feature(scales, lhs.ndim, batch, feature, lhs.shape[feature]))
            batch, feature = numbers.out_spec[:2]
            return out * _per_feature(scales, out.ndim, batch, feature, out.shape[feature]).astype(out.dtype)

    class QtProvider(GroupScaledConvolution, qwix.QtProvider):
        pass

    class PtqProvider(GroupScaledConvolution, qwix.PtqProvider):
        def __init__(self, rules: Sequence[object]) -> None:
            super().__init__(
                rules, _dot_general_fn=_scaled_in_float32(core["dot_general"].dot_general),
                _einsum_fn=_scaled_in_float32(core["einsum"].einsum),
                _conv_general_dilated_fn=_scaled_in_float32(core["conv_general"].conv_general_dilated))

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
    qwix = importlib.import_module("qwix")
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


def apply_quantization(model: nn.Module, spec: Quantization) -> nn.Module:
    """Wrap `model` so its trunk matmuls train in `spec`'s dtype.

    The returned module is a copy of the same class with the entry methods
    it defines of `METHODS` wrapped, so everything the registry, the
    objective and the checkpoint code read off the model still answers.
    Construction already refused what the value cannot ask for; without the
    package the call raises naming it.
    """
    rules = _rules(spec, training=True)
    methods = tuple(method for method in METHODS if hasattr(model, method))
    return importlib.import_module("qwix").quantize_model(model, _providers()[0](rules), methods=methods)


def quantize_for_serving(model: nn.Module, variables: Variables, spec: Quantization,
                         *args: Conditioning, **kwargs: Conditioning) -> tuple[nn.Module, Variables]:
    """`model` and `variables` with the weights `spec` names stored quantized.

    This is Qwix's post-training quantization. The returned variables hold
    each matched kernel as int8 or fp8 values with their scales, in place of
    the float kernel, so the weights take about a quarter of fp32's memory, and the
    returned module computes with them. Unless `spec.weight_only`, its
    activations quantize at each matmul from their own range, as training
    under `apply_quantization` does, and a matmul of two quantized operands
    runs in the quantized dtype. `args` and `kwargs` are one example call of
    the model, which Qwix traces abstractly to find the kernels its
    matmuls read. `spec`'s backward fields have nothing to do here.

    A weight-only spec leaves convolutions in float: Qwix 0.1.8's serving
    provider quantizes a convolution's weights only together with its
    activations.
    """
    rules = _rules(spec, training=False)
    qwix = importlib.import_module("qwix")
    methods = tuple(method for method in METHODS if hasattr(model, method))
    served = qwix.quantize_model(model, _providers()[1](rules), methods=methods)
    abstract = jax.eval_shape(functools.partial(served.init, jax.random.key(0), *args, **kwargs))
    return served, {**variables, "params": qwix.quantize_params(variables["params"], abstract["params"])}


@runtime_checkable
class ModelObjective(Protocol):
    """Trains one module, which is the shape `quantize` can wrap.

    The module is the objective's `model`, and every trace it runs reads it
    there."""

    model: nn.Module


def quantize(objective: object, spec: Quantization) -> None:
    """Quantize the trunk matmuls of the module `objective` trains.

    `apply_quantization` wraps a module before an objective is built, which
    is what a recipe that builds its own model does. A run that names
    `--trainer.quantization` has handed `RunConfig.train` the objective
    already, so the wrap lands on the objective's own model instead, before
    anything has initialised or traced it; the wrapped module is a copy of
    the same class, so what the objective read off the model at construction
    still holds.

    An objective that trains something other than one module has nothing to
    wrap and is refused by name.
    """
    if not isinstance(objective, ModelObjective):
        raise ValueError(
            f"--trainer.quantization quantizes the module an objective trains, and "
            f"{type(objective).__name__} keeps no `model`; train an objective that "
            f"holds one, or leave the quantization unset")
    objective.model = apply_quantization(objective.model, spec)
