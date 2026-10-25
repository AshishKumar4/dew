"""Quantization of a model's matmuls through Qwix, for training, serving and loading.

Qwix (google/qwix, Apache 2.0) describes quantization as rules over module
paths and applies them without editing the model. One call wraps the module,
and the matmuls that run inside the wrapped methods run quantized.

In Dew that call is `Quantization.apply`. A caller builds its model as
usual, then wraps it before the objective ever sees it. A run that sets
`--trainer.quantization` passes the objective to `RunConfig.train` instead,
and the model the objective holds is wrapped there, before anything
initialises it (`dew.training.quantization.quantize_trunk`). This module is
the model's: loading, adapters, training and serving read it, and it reads
nothing of training.

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

Two MaxText settings have no equivalent, and `dtype` accepts neither.
Static activation scaling (`fp8_full`) needs a calibration pass that Dew
has no hook for, and `nanoo_fp8` uses AMD-only kernels. KV-cache
quantization is not a weight quantization: the cache's own layout stores
int8 or float8 keys and values (`dew.nn.kv_cache`).

Qwix comes with the `quantization` extra (`pip install "dewml[quantization]"`).
It is imported inside the calls that need it, and without the package those
calls raise an error that names the extra.
"""
from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import math
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import core, linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

from dew.cache import QUANTIZED_TRACE
from dew.nn.kernels.generation import backend
from dew.nn.precision import correctly_rounded_divide

if TYPE_CHECKING:
    from qwix import QArray, QuantizationProvider

    from dew.diffusion.process import Conditioning
    from dew.objectives.base import Variables

QuantizedDtype = Literal["int8", "fp8"]
"""The gemm dtypes a run trains with: int8 on any backend, fp8 where the
backend lowers it (measured in docs/performance.md)."""

Rounding = Literal["uniform", "low_bit_uniform"]
"""How a quantized gradient rounds: Qwix's two stochastic modes."""


UNQUANTIZED = {
    "dew.nn.ssm.diagonal_recurrence":
        "a diagonal recurrence's chunk products, powers of its pole compounded over every step",
    "dew.nn.attention_residuals.depth_mixed":
        "depth attention's fp32 softmax over earlier layers' outputs and their weighted sum",
    "dew.nn.quantizers.nearest_codewords":
        "a vector quantizer's codeword distances, whose argmax a rounded product reorders",
    "dew.nn.kv_cache.rotated": "the rotation a quantized cache applies before it rounds keys and values",
    "dew.lora._updated": "an adapter's factor product merged into the kernel it updates, a weight and not an "
                         "activation",
}
"""The functions whose products a quantized model keeps in float, at the precision they pin, each
with why. Every other product a rule matches quantizes, whatever precision it pins: a quantized
product has none to honour. The providers run each through Qwix's intercept map (`_providers`), as
Qwix runs `pallas_call`, and inside a handler Qwix leaves the products unintercepted."""

def entry_methods(model: nn.Module) -> tuple[str, ...]:
    """Every method of `model`'s class that a caller can enter it through,
    the ones Qwix wraps: `__call__` and each public function the class or
    its bases define below `nn.Module`, `setup` aside. Qwix's interception is
    non-recursive over dynamic extent, so `__call__` reaching
    `hidden_states` quantizes once."""
    owned = [vars(layer) for layer in type(model).__mro__
             if issubclass(layer, nn.Module) and layer is not nn.Module]
    names = {name for space in owned for name, value in space.items()
             if (name == "__call__" or not name.startswith("_")) and name != "setup"
             and inspect.isfunction(value)}
    return tuple(sorted(names))


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
    """The weight calibration method, in the form Qwix parses (`absmax,0.8`,
    for example). Dew keeps no copy of its grammar: Qwix refuses a method it
    does not know where it first quantizes a matched weight, not when this
    spec is built."""
    tile_size: int | None = None
    """The number of elements per tile for sub-channel scales along the contraction axis.

    Unset keeps per-channel scales, which are coarser and cheaper."""
    bwd_qtype: QuantizedDtype | None = None
    """The dtype gradients quantize to in the backward pass; unset keeps
    them in the compute dtype."""
    # Qwix draws the stream in qwix/_src/providers/qt.py:361.
    bwd_stochastic_rounding: Rounding | None = None
    """The stochastic rounding mode for the quantized gradients; unset rounds deterministically.

    Qwix draws from the `stochastic_rounding` RNG stream, which Dew's
    objectives fold off each step's key (`dew.objectives.base.training_rngs`);
    a custom objective passes it the same way."""

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

    def rule_for(self, scope: Sequence[str]) -> int | None:
        """The index of the pattern that quantizes the module at `scope` (its path of names), the
        first whose regex fully matches the `/`-joined path as Qwix matches it, or None for a module
        no pattern selects."""
        joined = "/".join(scope)
        return next((index for index, pattern in enumerate(self.patterns) if re.fullmatch(pattern, joined)),
                    None)

    def apply(self, model: nn.Module) -> nn.Module:
        """Return a copy of `model` whose trunk matmuls train in this spec's dtype.

        The copy is an instance of Qwix's subclass of the model's class, with
        each of its `entry_methods` wrapped. Everything
        the registry, the objective and the checkpoint code read from the
        model therefore still works. Invalid settings were already refused
        when the spec was constructed, all but the calibration method, which
        Qwix parses where it first quantizes a matched weight. Without Qwix
        installed, the call raises an error that names the extra.
        """
        return _qwix_wrapped(model, _providers()[0](_rules(self, training=True)), self)


@runtime_checkable
class QuantizedClass(Protocol):
    """A model class Qwix made for a `Quantization` (`_quantized`), which `_dew_quantization` names for the
    programs that run it (`dew.nn.kernels.options.compiler_options` reads its dtypes,
    `dew.inference.projections` its rules); a wrapper's class over it inherits it."""

    _dew_quantization: ClassVar[Quantization]


def _qwix_wrapped(model: nn.Module, provider: QuantizationProvider, wrapper: Quantization) -> nn.Module:
    """`model` with the matmuls of its `entry_methods` under Qwix's `provider`, in
    the class Qwix makes for the call, which names `wrapper` and the class it
    wraps (`dew.registry.WrapperClass`) for the model's record.

    Qwix holds one provider per model: it makes its class over the class the
    model's last Qwix call wrapped (qwix/_src/model.py:116-119), so a second
    call would drop the first quantization and every wrapper made over it,
    an adapter included. A model that already computes under Qwix is refused.
    """
    held = next((layer for layer in type(model).__mro__ if "_unquantized_type" in vars(layer)), None)
    if held is not None:
        raise ValueError(f"{type(model).__name__} already computes under a Qwix quantization, which a "
                         f"second one would replace; quantize the model once")
    wrapped = _quantized(model, provider, wrapper)
    layer: Any = type(wrapped)
    layer._dew_wrapped = type(model)
    layer._dew_wrapper = staticmethod(lambda: wrapper)
    return wrapped


def _quantized(model: nn.Module, provider: QuantizationProvider, spec: Quantization) -> nn.Module:
    """`model` with the matmuls of its `entry_methods` under Qwix's `provider`, in the class Qwix
    makes for the call, a `QuantizedClass` of `spec`,
    each method tracing what it calls under the key `QUANTIZED_TRACE` gives the provider."""
    methods = entry_methods(model)
    quantized = _qwix().quantize_model(model, provider, methods=methods)
    layer = type(quantized)
    layer._dew_quantization = spec
    key = (type(provider).__name__, spec)
    for name in methods:
        setattr(layer, name, _keyed(getattr(layer, name), key))
    return quantized


def _keyed[**Arguments, Result](method: Callable[Arguments, Result],
                                 key: Hashable) -> Callable[Arguments, Result]:
    """`method` run under `key` in `QUANTIZED_TRACE`."""
    @functools.wraps(method)
    def keyed(*args: Arguments.args, **kwargs: Arguments.kwargs) -> Result:
        with QUANTIZED_TRACE(key):
            return method(*args, **kwargs)
    return keyed


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


QUANTIZATION = "quantization"
"""The collection that holds a projection site's checkpoint quantization, at
the site's own scope: every view of the variables (a scanned or staged
stack, a packed group, a pipeline) carries it with the weights it scales."""

InputKind = Literal["fp8", "nvfp4"]
"""A checkpoint input quantizer's format: ModelOpt's static FP8, or NVFP4."""

InputFormat = Literal["compressed-tensors", "modelopt"]
"""Whose NVFP4 local-scale arithmetic a checkpoint's inputs follow."""

INPUT_SCALES: dict[InputKind, str] = {"fp8": "fp8_input_scale", "nvfp4": "nvfp4_input_global_scale"}
"""Each input format's leaf in `QUANTIZATION`: ModelOpt's static FP8
multiplier (amax/448) or the NVFP4 global scale, one float32 scalar a site."""


FROZEN_FP8 = ("fp8_forward", "fp8_forward_scale", "fp8_dgrad", "fp8_dgrad_scale")
"""A frozen FP8 site's leaves in `QUANTIZATION`: its `[in, out]` kernel's E4M3
codes for the forward product, `[out, in]`, and for the input gradient,
`[in, out]`, each beside one float32 scale a row (`frozen_fp8_leaves`)."""


def frozen_fp8_leaves(kernel: jax.typing.ArrayLike) -> dict[str, np.ndarray]:
    """A frozen `[in, out]` kernel packed once for both products it takes part
    in, each direction's rows scaled on their own by the one FP8 block cast
    (`dew.interop.codecs.quantize_fp8_blocks` over `(1, width)` blocks): the
    forward's `[out, in]` rows contract the input, the input gradient's
    `[in, out]` rows the output's cotangent. A forward row's scale is a
    contracting-axis scale of the other direction, so the two are quantized
    apart. The two E4M3 copies replace the kernel in the train state
    (`frozen_fp8_kernel` decodes it back), two bytes a parameter, a bf16
    kernel's, so the gain over a bf16 base is the products' rate."""
    from dew.interop.codecs import quantize_fp8_blocks

    kernel = np.asarray(kernel, np.float32)
    if kernel.ndim != 2:
        raise ValueError(f"a frozen FP8 site packs a two-axis [in, out] kernel, got shape {kernel.shape}")
    forward, forward_scale = quantize_fp8_blocks(kernel.T, (1, kernel.shape[0]))
    dgrad, dgrad_scale = quantize_fp8_blocks(kernel, (1, kernel.shape[1]))
    return dict(zip(FROZEN_FP8, (forward, forward_scale, dgrad, dgrad_scale), strict=True))


def frozen_fp8_kernel(leaves: Mapping[str, jax.typing.ArrayLike]) -> jax.Array:
    """The `[in, out]` kernel a frozen FP8 site's `leaves` hold, in float32:
    each forward code widened times its row's scale (`frozen_fp8_leaves`), the
    rows the forward product multiplies by, for a merge, an export or a head's
    vocabulary rows (`CausalTransformer.applied_kernel`), traced or not."""
    forward = jnp.asarray(leaves[FROZEN_FP8[0]])
    return (forward.astype(jnp.float32) * jnp.asarray(leaves[FROZEN_FP8[1]], jnp.float32)).T


def _fp8_rows(values: jax.Array, rows: jax.Array, row_scales: jax.Array) -> jax.Array:
    """`values @ rows.T`, `[M, K]` by E4M3 `[N, K]`, in float32: `values`'
    rows cast by the FP8 block cast, the codes widened in the dot and both
    rows' scales applied after it, the form XLA's GPU compiler rewrites to an
    FP8 GEMM (cuBLASLt)."""
    from dew.interop.codecs import fp8_block_cast

    codes, scales = fp8_block_cast(values, (1, values.shape[1]))
    product = jax.lax.dot_general(codes.astype(jnp.float32), rows.astype(jnp.float32),
                                  (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    return product * scales * row_scales.T


@jax.custom_vjp
def frozen_fp8_product(x: jax.Array, forward: jax.Array, forward_scale: jax.Array, dgrad: jax.Array,
                       dgrad_scale: jax.Array) -> jax.Array:
    """`x @ kernel` in float32, `[M, in]` by a frozen kernel packed as
    `frozen_fp8_leaves`: FP8 products forward and for `x`'s gradient, and no
    gradient for the kernel, which is frozen."""
    return _fp8_rows(x, forward, forward_scale)


def _frozen_fp8_forward(x, forward, forward_scale, dgrad, dgrad_scale):
    # The input's dtype rides as an empty array, a residual JAX can hold.
    return _fp8_rows(x, forward, forward_scale), (jnp.zeros((0,), x.dtype), dgrad, dgrad_scale)


def _frozen_fp8_backward(residuals, cotangent):
    like, dgrad, dgrad_scale = residuals
    return _fp8_rows(cotangent, dgrad, dgrad_scale).astype(like.dtype), None, None, None, None


frozen_fp8_product.defvjp(_frozen_fp8_forward, _frozen_fp8_backward)


def frozen_fp8_dense(module: nn.Module, inputs: jax.Array) -> jax.Array:
    """The bound Dense `module`'s output from its frozen FP8 leaves, its
    kernel unread: the product (`frozen_fp8_product`) over the input's
    trailing axis, its bias added in float32, in the dtype the Dense
    computes in."""
    leaves = [module.get_variable(QUANTIZATION, leaf) for leaf in FROZEN_FP8]
    product = frozen_fp8_product(inputs.reshape(-1, inputs.shape[-1]), *leaves)
    if module.use_bias:
        product = product + module.get_variable("params", "bias").astype(jnp.float32)
    dtype = module.dtype or jnp.result_type(inputs.dtype, module.param_dtype)
    return product.astype(dtype).reshape(*inputs.shape[:-1], product.shape[-1])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class InputQuantizer:
    """One projection site's checkpoint input quantizer: its stored scale and
    the format's arithmetic, which `qdq` applies to the site's input.

    `fp8` and `nvfp4` build one from a file's value on the host and refuse
    an invalid one; a site builds one from the scale its variables hold,
    which is what traces under a scan or a jit.
    """

    scale: jax.Array
    kind: InputKind = dataclasses.field(metadata={"static": True})
    e4m3_scale: bool = dataclasses.field(default=True, metadata={"static": True})
    """Whether NVFP4 rounds its local scales to E4M3."""
    format: InputFormat = dataclasses.field(default="compressed-tensors", metadata={"static": True})
    """Whose NVFP4 local-scale arithmetic the input follows."""

    @classmethod
    def fp8(cls, scale: float) -> InputQuantizer:
        """ModelOpt's static E4M3 input multiplier, amax/448 in its exported files."""
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"ModelOpt FP8 input_scale must be positive and finite, got {scale}")
        return cls(jnp.asarray(scale, jnp.float32), "fp8")

    @classmethod
    def nvfp4(cls, global_scale: float, e4m3_scale: bool,
              format: InputFormat = "compressed-tensors") -> InputQuantizer:
        """One checkpoint Linear's stored global scale and published local-scale arithmetic."""
        if (not math.isfinite(global_scale) or global_scale < 0
                or (global_scale == 0 and format == "compressed-tensors")):
            raise ValueError(f"NVFP4 input_global_scale must be positive and finite, got {global_scale}")
        scheme = InputQuantization(e4m3_scale, format)
        return cls(jnp.asarray(global_scale, jnp.float32), "nvfp4", scheme.e4m3_scale, scheme.format)

    def qdq(self, x: jax.Array) -> jax.Array:
        """`x` quantized to the format and back, with the straight-through gradient."""
        return fp8_input_qdq(x, self) if self.kind == "fp8" else nvfp4_input_qdq(x, self)


def fp8_input_qdq(x: jax.Array, spec: InputQuantizer) -> jax.Array:
    """ModelOpt's static FP8 input QDQ using Qwix's E4M3 codes and stored scale.

    tensor_quant.py:_fp8_eager widens before scaling and returns the input
    dtype. The reference reconstructs amax from the file's amax/448,
    computes 448/amax and its reciprocal in fp32, casts directly to E4M3
    RN, and dequantizes by that reciprocal. Keeping both divisions holds
    the scale's last bits to the author's fake quantizer. Backward is Dew's STE.
    """
    from dew.nn.fake_quant import straight_through

    qwix = _qwix()
    values = jax.lax.stop_gradient(x).astype(jnp.float32)
    amax = jnp.asarray(spec.scale, jnp.float32) * jnp.float32(448)
    safe = jnp.where(amax <= jnp.float32(2 ** -24), 1, amax)
    # torch's scalar/tensor reverse divide is reciprocal then multiply.
    multiplier = correctly_rounded_divide(jnp.float32(1), safe) * jnp.float32(448)
    scale = correctly_rounded_divide(jnp.float32(1), multiplier)
    # The composed GPU forward must round this product before the FP8 cast.
    quotients = jax.lax.optimization_barrier(values * multiplier)
    broadcast_shape = (1,) * values.ndim
    quantized = _cast(qwix, quotients, jnp.float8_e4m3fn)
    quantized = quantized.replace(scale=scale.reshape(broadcast_shape))
    rounded = qwix.dequantize(quantized)
    return straight_through(x, rounded)


def _cast(qwix: ModuleType, quotients: jax.Array, qtype: jax.typing.DTypeLike):
    """`quotients`, already divided by their scales, cast to `qtype` as a Qwix QArray of unit scale:
    a fixed range of the type's own bound gives a scale of exactly one (448 over E4M3's 448, 6 over
    E2M1's 6), so the cast is the type's rounding alone and the caller sets the real scale. NVFP4
    names its 16-wide blocks along the last axis, which Qwix requires of it and a fixed range,
    always per tensor, leaves unread."""
    if qtype == jnp.float8_e4m3fn:
        return qwix.quantize(quotients, qtype, calibration_method="fixed,448")
    return qwix.quantize(quotients, qtype, tiled_axes={quotients.ndim - 1: 16}, calibration_method="fixed,6")


def nvfp4_input_qdq(x: jax.Array, spec: InputQuantizer) -> jax.Array:
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

    qwix = _qwix()
    if x.shape[-1] % 16:
        raise ValueError(f"NVFP4 input width must be divisible by 16, got {x.shape}")
    values = jax.lax.stop_gradient(x)
    groups = values.reshape(*x.shape[:-1], -1, 16)
    if spec.format == 'modelopt':
        return _modelopt_input_qdq(x, spec, qwix)
    largest = jnp.max(jnp.abs(groups), -1)
    divisor = jnp.full(largest.shape, 6, jnp.float32)
    # CT's local division must round before global scaling, including its bf16 cast.
    scale = jnp.asarray(
        rounded_operand(correctly_rounded_divide(largest.astype(jnp.float32), divisor), x.dtype), jnp.float32)
    overall = jnp.full(scale.shape, spec.scale, jnp.float32)
    scale = scale * overall
    if spec.e4m3_scale:
        scale = jnp.clip(scale, 0, 448).astype(jnp.float8_e4m3fn).astype(jnp.float32)
    eps = jnp.finfo(jnp.float8_e4m3fn if spec.e4m3_scale else jnp.float32).eps
    scale = jnp.where(scale == 0, jnp.asarray(eps, jnp.float32), scale)
    scale = correctly_rounded_divide(scale, overall)
    expanded = jnp.broadcast_to(scale[..., None], groups.shape).reshape(x.shape)
    quotients = correctly_rounded_divide(values.astype(jnp.float32), expanded)
    quantized = _cast(qwix, quotients, 'nvfp4').replace(scale=expanded)
    rounded = qwix.dequantize(quantized)
    # CT adds its symmetric zero point before casting, so exact -0 is +0.
    rounded = jnp.where(values == 0, jnp.zeros_like(rounded), rounded)
    return straight_through(x, rounded)


def _modelopt_input_qdq(x: jax.Array, spec: InputQuantizer, qwix: ModuleType) -> jax.Array:
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
    global_scale = jnp.where(spec.scale > 0, spec.scale, 1e-12).astype(jnp.float32)
    denominator = jnp.full(largest.shape, 6, jnp.float32) * global_scale
    normalized = largest * correctly_rounded_divide(jnp.ones_like(largest), denominator)
    saturated = jnp.minimum(normalized, 448)
    scales = saturated.astype(jnp.float8_e4m3fn).astype(jnp.float32) * global_scale
    scales = jnp.where(scales >= 1e-5, scales, 1)
    spread = jnp.broadcast_to(scales[..., None], groups.shape).reshape(x.shape)
    quotients = values * correctly_rounded_divide(jnp.ones_like(values), spread)
    rounded = qwix.dequantize(_cast(qwix, quotients, 'nvfp4').replace(scale=spread))
    return straight_through(x, jnp.where(rounded == 0, jnp.zeros_like(rounded), rounded))


@dataclasses.dataclass(frozen=True)
class InputQuantization:
    """The quantization at a model's Linear sites: a checkpoint's input
    quantization, which a model loaded from a ModelOpt or compressed-tensors
    FP8 or NVFP4 checkpoint computes under (`dew.interop.Pretrained`), and a
    LoRA run's frozen FP8 base (`dew.lora.LoRA.apply`'s `fp8_base`); a run
    records it.

    Each quantized Linear holds its scale, or its frozen FP8 kernel
    (`FROZEN_FP8`), in its own `QUANTIZATION` variables, so a scanned,
    staged or packed view of the layers keeps it, and a site without one
    computes as it is. This record holds what the files declare (NVFP4's
    local-scale rounding and arithmetic, and which sites take their input
    in which format) and which frozen sites multiply in FP8, as measured on
    `measured_on`. It is not a `Quantization`: nothing trains quantized, and
    the weights already hold the values the source reader decoded.
    """

    e4m3_scale: bool = True
    """Whether NVFP4 sites round their local scales to E4M3."""
    format: InputFormat = "compressed-tensors"
    """Whose NVFP4 local-scale arithmetic the sites follow."""
    sites: tuple[tuple[str, InputKind], ...] = ()
    """Each quantized Linear's scope in the stored variables tree, `/`-joined,
    and its format: what `init` declares, so a model rebuilt from this record
    initializes the scales a checkpoint restores. The values live in the
    variables alone."""
    frozen_fp8: tuple[str, ...] = ()
    """The frozen Dense sites whose products run in FP8 (`frozen_fp8_dense`),
    by their scope in the stored variables tree, `/`-joined: those the run
    names, whose FP8 forward and input gradient it measured faster than
    bf16's on `measured_on`. LoRA packs their kernels from the base it
    freezes, the one place a kernel is known never to train."""
    measured_on: str | None = None
    """The GPU generation `frozen_fp8` was measured on (`sm90`): where its
    sites measured faster in FP8, which another generation may not repeat.
    The product is ordinary arithmetic on the stored codes and scales and
    runs on any generation."""

    def __post_init__(self) -> None:
        if self.format == "modelopt" and not self.e4m3_scale:
            raise ValueError("ModelOpt NVFP4 always rounds its local scales to E4M3")
        object.__setattr__(self, "frozen_fp8", tuple(self.frozen_fp8))
        if self.frozen_fp8 and self.measured_on is None:
            raise ValueError("frozen FP8 sites name the GPU generation they were measured on (measured_on)")

    def layer_sites(self, layer: str) -> tuple[tuple[str, InputKind], ...]:
        """The quantized sites under the stored scope `layer` (`layers_3`),
        relative to it, with their formats: what two otherwise like layers
        must share to run as one scanned body or one stage position."""
        prefix = layer + "/"
        return tuple((path.removeprefix(prefix), kind) for path, kind in self.sites
                     if path.startswith(prefix))

    def declared(self) -> Variables:
        """The `QUANTIZATION` collection `sites` declare, every scale 1 until
        a checkpoint's values replace it."""
        return unflatten_dict({(*path.split("/"), INPUT_SCALES[kind]): np.ones((), np.float32)
                               for path, kind in self.sites})

    def quantizer(self, module: nn.Module) -> InputQuantizer | None:
        """The bound `module`'s input quantizer, from the scale its variables
        hold, or None for a module that holds none."""
        held: list[InputKind] = [kind for kind, leaf in INPUT_SCALES.items()
                                 if module.has_variable(QUANTIZATION, leaf)]
        if not held:
            return None
        if len(held) > 1:
            raise ValueError(f"{'/'.join(module.path)} holds {held} input scales; a Linear reads one")
        scale = module.get_variable(QUANTIZATION, INPUT_SCALES[held[0]])
        return InputQuantizer(scale, held[0], self.e4m3_scale, self.format)

    def joined(self, model: nn.Module) -> nn.Module:
        """`model` computing under this record's frozen FP8 sites, and under
        the input quantization it already computes under, if any: one
        wrapper, so one interceptor, serves both."""
        from dew.registry import rebuilt, wrapper_layers

        _, layers = wrapper_layers(type(model))
        held = layers[0]._dew_wrapper() if layers else None
        if not isinstance(held, InputQuantization):
            return self.apply(model)
        unwrapped = rebuilt(layers[0]._dew_wrapped, model)
        joined = dataclasses.replace(held, frozen_fp8=self.frozen_fp8, measured_on=self.measured_on)
        return joined.apply(unwrapped)

    def packed(self, frozen: Variables) -> Variables:
        """The `QUANTIZATION` leaves of `frozen_fp8`, packed from the kernels
        the `frozen` tree holds at their scopes (`frozen_fp8_leaves`)."""
        return unflatten_dict({(*site.split("/"), leaf): value for site in self.frozen_fp8
                               for leaf, value in frozen_fp8_leaves(_site_kernel(frozen, site)).items()})

    def apply(self, model: nn.Module) -> nn.Module:
        """Return `model` quantizing each Linear's input to its declared format
        and back with its scales, before the layer's dot, and computing each
        Linear that holds a frozen FP8 kernel from it (`frozen_fp8_dense`).

        Both run at the call of every module that holds a scale or a frozen
        kernel, under an interceptor `apply` installs, so every entry method
        of the model runs them, inside an adapter's branch. An input's dot
        is otherwise unchanged, so its dtype, accumulation precision, output
        sharding and the bias placement stay as the model defines them.
        """
        from dew.registry import rebuilt

        base, record = type(model), self
        if any("_dew_input_quantization" in vars(layer) for layer in base.__mro__):
            raise ValueError(f"{base.__name__} already computes under its checkpoint's input quantization")

        def quantize_inputs(next_fun, args, kwargs, context):
            module = context.module
            if context.method_name == "applied_kernel":
                # A model reads the kernel its submodule multiplies by
                # (`CausalTransformer.applied_kernel`): a frozen FP8 site's
                # is its codes decoded, which an adapter's branch updates.
                child = getattr(module, args[0] if args else kwargs["name"])
                if not child.has_variable(QUANTIZATION, FROZEN_FP8[0]):
                    return next_fun(*args, **kwargs)
                return frozen_fp8_kernel({leaf: child.get_variable(QUANTIZATION, leaf)
                                          for leaf in FROZEN_FP8})
            if context.method_name == "nonlinear_input":
                # A layer asks whether its submodule is linear in its input
                # (`MultiHeadLatentAttention.nonlinear_input`): one that
                # rounds its input or multiplies in FP8 is not.
                name = args[0] if args else kwargs["name"]
                child = getattr(module, name)
                return next_fun(*args, **kwargs) or (record.quantizer(child) is not None
                                                     or child.has_variable(QUANTIZATION, FROZEN_FP8[0]))
            calls = context.method_name == "__call__"
            quantizer = record.quantizer(module) if calls else None
            frozen = calls and module.has_variable(QUANTIZATION, FROZEN_FP8[0])
            if quantizer is None and not frozen:
                return next_fun(*args, **kwargs)
            # A Linear's input arrives first or by its parameter's name.
            inputs = args[0] if args else kwargs["inputs"]
            if not _trailing_contraction(module, inputs.ndim):
                raise ValueError(f"{'/'.join(module.path)} holds an input scale or a frozen FP8 kernel, "
                                 "which a Linear contracting its input's trailing axis alone reads")
            if quantizer is not None:
                inputs = quantizer.qdq(inputs)
            if frozen:
                return frozen_fp8_dense(module, inputs)
            if args:
                return next_fun(inputs, *args[1:], **kwargs)
            return next_fun(**{**kwargs, "inputs": inputs})

        class InputQuantized(base):
            _dew_input_quantization: ClassVar[bool] = True
            _dew_wrapped: ClassVar[type[nn.Module]] = base

            @staticmethod
            def _dew_wrapper() -> InputQuantization:
                return record

            @nn.nowrap
            def input_quantizer(self, module: nn.Module) -> InputQuantizer | None:
                """The quantizer `module`'s input takes (`dew.nn.protocols.QuantizesInputs`)."""
                return record.quantizer(module)

            def apply(self, variables, *args, **kwargs):
                with nn.intercept_methods(quantize_inputs):
                    return super().apply(variables, *args, **kwargs)

            def init_with_output(self, *args, **kwargs):
                output, variables = super().init_with_output(*args, **kwargs)
                return output, ({**variables, QUANTIZATION: record.declared()} if record.sites else variables)

        InputQuantized.__name__ = InputQuantized.__qualname__ = base.__name__
        return rebuilt(InputQuantized, model)


def _site_kernel(tree: Variables, site: str) -> jax.typing.ArrayLike:
    """The kernel `tree` holds at the `/`-joined scope `site`, refused by name where it holds none."""
    node: object = tree
    for name in (*site.split("/"), "kernel"):
        if not isinstance(node, Mapping) or name not in node:
            raise ValueError(f"frozen FP8 site {site} is not a Dense of this model's frozen tree")
        node = node[name]
    if not isinstance(node, jax.Array | np.ndarray):
        raise ValueError(f"frozen FP8 site {site} holds no kernel array")
    return node


def _trailing_contraction(module: nn.Module, ndim: int) -> bool:
    """Whether `module` is a Linear that contracts its rank-`ndim` input's trailing axis alone."""
    match module:
        case nn.Dense():
            return True
        case nn.DenseGeneral(axis=axis, batch_dims=batch_dims):
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            return not batch_dims and sorted(a % ndim for a in axes) == [ndim - 1]
    return False


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


def _widened(operands, preferred: jax.typing.DTypeLike | None) -> tuple:
    """`operands` with each float array narrower than `preferred` widened to
    it, which holds its values exactly; a quantized weight passes as it is."""
    if preferred is None:
        return tuple(operands)
    wide = jnp.dtype(preferred)
    return tuple(operand.astype(wide) if isinstance(operand, jax.Array | np.ndarray)
                 and jnp.issubdtype(operand.dtype, jnp.floating) and operand.dtype.itemsize < wide.itemsize
                 else operand for operand in operands)


def _honoured(out: jax.Array, preferred: jax.typing.DTypeLike | None, out_sharding) -> jax.Array:
    """A quantized product in the result dtype and sharding its call asked for."""
    out = out if preferred is None else out.astype(preferred)
    return out if out_sharding is None else jax.lax.with_sharding_constraint(out, out_sharding)


def _real(*dtypes: jax.typing.DTypeLike) -> bool:
    """Whether no dtype is complex; Qwix quantizes real values only."""
    return not any(jnp.issubdtype(dtype, jnp.complexfloating) for dtype in dtypes)


def _refuse_grouped_on_gpu() -> None:
    """Refuse a grouped convolution with quantized activations on a GPU,
    where jax 0.11.2 computed it wrong or failed to compile it (the error
    says where; in int8 on the A100 the 176M text-to-image model's CLIP score
    fell from 0.247 to 0.137). The refusal covers every GPU, dtype and group
    width until an H100 is measured."""
    if backend() == "gpu":
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
    qwix = _qwix()

    def quantized(arg) -> bool:
        return isinstance(arg, qwix.QArray)

    def scaled(*args: P.args, **kwargs: P.kwargs) -> jax.Array:
        operands = [arg for arg in jax.tree.leaves(args, is_leaf=quantized) if quantized(arg)]
        if len(operands) < 2:
            return op(*args, **kwargs)
        # The product's dtype as Qwix's rule gives it (qarray.get_accumulator_and_result_type): its
        # operands' scales', an 8-bit or narrower one read as bfloat16, an integer one as int32.
        dtype = jnp.result_type(*(
            (jnp.int32 if "int" in operand.scale.dtype.name else jnp.bfloat16)
            if operand.scale.dtype.itemsize <= 1 else operand.scale.dtype for operand in operands))
        rescaled = jax.tree.map(
            lambda arg: arg.astype(jnp.float32) if quantized(arg) else arg,
            args,
            is_leaf=quantized,
        )
        return op(*rescaled, **kwargs).astype(dtype)

    return scaled


@functools.cache
def _grouped_convolution_gradient() -> type:
    """Qwix's quantized-training provider with Dew's gradient for a grouped
    convolution; importing Qwix here keeps it optional."""
    qwix = _qwix()
    conv_general_qt = _qwix("qwix._src.core.conv_general_qt")

    @functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(2, 11)))
    def grouped_conv_qt(lhs, rhs, config, window_strides, padding, lhs_dilation, rhs_dilation,
                        dimension_numbers, feature_group_count, batch_group_count, out_sharding):
        return conv_general_qt.conv_general_qt_fwd(
            lhs, rhs, config, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers,
            feature_group_count, batch_group_count, out_sharding)[0]

    def straight_through(config, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers,
                         feature_group_count, batch_group_count, out_sharding, residuals, g):
        operands = [qwix.dequantize(operand) if isinstance(operand, qwix.QArray) else operand
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
            """The convolution under the call's contract, as `ComplexInFloat` keeps a matmul's: a
            quantized one returns the result dtype it asks for, its operands widened to it first,
            which is exact for the narrower floats it accumulates."""
            rule, _ = self._get_current_rule_and_op_id("conv_general_dilated", only_rule=True)
            quantized = rule is not None and rule.weight_qtype is not None
            returned = jnp.dtype(preferred_element_type or lhs.dtype)
            if quantized:
                lhs, rhs = _widened((lhs, rhs), preferred_element_type)
            convolve = functools.partial(
                super().conv_general_dilated,
                rhs=rhs, window_strides=window_strides, padding=padding, lhs_dilation=lhs_dilation,
                rhs_dilation=rhs_dilation, dimension_numbers=dimension_numbers,
                feature_group_count=feature_group_count, batch_group_count=batch_group_count,
                precision=precision, preferred_element_type=preferred_element_type, out_sharding=out_sharding)
            if feature_group_count == 1 or rule is None or rule.act_qtype is None:
                out = convolve(lhs=lhs)
                return _honoured(out, preferred_element_type, None) if quantized else out
            _refuse_grouped_on_gpu()
            numbers = jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
            scales = _group_scales(lhs, numbers, feature_group_count).astype(jnp.float32)
            batch, feature = numbers.lhs_spec[:2]
            out = convolve(lhs=lhs.astype(jnp.float32)
                           / _per_feature(scales, lhs.ndim, batch, feature, lhs.shape[feature]))
            batch, feature = numbers.out_spec[:2]
            return (out * _per_feature(scales, out.ndim, batch, feature, out.shape[feature])).astype(returned)

    return GroupScaledConvolution


@functools.cache
def _complex_in_float() -> type:
    """The providers' shared base (`ComplexInFloat`): a complex matmul unquantized, and the call's contract
    honoured on a quantized one (`_providers`)."""
    qwix = _qwix()

    class ComplexInFloat(qwix.QuantizationProvider):
        """Passes a matmul with a complex operand through unquantized, and
        honours the call's contract on a quantized one, which Qwix 0.1.8
        drops (qwix/_src/providers/qt.py:96-97,130-132, ptq.py:177-180,244):
        a result dtype by widening the operands to it, exact for the
        narrower floats it accumulates, and an output sharding by
        constraining the result where the provider drops it. The functions
        `UNQUANTIZED` names run unquantized, through the intercept map; any
        other matmul a rule matches quantizes, whatever precision it pins."""

        keeps_out_sharding: ClassVar[bool] = False

        def get_intercept_map(self):
            # A handler that calls its function again runs the original with this provider's
            # interception off for the thread (qwix/_src/interception.py:361-387), so none of
            # its products quantize.
            functions = {name: getattr(importlib.import_module(module), function)
                         for name in UNQUANTIZED for module, _, function in [name.rpartition(".")]}
            return {**super().get_intercept_map(), **functions}

        def _matched(self, op: str) -> bool:
            rule, _ = self._get_current_rule_and_op_id(op, only_rule=True)
            return rule is not None and rule.weight_qtype is not None

        def dot_general(self, lhs, rhs, dimension_numbers, precision=None,
                        preferred_element_type=None, *, out_sharding=None):
            if not _real(lhs.dtype, rhs.dtype):
                return jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=precision,
                                           preferred_element_type=preferred_element_type,
                                           out_sharding=out_sharding)
            if not self._matched("dot_general"):
                return super().dot_general(lhs, rhs, dimension_numbers, precision,
                                           preferred_element_type, out_sharding=out_sharding)
            kept = out_sharding if self.keeps_out_sharding else None
            out = super().dot_general(*_widened((lhs, rhs), preferred_element_type), dimension_numbers,
                                      out_sharding=kept)
            return _honoured(out, preferred_element_type, None if kept is not None else out_sharding)

        def einsum(self, einsum_str, *operands, **kwargs):
            # Qwix 0.1.8 quantizes a two-operand einsum alone and raises on any other
            # (qwix/_src/providers/qt.py:119-121), which runs here as the unwrapped model runs it.
            if len(operands) != 2 or not _real(*(operand.dtype for operand in operands)):
                return jnp.einsum(einsum_str, *operands, **kwargs)
            if not self._matched("einsum"):
                return super().einsum(einsum_str, *operands, **kwargs)
            preferred = kwargs.pop("preferred_element_type", None)
            out_sharding = kwargs.pop("out_sharding", None)
            kwargs.pop("precision", None)
            out = super().einsum(einsum_str, *_widened(operands, preferred), **kwargs)
            return _honoured(out, preferred, out_sharding)

    return ComplexInFloat


@functools.cache
def _serving_provider() -> type:
    """The serving provider (`_providers`): Qwix's post-training quantization with Dew's grouped
    convolution, its products scaled in float32 (`_scaled_in_float32`), a quantized kernel no rule matches
    multiplied by its own recorded recipe, and a quantized kernel's scales promoted with the module's."""
    qwix = _qwix()

    class PtqProvider(_group_scaled_convolution(), _complex_in_float(), qwix.PtqProvider):
        keeps_out_sharding: ClassVar[bool] = True  # ptq.py:180 passes it on

        def __init__(self, rules: Sequence[object]) -> None:
            super().__init__(
                rules, _dot_general_fn=_scaled_in_float32(qwix.dot_general),
                _einsum_fn=_scaled_in_float32(qwix.einsum),
                _conv_general_dilated_fn=_scaled_in_float32(qwix.conv_general_dilated))

        def _recorded(self, op: str, weight) -> bool:
            """Whether `weight` is a stored quantized kernel that reaches `op` with no rule matching
            it, as an unpacked task's member does under a rule that names its packed group
            (`dew.inference.projections.unpacked`): it multiplies by the recipe it records, its
            activation unquantized, where every rule leaves activations so; a rule that quantizes
            them records no recipe a member could take, which refuses."""
            if not isinstance(weight, Annotated) or self._matched(op):
                return False
            if any(rule.act_qtype is not None for rule in self._rules):
                raise ValueError(f"a quantized kernel reached a {op} no rule matches, and the rules quantize "
                                 "activations, whose recipe the kernel does not record; name its members in "
                                 "the patterns")
            return True

        def dot_general(self, lhs, rhs, dimension_numbers, precision=None,
                        preferred_element_type=None, *, out_sharding=None):
            if not self._recorded("dot_general", rhs):
                return super().dot_general(lhs, rhs, dimension_numbers, precision, preferred_element_type,
                                           out_sharding=out_sharding)
            out = self._dot_general_fn(*_widened((lhs, rhs.array), preferred_element_type),
                                       dimension_numbers, out_sharding=out_sharding)
            return _honoured(out, preferred_element_type, None)

        def einsum(self, einsum_str, *operands, **kwargs):
            if len(operands) != 2 or not self._recorded("einsum", operands[1]):
                return super().einsum(einsum_str, *operands, **kwargs)
            preferred = kwargs.pop("preferred_element_type", None)
            out_sharding = kwargs.pop("out_sharding", None)
            out = self._einsum_fn(einsum_str, *_widened((operands[0], operands[1].array), preferred))
            return _honoured(out, preferred, out_sharding)

        def promote_dtype(self, *args, **kwargs):
            # Qwix 0.1.8 passes a quantized kernel through, and its wrapper's own `astype` raises on
            # the QArray inside it (`flax_util.update_boxed` accepts boxes and arrays only): the
            # QArray's scales take the dtype, wherever the wrapper holds it.
            promoted = super().promote_dtype(*args, **kwargs)
            dtype = kwargs.get("dtype")
            if dtype is None:
                return promoted

            def quantized(value) -> bool:
                return isinstance(value, qwix.QArray)

            return jax.tree.map(lambda value: value.astype(dtype) if quantized(value) else value,
                                promoted, is_leaf=quantized)

    return PtqProvider


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

    class QtProvider(_group_scaled_convolution(), _complex_in_float(), _grouped_convolution_gradient()):
        pass

    return QtProvider, _serving_provider()


@runtime_checkable
class Annotated(Protocol):
    """A serving tree's quantized kernel as Qwix's post-training quantization annotates it: the QArray
    (`array`) with the recipe it was quantized by (`how`), a dataclass of those fields."""

    array: QArray
    how: object


def split_columns(leaf, widths: Sequence[int], *, axis: int = -1) -> list:
    """`leaf` split into one leaf a member along `axis` by `widths`, a packed projection's kernel
    taken back into its members' (`dew.inference.projections.unpacked`): a plain array, host or
    device, as `np.split` or `jnp.split` would; a Qwix QArray with its values split and each scale
    and zero point split as its own axis is, shared (one entry) kept whole by every member, per
    channel split at the same columns, per tile at the tiles' (a member that would cut a tile is
    refused); and the serving tree's wrapper of one (Qwix's annotated array, `array` its field) as
    the QArray inside it, its recipe kept, where no other field of it holds an array that would need
    a split of its own. A leaf this process can materialize whole (fully addressable, or replicated
    with an addressable copy) splits on the host and returns host arrays; any other splits where it
    lives. The widths are positive and sum to `leaf`'s size along `axis`, or the split is refused."""
    edges = np.cumsum(widths)[:-1].tolist()

    def checked(size: int):
        if not widths or min(widths) < 1 or sum(widths) != size:
            raise ValueError(f"members {list(widths)} do not partition an axis of {size}")

    def arrays(value, ratio: int = 1):
        if value is None:
            return [None] * len(widths)
        if isinstance(value, jax.Array) and not (value.is_fully_addressable or (
                value.is_fully_replicated and value.sharding.has_addressable_devices)):
            return jnp.split(value, [edge // ratio for edge in edges], axis=axis)
        return np.split(np.asarray(value), [edge // ratio for edge in edges], axis=axis)

    if isinstance(leaf, (jax.Array, np.ndarray)):
        checked(leaf.shape[axis])
        return arrays(leaf)
    qwix = _qwix()
    if isinstance(leaf, qwix.QArray):
        total = leaf.qvalue.shape[axis]
        checked(total)

        def scaled(value):
            if value is None or value.shape[axis] == 1:
                return [value] * len(widths)
            tile = total // value.shape[axis]
            if any(width % tile for width in widths):
                raise ValueError(f"members {list(widths)} cut the scales' tiles of {tile} columns")
            return arrays(value, tile)

        parts = zip(arrays(leaf.qvalue), scaled(leaf.scale), scaled(leaf.zero_point), strict=True)
        return [dataclasses.replace(leaf, qvalue=value, scale=scale, zero_point=zero)
                for value, scale, zero in parts]
    if (isinstance(leaf, Annotated) and isinstance(leaf.array, qwix.QArray) and dataclasses.is_dataclass(leaf)
            and not isinstance(leaf, type)):
        others = [field.name for field in dataclasses.fields(leaf) if field.name != "array"
                  and isinstance(getattr(leaf, field.name), (jax.Array, np.ndarray))]
        if others:
            raise TypeError(f"a {type(leaf).__name__} holds arrays beside its kernel ({', '.join(others)}), "
                            "whose split is not this one")
        return [dataclasses.replace(leaf, array=part)
                for part in split_columns(leaf.array, widths, axis=axis)]
    raise TypeError(f"a {type(leaf).__name__} is not a packed kernel this splits")


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
    served = _quantized(model, _providers()[1](rules), spec)
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


__all__ = ["InputQuantization", "Quantization", "split_columns"]
