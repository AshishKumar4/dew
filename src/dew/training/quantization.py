"""Quantized training through Qwix, applied to the model at build time.

Qwix (google/qwix, Apache 2.0) expresses quantization as rules over module
paths and applies them without editing the model: one call wraps the module
and the matmuls in the wrapped methods' extent run quantized. Dew's version
of that call is `apply`: a recipe builds its model from the registry as
always, then wraps it before the objective ever sees it.

What trains is fake-quantized. The parameter tree keeps fp32 master weights
with the same structure, so the checkpoint layout, the sharding derivation,
the Muon parameter split and Hugging Face loading are unchanged: the
quantization lives in the forward and backward matmuls, with a
straight-through estimator on the backward pass. The vocabulary head stays
fp32 with the rest of Dew's fp32 zones: its einsum lives in the objective's
chunked cross entropy, outside any model method Qwix wraps.

The value mirrors MaxText's knob set (configs/base.yml:128-167) where Qwix
has an equivalent: `dtype` is its `quantization` for the dynamic-range
forms, `patterns` is its `quant_cfg_path` written inline as the regexes
Qwix matches, and the backward fields are Qwix's finer-grained version of
the same idea. Three of its knobs have no equivalent and are refused with
the reason: static activation scaling (`fp8_full`) needs a calibration pass
Dew has no seam for, `nanoo_fp8` is AMD-only kernels, and KV-cache
quantization has no reader here since the cache holds the compute dtype.

Qwix is not a dependency. The import sits inside `apply`, and without the
package the call raises naming it, the way the tokamax branch of
`dew.nn.moe` behaves.
"""
import dataclasses
import importlib
import re
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

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
    """How a run quantizes its trunk matmuls with Qwix's quantized-training
    provider."""

    dtype: QuantizedDtype = "int8"
    """The dtype weights and activations quantize to, in the forward pass."""
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
        if self.dtype == "fp8_full":
            raise ValueError(
                "fp8_full is static activation scaling, which needs a "
                "calibration pass Dew has no seam for; Dew's fp8 is dynamic "
                "range")
        if self.dtype == "nanoo_fp8":
            raise ValueError(
                "nanoo_fp8 is kernels for AMD MI300 and MI325, which this "
                "hardware cannot run")
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
    """`dtype` as the JAX dtype Qwix quantizes to."""
    return jnp.int8 if dtype == "int8" else jnp.float8_e4m3fn


def apply(model: nn.Module, spec: Quantization) -> nn.Module:
    """`model` with its trunk matmuls training in `spec`'s dtype.

    The returned module is a copy of the same class with the entry methods
    it defines of `METHODS` wrapped, so everything the registry, the
    objective and the checkpoint code read off the model still answers.
    Construction already refused what the value cannot ask for; without the
    package the call raises naming it.
    """
    if not isinstance(spec, Quantization):
        raise ValueError(
            f"quantization is a Quantization value, got {spec!r}")
    qwix = importlib.import_module("qwix")
    rules = [
        qwix.QtRule(
            module_path=pattern,
            weight_qtype=_qtype(spec.dtype),
            act_qtype=_qtype(spec.dtype),
            tile_size=spec.tile_size,
            weight_calibration_method=spec.calibration,
            bwd_qtype=None if spec.bwd_qtype is None else _qtype(spec.bwd_qtype),
            bwd_stochastic_rounding=spec.bwd_stochastic_rounding,
        )
        for pattern in spec.patterns
    ]
    methods = tuple(method for method in METHODS if hasattr(model, method))
    return qwix.quantize_model(model, qwix.QtProvider(rules), methods=methods)
