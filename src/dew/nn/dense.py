"""Flax's dense layers, with their bias gradients summed at fp32 (in float64 on a CPU under Dew's tests).

`Dense` and `DenseGeneral` are flax 0.12's layers, fields, parameter names,
shapes, initializers, promotion and product hooks alike, so checkpoints,
LoRA and Qwix see the layers they did, but for a batched `DenseGeneral`'s
kernel, which draws each batch member alone (`_members`). Their call is their own: the same
product (the GEMM template's dense product where a launch takes it, `gemm.dot_general`, XLA's program
otherwise), rounded to its dtype as flax's is, then the bias added by `biased`,
which adds it in the dtype flax's addition gives them and broadcasts it at
fp32, so its gradient sums at fp32 as torch's addmm sums it, and in float64 on a CPU under
Dew's tests, as such a CPU sums every float32 product (`dew.nn.precision.accumulated`). A layer
computing in a 16-bit dtype whose bias is stored at fp32 rounds its product and bias once, as torch's
addmm does (`_rounds_once`, row 634). The bias is
added inside the layer's own managed call, so a method interceptor (an
adapter, an input quantization) and an intermediate capture see the
output as they saw flax's.

flax's own calls broadcast the bias in its storage dtype, where its gradient
then sums, and offer no step to widen it.
"""

import functools
import importlib.util
import inspect
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.module import InterceptorContext
from flax.typing import Array

from dew._cache import INTERCEPTED

from .derivatives import KernelGlue, contracted, declared_tangent
from .kernels import gemm
from .precision import accumulated, biased, float32_dot_general, refuse_recipe
from .sharding import declared

QWIX = importlib.util.find_spec("qwix") is not None
"""Whether Qwix is installed (`dew.quantization`'s extra), whose rules a layer's product may fall under."""


def intercepted() -> bool:
    """Whether a wrapper owns the layer's product, which reaches it as a `jax.lax.dot_general` call: a Dew
    interception scope (`dew.quantization`, `dew.lora`: `dew._cache.INTERCEPTED`), or a Qwix rule matching
    the op at this module under a direct `qwix.quantize_model` call (`qwix.pallas.get_current_rule`, Qwix's
    public query for code that runs its own kernels, which every active provider answers)."""
    if INTERCEPTED.value:
        return True
    if not QWIX:
        return False
    from qwix import pallas as qwix_pallas

    return qwix_pallas.get_current_rule("dot_general") is not None


def _product(layer: nn.Dense | nn.DenseGeneral):
    """The layer's product: its `dot_general_cls` (an interceptor's: a quantized product), as flax picks
    it, else its `dot_general`, or lax's at its stated precision (`accumulated`: the plain product, its fp32
    sum taken in float64 on a CPU under Dew's tests). A plain product (lax's or `float32_dot_general`: its
    operands' dtype out, at the layer's precision) goes through the GEMM template's dense product where a
    launch takes it (`gemm.dot_general`; XLA's program wherever the template declares XLA faster), but
    where a wrapper owns the products (`intercepted`: Qwix quantizes them as it intercepts `dot_general`). A
    product with a policy of its own (a head's fp32 result: `fp32_result_dot_general`, `head_dot_general`)
    is the layer's own: the template reads no policy from a callable."""
    if layer.dot_general_cls is not None:
        refuse_recipe("a quantizer's product class")
        return layer.dot_general_cls()
    product = layer.dot_general if layer.dot_general is not None else _accumulated_dot_general
    plain = layer.dot_general in (None, float32_dot_general)
    if intercepted():
        refuse_recipe("a wrapper's product (an adapter's, a quantizer's)")
    return gemm.dot_general(product) if plain and not intercepted() else product


def plain_product():
    """The product a plain `Dense` multiplies by (`_product` of a layer naming no product of its own):
    the GEMM template's where a launch takes it, else lax's with its fp32 sum taken in float64 on a CPU
    under Dew's tests, and lax's alone where a wrapper owns the products (`intercepted`). A product read
    apart from its layer, as a vocabulary head's table reads `project_out` (`Projection`), takes it."""
    return _accumulated_dot_general if intercepted() else gemm.dot_general(_accumulated_dot_general)


def _accumulated_dot_general(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None,
                             **kwargs) -> jax.Array:
    """`jax.lax.dot_general` as stated, its fp32 sum taken in float64 on a CPU under Dew's tests
    (`accumulated`)."""
    return accumulated(functools.partial(jax.lax.dot_general, dimension_numbers=dimension_numbers,
                                         precision=precision, **kwargs),
                       lhs, rhs, preferred_element_type=preferred_element_type)


def _biased(layer: nn.Dense | nn.DenseGeneral, inputs: Array, kernel: Array, dimensions, bias: Array | None,
            shaped, stored: Array | None = None):
    """The layer's product of `inputs` and `kernel` over `dimensions` as flax takes it, plus `bias`
    reshaped by `shaped` to broadcast over it (`biased`: added as flax adds it, its gradient summed
    at fp32, in float64 on a CPU under Dew's tests); where the layer computes in a 16-bit dtype and its
    bias is `stored` at fp32, the product and that bias rounded once (`_rounds_once`)."""
    if bias is not None and _rounds_once(layer, inputs, kernel, stored):
        widened = _widened(inputs, kernel, dimensions, layer.precision)
        return biased(widened, shaped(stored)).astype(inputs.dtype)
    product = _product(layer)(inputs, kernel, dimensions, precision=layer.precision)
    if bias is None:
        return product
    return biased(product, shaped(bias))


def _rounds_once(layer: nn.Dense | nn.DenseGeneral, inputs: Array, kernel: Array, stored: Array | None
                 ) -> bool:
    """Whether a biased layer rounds its product and bias once, as torch's addmm does: its operands
    16-bit, its bias stored at fp32, and its product the plain one (no product of its own, no wrapper
    owning the products: `_product`)."""
    narrow = jnp.result_type(inputs, kernel) in (jnp.bfloat16, jnp.float16)
    plain = layer.dot_general is None and layer.dot_general_cls is None and not intercepted()
    return narrow and plain and stored is not None and stored.dtype == jnp.float32


def _widened_value(inputs: Array, kernel: Array, dimensions, precision) -> jax.Array:
    """`_widened`'s value: XLA's product summed in fp32 and left unrounded (`_accumulated_dot_general`)."""
    return _accumulated_dot_general(inputs, kernel, dimensions, precision=precision,
                                    preferred_element_type=jnp.float32)


def _widened_tangent(dimensions, precision, primals, tangents):
    """`_widened`'s tangent: each operand's direction's product in the operands' dtype, widened, which reverse
    mode transposes into the operands' dtype's products of the cotangent rounded to it (exact: the
    cotangent of a 16-bit output), as the layer's 16-bit products' are."""
    from jax.custom_derivatives import SymbolicZero

    inputs, kernel = primals
    d_inputs, d_kernel = tangents
    value = _widened_value(inputs, kernel, dimensions, precision)
    terms = [jax.lax.dot_general(left, right, dimensions, precision=precision).astype(jnp.float32)
             for left, right, direction in ((d_inputs, kernel, d_inputs), (inputs, d_kernel, d_kernel))
             if not isinstance(direction, SymbolicZero)]
    return value, sum(terms[1:], terms[0]) if terms else jnp.zeros_like(value)


# A biased 16-bit layer's product summed in fp32 for its fp32 bias (`_rounds_once`): XLA's program, which
# fuses the bias's add and the rounding into its GEMM where the bias is a parameter (gemm_fusion.cc's
# epilogue rule; NSDIAG-BIAS: no fp32 product written), its tangent the layer's 16-bit products'.
_widened = contracted("test_a_once_rounded_biased_layer_holds_every_order")(
    declared_tangent(KernelGlue(), _widened_value, _widened_tangent, nondiff_argnums=(2, 3),
                     symbolic_zeros=True))


def _reshaped(init, flat_shape):
    """`init` drawn at `flat_shape` and reshaped to the shape asked of it, a boxed draw (a partitioned
    initializer's) reshaped inside its box, whose names are the reshaped kernel's: the reshape maps over
    the draw's values, so no constraint the names would apply meets the flat draw."""
    def drawn(rng, shape, dtype=jnp.float32):
        flat = init(rng, tuple(int(size) for size in flat_shape(shape)), dtype)
        return jax.tree.map(lambda value: jnp.reshape(value, shape), flat)
    return drawn



def _promoted(layer: nn.Dense | nn.DenseGeneral, inputs: Array, kernel: Array, bias: Array | None):
    """The layer's operands through its `promote_dtype`, as flax promotes them."""
    inputs, kernel, bias = layer.promote_dtype(inputs, kernel, bias, dtype=layer.dtype)
    if inputs is None or kernel is None:
        raise ValueError("a dense layer's promotion must return its input and kernel")
    return inputs, kernel, bias


def _axes(axes: Iterable[int] | int, ndim: int | None = None) -> tuple[int, ...]:
    """`axes` as a tuple, as flax canonicalizes them, and with `ndim` given, a negative axis
    counted from the end and the whole sorted; an axis out of range stays out of range."""
    axes = tuple(axes) if isinstance(axes, Iterable) else (axes,)
    return axes if ndim is None else tuple(sorted(axis if axis >= 0 else ndim + axis for axis in axes))


class _LinearEnvelope(nn.Module):
    """A managed arithmetic call inside an owner's input/output policy, at the same parameter scope."""

    @nn.compact
    def __call__(self, inputs: Array) -> jax.Array:
        return self._envelope(self._linear, inputs)

    @nn.nowrap
    def _envelope(self, linear: Callable[[Array], jax.Array], inputs: Array) -> jax.Array:
        return linear(inputs)

    def _linear(self, inputs: Array) -> jax.Array:
        raise NotImplementedError


class Dense(_LinearEnvelope, nn.Dense):
    """Flax's Dense, its bias added by `biased` (the module docstring). `kernel_axes` are its kernel's
    logical axes, `[inputs, features]`, which its module declares where it builds it, and its bias takes
    the features' (`declared`); None declares none."""

    kernel_axes: tuple[str | None, ...] | None = None

    def product_owned(self) -> bool:
        """Whether a wrapper owns this layer's product (`intercepted`), asked within its own scope, so a
        Qwix rule over its module path answers: what a parent joining it with others reads (`joined`)."""
        return intercepted()

    def _linear(self, inputs: Array) -> jax.Array:
        width = jnp.shape(inputs)[-1]
        axes = self.kernel_axes
        kernel = self.param('kernel', declared(self.kernel_init, axes), (width, self.features),
                            self.param_dtype)
        bias = (self.param('bias', declared(self.bias_init, None if axes is None else axes[-1:]),
                           (self.features,), self.param_dtype) if self.use_bias else None)
        stored = bias
        inputs, kernel, bias = _promoted(self, inputs, kernel, bias)
        dimensions = (((inputs.ndim - 1,), (0,)), ((), ()))
        return _biased(self, inputs, kernel, dimensions, bias, lambda bias: bias, stored)


def _members(init, batched: int, flat_shape):
    """`init` drawn for each member of a kernel's `batched` leading batch axes alone, at `flat_shape`
    of the rest, and stacked: each member's own fan-in, its inputs', where flax draws the whole kernel
    at `[batch * inputs, features]` and a variance-scaling initializer counts the batch members as
    inputs (DeepSeek V4's grouped `o_a_proj`, eight groups, would start sqrt(8) narrower than one
    group's lecun init). A boxed draw (a partitioned initializer's) is reshaped inside its box, as
    `_reshaped` reshapes it."""
    def drawn(rng, shape, dtype=jnp.float32):
        member = tuple(int(size) for size in flat_shape(shape[batched:]))
        keys = jax.random.split(rng, int(np.prod(shape[:batched])))
        stacked = jax.vmap(lambda key: init(key, member, dtype))(keys)
        return jax.tree.map(lambda value: jnp.reshape(value, shape), stacked)
    return drawn


class DenseGeneral(_LinearEnvelope, nn.DenseGeneral):
    """Flax's DenseGeneral, its bias added by `biased` (the module docstring), and a batched kernel
    drawn a batch member at a time (`_members`). `kernel_axes` are its kernel's logical axes, `[batch
    dims, contracted, features]`, and its bias takes the batch dims' and the features' (`declared`);
    None declares none."""

    kernel_axes: tuple[str | None, ...] | None = None

    def _linear(self, inputs: Array) -> jax.Array:
        features, batch_dims = _axes(self.features), _axes(self.batch_dims)
        if batch_dims and set(batch_dims) != set(range(max(batch_dims) + 1)):
            raise ValueError(f'batch_dims {batch_dims} must be consecutive leading dimensions '
                             'starting from 0.')
        axis, batch_dims = _contractions(self, inputs.ndim)
        batched = len(batch_dims)
        batch_shape = tuple(inputs.shape[dim] for dim in batch_dims)

        def flat(shape):
            """`shape` as `[inputs, features]`, the contracted axes and the features each flattened."""
            return np.prod(shape[:len(shape) - len(features)]), np.prod(shape[-len(features):])

        axes = self.kernel_axes
        init = declared(self.kernel_init, axes)
        drawn = _members(init, batched, flat) if batched else _reshaped(init, flat)
        shape = batch_shape + tuple(inputs.shape[dim] for dim in axis) + features
        kernel = self.param('kernel', drawn, shape, self.param_dtype)
        bias_axes = None if axes is None else (*axes[:batched], *axes[len(axes) - len(features):])
        bias = self.param(
            'bias', _reshaped(declared(self.bias_init, bias_axes),
                              lambda shape: (np.prod(shape[:batched]) * np.prod(shape[-len(features):]),)),
            batch_shape + features, self.param_dtype) if self.use_bias else None
        stored = bias
        inputs, kernel, bias = _promoted(self, inputs, kernel, bias)
        contracted = tuple(range(batched, batched + len(axis)))
        # The product is [batch dims, kept dims, features]; the bias broadcasts over the kept dims.
        kept = tuple(inputs.shape[dim] if dim in batch_dims else 1
                     for dim in range(inputs.ndim) if dim not in axis)
        return _biased(self, inputs, kernel, ((axis, contracted), (batch_dims, tuple(range(batched)))), bias,
                       lambda bias: jnp.reshape(bias, kept + features), stored)


def _contractions(module: nn.Dense | nn.DenseGeneral, ndim: int
                   ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Normalized input contraction and batch axes, shared with adapters and input quantization."""
    if isinstance(module, nn.DenseGeneral):
        return _axes(module.axis, ndim), _axes(module.batch_dims, ndim)
    return (ndim - 1,), ()


def _plain_dense(module: nn.Module) -> bool:
    """A bare Dense call, with no subclass envelope for a joined projection to bypass."""
    return type(module) is Dense


def _linear_site(context: InterceptorContext, target: str) -> bool:
    """Whether this targeted call is the managed linear arithmetic, rather than its envelope.

    Flax exposes the original method publicly. Its partial binding and decorators are inspected,
    never called here: execution through next_fun must retain every other interceptor.
    """
    module = context.module
    original = context.orig_method
    while isinstance(original, functools.partial):
        original = original.func
    original = inspect.unwrap(original)
    if isinstance(module, (Dense, DenseGeneral)):
        owner = DenseGeneral if isinstance(module, DenseGeneral) else Dense
        arithmetic = context.method_name == "_linear"
        expected = owner._linear if arithmetic else owner.__call__
        if original is not inspect.unwrap(expected):
            raise TypeError(f"{target} overrides its linear {context.method_name}; put input/output policy "
                            "in _envelope so adapters and quantizers can reach its arithmetic")
        return arithmetic
    if isinstance(module, (nn.Dense, nn.DenseGeneral)):
        owner = nn.DenseGeneral if isinstance(module, nn.DenseGeneral) else nn.Dense
        if context.method_name == "__call__" and original is inspect.unwrap(owner.__call__):
            return True
    raise TypeError(f"{target} does not execute a supported Dense or DenseGeneral arithmetic call")
