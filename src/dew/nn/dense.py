"""Flax's dense layers, with their bias gradients summed at fp32 (in float64 on a CPU under Dew's tests).

`Dense` and `DenseGeneral` are flax 0.12's layers, fields, parameter names,
shapes, initializers, promotion and product hooks alike, so checkpoints,
LoRA and Qwix see the layers they did, but for a batched `DenseGeneral`'s
kernel, which draws each batch member alone (`_members`). Their call is their own: the same
product (the GEMM template's dense product where a launch takes it, `projections.dot_general`, XLA's program
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

import dataclasses
import functools
import importlib.util
import inspect
import math
import sys
import types
from collections.abc import Callable, Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.module import InterceptorContext
from flax.typing import Array

from dew._cache import INTERCEPTED

from .derivatives import KernelGlue, contracted, declared_tangent, with_tangents_formed
from .kernels import projections
from .kernels.formats import NvFp4, TernaryWeights
from .precision import accumulated, biased, float32_dot_general, refuse_recipe
from .sharding import contracted_alike, declared

QWIX = importlib.util.find_spec("qwix") is not None
"""Whether Qwix is installed (`dew.quantization`'s extra), whose rules a layer's product may fall under."""


QUANTIZATION = "quantization"
"""The collection that holds a projection site's checkpoint quantization, at
the site's own scope: every view of the variables (a scanned or staged
stack, a packed group, a pipeline) carries it with the weights it scales."""


def packed_matrix(module: nn.Module, name: str, make: Callable[[], jax.Array]):
    """`module`'s matrix `name`: the packed matrix its `QUANTIZATION` collection holds under the name (a
    checkpoint's weight served as its codes: `dew.nn.kernels.formats.NvFp4`'s 4-bit, an expert bank's
    `[experts, in, out]`, row 4516, a Linear's kernel `[in, out]`, row 8550; a Linear's BitNet b1.58
    `TernaryWeights`, row 1464), else its parameter, `make()`.
    The one reader of a packed matrix every layer that multiplies one takes (row 11641)."""
    if module.has_variable(QUANTIZATION, name):
        return module.get_variable(QUANTIZATION, name)
    return make()


def packed_ternary(module: nn.Module, name: str = "kernel") -> bool:
    """Whether `module`'s matrix `name` is a packed BitNet b1.58 matrix (`packed_matrix`'s, row 1464): its
    layer quantizes its input a token at a time, so no matrix alone is its product, whatever its `params`
    hold (`quantization.LinearProducts.nonlinear_input`, `decoder_head.output_table`)."""
    return module.has_variable(QUANTIZATION, name) and isinstance(module.get_variable(QUANTIZATION, name),
                                                                   TernaryWeights)


def watched_or_owned(module: nn.Module) -> bool:
    """Whether `module`'s calls must run as Flax and a wrapper see them, no rule of Dew's standing in for its
    computation (`dew.nn.derivatives.portable_formed`, which differentiates a function of arrays apart from
    the module's scope): a quantization scope of Dew's (`INTERCEPTED`) or a direct Qwix provider's
    (`providing`: any of its ops, at any path its rules name, may be a rule's), a Flax method interceptor
    installed (`intercepting`: a watch's), or its caller capturing its calls' outputs (`observed`)."""
    return INTERCEPTED.value or providing() or intercepting(module) or observed(module)


def providing() -> bool:
    """Whether a direct Qwix provider's interception is on in this thread (`qwix.quantize_model`, its
    interception manager, qwix/_src/interception.py, as `dew._cache.unintercepted` reads it): read by turning
    it off and on again, no rule queried, so none counted as matched (`get_unused_rules`). False where Qwix is
    not loaded, or another thread is loading it (its module present, its manager not yet made: no provider
    can be on), row 11804."""
    interception = sys.modules.get("qwix._src.interception")
    interceptions = None if interception is None else vars(interception).get("interception_manager")
    if interceptions is None:
        return False
    providers = interceptions.disable_interception()
    try:
        return bool(providers)
    finally:
        interceptions.enable_interception(providers)


def intercepting(module: nn.Module | None = None) -> bool:
    """Whether a Flax method interceptor installed (`nn.intercept_methods`, Flax's own stack) may intercept
    `module`'s own call, which a rule standing in for the call would bypass (a joined projection's, base
    review 760be77fe F1: an interceptor zeroing `v_proj`'s output, the joined layer returned it; a recomputed
    merge's): one not a watch of Dew's (node perturbation's, a caller's own), or a watch naming `module`
    (`inspection.watched_by`; row 11820: a watch of the MLPs leaves a merge to its own rules). Without a
    module, any interceptor installed."""
    from flax.linen import module as flax_module

    from dew.nn import inspection

    installed = list(flax_module._global_interceptor_stack)
    if module is None:
        return bool(installed)
    return any(inspection.watched_by(interceptor, module) for interceptor in installed)


PRE_CALL_CAPTURES: set[Callable] = set()
"""The `capture_intermediates` filters that decide from a module's name and method alone, before its call
runs (`pre_call_capture`): `observed` asks them; any other filter may read what the call sets up (its
parameters), which Flax asks after it, so a module under one is observed (row 11820's review)."""


def pre_call_capture(keep: Callable) -> Callable:
    """`keep`, a `capture_intermediates` filter that reads a module's name and method alone, registered
    as one `observed` may ask before the call (`PRE_CALL_CAPTURES`)."""
    PRE_CALL_CAPTURES.add(keep)
    return keep


def captures_any_method(module: nn.Module) -> bool:
    """Whether a caller's capture (`capture_intermediates`, `module`'s `intermediates` collection open) may
    keep the output of a method other than a call: a filter of the caller's own, which may take a helper
    method returning no array (a layer's resolved kind, a submodule), where Flax's call filter and Dew's
    pre-call filters (`PRE_CALL_CAPTURES`) keep modules' calls."""
    from flax.linen import module as flax_module

    if not module.is_mutable_collection('intermediates'):
        return False
    captures = flax_module._context.capture_stack
    keep = captures[-1] if captures else None
    if not keep:
        return False
    method = keep.__func__ if isinstance(keep, types.MethodType) else keep
    return not (method is flax_module.capture_call_intermediates
                or any(method is each for each in PRE_CALL_CAPTURES))


def observed(module: nn.Module) -> bool:
    """Whether a caller keeps `module`'s own calls' outputs as Flax captures a module's (its
    `intermediates` collection open and the capture's filter, `capture_intermediates`', taking its call;
    row 11820: a capture of the layers' states leaves a merge to its own rules), a filter that may read what
    the call sets up taking it (`PRE_CALL_CAPTURES`); a caller intercepting them is `intercepting`'s (a
    watch's, node perturbation's)."""
    from flax.linen import module as flax_module

    if not module.is_mutable_collection('intermediates'):
        return False
    captures = flax_module._context.capture_stack
    keep = captures[-1] if captures else None
    if not keep:
        return False
    method = keep.__func__ if isinstance(keep, types.MethodType) else keep
    # Matched by identity: a caller's filter may be an unhashable callable (row 11876).
    if method is flax_module.capture_call_intermediates or any(method is each for each in PRE_CALL_CAPTURES):
        return bool(keep(module, "__call__"))
    return True


def intercepted() -> bool:
    """Whether a wrapper owns the layer's product, which reaches it as a `jax.lax.dot_general` call: a Dew
    interception scope (`dew.quantization`, `dew.lora`: `dew._cache.INTERCEPTED`), or a Qwix rule matching
    the op at this module under a direct `qwix.quantize_model` call (`qwix.pallas.get_current_rule`, Qwix's
    public query for code that runs its own kernels, which every active provider answers)."""
    return INTERCEPTED.value or _ruled()


def _ruled() -> bool:
    """Whether a Qwix rule matches the op at this module under a direct `qwix.quantize_model` call
    (`qwix.pallas.get_current_rule`, which every active provider answers)."""
    if not QWIX:
        return False
    from qwix import pallas as qwix_pallas

    return qwix_pallas.get_current_rule("dot_general") is not None


def _product(layer: nn.Dense | nn.DenseGeneral, inputs: Array, kernel: Array, dimensions,
             bias: Array | None = None, tensor_scale: Array | None = None,
             formed: projections.FormedInput | None = None) -> jax.Array:
    """The layer's product of `inputs` and `kernel` over `dimensions`, plus `bias` where given, shaped to
    broadcast over it (`biased`). The product is its `dot_general_cls` (an interceptor's: a quantized
    product), as flax picks it, else its `dot_general`, or lax's at its stated precision (`accumulated`: the
    plain product, its fp32 sum taken in float64 on a CPU under Dew's tests). A plain product (lax's or
    `float32_dot_general`: its operands' dtype out, at the layer's precision) goes through the GEMM
    template's dense product where a launch takes it (`projections.dot_general`; XLA's program wherever the
    template declares XLA faster), which takes the bias inside its own rule, but where a wrapper owns the
    products (`intercepted`: Qwix quantizes them as it intercepts `dot_general`). A product with a policy of
    its own (a head's fp32 result: `fp32_result_dot_general`, `head_dot_general`) is the layer's own: the
    template reads no policy from a callable, and the bias is added past it. `tensor_scale`, the rows' under
    a delayed-scaling NVFP4 recipe that takes the product (`history_scale`), which the template's product
    hands its recipe (`projections.recipe_general`). `formed`, `inputs` formed entrywise from other values
    (`projections.FormedInput`), which the plain product multiplies as the template's formed operand; every
    other product multiplies `inputs` as any."""
    extents = projections.extents(inputs.shape, kernel.shape, dimensions)
    product = layer.dot_general if layer.dot_general is not None else _accumulated_dot_general
    if layer.dot_general_cls is not None:
        refuse_recipe("a quantizer's product class", *extents)
        product = layer.dot_general_cls()
    elif intercepted():
        refuse_recipe("a wrapper's product (an adapter's, a quantizer's)", *extents)
    elif layer.dot_general in (None, float32_dot_general):
        return projections.dot_general(product, projections.kernel_path(layer))(
            inputs, kernel, dimensions, precision=layer.precision, bias=bias, tensor_scale=tensor_scale,
            formed=formed)
    out = product(inputs, kernel, dimensions, precision=layer.precision)
    return out if bias is None else biased(out, bias)


def plain_product(path: tuple | None = None):
    """The product a plain `Dense` multiplies by (`_product` of a layer naming no product of its own):
    the GEMM template's where a launch takes it, else lax's with its fp32 sum taken in float64 on a CPU
    under Dew's tests, and lax's where a wrapper's scope takes the products past the template
    (`intercepted`). A product read apart from its layer, as a vocabulary head's table reads
    `project_out` (`Projection`), takes it, its kernel the parameter at `path` (`projections.kernel_path`),
    which a mesh splitting the rows places per shard (`projections.mapped_shards`)."""
    plain = _accumulated_dot_general
    return plain if intercepted() else projections.dot_general(plain, path)


def _accumulated_dot_general(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None,
                             **kwargs) -> jax.Array:
    """`jax.lax.dot_general` as stated, its fp32 sum taken in float64 on a CPU under Dew's tests
    (`accumulated`), landing where a contraction Explicit axes split on both operands lands its sum
    (`contracted_alike`) where the call states no placement of its own."""
    if kwargs.get("out_sharding") is None:
        lhs, rhs, placed = contracted_alike(lhs, rhs, dimension_numbers)
        kwargs = kwargs if placed is None else {**kwargs, "out_sharding": placed}
    return accumulated(functools.partial(jax.lax.dot_general, dimension_numbers=dimension_numbers,
                                         precision=precision, **kwargs),
                       lhs, rhs, preferred_element_type=preferred_element_type)


def _biased(layer: nn.Dense | nn.DenseGeneral, inputs: Array, kernel: Array, dimensions, bias: Array | None,
            shaped, stored: Array | None = None, *, taken: Callable | None = None,
            tensor_scale: Array | None = None, formed: projections.FormedInput | None = None):
    """The layer's product of `inputs` and `kernel` over `dimensions` as flax takes it, plus `bias`
    reshaped by `shaped` to broadcast over it (`biased`: added as flax adds it, its gradient summed
    at fp32, in float64 on a CPU under Dew's tests); where the layer computes in a 16-bit dtype and its
    bias is `stored` at fp32, the product and that bias rounded once (`_rounds_once`). `taken`, where
    given, takes the product, `taken(product)` of `product(inputs, kernel)`, as its caller forms it (a
    joined projection's kernels joined on each device's rows, `dew.nn.joined.joined_dense`; `kernel` then
    one of them, its dtype the product's), the bias added past it as here. `tensor_scale` and `formed`,
    `_product`'s, which a product rounded once with its bias takes as any input."""
    take = taken or (lambda product: product(inputs, kernel))
    if bias is not None and _rounds_once(layer, inputs, kernel, bias, stored):
        widened = take(lambda inputs, kernel: _summed(inputs, kernel, dimensions, layer.precision,
                                                      tensor_scale))
        return biased(widened, shaped(stored)).astype(inputs.dtype)
    if taken is None:
        return _product(layer, inputs, kernel, dimensions, None if bias is None else shaped(bias),
                        tensor_scale, formed)
    product = taken(lambda inputs, kernel: _product(layer, inputs, kernel, dimensions,
                                                    tensor_scale=tensor_scale))
    return product if bias is None else biased(product, shaped(bias))


def _summed(inputs: Array, kernel: Array, dimensions, precision,
            tensor_scale: Array | None = None) -> jax.Array:
    """A biased layer's product left unrounded in fp32, its bias added past it (`_biased`): the run's
    low-precision recipe's where it takes the projection (`projections.recipe_general`: its fp32 sums and its
    own gradients, its rows at `tensor_scale` under delayed scaling), else XLA's (`_widened`)."""
    recipe = projections.recipe_general(inputs, kernel, dimensions, jnp.float32, tensor_scale=tensor_scale)
    return recipe if recipe is not None else _widened(inputs, kernel, dimensions, precision)


def _rounds_once(layer: nn.Dense | nn.DenseGeneral, inputs: Array, kernel: Array, bias: Array,
                 stored: Array | None) -> bool:
    """Whether a biased layer rounds its product and bias once, as torch's addmm does: its operands
    16-bit, its bias stored at fp32 and promoted to their dtype (flax's promotion, whose sum it rounds
    in that dtype; a promotion that keeps the bias wider adds it as it gives, row 4870), and its product
    the plain one (no product of its own, no wrapper owning the products: `_product`)."""
    narrow = jnp.result_type(inputs, kernel)
    plain = layer.dot_general is None and layer.dot_general_cls is None and not intercepted()
    return (narrow in (jnp.bfloat16, jnp.float16) and bias.dtype == narrow and plain and stored is not None
            and stored.dtype == jnp.float32)


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
    # Its reverse mode returns the inputs' cotangent only with the kernel's gradient formed.
    d_inputs, d_kernel = with_tangents_formed(d_inputs, d_kernel)
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



NVFP4_HISTORY = 'nvfp4'
"""The collection of a delayed-scaling NVFP4 run's amax histories (`LowPrecision.tensor_scaling`, row
12721): each projection the run's recipe takes holds one (a `Dense`'s `'amax_history'`, an expert layer's
each projection's by its name), float32 `[amax_history]` (`[layers, amax_history]` in a scanned stack), its
rows' largest magnitudes of its last steps, newest first; zeros until a step records one (the scale of an
all-zero history is 1). The forward reads it alone; a training step rolls it once per committed update with
the step's recorded largest (`rolled`), no optimizer seeing it."""
NVFP4_AMAX = 'nvfp4_amax'
"""The sown collection of each such projection's rows' largest magnitude a call (`projections.nvfp4_amax`,
a float32 scalar, one a layer in a scanned stack), named and placed as its history (`NVFP4_HISTORY`), which
a training step takes the largest of across its calls and microbatches and rolls in (`rolled`)."""


NVFP4_PENDING = 'nvfp4_pending'
"""The collection of the largest magnitudes a delayed-scaling NVFP4 run recorded in calls an optimizer that
holds updates back (`optax.MultiSteps`) took in but applied no update in, which the call that applies one
rolls in with its own (`rolled_waiting`, row 12994): each history's (`NVFP4_HISTORY`) mirror, float32 `[]`
(`[layers]` in a scanned stack), -1 where nothing waits (a magnitude is never below zero, and zero is one). A
run whose optimizer applies every update holds none."""


def pending_amax(histories: Mapping) -> dict:
    """An `NVFP4_PENDING` collection for `histories` with nothing waiting: each history's newest place, -1."""
    return {name: pending_amax(value) if isinstance(value, Mapping) else jnp.full_like(value[..., 0], -1.0)
            for name, value in histories.items()}


def rolled_waiting(histories: Mapping, pending: Mapping, amaxes: Mapping, applied: jax.Array,
                   taken: jax.Array) -> tuple[dict, dict]:
    """`histories` and their `NVFP4_PENDING` collection `pending` after a step under an optimizer that holds
    updates back: each history `amaxes` holds a largest for waits on the larger of it and what waited where
    the optimizer `taken` the step's input (a skipped input's records wait for no update); where the step
    `applied` an update, every history something waits for rolls it in, newest (`rolled`), a projection an
    earlier call ran and this one did not among them, and nothing waits after."""
    def at(name, history):
        if isinstance(history, Mapping):
            return rolled_waiting(history, pending[name], amaxes.get(name, {}), applied, taken)
        waiting = pending[name]
        if name in amaxes:
            recorded = jnp.reshape(jnp.asarray(amaxes[name], waiting.dtype), waiting.shape)
            waiting = jnp.where(taken, jnp.maximum(waiting, recorded), waiting)
        newest = jnp.reshape(waiting.astype(history.dtype), (*history.shape[:-1], 1))
        moved = jnp.concatenate([newest, history[..., :-1]], axis=-1)
        rolls = jnp.reshape(applied & (waiting >= 0), (*history.shape[:-1], 1))
        return jnp.where(rolls, moved, history), jnp.where(applied, jnp.full_like(waiting, -1.0), waiting)

    pairs = {name: at(name, value) for name, value in histories.items()}
    return {name: pair[0] for name, pair in pairs.items()}, {name: pair[1] for name, pair in pairs.items()}


def amax_history(layer: nn.Module, name: str, shape: tuple[int, ...]) -> nn.Variable | None:
    """`layer`'s amax history for its projection `name` by a `shape` (`[..., depth, width]`, a product's
    extents as matrices) matrix, held at its path as `name` (`NVFP4_HISTORY`), where the run trains in a
    delayed-scaling NVFP4 recipe that takes the projection in the role traced (`projections.recipe_takes`);
    None elsewhere (row 12721). Declared where `layer` declares its variables (its `setup` or a compact
    method); a delayed run whose variables hold none, initialized under another recipe, is refused."""
    from dew.nn.precision import recipe_role

    held = recipe_role()
    if held is None or held[0].tensor_scaling != "delayed" or not projections.recipe_takes(*shape[-2:]):
        return None
    if not (layer.has_variable(NVFP4_HISTORY, name) or layer.is_mutable_collection(NVFP4_HISTORY)):
        raise ValueError(f"{layer.name}'s run trains in delayed-scaling NVFP4, whose projections read their "
                         f"amax histories (the {NVFP4_HISTORY!r} collection), and its variables hold none: "
                         "initialize the model under the run's recipe")
    length = held[0].amax_history
    return layer.variable(NVFP4_HISTORY, name, lambda: jnp.zeros((length,), jnp.float32))


def history_scale(layer: nn.Module, name: str, shape: tuple[int, ...]) -> jax.Array | None:
    """The tensor scale `layer`'s projection `name` quantizes its rows at: its amax history's
    (`amax_history`, `projections.delayed_tensor_scale`), None where it holds none. Its rows' largest
    magnitude the caller records (`recorded`)."""
    history = amax_history(layer, name, shape)
    return None if history is None else projections.delayed_tensor_scale(history.value)


def recorded(layer: nn.Module, name: str, rows: Array) -> None:
    """`rows`' largest magnitude (`projections.nvfp4_amax`) sown as `layer`'s projection `name`'s this call
    (`noted_amax`)."""
    noted_amax(layer, name, projections.nvfp4_amax(rows))


def noted_amax(layer: nn.Module, name: str, amax: Array) -> None:
    """`amax`, the largest magnitude among the rows `layer`'s projection `name` quantized this call, sown as
    its (`NVFP4_AMAX`) where the call keeps it, from `layer`'s own call (no transformation's body): what its
    history rolls in (`rolled`)."""
    layer.sow(NVFP4_AMAX, name, amax)


def rolled(histories: Mapping, amaxes: Mapping) -> dict:
    """`histories` (an `NVFP4_HISTORY` collection) with each history `amaxes` holds a largest magnitude for
    (a tree mirroring them, each leaf the largest over its projection's calls and a step's microbatches, a
    float32 scalar, or one a layer `[layers]` for a scanned stack's `[layers, steps]` history) shifted by one
    along its steps, that largest newest, its oldest dropped; every other history as it was (a
    projection no call of the step ran). A training step's update of a delayed-scaling NVFP4 run's state,
    once per committed update (row 12721)."""
    def roll(history, amax):
        # A scanned stack's history `[layers, steps]` and its amax `[layers]`: each layer's rolled.
        newest = jnp.reshape(jnp.asarray(amax, history.dtype), (*history.shape[:-1], 1))
        return jnp.concatenate([newest, history[..., :-1]], axis=-1)

    def at(name, value):
        if isinstance(value, Mapping):
            return rolled(value, amaxes.get(name, {}))
        return roll(value, amaxes[name]) if name in amaxes else value

    return {name: at(name, value) for name, value in histories.items()}


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
    the features' (`declared`); None declares none. A call's `formed` (`projections.FormedInput`) is read
    only where its `operand` is the call's `inputs` itself: an interceptor handing the layer other inputs
    (an input quantizer's) leaves it unread."""

    kernel_axes: tuple[str | None, ...] | None = None

    @nn.compact
    def __call__(self, inputs: Array, formed: projections.FormedInput | None = None) -> jax.Array:
        """The layer of `inputs`. `formed`, where given, states `inputs` formed entrywise from other values
        (`projections.FormedInput`: a gated MLP's hidden activation), which the layer's plain product
        multiplies as the template's formed operand, holding the values it was formed from in place of
        `inputs` for reverse mode (`_product`), under the class's identity rule."""
        held = formed if formed is not None and formed.operand is inputs else None
        return self._envelope(functools.partial(self._linear, formed=held), inputs)

    @property
    def reads_formed(self) -> bool:
        """Whether its call takes `formed` (`protocols.ReadsFormed`): a plain Dense's does."""
        return True

    def product_owned(self) -> bool:
        """Whether a wrapper owns this layer's product (`intercepted`), asked within its own scope, so a
        Qwix rule over its module path answers: what a parent joining it with others reads (`joined`)."""
        return intercepted()

    def _linear(self, inputs: Array, formed: projections.FormedInput | None = None) -> jax.Array:
        width = jnp.shape(inputs)[-1]
        axes = self.kernel_axes
        kernel = packed_matrix(self, 'kernel', lambda: self.param(
            'kernel', declared(self.kernel_init, axes), (width, self.features), self.param_dtype))
        if isinstance(kernel, (NvFp4, TernaryWeights)):
            return self._packed(inputs, kernel)
        bias = (self.param('bias', declared(self.bias_init, None if axes is None else axes[-1:]),
                           (self.features,), self.param_dtype) if self.use_bias else None)
        stored = bias
        inputs, kernel, bias = _promoted(self, inputs, kernel, bias)
        dimensions = (((inputs.ndim - 1,), (0,)), ((), ()))
        # Under a delayed-scaling NVFP4 recipe that takes the product, its rows' scale its amax history's.
        scale = history_scale(self, 'amax_history', kernel.shape)
        if scale is not None:
            recorded(self, 'amax_history', inputs)
        # The formed input as promoted: the product reads its hook's values in the dtype it multiplies.
        formed = None if formed is None else dataclasses.replace(formed, operand=inputs)
        return _biased(self, inputs, kernel, dimensions, bias, lambda bias: bias, stored, tensor_scale=scale,
                       formed=formed)


    @nn.nowrap
    def _packed(self, inputs: Array, matrix: NvFp4 | TernaryWeights) -> jax.Array:
        """The layer over its packed kernel (`packed_matrix`, a checkpoint's weight served as its codes):
        an NVFP4 matrix's product (`gemm.nvfp4_product`, row 8550: the inputs rounded to bfloat16, each code
        times its block's scale exact, the sums in fp32 and the matrix's tensor scale after) or a BitNet
        b1.58 ternary matrix's (`gemm.ternary_weight_product`, row 1464: the inputs quantized a token at a
        time to int8, each {-1, 0, 1} code's product exact, the scales after), its fp32 sum plus its bias
        rounded once to the layer's dtype. Frozen, as the codes are a
        checkpoint's; a quantization rule over the site (Qwix's) quantizes a kernel, which a packed one is
        not, and is refused. An input quantization of other sites (a source's stored input scales) leaves
        it as it is: a packed weight's source stores none for it (`codecs.SourceQuantization.as_nvfp4`)."""
        from dew.nn.kernels.gemm import nvfp4_product, ternary_weight_product

        if _ruled():
            raise ValueError(f"{self.name}'s kernel is packed (a checkpoint's codes, served as they "
                             "are), and a quantization rule owns its product: load it without packed_weights")
        width = jnp.shape(inputs)[-1]
        if tuple(matrix.shape) != (width, self.features):
            raise ValueError(f"{self.name}'s packed kernel is {list(matrix.shape)}, its layer "
                             f"[{width}, {self.features}]")
        bias = (self.param('bias', declared(self.bias_init, None if self.kernel_axes is None
                                            else self.kernel_axes[-1:]), (self.features,), self.param_dtype)
                if self.use_bias else None)
        dtype = self.dtype if self.dtype is not None else jnp.result_type(inputs)
        # The rows counted, not inferred: a packed matrix of no depth takes inputs of no width.
        rows = jnp.reshape(inputs, (math.prod(jnp.shape(inputs)[:-1]), width))
        if isinstance(matrix, TernaryWeights):
            # Frozen, its scale too: the product differentiates in its scale (`ternary_weight_product`).
            frozen = dataclasses.replace(matrix, scale=jax.lax.stop_gradient(matrix.scale))
            summed = ternary_weight_product(rows, frozen, out_dtype=jnp.float32)
        else:
            summed = nvfp4_product(rows, matrix, out_dtype=jnp.float32)
        if bias is not None:
            summed = summed + jnp.asarray(bias, jnp.float32)
        return jnp.reshape(summed.astype(dtype), (*jnp.shape(inputs)[:-1], self.features))


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
        dimensions = ((axis, contracted), (batch_dims, tuple(range(batched))))
        # Under a delayed-scaling NVFP4 recipe that takes the product (its depth by width as matrices), its
        # rows' scale its amax history's.
        extents = projections.extents(inputs.shape, kernel.shape, dimensions)
        scale = history_scale(self, 'amax_history', extents)
        if scale is not None:
            recorded(self, 'amax_history', inputs)
        return _biased(self, inputs, kernel, dimensions, bias,
                       lambda bias: jnp.reshape(bias, kept + features), stored, tensor_scale=scale)


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
