"""A program's rounding scale, read from its jaxpr in float64 (`rounding_scale`): what check 1's conditioning
weighs an element's error by (tests/reference_error.py, `assert_as_exact`'s `conditioning`).

Each rounding the program makes moves a value v by at most w(v) u, u its format's unit roundoff: a correctly
rounded operation (an add, a product, a quotient, a root, a narrowing conversion) by its result's
magnitude, a sum of m terms in any order by (m - 1) times the sum of the terms' magnitudes, an inner
product of m terms by m times it (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., 2002,
sections 2.2, 3.1 and 4.2), an integer converted to a float too narrow to hold it by its magnitude, and an
elementary function by its implementation's ulps times its result (`ULPS`). The program is evaluated in
float64 with each rounding moved by w(v) u times a standard normal draw (a tap), every operation its own
draw (`TapRule`), and each output element's scale c is the root mean square, over `DRAWS` draws, of how
far the taps move it from the untapped evaluation: its rounding error's size where every rounding moves its
value by a random multiple of its bound, the program's own conditioning included, first order or not. An
element no rounding reaches has c = 0.

The float64 evaluation adds its own rounding, u64 = 2^-53 a value, to each draw's displacement; a tap is
w(v) u at the narrowest format the program rounds to, float32's u = 2^-24 or bfloat16's 2^-8, so the
evaluation's noise sits 2^-29 or 2^-45 below the tap it is measured beside, each carried through the same
program alike. The scale is a deterministic weight of the program and its inputs, keyed by a seed; it is not
a bound.
"""

from __future__ import annotations

import dataclasses
import functools
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from float64_terms import SUMS, _evaluated, _static, _target, evaluated_on

EXACT = frozenset({
    "neg", "abs", "max", "min", "select_n", "reshape", "broadcast_in_dim", "transpose", "squeeze",
    "expand_dims", "slice", "dynamic_slice", "dynamic_update_slice", "concatenate", "pad", "gather",
    "copy", "copy_p", "rev", "iota", "sign", "floor", "ceil", "round", "reduce_max", "reduce_min",
    "argmax", "argmin", "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not", "xor", "clamp",
    "is_finite", "stop_gradient", "optimization_barrier", "cummax", "cummin", "sort",
    "shift_left", "shift_right_logical", "shift_right_arithmetic", "population_count", "rem",
    "name", "pjit_name", "nextafter", "random_bits", "random_seed", "random_wrap", "random_unwrap",
    "threefry2x32", "split", "scatter", "stack", "unstack", "bitcast_convert_type", "device_put",
    "sharding_constraint", "mesh_cast", "pvary", "reshard", "tie_in", "reduce_and", "reduce_or",
    "select_and_gather_add_exact", "empty2", "all_gather", "all_gather_invariant", "all_to_all", "ppermute",
    "pbroadcast", "axis_index"})
"""Operations whose float results are their operands' values moved or selected, or exact: no rounding.
An integer result rounds nothing either, nor does an array made to be written over (`empty2`); a collective
that moves blocks between devices (a gather, an all-to-all, a permutation) is a move too."""

COLLECTED = {"psum": "axes", "psum_invariant": "axes", "reduce_scatter": "axis_name"}
"""The collectives that sum over devices, by the parameter naming their axes: sums of terms, one a device of
those axes, bounded as any sum is (`_weight`)."""

ONCE = frozenset({"add", "add_any", "sub", "mul", "div", "sqrt", "square", "reduce_precision"})
"""IEEE 754's correctly rounded operations: one rounding, at most u times the result."""

ULPS = {"exp": 1.42, "log": 2.05, "log1p": 3.64, "expm1": 1.21, "logistic": 2.49, "tanh": 4.9,
        "erf": 6.71, "rsqrt": 2.8}
"""Each elementary function's error bound, in units of u times its true result: the largest of XLA
CPU's float32 implementations over every normal float32 input whose result is a normal float32, read
exhaustively against float64 references (2^32 inputs each, 11-02), rounded up at the second decimal.
XLA CPU flushes subnormal inputs to zero, so the bound holds where no elementary function reads one."""

SCATTERS = {"scatter": jax.lax.scatter, "scatter-add": jax.lax.scatter_add,
            "scatter-mul": jax.lax.scatter_mul, "scatter-min": jax.lax.scatter_min,
            "scatter-max": jax.lax.scatter_max}
"""The scatters by primitive, each of whose equations holds its combining computation in its operands' dtype,
which a float64 evaluation of a float32 program builds again (`TapRule.primitive`)."""

DRAWS = 8
"""The tap draws a rounding scale averages (`rounding_scale`): over 8 draws its root mean square is at least
0.41 of its expectation in 995 of 1000 elements (a chi-square of 8 degrees of freedom's 0.005 quantile is
1.344, and sqrt(1.344 / 8) = 0.41)."""

SCALES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""Each function's compiled rounding scale (`rounding_scale`) of a static program, by its arguments' tree,
shapes, dtypes and weak types and the matmul precision it traced under, held while the function lives."""


@dataclasses.dataclass(frozen=True)
class TapRule:
    """The walk's rule of a tapped evaluation (`float64_terms.Rule`): each primitive evaluated in float64,
    its parameters' dtypes widened, and each floating result it rounds moved by `scale` times its rounding's
    bound (module docstring) times a standard normal draw of its own, keyed by `key`, a uint64 counter mixed
    with the walk's path to it (`at`); `scale` 0 the untapped evaluation."""

    key: jax.Array
    scale: float

    def at(self, index: int) -> TapRule:
        return TapRule(_mixed(self.key + _GOLDEN * (jnp.asarray(index, jnp.int64).astype(jnp.uint64) + 1)),
                       self.scale)

    def primitive(self, equation, inputs: list) -> list:
        name, params = equation.primitive.name, dict(equation.params)
        for key in ("new_dtype", "preferred_element_type", "dtype"):
            if key in params:
                params[key] = _wide(params[key])
        if isinstance(params.get("precision"), jax.lax.DotAlgorithm | jax.lax.DotAlgorithmPreset):
            # An explicit algorithm fixes its products' and sums' format: the replay's are float64's.
            params["precision"] = jax.lax.DotAlgorithmPreset.F64_F64_F64
        operands = [_widened(value) for value in inputs]
        # A reduce_precision's float64 evaluation keeps its value, the rounding it is weighed.
        outputs = list(operands) if name == "reduce_precision" else _bound(equation, params, operands)
        if name in EXACT:
            return outputs
        tapped = []
        for index, (var, value) in enumerate(zip(equation.outvars, outputs, strict=True)):
            dtype = var.aval.dtype
            if not jnp.issubdtype(dtype, jnp.floating):
                tapped.append(value)
                continue
            weight = _weight(equation, name, params, operands, value)
            if weight is None:
                tapped.append(value)
                continue
            unit = _unit(name, params, dtype)
            tap = _mixed(self.key ^ (_GOLDEN * jnp.asarray(index + 1, jnp.uint64)))
            draw = _normal(tap, jnp.shape(value))
            tapped.append(value + self.scale * unit * weight * draw)
        return tapped


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
"""The 64-bit golden ratio, the step between the counters `_mixed` turns into independent words."""


def _mixed(x: jax.Array) -> jax.Array:
    """SplitMix64's finalizer of uint64 `x`: a word each bit of `x` reaches half the time (Steele, Lea and
    Flood, "Fast splittable pseudorandom number generators", 2014), the taps' counter-based generator."""
    x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def _normal(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Standard normal float64 draws of `shape` keyed by `key`, a uint64 word: two uniform draws an element,
    each counter's mixed word (`_mixed`), by Box and Muller's transform. A program holding a draw at every
    rounding compiles in seconds, where `jax.random`'s threefry keys and inverse error function at each
    took minutes."""
    count = int(np.prod(shape))
    counters = jnp.arange(2 * count, dtype=jnp.uint64).reshape((2, *shape))
    bits = _mixed(key + _GOLDEN * (counters + 1))
    # Each uniform in (0, 1] from the top 53 bits, so the logarithm is finite.
    uniform = ((bits >> 11).astype(jnp.float64) + 1) * 2.0 ** -53
    return jnp.sqrt(-2 * jnp.log(uniform[0])) * jnp.cos(2 * jnp.pi * uniform[1])


def rounding_scale(fn, *args, seed: int = 0):
    """`fn(*args)` in float64, and each floating output element's rounding scale c (module docstring), its
    rounding error's root mean square where each of the program's roundings moves its value by a standard
    normal multiple of its bound, over `DRAWS` draws keyed by `seed`, so the same draw of the same program
    weighs alike for every arm. The program is traced at `args` as given, its roundings those formats', and
    evaluated in float64; a non-floating output's scale is None. A static program (no branch or loop its
    values pick) evaluates compiled, once a signature and matmul precision; any other walks eagerly."""
    closed = jax.make_jaxpr(fn)(*args)
    tree = jax.tree.structure(jax.eval_shape(fn, *args))
    # Traced in the caller's arithmetic (its default matmul precision among it), replayed in float64's own.
    traced = str(jax.config.jax_default_matmul_precision)
    with evaluated_on(jax.tree.leaves(args)), jax.enable_x64(new_val=True), \
            jax.default_matmul_precision("highest"):
        leaves = [jnp.asarray(_widened(leaf)) for leaf in jax.tree.leaves(args)]
        consts = [_widened(value) for value in closed.consts]
        key = _mixed(jnp.asarray(seed, jnp.uint64))
        if _static(closed.jaxpr):
            signature = (jax.tree.structure(args), tuple((np.shape(leaf), jnp.result_type(leaf),
                                                          getattr(jax.typeof(leaf), "weak_type", False))
                                                         for leaf in jax.tree.leaves(args)),
                         traced, _target())
            held = SCALES.setdefault(fn, {})
            if signature not in held:
                jaxpr = closed.jaxpr
                # One tapped evaluation compiled, which every draw runs with its own key: a program of
                # one copy of `fn`, not one of each draw's.
                held[signature] = jax.jit(lambda consts, leaves, key, scale: _evaluated(
                    jaxpr, consts, leaves, rule=TapRule(key, scale))[0])
            outputs, scales = _scaled(closed.jaxpr, functools.partial(held[signature], consts, leaves), key)
        else:
            outputs, scales = _scaled(closed.jaxpr, lambda key, scale: _evaluated(
                closed.jaxpr, consts, leaves, rule=TapRule(key, scale))[0], key)
    return (tree.unflatten([np.asarray(value) for value in outputs]),
            tree.unflatten([None if scale is None else np.asarray(scale) for scale in scales]))


def _scaled(jaxpr, tapped, key):
    """`jaxpr`'s untapped float64 outputs and each floating output's scale (`rounding_scale`), from
    `tapped(key, scale)`, its evaluation with every rounding moved by `scale` times a draw keyed by `key`:
    the root mean square of each draw's displacement from the untapped evaluation, each element taken
    relative to its largest displacement, so no square overflows and a nonfinite displacement stays
    nonfinite."""
    plain = tapped(key, 0.0)
    draws = [tapped(_mixed(key + _GOLDEN * jnp.asarray(draw + 1, jnp.uint64)), 1.0) for draw in range(DRAWS)]
    scales = []
    for at, (var, value) in enumerate(zip(jaxpr.outvars, plain, strict=True)):
        if not jnp.issubdtype(var.aval.dtype, jnp.floating):
            scales.append(None)
            continue
        moved = jnp.stack([jnp.abs(draw[at] - value) for draw in draws])
        largest = jnp.max(moved, axis=0)
        held = jnp.where(largest > 0, largest, 1)
        scales.append(jnp.where(largest > 0, largest * jnp.sqrt(jnp.mean(jnp.square(moved / held), axis=0)),
                                largest))
    return plain, scales


def _widened(leaf):
    """`leaf` in float64 where it is a float (bfloat16 and float16 among them), as it is otherwise."""
    if jnp.issubdtype(jnp.result_type(leaf), jnp.floating):
        return jnp.asarray(leaf, jnp.float64)
    return leaf


def _wide(dtype):
    """A float dtype parameter as float64, any other as it is."""
    return jnp.float64 if dtype is not None and jnp.issubdtype(dtype, jnp.floating) else dtype


def _weight(equation, name, params, operands, value):
    """The most one rounding at `equation` moves its float64 result `value` (module docstring), in units of
    its format's u; None where it rounds nothing."""
    magnitude = jnp.abs(value)
    if name == "convert_element_type":
        return _converted(equation, operands[0], magnitude)
    if name in ONCE:
        return magnitude
    if name in ULPS:
        return ULPS[name] * magnitude
    if name == "integer_pow":
        return max(abs(int(params["y"])) - 1, 1) * magnitude
    if name == "one_minus_square":
        # The square rounds by its magnitude, which the difference carries, and the difference by its own.
        return jnp.square(operands[0]) + magnitude
    if name not in SUMS and name not in COLLECTED:
        raise ValueError(f"the rounding scale has no rule for {name}: add it to EXACT, ONCE, ULPS, COLLECTED "
                         "or float64_terms.SUMS")
    (terms,) = _bound(equation, params, [jnp.abs(operand) for operand in operands])
    return _terms_count(equation, name, params) * terms


def _bound(equation, params, operands) -> list:
    """`equation`'s primitive applied to `operands`, widened past its program's dtypes, with `params`: a
    scatter's combining computation, traced for its program's dtype, built again for theirs (`SCATTERS`)."""
    name = equation.primitive.name
    if name in SCATTERS:
        operand, indices, updates = operands
        return [SCATTERS[name](operand, indices, updates, params["dimension_numbers"],
                               indices_are_sorted=params["indices_are_sorted"],
                               unique_indices=params["unique_indices"], mode=params["mode"])]
    outputs = equation.primitive.bind(*operands, **equation.primitive.get_bind_params(params))
    return list(outputs) if equation.primitive.multiple_results else [outputs]


def _converted(equation, operand, magnitude):
    """A conversion's rounding weight: a float narrowed rounds by its magnitude, an integer converted to a
    float too narrow to hold it exactly (at or past 2^(its significand's bits)) by its magnitude, each
    element its own; a widening rounds nothing."""
    source, target = equation.invars[0].aval.dtype, equation.outvars[0].aval.dtype
    if not jnp.issubdtype(target, jnp.floating):
        return None
    if jnp.issubdtype(source, jnp.floating):
        # A narrower significand or exponent rounds (float16 to bfloat16 alike: same width, fewer bits).
        narrower = (jnp.finfo(target).nmant < jnp.finfo(source).nmant
                    or jnp.finfo(target).maxexp < jnp.finfo(source).maxexp)
        return magnitude if narrower else None
    exact = 2.0 ** (jnp.finfo(target).nmant + 1)
    return jnp.where(jnp.abs(jnp.asarray(operand, jnp.float64)) >= exact, magnitude, 0)


def _unit(name: str, params, dtype) -> float:
    """The unit roundoff of the format an equation rounds its result `dtype` to: a reduce_precision's
    stated significand (`mantissa_bits`), else its result's own (`jnp.finfo`), the format it is stored in
    being the one it rounds to."""
    if name == "reduce_precision":
        return 2.0 ** -(int(params["mantissa_bits"]) + 1)
    return float(jnp.finfo(dtype).eps) / 2


def _terms_count(equation, name, params) -> int:
    """How many terms each element of a sum `equation` adds (`float64_terms.SUMS`, `COLLECTED`), its rounding
    weight's factor: an inner product's contraction length, a reduction's `m - 1`, a convolution's window
    times its input features a group, a scatter's updates an element at most, a device sum's devices less
    one."""
    shape = equation.invars[0].aval.shape
    if name == "dot_general":
        (contracting, _), _ = params["dimension_numbers"]
        return int(np.prod([shape[axis] for axis in contracting]))
    if name == "ragged_dot_general":
        (contracting, _), _ = params["ragged_dot_dimension_numbers"].dot_dimension_numbers
        return int(np.prod([shape[axis] for axis in contracting]))
    if name == "conv_general_dilated":
        kernel = equation.invars[1].aval.shape
        numbers = params["dimension_numbers"]
        features = kernel[numbers.rhs_spec[1]]
        return int(features * np.prod([kernel[axis] for axis in numbers.rhs_spec[2:]]))
    if name == "reduce_sum":
        return max(int(np.prod([shape[axis] for axis in params["axes"]])) - 1, 1)
    if name == "scatter-add":
        numbers = params["dimension_numbers"]
        updates = equation.invars[2].aval.shape
        return int(np.prod([size for axis, size in enumerate(updates)
                            if axis not in numbers.update_window_dims]))
    if name in COLLECTED:
        axes = params[COLLECTED[name]]
        named = [axis for axis in (axes if isinstance(axes, tuple | list) else (axes,))
                 if isinstance(axis, str)]
        return max(int(np.prod([jax.lax.axis_size(axis) for axis in named])) - 1, 1)
    return max(shape[params["axis"]] - 1, 1)
