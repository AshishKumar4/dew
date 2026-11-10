"""A program's rounding scale, read from its jaxpr in float64 (`rounding_scale`): what check 1's conditioning
weighs an element's error by (tests/reference_error.py, `assert_as_exact`'s `conditioning`).

Each rounding the program makes moves a value v by at most w(v) u, u its format's unit roundoff: a correctly
rounded operation (an add, a product, a quotient, a root, a narrowing conversion) by its result's
magnitude, a sum of m terms in any order by (m - 1) times the sum of the terms' magnitudes, an inner
product of m terms by m times it (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., 2002,
sections 2.2, 3.1 and 4.2), an integer converted to a float too narrow to hold it by its magnitude, and an
elementary function by its implementation's ulps times its result (`ULPS`). The program is evaluated in
float64 with each rounding moved by w(v) u times a random sign, + or - alike (a tap, a draw of unit variance),
every operation its own draw (`TapRule`), and each output element's scale c is the root mean square, over
`DRAWS` draws, of how
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
from arm_arithmetic import unrewritten
from float64_terms import SUMS, _static, _target, evaluated_on, walked
from jax.extend.core import primitives

from _dew_torch.rounding import DRAWS, ULPS
from dew.nn import programs
from dew.nn.precision import compensated_add, correctly_rounded_divide_p

EXACT = frozenset({jax.lax.neg_p, jax.lax.abs_p, jax.lax.max_p, jax.lax.min_p, jax.lax.select_n_p,
    jax.lax.copy_p, jax.lax.iota_p, jax.lax.sign_p, jax.lax.floor_p, jax.lax.ceil_p, jax.lax.round_p,
    jax.lax.reduce_max_p, jax.lax.reduce_min_p, jax.lax.argmax_p, jax.lax.argmin_p, jax.lax.eq_p,
    jax.lax.ne_p, jax.lax.lt_p, jax.lax.le_p, jax.lax.gt_p, jax.lax.ge_p, jax.lax.and_p, jax.lax.or_p,
    jax.lax.not_p, jax.lax.xor_p, jax.lax.clamp_p, jax.lax.is_finite_p, jax.lax.stop_gradient_p,
    jax.lax.optimization_barrier_p, jax.lax.cummax_p, jax.lax.cummin_p, jax.lax.sort_p, jax.lax.shift_left_p,
    jax.lax.shift_right_logical_p, jax.lax.shift_right_arithmetic_p, jax.lax.population_count_p,
    jax.lax.rem_p, primitives.name_p, jax.lax.nextafter_p, primitives.random_bits_p, primitives.random_seed_p,
    primitives.random_wrap_p, primitives.random_unwrap_p, primitives.threefry2x32_p, jax.lax.scatter_p,
    jax.lax.stack_p, jax.lax.unstack_p, jax.lax.bitcast_convert_type_p, jax.lax.device_put_p,
    jax.lax.sharding_constraint_p, primitives.pvary_p, primitives.reshard_p, jax.lax.reduce_and_p,
    jax.lax.reduce_or_p, primitives.empty2_p, jax.lax.all_gather_p, primitives.all_gather_invariant_p,
    jax.lax.all_to_all_p, jax.lax.ppermute_p, primitives.pbroadcast_p, jax.lax.axis_index_p})
"""Operations whose float results are their operands' values selected, or exact: no rounding. An operation
that only moves entries is exact too (`dew.nn.programs.rearranges`, read beside these, `_exact`). An integer
result rounds nothing either, nor does an array made to be written over (`empty2`); a collective that moves
blocks between devices (a gather, an all-to-all, a permutation) is a move too."""


def _exact(primitive) -> bool:
    """Whether `primitive` rounds nothing: a move of entries, or one of `EXACT`."""
    return programs.rearranges(primitive) or primitive in EXACT


COLLECTED = {jax.lax.psum_p: "axes", primitives.psum_invariant_p: "axes",
             primitives.reduce_scatter_p: "axis_name"}
"""The collectives that sum over devices, by the parameter naming their axes: sums of terms, one a device of
those axes, bounded as any sum is (`_weight`)."""

ONCE = frozenset({jax.lax.add_p, primitives.add_jaxvals_p, jax.lax.sub_p, jax.lax.mul_p, jax.lax.div_p,
                  jax.lax.sqrt_p, jax.lax.square_p, jax.lax.reduce_precision_p, correctly_rounded_divide_p})
"""IEEE 754's correctly rounded operations, Dew's correctly rounded division among them: one rounding, at
most u times the result."""

FUNCTIONS = {getattr(jax.lax, f"{name}_p"): bound for name, bound in ULPS.items()}
"""Each elementary function's error bound (`ULPS`, by the function's name in JAX and torch alike), by JAX's
primitive of that name."""

SCATTERS = {jax.lax.scatter_p: jax.lax.scatter, jax.lax.scatter_add_p: jax.lax.scatter_add,
            jax.lax.scatter_mul_p: jax.lax.scatter_mul, jax.lax.scatter_min_p: jax.lax.scatter_min,
            jax.lax.scatter_max_p: jax.lax.scatter_max}
"""The scatters by primitive, each of whose equations holds its combining computation in its operands' dtype,
which a float64 evaluation of a float32 program builds again (`TapRule.primitive`)."""

SCALES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""Each function's compiled rounding scale (`rounding_scale`) of a static program, by its arguments' tree,
shapes, dtypes and weak types and the matmul precision it traced under, held while the function lives."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TapRule:
    """The walk's rule of a tapped evaluation (`float64_terms.Rule`): each primitive evaluated in float64,
    its parameters' dtypes widened, and each floating result it rounds moved by `scale` times its rounding's
    bound (module docstring) times a random sign of its own, keyed by `key`, a uint64 counter mixed
    with the walk's path to it (`at`); `scale` 0 the untapped evaluation."""

    key: jax.Array
    scale: float
    pruned = True

    def at(self, index: int, name: str | None = None) -> TapRule:
        return TapRule(_mixed(self.key + _GOLDEN * (jnp.asarray(index, jnp.int64).astype(jnp.uint64) + 1)),
                       self.scale)

    def held(self, dtype):
        return _wide(dtype)

    def primitive(self, equation, inputs: list) -> list:
        primitive, params = equation.primitive, dict(equation.params)
        for key in ("new_dtype", "preferred_element_type", "dtype"):
            if key in params:
                params[key] = _wide(params[key])
        if isinstance(params.get("precision"), jax.lax.DotAlgorithm | jax.lax.DotAlgorithmPreset):
            # An explicit algorithm fixes its products' and sums' format: the replay's are float64's.
            params["precision"] = jax.lax.DotAlgorithmPreset.F64_F64_F64
        operands = [_widened(value) for value in inputs]
        # A reduce_precision's float64 evaluation keeps its value, the rounding it is weighed.
        rounded_to = primitive is jax.lax.reduce_precision_p
        outputs = list(operands) if rounded_to else _bound(equation, params, operands)
        if _exact(primitive):
            return outputs
        tapped = []
        for index, (var, value) in enumerate(zip(equation.outvars, outputs, strict=True)):
            dtype = var.aval.dtype
            if not jnp.issubdtype(dtype, jnp.floating):
                tapped.append(value)
                continue
            weight = _weight(equation, params, operands, value)
            if weight is None:
                tapped.append(value)
                continue
            unit = _unit(primitive, params, dtype)
            tap = _mixed(self.key ^ (_GOLDEN * jnp.asarray(index + 1, jnp.uint64)))
            draw = _signs(tap, jnp.shape(value))
            # An infinite result (a masked logit's -inf carried on) is exact: nothing rounds it, and its
            # weight, a multiple of its magnitude, is no rounding's.
            weight = jnp.where(jnp.isfinite(value), weight, 0.0)
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


def _signs(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Random signs of `shape`, each + or - alike, as float64, keyed by `key`, a uint64 word: each element's
    counter's mixed word's top bit (`_mixed`). A draw of unit variance, as a standard normal's, whose
    displacements' mean square is the same sum of squared weights (Hutchinson's estimator, "A stochastic
    estimator of the trace of the influence matrix", 1990, draws signs so too); a program holding a draw at
    every rounding compiles in seconds, where normals' logarithm and cosine at each took minutes."""
    counters = jnp.arange(int(np.prod(shape)), dtype=jnp.uint64).reshape(shape)
    bits = _mixed(key + _GOLDEN * (counters + 1))
    return 1.0 - 2.0 * (bits >> 63).astype(jnp.float64)


def rounding_scale(fn, *args, seed: int = 0):
    """`fn(*args)` in float64, and each floating output element's rounding scale c (module docstring), its
    rounding error's root mean square where each of the program's roundings moves its value by a random
    sign times its bound, over `DRAWS` draws keyed by `seed`, so the same draw of the same program
    weighs alike for every arm. The program is traced at `args` as given, its roundings those formats', and
    evaluated in float64; a non-floating output's scale is None. A static program (no branch or loop its
    values pick) evaluates compiled, once a signature and matmul precision; any other walks eagerly."""
    # The program as written: traced with the algebra pass off, a reference's (`arm_arithmetic.unrewritten`).
    written = unrewritten(fn)
    closed = jax.make_jaxpr(written)(*args)
    tree = jax.tree.structure(jax.eval_shape(written, *args))
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
                held[signature] = jax.jit(lambda consts, leaves, key, scale: walked(
                    jaxpr, consts, leaves, TapRule(key, scale)))
            outputs, scales = _scaled(closed.jaxpr, functools.partial(held[signature], consts, leaves), key)
        else:
            # Its runs of static steps between a branch and a loop compiled, the rule passed in (`_span`).
            outputs, scales = _scaled(closed.jaxpr, lambda key, scale: walked(
                closed.jaxpr, consts, leaves, TapRule(key, jnp.float64(scale)), compiling=True), key)
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


def _weight(equation, params, operands, value):
    """The most one rounding at `equation` moves its float64 result `value` (module docstring), in units of
    its format's u; None where it rounds nothing. An operation with no rule is refused by name."""
    primitive, magnitude = equation.primitive, jnp.abs(value)
    if primitive is jax.lax.convert_element_type_p:
        return _converted(equation, operands[0], magnitude)
    if primitive in ONCE:
        return magnitude
    if primitive in FUNCTIONS:
        return FUNCTIONS[primitive] * magnitude
    if primitive is jax.lax.integer_pow_p:
        return max(abs(int(params["y"])) - 1, 1) * magnitude
    if primitive is jax.lax.reduce_p and params["computation"] is compensated_add:
        # Dew's compensated sum (`dew.nn.precision.compensated_sum`): a double-float pair's sum, its error
        # at most u |sum| + gamma_{n-1}^2 sum |x| (Ogita, Rump and Oishi, Sum2), in units of u; each of
        # the pair's halves by it, as the high half's rounding moves the low half by as much.
        terms = int(np.prod([equation.invars[0].aval.shape[axis] for axis in params["dimensions"]]))
        u = _unit(primitive, params, equation.outvars[0].aval.dtype)
        gamma = (terms - 1) * u / (1 - (terms - 1) * u)
        pair = operands[:len(operands) // 2]
        total = jnp.abs(sum(jnp.sum(operand, params["dimensions"]) for operand in pair))
        spread = sum(jnp.sum(jnp.abs(operand), params["dimensions"]) for operand in pair)
        return total + gamma * gamma / u * spread
    if (primitive is jax.lax.reduce_p
            and isinstance(started := equation.invars[len(equation.invars) // 2], jax.extend.core.Literal)
            and started.val == -np.inf):
        # A running logsumexp (`dew.objectives.lm.chunked`'s, its maximum started at -inf), in float64
        # under Dew's wide CPU arithmetic (`dew.nn.precision.widened_reduction`) and float32 elsewhere:
        # each combine rescales a running sum by exp(a - larger), the difference rounding at most u
        # times the row's range, which moves the exponential by as much relative, beside the
        # exponential's ulps and two roundings of the result's magnitude.
        terms = int(np.prod([equation.invars[0].aval.shape[axis] for axis in params["dimensions"]]))
        # A masked column (-inf, `chunked._without`) holds no mass and rescales nothing: the range is the
        # finite logits', none where the row holds no finite one.
        logits = operands[0]
        finite = jnp.isfinite(logits)
        highest = jnp.max(jnp.where(finite, logits, -jnp.inf), params["dimensions"])
        lowest = jnp.min(jnp.where(finite, logits, jnp.inf), params["dimensions"])
        spread = jnp.where(jnp.any(finite, params["dimensions"]), highest - lowest, 0.0)
        return max(terms - 1, 1) * (2 + ULPS["exp"] + spread) * magnitude
    if primitive is primitives.one_minus_square_p:
        # The square rounds by its magnitude, which the difference carries, and the difference by its own.
        return jnp.square(operands[0]) + magnitude
    if primitive not in SUMS and primitive not in COLLECTED:
        raise ValueError(f"the rounding scale has no rule for {primitive.name}: give it one in EXACT, ONCE, "
                         "ULPS, COLLECTED or float64_terms.SUMS")
    (terms,) = _bound(equation, params, [jnp.abs(operand) for operand in operands])
    return _terms_count(equation, params) * terms


def _bound(equation, params, operands) -> list:
    """`equation`'s primitive applied to `operands`, widened past its program's dtypes, with `params`: a
    scatter's combining computation, traced for its program's dtype, built again for theirs (`SCATTERS`)."""
    primitive = equation.primitive
    if primitive is correctly_rounded_divide_p:
        # Its float32 rounding is what it is weighed for (`ONCE`): the replay's quotient is float64's.
        return [operands[0] / operands[1]]
    if primitive is jax.lax.reduce_p:
        # Its combining computation, traced for its program's dtypes, is called again on theirs.
        count = len(operands) // 2
        return list(jax.lax.reduce(tuple(operands[:count]), tuple(operands[count:]), params["computation"],
                                   params["dimensions"]))
    if primitive in SCATTERS:
        operand, indices, updates = operands
        return [SCATTERS[primitive](operand, indices, updates, params["dimension_numbers"],
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


def _unit(primitive, params, dtype) -> float:
    """The unit roundoff of the format an equation rounds its result `dtype` to: a reduce_precision's
    stated significand (`mantissa_bits`), else its result's own (`jnp.finfo`), the format it is stored in
    being the one it rounds to."""
    if primitive is jax.lax.reduce_precision_p:
        return 2.0 ** -(int(params["mantissa_bits"]) + 1)
    return float(jnp.finfo(dtype).eps) / 2


def _terms_count(equation, params) -> int:
    """How many terms each element of a sum `equation` adds (`float64_terms.SUMS`, `COLLECTED`), its rounding
    weight's factor: an inner product's contraction length, a reduction's `m - 1`, a convolution's window
    times its input features a group, a scatter's updates an element at most, a device sum's devices less
    one."""
    primitive, shape = equation.primitive, equation.invars[0].aval.shape
    if primitive is jax.lax.dot_general_p:
        (contracting, _), _ = params["dimension_numbers"]
        return int(np.prod([shape[axis] for axis in contracting]))
    if primitive is primitives.ragged_dot_general_p:
        (contracting, _), _ = params["ragged_dot_dimension_numbers"].dot_dimension_numbers
        return int(np.prod([shape[axis] for axis in contracting]))
    if primitive is jax.lax.conv_general_dilated_p:
        kernel = equation.invars[1].aval.shape
        numbers = params["dimension_numbers"]
        features = kernel[numbers.rhs_spec[1]]
        return int(features * np.prod([kernel[axis] for axis in numbers.rhs_spec[2:]]))
    if primitive is jax.lax.reduce_sum_p:
        return max(int(np.prod([shape[axis] for axis in params["axes"]])) - 1, 1)
    if primitive is jax.lax.scatter_add_p:
        numbers = params["dimension_numbers"]
        updates = equation.invars[2].aval.shape
        return int(np.prod([size for axis, size in enumerate(updates)
                            if axis not in numbers.update_window_dims]))
    if primitive in COLLECTED:
        axes = params[COLLECTED[primitive]]
        named = [axis for axis in (axes if isinstance(axes, tuple | list) else (axes,))
                 if isinstance(axis, str)]
        return max(int(np.prod([jax.lax.axis_size(axis) for axis in named])) - 1, 1)
    return max(shape[params["axis"]] - 1, 1)
