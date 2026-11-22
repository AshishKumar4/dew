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
from collections.abc import Hashable

import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import (
    FORMATTED,
    as_traced,
    on_float64_device,
    on_float64_host,
    rounds_nothing,
    unrewritten,
)
from float64_terms import SPAN_STEPS, SUMS, _target, evaluated_on, released, static, steps, walked
from jax.extend.core import primitives

from _dew_torch.rounding import DRAWS, ULPS
from dew.nn import programs
from dew.nn.precision import compensated_add, compensated_sum, correctly_rounded_divide_p, summed_pairs

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

@functools.cache
def compensating() -> frozenset:
    """The combining computations of Dew's compensated sums, each a double-float pair's, its first two
    operands and results the pair, any third the plain sum carried beside it: `compensated_add`, and the
    ones `compensated_sum`'s reduction (`summed_total`, the pair and its plain sum in one pass) and the pair
    reduction alone (`summed_pairs`) combine by, read from their own traces, so a renamed combine is still
    known by its object."""
    shape = jax.ShapeDtypeStruct((2,), jnp.float32)
    combines = set()
    for summed in (lambda x: compensated_sum(x, (0,)), lambda x: summed_pairs((x, jnp.zeros_like(x)), (0,))):
        traced = jax.make_jaxpr(summed)(shape)
        combines |= {equation.params["computation"] for equation in programs.equations(traced.jaxpr)
                     if equation.primitive is jax.lax.reduce_p}
    return frozenset({compensated_add, *combines})


SCALES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""Each function's compiled rounding scale (`rounding_scale`) of a static program, by its trace's key
(`_key`) and the device it evaluates on, held while the function lives."""

TRACED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""Each function's program as `rounding_scale` traces it (`_traced`), by its trace's key (`_key`), held
while the function lives: a weighed site's recording asks its reference's scale at many draws of one
program, which traced it again at each (tests/test_weighed_recording.py 77.3 s, 70.8 s held; row 5864)."""


def released_scales() -> None:
    """Forget every program `rounding_scale` holds (`TRACED`, `SCALES`) and the walk's traces and compiled
    runs (`float64_terms.released`): a conditioned site's weighing's, once it is done (`reference_error`'s
    `_deferred`), so a process weighing many sites holds one site's at a time (row 9780)."""
    TRACED.clear()
    SCALES.clear()
    released()


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
        # A reduce_precision's float64 evaluation keeps its value, the rounding it is weighed; any other
        # operation whose meaning is its format's (`FORMATTED`: a bit cast, a step to a neighbour) runs at the
        # dtypes it was traced at (`as_traced`, the float64 replay's rule), its float results widened after
        # (bound on float64, a float32 read as uint32 was two words an element).
        rounded_to = primitive is jax.lax.reduce_precision_p
        if rounded_to:
            outputs = list(operands)
        elif primitive in FORMATTED:
            outputs = [_widened(value) for value in as_traced(equation, operands)]
        else:
            outputs = _bound(equation, params, operands)
        if rounds_nothing(primitive):
            return outputs
        tapped = []
        for index, (var, value) in enumerate(zip(equation.outvars, outputs, strict=True)):
            dtype = var.aval.dtype
            if not jnp.issubdtype(dtype, jnp.floating):
                tapped.append(value)
                continue
            weight = _weight(equation, params, operands, value, index)
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
    Flood, "Fast splittable pseudorandom number generators", 2014), the taps' counter-based generator; in
    JAX's arithmetic or NumPy's, wrapping alike (a draw's keys are NumPy's, `_keys`)."""
    x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def _keys(seed: int) -> tuple[np.uint64, list[np.uint64]]:
    """The keys of `rounding_scale`'s evaluations at `seed`: the untapped one's and each of its DRAWS
    draws', mixed on the host (`_mixed` in NumPy's uint64, which wraps as JAX's does), so a scale makes
    no host-to-device round trip a key."""
    with np.errstate(over="ignore"):
        key = _mixed(np.uint64(seed))
        return key, [_mixed(key + _GOLDEN * np.uint64(draw + 1)) for draw in range(DRAWS)]


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
    values pick) of at most `SPAN_STEPS` steps evaluates compiled whole, once a trace key; any other walks
    its static runs compiled, a span at a time."""
    key = _key(args)
    closed, tree = _traced(fn, args, key)
    moved = on_float64_host(jax.tree.leaves(args))
    with on_float64_device(), evaluated_on(moved), jax.enable_x64(new_val=True), \
            jax.default_matmul_precision("highest"):
        leaves = [jnp.asarray(_widened(leaf)) for leaf in moved]
        consts = [_widened(value) for value in on_float64_host(closed.consts)]
        keys = _keys(seed)
        # A static program compiles whole, one of more than a span's steps (its calls' bodies' counted,
        # `float64_terms.steps`; row 10950) in spans (`float64_terms._runs`).
        if static(closed.jaxpr) and steps(closed.jaxpr) <= SPAN_STEPS:
            held = SCALES.setdefault(fn, {})
            # Compiled for the device it evaluates on (`float64_terms.evaluated_on`), which its trace is not.
            compiled = (key, _target())
            if compiled not in held:
                jaxpr = closed.jaxpr
                # One tapped evaluation compiled, which every draw runs with its own key: a program of
                # one copy of `fn`, not one of each draw's.
                held[compiled] = jax.jit(lambda consts, leaves, key, scale: walked(
                    jaxpr, consts, leaves, TapRule(key, scale)))
            outputs, scales = _scaled(closed.jaxpr, functools.partial(held[compiled], consts, leaves), keys)
        else:
            # Its runs of static steps between a branch and a loop compiled, the rule passed in (`_span`).
            outputs, scales = _scaled(closed.jaxpr, lambda key, scale: walked(
                closed.jaxpr, consts, leaves, TapRule(key, jnp.float64(scale)), compiling=True), keys)
    return (tree.unflatten([np.asarray(value) for value in outputs]),
            tree.unflatten([None if scale is None else np.asarray(scale) for scale in scales]))


def _key(args) -> Hashable:
    """What a program `rounding_scale` traces at `args` is held by (`TRACED`; `SCALES` beside the device it
    evaluates on): the arguments as a lowering reads them (`options.abstract_signature`) and the context a
    trace reads beside them (`options.trace_context`: JAX's traced configuration, the caller's matmul
    precision and x64 among it, its meshes, Dew's contexts, the logical axis rules and Dew's environment),
    the one key a held trace is kept by (row 10878: a second, hand-picked here, missed Dew's kernel
    choices)."""
    from dew.nn.kernels import options

    return options.abstract_signature(args), options.trace_context()


def _traced(fn, args, key: Hashable) -> tuple:
    """`fn`'s program as written at `args`, traced with the algebra pass off, a reference's
    (`arm_arithmetic.unrewritten`), and its outputs' tree from the same trace, once a key (`_key`,
    `TRACED`): a function is its program, as `SCALES` holds it."""
    held = TRACED.setdefault(fn, {})
    if key not in held:
        closed, shapes = jax.make_jaxpr(unrewritten(fn), return_shape=True)(*args)
        held[key] = (closed, jax.tree.structure(shapes))
    return held[key]


def _scaled(jaxpr, tapped, keys):
    """`jaxpr`'s untapped float64 outputs and each floating output's scale (`rounding_scale`), from
    `tapped(key, scale)`, its evaluation with every rounding moved by `scale` times a draw keyed by `key`
    (`keys`: the untapped evaluation's and each draw's, `_keys`): the root mean square of each draw's
    displacement from the untapped evaluation, folded in a draw at a time (`_folded`), so the untapped
    outputs, one draw's and the running fold are all it holds, where it held every draw's at once (row 9780:
    a 10-layer model's loss and gradients, nine float64 copies of them, were killed for memory)."""
    key, drawn = keys
    plain = tapped(key, 0.0)
    floating = [jnp.issubdtype(var.aval.dtype, jnp.floating) for var in jaxpr.outvars]
    # The largest displacement at the outputs' own width (a float64 output's may lie below float32's range),
    # the scaled squares, at most the count, in float32: held, the untapped outputs, one draw's and about
    # one more output's (rows 9780 on: a 2 GiB float64 gradient tree's walk peaked at 6.15 GiB).
    folds = [(jnp.zeros_like(value), jnp.zeros(jnp.shape(value), jnp.float32)) if kept else None
             for value, kept in zip(plain, floating, strict=True)]
    for each in drawn:
        draw = tapped(each, 1.0)
        folds = [None if fold is None else _folded(value, moved, *fold)
                 for value, moved, fold in zip(plain, draw, folds, strict=True)]
        del draw
    return plain, [None if fold is None else _spread(*fold, len(drawn)) for fold in folds]


@jax.jit
def _folded(value, draw, largest, squares):
    """A draw's displacement from `value` folded into its running root mean square: the largest
    displacement so far and the sum of the squares of each displacement relative to it, rescaled where a
    larger one comes (LAPACK's `dnrm2` update), so no square overflows and a nonfinite displacement keeps the
    fold nonfinite. One program a shape, where its operations eagerly were a dispatch each."""
    moved = jnp.abs(draw - value)
    larger = moved > largest
    held = jnp.where(larger, moved, jnp.where(largest > 0, largest, 1))
    # Each ratio at most 1 at the outputs' width, then the sum's float32 (`_scaled`).
    squares = jnp.where(larger, squares * jnp.square(largest / held).astype(squares.dtype) + 1,
                        squares + jnp.where(largest > 0, jnp.square(moved / held), 0).astype(squares.dtype))
    return jnp.maximum(largest, moved), squares


def _spread(largest, squares, count: int):
    """The root mean square of `count` folded displacements (`_folded`), zero where none moved, at the
    largest's width."""
    return jnp.where(largest > 0, largest * jnp.sqrt(squares / count).astype(largest.dtype), largest)


def _widened(leaf):
    """`leaf` in float64 where it is a float (bfloat16 and float16 among them), as it is otherwise."""
    if jnp.issubdtype(jnp.result_type(leaf), jnp.floating):
        return jnp.asarray(leaf, jnp.float64)
    return leaf


def _wide(dtype):
    """A float dtype parameter as float64, any other as it is."""
    return jnp.float64 if dtype is not None and jnp.issubdtype(dtype, jnp.floating) else dtype


def _weight(equation, params, operands, value, index: int = 0):
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
    if primitive is jax.lax.pow_p:
        # A float's power (Adam's bias correction, `1 - b**count`): at an integer exponent rounding as an
        # integer power's, `max(|e| - 1, 1)` times, else through `exp(e log x)`, by each function's ulps;
        # `_dew_torch.rounding`'s rule for aten.pow, element by element.
        exponent = jnp.broadcast_to(operands[1], jnp.shape(value))
        whole = jnp.maximum(jnp.abs(exponent) - 1, 1)
        return jnp.where(exponent == jnp.round(exponent), whole, ULPS["exp"] + ULPS["log"]) * magnitude
    if primitive is jax.lax.reduce_p and params["computation"] in compensating():
        # Dew's compensated sum (`dew.nn.precision.compensated_sum`): a double-float pair's sum, its error
        # at most u |sum| + gamma_{n-1}^2 sum |x| (Ogita, Rump and Oishi, Sum2), in units of u; each of
        # the pair's halves by it, as the high half's rounding moves the low half by as much. A plain sum
        # carried beside the pair (`summed_total`'s third operand and result) rounds as any sum of its
        # terms does, in whatever order the reduction adds them.
        terms = int(np.prod([equation.invars[0].aval.shape[axis] for axis in params["dimensions"]]))
        summed = operands[:len(equation.outvars)]
        if index >= 2:
            return max(terms - 1, 1) * jnp.sum(jnp.abs(summed[index]), params["dimensions"])
        u = _unit(primitive, params, equation.outvars[0].aval.dtype)
        gamma = (terms - 1) * u / (1 - (terms - 1) * u)
        pair = summed[:2]
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
    unrounded = programs.UNROUNDED.get(primitive)
    if unrounded is not None:
        # Its rounding to its own format is what it is weighed for (`ONCE`): widened, it is the operation
        # it rounds.
        return [unrounded(*operands)]
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
