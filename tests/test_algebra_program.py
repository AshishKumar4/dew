"""The algebra pass where Dew compiles its programs (docs/design/algebra.md §2, §6.7): every outer program and
the resident training step jit their function rewritten, under the switch the gate opens per arm (`passes`);
the bodies of nested programs, scans, loops, conditionals and checkpoints are rewritten in place; products
(`dot_general`) are terms the rules read; and each rewrite is recorded in the one kernel report."""

import dataclasses
import gc
import json
import os
import subprocess
import sys
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, assert_computes_the_oracle, equations
from test_algebra import cheaper_spliced, count_above, rows, traces_cheaper, two_sums

from dew import algebra
from dew.algebra import cost, program, terms
from dew.nn import programs
from dew.nn.kernels import generation, options
from dew.nn.kernels.templates import cost as kernel_cost, report
from dew.nn.precision import FP32_SUMS


def nested(kind: str):
    """`two_sums` inside a body of `kind`: a nested jit, a checkpoint, a scan over two steps, a conditional,
    a loop of one turn. A nested jit's and a checkpoint's function is each trace's own, so no trace reads
    another's body from JAX's cache and each records its own rewrite."""
    def body(x, y, a):
        if kind == "jit":
            return jax.jit(lambda x, y, a: two_sums(x, y, a))(x, y, a)
        if kind == "checkpoint":
            return jax.checkpoint(lambda x, y, a: two_sums(x, y, a))(x, y, a)
        if kind == "scan":
            return jax.lax.scan(lambda carry, step: (carry + two_sums(x * step, y, a), None),
                                jnp.zeros(x.shape[:-1], x.dtype), jnp.arange(2.0, dtype=x.dtype))[0]
        if kind == "cond":
            return jax.lax.cond(x[0, 0] > 0, two_sums, lambda x, y, a: two_sums(y, x, a), x, y, a)
        return jax.lax.while_loop(lambda state: state[0] < 1,
                                  lambda state: (state[0] + 1, two_sums(x, y, a)),
                                  (0, jnp.zeros(x.shape[:-1], x.dtype)))[1]
    return body


KINDS = ("jit", "checkpoint", "scan", "cond", "while")


def test_an_infinite_scale_stays_inside_a_maximum():
    """`max(x * inf)` over `x = [0, 1, ..., 1023]`: `0 * inf` is NaN, so the maximum is NaN; the pass took an
    infinite scale for a positive one and hoisted it out of the maximum, `max(x) * inf`, infinite, its work
    the less (spec alignment, 8992). A scale out of an extremum is a finite positive real's alone."""
    def scaled(x):
        return jnp.max(x * jnp.inf)

    x = jnp.arange(1024, dtype=jnp.float32)
    assert np.isnan(np.asarray(jax.jit(algebra.rewritten(scaled))(x)))


def test_a_window_proves_nothing_of_a_value_its_program_proves_no_number():
    """`sum(tanh(a / b) * c)`, its quotient outside the window a reduction heads (three equations deep):
    `a / b` may be `0 / 0`, so the program proves `tanh` of it no number and bounds nothing it makes; each
    window cut from it holds no bound and no finiteness the program does not (`program._declared`: the
    quotient declared as the program proves it, `terms.Proven`), where the window took its input for a finite
    real and bounded the tanh, its product and its sum, which a reordering reads (spec alignment, 8993)."""
    from jax.extend import core as jcore

    def program_of(a, b, c):
        return jnp.sum(jnp.tanh(a / b) * c)

    closed = jax.make_jaxpr(program_of)(*(jnp.ones(8, jnp.float32) for _ in range(3)))
    whole = terms.region(closed)
    windows = program._windows(closed.jaxpr)
    assert windows
    proven = program._held(closed.jaxpr)
    for window, _ in windows:
        cut = terms.region(jcore.ClosedJaxpr(window, []), program._declared(window, proven))
        made = [var for equation in window.eqns for var in equation.outvars]
        assert [var for var in made if cut.held.get(var) is not None and whole.held.get(var) is None] == []
        assert [var for var in made if cut.apart[var].finite and not whole.apart[var].finite] == []


@pytest.mark.parametrize(("function", "finite"), [(jnp.tanh, True), (jax.lax.erf, True), (jnp.sin, False)],
                         ids=["tanh", "erf", "sin"])
def test_a_saturating_function_of_a_product_that_may_overflow_keeps_its_bound(function, finite):
    """`f(a * b)` of two inputs, finite reals whose product may overflow to an infinity: a tanh's and an erf's
    is finite, so its bound stands (a bound is stated of a finite value alone; an infinity's tanh is its
    side's 1), and a window reading the product as its program proves it bounds them still; a sine's of an
    infinity is no number, and stays unbounded (spec alignment, 8993)."""
    closed = jax.make_jaxpr(lambda a, b: function(a * b))(*(jnp.ones(8, jnp.float32) for _ in range(2)))
    made = closed.jaxpr.outvars[0]
    assert (terms.region(closed).held.get(made) is not None) is finite


@pytest.mark.parametrize("kind", KINDS)
def test_a_bodys_math_is_rewritten_in_place(kind):
    """Two sums inside a nested jit, a checkpoint, a scan, a conditional or a loop are rewritten inside that
    body into a form the model prices cheaper, the body kept a body (never unrolled), and the program is on
    par with the unrewritten one against the float64 truth (check 1)."""
    fn = nested(kind)
    x, y, a = rows(0, (8, 64), (8, 64), (8, 64))
    before, after = jax.make_jaxpr(fn)(x, y, a), jax.make_jaxpr(algebra.rewritten(fn))(x, y, a)
    assert cheaper_spliced(fn, x, y, a)

    def bodies(closed) -> list[str]:
        return sorted(equation.primitive.name for equation in closed.jaxpr.eqns if programs.bodies(equation))

    assert bodies(after) == bodies(before) and len(bodies(after)) == 1

    def draw(seed):
        return rows(seed, (8, 64), (8, 64), (8, 64))

    assert_as_exact(jax.jit(algebra.rewritten(fn)), jax.jit(fn), draw, f"two sums in a {kind}", seeds=16)


def test_statics_given_both_ways_are_each_taken_as_given_as_jit_takes_them():
    """Where both spellings are given, jit takes each as given and infers neither from the other: a count
    static by name alone and passed by position is traced, and a loop over it is refused. The rewritten
    function binds its statics by the same inference (`jax._src.api_util.infer_argnums_and_argnames`), so it
    refuses the same call, where its own union of the two spellings bound the count and ran it; passed by
    name, the rewritten function is on par with the function as written (check 1)."""
    def repeated(scale, x, y, a, n):
        for _ in range(n):
            x = two_sums(x, y, a)[:, None] * x
        return scale * two_sums(x, y, a)

    statics = {"static_argnums": 0, "static_argnames": "n"}
    x, y, a = rows(0, (8, 16), (8, 16), (8, 16))
    for function in (jax.jit(repeated, **statics), algebra.rewritten(repeated, **statics)):
        with pytest.raises(jax.errors.TracerIntegerConversionError):
            function(2.0, x, y, a, 2)
    rewritten = algebra.rewritten(repeated, **statics)
    assert_as_exact(jax.jit(lambda x, y, a: rewritten(2.0, x, y, a, n=2)),
                    jax.jit(lambda x, y, a: repeated(2.0, x, y, a, n=2)),
                    lambda seed: rows(seed, (8, 16), (8, 16), (8, 16)), "statics bound as jit binds them",
                    seeds=16)


def test_a_product_whose_mode_states_no_rounding_is_factored_by_no_rule():
    """A rule that factors a product makes another in its mode, bounded by that mode's rounding
    (`terms.contraction_slack`). A scaled operand's product at the default precision is rewritten; the same
    product under a dot algorithm given by its own types, whose rounding no source states, is left as
    written: nothing bounds the product a rule would make, so none fires, where the guards read only the
    operands' powers and the length and factored it as an ordinary product."""
    x, y, s = rows(1, (64, 32), (32, 16), (64,))

    def scaled(precision):
        return lambda x, y, s: jnp.dot(jnp.tanh(x) * jnp.tanh(s)[:, None], jnp.tanh(y), precision=precision)

    assert cheaper_spliced(scaled(None), x, y, s)
    assert not cheaper_spliced(scaled(jax.lax.DotAlgorithm(np.float32, np.float32, np.float32)), x, y, s)


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_transposed_byte_product_lifts_its_scale_only_with_exact_integer_accumulation(side):
    dims = (((1,), (1,)), ((), ()))
    full = jax.lax.DotAlgorithmPreset.F32_F32_F32

    def scaled(x, y, scale, precision, mesh=None):
        s = scale
        x, y = x.astype(s.dtype), y.astype(s.dtype)
        if side == "left":
            x = x * s[:, None]
        else:
            y = y * s[:, None]
        if mesh is None:
            return jax.lax.dot_general(x, y, dims, precision=precision)
        with jax.sharding.use_abstract_mesh(mesh):
            return jax.lax.dot_general(x, y, dims, precision=precision)

    def draw(seed, length=128):
        rng = np.random.default_rng(seed)
        scales = 8 if side == "left" else 16
        return (jnp.asarray(rng.integers(-128, 128, (8, length), dtype=np.int8)),
                jnp.asarray(rng.integers(0, 256, (16, length), dtype=np.uint8)),
                jnp.asarray(rng.uniform(0.125, 4.0, scales), jnp.float32))

    def selected(s):
        return jax.lax.select(s > 1, jax.lax.full(s.shape, 2.0, s.dtype), jax.lax.full(s.shape, 0.5, s.dtype))

    def plain(x, y, s):
        return scaled(x, y, selected(s), full)

    def truth(x, y, s):
        return scaled(x, y, selected(s), None)

    assert cheaper_spliced(plain, *draw(0))
    assert not cheaper_spliced(plain, *draw(0, length=257))
    assert not cheaper_spliced(truth, *draw(0))  # an unspecified dot mode is not an integer proof
    assert_as_exact(jax.jit(algebra.rewritten(plain)), jax.jit(plain), draw,
                    f"transposed byte {side} dequantization", seeds=8, truth_fn=truth)

    a = [3, -5] if side == "left" else [-5, 3]
    b = [5, 3] if side == "left" else [3, 5]
    x = jnp.broadcast_to(jnp.asarray([*a, *([0] * 126)], jnp.int8), (8, 128))
    y = jnp.broadcast_to(jnp.asarray([*b, *([0] * 126)], jnp.uint8), (16, 128))
    s = jnp.full((8 if side == "left" else 16,), 2.0, jnp.float32)

    def rounded(x, y, s):
        bad = jax.lax.full(s.shape, 1 + 2 ** -22, s.dtype)
        other = jax.lax.full(s.shape, 0.5, s.dtype)
        return scaled(x, y, jax.lax.select(s > 1, bad, other), full)

    assert not cheaper_spliced(rounded, x, y, s)
    np.testing.assert_array_equal(jax.jit(algebra.rewritten(rounded))(x, y, s), jax.jit(rounded)(x, y, s))

    # Two dots of the same parameters in one region may have different contracting lengths.
    def together(x, y, s, lx, ly):
        return plain(x, y, s), plain(lx, ly, s)

    lx, ly, _ = draw(0, length=257)
    region = terms.region(jax.make_jaxpr(together)(x, y, s, lx, ly))
    graph, _, (short, long) = program.graphed(region)
    assert graph.check_bool(algebra.rules.exact_int8_product(short))
    assert not graph.check_bool(algebra.rules.exact_int8_product(long))

    # The caller and the e-graph are outside the declared mesh, but the written dot is not.
    from jax.sharding import AbstractMesh, AxisType

    mesh = AbstractMesh((2,), ("data",), axis_types=(AxisType.Explicit,))
    x_shape, y_shape, s_shape = (jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (x, y, s))
    with jax.sharding.use_abstract_mesh(mesh):
        placed = jax.make_jaxpr(plain)(x_shape, y_shape, s_shape)
    graph, _, (placed_dot,) = program.graphed(terms.region(placed))
    assert not graph.check_bool(algebra.rules.exact_int8_product(placed_dot))

    # Warm the shape-identical windows in both orders. Only the dot enters the local scope.
    def scoped(x, y, s):
        return scaled(x, y, selected(s), full, mesh)

    for length, arms in ((128, (plain, scoped)), (96, (scoped, plain))):
        inputs = draw(0, length)
        for arm in arms:
            assert cheaper_spliced(arm, *inputs) == (arm is plain)


def test_a_saturation_of_no_growth_or_depth_is_refused():
    """A saturation's limits are each at least one (docs/design/algebra.md 6.6): one of none would stop
    every region before its first iteration and report it saturated."""
    for limits in ({"growth": 0, "depth": 4}, {"growth": 64, "depth": 0}):
        with pytest.raises(ValueError):
            algebra.Saturation(**limits)


@pytest.mark.parametrize("pass_n", ["positionally", "by keyword"])
@pytest.mark.parametrize("statics", [{"static_argnames": "n"}, {"static_argnums": 3}], ids=["name", "number"])
def test_a_rewritten_function_binds_its_static_arguments_as_jit_does(statics, pass_n):
    """A function whose loop count is static, rewritten and jitted with the same static arguments, named or
    numbered, the count passed positionally or by keyword: it runs as the function does (check 1)."""
    def repeated(x, y, a, n):
        for _ in range(n):
            x = two_sums(x, y, a)[:, None] * x
        return two_sums(x, y, a)

    rewritten = jax.jit(algebra.rewritten(repeated, **statics), **statics)
    plain = jax.jit(repeated, **statics)

    def called(fn):
        return (lambda x, y, a: fn(x, y, a, 2)) if pass_n == "positionally" else (
            lambda x, y, a: fn(x, y, a, n=2))

    # 64 rows a draw: at 8 the draws' log ratios spread past what the gate resolves in 80 seeds.
    assert_as_exact(called(rewritten), called(plain), lambda seed: rows(seed, (64, 64), (64, 64), (64, 64)),
                    f"a static count passed {pass_n}", seeds=16)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn],
                         ids=lambda dtype: jnp.dtype(dtype).name)
def test_a_literal_of_any_float_dtype_states_its_magnitude_and_scale(dtype):
    """A literal 2 of each float dtype, numpy's own (float32) and ml_dtypes' (bfloat16, float8), which numpy's
    type tree does not place among its floats: its magnitude and its positive scale are stated alike."""
    from egglog import EGraph, i64

    closed = jax.make_jaxpr(lambda x: x * jnp.asarray(2.0, dtype))(jnp.ones((4,), dtype))
    region = terms.region(closed)
    literal = terms.Term.literal(i64(0))
    graph = EGraph()
    program.stated(graph, region)
    assert region.magnitudes[terms.key(literal)] == 1
    assert graph.check_bool(algebra.rules.positive_scale(literal))


def test_each_contraction_states_its_own_length():
    """Two products of one parameters' index (one dimension numbers, one dtype) contracting 2 and 4096
    elements: each states its own contracted length's power (`rules.contracted_length`, by its product), where
    one stated per parameters' index let a reordering of the longer read the shorter's and admit a sum past
    the program's reach (row 3577)."""
    from egglog import EGraph, eq, get_callable_args, i64

    closed = jax.make_jaxpr(lambda b, c, a, x: (jnp.dot(b, c), jnp.dot(a, x)))(
        *rows(1, (8, 2), (2, 16), (8, 4096), (4096, 16)))
    region = terms.region(closed)
    graph = EGraph()
    program.stated(graph, region)
    short, long = (term for term, _ in region.shapes if get_callable_args(term, terms.Term.dot) is not None)
    assert graph.check_bool(eq(algebra.rules.contracted_length(short)).to(i64(1)))
    assert graph.check_bool(eq(algebra.rules.contracted_length(long)).to(i64(12)))


def test_a_scale_lifted_past_a_product_is_priced_at_its_shape():
    """A row scale lifted past a matrix product (`linear`'s lift, `(s x) @ y` as `s (x @ y)`): its broadcast
    to the product's extent states that extent, so the region's price counts the lifted broadcast and its
    multiply, where a broadcast of no stated shape was priced free (row 3581)."""
    from egglog import Vec, eq, i64

    closed = jax.make_jaxpr(lambda x, y, s: (x * s[:, None]) @ y)(*rows(1, (64, 32), (32, 16), (64,)))
    # Operands of declared magnitudes, so the lift's products stay under the program's reach (`dominated`).
    region = terms.region(closed, {0: 1.0, 1: 1.0, 2: 1.0})
    graph, _, _ = program.graphed(region)
    done = program.saturate(graph, region, (algebra.rules.RULES["linear"],), algebra.SATURATION)
    assert done.fired["linear"] > 0
    scale = terms.Term.leaf(i64(2))
    lift = graph.extract(algebra.rules.lifted(Vec[i64](i64(64), i64(16)), i64(0)))
    broadcast = scale.shaped("broadcast_in_dim", lift)
    assert graph.check_bool(eq(algebra.rules.shape(broadcast)).to(Vec[i64](i64(64), i64(16))))


def _distributed(root):
    """`a x + a y`'s factored form, `a (x + y)`."""
    from egglog import get_callable_args

    left, _, right, added = get_callable_args(root, terms.Term.binary)
    a, _, x, multiplied = get_callable_args(left, terms.Term.binary)
    _, _, y, _ = get_callable_args(right, terms.Term.binary)
    return a.binary("mul", x.binary("add", y, added), multiplied)


def _summed_out(root):
    """`sum(a[:, None] x, 1)`'s factored form, `a sum(x, 1)`."""
    from egglog import get_callable_args

    product, monoid, axes, summed = get_callable_args(root, terms.Term.reduce)
    laid, _, x, multiplied = get_callable_args(product, terms.Term.binary)
    a, _, _ = get_callable_args(laid, terms.Term.shaped)
    return a.binary("mul", x.reduce(monoid, axes, summed), multiplied)


def _lifted_out(root, graph):
    """`(x s[:, None]) @ y`'s factored form, `s (x @ y)`, its scale laid to the product's extent."""
    from egglog import Vec, get_callable_args, i64

    scaled, y, params = get_callable_args(root, terms.Term.dot)
    x, _, laid, multiplied = get_callable_args(scaled, terms.Term.binary)
    s, _, _ = get_callable_args(laid, terms.Term.shaped)
    lift = graph.extract(algebra.rules.lifted(Vec[i64](i64(64), i64(16)), i64(0)))
    return s.shaped("broadcast_in_dim", lift).binary("mul", x.dot(y, params), multiplied)


FACTORED = {
    "distributivity": (lambda x, y, a: a * x + a * y, ((64, 32), (64, 32), (64, 32)), _distributed),
    "a row scalar out of a sum": (lambda x, y, a: jnp.sum(a[:, None] * x, axis=1),
                                  ((64, 32), (64, 32), (64,)), _summed_out),
    "a row scale out of a product": (lambda x, y, a: (x * a[:, None]) @ y, ((64, 32), (32, 16), (64,)),
                                     _lifted_out),
}
"""Each rule that factors a scale out of a sum or a product: its program over `(x, y, a)`, their shapes, and
the factored form of its root (`a (x + y)`, `a sum(x)`, `a (x @ y)`)."""


@pytest.mark.parametrize("name", list(FACTORED))
def test_a_factored_product_is_admitted_only_under_the_reach(name):
    """A rule's factored product rounds the sum it factors before multiplying it, so it may pass float32's
    largest where the written terms each round to it (float32's `a x + a y` at `a = (2**24 - 4) 2**104`,
    `x = 1`, `y = 3 2**-24`; §6.3). Each rule admits the value it gives under the program's reach alone
    (`rules.dominated` of it): over operands of magnitude one, a scale of magnitude one makes the factored
    form one with the program's root, and a scale of float32's largest magnitude does not."""
    from egglog import eq

    program_of, shapes, factored_of = FACTORED[name]
    closed = jax.make_jaxpr(program_of)(*rows(1, *shapes))
    largest = float(np.finfo(np.float32).max)
    held = {}
    for scale in (1.0, largest):
        region = terms.region(closed, {0: 1.0, 1: 1.0, 2: scale})
        graph, _, (root,) = program.graphed(region)
        program.saturate(graph, region, (algebra.rules.RULES["linear"],), algebra.SATURATION)
        factored = factored_of(root, graph) if name == "a row scale out of a product" else factored_of(root)
        held[scale] = graph.check_bool(eq(root).to(factored))
    assert held == {1.0: True, largest: False}


REUSED = {
    "a reciprocal": (lambda a, w: (a / jnp.maximum(jnp.abs(w), 1e-3), 1 / jnp.maximum(jnp.abs(w), 1e-3)),
                     lambda a, w: (a / jnp.abs(w), 1 / jnp.abs(w))),
    "a reciprocal root": (
        lambda a, w: (a / jnp.sqrt(jnp.square(w) + 1e-3), jax.lax.rsqrt(jnp.square(w) + 1e-3)),
        lambda a, w: (a / jnp.sqrt(jnp.square(w)), jax.lax.rsqrt(jnp.square(w)))),
}
"""Each reuse's program, its quotient beside the reciprocal it reads: its divisor held away from zero by an
epsilon (a normalization's `max(|w|, eps)`, a norm's `w^2 + eps`), and the same with no epsilon."""


@pytest.mark.parametrize("name", list(REUSED))
@pytest.mark.parametrize(("floored", "bounded", "admitted"), [(True, True, True), (False, True, False),
                                                              (True, False, False)],
                         ids=["held", "no floor", "no bound"])
def test_a_quotient_reads_the_reciprocal_its_program_holds_only_where_its_divisor_is_held(name, floored,
                                                                                       bounded, admitted):
    """A quotient by a value whose reciprocal (or reciprocal root) the program holds is that reciprocal's
    product (the reuse set) only where the reciprocal is a normal finite float and the product finite
    (§6.3): its divisor held away from zero by an epsilon and bounded by its operand's declared range, the
    dividend bounded too. With no epsilon, a subnormal divisor's reciprocal overflows where the quotient is
    finite; with no bound, a large divisor's flushes to zero; neither is shared."""
    from egglog import eq, i64

    program_of = REUSED[name][0 if floored else 1]
    closed = jax.make_jaxpr(program_of)(*rows(1, (64, 32), (64, 32)))
    region = terms.region(closed, {0: 4.0, 1: 4.0} if bounded else {0: 4.0})
    graph, _, (quotient, reciprocal) = program.graphed(region)
    program.saturate(graph, region, (algebra.rules.RULES["reuse"],), algebra.SATURATION)
    product = terms.Term.leaf(i64(0)).binary("mul", reciprocal, i64(region.stock["mul"]))
    assert graph.check_bool(eq(quotient).to(product)) == admitted


@pytest.mark.parametrize("dtype, folds", [(jnp.float32, True), (jnp.bfloat16, False)])
def test_two_negations_are_an_identity_in_the_float32_window(dtype, folds):
    from egglog import eq, i64

    def twice(x):
        return jax.lax.neg(jax.lax.neg(x))

    closed = jax.make_jaxpr(twice)(jax.ShapeDtypeStruct((8,), dtype))
    region = terms.region(closed)
    graph, _, (result,) = program.graphed(region)
    program.saturate(graph, region, (algebra.rules.RULES["identities"],), algebra.SATURATION)
    assert graph.check_bool(eq(result).to(terms.Term.leaf(i64(0)))) is folds


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("case", ["normal", "infinity", "subnormal", "signaling", "quiet", "unknown",
                                  "placed"])
def test_a_widening_window_preserves_the_original_narrow_bits(dtype, case):
    normal, infinity = (0x0400, 0x7C00) if dtype == jnp.float16 else (0x0080, 0x7F80)
    other = {"normal": normal, "infinity": infinity, "subnormal": 1, "signaling": infinity | 1,
             "quiet": infinity | (normal >> 1) | 1, "unknown": infinity | 1, "placed": normal}[case]
    negative_zero = np.uint16(0x8000).view(dtype)
    endpoint = np.uint16(other).view(dtype)

    def layer(mask, payload):
        x = (payload if case == "unknown" else
             jax.lax.select(mask, jax.lax.full(mask.shape, negative_zero, dtype),
                            jax.lax.full(mask.shape, endpoint, dtype)))
        narrowed = x.astype(jnp.float32).astype(dtype)
        return jnp.max(narrowed * dtype(2)), jax.lax.bitcast_convert_type(narrowed, jnp.uint16)

    mask = np.arange(64) % 2 == 0
    payload = np.where(mask, np.uint16(0x8000), np.uint16(other)).view(dtype)
    rewritten = algebra.rewritten(layer)
    folds = case in ("normal", "infinity")
    if case == "placed":
        mesh = jax.sharding.AbstractMesh((1,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
        with jax.sharding.use_abstract_mesh(mesh):
            traced = jax.make_jaxpr(rewritten)(mask, payload)
    else:
        traced = jax.make_jaxpr(rewritten)(mask, payload)
    casts = [eq for eq in equations(traced.jaxpr) if eq.primitive is jax.lax.convert_element_type_p]
    assert len(casts) == (0 if folds else 2)
    bits = jax.jit(rewritten)(mask, payload)[1]
    want = payload.view(np.uint16) if folds else jax.jit(layer)(mask, payload)[1]
    np.testing.assert_array_equal(bits, want)


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("floored", [False, True])
def test_a_clamped_divider_reuses_its_held_negative_power_only_in_range(broadcast, floored):
    def layer(a, z):
        divisor = jax.lax.clamp(z.dtype.type(floored), jnp.abs(z), z.dtype.type(4))
        reciprocal = jax.lax.integer_pow(divisor, -1)
        return a / (divisor[:, None] if broadcast else divisor), reciprocal

    def draw(seed):
        a, z = rows(seed, (16, 32) if broadcast else (16,), (16,))
        return a, np.where(z < 0, np.float32(0.5), np.float32(2))

    with algebra.rules.enabling_dev("reuse"):
        rewritten = algebra.rewritten(layer)
        traced = jax.make_jaxpr(rewritten)(*draw(0))
        ops = [eq.primitive for eq in equations(traced.jaxpr)]
        assert ops.count(jax.lax.div_p) == (0 if floored else 1)
        assert ops.count(jax.lax.integer_pow_p) == 1
        if floored:
            assert_as_exact(jax.jit(rewritten), jax.jit(layer), draw,
                            "clamped dividend with a held reciprocal", seeds=8)


@pytest.mark.parametrize("fn", [
    lambda x, y, s: jnp.max(jnp.tanh(x) * 0.5, axis=1),
    lambda x, y, s: jnp.min(0.5 * jnp.tanh(x), axis=1),
    lambda x, y, s: jnp.dot(jnp.tanh(x) * jnp.tanh(s)[:, None], jnp.tanh(y)),
], ids=["max of a positive scale", "min of a positive scale", "contraction of a scaled operand"])
def test_a_rule_whose_left_side_heads_at_a_reduction_or_a_contraction_rewrites_it(fn):
    """A maximum or minimum of a positively scaled operand (`linear`'s positive scale out of an extremum) and
    a contraction of a scaled operand (its matrix scale, a dequant's): each is rewritten into a form the
    model prices cheaper, where no head read a reduction of an extremum over a product or a contraction over
    one, so neither rule ever ran (row 3580)."""
    assert cheaper_spliced(fn, *rows(1, (64, 32), (32, 16), (64,)))


def test_derived_forms_trace_under_each_manual_mesh_they_meet():
    """A tile's derived forms (`recurrence_algebra.derived_tile`, cached for the process) traced under a
    manual mesh of two devices, then of one: each trace binds its literals under its own mesh, the forms
    derived per trace context (`derive.derive`'s), where forms cached from the first bound its literals'
    avals, mesh and all, under the second and the trace refused (`For primitive gt, context mesh ... should
    match the aval mesh`; test_lowering's sharded recurrences, red at e82eab19a)."""
    from jax.sharding import AbstractMesh, AxisType

    from dew.nn.kernels import recurrence_algebra

    tile = jax.ShapeDtypeStruct((4, 4), jnp.float32)
    scale = jax.ShapeDtypeStruct((4,), jnp.float32)
    for devices in (2, 1):
        mesh = AbstractMesh((devices,), ("data",), axis_types=(AxisType.Manual,))
        with jax.sharding.use_abstract_mesh(mesh):
            jax.make_jaxpr(lambda p, d, s: recurrence_algebra.derived_tile("pairs").mapped(
                {"products": p, "decays": d, "scales": s}))(tile, tile, scale)


def _shared_products(a, x, y):
    """Two matrix products of a shared operand, summed, their operands bounded (tanh's)."""
    a, x, y = jnp.tanh(a), jnp.tanh(x), jnp.tanh(y)
    return jnp.dot(a, x) + jnp.dot(a, y)


def test_two_roots_of_one_shared_product_are_made_once():
    """Two outputs reading products of a shared operand, `a x + a y` and `a y + a x` (`_shared_products`'s
    terms, in either order): the pass prices them together as one factored product, and the program it
    writes makes that product once, the second output the first's value; each traced apart wrote the
    product twice, so the program computed what extraction priced as shared twice (spec alignment, 8995).
    Its values are the written program's on par from float64 (check 1)."""
    def both(a, x, y):
        a, x, y = jnp.tanh(a), jnp.tanh(x), jnp.tanh(y)
        return jnp.dot(a, x) + jnp.dot(a, y), jnp.dot(a, y) + jnp.dot(a, x)

    shapes = ((256, 64), (64, 64), (64, 64))
    written = jax.make_jaxpr(algebra.rewritten(both))(*rows(1, *shapes))
    assert [equation.primitive for equation in equations(written.jaxpr)].count(jax.lax.dot_general_p) == 1
    assert_as_exact(jax.jit(algebra.rewritten(both)), jax.jit(both), lambda seed: rows(seed, *shapes),
                    "two roots of one shared product", seeds=16)


def test_a_run_under_the_accurate_forms_keeps_its_program_as_written():
    """Under the run's accurate numerics (`precision="accurate"`, `generation.run_precision`) the pass admits
    the `exact` sets alone (docs/design/algebra.md §1.4): two products of a shared operand, summed, which the
    default numerics rewrite into a cheaper form, are spliced into none."""
    from dew.nn.precision import ACCURATE

    operands_ = rows(1, (256, 64), (64, 64), (64, 64))
    with generation.run_precision(ACCURATE), report.refusals():
        jax.make_jaxpr(algebra.rewritten(_shared_products))(*operands_)
        assert not any(made.spliced for made in report.rewrites())
    assert cheaper_spliced(_shared_products, *operands_)


def test_a_program_the_pass_rewrites_is_traced_once_at_the_compile_point():
    """The compile point (`options.program`) lowers a program the pass rewrites from the rewritten program
    itself, never traced again (row 6977: its replay was a whole second trace of every rewritten step), and
    what it runs is that program's: two sums the pass joins, as the rewritten function computes them."""
    from dew.telemetry.compiles import Compiles

    operands = rows(3, (64, 128), (64, 128), (64, 128))
    compiled = options.program(two_sums)
    with Compiles("two_sums") as heard:
        traced = compiled.trace(*operands)
    assert [found.kind for found in heard.programs].count("traced") == 1
    np.testing.assert_array_equal(traced.lower().compile()(*operands),
                                  jax.jit(algebra.rewritten(two_sums))(*operands))


@pytest.mark.parametrize(("length", "reductions"), [(128, 1), (129, 2)])
def test_floating_count_join_respects_the_accumulators_exact_integer_range(length, reductions):
    def counted(x, y):
        left = jax.lax.reduce_sum((x > 0).astype(x.dtype), (0,))
        right = jax.lax.reduce_sum((y > 0).astype(y.dtype), (0,))
        return left + right

    x = jnp.ones((length,), jnp.bfloat16)
    y = x.at[0].set(0)
    run = algebra.rewritten(counted)
    traced = jax.make_jaxpr(run)(x, y)
    assert sum(eq.primitive is jax.lax.reduce_sum_p for eq in equations(traced.jaxpr)) == reductions
    np.testing.assert_array_equal(jax.jit(run)(x, y), jax.jit(counted)(x, y))


def _clamped_products(x, a, b, lower, upper, operand=jnp.tanh):
    """A clamped value read by both products of an elementwise sum, eligible for distributivity."""
    clamped = jax.lax.clamp(lower, operand(x), upper)
    return jnp.sum(clamped * jnp.tanh(a) + clamped * jnp.tanh(b), axis=-1)


CLAMP_EDGES = np.asarray([np.inf, -np.inf, 0.0, -0.0, 0.3, 1.0, -2.0, 5.0, 2.0 ** -140], np.float32)
"""A clamp's operands at its edges: both infinities, signed zeros (a bound at 0's), values between, on and
past bounds of 0 and 1, and a subnormal; a NaN beside them in one row of their program (`_clamp_edges`)."""


def _clamp_edges() -> jax.Array:
    """`CLAMP_EDGES` over a 64 by 128 program, a NaN in its first row alone, so every other row's sums are
    finite and the first's carries the NaN through."""
    edges = np.resize(CLAMP_EDGES, (64, 128))
    edges[0, 0] = np.nan
    return jnp.asarray(edges)


def _clamp_draw(seed: int) -> tuple[jax.Array, ...]:
    """A clamp's finite edges and bounds (0, 1, and +-0.5) beside drawn values, over a 64 by 128 program, and
    the two products' operands."""
    rng = np.random.default_rng(seed)
    finite = CLAMP_EDGES[np.isfinite(CLAMP_EDGES)]
    x = rng.choice(np.concatenate([finite, [0.5, -0.5], rng.standard_normal(16)]), (64, 128))
    return (jnp.asarray(x, jnp.float32), *rows(seed, (64, 128), (64, 128)))


@pytest.mark.parametrize("bounds", ["scalar", "crossed", "array", "of an unbounded operand"])
def test_a_rewritten_clamps_derivative_is_zero_at_its_bounds_and_its_program_on_par(bounds):
    """A clamp is an operation of its own (`Term.clamp`), emitted as itself: the pass joins two products a
    clamped value is read by, scalar, crossed (lower above upper) and array bounds, and of an operand nothing
    bounds (a clamp is bounded by its bounds alone, an infinity clamped to one). Its derivative is zero
    exactly wherever the written program's is, at a bound and past one, and a NaN where it is one, over the
    clamp's edges (a maximum and a minimum would halve it at a bound: 0.76 where it is 0, row 7286's
    finding); the rewritten program and its gradient are on par with the written ones (check 1)."""
    x = _clamp_edges()
    a, b = rows(5, x.shape, x.shape)
    def scalars(*values):
        return lambda a, b: tuple(jnp.asarray(value, a.dtype) for value in values)

    # The bounds in the operands' dtype (a clamp's are its operand's), check 1's truth's float64 among them.
    bounded = {"scalar": scalars(0, 1), "crossed": scalars(1, 0),
               # Bounds of their own values, bounded (a clamp's magnitude is its bounds'), x's shape.
               "array": lambda a, b: (-0.5 * jnp.abs(jnp.tanh(b)), jnp.abs(jnp.tanh(a))),
               "of an unbounded operand": scalars(-0.5, 0.5)}[bounds]
    operand = (lambda x: x) if bounds == "of an unbounded operand" else jnp.tanh

    def clamped(x, a, b):
        return _clamped_products(x, a, b, *bounded(a, b), operand=operand)

    def gradient(form):
        return jax.grad(lambda x, a, b: jnp.sum(form(x, a, b)))

    assert cheaper_spliced(clamped, x, a, b)
    rewritten = algebra.rewritten(clamped)
    found, wanted = (np.asarray(jax.jit(gradient(form))(x, a, b)) for form in (rewritten, clamped))
    held = wanted == 0
    np.testing.assert_array_equal(found[held], wanted[held])
    np.testing.assert_array_equal(np.isnan(found), np.isnan(wanted))
    for arm, reference in ((rewritten, clamped), (gradient(rewritten), gradient(clamped))):
        assert_as_exact(jax.jit(arm), jax.jit(reference), _clamp_draw, f"a clamp's products, {bounds}",
                        seeds=16)


def test_a_clamp_is_priced_as_a_maximum_and_a_minimum_of_its_operand_read_once():
    """A clamped value's price (`cost.work`) is its operand's and two operations an element, a maximum's and
    a minimum's, its operand read once: as an operation the language did not speak, a clamp was a leaf of its
    own, its output read again and none of its arithmetic counted (row 7286)."""
    def priced(fn):
        region = terms.region(jax.make_jaxpr(fn)(jnp.ones((64, 128), jnp.float32)))
        graph, bundle, _ = program.graphed(region)
        _, price = graph.extract(bundle, include_cost=True, extractor="greedy-dag",
                                 cost_model=cost.model(region, cost.fused(cost.target("sm90"))))
        return price.work

    plain = priced(jnp.tanh)
    clamped = priced(lambda x: jax.lax.clamp(np.float32(-0.5), jnp.tanh(x), np.float32(0.5)))
    assert (clamped.operations - plain.operations, clamped.device_bytes) == (2 * 64 * 128, plain.device_bytes)


def test_products_with_a_shared_operand_join_into_one():
    """Two matrix products with one shared operand (`dot_general`), summed, are spliced into a form the model
    prices cheaper (one product of the shared operand with the summed other), their operands bounded
    (tanh's) so the range facts bound the new sum (§6.3), and compute what they compute (check 1)."""
    shapes = ((256, 64), (64, 64), (64, 64))
    assert cheaper_spliced(_shared_products, *rows(1, *shapes))

    def draw(seed):
        return rows(seed, *shapes)

    assert_as_exact(jax.jit(algebra.rewritten(_shared_products)), jax.jit(_shared_products), draw,
                    "two products", seeds=16)


@pytest.mark.parametrize("shared", ["left", "right"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("overflow", ["product", "sum"])
def test_factoring_products_keeps_finite_cancellation(shared, compiled, overflow):
    """Two contractions cancel exactly; factoring their bounded operands overflows a term product or a
    partial sum, although the new operand itself stays finite."""
    def contracted(a, z, w):
        z, w = jnp.tanh(z), jnp.tanh(w)
        return a @ z + a @ w if shared == "left" else z.T @ a.T + w.T @ a.T

    length = 2 if overflow == "product" else 4
    a = jnp.full((1, length), 2.0 ** (127 if overflow == "product" else 126), jnp.float32)
    z = jnp.repeat(jnp.array([20.0, -20.0], jnp.float32), length // 2)
    z = jnp.broadcast_to(z[:, None], (length, 64))
    original, rewritten = contracted, algebra.rewritten(contracted)
    if compiled:
        original, rewritten = jax.jit(original), jax.jit(rewritten)
    with jax.default_matmul_precision(FP32_SUMS):
        expected = original(a, z, z)
        np.testing.assert_array_equal(expected, jnp.zeros_like(expected))
        np.testing.assert_array_equal(rewritten(a, z, z), expected)


def test_a_program_closing_over_a_typed_key_compiles_through_the_pass():
    """A user's function compiled through Dew's seam (`dew.jit`, `options.program`) that closes over a typed
    PRNG key and binds an exponential, so the pass keys its body by its constants' bits (the attention
    family's anchor, `recognize._body_key`): it compiles, and its draw is the key's, bit for bit, as
    `jax.jit` draws it. NumPy reads no typed key's bits, which refused every such program (row 5860)."""
    key = jax.random.key(7)

    def sampled(x):
        return jnp.exp(x), jax.random.normal(key, x.shape)

    x = jnp.linspace(-1.0, 1.0, 16)
    _, drawn = options.program(sampled)(x)
    np.testing.assert_array_equal(np.asarray(drawn), np.asarray(jax.jit(sampled)(x)[1]))


ATTENTION_KEY_SEEDS = 8
"""The captured-key attentions' declared draws against the function under `jax.jit` (check 1)."""


def keyed_attention(q, k, v, key):
    """Multi-head attention `[batch, time, heads, width]` whose logits add a bias `key` draws."""
    bias = jax.random.normal(key, (q.shape[0], q.shape[2], q.shape[1], k.shape[1]), q.dtype)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * 0.25 + bias
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), v)


def attention_draw(salt: int):
    """`keyed_attention`'s queries, keys and values, drawn from `salt`'s stream."""
    def draw(seed):
        rng = np.random.default_rng([salt, seed])
        return tuple(jnp.asarray(rng.standard_normal((2, 16, 2, 8)), jnp.float32) for _ in range(3))
    return draw


def test_the_normal_form_keeps_payload_bits_observed_after_reduce_precision():
    from jax.extend import core

    from dew.algebra import normal_form

    def read(words):
        source = jax.lax.bitcast_convert_type(words, jnp.float32)
        rounded = jax.lax.reduce_precision(source, exponent_bits=8, mantissa_bits=7)
        return jax.lax.bitcast_convert_type(rounded, jnp.uint32)

    words = np.asarray([0x7FC12345, 0xFFC12345, 0x80000000, 0x00400000, 0x7F800000, 0x3F812345], np.uint32)
    canonical = normal_form.canonical(jax.make_jaxpr(read)(words))
    got = jax.jit(core.jaxpr_as_fun(canonical))(words)[0]
    np.testing.assert_array_equal(got, jax.jit(read)(words))


def test_an_attention_closing_over_a_typed_key_compiles_through_the_pass():
    """A multi-head attention whose logits add a bias drawn from a typed PRNG key the function closes over,
    compiled through Dew's seam: the pass recognizes the attention, its bias's draw inside the candidate
    (`recognize`), its normal form taking the key as the constant it is (`normal_form.canonical`), and the
    program it compiles is on par with the function under `jax.jit` from the function in float64 (check
    1). NumPy reads no typed key's bits, which refused every such program (row 5860)."""
    from reference_error import dew_programs

    key = jax.random.key(5861)

    def attended(q, k, v):
        return keyed_attention(q, k, v, key)

    assert_as_exact(dew_programs(options.program(attended)), attended, attention_draw(5861),
                    "an attention with a captured key's bias", seeds=ATTENTION_KEY_SEEDS)


def test_an_attention_selecting_one_of_two_captured_keys_compiles_through_the_pass():
    """A multi-head attention whose bias draws from one of two typed PRNG keys it closes over, the key chosen
    by its query's sign: the key a selection moves is a value of the attention's candidate, which the pass
    reads as no number (`terms`, its bound and its literals' facts), and the program it compiles is on par
    with the function under `jax.jit` from the function in float64 (check 1, row 5860)."""
    from reference_error import dew_programs

    keys = jax.random.key(5862), jax.random.key(5863)

    def attended(q, k, v):
        return keyed_attention(q, k, v, jax.lax.select(q[0, 0, 0, 0] > 0, *keys))

    assert_as_exact(dew_programs(options.program(attended)), attended, attention_draw(5862),
                    "an attention with a selected key's bias", seeds=ATTENTION_KEY_SEEDS)


def test_every_outer_program_is_rewritten_unless_its_arm_turns_the_pass_off():
    """An outer program (`options.program`) traces its function rewritten, so its lowering, its trace and its
    executable read one program; under `algebra.passes(on=False)`, which the gate opens for a reference arm,
    the same program traces unrewritten, and a program already held retraces as the switch changes, and as
    the target its forms are priced for does (`generation.assumed`): each a new executable."""
    x, y, a = rows(2, (8, 64), (8, 64), (8, 64))
    program = options.program(two_sums)

    def spliced() -> bool:
        with report.refusals():
            program.trace(x, y, a)
            return any(made.spliced for made in report.rewrites())

    assert spliced()
    with algebra.passes(on=False):
        assert not spliced()
        off = program.executable(x, y, a)
    on = program.executable(x, y, a)
    assert off is not on
    with generation.assumed("sm90"):
        elsewhere = program.executable(x, y, a)
    assert elsewhere is not on and elsewhere is not off
    np.testing.assert_array_equal(np.asarray(program(x, y, a)), np.asarray(on(x, y, a)))


def test_a_rewrite_is_recorded_in_the_kernel_report():
    """Each program the pass rewrites is recorded in the one kernel report (`report.rewrites`): the equations
    it read, the rule sets that fired and the nodes they introduced, the region's price before and after, the
    pass's seconds, and the roots whose cheaper form the program took, counted once the splice lands (a
    report recorded before its splice claimed forms the program may not have taken; row 3582)."""
    def joined(x, y, a):
        return two_sums(x, y, a)

    x, y, a = rows(3, (8, 64), (8, 64), (8, 64))
    with report.refusals():
        options.program(joined).compiled(x, y, a)
        [made] = report.rewrites()
    assert made.fired.get("joined", 0) > 0
    assert made.after < made.before
    assert made.seconds > 0
    assert made.spliced == 1


def test_a_pass_without_a_rule_set_leaves_its_rewrites_untaken():
    """A pass that leaves sets out admits none of their rewrites: only `joined` can join these counts.
    Without it they compute what the program with no pass computes. A derivation's named set can be
    left out; a dev-time set, a derivation's defining set and an unregistered name are refused."""
    from dew.algebra.program import passes

    def joined(x, y, a):
        return two_sums(x, y, a)

    x, y, a = rows(3, (8, 64), (8, 64), (8, 64))
    with report.refusals(), passes(without=("linear",)):
        options.program(joined).compiled(x, y, a)
        [made] = report.rewrites()
    assert "linear" not in made.admitted and "linear" not in made.fired and made.fired.get("joined", 0) > 0
    with report.refusals(), passes(without=("linear", "joined")):
        computed = options.program(joined).compiled(x, y, a)(x, y, a)
        assert not report.rewrites()
    with passes(on=False):
        traced = options.program(joined).compiled(x, y, a)(x, y, a)
    np.testing.assert_array_equal(np.asarray(computed), np.asarray(traced))
    with passes(without=("carried",)):
        pass
    for unnamed in ("stable", "decoupled", "lineal"):
        with pytest.raises(ValueError), passes(without=(unnamed,)):
            pass


def test_a_programs_whole_pass_is_recorded_by_either_entry():
    """Each program's whole pass is recorded in seconds whichever entry runs it (`report.passes`, row 5339):
    the compile point's rewrite of a jit's trace (`options.program`) and a function's own
    (`algebra.rewritten`, which recorded none), each at least what its region's derivation took, which it
    holds."""
    x, y, a = rows(3, (8, 64), (8, 64), (8, 64))
    for run in (lambda: options.program(lambda x, y, a: two_sums(x, y, a)).compiled(x, y, a),
                lambda: algebra.rewritten(two_sums)(x, y, a)):
        with report.refusals():
            run()
            passes, [made] = report.passes(), report.rewrites()
        assert passes
        assert sum(passes) >= made.seconds > 0


def two_products_summed(a, x, y):
    """`sum(a * x) + sum(a * y)` over the last axis: two sums of products that share a factor, which the
    program bounds none of."""
    return jnp.sum(a * x, axis=-1) + jnp.sum(a * y, axis=-1)


def test_a_deep_computation_traced_twice_is_on_par_with_the_function():
    """A chain of 40 steps reading each value twice (`a cos(a)`, which stays normal) traced twice over one
    input (JAX merges no equal values, so the two are apart), its results added (a distributivity head, so
    the pass bounds the whole program's values, `program._held`): rewritten, on par with the function by
    check 1. Its skeletons are one object for each form (`terms.Skeleton.made`): compared as nested tuples,
    each equal pair walked every path through their sharing, 2**40 of them, past the test's deadline (train
    902 F1)."""
    def doubled(x):
        a = b = x
        for _ in range(40):
            a = a * jnp.cos(a)
        for _ in range(40):
            b = b * jnp.cos(b)
        return a + b

    def draw(seed):
        return rows(seed, (8, 64))

    assert_as_exact(jax.jit(algebra.rewritten(doubled)), jax.jit(doubled), draw, "two equal deep chains",
                    seeds=16)


def test_a_region_no_rule_may_fire_in_is_recorded_unpriced():
    """A window whose values the program bounds none of builds no terms (`rules.reorderable`), so nothing
    prices it: its record holds no price where one a rule fires in holds both (row 5340: its 0.0 before and
    after read as a region that costs nothing)."""
    a, x, y = rows(5, (8, 64), (8, 64), (8, 64))
    with report.refusals():
        algebra.rewritten(two_products_summed)(a, x, y)
        [left] = report.rewrites()
        algebra.rewritten(two_sums)(x, y, a)
        fired = report.rewrites()[-1]
    assert (left.before, left.after, left.iterations, left.spliced) == (None, None, 0, 0)
    assert fired.after < fired.before


def test_a_program_the_rules_cannot_rewrite_is_kept_as_traced():
    """A program whose math no rule's left side reads (no sum of two reductions, products or contractions,
    no reduction of a product), in its own body or in a scan's, is left as traced: none of its regions is
    saturated or recorded, and it computes what it traced, so a program of many such bodies pays the pass
    nothing (row 3279's rollout pool paid 20 to 40 ms a region)."""
    def stepped(x):
        return jax.lax.scan(lambda carry, row: (jnp.tanh(carry + row), carry * 2.0), x[0], x)[1]

    x = rows(4, (6, 8))[0]
    with report.refusals():
        before = jax.make_jaxpr(stepped)(x)
        after = jax.make_jaxpr(algebra.rewritten(stepped))(x)
        assert report.rewrites() == ()
    np.testing.assert_array_equal(np.asarray(jax.core.eval_jaxpr(after.jaxpr, after.consts, x)[0]),
                                  np.asarray(jax.core.eval_jaxpr(before.jaxpr, before.consts, x)[0]))

PINNED = """
import gc, weakref
import dew


def first_compile():
    class Held:
        pass

    held = Held()
    from dew.nn.kernels import options

    options.program(lambda x: x + 1.0)(1.0)
    return weakref.ref(held)


reference = first_compile()
gc.collect()
print(reference() is None)
"""
"""A fresh process whose first program, compiled inside a call, imports the pass's e-graph library there."""


def test_a_calls_locals_go_when_its_first_program_imports_the_pass():
    """A call that compiles a process's first program, importing the pass and its e-graph library (egglog,
    whose declarations hold the frame that first imports it), keeps none of its locals once it returns:
    `import dew` imports the library on its own shallow frame first."""
    done = subprocess.run([sys.executable, "-c", PINNED], capture_output=True, text=True, check=True,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.stdout.strip().splitlines()[-1] == "True"

def test_a_program_thousands_of_equations_deep_is_rewritten_around_what_the_rules_read():
    """A program whose math runs four thousand equations deep before two sums the rules join (a step's
    backward through every layer) is rewritten in a window around those sums, the deep chain read as
    values from outside it: the sums take their cheaper form and the program is on par with the unrewritten
    one, where holding the whole chain as one term would pass the interpreter's recursion limit."""
    def deep(x, y, a):
        for _ in range(4000):
            x = jnp.sin(x)
        return two_sums(x, y, a)

    assert cheaper_spliced(deep, *rows(5, (4, 16), (4, 16), (4, 16)))

    def draw(seed):
        return rows(seed, (4, 16), (4, 16), (4, 16))

    assert_as_exact(jax.jit(algebra.rewritten(deep)), jax.jit(deep), draw, "two sums after a deep chain",
                    seeds=8)

@pytest.mark.parametrize("factor", [0.25, 0.3])
def test_a_scale_after_a_contraction_stays_after_the_product(factor):
    """A bounded contraction has no lower-range proof for its operands or their products."""
    def scaled(g, i):
        return jnp.dot(jnp.tanh(g), jnp.tanh(i)) * factor

    g, i = rows(6, (4, 4), (4, 16))
    assert not cheaper_spliced(scaled, g, i)

    def draw(seed):
        return rows(seed, (4, 4), (4, 16))

    assert_as_exact(jax.jit(algebra.rewritten(scaled)), jax.jit(scaled), draw,
                    "a scalar kept after its contraction", seeds=16)


@pytest.mark.parametrize("scaled", ["left", "right"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("overflow", ["product", "sum"])
def test_moving_a_scale_keeps_finite_contraction_products(scaled, compiled, overflow):
    """Scaling the small operand stays finite, but its replacement contraction overflows a term product or
    a partial sum where the original grouping cancels exactly."""
    length = 2 if overflow == "product" else 4

    def contracted(y):
        x = jnp.full((1, length), 2.0 ** (120 if overflow == "product" else 125), jnp.float32)
        product = x @ y if scaled == "left" else y.T @ x.T
        return product * jnp.float32(2)

    magnitude = 128.0 if overflow == "product" else 2.0
    y = jnp.repeat(jnp.array([magnitude, -magnitude], jnp.float32), length // 2)
    y = jnp.broadcast_to(y[:, None], (length, 4 if overflow == "product" else 64))
    original, rewritten = contracted, algebra.rewritten(contracted)
    if compiled:
        original, rewritten = jax.jit(original), jax.jit(rewritten)
    with jax.default_matmul_precision(FP32_SUMS):
        expected = original(y)
        np.testing.assert_array_equal(expected, jnp.zeros_like(expected))
        np.testing.assert_array_equal(rewritten(y), expected)


def test_a_rewrites_persisted_record_names_the_limits_its_saturation_had():
    """A rewrite's record as a fit search persists it (`planning.rewrite_record`, `FitSearch.kernels`): the
    node and iteration limits its saturation had beside whether one stopped it (docs/design/algebra.md
    §6.8), through JSON whole."""
    import json

    from dew.training import planning

    with report.refusals():
        jax.make_jaxpr(algebra.rewritten(two_sums))(*rows(0, (8, 64), (8, 64), (8, 64)))
        made = report.rewrites()[-1]
    record = json.loads(json.dumps(planning.rewrite_record(made)))
    assert (tuple(record["limits"]), record["limited"]) == (tuple(made.limits), made.limited)


@jax.jit
def _layer(x, w):
    """A jitted layer a model calls at each of its call sites: one body JAX shares across them."""
    return jnp.tanh(x @ w) + x


def test_a_body_shared_by_its_call_sites_stays_shared():
    """A model calling one jitted layer eight times, nothing to rewrite: with the pass on the model compiles
    as many programs as it does with the pass off (`dew.telemetry.compiles`), so a model's layers lower and
    compile once, their kernels too, not once a call site (each rebuilt body made a Qwen3.5-4B training
    step's compile 52 to 487 seconds on an H100)."""
    def model(x, ws):
        for w in ws:
            x = _layer(x, w)
        return jnp.sum(x)

    x, *ws = rows(0, *[(64, 64)] * 9)

    from dew.telemetry.compiles import Compiles

    def compiled(*, on: bool) -> int:
        jax.clear_caches()
        with algebra.passes(on), Compiles(None) as heard:
            jax.jit(algebra.rewritten(model))(x, ws).block_until_ready()
        return len(heard.programs)

    assert compiled(on=True) == compiled(on=False)


def test_the_pass_holds_no_body_past_its_life():
    """A body the pass read and left as it is goes with the program that held it: the pass's record of a
    body's rewrites holds the body weakly and never the body itself, so a process tracing many programs
    (a rollout pool retracing a length at a time) keeps none it dropped."""
    x, w = rows(0, (8, 8), (8, 8))
    with algebra.passes(on=True):
        closed = jax.make_jaxpr(algebra.rewritten(lambda x, w: jnp.sum(_layer(x, w))))(x, w)
    [body] = [equation.params["jaxpr"] for equation in closed.jaxpr.eqns
              if equation.primitive.name in ("pjit", "jit")]
    held = weakref.ref(body)
    del closed, body
    jax.clear_caches()
    gc.collect()
    assert held() is None




def _rewrites(fn, operands_) -> tuple:
    """The rewrites the pass records as `fn` traces over `operands_` (`report.rewrites`)."""
    with report.refusals():
        jax.make_jaxpr(algebra.rewritten(fn))(*operands_)
        return report.rewrites()


def test_a_window_of_one_structure_is_derived_once_across_the_programs_a_process_compiles():
    """Two programs whose windows have one structure, their values named apart (every compile a planner makes
    of one step, a resumed run's relowering, every layer of a stack), take one derivation: the second
    program's window is rewritten with the first's forms, reused, not saturated again (row 3764), and
    computes what its program computes (check 1)."""
    shapes = ((3, 40), (3, 40), (3, 40))

    def swapped(x, y, a):
        return two_sums(y, x, a)

    _rewrites(two_sums, rows(7, *shapes))
    [made] = _rewrites(swapped, rows(7, *shapes))
    assert (made.reused, made.spliced) == (True, 1)
    assert_as_exact(jax.jit(algebra.rewritten(swapped)), jax.jit(swapped), lambda seed: rows(seed, *shapes),
                    "a reused window's forms", seeds=8)


def test_windows_apart_in_a_literal_or_a_shape_are_derived_apart():
    """A window's derivation is keyed by all it reads: after a program of one structure, another of a
    different literal (a scale that is no power of two) or a different operand's shape derives its own, and
    each computes what its program computes (check 1)."""
    def scaled(scale):
        def sums(x, y, a):
            x, y, a = jnp.tanh(x), jnp.tanh(y), jnp.tanh(a)
            return jnp.sum(scale * a * x, axis=-1) + jnp.sum(scale * a * y, axis=-1)
        return sums

    _rewrites(scaled(0.5), rows(8, (3, 44), (3, 44), (3, 44)))
    for scale, width in ((0.3, 44), (0.5, 52)):
        shapes = ((3, width),) * 3
        assert not any(made.reused for made in _rewrites(scaled(scale), rows(8, *shapes)))
        assert_as_exact(jax.jit(algebra.rewritten(scaled(scale))), jax.jit(scaled(scale)),
                        lambda seed, shapes=shapes: rows(seed, *shapes), f"scale {scale} at {width}", seeds=8)


@pytest.mark.mesh(devices=4)
def test_a_window_over_values_an_explicit_mesh_places_is_derived_apart_from_one_devices():
    """A window's values' placement is part of what derives it: after one program over one device's values,
    the same program over values an explicit mesh shards derives its own, whose forms take its mesh, and
    computes what its program computes (check 1; the matrix's explicit layouts after their one-device
    reference)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    shapes = ((4, 56), (4, 56), (4, 56))
    _rewrites(two_sums, rows(9, *shapes))
    mesh = jax.make_mesh((4,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:4])

    def placed(seed):
        split = NamedSharding(mesh, P("data", None))
        return tuple(jax.device_put(operand, split) for operand in rows(seed, *shapes))

    with jax.set_mesh(mesh):
        assert not any(made.reused for made in _rewrites(two_sums, placed(9)))
        assert_as_exact(jax.jit(algebra.rewritten(two_sums)), jax.jit(two_sums), placed,
                        "an explicit mesh's window", seeds=8)



def test_a_region_holds_its_own_copy_of_a_literal_a_captured_scalar_views():
    """A literal may view a NumPy scalar its function captured, which its caller may change after a region is
    derived and kept for the later programs of its structure (`program._derived`): the region holds its own
    value, its type kept, not the caller's storage."""
    captured = np.array(0.5, np.float32)
    closed = jax.make_jaxpr(lambda x: jnp.sum(captured * jnp.tanh(x)))(jnp.ones((3, 8), jnp.float32))
    region = terms.region(closed)
    captured[...] = 4.0
    [held] = region.literals
    assert (float(held), np.asarray(held).dtype) == (0.5, np.float32)


SHAPES_MARKED = ((8, 48), (8, 48), (8, 48))
"""The operands of the marked spans' programs below: their shapes no other test of this module draws."""


def _marked_kinds(closed) -> list:
    """The kind of every marked call `closed` holds, its bodies' included."""
    from dew.nn.derivatives import marked_call_p

    return [equation.params["kind"] for equation in equations(closed) if equation.primitive is marked_call_p]


@pytest.mark.parametrize("generation_name", ["sm80", "sm90", "sm120"])
def test_a_template_walks_span_is_priced_at_its_gpus_kernel_step(generation_name):
    """A template's tile walk prices its span, a chain of dependent steps each reading the last, at its GPU's
    in-kernel step (one dependent reduction of a tile within a program, measured as the slope over 1,024 to
    16,384 steps, FS-STEP-SECONDS1's kernel path), so a walk of 10,000 such steps and little work costs
    10,000 times a walk of one, and no walk is free (row 4080)."""
    target = generation.gpu_target(generation_name)
    step = kernel_cost.walked(kernel_cost.Work(flops=1.0, span=1), target)
    assert kernel_cost.walked(kernel_cost.Work(flops=1.0, span=10_000), target) == 10_000 * step > 0


GH200_LATENCIES = {"product": (20.6e-9, 1.156e-9), "barrier": 85.0e-9, "copy": (464.5e-9, 5.17e-12)}
"""A GH200's step latencies as M-FS-19 measured them (FS-4123-LATENCIES2-gh200, clock locked at 1.98 GHz,
two rounds within 0.5%): a waited 64 x N product's fixed seconds and seconds a column, a two-warpgroup
barrier's round trip, a tile copy's from device memory past an L2 eviction, fixed and a byte."""


def test_a_template_walks_chain_is_priced_at_its_parts_measured_latencies():
    """A tile walk's dependent path (`Chain`) waits on its products, its warpgroups' barriers and its cold
    tile copies, each at the part's measured latency (M-FS-19's receipt, `GH200_LATENCIES`, independent of
    the part's own record): a recurrence chunk step waiting on four products of 128 columns and two round
    trips costs their sum, where its span alone priced one reduction (row 4123: a states launch's chunk step
    measured 13 times one), and one waiting on a 32 KiB tile copied cold through a ring of one stage the
    copy's fixed seconds and its slope's (the review of pricing-5 146aadb571f, row 12371)."""
    target = generation.gpu_target("NVIDIA GH200 480GB")
    (fixed, per_column), barrier = GH200_LATENCIES["product"], GH200_LATENCIES["barrier"]
    chain = kernel_cost.Chain(span=1, products=4, columns=4 * 128, barriers=2)
    chained = kernel_cost.walked(kernel_cost.Work(flops=1.0, chains=(chain,)), target)
    spanned = kernel_cost.walked(kernel_cost.Work(flops=1.0, span=1), target)
    expected = target.kernel_step_seconds + 4 * fixed + 512 * per_column + 2 * barrier
    assert_computes_the_oracle(np.float64(chained), np.float64(expected), "the chain's seconds", roundings=8)
    assert chained > 4 * spanned
    copied, per_byte = GH200_LATENCIES["copy"]
    cold = kernel_cost.Chain(span=1, products=4, columns=4 * 128, barriers=2, copies=1, copy_bytes=32768)
    waited = kernel_cost.walked(kernel_cost.Work(flops=1.0, chains=(cold,)), target)
    assert_computes_the_oracle(np.float64(waited), np.float64(expected + copied + 32768 * per_byte),
                               "the cold copy's seconds", roundings=8)


def test_a_rings_stages_hide_a_tile_copy_its_steps_outlast():
    """A step's tile copy through a ring of S stages is waited on only past what S - 1 steps ahead hide: a
    step of C seconds takes max(C, (L + C) / S) for a copy of L seconds, so a 4-stage ring hides a copy of
    1.5 of a step's 663 ns and a 2-stage ring of the same copy waits on part of it (row 4123)."""
    target = generation.gpu_target("NVIDIA GH200 480GB")
    copy = dataclasses.replace(target.part, copy=(1000e-9, 0.0))
    target = dataclasses.replace(target, measured={target.device_kind: copy})
    step = kernel_cost.Chain(products=4, columns=4 * 128)
    work = kernel_cost.walked(kernel_cost.Work(chains=(step,)), target)
    hidden = kernel_cost.Chain(products=4, columns=4 * 128, copies=1, stages=4)
    waited = dataclasses.replace(hidden, stages=2)
    assert kernel_cost.walked(kernel_cost.Work(chains=(hidden,)), target) == work
    waiting = kernel_cost.walked(kernel_cost.Work(chains=(waited,)), target)
    assert_computes_the_oracle(np.float64(waiting), np.float64((1000e-9 + work) / 2),
                               "the waited ring's step", roundings=4)


def test_work_side_by_side_keeps_every_chain_another_does_not_outlast():
    """Works summed side by side keep each chain no other outlasts at every part's latencies, so their price
    is the longest of their chains whatever the part, and adding work never lowers it: a chain of more of
    every event through no more stages replaces the lesser, two that stress different events are both kept."""
    products, barriers = kernel_cost.Chain(products=4, columns=512), kernel_cost.Chain(barriers=6)
    more = kernel_cost.Chain(products=5, columns=640, barriers=6)
    assert (kernel_cost.Work(chains=(products,)) + kernel_cost.Work(chains=(barriers,))).chains == (
        products, barriers)
    both = kernel_cost.Work(chains=(products, barriers))
    assert (both + kernel_cost.Work(chains=(more,))).chains == (more,)
    assert (kernel_cost.Work(flops=1.0) + kernel_cost.Work(chains=(more,))).chains == (more,)


def test_the_exact_extractor_reads_a_walks_longest_path_as_the_greedy_price_does():
    """The algebra's walk schedule's peak (`Walk.spanned`, what the exact extractor reads) is the longest
    dependent path the greedy price takes (`templates.cost.walked`): on a GH200 a form of one reduction and
    four barrier round trips (463 ns) costs more than one of two reductions (285 ns), where its span alone
    read 143 ns (base review 1e0a501f0 F1)."""
    target = generation.gpu_target("NVIDIA GH200 480GB")
    schedule = cost.walk(target)
    waits = kernel_cost.Work(chains=(kernel_cost.Chain(span=1, barriers=4),))
    reductions = kernel_cost.Work(span=2)
    assert schedule.spanned(waits) == kernel_cost.walked(waits, target)
    assert schedule.spanned(waits) > schedule.spanned(reductions)


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_integer_operations_take_the_integer_units_rate(generation_name):
    """An integer operation runs on a multiprocessor's ALU at 64 results a clock on sm80 and sm90 (CUDA C++
    Programming Guide 12.6, §5.4.1 Table 4: 32-bit integer add, shift, logic, compare), beside the fp32 lanes,
    not on them: a walk of integer work alone costs its count over that rate, and fp32 work beside it no more
    (row 5386)."""
    target = generation.gpu_target(generation_name)
    integers = kernel_cost.Work(integer_operations=64.0 * 10_000)
    alone = kernel_cost.walked(integers, target)
    assert_computes_the_oracle(np.float64(alone), np.float64(10_000 / target.loaded_clock_hz),
                               "integer work's seconds", roundings=4)
    assert kernel_cost.walked(integers + kernel_cost.Work(operations=128.0), target) == alone


def test_a_protected_span_is_kept_as_traced_however_often_the_pass_runs():
    """A protected span (`dew.nn.derivatives.Protected`: an error-free transform's) is a boundary no region
    crosses and whose inside no rule touches (docs/design/algebra.md §1.2): two sums the pass rewrites
    unmarked are spliced into nothing inside their one marked call, once rewritten and again, a program
    already rewritten rewritten anew."""
    from dew.nn.derivatives import Protected, marked

    operands_ = rows(11, *SHAPES_MARKED)
    assert cheaper_spliced(two_sums, *operands_)
    spanned = marked(Protected(), two_sums)
    for rewrite in (algebra.rewritten(spanned), algebra.rewritten(algebra.rewritten(spanned))):
        with report.refusals():
            rewritten = jax.make_jaxpr(rewrite)(*operands_)
            assert not any(made.spliced for made in report.rewrites())
        assert _marked_kinds(rewritten) == [Protected()]


@pytest.mark.parametrize("kind", ["Surrogate", "KernelGlue"])
def test_a_surrogates_and_a_kernel_glues_body_is_rewritten_within_its_call(kind):
    """A surrogate's bodies and a kernel derivative's glue are rewritten as any other math, within their
    marked call, which stays one call of its kind (§1.2): two sums inside take their cheaper form there, and
    the program computes what it computes (check 1)."""
    from dew.nn import derivatives

    spanned = derivatives.marked(getattr(derivatives, kind)(), two_sums)
    assert cheaper_spliced(spanned, *rows(12, *SHAPES_MARKED))
    rewritten = jax.make_jaxpr(algebra.rewritten(spanned))(*rows(12, *SHAPES_MARKED))
    assert len(_marked_kinds(rewritten)) == 1
    assert_as_exact(jax.jit(algebra.rewritten(spanned)), jax.jit(two_sums),
                    lambda seed: rows(seed, *SHAPES_MARKED), f"{kind}'s body", seeds=8)


def test_an_algebra_span_joins_the_windows_around_it():
    """A derivative written only to restructure algebra (`Algebra`) is rewritten freely (§6.10): its body
    opens into the region around it, so a sum inside it and a sum outside it take one cheaper form, no
    marked call left, and the program computes what it computes (check 1)."""
    from dew.nn.derivatives import Algebra, marked

    def spanning(x, y, a):
        return marked(Algebra(), count_above)(x, a) + count_above(y, a)

    assert cheaper_spliced(spanning, *rows(13, *SHAPES_MARKED))
    assert _marked_kinds(jax.make_jaxpr(algebra.rewritten(spanning))(*rows(13, *SHAPES_MARKED))) == []
    assert_as_exact(jax.jit(algebra.rewritten(spanning)), jax.jit(two_sums),
                    lambda seed: rows(seed, *SHAPES_MARKED), "an opened span", seeds=8)


def test_an_algebra_span_no_rewrite_crosses_keeps_its_marker():
    """An Algebra span that no rewrite reaches into (`cumsum(sin(x))`, nothing the rules lower) beside two
    sums the pass joins elsewhere in the program: the joined sums are spliced, and the span stays one marked
    call of its kind, as written, where one splice anywhere had opened every span for good (spec alignment,
    8998, §6.10); the program computes what it computes (check 1)."""
    from dew.nn.derivatives import Algebra, marked

    def both(x, y, a):
        kept = marked(Algebra(), lambda x: jnp.cumsum(jnp.sin(x), axis=-1))(x)
        return kept, two_sums(x, y, a)

    assert cheaper_spliced(lambda x, y, a: both(x, y, a)[1], *rows(14, *SHAPES_MARKED))
    assert _marked_kinds(jax.make_jaxpr(algebra.rewritten(both))(*rows(14, *SHAPES_MARKED))) == [Algebra()]
    assert_as_exact(jax.jit(algebra.rewritten(both)), jax.jit(both), lambda seed: rows(seed, *SHAPES_MARKED),
                    "a span no rewrite crosses beside joined sums", seeds=8)


def test_an_algebra_span_a_rewrite_lands_inside_stays_rewritten():
    """An Algebra span whose body joins two sums (`two_sums`) and takes their cumulative sum, an equation no
    rule rewrites: the join lands inside the span, which stays opened, so the program holds one sum where it
    held two; closing the span again, its replaced equations being in the program no more, restored its body
    as written and undid the join the pass had reported (head review). It computes what it computes
    (check 1)."""
    from dew.nn.derivatives import Algebra, marked

    def program_of(x, y, a):
        return marked(Algebra(), lambda x, y, a: jnp.cumsum(two_sums(x, y, a), axis=-1))(x, y, a)

    def summed(closed) -> int:
        return [equation.primitive for equation in equations(closed.jaxpr)].count(jax.lax.reduce_sum_p)

    shapes = SHAPES_MARKED
    assert summed(jax.make_jaxpr(algebra.rewritten(program_of))(*rows(16, *shapes))) < summed(
        jax.make_jaxpr(program_of)(*rows(16, *shapes)))
    assert_as_exact(jax.jit(algebra.rewritten(program_of)), jax.jit(program_of),
                    lambda seed: rows(seed, *shapes), "a span a join lands inside", seeds=8)


def test_an_algebra_span_returning_one_value_twice_stays_open_beside_a_splice():
    """An Algebra span returning one value as two outputs, both read, beside sums the pass joins elsewhere:
    the program the pass writes is a valid program (JAX's own check), the span left opened, where calling
    it again bound that one value's name twice (head review); it computes what it computes (check 1)."""
    from jax._src import core as jax_core

    from dew.nn.derivatives import Algebra, marked

    def both(x, y, a):
        def twice(x):
            made = jnp.cumsum(jnp.sin(x), axis=-1)
            return made, made

        first, second = marked(Algebra(), twice)(x)
        return first * second, two_sums(x, y, a)

    closed = jax.make_jaxpr(both)(*rows(15, *SHAPES_MARKED))
    jax_core.check_jaxpr(algebra.rewritten_program(closed).jaxpr)
    assert_as_exact(jax.jit(algebra.rewritten(both)), jax.jit(both), lambda seed: rows(seed, *SHAPES_MARKED),
                    "a span returning one value twice beside joined sums", seeds=8)


def test_an_algebra_span_with_effects_is_rewritten_within_its_call():
    """An Algebra span whose body has effects (a callback's) is not opened into the region around it, whose
    renaming would leave its effects naming the body's own variables: it is rewritten within its call, as a
    surrogate's body is, which stays one call of its kind, and it computes what it computes (check 1)."""
    from dew.nn.derivatives import Algebra, marked

    def noted(x, y, a):
        jax.debug.callback(lambda value: None, jnp.sum(x))
        return two_sums(x, y, a)

    spanned = marked(Algebra(), noted)
    assert cheaper_spliced(spanned, *rows(15, *SHAPES_MARKED))
    rewritten = jax.make_jaxpr(algebra.rewritten(spanned))(*rows(15, *SHAPES_MARKED))
    assert _marked_kinds(rewritten) == [Algebra()]
    assert_as_exact(jax.jit(algebra.rewritten(spanned)), jax.jit(two_sums),
                    lambda seed: rows(seed, *SHAPES_MARKED), "an effectful span", seeds=8)


def test_an_algebra_span_no_rule_changes_keeps_the_program_as_traced():
    """An Algebra span opened for the pass, around which no form is cheaper, leaves the program jit traced as
    it is, its marked call kept (`rewritten_program`: the same object, lowered as traced)."""
    from dew.nn.derivatives import Algebra, marked

    def stepped(x):
        return marked(Algebra(), lambda x: jnp.cumsum(jnp.sin(x), axis=-1))(x) * 2.0

    traced = jax.make_jaxpr(stepped)(rows(14, (8, 48))[0])
    assert algebra.rewritten_program(traced) is traced


def _parameter_pairs():
    """Equation parameters the algebra pass and the kernel tuner key (`terms.parameter_key`), as pairs of
    values each (equal, unequal) and the value as JAX holds it: two that print alike or key alike but differ
    (a mesh over the same devices in another order, a sharding of one mesh over its devices in another order;
    two long arrays apart in an element their printed forms elide; two abstract values apart in a layout,
    which their printed forms leave out), and
    two that are one parameter however they were built (an integer and NumPy's, a dtype and its scalar
    type, two shardings, dimension numbers, precisions and abstract values built apart)."""
    from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P

    devices = jax.devices()[:4]
    ordered, reordered = (jax.sharding.Mesh(np.asarray(order), ("data",))
                          for order in (devices, devices[::-1]))
    long, changed = np.arange(4096, dtype=np.int32), np.arange(4096, dtype=np.int32)
    changed[2048] = -1
    abstract = AbstractMesh((4,), ("data",), axis_types=(AxisType.Explicit,))
    numbers = jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,),
                                             start_index_map=(0,))
    placed = NamedSharding(ordered, P("data"))
    permuted = NamedSharding(ordered, P("data"), _logical_device_ids=(3, 2, 1, 0))
    from jax.core import ShapedArray
    from jax.experimental.layout import Layout

    laid = (ShapedArray((4, 8), jnp.float32), ShapedArray((4, 8), jnp.float32, layout=Layout((0, 1))))
    unequal = [(ordered, reordered), (long, changed), (placed, permuted), laid]
    equal = [((1, 2), (np.int64(1), np.int64(2))), (np.dtype(jnp.bfloat16), jnp.dtype("bfloat16")),
             (NamedSharding(abstract, P("data", None)), NamedSharding(abstract, P("data", None))),
             (numbers, jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,),
                                                      start_index_map=(0,))),
             (jax.lax.Precision.HIGHEST, jax.lax.Precision("highest")),
             (jax.lax.DotAlgorithmPreset.BF16_BF16_F32, jax.lax.DotAlgorithmPreset.BF16_BF16_F32),
             (ShapedArray((4, 8), jnp.float32, sharding=NamedSharding(abstract, P("data", None))),
              ShapedArray((4, 8), np.dtype("float32"), sharding=NamedSharding(abstract, P("data", None)))),
             (jax.typeof(jax.random.key(0)), jax.typeof(jax.random.key(1)))]
    return equal, unequal


@pytest.mark.mesh(devices=4)
def test_a_parameter_is_keyed_by_what_it_is_not_by_its_printed_form():
    """A window's derivation and a kernel launch's program are keyed by each equation's parameters
    (`terms.parameter_key`, the one structural key both read): parameters that print alike but differ take
    two keys, so a window derived for one is never another's; one parameter however built takes one; and
    a key is plain data, the same in every process (its printed form reads back as itself), never an
    object's printed form."""
    import ast

    equal, unequal = _parameter_pairs()
    for first, second in unequal:
        assert terms.parameter_key(first) != terms.parameter_key(second), (first, second)
        assert (terms.parameters_key(jax.lax.add_p, {"p": first})
                != terms.parameters_key(jax.lax.add_p, {"p": second}))
    for first, second in equal:
        assert terms.parameter_key(first) == terms.parameter_key(second), (first, second)
    for value in [value for pair in (*equal, *unequal) for value in pair]:
        key = terms.parameter_key(value)
        assert ast.literal_eval(repr(key)) == key, value


REGISTERED_AFTER = """
import json

import jax
import jax.numpy as jnp
import numpy as np
from egglog import eq, i64, rule, union, var
from jax.core import eval_jaxpr

from dew import algebra
from dew.algebra import rules
from dew.algebra.terms import Term


def sums(x, y):
    return jnp.sum(jnp.exp(x), -1) + jnp.sum(jnp.tanh(y), -1)


def branched(x, y):
    return jax.lax.cond(jnp.sum(x) > 0, sums, lambda x, y: sums(y, x), x, y)


def alone(x, y):
    return jnp.exp(jnp.tanh(x))


rng = np.random.default_rng(0)
x, y = (jnp.asarray(rng.standard_normal((5, 38)), jnp.float32) for _ in range(2))
closed = jax.make_jaxpr(branched)(x, y)


def computed():
    window = jax.jit(algebra.rewritten(sums))(x, y)
    rewritten = algebra.rewritten_program(closed)
    body = eval_jaxpr(rewritten.jaxpr, rewritten.consts, x, y)[0]
    lone = jax.jit(algebra.rewritten(alone))(x, y)
    return (float(jnp.max(jnp.abs(window - sums(x, y)))), float(jnp.max(jnp.abs(body - sums(x, y)))),
            float(jnp.max(jnp.abs(lone - alone(x, y)))))


before = computed()
variable, root, params = var("x", Term), var("root", Term), var("params", i64)
rewritten = rule(eq(variable.unary("exp", params)).to(root)).then(union(root).with_(variable))
exponential = rules.RuleFamily("the exponential as its argument", (rewritten,), (jnp.exp, lambda x: x, None))
rules.register(rules.Rules.declared("registered after", "reorders", (exponential,), "a test's"))
print(json.dumps({"before": before, "after": computed()}))
"""
"""A process that rewrites a program's window, a branch body of one held program and a program holding an
exponential alone, registers a rule set (one that rewrites an exponential to its argument), and rewrites
them again."""


def test_a_rule_set_registered_after_a_rewrite_reaches_the_programs_rewritten_after_it():
    """A window's and a body's rewrite are kept by everything their derivation reads, the rule sets the
    registry admits among them: a set registered after a program was rewritten (a user extending the
    algebra in a running process) reaches the next rewrite of a window of its structure and of the same held
    branch body, where a rewrite kept from before it would stand; and a program whose only math the set
    reads is the set's left side (an exponential alone, no sum beside it) is rewritten too: the pass opens
    its windows at each registered rule's own head (`rules.heads`), read from the term its union rewrites
    however the rule spells that term's equality (here the term first, its root second), where a table
    written beside the rules listed no exponential and left such a rule unreachable."""
    done = subprocess.run([sys.executable, "-c", REGISTERED_AFTER], capture_output=True, text=True,
                          check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert max(found["before"]) < 1e-4
    assert min(found["after"]) > 0.1


EXACT_ON_UNBOUNDED = """
import json

import jax
import jax.numpy as jnp
import numpy as np
from egglog import eq, i64, rule, union, var

from dew import algebra
from dew.algebra import rules
from dew.algebra.terms import Term


def twice(x):
    return jnp.abs(jnp.abs(x))


x = jnp.asarray(np.random.default_rng(0).standard_normal((5, 38)), jnp.float32)
closed = jax.make_jaxpr(twice)(x)


def magnitudes():
    return sum(equation.primitive.name == "abs" for equation in algebra.rewritten_program(closed).jaxpr.eqns)


before = magnitudes()
value, root, inner, outer = var("value", Term), var("root", Term), var("inner", i64), var("outer", i64)
once = rule(eq(value.unary("abs", inner).unary("abs", outer)).to(root)).then(
    union(root).with_(value.unary("abs", inner)))
family = rules.RuleFamily("a magnitude's magnitude", (once,), (twice, jnp.abs, None))
rules.register(rules.Rules.declared("registered exact", "exact", (family,), "a test's"))
print(json.dumps({"before": before, "after": magnitudes()}))
"""
"""A process that rewrites a program of a magnitude's magnitude over an operand nothing bounds, registers an
exact set that reads it as the magnitude, and rewrites it again."""


def test_a_registered_exact_rule_rewrites_a_window_of_values_nothing_bounds():
    """An exact rule reads no bound, so it fires in a window whose values have none: a registered exact set
    that reads `abs(abs(x))` as `abs(x)` leaves one magnitude of the two over an input the program states no
    range of. The pass left every such window as traced, whatever sets were admitted, because no reordering
    rule could fire in it (`rules.reorderable`)."""
    done = subprocess.run([sys.executable, "-c", EXACT_ON_UNBOUNDED], capture_output=True, text=True,
                          check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"before": 2, "after": 1}


def test_a_value_is_priced_at_the_bytes_its_dtype_holds():
    """A region's values are priced at their own element size (`cost.work`): a program reading bfloat16
    operands reads half the device bytes one reading float32 operands of the same shape does, and its product
    stages half the bytes into its tile walk, so the pass weighs a narrow program's memory as it is."""
    def priced(dtype):
        def contracted(a, b):
            return jnp.dot(a + b, b, preferred_element_type=jnp.float32)

        a, b = (jnp.ones((64, 64), dtype) for _ in range(2))
        region = terms.region(jax.make_jaxpr(contracted)(a, b))
        graph, bundle, _ = program.graphed(region)
        _, price = graph.extract(bundle, include_cost=True, extractor="greedy-dag",
                                 cost_model=cost.model(region, cost.fused(cost.target("sm90"))))
        return price.work

    narrow, wide = priced(jnp.bfloat16), priced(jnp.float32)
    assert (wide.device_bytes, wide.shared_bytes) == (2 * 64 * 64 * 4, 2 * 64 * 64 * 4)
    assert (narrow.device_bytes, narrow.shared_bytes) == (2 * 64 * 64 * 2, 2 * 64 * 64 * 2)


@pytest.mark.parametrize("function, exponentials", [
    (jnp.exp, 64 * 64), (jnp.sqrt, 64 * 64), (jax.lax.erf, 0), (jnp.expm1, 0)],
    ids=lambda f: getattr(f, "__name__", None))
def test_a_functions_work_is_the_units_the_kernels_counter_names(function, exponentials):
    """A region's elementwise function is priced on the units the kernels' counter states it issues on
    (`emission.SPECIAL_FUNCTIONS`, `cost.work`): an exp and a square root on the special-function units, an
    erf and an expm1 on the CUDA cores (XLA's polynomial; CUDA's expm1f, FMAs alone), where the algebra's
    own table charged all four as exponentials (base review 05bd9f77b F2)."""
    region = terms.region(jax.make_jaxpr(function)(jnp.ones((64, 64), jnp.float32)))
    graph, bundle, _ = program.graphed(region)
    _, price = graph.extract(bundle, include_cost=True, extractor="greedy-dag",
                             cost_model=cost.model(region, cost.fused(cost.target("sm90"))))
    assert price.work.exponentials == exponentials, price.work


def test_a_window_rewritten_under_another_matmul_precision_is_derived_again():
    """A window's derivation is keyed by the matmul precision its pricing reads for a product of no stated
    precision (`jax.default_matmul_precision`: TF32 at the default, the CUDA cores' IEEE fp32 above it): a
    held program rewritten again under another precision is derived afresh, not read from the first."""
    def scaled(x, y, b):
        x, y = jnp.tanh(x), jnp.tanh(y)
        return jnp.dot(x, y.T) * 0.5 + jnp.dot(b, y.T) * 0.5

    closed = jax.make_jaxpr(scaled)(*rows(16, (12, 72), (12, 72), (12, 72)))
    reused = []
    for precision in ("default", "highest"):
        with report.refusals(), jax.default_matmul_precision(precision):
            algebra.rewritten_program(closed)
            reused.append([made.reused for made in report.rewrites()])
    assert reused[0] and not any(reused[0]) and not any(reused[1])



def test_a_branch_a_pass_writes_an_ordered_effect_into_compiles_and_runs_both_ways():
    """A conditional one of whose branches a pass writes an equation into that holds an ordered effect (an
    interpreted kernel's IO callbacks, as fusion writes a launch's portable branch: `programs.with_bodies_of`,
    `with_equations`), lowered as jit lowers a rewritten program, never traced again (`program.traced_as`),
    and run down both branches: the conditional and the program holding it state the effect, so its lowering
    threads the effect's token, where they kept the effects they were traced with and the lowering raised a
    `KeyError` on the effect (an interpreted launch's `OrderedIOEffect`, row 12217; the models review)."""
    def branched(p, x):
        return jax.lax.cond(p, lambda x: x + 1, lambda x: x * 2, x)

    x, taken = jnp.float32(3.0), jnp.ones((), bool)
    traced = jax.jit(branched).trace(taken, x)
    closed = traced.jaxpr
    printed = jax.make_jaxpr(lambda x: (jax.debug.print("{}", x, ordered=True), x + 1)[1])(x)
    (conditional,) = [equation for equation in closed.jaxpr.eqns if equation.primitive.name == "cond"]
    params = {**conditional.params, "branches": (conditional.params["branches"][0], printed)}
    written = programs.with_equations(closed.jaxpr, [
        programs.with_bodies_of(equation, params) if equation is conditional else equation
        for equation in closed.jaxpr.eqns])
    run = program.traced_as(traced, closed.replace(jaxpr=written)).lower().compile()
    assert [float(run(taken, x)), float(run(~taken, x))] == [4.0, 6.0]


@pytest.mark.parametrize("rows", [1, 8])
def test_the_raw_decode_splice_executes_without_retracing(rows):
    """The spliced Jaxpr can run inside the original trace's compilation metadata, not a replay's."""
    from reference_error import as_given
    from test_algebra_fusion import _decode_draw, _decode_layer, _wide_decode

    from dew.algebra.fusion import lower_decode

    traced = jax.jit(_decode_layer).trace(*_decode_draw(rows, 6107)[:-1])
    selected = lower_decode(traced.jaxpr.jaxpr, rows=rows, enabled=True)
    run = program.traced_as(traced, traced.jaxpr.replace(jaxpr=selected)).lower().compile()
    assert_as_exact(as_given(run), _decode_layer, lambda seed: _decode_draw(rows, seed)[:-1],
                    "raw spliced decode forward", seeds=8, truth_fn=_wide_decode)


def test_a_held_body_rewritten_under_another_tile_pin_takes_that_pin():
    """A body's rewrite is keyed by the tiles a tuning pins (`tuning.pinned`, which a launch reads at its
    rewrite, `KernelOp.pinned`): one held program rewritten again under another legal pin of its product's
    tile launches the product at that pin, not as the first rewrite wrote it (row 12169: the body's cache
    keyed the kernel choices but not the pins, so the second pin was never read)."""
    from jax.sharding import PartitionSpec

    from dew.nn.kernels import grouped_matmul, tuning
    from dew.nn.kernels.templates.gemm import tiling

    mesh = jax.make_mesh((1,), ("data",), devices=jax.devices()[:1])

    def projected(x, w):
        def local(x, w):
            return jnp.dot(x, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

        return jax.shard_map(local, mesh=mesh, in_specs=(PartitionSpec(), PartitionSpec()),
                             out_specs=PartitionSpec())(x, w)

    closed = jax.make_jaxpr(projected)(jax.ShapeDtypeStruct((256, 512), jnp.bfloat16),
                                       jax.ShapeDtypeStruct((512, 256), jnp.bfloat16))
    for tile in (tiling.Tiles(128, 128, 64, 4, 1, warpgroups=2, family="wgmma"),
                 tiling.Tiles(128, 64, 64, 2, 1, warpgroups=2, family="wgmma")):
        with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
              tuning.pinned({grouped_matmul.MosaicGmm: tile}), tuning.seen() as launches):
            algebra.rewritten_program(closed)
        taken = [config for config, _ in launches.taken]
        assert len(taken) == 1, taken
        assert [(config.block_n, config.stages) for config in taken if isinstance(config, tiling.Tiles)] == [
            (tile.block_n, tile.stages)], taken
    # Pins of two kinds named in either order are one set of pins, one trace context: a program jitted under
    # one order runs under the other without tracing again (the head review: a tuple's order keyed two
    # contexts, two traces and two rewrites of one body).
    traces = []

    def counted(x):
        traces.append(None)
        return x + 1

    stepped = jax.jit(counted)
    for pins in ({grouped_matmul.MosaicGmm: tile, grouped_matmul.MosaicTgmm: tile},
                 {grouped_matmul.MosaicTgmm: tile, grouped_matmul.MosaicGmm: tile}):
        with tuning.pinned(pins):
            stepped(jnp.ones(()))
    assert len(traces) == 1, len(traces)

def _scaled_twice(s, q_pos, k_pos, a, b):
    """A score hook the pass rewrites: two bounded scales of the bounded logits summed (the magnitudes the
    pass's reordering reads), one product of their sum."""
    capped = jnp.tanh(s)
    return capped * jnp.tanh(a) + capped * jnp.tanh(b)


def _scaled_operands(seed: int, shape: tuple[int, ...]) -> tuple:
    """A tile of `shape` and two scalar scales, drawn by `seed`."""
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.standard_normal(shape), jnp.float32),
            *(jnp.asarray(rng.uniform(0.5, 2.0), jnp.float32) for _ in range(2)))


def test_a_score_hook_and_its_slopes_pass_the_algebra_pass_before_a_kernel_places_them():
    """A score hook and its derivative are rewritten by the algebra pass before a kernel's tile places them
    (docs/design/algebra.md §2): the tile's hook (`attention_algebra.hooked`) and its slopes
    (`attention_gradient.scored`) trace the hook's cheaper form (the rewrite report's), which they compute as
    the hook does (check 1); with the pass off, the hook as written."""
    from dew.nn.kernels.templates.attention_algebra import hooked
    from dew.nn.kernels.templates.attention_gradient import scored

    q_pos, k_pos = jnp.arange(16)[:, None], jnp.arange(32)[None, :]

    def tile(s, a, b):
        return hooked(_scaled_twice, s, q_pos, k_pos, (a, b))

    def sloped(s, a, b):
        return scored(_scaled_twice, s, q_pos, k_pos, (a, b), slopes=True)

    def draw(seed):
        return _scaled_operands(seed, (16, 32))

    assert traces_cheaper(tile, *draw(0)) and traces_cheaper(sloped, *draw(0))
    with algebra.passes(on=False):
        assert not traces_cheaper(tile, *draw(0))
    assert_as_exact(jax.jit(tile), jax.jit(lambda s, a, b: _scaled_twice(s, q_pos, k_pos, a, b)), draw,
                    "a rewritten score hook", seeds=8)


def test_a_fused_gemms_producer_passes_the_algebra_pass_before_the_fuser_places_it():
    """A fused GEMM's producer is rewritten by the algebra pass before the fuser stages it (algebra.md §2):
    a producer of two scales of one operand summed is staged as its cheaper form, its window's rewrite
    spliced (the fused products' values against their references: tests/test_gemm_template.py)."""
    from dew.nn.kernels import fused_gemm

    def producer(values):
        x, a, b = values
        capped = jnp.tanh(x)
        return capped * jnp.tanh(a) + capped * jnp.tanh(b)

    with report.refusals():
        fused_gemm._split(producer, _scaled_operands(1, (64, 128)))
        assert any(made.spliced for made in report.rewrites())


def test_a_tiles_transient_hook_is_released_once_its_trace_is_done():
    """A tile's hook is often a closure made for its trace (a keys-first tile's transposed hook,
    `layout.transposed`), which the pass's rewrite of it holds no longer than the trace: once traced and
    dropped, it is collected."""
    import gc
    import weakref

    from dew.nn.kernels.templates.attention_algebra import hooked

    def made():
        def transient(s, q_pos, k_pos, a, b):
            return _scaled_twice(s, q_pos, k_pos, a, b)
        return transient

    def traced(hook) -> weakref.ref:
        jax.make_jaxpr(lambda s, a, b: hooked(hook, s, jnp.arange(16)[:, None], jnp.arange(32)[None, :],
                                              (a, b)))(*_scaled_operands(2, (16, 32)))
        return weakref.ref(hook)

    held = traced(made())
    gc.collect()
    assert held() is None


def _mapped_over_rows(program, specs, devices: int = 2):
    """`program` mapped per shard (`jax.shard_map`) over `devices` of this process's devices, its axis
    `data`, its inputs split as `specs` (each a PartitionSpec's axes) and its output along its rows."""
    from jax.sharding import Mesh, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices()[:devices]), ("data",))
    return jax.shard_map(program, mesh=mesh, in_specs=tuple(PartitionSpec(*spec) for spec in specs),
                         out_specs=PartitionSpec("data"))


@pytest.mark.mesh(devices=2)
def test_a_mapped_bodys_sums_take_their_cheaper_form_varying_as_the_values_they_replace():
    """Two sums a manual mesh's body joins (`two_sums`, every input split along their rows): the cheaper
    form, traced over the window's inputs as the body types them (their manual axes among them,
    `fusion.typed`), varies as the values it replaces do, so it is spliced (row 6578's review, F5: traced
    over unsplit types, its values varied over nothing and the output-type gate discarded it), and the
    program computes what it computes (check 1)."""
    mapped = _mapped_over_rows(two_sums, (("data",), ("data",), ("data",)))
    assert cheaper_spliced(mapped, *rows(17, *SHAPES_MARKED))
    assert_as_exact(jax.jit(algebra.rewritten(mapped)), jax.jit(mapped),
                    lambda seed: rows(seed, *SHAPES_MARKED), "a mapped body's joined sums", seeds=8)


def _span_of_a_constant_beside_joined_sums(x, y, a):
    """An Algebra span closing over an array (`cumsum(sin(x * held))`, nothing the rules lower) beside two
    sums the pass joins (`two_sums`): the constant-span witnesses' body."""
    from dew.nn.derivatives import Algebra, marked

    held = np.linspace(0.5, 1.5, x.shape[-1], dtype=np.float32)
    kept = marked(Algebra(), lambda x: jnp.cumsum(jnp.sin(x * held), axis=-1))(x)
    return kept, two_sums(x, y, a)


@pytest.mark.mesh(devices=2)
def test_a_mapped_body_whose_rewrite_holds_a_constant_binds_it_as_an_operand():
    """A manual mesh's body holding an Algebra span that closes over an array (`cumsum(sin(x * held))`,
    nothing the rules lower) beside two sums the pass joins (`two_sums`): the join is spliced and the span,
    its array a constant of its own, is rewritten within its call, so no constant reaches the map's body,
    which JAX's lowering binds none of (row 6578's review, F1: opened, its array was the body's and the
    map's lowering refused it), and the program computes what it computes."""
    from jax.sharding import Mesh, PartitionSpec

    split = PartitionSpec("data")
    mesh = Mesh(np.asarray(jax.devices()[:2]), ("data",))
    mapped = jax.shard_map(_span_of_a_constant_beside_joined_sums, mesh=mesh, in_specs=(split,) * 3,
                           out_specs=(split, split))
    assert cheaper_spliced(mapped, *rows(18, *SHAPES_MARKED))
    assert_as_exact(jax.jit(algebra.rewritten(mapped)), jax.jit(mapped),
                    lambda seed: rows(seed, *SHAPES_MARKED), "a mapped body's span beside joined sums",
                    seeds=8)


def test_a_checkpoints_body_holding_a_span_of_a_constant_lowers_as_written():
    """A checkpoint (`jax.checkpoint`) whose body holds an Algebra span that closes over an array beside two
    sums the pass joins: the join is spliced and the span, its array a constant of its own, is rewritten
    within its call, so no constant reaches the checkpoint's body, which JAX stages and lowers with none
    (row 6578's review: opened, its array was the body's and the checkpoint's lowering refused it), and the
    program computes what it computes."""
    both = jax.checkpoint(_span_of_a_constant_beside_joined_sums)
    assert cheaper_spliced(both, *rows(20, *SHAPES_MARKED))
    assert_as_exact(jax.jit(algebra.rewritten(both)), jax.jit(both), lambda seed: rows(seed, *SHAPES_MARKED),
                    "a checkpoint's span of a constant beside joined sums", seeds=8)


@pytest.mark.parametrize(
    "case", ["same_shape", "broadcast_after_sqrt", "sqrt_after_broadcast", "unbounded_a"])
def test_reuse_pulls_sibling_rsqrt_into_division_windows_and_drops_sqrt(case):
    """When `reuse` is active, `program._windows` pulls sibling `rsqrt(v)` equations into `a / sqrt(v)`'s
    window (same-shape, broadcast after `sqrt`, `sqrt` after broadcast, and unbounded dividend `a` when
    `|v| >= 1`, i.e. `floor >= 0`), eliminating `sqrt` and `div` while holding check 1 against float64."""
    def fn(a, z):
        v = jnp.sum(jnp.square(jnp.tanh(z)), axis=-1) / float(z.shape[-1]) + 1.0
        r = jax.lax.rsqrt(v)
        if case == "same_shape":
            q = jnp.tanh(a[:, 0]) / jnp.sqrt(v)
        elif case == "broadcast_after_sqrt":
            q = jnp.tanh(a) / jnp.sqrt(v)[:, None]
        elif case == "sqrt_after_broadcast":
            q = jnp.tanh(a) / jnp.sqrt(v[:, None])
        else:
            q = a / jnp.sqrt(v)[:, None]
        return q, r

    with algebra.rules.enabling_dev("reuse"):
        x0, z0 = rows(0, (16, 32), (16, 32))
        before = jax.make_jaxpr(fn)(x0, z0)
        after = jax.make_jaxpr(algebra.rewritten(fn))(x0, z0)
        before_eqs, after_eqs = list(equations(before.jaxpr)), list(equations(after.jaxpr))
        assert any(eq.primitive is jax.lax.sqrt_p for eq in before_eqs)
        assert not any(eq.primitive is jax.lax.sqrt_p for eq in after_eqs)
        assert sum(eq.primitive is jax.lax.rsqrt_p for eq in after_eqs) == 1
        divs_before = sum(eq.primitive is jax.lax.div_p for eq in before_eqs)
        divs_after = sum(eq.primitive is jax.lax.div_p for eq in after_eqs)
        assert divs_after == divs_before - 1
        assert_as_exact(jax.jit(algebra.rewritten(fn)), jax.jit(fn),
                        lambda seed: rows(seed, (16, 32), (16, 32)), f"reuse {case}", seeds=8)


def test_reuse_refuses_unfloored_root_rewrites_reduce_sum_and_keys_jit_context():
    """An unfloored divisor `v` is never rewritten from `a / sqrt(v)` to `a * rsqrt(v)`, a floored
    `reduce_sum` divisor rewrites both `a / sqrt(y)[:, None]` (dropping `sqrt_p` and one `div_p`) and direct
    reciprocal reuse `b + a / y[:, None]` with sibling `1 / y` (leaving only the one shared reciprocal
    `div_p` and matching float64 Check 1), `enabling_dev` refuses accuracy-measured `bounded` sets (`stable`),
    and one `jax.jit(rewritten(fn))` callable traces separately across off -> on -> off
    `enabling_dev("reuse")` transitions."""
    def unfloored(a, v):
        pos = jnp.abs(v)
        return jnp.tanh(a) / jnp.sqrt(pos)[:, None], jax.lax.rsqrt(pos)

    def reduce_sum_root(a, z):
        y = jnp.sum(jnp.square(jnp.tanh(z)) + 1.0, axis=-1)
        return a / jnp.sqrt(y)[:, None], jax.lax.rsqrt(y)

    def softmax_like(a, b, z):
        y = jnp.sum(jnp.square(jnp.tanh(z)) + 1.0, axis=-1)
        inv = 1.0 / y
        return b + a / y[:, None], inv

    def gated(a, z):
        v = jnp.sum(jnp.square(jnp.tanh(z)), axis=-1) / float(z.shape[-1]) + 1.0
        return jnp.tanh(a) / jnp.sqrt(v)[:, None], jax.lax.rsqrt(v)

    with pytest.raises(ValueError), algebra.rules.enabling_dev("stable"):
        pass
    a0, b0, z0 = rows(0, (8, 16), (8, 16), (8, 16))
    v0 = np.full((8,), 2.0, np.float32)
    with algebra.rules.enabling_dev("reuse"):
        after_unfloored = jax.make_jaxpr(algebra.rewritten(unfloored))(a0, v0)
        assert any(eq.primitive is jax.lax.sqrt_p for eq in equations(after_unfloored.jaxpr))

        before_rs = list(equations(jax.make_jaxpr(reduce_sum_root)(a0, z0).jaxpr))
        after_rs = list(equations(jax.make_jaxpr(algebra.rewritten(reduce_sum_root))(a0, z0).jaxpr))
        assert any(eq.primitive is jax.lax.sqrt_p for eq in before_rs)
        assert not any(eq.primitive is jax.lax.sqrt_p for eq in after_rs)
        assert sum(eq.primitive is jax.lax.rsqrt_p for eq in after_rs) == 1
        assert (sum(eq.primitive is jax.lax.div_p for eq in after_rs)
                == sum(eq.primitive is jax.lax.div_p for eq in before_rs) - 1)
        assert_as_exact(jax.jit(algebra.rewritten(reduce_sum_root)), jax.jit(reduce_sum_root),
                        lambda seed: rows(seed, (8, 16), (8, 16)), "reuse reduce_sum root", seeds=8)

        before_sm = list(equations(jax.make_jaxpr(softmax_like)(a0, b0, z0).jaxpr))
        after_sm = list(equations(jax.make_jaxpr(algebra.rewritten(softmax_like))(a0, b0, z0).jaxpr))
        assert sum(eq.primitive is jax.lax.div_p for eq in before_sm) == 2
        assert sum(eq.primitive is jax.lax.div_p for eq in after_sm) == 1
        assert_as_exact(jax.jit(algebra.rewritten(softmax_like)), jax.jit(softmax_like),
                        lambda seed: rows(seed, (8, 16), (8, 16), (8, 16)), "reuse reduce_sum", seeds=8)
    # One jitted callable must trace separately across off -> on -> off scope transitions.
    jitted = jax.jit(algebra.rewritten(gated))
    off_before = equations(jitted.trace(a0, z0).jaxpr.jaxpr)
    assert any(eq.primitive is jax.lax.sqrt_p for eq in off_before)
    with algebra.rules.enabling_dev("reuse"):
        on_inside = equations(jitted.trace(a0, z0).jaxpr.jaxpr)
        assert not any(eq.primitive is jax.lax.sqrt_p for eq in on_inside)
    off_after = equations(jitted.trace(a0, z0).jaxpr.jaxpr)
    assert any(eq.primitive is jax.lax.sqrt_p for eq in off_after)


def test_dead_nested_result_behind_custom_vmap_is_cleaned_and_unreported():
    """`custom_vmap_p` registers no JAX DCE rule, so `_within` must still run `_read` inside its body: a
    dead secondary SwiGLU inside a nested `jit` called by a live `custom_vmap` is removed and emits no
    `swiglu` route."""
    from jax.custom_batching import custom_vmap

    @jax.jit
    def inner(a, b):
        live = a + b
        dead_swiglu = jax.nn.silu(a.astype(jnp.float32)).astype(a.dtype) * b
        return live, dead_swiglu

    @custom_vmap
    def wrapped_dead(a, b):
        live, _ = inner(a, b)
        return live

    wrapped_dead.def_vmap(lambda size, batched, a, b: (wrapped_dead(a, b), batched[0]))

    @custom_vmap
    def wrapped_live(a, b):
        live, used_swiglu = inner(a, b)
        return live + used_swiglu

    wrapped_live.def_vmap(lambda size, batched, a, b: (wrapped_live(a, b), batched[0]))

    shapes = [jax.ShapeDtypeStruct((256, 1024), jnp.bfloat16)] * 2
    with generation.assumed("sm90"), generation.routes() as found_dead:
        jax.jit(algebra.rewritten(wrapped_dead, kernels=True)).trace(*shapes)
    assert not any(route.choice == "swiglu" for route in found_dead)
    with generation.assumed("sm90"), generation.routes() as found_live:
        jax.jit(algebra.rewritten(wrapped_live, kernels=True)).trace(*shapes)
    assert any(route.choice == "swiglu" and route.kernel != "xla" for route in found_live)


def test_shared_divisor_fanout_stays_bounded_and_proven_matches_region_on_chained_broadcasts():
    """High-fanout shared producers hitting `CLUSTER` stay capped in `program._windows`, and `terms.proven`
    matches `terms.region` bit-for-bit on chained broadcasts of unbounded float inputs."""
    def fanout(xs, z):
        v = jnp.sum(jnp.square(jnp.tanh(z)), axis=-1) / float(z.shape[-1]) + 1.0
        root = jnp.sqrt(v)[:, None]
        inv = jax.lax.rsqrt(v)
        outs = [xs[i] / root for i in range(len(xs))]
        return tuple(outs), inv

    z0 = np.ones((8, 16), np.float32)
    xs0 = tuple(np.ones((8, 16), np.float32) for _ in range(120))
    with algebra.rules.enabling_dev("reuse"):
        closed_fan = jax.make_jaxpr(fanout)(xs0, z0)
        wins = program._windows(closed_fan.jaxpr)
    assert wins and all(len(win.eqns) <= program.CLUSTER for win, _ in wins)

    def chained_bcast(x):
        b1 = jax.lax.broadcast_in_dim(x, (4, 8, 1), (0, 1))
        return jax.lax.broadcast_in_dim(b1, (4, 8, 16), (0, 1, 2))

    closed_b = jax.make_jaxpr(chained_bcast)(jax.ShapeDtypeStruct((4, 8), jnp.float32))
    reg = terms.region(closed_b)
    p_held, p_apart = terms.proven(closed_b.jaxpr)
    assert p_held == reg.held
    assert p_apart == reg.apart


def test_jaxpr_parameter_key_hashes_structure_and_isolates_opaques_and_consts():
    """`terms.parameter_key` hashes `jcore.Jaxpr` structurally (`terms.jaxpr_digest`), keeping pure jaxprs
    JSON-safe (`_stated._plain`) and cross-process stable (`terms.opaque`) while distinguishing different
    arithmetic, constants, and opaque `custom_vjp` `WrappedFun` closures."""
    from dew import _stated

    shape = jax.ShapeDtypeStruct((4, 8), jnp.float32)
    c1 = jax.make_jaxpr(lambda x: jnp.sin(x) + 1.0)(shape)
    c2 = jax.make_jaxpr(lambda x: jnp.sin(x) + 1.0)(shape)
    c3 = jax.make_jaxpr(lambda x: jnp.cos(x) + 1.0)(shape)
    k1, k2, k3 = terms.parameter_key(c1), terms.parameter_key(c2), terms.parameter_key(c3)
    assert k1 == k2 and k1 != k3
    assert terms.jaxpr_digest(c1) == terms.jaxpr_digest(c2)
    assert terms.opaque(k1) == k1
    assert isinstance(_stated._plain(k1, "pure_jaxpr"), list)

    # Closed-over constants differentiate `parameter_key` even when the jaxpr body matches structurally.
    arr_a, arr_b = np.ones((4, 8), np.float32), np.full((4, 8), 2.0, np.float32)
    ca = jax.make_jaxpr(lambda x: x + arr_a)(shape)
    cb = jax.make_jaxpr(lambda x: x + arr_b)(shape)
    assert terms.jaxpr_digest(ca) == terms.jaxpr_digest(cb)
    assert terms.parameter_key(ca) != terms.parameter_key(cb)

    # Distinct `custom_vjp` rules embed distinct opaque `WrappedFun`s that never collide and refuse `_plain`.
    def make_vjp(scale: float):
        @jax.custom_vjp
        def f(x):
            return x * 2.0

        f.defvjp(lambda x: (x * 2.0, None), lambda _, g: (g * scale,))
        return jax.make_jaxpr(f)(shape)

    v1, v2 = make_vjp(2.0), make_vjp(3.0)
    kv1, kv2 = terms.parameter_key(v1), terms.parameter_key(v2)
    assert kv1 != kv2
    assert terms.opaque(kv1) == terms.opaque(kv2)
    with pytest.raises(_stated.Unestablished):
        _stated._plain(kv1, "custom_vjp_jaxpr")


def test_repeated_primitive_slots_keep_distinct_primitive_identities():
    from jax.extend import core

    def negate_twice(x):
        return jax.lax.neg(jax.lax.neg(x))

    traced = jax.make_jaxpr(negate_twice)(jax.ShapeDtypeStruct((8,), jnp.float32))
    first, second = traced.eqns
    other = core.Primitive(second.primitive.name)
    changed = traced.replace(jaxpr=traced.jaxpr.replace(eqns=[first, second.replace(primitive=other)]))
    assert terms.jaxpr_digest(traced, in_process=True) != terms.jaxpr_digest(changed, in_process=True)


def test_in_process_digest_distinguishes_which_link_each_equation_reads():
    shape = jax.ShapeDtypeStruct((), jnp.float32)

    def step(x, y):
        both = jax.lax.add(x, y)
        return jax.lax.sub(both, x)

    def other(x, y):
        both = jax.lax.add(x, y)
        return jax.lax.sub(both, y)

    first, clone, different = (jax.make_jaxpr(fn)(shape, shape) for fn in (step, jax.jit(step), other))
    key = terms.jaxpr_digest(first, in_process=True)
    assert key == terms.jaxpr_digest(clone.eqns[0].params["jaxpr"], in_process=True)
    assert key != terms.jaxpr_digest(different, in_process=True)


def test_process_digest_keeps_repeated_descriptors_in_equation_order():
    def formed(x, first, second):
        return jax.lax.reduce_precision(jax.lax.reduce_precision(x, 8, first), 8, second)

    shape = jax.ShapeDtypeStruct((4,), jnp.float32)
    direct = jax.make_jaxpr(lambda x: formed(x, first=4, second=8))(shape)
    reboxed = jax.make_jaxpr(lambda x: formed(x, 4, 8))(shape)
    reversed_ = jax.make_jaxpr(lambda x: formed(x, first=8, second=4))(shape)
    assert terms.body_key(direct) == terms.body_key(reboxed)
    assert terms.body_key(direct) != terms.body_key(reversed_)


def _bf16_rms(x, w):
    """Plain RMSNorm in fp32 statistics for structural body-cache tests."""
    mean_sq = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x.astype(jnp.float32) * jax.lax.rsqrt(mean_sq + 1e-6)).astype(x.dtype) * w


def _named_bf16_rms(name: str):
    """Freshly named RMSNorm function so each `make_jaxpr` carries distinct `debug_info`."""
    def norm(x, w):
        return _bf16_rms(x, w)

    norm.__name__ = name
    return norm


def test_structural_closed_reuses_const_free_body_preserving_debug_info_and_reports():
    """Fresh structurally identical const-free bodies hit `program._STRUCTURAL_CLOSED`, replaying routes and
    rewrite reports while keeping each body's own `debug_info` and computing identical values."""
    from jax._src import core as jax_core

    from dew.nn.kernels.templates import report as kernel_report

    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    ca = jax.make_jaxpr(_named_bf16_rms("norm_a"))(*shapes)
    cb = jax.make_jaxpr(_named_bf16_rms("norm_b"))(*shapes)
    assert ca is not cb and ca.debug_info != cb.debug_info
    with generation.assumed("sm90"):
        with generation.routes() as routes_a, kernel_report.recording():
            out_a = program._closed(ca, accurate=False, saturation=algebra.SATURATION, kernels=True)
        with generation.routes() as routes_b, kernel_report.recording():
            out_b = program._closed(cb, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert out_a is not ca and out_b is not cb
    assert out_a.debug_info == ca.debug_info
    assert out_b.debug_info == cb.debug_info
    assert any(r.choice == "rms_norm" and r.kernel != "xla" for r in routes_a)
    assert any(r.choice == "rms_norm" and r.kernel != "xla" for r in routes_b)
    jax_core.check_jaxpr(out_b.jaxpr)

    # `rules.head_primitives()` and `rules.heads()` stay aligned with and without dev rule sets enabled.
    base_prims = algebra.rules.head_primitives()
    assert "add" in base_prims and "div" not in base_prims
    with algebra.rules.enabling_dev("reuse"):
        reuse_prims = algebra.rules.head_primitives()
        assert "div" in reuse_prims
        assert algebra.rules.heads("div", [None, "sqrt"])
        assert not algebra.rules.heads("div", [None, None])


def test_structural_closed_isolates_callbacks_device_consts_and_prng_impls():
    """`program._STRUCTURAL_CLOSED` and `terms.body_key` distinguish `pure_callback` closures from the same
    factory, same-tag `define_prng_impl` specs with distinct implementations, and decline structural sharing
    without touching device memory when a nested `pjit` holds a concrete `jax.Array`."""
    from jax.core import ShapedArray, eval_jaxpr
    from jax.extend import random as jrandom

    def make_cb_body(offset: float):
        def bump(x):
            return x + np.asarray(offset, dtype=x.dtype)

        def body(x, w):
            normed = _bf16_rms(x, w)
            return jax.pure_callback(bump, jax.ShapeDtypeStruct(normed.shape, normed.dtype), normed)

        return body

    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    cb1, cb5 = jax.make_jaxpr(make_cb_body(1.0))(*shapes), jax.make_jaxpr(make_cb_body(5.0))(*shapes)
    assert terms.body_key(cb1) != terms.body_key(cb5)
    with generation.assumed("sm90"):
        r1 = program._closed(cb1, accurate=False, saturation=algebra.SATURATION, kernels=True)
        r5 = program._closed(cb5, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert r1 is not cb1 and r5 is not cb5 and r1.jaxpr is not r5.jaxpr
    eq1 = next(eq for eq in r1.jaxpr.eqns if eq.primitive.name == "pure_callback")
    eq5 = next(eq for eq in r5.jaxpr.eqns if eq.primitive.name == "pure_callback")
    assert eq1.params["callback"] is not eq5.params["callback"]
    x_in, w_in = jnp.ones((256, 256), jnp.bfloat16), jnp.ones((256,), jnp.bfloat16)
    cpu1 = program._closed(cb1, accurate=False, saturation=algebra.SATURATION, kernels=True)
    cpu5 = program._closed(cb5, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert float(eval_jaxpr(cpu1.jaxpr, cpu1.consts, x_in, w_in)[0][0, 0]) == 2.0
    assert float(eval_jaxpr(cpu5.jaxpr, cpu5.consts, x_in, w_in)[0][0, 0]) == 6.0

    # Nested `jit` holding a concrete device `jax.Array` declines structural sharing without host copy.
    captured = jnp.ones((256,), jnp.bfloat16)

    @jax.jit
    def inner_with_const(x):
        return x + captured

    def outer_const_free(x, w):
        return inner_with_const(_bf16_rms(x, w))

    nested_closed = jax.make_jaxpr(outer_const_free)(*shapes)
    assert not nested_closed.consts
    assert terms.body_key(nested_closed) is None

    # Same-tag `define_prng_impl` specs with distinct `random_bits` implementations get distinct keys.
    def make_prng(fill: int):
        return jrandom.define_prng_impl(
            key_shape=(2,),
            seed=lambda seed: jnp.zeros((2,), jnp.uint32),
            split=lambda key, shape: jnp.broadcast_to(key, (*shape, 2)),
            random_bits=lambda key, bit_width, shape: jnp.full(shape, fill, jnp.uint32),
            fold_in=lambda key, word: key,
            name="shared_test_name",
            tag="shared_test_tag",
        )
    impl_a, impl_b = make_prng(11), make_prng(29)
    key_a, key_b = jax.random.key(0, impl=impl_a), jax.random.key(0, impl=impl_b)

    def draw_and_norm(k, x, w):
        return jax.random.bits(k, (4,), jnp.uint32), _bf16_rms(x, w)

    p_a = jax.make_jaxpr(draw_and_norm)(jax.typeof(key_a), *shapes)
    p_b = jax.make_jaxpr(draw_and_norm)(jax.typeof(key_b), *shapes)
    assert terms.body_key(p_a) != terms.body_key(p_b)
    with generation.assumed("sm90"):
        rp_a = program._closed(p_a, accurate=False, saturation=algebra.SATURATION, kernels=True)
        rp_b = program._closed(p_b, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert rp_a is not p_a and rp_b is not p_b and rp_a.jaxpr is not rp_b.jaxpr
    a0, b0 = rp_a.in_avals[0], rp_b.in_avals[0]
    assert isinstance(a0, ShapedArray) and isinstance(b0, ShapedArray) and a0.dtype != b0.dtype
    cpu_a = program._closed(p_a, accurate=False, saturation=algebra.SATURATION, kernels=True)
    cpu_b = program._closed(p_b, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert int(eval_jaxpr(cpu_a.jaxpr, cpu_a.consts, key_a, x_in, w_in)[0][0]) == 11
    assert int(eval_jaxpr(cpu_b.jaxpr, cpu_b.consts, key_b, x_in, w_in)[0][0]) == 29


def test_structural_closed_keys_on_drawing_context_in_both_orders():
    """Fresh structurally identical bodies traced under `generation.drawing()` and live `assumed('sm90')`
    never share a `_STRUCTURAL_CLOSED` entry in either cache order."""
    from dew.nn.kernels.templates import report as kernel_report

    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    d1, n1 = (jax.make_jaxpr(_named_bf16_rms(name))(*shapes) for name in ("draw_first", "real_after"))
    n2, d2 = (jax.make_jaxpr(_named_bf16_rms(name))(*shapes) for name in ("real_first", "draw_after"))
    with generation.assumed("sm90"):
        with generation.drawing(), generation.routes() as draw_r1, kernel_report.recording():
            out_d1 = program._closed(d1, accurate=False, saturation=algebra.SATURATION, kernels=True)
        with generation.routes() as real_r1, kernel_report.recording():
            out_n1 = program._closed(n1, accurate=False, saturation=algebra.SATURATION, kernels=True)
        with generation.routes() as real_r2, kernel_report.recording():
            out_n2 = program._closed(n2, accurate=False, saturation=algebra.SATURATION, kernels=True)
        with generation.drawing(), generation.routes() as draw_r2, kernel_report.recording():
            out_d2 = program._closed(d2, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert not any(r.choice == "rms_norm" and r.kernel != "xla" for r in draw_r1)
    assert any(r.choice == "rms_norm" and r.kernel != "xla" for r in real_r1)
    assert any(r.choice == "rms_norm" and r.kernel != "xla" for r in real_r2)
    assert not any(r.choice == "rms_norm" and r.kernel != "xla" for r in draw_r2)
    assert out_d1 is d1 and out_n1 is not n1 and out_n2 is not n2 and out_d2 is d2


def test_body_key_handles_float0_dag_scaling_equation_context_and_scoped_array_gc():
    """`terms.body_key` and `_STRUCTURAL_CLOSED` handle `float0` tangent dtypes from integer gradients, stay
    linear on depth-30 shared nested-jit DAGs, preserve equation-local `cur_abstract_mesh` contexts in both
    cache orders, and never pin a `jax.Array` held only in a scoped `ContextVar`."""
    import contextvars
    import gc

    from jax.sharding import AbstractMesh, AxisType

    import dew

    grad_int = jax.grad(lambda x, i: (x * i).sum(), allow_int=True)
    f0_closed = jax.make_jaxpr(grad_int)(
        jax.ShapeDtypeStruct((4,), jnp.float32), jax.ShapeDtypeStruct((4,), jnp.int32))
    assert terms.body_key(f0_closed) is not None and terms.jaxpr_digest(f0_closed, in_process=True)[0]
    assert program._closed(f0_closed, accurate=False, saturation=algebra.SATURATION, kernels=True) is not None
    got_g = dew.jit(grad_int)(jnp.ones((4,), jnp.float32), jnp.arange(4, dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(got_g), np.arange(4, dtype=jnp.float32))

    def build_dag(offset: float):
        fn = jax.jit(lambda x: x + offset)
        for _ in range(30):
            prev = fn
            fn = jax.jit(lambda x, f=prev: f(x) + f(x + 1.0))
        return jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((4,), jnp.float32))

    dag_a, dag_b, dag_diff = build_dag(1.0), build_dag(1.0), build_dag(2.0)
    ka, kb, kd = terms.body_key(dag_a), terms.body_key(dag_b), terms.body_key(dag_diff)
    assert ka is not None and ka == kb and ka != kd
    assert terms.jaxpr_digest(dag_a, in_process=True) == terms.jaxpr_digest(dag_b, in_process=True)
    assert terms.jaxpr_digest(dag_a, in_process=True) != terms.jaxpr_digest(dag_diff, in_process=True)

    mesh_a = AbstractMesh((2,), ("data",), axis_types=(AxisType.Explicit,))
    mesh_b = AbstractMesh((4,), ("data",), axis_types=(AxisType.Explicit,))
    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))

    def make_scoped_body(mesh: AbstractMesh, name: str):
        def body(x, w):
            normed = _bf16_rms(x, w)
            with jax.sharding.use_abstract_mesh(mesh):
                return normed + jnp.asarray(1.0, normed.dtype)
        body.__name__ = name
        return jax.make_jaxpr(body)(*shapes)

    for first_mesh, second_mesh, tag in ((mesh_a, mesh_b, "ab"), (mesh_b, mesh_a, "ba")):
        c1, c2 = make_scoped_body(first_mesh, f"ctx_{tag}_1"), make_scoped_body(second_mesh, f"ctx_{tag}_2")
        assert terms.body_key(c1) != terms.body_key(c2) and terms.jaxpr_digest(c1) != terms.jaxpr_digest(c2)
        with generation.assumed("sm90"):
            r1 = program._closed(c1, accurate=False, saturation=algebra.SATURATION, kernels=True)
            r2 = program._closed(c2, accurate=False, saturation=algebra.SATURATION, kernels=True)
        assert r1.jaxpr.eqns[-1].ctx.cur_abstract_mesh == first_mesh
        assert r2.jaxpr.eqns[-1].ctx.cur_abstract_mesh == second_mesh

    scoped_var: contextvars.ContextVar = contextvars.ContextVar("test_scoped_device_array", default=None)

    def run_in_scope() -> weakref.ref:
        buf = jnp.ones((16, 16), jnp.float32)
        ref, token = weakref.ref(buf), scoped_var.set(buf)
        try:
            traced = jax.make_jaxpr(lambda x: x + 1.0)(jax.ShapeDtypeStruct((4,), jnp.float32))
            program._closed(traced, accurate=False, saturation=algebra.SATURATION, kernels=True)
        finally:
            scoped_var.reset(token)
        return ref

    arr_ref = run_in_scope()
    gc.collect()
    assert arr_ref() is None


def test_body_key_distinguishes_callable_dataclasses_and_norm_rejects_bad_layouts():
    """Executable callable dataclasses (factory-made, `eq=False`, cyclic `eq=False`, and frozen cyclic
    `compare=False` in bare nested `jax.jit`) never hit `RecursionError` or cross-hit `_STRUCTURAL_CLOSED`;
    `_NormTrack` rejects permuting reshapes, singleton expansion, and lossy `bf16->fp16` casts."""
    import dew

    def make_factory_callback(offset: float):
        @dataclasses.dataclass(frozen=True)
        class OffsetCallback:
            def __call__(self, arr):
                return arr + np.float32(offset)
        return OffsetCallback()

    @dataclasses.dataclass(frozen=True, eq=False)
    class IdentityCallback:
        def __call__(self, arr):
            return arr + np.float32(1.0 if self is id_first else 2.0)

    @dataclasses.dataclass(eq=False)
    class CyclicCallback:
        offset: float
        parent: object = None
        def __call__(self, arr):
            return arr + np.float32(self.offset)

    @dataclasses.dataclass(frozen=True)
    class FrozenCyclic:
        offset: float
        parent: list = dataclasses.field(default_factory=list, compare=False, repr=False)
        def __call__(self, arr):
            return arr + np.float32(self.offset)

    id_first, id_second = IdentityCallback(), IdentityCallback()
    cyc1, cyc2 = CyclicCallback(1.0), CyclicCallback(2.0)
    cyc1.parent, cyc2.parent = cyc1, cyc2
    fc1, fc2 = FrozenCyclic(1.0), FrozenCyclic(2.0)
    fc1.parent.append(fc1)
    fc2.parent.append(fc2)
    rms_shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    x_in, w_in = jnp.ones((256, 256), jnp.bfloat16), jnp.ones((256,), jnp.bfloat16)
    exp2, exp3 = np.full((256, 256), 2.0, np.float32), np.full((256, 256), 3.0, np.float32)
    def make_body(cb, *, nested: bool = False, bare: bool = False):
        inner = jax.jit(lambda y: jax.pure_callback(cb, jax.ShapeDtypeStruct(y.shape, y.dtype), y))
        return lambda x, w: (lambda y: inner(y) if nested else jax.pure_callback(
            cb, jax.ShapeDtypeStruct(y.shape, y.dtype), y))(
            x.astype(jnp.float32) if bare else _bf16_rms(x, w).astype(jnp.float32))
    cases = ((make_factory_callback(1.0), make_factory_callback(2.0), True, False, False),
             (id_first, id_second, False, False, False), (cyc1, cyc2, False, True, False),
             (fc1, fc2, False, True, True))
    for cb1, cb2, shareable, is_nested, is_bare in cases:
        f1, f2 = (make_body(cb, nested=is_nested, bare=is_bare) for cb in (cb1, cb2))
        c1, c2 = jax.make_jaxpr(f1)(*rms_shapes), jax.make_jaxpr(f2)(*rms_shapes)
        k1, k2 = terms.body_key(c1), terms.body_key(c2)
        assert (k1 is not None and k2 is not None and k1 != k2) if shareable else (k1 is None and k2 is None)
        assert terms.jaxpr_digest(c1, in_process=True) != terms.jaxpr_digest(c2, in_process=True)
        assert shareable or terms.jaxpr_digest(c1) != terms.jaxpr_digest(c2)
        with generation.assumed("sm90"):
            r1 = program._closed(c1, accurate=False, saturation=algebra.SATURATION, kernels=True)
            r2 = program._closed(c2, accurate=False, saturation=algebra.SATURATION, kernels=True)
        assert r1.jaxpr != r2.jaxpr
        np.testing.assert_array_equal(np.asarray(dew.jit(f1)(x_in, w_in)), exp2)
        np.testing.assert_array_equal(np.asarray(dew.jit(f2)(x_in, w_in)), exp3)
    bad_norms = (
        (lambda x: x * jax.lax.rsqrt(jax.lax.reshape(
            jnp.sum(jnp.square(x), axis=-1, keepdims=True) + jnp.float32(1e-6),
            (4, 2, 2, 1), dimensions=(0, 2, 1, 3))), (4, 2, 2, 8), jnp.float32),
        (lambda x: x * jax.lax.broadcast_in_dim(jax.lax.rsqrt(
            jnp.sum(jnp.square(x), axis=-1, keepdims=True) + jnp.float32(1e-6)),
            (4, 2, 2, 64), (0, 1, 2, 3)), (4, 2, 2, 1), jnp.float32),
        (lambda x: x.astype(jnp.float32) * jax.lax.rsqrt(
            jnp.sum(jnp.square(x.astype(jnp.float16).astype(jnp.float32)), axis=-1, keepdims=True)
            + jnp.float32(2.0 ** -100)), (4, 2, 2, 64), jnp.bfloat16),
        (lambda x: x.astype(jnp.float32) * jax.lax.rsqrt(
            jnp.sum(jnp.square(x), axis=-1, keepdims=True).astype(jnp.float32)
            + jnp.float32(2.0 ** -100)), (4, 2, 2, 64), jnp.float16),
    )
    for bad_fn, shape, dt in bad_norms:
        closed = jax.make_jaxpr(bad_fn)(jax.ShapeDtypeStruct(shape, dt))
        held, apart = terms.proven(closed.jaxpr, targets=(closed.jaxpr.outvars[0],), deep=True)
        assert held[closed.jaxpr.outvars[0]] is None and not apart[closed.jaxpr.outvars[0]].unit_norm
