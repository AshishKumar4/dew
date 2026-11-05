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
from reference_error import assert_as_exact, equations
from test_algebra import count, rows, two_sums

from dew import algebra
from dew.algebra import cost, program, terms
from dew.nn.kernels import generation, options
from dew.nn.kernels.templates import cost as kernel_cost, report
from dew.nn.precision import FP32_SUMS


def reductions(closed) -> int:
    """The sums a program's jaxpr holds, its bodies' included."""
    return sum(equation.primitive.name == "reduce_sum" for equation in equations(closed))


def nested(kind: str):
    """`two_sums` inside a body of `kind`: a nested jit, a checkpoint, a scan over two steps, a conditional,
    a loop of one turn."""
    def body(x, y, a):
        if kind == "jit":
            return jax.jit(two_sums)(x, y, a)
        if kind == "checkpoint":
            return jax.checkpoint(two_sums)(x, y, a)
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


@pytest.mark.parametrize("kind", KINDS)
def test_a_bodys_math_is_rewritten_in_place(kind):
    """Two sums inside a nested jit, a checkpoint, a scan, a conditional or a loop are joined into one inside
    that body, the body kept a body (never unrolled), and the program is on par with the unrewritten one
    against the float64 truth (check 1)."""
    fn = nested(kind)
    x, y, a = rows(0, (8, 64), (8, 64), (8, 64))
    before, after = jax.make_jaxpr(fn)(x, y, a), jax.make_jaxpr(algebra.rewritten(fn))(x, y, a)
    branches = 2 if kind == "cond" else 1
    assert (reductions(before), reductions(after)) == (2 * branches, branches)

    def bodies(closed) -> list[str]:
        return sorted(equation.primitive.name for equation in closed.jaxpr.eqns
                      if equation.primitive.name in algebra.program.BODIES)

    assert bodies(after) == bodies(before) and len(bodies(after)) == 1

    def draw(seed):
        return rows(seed, (8, 64), (8, 64), (8, 64))

    assert_as_exact(jax.jit(algebra.rewritten(fn)), jax.jit(fn), draw, f"two sums in a {kind}", seeds=16)


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


def test_a_program_no_rule_changes_is_staged_once():
    """An outer program (`options.program`) whose math no rule changes (an opaque equation beside an unbounded
    sum), lowered with the pass on: each of its equations staged once, by jit's own trace, which lowers as it
    is (`program.rewritten_program`), where the pass's replay of its traced program staged each again (a
    Dory-4B training step's 1.8 seconds of compile, nothing rewritten)."""
    from jax.extend import core as jex_core
    from jax.interpreters import mlir

    staged = []

    class Counted(jex_core.Primitive):
        def bind(self, *args, **params):
            staged.append(args)
            return super().bind(*args, **params)

    counted = Counted("counted_staging")
    counted.def_impl(lambda x: x)
    counted.def_abstract_eval(lambda x: x)
    mlir.register_lowering(counted, lambda ctx, x: [x])

    def step(x):
        return counted.bind(x) + x

    with report.refusals():
        options.program(step).lower(jnp.ones((4, 8), jnp.float32))
    assert len(staged) == 1


@pytest.mark.parametrize("fn", [
    lambda x, y: jnp.tanh(x) * 2.0 + jnp.tanh(x) * 3.0,
    lambda x, y: x * 2.0 + x * 4.0,
    lambda x, y: x * 2.0 + y,
    lambda x, y: jnp.dot(x, y) + x,
    lambda x, y: jnp.exp(x) + y * y,
], ids=["bounded", "scaled twice", "scaled once", "unbounded product", "unbounded exponential"])
def test_a_windows_structure_decides_reordering_as_its_terms_do(fn):
    """Whether a rule may reorder a program (`rules.reorderable`) read off its structure alone
    (`terms.Skeleton`, the pass's first read of every window) as off its egglog terms: a bounded value, one
    value scaled by literals twice, and the program's own unbounded math each decided alike."""
    closed = jax.make_jaxpr(fn)(*rows(1, (16, 16), (16, 16)))
    assert (algebra.rules.reorderable(terms.region(closed, forms=terms.Skeleton))
            == algebra.rules.reorderable(terms.region(closed)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn],
                         ids=lambda dtype: jnp.dtype(dtype).name)
def test_a_literal_of_any_float_dtype_states_its_magnitude_and_scale(dtype):
    """A literal 2 of each float dtype, numpy's own (float32) and ml_dtypes' (bfloat16, float8), which numpy's
    type tree does not place among its floats: its magnitude (`terms.region`) and its power of two's exact and
    positive scale (`rules.exact_scale`, `rules.positive_scale`) stated alike, where a bfloat16 or float8
    literal had none."""
    from egglog import EGraph, i64

    closed = jax.make_jaxpr(lambda x: x * jnp.asarray(2.0, dtype))(jnp.ones((4,), dtype))
    region = terms.region(closed)
    literal = terms.Term.literal(i64(0))
    graph = EGraph()
    program.stated(graph, region)
    assert region.magnitudes[terms.key(literal)] == 1
    assert graph.check_bool(algebra.rules.exact_scale(literal))
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


@pytest.mark.parametrize("fn", [
    lambda x, y, s: jnp.max(x * 0.5, axis=1),
    lambda x, y, s: jnp.min(0.5 * x, axis=1),
    lambda x, y, s: jnp.dot(x * s[:, None], y),
], ids=["max of a positive scale", "min of a positive scale", "contraction of a scaled operand"])
def test_a_rule_whose_left_side_heads_at_a_reduction_or_a_contraction_gets_a_window(fn):
    """A maximum or minimum of a positively scaled operand (`linear`'s positive scale out of an extremum) and
    a contraction of a scaled operand (its matrix scale, a dequant's): each opens a window its rule may fire
    in (`rules.heads`), where no head read a reduction of an extremum over a product or a contraction over
    one, so neither rule ever ran (row 3580)."""
    closed = jax.make_jaxpr(fn)(*rows(1, (64, 32), (32, 16), (64,)))
    assert program._windows(closed.jaxpr)


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


def test_a_run_under_the_accurate_forms_keeps_its_program_as_written():
    """Under the run's accurate numerics (`precision="accurate"`, `generation.run_precision`) the pass admits
    the `exact` sets alone (docs/design/algebra.md §1.4): two products of a shared operand, summed, which the
    default numerics join into one, stay two (`dot_general` counted), no window splicing a form."""
    from dew.nn.precision import ACCURATE

    operands_ = rows(1, (256, 64), (64, 64), (64, 64))
    with generation.run_precision(ACCURATE), report.refusals():
        accurate = jax.make_jaxpr(algebra.rewritten(_shared_products))(*operands_)
        assert not any(made.spliced for made in report.rewrites())
    default = jax.make_jaxpr(algebra.rewritten(_shared_products))(*operands_)
    assert (count(accurate, "dot_general"), count(default, "dot_general")) == (2, 1)


def test_two_sums_join_into_one_where_the_model_prices_it_cheaper():
    """Two sums of bounded values (`sum(tanh(x)) + sum(tanh(y))`): the model reads each operand once either
    way and counts one reduction's operations and one addition fewer joined, so it joins them, and the
    program computes what it computes (check 1)."""
    def two(x, y):
        return jnp.sum(jnp.tanh(x), axis=-1) + jnp.sum(jnp.tanh(y), axis=-1)

    shapes = ((64, 128), (64, 128))
    operands_ = rows(3, *shapes)
    assert count(jax.make_jaxpr(algebra.rewritten(two))(*operands_), "reduce_sum") == 1
    assert_as_exact(jax.jit(algebra.rewritten(two)), jax.jit(two), lambda seed: rows(seed, *shapes),
                    "two sums", seeds=16)


def test_products_with_a_shared_operand_join_into_one():
    """Two matrix products with one shared operand (`dot_general`), summed, become one product of the shared
    operand with the summed other, their operands bounded (tanh's) so the range facts bound the new sum
    (§6.3): a contraction is bilinear, and the product's FLOPs halved outweigh the sum's operations."""
    shapes = ((256, 64), (64, 64), (64, 64))
    after = jax.make_jaxpr(algebra.rewritten(_shared_products))(*rows(1, *shapes))
    assert sum(equation.primitive.name == "dot_general" for equation in equations(after)) == 1

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


def test_every_outer_program_is_rewritten_unless_its_arm_turns_the_pass_off():
    """An outer program (`options.program`) traces its function rewritten, so its lowering, its trace and its
    executable read one program; under `algebra.passes(on=False)`, which the gate opens for a reference arm,
    the same program traces unrewritten, and a program already held retraces as the switch changes."""
    x, y, a = rows(2, (8, 64), (8, 64), (8, 64))
    program = options.program(two_sums)
    assert reductions(program.trace(x, y, a).jaxpr) == 1
    with algebra.passes(on=False):
        assert reductions(program.trace(x, y, a).jaxpr) == 2
        off = program.executable(x, y, a)
    on = program.executable(x, y, a)
    assert off is not on
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
    assert made.fired.get("linear", 0) > 0
    assert made.after < made.before
    assert made.seconds > 0
    assert made.spliced == 1

def test_a_program_the_rules_cannot_rewrite_is_kept_as_traced():
    """A program whose math no rule's left side reads (no sum of two reductions, products or contractions,
    no reduction of a product), in its own body or in a scan's, is left as traced: none of its regions is
    saturated or recorded, and its scan's body is the one it traced, so a program of many such bodies pays
    the pass nothing (row 3279's rollout pool paid 20 to 40 ms a region)."""
    def stepped(x):
        return jax.lax.scan(lambda carry, row: (jnp.tanh(carry + row), carry * 2.0), x[0], x)[1]

    x = rows(4, (6, 8))[0]
    with report.refusals():
        before = jax.make_jaxpr(stepped)(x)
        after = jax.make_jaxpr(algebra.rewritten(stepped))(x)
        assert report.rewrites() == ()
    [scan] = [equation for equation in after.jaxpr.eqns if equation.primitive.name == "scan"]
    [traced] = [equation for equation in before.jaxpr.eqns if equation.primitive.name == "scan"]
    assert str(scan.params["jaxpr"]) == str(traced.params["jaxpr"])

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
    values from outside it: the sums are joined and the program is on par with the unrewritten one, where
    holding the whole chain as one term would pass the interpreter's recursion limit."""
    def deep(x, y, a):
        for _ in range(4000):
            x = jnp.sin(x)
        return two_sums(x, y, a)

    x, y, a = rows(5, (4, 16), (4, 16), (4, 16))
    assert reductions(jax.make_jaxpr(algebra.rewritten(deep))(x, y, a)) == 1

    def draw(seed):
        return rows(seed, (4, 16), (4, 16), (4, 16))

    assert_as_exact(jax.jit(algebra.rewritten(deep)), jax.jit(deep), draw, "two sums after a deep chain",
                    seeds=8)

def test_a_scale_after_a_contraction_moves_onto_its_smaller_operand():
    """A contraction's result scaled by a power of two (a derived walk's `ds k * scale`, docs/design/
    algebra.md §1.4) scales its smaller operand instead, the product being linear in each and the scale
    rounding nothing: the rewritten program multiplies 4 x 4 values where it multiplied 4 x 16, on par with
    the program as written against the float64 truth (check 1). A scale of another value stays where the
    program put it, as moving it reorders rounding."""
    def scaled(g, i):
        return jnp.dot(jnp.tanh(g), jnp.tanh(i)) * 0.25

    def other(g, i):
        return jnp.dot(jnp.tanh(g), jnp.tanh(i)) * 0.3

    g, i = rows(6, (4, 4), (4, 16))
    kept = jax.make_jaxpr(algebra.rewritten(other))(g, i)
    assert [equation.outvars[0].aval.shape for equation in equations(kept)
            if equation.primitive.name == "mul"] == [(4, 16)]

    after = jax.make_jaxpr(algebra.rewritten(scaled))(g, i)
    assert [equation.outvars[0].aval.shape for equation in equations(after)
            if equation.primitive.name == "mul"] == [(4, 4)]

    def draw(seed):
        return rows(seed, (4, 4), (4, 16))

    assert_as_exact(jax.jit(algebra.rewritten(scaled)), jax.jit(scaled), draw,
                    "a scale through a contraction", seeds=16)


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



def test_an_uncatalogued_gpu_prices_forms_by_its_own_facts():
    """A GPU of a known family no row names (an sm121, its multiprocessors, clock and bandwidth its own):
    the pass prices a program by its facts (`cost.target`), not an H100's, so a form is chosen for the
    device the program compiles for."""
    unseen = generation.GpuTarget("sm121", "an unseen consumer Blackwell", 99 * 1024, multiprocessors=48,
                                  clock_hz=2.4e9, memory_bandwidth=273e9)
    with generation.assumed(unseen):
        priced = cost.target()
    assert (priced.generation, priced.multiprocessors, priced.memory_bandwidth) == ("sm121", 48, 273e9)


@pytest.mark.parametrize(("generation_name", "speedup"), [("sm90", 2), ("sm80", 1)])
def test_an_fp8_product_is_priced_in_the_tensor_mode_its_gpu_runs_it_in(generation_name, speedup):
    """An FP8 product's seconds on a GPU: a bf16 product's halved where its tensor cores multiply FP8
    (Hopper's warpgroup MMA), a bf16 product's where its codes are unpacked into 16-bit products (an A100's
    mma.sync), and on no GPU the CUDA cores' (row 3612)."""
    target = generation.TARGETS[generation_name]

    def seconds(dtype):
        priced, unpriced = kernel_cost.units(cost.product_work(4096, 4096, 4096, dtype=dtype),
                                             kernel_cost.rates(target, device=True))
        assert not any(unpriced)
        return max(priced)

    # Exact: each mode's price is the same FLOPs over the same facts, its rate a power of two apart.
    assert seconds(jnp.float8_e4m3fn) == seconds(jnp.bfloat16) / speedup


def test_device_memory_is_priced_at_the_parts_measured_stream():
    """A GPU's device-memory bytes are priced at its part's measured stream (`GpuTarget.part`,
    tools/device_facts.py's plateau), whatever bandwidth its memory clock and bus state, which no transfer
    reaches; a part no one measured, at its clocks' (row 3612)."""
    hopper = generation.TARGETS["sm90"]

    def memory_rate(device_kind, clocked):
        target = dataclasses.replace(hopper, device_kind=device_kind, memory_bandwidth=clocked)
        return kernel_cost.rates(target, device=True)[-1]

    (measured,) = {memory_rate("NVIDIA GH200 480GB", clocked) for clocked in (4.0e12, 4.9e12)}
    assert measured is not None and measured < 4.0e12
    # A part no device reports (its kind the test's own), which no row's streams can name: its clocks'.
    assert memory_rate("a Hopper part no one measured", 2.0e12) == 2.0e12


@pytest.mark.parametrize("tpu", ["v5e", "v6e"])
def test_a_tpus_program_is_rewritten_by_the_rates_it_states(tpu):
    """A TPU's target states its MXU's rate and not its vector or transcendental units' (`cost.cheaper`): a
    program of a product reduced (`sum(a * b)` and two joinable sums of products) traces and runs rewritten
    for it, on par with the program (check 1), a form taken only where it does no more work at a rate the
    TPU states none of."""
    def program_(a, b, x):
        return jnp.sum(a * b, axis=-1) + jnp.sum(a * x, axis=-1)

    with algebra.passes(target=tpu):
        assert_as_exact(jax.jit(algebra.rewritten(program_)), jax.jit(program_),
                        lambda seed: rows(seed, (8, 64), (8, 64), (8, 64)), f"a {tpu} program", seeds=16)


def test_a_products_mode_prices_it():
    """A product of fp32 operands at the default precision (`Precision.DEFAULT`, stated: the suite configures
    the highest) takes the tensor cores' TF32 mode, one of 16-bit operands their 16-bit mode, and one at
    `Precision.HIGHEST` the CUDA cores (`cost.work`): an H100's TF32 rate is half its 16-bit one and its
    CUDA cores' far below both, so the three price apart."""
    def product(precision, dtype):
        def contracted(a, b):
            return jnp.dot(a, b, precision=precision, preferred_element_type=jnp.float32)

        a, b = (jnp.ones((64, 64), dtype) for _ in range(2))
        region = terms.region(jax.make_jaxpr(contracted)(a, b))
        graph, bundle, _ = program.graphed(region)
        _, price = graph.extract(bundle, include_cost=True, extractor="greedy-dag",
                                 cost_model=cost.model(region, cost.fused(cost.target("sm90"))))
        return price.work

    half, tf32, highest = (product(None, jnp.bfloat16), product(jax.lax.Precision.DEFAULT, jnp.float32),
                           product(jax.lax.Precision.HIGHEST, jnp.float32))
    flops = 2.0 * 64 ** 3
    assert (half.flops, tf32.tf32_flops, highest.operations) == (flops, flops, flops)
    # A dot algorithm by its fields is its preset's: three 16-bit products.
    spelled = jax.lax.DotAlgorithm(jnp.bfloat16, jnp.bfloat16, jnp.float32, num_primitive_operations=3)
    assert product(spelled, jnp.float32) == product(jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3, jnp.float32)
    assert product(spelled, jnp.float32).flops == 3 * flops
    # An FP8 dot algorithm's products are the 8-bit mode's, as FP8 operands' are.
    eight = jax.lax.DotAlgorithm(jnp.float8_e4m3fn, jnp.float8_e4m3fn, jnp.float32)
    assert product(eight, jnp.float8_e4m3fn).fp8_flops == flops


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
    """A model calling one jitted layer eight times, nothing to rewrite: with the pass on, every call site
    holds one body, as with it off, and the lowered module has one function for the layer, so a model's
    layers lower and compile once, their kernels too, not once a call site (each rebuilt body made a
    Qwen3.5-4B training step's compile 52 to 487 seconds on an H100). A retrace takes the same rewritten
    body."""
    def model(x, ws):
        for w in ws:
            x = _layer(x, w)
        return jnp.sum(x)

    x, *ws = rows(0, *[(64, 64)] * 9)

    def shared(*, on: bool):
        with algebra.passes(on):
            closed = jax.make_jaxpr(algebra.rewritten(model))(x, ws)
            text = jax.jit(algebra.rewritten(model)).lower(x, ws).as_text()
        bodies = [equation.params["jaxpr"] for equation in closed.jaxpr.eqns
                  if equation.primitive.name in ("pjit", "jit")]
        return {id(body) for body in bodies}, text.count("func.func")

    off, on, again = shared(on=False), shared(on=True), shared(on=True)
    assert (len(on[0]), on[1]) == (len(off[0]), off[1]) == (1, 2)
    assert again[0] == on[0]


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


def test_a_protected_span_is_kept_as_traced_however_often_the_pass_runs():
    """A protected span (`dew.nn.derivatives.Protected`: an error-free transform's) is a boundary no region
    crosses and whose inside no rule touches (docs/design/algebra.md §1.2): two sums the pass joins unmarked
    stay two inside their one marked call, once rewritten and again, a program already rewritten rewritten
    anew."""
    from dew.nn.derivatives import Protected, marked

    operands_ = rows(11, *SHAPES_MARKED)
    assert reductions(jax.make_jaxpr(algebra.rewritten(two_sums))(*operands_)) == 1
    spanned = marked(Protected(), two_sums)
    for rewrite in (algebra.rewritten(spanned), algebra.rewritten(algebra.rewritten(spanned))):
        rewritten = jax.make_jaxpr(rewrite)(*operands_)
        assert (reductions(rewritten), _marked_kinds(rewritten)) == (2, [Protected()])


@pytest.mark.parametrize("kind", ["Surrogate", "KernelGlue"])
def test_a_surrogates_and_a_kernel_glues_body_is_rewritten_within_its_call(kind):
    """A surrogate's bodies and a kernel derivative's glue are rewritten as any other math, within their
    marked call, which stays one call of its kind (§1.2): two sums inside join into one there, and the
    program computes what it computes (check 1)."""
    from dew.nn import derivatives

    spanned = derivatives.marked(getattr(derivatives, kind)(), two_sums)
    rewritten = jax.make_jaxpr(algebra.rewritten(spanned))(*rows(12, *SHAPES_MARKED))
    assert (reductions(rewritten), len(_marked_kinds(rewritten))) == (1, 1)
    assert_as_exact(jax.jit(algebra.rewritten(spanned)), jax.jit(two_sums),
                    lambda seed: rows(seed, *SHAPES_MARKED), f"{kind}'s body", seeds=8)


def test_an_algebra_span_joins_the_windows_around_it():
    """A derivative written only to restructure algebra (`Algebra`) is rewritten freely (§6.10): its body
    opens into the region around it, so a sum inside it joins a sum outside it, and the program computes
    what it computes (check 1)."""
    from dew.nn.derivatives import Algebra, marked

    def inside(x, a):
        return jnp.sum(jnp.tanh(a) * jnp.tanh(x), axis=-1)

    def spanning(x, y, a):
        return marked(Algebra(), inside)(x, a) + jnp.sum(jnp.tanh(a) * jnp.tanh(y), axis=-1)

    rewritten = jax.make_jaxpr(algebra.rewritten(spanning))(*rows(13, *SHAPES_MARKED))
    assert (reductions(rewritten), _marked_kinds(rewritten)) == (1, [])
    assert_as_exact(jax.jit(algebra.rewritten(spanning)), jax.jit(two_sums),
                    lambda seed: rows(seed, *SHAPES_MARKED), "an opened span", seeds=8)


def test_an_algebra_span_with_effects_is_rewritten_within_its_call():
    """An Algebra span whose body has effects (a callback's) is not opened into the region around it, whose
    renaming would leave its effects naming the body's own variables: it is rewritten within its call, as a
    surrogate's body is, which stays one call of its kind, and it computes what it computes (check 1)."""
    from dew.nn.derivatives import Algebra, marked

    def noted(x, y, a):
        jax.debug.callback(lambda value: None, jnp.sum(x))
        return two_sums(x, y, a)

    spanned = marked(Algebra(), noted)
    rewritten = jax.make_jaxpr(algebra.rewritten(spanned))(*rows(15, *SHAPES_MARKED))
    assert (reductions(rewritten) - 1, _marked_kinds(rewritten)) == (1, [Algebra()])
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
    two long arrays apart in an element their printed forms elide), and
    two that are one parameter however they were built (an integer and NumPy's, a dtype and its scalar
    type, two shardings, dimension numbers and precisions built apart)."""
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
    unequal = [(ordered, reordered), (long, changed), (placed, permuted)]
    equal = [((1, 2), (np.int64(1), np.int64(2))), (np.dtype(jnp.bfloat16), jnp.dtype("bfloat16")),
             (NamedSharding(abstract, P("data", None)), NamedSharding(abstract, P("data", None))),
             (numbers, jax.lax.GatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,),
                                                      start_index_map=(0,))),
             (jax.lax.Precision.HIGHEST, jax.lax.Precision("highest")),
             (jax.lax.DotAlgorithmPreset.BF16_BF16_F32, jax.lax.DotAlgorithmPreset.BF16_BF16_F32)]
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
from egglog import i64, rewrite, var
from jax.core import eval_jaxpr

from dew import algebra
from dew.algebra import rules
from dew.algebra.terms import Term


def sums(x, y):
    return jnp.sum(jnp.exp(x), -1) + jnp.sum(jnp.tanh(y), -1)


def branched(x, y):
    return jax.lax.cond(jnp.sum(x) > 0, sums, lambda x, y: sums(y, x), x, y)


rng = np.random.default_rng(0)
x, y = (jnp.asarray(rng.standard_normal((5, 38)), jnp.float32) for _ in range(2))
closed = jax.make_jaxpr(branched)(x, y)


def computed():
    window = jax.jit(algebra.rewritten(sums))(x, y)
    rewritten = algebra.program_of(closed)
    body = eval_jaxpr(rewritten.jaxpr, rewritten.consts, x, y)[0]
    return float(jnp.max(jnp.abs(window - sums(x, y)))), float(jnp.max(jnp.abs(body - sums(x, y))))


before = computed()
variable, params = var("x", Term), var("params", i64)
exponential = rules.RuleFamily("the exponential as its argument", (rewrite(variable.unary("exp", params)).to(
    variable),), (jnp.exp, lambda x: x, None))
rules.register(rules.Rules.declared("registered after", "reorders", (exponential,), "a test's"))
print(json.dumps({"before": before, "after": computed()}))
"""
"""A process that rewrites a program's window and a branch body of one held program, registers a rule set
(one that rewrites an exponential to its argument), and rewrites them again."""


def test_a_rule_set_registered_after_a_rewrite_reaches_the_programs_rewritten_after_it():
    """A window's and a body's rewrite are kept by everything their derivation reads, the rule sets the
    registry admits among them: a set registered after a program was rewritten (a user extending the
    algebra in a running process) reaches the next rewrite of a window of its structure and of the same held
    branch body, where a rewrite kept from before it would stand."""
    done = subprocess.run([sys.executable, "-c", REGISTERED_AFTER], capture_output=True, text=True,
                          check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert max(found["before"]) < 1e-4
    assert min(found["after"]) > 0.1


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
            algebra.program_of(closed)
            reused.append([made.reused for made in report.rewrites()])
    assert reused[0] and not any(reused[0]) and not any(reused[1])


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


def _multiplies(closed) -> int:
    return sum(equation.primitive is jax.lax.mul_p for equation in equations(closed))


def test_a_score_hook_and_its_slopes_pass_the_algebra_pass_before_a_kernel_places_them():
    """A score hook and its derivative are rewritten by the algebra pass before a kernel's tile places them
    (docs/design/algebra.md §2): the tile's hook (`attention_algebra.hooked`) and its slopes
    (`attention_gradient.scored`) trace the hook's cheaper form, one multiply of the two scales' sum, which
    they compute as the hook does (check 1); with the pass off, the hook as written."""
    from dew.nn.kernels.templates.attention_algebra import hooked
    from dew.nn.kernels.templates.attention_gradient import scored

    q_pos, k_pos = jnp.arange(16)[:, None], jnp.arange(32)[None, :]

    def tile(s, a, b):
        return hooked(_scaled_twice, s, q_pos, k_pos, (a, b))

    def sloped(s, a, b):
        return scored(_scaled_twice, s, q_pos, k_pos, (a, b), slopes=True)

    def draw(seed):
        return _scaled_operands(seed, (16, 32))

    assert _multiplies(jax.make_jaxpr(tile)(*draw(0))) == 1
    with algebra.passes(on=False):
        assert _multiplies(jax.make_jaxpr(tile)(*draw(0))) == 2
    rewritten, written = jax.make_jaxpr(sloped)(*draw(0)), None
    with algebra.passes(on=False):
        written = jax.make_jaxpr(sloped)(*draw(0))
    assert _multiplies(rewritten) < _multiplies(written)
    assert_as_exact(jax.jit(tile), jax.jit(lambda s, a, b: _scaled_twice(s, q_pos, k_pos, a, b)), draw,
                    "a rewritten score hook", seeds=8)


def test_a_fused_gemms_producer_passes_the_algebra_pass_before_the_fuser_places_it():
    """A fused GEMM's producer is rewritten by the algebra pass before the fuser stages it (algebra.md §2):
    a producer of two scales of one operand summed is staged as its cheaper form, its window's rewrite
    spliced (the fused products' values against their references: tests/test_gemm_template.py)."""
    from dew.nn.kernels.templates.gemm import fused

    def producer(values):
        x, a, b = values
        capped = jnp.tanh(x)
        return capped * jnp.tanh(a) + capped * jnp.tanh(b)

    with report.refusals():
        fused._split(producer, _scaled_operands(1, (64, 128)))
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
