"""The generator (docs/design/algebra.md §6.11, `dew.algebra.generate`): each written module is what
generation writes (its `--check`, run here on every train), each generated form is on par with its
declaration (check 1), a printed program computes what its jaxpr does bit for bit and its compensated print
keeps what its sums round away (`dew.algebra.printed`), and §1.5's acceptance cases are generated as their
error decides (`dew.algebra.acceptance`)."""

import functools
import importlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew import algebra
from dew.algebra import acceptance, component_cases, error, generate, printed

FORMS = [(name, policy) for name, form in generate.DECLARATIONS.items() for policy in form.cases.policies]


def _derived(name: str):
    """The generated function of a declaration, from its module beside the declaring one."""
    module, function = name.rsplit(".", 1)
    return getattr(importlib.import_module(f"{module}_derived"), function)


@pytest.fixture
def on_its_platform():
    """The platform a generated module was measured on, which its header names and its regeneration
    measures on again: armada's CPU, the one CI pins."""
    if jax.default_backend() != "cpu":
        pytest.skip("a generated module was measured on the CPU, and regenerates as its header says there")


def test_the_written_modules_are_what_generation_writes(on_its_platform):
    """Every declaring module's generated module, regenerated in memory, is the one written (`--check`): a
    changed rule, declaration or chooser lands as a reviewed diff of generated code."""
    made = generate.generate(check=True)
    assert {name for found in made for name in found.declarations} == set(generate.DECLARATIONS)


def test_the_command_checks_the_declaring_modules_own_declarations(on_its_platform):
    """`python -m dew.algebra.generate --check`, run as a script (its module `__main__`, beside the one the
    declaring modules register with), checks the declarations they register, each named: it had held an
    empty registry of its own, which knew none of them (and passed checking nothing where none was
    named)."""
    done = subprocess.run([sys.executable, "-m", "dew.algebra.generate", "--check",
                           *sorted(generate.DECLARATIONS)], check=False)
    assert done.returncode == 0


@pytest.mark.parametrize(("name", "policy"), FORMS, ids=[f"{name}-{policy}" for name, policy in FORMS])
def test_each_generated_form_is_on_par_with_its_declaration(name, policy):
    """Each generated form at each of its declaration's cases is on par with the declaration as written
    against its float64 truth (check 1), on the declaration's own draws."""
    form = generate.DECLARATIONS[name]
    truth = form.truth if isinstance(form, generate.ComponentForm) else form.function
    for operands in form.cases.operands:
        assert_as_exact(jax.jit(functools.partial(_derived(name), policy=policy)), jax.jit(form.function),
                        lambda seed, operands=operands: form.cases.draw(seed, operands), f"{name} ({policy})",
                        seeds=form.cases.seeds, truth_fn=truth)


@pytest.mark.parametrize("name", ["query_weights", "key_weights", "query_dropped_cotangents",
                                  "key_dropped_cotangents", "selective_decay", "key_memory_decay",
                                  "softcap_tile", "sink_state", "sliding_visibility", "selective_adjoint",
                                  "kda_write", "gdn_value_pullback", "ssd_chunk_forward", "ssd_chunk_pullback",
                                  "gemm_partial_scale", "fp8_epilogue"])
@pytest.mark.parametrize("arm", ["written", "generated"])
def test_new_template_components_certify_every_primitive_on_the_cpu(name, arm, on_its_platform):
    from certified import assert_certified

    qualified = f"{component_cases.__name__}.{name}"
    declaration = generate.DECLARATIONS[qualified]
    for policy in declaration.cases.policies:
        function = (declaration.function if arm == "written" else
                    functools.partial(_derived(qualified), policy=policy))
        for index, operands in enumerate(declaration.cases.operands):
            assert_certified(function, functools.partial(declaration.cases.draw, operands=operands),
                             f"{name} ({arm}, {policy}, {index})", seeds=declaration.cases.seeds)


def test_shared_component_preserves_layouts_constants_and_live_results():
    from dew.algebra import _component_sharing, terms

    def tile(value, row, *, axis, keep_product, bias):
        extended = jax.lax.broadcast_in_dim(row + bias, value.shape, (axis,))
        product = value * extended
        base = jnp.abs(product + value)
        root = jnp.sqrt(base)
        shifted = root + extended
        result = jax.lax.logistic(shifted)
        return (result, product) if keep_product else (result,)

    layouts = {"row_first": ((7, 13), 0, False, -0.5), "key_first": ((13, 7), 1, True, 0.5)}
    arguments, separate = {}, {}
    for name, (shape, axis, keep_product, shift) in layouts.items():
        arguments[name] = (jax.ShapeDtypeStruct(shape, jnp.bfloat16),
                           jax.ShapeDtypeStruct((shape[axis],), jnp.bfloat16))
        bias = (np.arange(shape[axis]) * 0.25 + shift).astype(jnp.bfloat16)
        fn = functools.partial(tile, axis=axis, keep_product=keep_product, bias=bias)
        separate[name] = printed.source(jax.make_jaxpr(fn)(*arguments[name]), name, borrow_shapes=True)[0]
    # An existing emitted body and a public wrapper can both occupy the helper's preferred name.
    occupied = "_shared_component_0"
    arguments[occupied] = (jax.ShapeDtypeStruct((7,), jnp.bfloat16),)
    separate[occupied] = printed.source(jax.make_jaxpr(lambda x: (x * x,))(*arguments[occupied]),
                                        occupied, borrow_shapes=True)[0]
    reserved = "_shared_component_1"
    shared = _component_sharing.shared(separate, reserved=(reserved,))
    assert len(shared) == len(separate) + 1 and reserved not in shared
    for name, operands in arguments.items():
        before = printed.loaded(separate[name], name)
        after = printed.loaded("\n\n".join(shared.values()), name)
        original = jax.make_jaxpr(before)(*operands)
        composed = jax.make_jaxpr(after)(*operands)
        values = tuple(np.full(operand.shape, 0.75, operand.dtype) for operand in operands)
        for actual, expected in zip(jax.jit(after)(*values), jax.jit(before)(*values), strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert terms.jaxpr_digest(original, in_process=True) == terms.jaxpr_digest(composed, in_process=True)


@pytest.mark.parametrize("kind", ["placed", "duplicate"])
def test_component_dispatch_refuses_cases_its_flat_key_cannot_distinguish(kind):
    (example,) = component_cases.ROUNDED.cases.operands[0]
    if kind == "placed":
        mesh = jax.make_mesh((1,), ("data",), devices=jax.devices()[:1])
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        operands = ((jax.ShapeDtypeStruct(example.shape, example.dtype, sharding=sharding),),)
    else:
        operands = ((example,), (example,))
    form = component_cases.ROUNDED
    cases = generate.Cases(operands, form.cases.draw, seeds=form.cases.seeds)
    form = generate.ComponentForm(form.function, form.truth, cases)
    with pytest.raises(ValueError):
        generate._chosen("bad.components", form)


def test_a_component_preserves_one_result_convention_across_cases():
    def changed(x):
        return x if x.shape == (1,) else (x, x + 1)

    def draw(seed, operands):
        return (np.full(operands[0].shape, seed, np.float32),)

    shapes = tuple((jax.ShapeDtypeStruct((n,), jnp.float32),) for n in (1, 2))
    cases = generate.Cases(shapes, draw, seeds=2)
    with pytest.raises(ValueError):
        generate._chosen("bad.component", generate.ComponentForm(changed, changed, cases))


def test_generated_query_cotangents_default_to_the_accurate_small_weight_order():
    from dew.algebra.component_cases_derived import queries_cotangent

    cases = component_cases.QUERIES_COTANGENT.cases
    inputs = cases.draw(1, cases.operands[0])
    _, weights, cotangents = jax.jit(queries_cotangent)(*inputs)
    assert weights[0, 1] == 0
    # The shared Check 1 draw also checks this value against independent float64 arithmetic.
    assert cotangents[0, 1] > jnp.finfo(cotangents.dtype).tiny


def test_generated_online_tiles_and_split_merge_compute_the_same_full_row():
    from dew.algebra.component_cases_derived import online_softmax_step, split_kv_merge

    def empty(logits, values):
        rows = logits.shape[0]
        return (jnp.zeros((rows, values.shape[1]), logits.dtype),
                jnp.full((rows,), jnp.finfo(jnp.float32).min, logits.dtype), jnp.zeros((rows,), logits.dtype))

    def generated(left, left_values, right, right_values):
        start = empty(left, left_values)
        first = online_softmax_step(*start, left, left_values)
        second = online_softmax_step(*start, right, right_values)
        return split_kv_merge(*first, *second)

    def split(left, left_values, right, right_values):
        start = empty(left, left_values)
        first = component_cases._online_truth(*start, left, left_values)
        second = component_cases._online_truth(*start, right, right_values)
        return component_cases._split_kv_truth(*first, *second)

    def whole(left, left_values, right, right_values):
        logits = jnp.concatenate((left, right), axis=1)
        values = jnp.concatenate((left_values, right_values), axis=0)
        return component_cases._online_truth(*empty(logits, values), logits, values)

    def draw(seed):
        rng = np.random.default_rng(seed)
        left, right = rng.normal(size=(2, 32, 64)).astype(np.float32)
        left[0], right[0], right[1] = -np.inf, -np.inf, -np.inf
        right[2] += 12
        first, second = rng.normal(size=(2, 64, 64)).astype(np.float32)
        return left, first, right, second

    with jax.default_matmul_precision("highest"):
        assert_as_exact(jax.jit(generated), jax.jit(split), draw, "generated split-KV tile composition",
                        seeds=8, truth_fn=whole)


def _generated_inverse(system):
    from dew.algebra.component_cases_derived import chunk_solve_level

    size = system.shape[0]
    place = jnp.arange(size)
    identity = jnp.eye(size, dtype=system.dtype)
    paired = (place[:, None] % 2 == 1) & (place[None, :] == place[:, None] - 1)
    inverse = identity - jnp.where(paired, system, 0)
    for level in range(1, size.bit_length() - 1):
        inverse = chunk_solve_level(system, inverse, jnp.asarray(level, jnp.int32))
    return inverse - identity


@pytest.mark.parametrize(("size", "columns"), [(16, 32), (32, 32), (64, 64)])
def test_the_generated_chunk_solve_and_application_satisfy_the_same_triangular_system(size, columns):
    from dew.algebra.component_cases_derived import chunk_weighted
    from dew.nn.kernels import recurrence_algebra

    def generated(system, values, strengths):
        return chunk_weighted(_generated_inverse(system), values, strengths)

    def written(system, values, strengths):
        lower = recurrence_algebra.solved(system, component_cases._carry_product)
        return component_cases.CHUNK_WEIGHTED.function(lower, values, strengths)

    def truth(system, values, strengths):
        rhs = values * strengths[:, None]
        return jax.lax.linalg.triangular_solve(system, rhs, left_side=True, lower=True, unit_diagonal=True)

    def draw(seed):
        system = component_cases._chunk_system(seed, size)
        rng = np.random.default_rng(seed)
        rhs = rng.uniform(-0.25, 0.25, (size, columns)).astype(np.float32)
        strength = rng.uniform(0.125, 0.5, size).astype(np.float32)
        strength[1] = 0
        return system, rhs, strength

    assert_as_exact(generated, written, draw, "generated recurrence solve and value application",
                    seeds=8, truth_fn=truth)


@pytest.mark.parametrize("size", [16, 32, 64])
def test_generated_inverse_adjoint_transposes_the_same_triangular_system(size):
    from dew.algebra.component_cases_derived import chunk_inverse_pullback
    from dew.nn.kernels import recurrence_algebra

    def generated(system, cotangent):
        return chunk_inverse_pullback(_generated_inverse(system), cotangent)

    def written(system, cotangent):
        lower = recurrence_algebra.solved(system, component_cases._carry_product)
        return component_cases.CHUNK_INVERSE_PULLBACK.function(lower, cotangent)

    def truth(system, cotangent):
        identity = jnp.eye(size, dtype=system.dtype)

        def solve(a):
            return jax.lax.linalg.triangular_solve(
                a, identity, left_side=True, lower=True, unit_diagonal=True)

        _, pull = jax.vjp(solve, system)
        return pull(jnp.tril(cotangent, -1))[0]

    def draw(seed):
        rng = np.random.default_rng(seed)
        # Same-shape adversarial draws can also use the system as the cotangent. Keep the closed
        # solve and its transpose representable together; the component gate covers the isolated edge.
        system = np.tril(rng.uniform(-0.125 / size, 0.125 / size, (size, size)).astype(np.float32), -1)
        cotangent = rng.normal(size=system.shape).astype(np.float32)
        return system, cotangent

    assert_as_exact(generated, written, draw, "generated inverse VJP of a triangular solve",
                    seeds=8, truth_fn=truth)


def test_generated_value_gradient_accumulates_two_query_tiles():
    from dew.algebra.component_cases_derived import values_gradient

    def generated(scores, cotangent, peak, total):
        return (values_gradient(scores[:32], cotangent[:32], peak[:32], total[:32])
                + values_gradient(scores[32:], cotangent[32:], peak[32:], total[32:]))

    def written(scores, cotangent, peak, total):
        return component_cases.VALUES_GRADIENT.function(scores, cotangent, peak, total)

    def truth(scores, cotangent, peak, total):
        keep = jnp.isfinite(peak)
        weights = jnp.exp(jnp.where(keep[:, None], scores - peak[:, None], -jnp.inf))
        return weights.T @ (cotangent / jnp.where(total > 0, total, 1)[:, None])

    def draw(seed):
        case = component_cases.VALUES_GRADIENT.cases
        left = case.draw(seed, case.operands[0])
        right = case.draw(seed + 19, case.operands[0])
        return tuple(np.concatenate((x, y)) for x, y in zip(left, right, strict=True))

    assert_as_exact(generated, written, draw, "generated dV across query tiles", seeds=8, truth_fn=truth)


def test_generated_rounded_bodies_keep_the_written_surrogate_law():
    from dew.algebra import component_cases_derived as emitted
    from dew.nn import derivatives, precision

    def tangent(primals, tangents, value):
        return emitted.rounded_tangent(primals[0], tangents[0], value)

    rounded = derivatives.surrogate(emitted.rounded, tangent)
    x = np.full((16, 32), 1.003, np.float32)
    dx = np.full_like(x, 1.007)
    primal, direction = jax.jit(lambda x, dx: jax.jvp(rounded, (x,), (dx,)))(x, dx)
    np.testing.assert_array_equal(primal, precision.rounded_operand(x, jnp.bfloat16))
    np.testing.assert_array_equal(direction, dx)
    _, pullback = jax.vjp(rounded, jnp.asarray(x))
    np.testing.assert_array_equal(pullback(jnp.asarray(dx))[0], dx)
    gradient = jax.grad(lambda x: jnp.sum(rounded(x) ** 2))
    _, higher = jax.jit(lambda x, dx: jax.jvp(gradient, (x,), (dx,)))(x, dx)
    np.testing.assert_array_equal(higher, 2 * dx)


def nested(x, y):
    """A jit within the program and a constant it closes over."""
    table = jnp.asarray(np.arange(16.0, dtype=np.float32).reshape(4, 4))
    return jax.jit(lambda a: jnp.tanh(a) @ table)(x) + y


PAYLOAD = np.frombuffer(np.uint32(0x7FC00001).tobytes(), np.float32)[0]
"""A float32 NaN of a payload of its own, not the canonical one."""


PROGRAMS = {
    "products and reductions": lambda x, y: (jnp.dot(x, y.T), jnp.sum(x * 2.5, axis=1), jnp.max(y, axis=0)),
    "a nested call and a constant": nested,
    "selection, powers and casts": lambda x, y: jnp.where(x > y, x ** 3,
                                                          x.astype(jnp.bfloat16).astype(jnp.float32)),
    "a sort's several results": lambda x, y: jax.lax.sort((x, y), num_keys=1),
    "a sort's one result": lambda x, y: jax.lax.sort(x),
    "a NaN's payload": lambda x, y: jax.lax.bitcast_convert_type(jnp.full_like(x, PAYLOAD), jnp.int32),
}


@pytest.mark.parametrize("name", list(PROGRAMS))
def test_a_printed_program_computes_what_its_jaxpr_does_bit_for_bit(name):
    """A program printed as plain JAX (`printed.source`, each equation its primitive bound at its own
    parameters, a call's body inlined, a constant by its bytes) and loaded computes each output bit for bit
    as its jaxpr does."""
    rng = np.random.default_rng(0)
    operands = tuple(jnp.asarray(rng.standard_normal((4, 4)), jnp.float32) for _ in range(2))
    closed = jax.make_jaxpr(PROGRAMS[name])(*operands)
    text, compensated = printed.source(closed, "program")
    assert not compensated
    got = printed.loaded(text, "program")(*operands)
    expected = jax.core.eval_jaxpr(closed.jaxpr, closed.consts, *operands)
    assert len(got) == len(expected)
    for ours, theirs in zip(got, expected, strict=True):
        ours, theirs = np.asarray(ours), np.asarray(theirs)
        assert (ours.dtype, ours.shape) == (theirs.dtype, theirs.shape)
        assert ours.tobytes() == theirs.tobytes()


def test_a_compensated_print_keeps_what_its_sums_round_away():
    """A sum's terms that its float rounds away kept by its compensated print (`printed.source`'s
    `compensated`, the error-free derivation): `(a + b) - a` of `a = 1e8`, `b = 0.75` is `b` where its plain
    print gives zero, and a sum's reduction of `[1e8, 1, -1e8]` is one where its plain reduction's need not
    be."""
    def kept(a, b, row):
        return (a + b) - a, jnp.sum(row)

    a, b = jnp.float32(1e8), jnp.float32(0.75)
    row = jnp.asarray([1e8, 1.0, -1e8], jnp.float32)
    closed = jax.make_jaxpr(kept)(a, b, row)
    text, compensated = printed.source(closed, "kept", compensated=True)
    assert compensated
    difference, total = printed.loaded(text, "kept")(a, b, row)
    assert float(jax.jit(lambda a, b: (a + b) - a)(a, b)) == 0.0
    assert (float(difference), float(total)) == (0.75, 1.0)


def test_a_nested_program_called_twice_is_printed_with_each_calls_own_operands():
    """One nested program called on two operands (its one trace read by both calls), printed plain and
    compensated: each call adds one to its own operand, the second's never the first's (a name, a double
    word or a read of the first call's left bound to the second's)."""
    inner = jax.jit(lambda x: x + 1.0)

    def twice(a, b, c):
        return inner(a + b), inner(c)

    operands = (jnp.float32(2.0), jnp.float32(3.0), jnp.float32(7.0))
    closed = jax.make_jaxpr(twice)(*operands)
    for compensated in (False, True):
        text, _ = printed.source(closed, "twice", compensated=compensated)
        assert [float(value) for value in printed.loaded(text, "twice")(*operands)] == [6.0, 8.0]


def test_an_output_whose_weak_type_a_plain_call_cannot_keep_is_refused():
    """A program returning a Python float, a literal it holds weak: its printed literal is a typed array,
    strong, so the printed program's output is not the program's and it is refused, never printed with
    another type."""
    closed = jax.make_jaxpr(lambda x: (x, 1.0))(jnp.ones((4,), jnp.float32))
    with pytest.raises(printed.Unprintable):
        printed.source(closed, "weak")


def test_an_equation_no_plain_call_writes_is_refused():
    """A custom derivative's call (a ReLU's) is no value's equation a plain call writes as the jaxpr holds
    it, so its program is refused, never printed without its rule."""
    closed = jax.make_jaxpr(jax.nn.relu)(jnp.ones((4,), jnp.float32))
    with pytest.raises(printed.Unprintable):
        printed.source(closed, "relu")


def test_the_folded_probabilities_are_generated_without_the_saved_logarithm_under_the_accurate_policy():
    """§1.5's log-sum-exp case: under the accurate policy the generated probabilities never round the
    total's logarithm into a logit-sized value (no logarithm in its program), where the form as written
    takes one."""
    (operands,) = generate.DECLARATIONS[f"{acceptance.__name__}.lse_folded_probabilities"].cases.operands
    accurate = jax.make_jaxpr(functools.partial(_derived(f"{acceptance.__name__}.lse_folded_probabilities"),
                                                policy="accurate"))(*operands)
    written = jax.make_jaxpr(acceptance.lse_folded_probabilities)(*operands)
    assert [equation.primitive for equation in written.jaxpr.eqns].count(jax.lax.log_p) == 1
    assert jax.lax.log_p not in [equation.primitive for equation in accurate.jaxpr.eqns]


def test_dvs_normalization_is_proposed_on_either_side_of_its_contraction():
    """§1.5's dV case: the chooser proposes the normalization by each query's total before the contraction,
    as written, and moved onto the cotangent (the stable set's scale along a contraction), so the choice
    between them is the error's; the written form divides the probabilities alone."""
    (operands,) = generate.DECLARATIONS[f"{acceptance.__name__}.normalized_values_product"].cases.operands
    cotangent = operands[2].shape
    choice = error.choose(acceptance.normalized_values_product, operands, accurate=False,
                          saturation=algebra.SATURATION, admits=lambda candidate, reference: True)

    def divides_the_cotangent(candidate: error.Candidate) -> bool:
        closed = jax.make_jaxpr(candidate.form or candidate.function)(*operands)
        return any(equation.primitive is jax.lax.div_p and equation.outvars[0].aval.shape == cotangent
                   for equation in closed.jaxpr.eqns)

    found = [divides_the_cotangent(candidate) for candidate in choice.candidates]
    assert not found[[candidate.name for candidate in choice.candidates].index("as written")]
    assert any(found)
