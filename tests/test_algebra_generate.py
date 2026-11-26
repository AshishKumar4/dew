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
