"""The generator (docs/design/algebra.md §6.11, `dew.algebra.generate`): each written module is what
generation writes (its `--check`, run here on every train), each generated form is on par with its
declaration (check 1), a printed program computes what its jaxpr does bit for bit and its compensated print
keeps what its sums round away (`dew.algebra.printed`), and §1.5's acceptance cases are generated as their
error decides (`dew.algebra.acceptance`)."""

import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew import algebra
from dew.algebra import acceptance, error, generate, printed

FORMS = [(name, policy) for name, form in generate.DECLARATIONS.items() for policy in form.cases.policies]


def _derived(name: str):
    """The generated function of a declaration, from its module beside the declaring one."""
    module, function = name.rsplit(".", 1)
    return getattr(importlib.import_module(f"{module}_derived"), function)


def test_the_written_modules_are_what_generation_writes():
    """Every declaring module's generated module, regenerated in memory, is the one written (`--check`): a
    changed rule, declaration or chooser lands as a reviewed diff of generated code."""
    made = generate.generate(check=True)
    assert {name for found in made for name in found.declarations} == set(generate.DECLARATIONS)


@pytest.mark.parametrize(("name", "policy"), FORMS, ids=[f"{name}-{policy}" for name, policy in FORMS])
def test_each_generated_form_is_on_par_with_its_declaration(name, policy):
    """Each generated form at each of its declaration's cases is on par with the declaration as written
    against its float64 truth (check 1), on the declaration's own draws."""
    form = generate.DECLARATIONS[name]
    for operands in form.cases.operands:
        assert_as_exact(jax.jit(functools.partial(_derived(name), policy=policy)), jax.jit(form.function),
                        lambda seed, operands=operands: form.cases.draw(seed, operands), f"{name} ({policy})",
                        seeds=form.cases.seeds, truth_fn=form.function)


def nested(x, y):
    """A jit within the program and a constant it closes over."""
    table = jnp.asarray(np.arange(16.0, dtype=np.float32).reshape(4, 4))
    return jax.jit(lambda a: jnp.tanh(a) @ table)(x) + y


PROGRAMS = {
    "products and reductions": lambda x, y: (jnp.dot(x, y.T), jnp.sum(x * 2.5, axis=1), jnp.max(y, axis=0)),
    "a nested call and a constant": nested,
    "selection, powers and casts": lambda x, y: jnp.where(x > y, x ** 3,
                                                          x.astype(jnp.bfloat16).astype(jnp.float32)),
    "a sort's several results": lambda x, y: jax.lax.sort((x, y), num_keys=1),
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
