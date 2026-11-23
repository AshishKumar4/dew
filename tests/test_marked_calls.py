"""Marked calls (docs/design/algebra.md §6.10): a span of a computation carried as one call of its kind
(`dew.nn.derivatives.marked`), which every transform keeps a call of that kind, so the algebra pass reads a
span's kind from the program it rewrites, its forward, tangent and cotangent alike."""

import datetime
import functools
import io
import typing
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.custom_derivatives import SymbolicZero
from jax.experimental import checkify
from jax.sharding import Mesh, PartitionSpec
from reference_error import equations

from dew.nn import derivatives
from dew.nn.derivatives import KernelGlue, Protected, Surrogate, marked, marked_call_p
from dew.nn.programs import called

STATED = {derivatives.Activation: [derivatives.Activation("swiglu", gated=True),
                                   derivatives.Activation("gelu", gated=False)]}
"""The kinds a span states fields of, each as the spans that mark it state them: an activation's region by its
activation's name, gated and ungated."""

KINDS = [stated for kind in typing.get_args(derivatives.Kind.__value__)
         for stated in (STATED[kind] if kind in STATED else [kind()])]
"""Every kind a span is marked as, each through either builder (`derivatives.Kind`'s)."""


def kinds_of(closed) -> list:
    """The kind of every marked call a program holds, its bodies' included."""
    return [equation.params["kind"] for equation in equations(closed) if equation.primitive is marked_call_p]


def body(x, y):
    return jnp.sin(x) * y + jnp.sum(x * y)


@pytest.mark.parametrize("kind", KINDS, ids=repr)
def test_a_marked_span_computes_as_its_body_and_stays_one_call_of_its_kind_through_every_transform(kind):
    """A marked span computes what its body does, and its program holds it as one call of its kind; so do
    its derivatives forward and backward (`jax.jvp`, `jax.linearize`, `jax.grad`), its batched form
    (`jax.vmap`), its checkpointed one (`jax.checkpoint`), its sharded one (`jax.shard_map`) and its
    checked one (`checkify.checkify`, as a sampler's decode step checks its program), which runs as its body
    does."""
    spanned = marked(kind, body)
    x, y = jnp.linspace(0.1, 1.0, 8), jnp.linspace(1.0, 2.0, 8)
    np.testing.assert_array_equal(np.asarray(jax.jit(spanned)(x, y)), np.asarray(jax.jit(body)(x, y)))
    error, checked = jax.jit(checkify.checkify(spanned))(x, y)
    error.throw()
    np.testing.assert_array_equal(np.asarray(checked), np.asarray(jax.jit(body)(x, y)))

    def loss(x, y):
        return jnp.sum(spanned(x, y))

    transformed = {
        "call": lambda x, y: spanned(x, y),
        "jvp": lambda x, y: jax.jvp(spanned, (x, y), (y, x))[1],
        "linearize": lambda x, y: jax.linearize(spanned, x, y)[1](y, x),
        "grad": jax.grad(loss, argnums=(0, 1)),
        "vmap": jax.vmap(spanned),
        "checkpoint": jax.grad(jax.checkpoint(loss)),
        "shard_map": jax.shard_map(spanned, mesh=Mesh(np.array(jax.devices()[:1]), ("rows",)),
                                   in_specs=PartitionSpec("rows"), out_specs=PartitionSpec("rows")),
        "checkify": checkify.checkify(spanned),
    }
    for name, function in transformed.items():
        found = kinds_of(jax.make_jaxpr(function)(x, y) if name != "vmap"
                         else jax.make_jaxpr(function)(x[None], y[None]))
        assert found and all(seen == kind for seen in found), name


def test_a_marked_spans_gradient_is_its_bodys():
    """A marked span differentiates as its body does: its gradient on par with the body's, exactly, the
    marker computing nothing of its own."""
    x, y = jnp.linspace(0.1, 1.0, 8), jnp.linspace(1.0, 2.0, 8)
    spanned = jax.grad(lambda x, y: jnp.sum(marked(Protected(), body)(x, y)), argnums=(0, 1))
    plain = jax.grad(lambda x, y: jnp.sum(body(x, y)), argnums=(0, 1))
    for got, expected in zip(jax.jit(spanned)(x, y), jax.jit(plain)(x, y), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def _scaled_backward(scale: float):
    """A function `sin` whose declared backward scales its cotangent by `scale`, so a test sees the rule ran,
    its static `scale` passed beside the arrays."""
    def primal(x, factor):
        return jnp.sin(x) * factor

    def forward(x, factor):
        return primal(x, factor), (x, factor)

    def backward(saved, cotangent):
        x, factor = saved
        return scale * cotangent * jnp.cos(x) * factor, jnp.zeros_like(factor)

    return primal, forward, backward


@pytest.mark.parametrize("kind", KINDS, ids=repr)
def test_a_declared_derivative_is_one_marked_call_and_differentiates_as_its_rule(kind):
    """A custom derivative declared with its kind (`declared_derivative`: no bare `jax.custom_vjp`) is one
    marked call of that kind, its primal, forward and backward together, which every transform keeps: its
    value is its primal's, reverse mode runs its declared backward (a cotangent scaled by 3) in calls of its
    kind, a second order differentiates that backward, and forward mode refuses it exactly as a bare
    `jax.custom_vjp` does."""
    primal, forward, backward = _scaled_backward(3.0)
    declared = derivatives.declared_derivative(kind, primal, forward, backward)
    bare = jax.custom_vjp(primal)
    bare.defvjp(forward, backward)
    x, factor = jnp.linspace(0.1, 1.0, 8), jnp.full(8, 2.0)
    np.testing.assert_array_equal(np.asarray(declared(x, factor)), np.asarray(primal(x, factor)))
    assert kinds_of(jax.make_jaxpr(declared)(x, factor)) == [kind]

    def loss(function):
        return lambda x: jnp.sum(function(x, factor))

    np.testing.assert_array_equal(np.asarray(jax.grad(loss(declared))(x)),
                                  np.asarray(jax.grad(loss(bare))(x)))
    assert not np.array_equal(np.asarray(jax.grad(loss(bare))(x)), np.asarray(jax.grad(loss(primal))(x)))
    # Reverse mode splits the call into its forward and its transpose, each a call of its kind.
    found = kinds_of(jax.make_jaxpr(jax.grad(loss(declared)))(x))
    assert found and all(seen == kind for seen in found)
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda x: jnp.sum(jax.grad(loss(declared))(x)))(x)),
                                  np.asarray(jax.grad(lambda x: jnp.sum(jax.grad(loss(bare))(x)))(x)))
    with pytest.raises(TypeError) as marked_error:
        jax.jvp(lambda x: declared(x, factor), (x,), (x,))
    with pytest.raises(TypeError) as bare_error:
        jax.jvp(lambda x: bare(x, factor), (x,), (x,))
    assert str(marked_error.value) == str(bare_error.value)
    np.testing.assert_array_equal(np.asarray(jax.vmap(declared)(x[:, None], factor[:, None])),
                                  np.asarray(jax.vmap(bare)(x[:, None], factor[:, None])))


@pytest.mark.parametrize("kind", [kind() for kind in typing.get_args(derivatives.TangentKind.__value__)],
                         ids=lambda kind: type(kind).__name__)
def test_a_declared_tangent_is_one_marked_call_and_differentiates_as_its_rule(kind):
    """A tangent rule declared with its kind (`declared_tangent`: no bare `jax.custom_jvp`; a surrogate's is
    `surrogate`'s) is one marked call of that kind, which forward mode and reverse mode (its tangent
    transposed) keep, each from the declared tangent (doubled here, so a test sees it ran)."""
    def primal(x):
        return jnp.sin(x)

    def tangent(primals, tangents):
        [x], [dx] = primals, tangents
        return jnp.sin(x), 2.0 * jnp.cos(x) * dx

    declared = derivatives.declared_tangent(kind, primal, tangent)
    bare = jax.custom_jvp(primal)
    bare.defjvp(tangent)
    x = jnp.linspace(0.1, 1.0, 8)
    assert kinds_of(jax.make_jaxpr(declared)(x)) == [kind]
    for function in (lambda f: jax.jvp(f, (x,), (jnp.ones_like(x),))[1],
                     lambda f: jax.grad(lambda x: jnp.sum(f(x)))(x)):
        np.testing.assert_array_equal(np.asarray(function(declared)), np.asarray(function(bare)))
    assert not np.array_equal(np.asarray(jax.grad(lambda x: jnp.sum(bare(x)))(x)), np.asarray(jnp.cos(x)))
    found = kinds_of(jax.make_jaxpr(jax.grad(lambda x: jnp.sum(declared(x))))(x))
    assert found and all(seen == kind for seen in found)


def test_an_eager_call_of_a_marked_span_traces_no_program():
    """A marked span called eagerly, outside any trace, a call no program records, is its function's own
    call: once its operations have run, it traces and compiles nothing again (a decision task's eager
    scores in a step's steady state, train 586), and computes its function's values. Inside a trace it is
    one call of its custom rule whose primal is one marked call, its arguments traced or concrete."""
    from dew.nn.categorical import label_terms
    from dew.telemetry.compiles import Compiles

    logits = jnp.asarray(np.random.default_rng(0).normal(size=(4, 8)), jnp.float32)
    labels = jnp.asarray([0, 3, 7, 2])
    first = label_terms(logits, labels)
    with Compiles(None) as heard:
        again = label_terms(logits, labels)
    assert not heard.programs, [(program.kind, program.name, program.where) for program in heard.programs]
    for each, expected in zip(again, first, strict=True):
        np.testing.assert_array_equal(np.asarray(each), np.asarray(expected))
    for traced in (jax.make_jaxpr(label_terms)(logits, labels).jaxpr,
                   jax.make_jaxpr(lambda: label_terms(logits, labels))().jaxpr):
        (rule,) = traced.eqns
        assert [equation.primitive for equation in called(rule).eqns] == [derivatives.marked_call_p], traced


def test_a_declared_rule_with_symbolic_zeros_meets_one_where_no_tangent_reaches():
    """With `symbolic_zeros`, an operand no tangent reaches is a `SymbolicZero` in a declared tangent rule
    (`declared_tangent`) and unperturbed in a declared forward (`declared_derivative`), through the marked
    call as the bare `jax.custom_jvp` and `jax.custom_vjp` give it, and each differentiates as the bare one
    does: a rule that skips a zero's term (a tangent JAX cannot transpose through a barrier) sees it."""
    seen = []

    def primal(x, y):
        return jnp.sin(x) * y

    def tangent(primals, tangents):
        (x, y), (dx, dy) = primals, tangents
        seen.append(type(dy))
        return primal(x, y), 2.0 * jnp.cos(x) * y * dx

    def forward(x, y):
        seen.append(y.perturbed)
        return primal(x.value, y.value), (x.value, y.value)

    def backward(residuals, cotangent):
        x, y = residuals
        return 3.0 * cotangent * jnp.cos(x) * y, jnp.zeros_like(y)

    tangents = derivatives.declared_tangent(Protected(), primal, tangent, symbolic_zeros=True)
    bare_tangents = jax.custom_jvp(primal)
    bare_tangents.defjvp(tangent, symbolic_zeros=True)
    derivative = derivatives.declared_derivative(Protected(), primal, forward, backward, symbolic_zeros=True)
    bare_derivative = jax.custom_vjp(primal)
    bare_derivative.defvjp(forward, backward, symbolic_zeros=True)
    x, y = jnp.linspace(0.1, 1.0, 8), jnp.full(8, 2.0)
    for declared, bare, unreached in ((tangents, bare_tangents, SymbolicZero),
                                      (derivative, bare_derivative, False)):
        seen.clear()
        np.testing.assert_array_equal(np.asarray(jax.grad(lambda x, f=declared: jnp.sum(f(x, y)))(x)),
                                      np.asarray(jax.grad(lambda x, f=bare: jnp.sum(f(x, y)))(x)))
        assert seen and set(seen) == {unreached}, seen
        # Its rules keep their kind: the gradient computes no sine outside a span of it.
        gradient = jax.make_jaxpr(jax.grad(lambda x, f=declared: jnp.sum(f(x, y))))(x)
        assert "sin" not in unmarked(gradient.jaxpr), gradient


def unmarked(jaxpr) -> list[str]:
    """The primitives `jaxpr` runs outside every marked call, its other bodies' included."""
    names = []
    for equation in jaxpr.eqns:
        if equation.primitive is marked_call_p:
            continue
        names.append(equation.primitive.name)
        for value in equation.params.values():
            for inner in (value if isinstance(value, (tuple, list)) else (value,)):
                if isinstance(inner, jax.extend.core.Jaxpr):
                    names += unmarked(inner)
    return names


def test_a_declared_rule_may_return_a_known_zero_tangent():
    """A declared tangent rule with `symbolic_zeros` may return a known zero (`SymbolicZero`) as an output's
    tangent, as a bare `jax.custom_jvp`'s may: differentiated along an operand it ignores, its tangent is
    zero, as the bare rule's is."""
    def tangent(primals, tangents):
        return primals[0], tangents[0]

    declared = derivatives.declared_tangent(Protected(), lambda x, y: x, tangent, symbolic_zeros=True)
    bare = jax.custom_jvp(lambda x, y: x)
    bare.defjvp(tangent, symbolic_zeros=True)
    x, y = jnp.linspace(0.1, 1.0, 8), jnp.full(8, 2.0)
    for rule in (declared, bare):
        _, along = jax.jit(lambda y, f=rule: jax.jvp(lambda y: f(x, y), (y,), (jnp.ones_like(y),)))(y)
        np.testing.assert_array_equal(np.asarray(along), np.zeros(8))


def test_a_surrogate_is_one_surrogate_call():
    """A surrogate (`surrogate`: a value and its declared derivative rule) is one marked call of its kind,
    which reverse mode keeps, its derivative its rule's."""
    rounded = derivatives.surrogate(lambda x: jnp.round(x), lambda primals, tangents, value: tangents[0])
    x = jnp.linspace(0.1, 4.0, 8)
    assert kinds_of(jax.make_jaxpr(rounded)(x)) == [Surrogate()]
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda x: jnp.sum(rounded(x)))(x)), np.ones(8))


def test_a_declared_derivatives_nondifferentiable_array_is_bound_not_traced():
    """An array at a declared derivative's nondifferentiable position (`nondiff_argnums`, a table the rule
    reads as a constant) reaches the rule as the bare `jax.custom_vjp` takes it, concrete, eagerly and under a
    jit, not as a traced operand of the marked call, which the custom derivative would refuse."""
    def primal(table, x):
        return jnp.sin(x) * table[0]

    def forward(table, x):
        return primal(table, x), x

    def backward(table, x, cotangent):
        return (3.0 * cotangent * jnp.cos(x) * table[0],)

    declared = derivatives.declared_derivative(KernelGlue(), primal, forward, backward, nondiff_argnums=(0,))
    bare = jax.custom_vjp(primal, nondiff_argnums=(0,))
    bare.defvjp(forward, backward)
    table, x = jnp.asarray([2.0]), jnp.linspace(0.1, 1.0, 8)

    def gradient(function, run):
        return run(jax.grad(lambda x: jnp.sum(function(table, x))))(x)

    for run in (lambda f: f, jax.jit):
        np.testing.assert_array_equal(np.asarray(gradient(declared, run)), np.asarray(gradient(bare, run)))


def test_a_marked_span_called_again_in_one_state_is_traced_once(monkeypatch):
    """A marked span called again in the same state, its function's reads, its arguments' types and the
    context a trace reads alike, is the program its first call traced (`derivatives._SPANS`), never traced
    again: precision's error-free transforms were traced afresh at every call, 7.5 percent of a padded
    merge's grad of grad (row 10762). Each trace of a span counted as JAX makes it."""
    traced = []
    making = jax.make_jaxpr

    def counted(fn, *args, **kwargs):
        # A marked call's own trace of its span, among whatever else the trace makes.
        traced.append(getattr(fn, "__name__", None))
        return making(fn, *args, **kwargs)

    span = marked(Protected(), body)
    x, y = jnp.ones(3), jnp.full(3, 2.0)
    monkeypatch.setattr(jax, "make_jaxpr", counted)
    first = jax.jit(lambda x, y: span(x, y)).trace(x, y)
    second = jax.jit(lambda x, y: span(x, y) + 0.0).trace(x, y)
    assert traced.count("span") == 1
    assert kinds_of(first.jaxpr) == kinds_of(second.jaxpr) == [Protected()]


def test_a_marked_span_with_an_effect_is_traced_at_every_call_and_runs_it(monkeypatch):
    """A marked span whose program has an effect (a host callback), which no call may take from another's
    trace (`derivatives._holdable`), is traced at each call, as one no key holds (`_span_key` of None), in the
    call's own collections, and runs its effect with each call's values: its held None never reads as a
    program to take (Train 1205's review F1, its facts held by `generation.held_trace`)."""
    traced, seen = [], []
    making = jax.make_jaxpr

    def counted(fn, *args, **kwargs):
        traced.append(getattr(fn, "__name__", None))
        return making(fn, *args, **kwargs)

    def logged(x, y):
        jax.debug.callback(lambda total: seen.append(float(total)), x.sum())
        return x * y

    span = marked(Protected(), logged)
    monkeypatch.setattr(jax, "make_jaxpr", counted)
    first = jax.jit(lambda x, y: span(x, y))(jnp.ones(3), jnp.full(3, 2.0))
    second = jax.jit(lambda x, y: span(x, y) + 0.0)(jnp.full(3, 3.0), jnp.full(3, 2.0))
    jax.effects_barrier()
    assert traced.count("span") == 2, traced
    assert seen == [3.0, 9.0]
    np.testing.assert_array_equal(np.asarray(first), np.full(3, 2.0, np.float32))
    np.testing.assert_array_equal(np.asarray(second), np.full(3, 6.0, np.float32))


SCALE = 2.0
"""A module global a marked span reads (`test_a_marked_span_reads_the_state_its_function_reads_now`)."""


def test_a_marked_span_reads_the_state_its_function_reads_now(monkeypatch):
    """A marked span's program is its function's in the state it reads now: a module global rebound, a
    closure's dict changed, between two calls, the second computes with the new values, never a program
    the first traced (row 10762's reviews: a key that missed a function's reads served the old one)."""
    held = {"shift": 1.0}

    def scaled(x):
        return x * SCALE + held["shift"]

    span = marked(Protected(), scaled)
    x = jnp.ones(3)
    first = jax.jit(lambda x: span(x))(x)
    monkeypatch.setattr(__import__(__name__), "SCALE", 3.0)
    held["shift"] = 5.0
    second = jax.jit(lambda x: span(x) + 0.0)(x)
    np.testing.assert_array_equal(np.asarray(first), np.full(3, 3.0, np.float32))
    np.testing.assert_array_equal(np.asarray(second), np.full(3, 8.0, np.float32))


def _span_traces(monkeypatch) -> list:
    """The marked spans' traces JAX makes from here on (`derivatives.marked`'s `span`), each counted."""
    traced = []
    making = jax.make_jaxpr

    def counted(fn, *args, **kwargs):
        traced.append(getattr(fn, "__name__", None))
        return making(fn, *args, **kwargs)

    monkeypatch.setattr(jax, "make_jaxpr", counted)
    return traced


def test_an_error_free_transform_called_again_in_one_state_is_traced_once(monkeypatch):
    """An error-free transform (`precision.two_sum`, one `bound_span` of `Protected`) called again in one
    state is the program its first call traced: its span closes over stated values alone, never the call's
    bound arguments, which no statement holds (row 11003: every bound span, the error-free transforms row
    10762 names among them, was traced at each call)."""
    from dew.nn.precision import two_sum

    x, y = jnp.ones(3), jnp.full(3, 2.0)
    traced = _span_traces(monkeypatch)
    jax.jit(lambda x, y: two_sum(x, y)).trace(x, y)
    jax.jit(lambda x, y: jax.tree.map(lambda z: z + 0.0, two_sum(x, y))).trace(x, y)
    assert traced.count("span") == 1


def test_a_held_span_reports_its_routes_wherever_it_is_taken():
    """A marked span whose body takes a route (a kernel's choice) reports it in every trace that takes the
    span, its program held or not: a held span reported nothing again, and a kernel report and the tuner
    missed its route and launch (row 11002)."""
    from dew.nn.kernels import generation

    def routed(x):
        generation.route("rms_norm", kernel="xla")
        return x * 2.0

    span = marked(Protected(), routed)
    found = []
    for program in (lambda x: span(x), lambda x: span(x) + 0.0):
        with generation.routes() as routes:
            jax.jit(program).trace(jnp.ones(3))
        found.append([route.choice for route in routes])
    assert found == [["rms_norm"], ["rms_norm"]]



def test_a_marked_span_closing_over_its_enclosing_traces_value_is_traced_as_it_is():
    """A marked span whose function closes over a value of the trace enclosing it (an outer jit's argument)
    computes with that value, traced at its call: no statement holds a trace's value, which raised as its
    placement was read (row 11002's review, F3: `ConcretizationTypeError` where an unstated span was
    meant)."""
    def scaled(x, y):
        return marked(Protected(), lambda v: v * y)(x)

    out = jax.jit(scaled)(jnp.ones(3), jnp.full(3, 2.0))
    np.testing.assert_array_equal(np.asarray(out), np.full(3, 2.0, np.float32))


def test_a_bound_flax_modules_statement_runs_none_of_its_setup():
    """A Flax module bound to its variables (a layer within its model's `init` or `apply`) has no statement:
    its identity is its variables' as well as its fields, and stating it ran its `setup`, which a second run
    of found its submodules' names in use (train 1169: a held span's statement over a model's layer raised
    `NameInUseError` in 49 cells). Nothing of it runs."""
    import flax.linen as nn

    from dew._stated import Unestablished, stated

    made = []

    class Made(nn.Module):
        def setup(self):
            made.append(None)
            self.dense = nn.Dense(2)

        def __call__(self, x):
            return self.dense(x)

    variables = Made().init(jax.random.key(0), jnp.ones((1, 3)))
    made.clear()
    bound = Made().bind(variables)
    with pytest.raises(Unestablished):
        stated(bound, "a bound module")
    assert made == []


def test_a_local_functions_bound_span_called_again_in_one_state_is_traced_once(monkeypatch):
    """A bound span of a local function (a kernel's glue: `declared_tangent` of a closure) called again in one
    state is the program its first call traced, as a module function's is: its statement holds it (row
    11650: local functions were traced at each call, an exception for a recursion its launch's digest made,
    not a collision of statements)."""
    from dew.nn.derivatives import declared_tangent

    def glued(scale):
        def applied(y):
            return y * scale

        def tangent(primals, tangents):
            return primals[0] * scale, tangents[0] * scale

        return declared_tangent(Protected(), applied, tangent)

    x = jnp.ones(3)
    traced = _span_traces(monkeypatch)
    jax.jit(lambda x: glued(2.0)(x)).trace(x)
    jax.jit(lambda x: glued(2.0)(x) + 0.0).trace(x)
    assert traced.count("span") == 1



class Doubled(functools.partial):
    """A partial doubling what it calls (`test_a_partial_subclass_overriding_its_call_is_stated_apart_from_a_
    plain_partial`)."""

    def __call__(self, *args, **kwargs):
        return 2 * super().__call__(*args, **kwargs)


def test_a_partial_subclass_overriding_its_call_is_stated_apart_from_a_plain_partial():
    """A `functools.partial` subclass whose `__call__` is its own (one doubling what it calls) is stated apart
    from a plain partial of the same function and arguments, its class among its statement: two marked spans
    over them are two programs (train 1174's return: test_cpu_accumulation's doubled contraction took the
    plain one's held program, 1x where 2x was asked)."""
    from dew._stated import stated

    plain = functools.partial(jnp.multiply, 3.0)
    assert stated(Doubled(jnp.multiply, 3.0), "doubled") != stated(plain, "plain")


class _Echo(nn.Module):
    """A module of the user's own handing back its prediction
    (`test_a_function_reading_ones_own_flax_module_is_stated_and_told_apart`)."""

    def __call__(self, prediction):
        return prediction


class _Doubling(nn.Module):
    """A module of the user's own doubling its prediction
    (`test_a_function_reading_ones_own_flax_module_is_stated_and_told_apart`)."""

    def __call__(self, prediction):
        return 2 * prediction


def _echoed(x):
    return _Echo().apply({}, x)


def _doubled(x):
    return _Doubling().apply({}, x)


def test_a_function_reading_ones_own_flax_module_is_stated_and_told_apart():
    """A function reading a Flax module class of the user's own is stated (its class's `parent` descriptor
    and that field's default, Flax's unspecified-parent sentinel, instances holding nothing of their own),
    and two reading modules that compute otherwise are two statements: an APG arm reading the guidance's own
    model class had no statement, and a GPU tier part's prepare left its cell unprepared."""
    from dew._stated import stated

    assert stated(_echoed, "echoed") != stated(_doubled, "doubled")


class _Held:
    """A plain object its module holds by name, which holds a value of its own
    (`test_a_module_held_instance_holding_a_value_has_no_statement`)."""


_HELD = _Held()
_HELD.value = 1


class _Rate(float):
    """A Python subclass of a C value type, its value held in the C base, its `__dict__` empty
    (`test_values_a_c_base_holds_apart_never_share_a_statement`)."""


class _Reference(weakref.ref):
    """A Python subclass of a C reference, its referent held in the C base
    (`test_values_a_c_base_holds_apart_never_share_a_statement`)."""


class _Day(datetime.date):
    """A Python subclass of a date (`test_values_a_c_base_holds_apart_never_share_a_statement`)."""


class _Bytes(bytes):
    """A Python subclass of a C value of variable size
    (`test_values_a_c_base_holds_apart_never_share_a_statement`)."""


class _When(datetime.datetime):
    """A Python subclass of a time (`test_values_a_c_base_holds_apart_never_share_a_statement`)."""


class _Failure(ValueError):
    """A Python subclass of an exception (`test_values_a_c_base_holds_apart_never_share_a_statement`)."""


class _Referent:
    """An object a reference reads (`test_values_a_c_base_holds_apart_never_share_a_statement`)."""


_REFERENTS = (_Referent(), _Referent())


def test_values_a_c_base_holds_apart_never_share_a_statement():
    """Instances of a C type, or of a Python subclass of one, keep their value in the C layout, which no
    `__dict__` shows, so none is stated by its type (rows 11772 and 11777: a kept program or an exact resume
    read a reference to one object as a reference to another, and a buffer of b"a" as one of b"b"): a
    float's subclass is stated by its value, two of them apart, and a reference's, a date's, a time's, a
    byte string's, an exception's and an in-memory stream's, which no form states, have no statement, on
    every Python the suite runs (3.14 states a C layout's subclasses as 3.12 does)."""
    from dew._stated import Unestablished, stated

    assert stated(_Rate(0.1), "a rate") != stated(_Rate(0.2), "a rate")
    for value in (_Reference(_REFERENTS[0]), _Day(2024, 1, 1), _When(2024, 1, 1), _Bytes(b"a"), _Failure("a"),
                  io.BytesIO(b"a"), io.StringIO("a")):
        with pytest.raises(Unestablished):
            stated(value, "a value its C base holds")


def test_a_module_held_instance_holding_a_value_has_no_statement():
    """An instance holding a value of its own (`_HELD.value`) has no statement: stated by its type alone, a
    change of its value would read as the same, so only one holding nothing is stated so."""
    from dew._stated import Unestablished, stated

    with pytest.raises(Unestablished):
        stated(_HELD, "a held instance")
