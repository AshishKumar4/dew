"""An operation is known by its object (dew.nn.programs): what JAX states of its own primitives each analysis
reads, and what each analysis does with a primitive it does not know, one of another library's or a user's
under any name."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.core import ShapedArray
from jax.extend import core as jcore
from jax.extend.core import primitives
from jax.interpreters import mlir
from test_algebra import rows

from dew import algebra
from dew.nn import flops, programs
from dew.nn.kernels.templates import attention, attention_mosaic_program, layout
from dew.nn.kernels.templates.hooks import Hook


def named(name: str, function, operands: int = 1) -> jcore.Primitive:
    """A primitive of a user's under `name`, computing `function` over `operands` arrays of one shape."""
    made = jcore.Primitive(name)
    made.def_impl(function)
    made.def_abstract_eval(lambda first, *rest: first)
    mlir.register_lowering(made, mlir.lower_fun(function, multiple_results=False))
    return made


def equation_of(primitive: jcore.Primitive, function, *arguments) -> jcore.JaxprEqn:
    """The one equation of `primitive` in `function`'s program over `arguments`."""
    [equation] = [equation for equation in jax.make_jaxpr(function)(*arguments).jaxpr.eqns
                  if equation.primitive is primitive]
    return equation


def test_a_call_is_read_through_its_program_and_a_pure_one_alone_runs_in_its_caller():
    """A jit's call and a checkpoint's are their programs on their operands (`called`); the jit's alone
    does nothing else (`purely_called`), a checkpoint keeping its policy as a call. A scan's body runs a step
    and a conditional's branches where they are taken, so neither is a call; each is read by its own
    reader, which refuses another operation's equation."""
    x = jnp.ones((4,), jnp.float32)
    jitted = equation_of(primitives.jit_p, lambda x: jax.jit(jnp.sin)(x), x)
    remat = equation_of(primitives.remat_p, lambda x: jax.checkpoint(jnp.sin)(x), x)
    scanned = equation_of(jax.lax.scan_p, lambda x: jax.lax.scan(lambda c, v: (c + v, c), 0.0, x)[1], x)
    chosen = equation_of(jax.lax.cond_p, lambda i, x: jax.lax.switch(i, (jnp.sin, jnp.cos), x),
                         jnp.int32(0), x)
    for call in (jitted, remat):
        assert [eqn.primitive for eqn in programs.called(call).eqns] == [jax.lax.sin_p]
    assert programs.purely_called(jitted) is programs.called(jitted)
    assert programs.purely_called(remat) is None
    for other in (scanned, chosen):
        assert programs.called(other) is None
    assert jax.lax.add_p in [eqn.primitive for eqn in programs.scan_body(scanned).eqns]
    assert len(programs.branches(chosen)) == 2
    with pytest.raises(TypeError):
        programs.scan_body(chosen)


def test_the_algebra_reads_another_librarys_operation_as_a_leaf_whatever_its_name():
    """A user's primitive named `dot_general`, bound with a product's parameters, that squares its second
    operand is no operation of the language: the sum of two of its results over bounded operands stays as
    written, where reading it by its name factored it over the sum of its operands, another function (row
    5341)."""
    def squared_product(a, x, **_):
        return a @ (x * x)

    squaring = jcore.Primitive("dot_general")
    squaring.def_impl(squared_product)
    squaring.def_abstract_eval(lambda a, x, **_: ShapedArray((a.shape[0], x.shape[1]), a.dtype))
    mlir.register_lowering(squaring, mlir.lower_fun(squared_product, multiple_results=False))
    product = {"dimension_numbers": (((1,), (0,)), ((), ())), "precision": None,
               "preferred_element_type": None, "out_sharding": None}

    def summed(x, y, a):
        # Bounded operands, which the rules' range facts need before they factor a sum (`test_algebra`'s).
        x, y, a = jnp.tanh(x), jnp.tanh(y), jnp.tanh(a)
        return squaring.bind(a, x, **product) + squaring.bind(a, y, **product)

    x, y, a = rows(2, (64, 32), (64, 32), (8, 64))
    np.testing.assert_array_equal(np.asarray(algebra.rewritten(summed)(x, y, a)), np.asarray(summed(x, y, a)))


def test_the_flop_count_reads_a_product_by_its_object():
    """A user's primitive named `dot_general` is not JAX's product: the traced count refuses it as an
    operation it has no count for (`MissingFlopsCount`), where reading it by its name raised KeyError on the
    product's parameters it does not hold; JAX's own product is counted, two FLOPs a multiply-add."""
    doubled = named("dot_general", lambda a: a + a)
    x = jnp.ones((4, 8), jnp.float32)
    with pytest.raises(flops.MissingFlopsCount):
        flops.traced_flops(lambda x: doubled.bind(x) @ x.T, x)
    assert flops.traced_flops(lambda x: x @ x.T, x) == 2 * 4 * 4 * 8


def test_a_tile_layout_refuses_an_operation_it_does_not_know():
    """A function of a tile through a user's primitive is not held transposed: the layout refuses an
    operation it has no rule for, whatever JAX operation its name spells."""
    reversed_rows = named("exp", lambda tile: tile[::-1])
    with pytest.raises(layout.Untransposable):
        layout.transposed(reversed_rows.bind)(jnp.ones((3, 5), jnp.float32))


def test_a_tile_layout_takes_a_placement_inside_a_nested_call_as_the_identity():
    """A function of a tile whose nested call places a value on a mesh (a sharding constraint, as a GPU's
    trace of a plain program's attention holds a reshard in its score hook: rows 9570 and 9571) is held
    transposed as the function without it: the placement binds nothing, the tile being placed already, so
    the transposed program holds none for a kernel's lowering to meet; and as a score hook the Mosaic
    backward takes a reshard (`transposable`), which its queries-first walks, running the hook as written,
    lower too, and refuses a sharding constraint, which no Mosaic rule lowers (head review)."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("rows",))
    placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def scored(tile):
        return jax.jit(lambda t: jax.lax.with_sharding_constraint(t * 2.0, placed))(tile) + tile

    tile = jnp.asarray(np.random.default_rng(9570).standard_normal((3, 5)), jnp.float32)
    held = layout.transposed(scored)
    np.testing.assert_array_equal(np.asarray(held(tile.T)), np.asarray(scored(tile).T))
    traced = jax.make_jaxpr(held)(tile.T)
    assert not any(eqn.primitive in (jax.lax.sharding_constraint_p, primitives.reshard_p)
                   for eqn in traced.jaxpr.eqns)
    # As a score hook the Mosaic backward takes a reshard, which a tile run as written lowers too, and
    # refuses a sharding constraint, which none does (head review).
    explicit = jax.make_mesh((1,), ("rows",), axis_types=(jax.sharding.AxisType.Explicit,))

    def resharded(logits, q_pos, k_pos):
        return jax.jit(lambda t: jax.sharding.reshard(
            t * 2.0, jax.sharding.NamedSharding(explicit, jax.sharding.PartitionSpec())))(logits)

    with jax.set_mesh(explicit):
        assert attention_mosaic_program.transposable(attention.Hooks(score=Hook(resharded)))
    assert not attention_mosaic_program.transposable(
        attention.Hooks(score=Hook(lambda logits, q_pos, k_pos: scored(logits))))

