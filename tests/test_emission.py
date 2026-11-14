"""A launch's work counted from its kernel's emission (`templates.emission`, row 5895): a step of its walk
counted from its body, as each warpgroup runs it."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import derivatives
from dew.nn.kernels.templates import emission
from dew.nn.kernels.templates.cost import Work

_STEP = derivatives.marked(derivatives.WalkStep(), lambda index: index)
"""A walk step's mark, as `mosaic.stepped` makes it."""

_ROW = jax.ShapeDtypeStruct((128,), jnp.float32)


def _operations(work: Work) -> float:
    return work.operations


def _walked(work):
    """A walk of four steps, each marked and doing `work` to the row."""
    def step(row, index):
        _STEP(index)
        return work(row), None

    return lambda row: lax.scan(step, row, jnp.arange(4))[0]


def _emitted(fn, *args) -> emission.Emitted:
    """`fn` counted as each of two warpgroups along `wg` runs it."""
    return emission.emitted(jax.make_jaxpr(fn, axis_env=[("wg", 2)])(*args), 2, "wg")


def test_a_step_counts_each_trip_of_a_loop_inside_it():
    """A walk step holding a loop of four trips counts its body four times."""
    def walk(row):
        def step(row, index):
            _STEP(index)
            return lax.scan(lambda row, _: (row + 1, None), row, length=4)[0], None

        return lax.scan(step, row, jnp.arange(8))[0]

    assert [role.step(_operations) for role in _emitted(walk, _ROW).roles] == [4 * 128] * 2


def _one(row, index):
    _STEP(index)
    return row + 1


def _two(row, index):
    return _one(_one(row, index), index)


def test_a_step_divides_by_the_steps_its_warpgroups_branch_takes():
    """A trip whose warpgroup's branch takes one step for one warpgroup and two for the other counts each
    one step's work: a step's work over the marks the warpgroup runs, not the most any branch holds."""
    def walk(row):
        def step(row, index):
            return lax.cond(lax.axis_index("wg") == 0, _one, _two, row, index), None

        return lax.scan(step, row, jnp.arange(8))[0]

    assert [role.step(_operations) for role in _emitted(walk, _ROW).roles] == [128, 128]


def test_a_scoped_bodys_results_resolve_the_roles_after_it():
    """A warpgroup's index a scoped body returns (`pl.run_scoped`'s) resolves the role's branch after it."""
    def walk(row):
        def step(row, index):
            role = pl.run_scoped(lambda scratch: lax.axis_index("wg"), plgpu.SMEM((8,), jnp.float32))
            return lax.cond(role == 0, _one, _two, row, index), None

        return lax.scan(step, row, jnp.arange(8))[0]

    assert [role.step(_operations) for role in _emitted(walk, _ROW).roles] == [128, 128]


def test_an_unknown_branch_counts_the_most_of_each_unit_any_branch_issues():
    """A conditional its data decides counts its conversion-only branch's conversions, though its other
    branch issues nothing."""
    def converted(row, index):
        return lax.switch(index, [lambda row: row, lambda row: row.astype(jnp.bfloat16).astype(jnp.float32)],
                          row)

    index = jax.ShapeDtypeStruct((), jnp.int32)
    assert _emitted(converted, _ROW, index).roles[0].fixed.conversions == 2 * 128


def test_an_unknown_branch_between_walks_steps_the_costliest():
    """A conditional its data decides between a walk of two segments of one operation a row a step and a
    walk of three a step counts a step of three."""
    once, thrice = _walked(lambda row: row + 1), _walked(lambda row: row + 1 + 1 + 1)

    def either(row, index):
        return lax.switch(index, [lambda row: once(once(row)), thrice], row)

    index = jax.ShapeDtypeStruct((), jnp.int32)
    assert _emitted(either, _ROW, index).roles[0].step(_operations) == 3 * 128


def test_a_peeled_walks_step_is_its_parts_mean():
    """A walk reading ahead, its first and last steps peeled from its loop (`RingFamily._walked`): its first
    step two operations a row, its loop's one, its last none, counts one a step, under a guard its data
    decides too (a program's active tile's)."""
    first, loop, last = (_walked(lambda row: row + 1 + 1), _walked(lambda row: row + 1),
                         _walked(lambda row: row))

    def peeled(row):
        return last(loop(first(row)))

    def guarded(row, active):
        return lax.cond(active, peeled, lambda row: row, row)

    active = jax.ShapeDtypeStruct((), jnp.bool_)
    assert [role.step(_operations) for role in _emitted(peeled, _ROW).roles] == [128, 128]
    assert [role.step(_operations) for role in _emitted(guarded, _ROW, active).roles] == [128, 128]
