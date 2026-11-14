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


def _exponentials(work: Work) -> float:
    return work.exponentials


_TRIPS = jax.ShapeDtypeStruct((), jnp.int32)
"""A walk's trips, which its data decides (a while loop's)."""


def _walked(work):
    """A walk its data decides the trips of, each step marked and doing `work` to the row."""
    def step(index, row):
        _STEP(index)
        return work(row)

    return lambda row, trips: lax.fori_loop(0, trips, step, row)


def _emitted(fn, *args) -> emission.Emitted:
    """`fn` counted as each of two warpgroups along `wg` runs it."""
    return emission.emitted(jax.make_jaxpr(fn, axis_env=[("wg", 2)])(*args), 2, "wg")


def test_a_step_counts_each_trip_of_a_loop_inside_it():
    """A walk step holding a loop of four trips counts its body four times."""
    walk = _walked(lambda row: lax.scan(lambda row, _: (jnp.exp(row), None), row, length=4)[0])
    assert [role.step(_exponentials) for role in _emitted(walk, _ROW, _TRIPS).roles] == [4 * 128] * 2


def test_a_walk_of_stated_trips_counts_each_trip_where_it_stands():
    """A walk whose trips its jaxpr states (a scan's) counts its body each trip, its work the program's own:
    no step a launch's statement multiplies."""
    def walk(row):
        def step(row, index):
            _STEP(index)
            return row + 1, None

        return lax.scan(step, row, jnp.arange(4))[0]

    counted = _emitted(walk, _ROW)
    assert [(role.fixed.operations, role.step(_operations)) for role in counted.roles] == [(512, None)] * 2


def _one(row, index):
    _STEP(index)
    return jnp.exp(row)


def _two(row, index):
    return _one(_one(row, index), index)


def test_a_step_divides_by_the_steps_its_warpgroups_branch_takes():
    """A trip whose warpgroup's branch takes one step for one warpgroup and two for the other counts each
    one step's work: a step's work over the marks the warpgroup runs, not the most any branch holds."""
    def walk(row, trips):
        def step(index, row):
            return lax.cond(lax.axis_index("wg") == 0, _one, _two, row, index)

        return lax.fori_loop(0, trips, step, row)

    assert [role.step(_exponentials) for role in _emitted(walk, _ROW, _TRIPS).roles] == [128, 128]


def test_a_scoped_bodys_results_resolve_the_roles_after_it():
    """A warpgroup's index a scoped body returns (`pl.run_scoped`'s) resolves the role's branch after it."""
    def walk(row, trips):
        def step(index, row):
            role = pl.run_scoped(lambda scratch: lax.axis_index("wg"), plgpu.SMEM((8,), jnp.float32))
            return lax.cond(role == 0, _one, _two, row, index)

        return lax.fori_loop(0, trips, step, row)

    assert [role.step(_exponentials) for role in _emitted(walk, _ROW, _TRIPS).roles] == [128, 128]


def test_an_unknown_branch_counts_the_most_of_each_unit_any_branch_issues():
    """A conditional its data decides counts its conversion-only branch's conversions, though its other
    branch issues nothing."""
    def converted(row, index):
        return lax.switch(index, [lambda row: row, lambda row: row.astype(jnp.bfloat16).astype(jnp.float32)],
                          row)

    index = jax.ShapeDtypeStruct((), jnp.int32)
    assert _emitted(converted, _ROW, index).roles[0].fixed.conversions == 2 * 128


def test_an_unknown_branch_between_walks_steps_the_costliest():
    """A conditional its data decides between a walk of two segments of one exponential a row a step and a
    walk of three a step counts a step of three."""
    once, thrice = _walked(jnp.exp), _walked(lambda row: jnp.exp(jnp.exp(jnp.exp(row))))

    def either(row, index, trips):
        return lax.switch(index, [lambda row: once(once(row, trips), trips), lambda row: thrice(row, trips)],
                          row)

    index = jax.ShapeDtypeStruct((), jnp.int32)
    assert _emitted(either, _ROW, index, _TRIPS).roles[0].step(_exponentials) == 3 * 128


def test_a_split_walks_step_is_its_parts_mean():
    """A walk split into parts, its first step's issuing one ahead and its last none (a pipelined walk's
    opening, steady and closing loops): its first step two exponentials a row, its loop's one, its last none,
    counts one a step, under a guard its data decides too (a program's active tile's)."""
    first, loop, last = _walked(lambda row: jnp.exp(jnp.exp(row))), _walked(jnp.exp), _walked(lambda row: row)

    def split(row, trips):
        return last(loop(first(row, trips), trips), trips)

    def guarded(row, trips, active):
        return lax.cond(active, split, lambda row, trips: row, row, trips)

    active = jax.ShapeDtypeStruct((), jnp.bool_)
    assert [role.step(_exponentials) for role in _emitted(split, _ROW, _TRIPS).roles] == [128, 128]
    assert [role.step(_exponentials) for role in _emitted(guarded, _ROW, _TRIPS, active).roles] == [128, 128]


def test_a_walks_guards_on_its_steps_place_count_at_their_trips():
    """A walk over a range whose first step takes its own exponential and every step but its last the next
    one's (`index == lo`, `index + 1 < hi`, as a walk reading ahead guards them) counts one a row a step and
    none apart: each guard read at the trips it decides, not taken every step."""
    def walk(row, lo, hi):
        def step(index, row):
            _STEP(index)
            row = lax.cond(index == lo, jnp.exp, lambda row: row, row)
            return lax.cond(index + 1 < hi, jnp.exp, lambda row: row, row)

        return lax.fori_loop(lo, hi, step, row)

    bound = jax.ShapeDtypeStruct((), jnp.int32)
    counted = _emitted(walk, _ROW, bound, bound)
    assert [(role.step(_exponentials), role.fixed.exponentials) for role in counted.roles] == [(128, 0)] * 2


def test_a_walks_guards_on_its_steps_parity_count_one_branch_a_step():
    """A walk whose step takes one of two exponentials by its parity (`index % 2`, the accumulator of two a
    walk reading ahead issues into) counts one a row a step, not both guards' each step."""
    def walk(row, lo, hi):
        def step(index, row):
            _STEP(index)
            row = lax.cond(index % 2 == 0, jnp.exp, lambda row: row, row)
            return lax.cond(index % 2 == 1, jnp.exp, lambda row: row, row)

        return lax.fori_loop(lo, hi, step, row)

    bound = jax.ShapeDtypeStruct((), jnp.int32)
    counted = _emitted(walk, _ROW, bound, bound)
    assert [(role.step(_exponentials), role.fixed.exponentials) for role in counted.roles] == [(128, 0)] * 2
