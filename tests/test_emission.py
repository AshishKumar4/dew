"""A launch's work counted from its kernel's emission (`templates.emission`, row 5895): a step of its walk
counted from its body, as each warpgroup runs it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import derivatives, programs
from dew.nn.kernels.templates import emission
from dew.nn.kernels.templates.cost import Stated, Work

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


@pytest.mark.parametrize("operation", [jnp.exp, jnp.sqrt, lax.rsqrt, jnp.tanh], ids=lambda f: f.__name__)
def test_a_square_root_counts_on_the_special_function_units_as_an_exponential_does(operation):
    """A row's sqrt or rsqrt counts one special-function operation an element and no operation of the CUDA
    cores, as an exp or a tanh does: each an fp32 MUFU on sm80, sm90 and sm100 (`emission.SPECIAL_FUNCTIONS`,
    their SASS), where a square root counted on the CUDA cores."""
    role = _emitted(operation, _ROW).roles[0]
    assert (role.fixed.exponentials, role.fixed.operations) == (128, 0), role.fixed


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


def _converted(row, index):
    _STEP(index)
    return row.astype(jnp.bfloat16).astype(jnp.float32)


def test_a_switch_on_the_warpgroups_role_counts_each_warpgroups_branch_alone():
    """A switch whose index the warpgroup's role decides, as a schedule's exclusive roles take theirs
    (`handoffs.Schedule.run`: the index from the warpgroup through comparisons, a sum and a select, which
    `lax.switch` clamps to its branches): each warpgroup counts its own branch, the producer none of a
    consumer's work. Unfolded, every warpgroup counted every branch's (ET's sm90 forward: a consumer's
    products on its producer too)."""
    def walk(row, trips):
        def walked(step):
            return lambda row: lax.fori_loop(0, trips, lambda index, row: step(row, index), row)

        wg = lax.axis_index("wg")
        role = sum((wg >= bound).astype(jnp.int32) for bound in (1,))
        return lax.switch(jnp.where(wg >= 2, 0, role), [walked(_one), walked(_converted)], row)

    roles = _emitted(walk, _ROW, _TRIPS).roles
    assert [(role.step(_exponentials), role.step(lambda work: work.conversions)) for role in roles] == [
        (128, 0), (0, 2 * 128)]


@pytest.mark.parametrize("predicate, exponentials", [
    # An integer quotient truncates toward zero: -1 over 2 is 0 at every warpgroup.
    (lambda wg: lax.div(1 - lax.clamp(2, wg, 3), 2) == 0, [128, 128]),
    # By a negative divisor too: 7 over -2 is -3 at the first warpgroup, 8 over -2 -4 at the second.
    (lambda wg: lax.div(7 + wg, -2) == -3, [128, 0]),
    # A remainder takes its dividend's sign: -7 rem 2 is -1, -6 rem 2 is 0.
    (lambda wg: lax.rem(wg - 7, 2) == -1, [128, 0]),
    # int8 wraps 128 to -128 at the second warpgroup.
    (lambda wg: (wg + 127).astype(jnp.int8) < 0, [0, 128]),
    # A logical shift of -1 by 31 is 1; of 0, 0.
    (lambda wg: lax.shift_right_logical(wg - 1, 31) == 1, [128, 0]),
    # A shift past the width is 0, an arithmetic one's of a negative value -1.
    (lambda wg: lax.shift_left(wg + 1, 32) == 0, [128, 128]),
    (lambda wg: lax.shift_right_arithmetic(wg - 1, 40) == -1, [128, 0]),
    # A float predicate is no warpgroup arithmetic the fold computes: both branches at every warpgroup.
    (lambda wg: wg.astype(jnp.float32) == jnp.float32(0), [128, 128]),
], ids=["truncated-quotient", "negative-divisor", "remainder-sign", "int8-wrap", "logical-shift",
        "shift-past-width", "arithmetic-shift-past-width", "float-unknown"])
def test_scalar_arithmetic_on_the_warpgroup_folds_as_the_program_computes_it(predicate, exponentials):
    """A conditional on integer arithmetic over the warpgroup's index (a quotient by either sign, a
    remainder, a narrower integer's wrap, shifts within and past the width) counts the branch each
    warpgroup's program takes, folded as `lax` computes it at its own dtype, where Python's arithmetic floored
    an integer quotient (base review 951e3f9e7 F3); a float predicate, which no warpgroup's index arithmetic
    is, counts both branches at every warpgroup, never one dropped: the fold computes integers and booleans
    alone, where IEEE semantics' edges (subnormals, signed zeros, float64, a complex value) were where its
    defects lived (base reviews 127d2d95c to edd99fc1d)."""
    def walk(row):
        wg = lax.axis_index("wg").astype(jnp.int32)
        return lax.cond(predicate(wg), jnp.exp, lambda row: row, row)

    assert [role.fixed.exponentials for role in _emitted(walk, _ROW).roles] == exponentials


@pytest.mark.parametrize("wide, predicate, exponentials", [
    # The second warpgroup's int8 16 times itself at int32 is 256, which int8 wraps to 0.
    (False, lambda wg: lax.mul(*[(wg * 16).astype(jnp.int8)] * 2, out_dtype=jnp.int32) == 256, 128),
    # The first warpgroup's 2**63 + 2048 times 3 at uint64 wraps to 2**63 + 6144, under x64.
    (True, lambda wg: lax.mul(wg.astype(jnp.uint64) + jnp.uint64(2 ** 63 + 2048), jnp.uint64(3))
     == jnp.uint64(2 ** 63 + 6144), 128),
    # The second warpgroup's 10 wraps to int4's -6.
    (False, lambda wg: (wg * 10).astype(jnp.int4) == -6, 128),
    # A float predicate counts both branches at both warpgroups.
    (False, lambda wg: wg.astype(jnp.float32) == jnp.float32(2.0 ** -140), 256),
], ids=["product-out-dtype", "uint64-wrap", "int4-wrap", "float-unknown"])
def test_a_launchs_cost_counts_a_scalar_predicate_as_the_gpu_computes_it(wide, predicate, exponentials):
    """A kernel of two warpgroups exponentiating its row where a predicate on the warpgroup's index holds:
    its launch's cost (`emission.costed`, the estimate XLA and Dew's FLOP count read) states the
    exponentials of the warpgroups an integer predicate holds for, folded as the GPU computes it (a product
    at the output dtype it names, base review f9953e23a F1; a uint64's wrap, edd99fc1d F1; int4's, 0f72bd41c
    F1), and both branches of each warpgroup where the predicate is a float's, which the fold computes none
    of."""
    def body(x_ref, o_ref, row):
        wg = lax.axis_index("wg").astype(jnp.int32)
        row[...] = lax.cond(predicate(wg), jnp.exp, lambda value: value, row[...])

    with jax.enable_x64(new_val=wide):
        launch = plgpu.kernel(body, out_type=_ROW, num_threads=2, thread_name="wg",
                              scratch_types=[plgpu.SMEM((128,), jnp.float32)])
        traced = jax.make_jaxpr(emission.costed(launch, Stated(steps=1, bytes_accessed=0)))(_ROW)
    assert [equation.params["cost_estimate"].transcendentals for equation in programs.equations(traced.jaxpr)
            if "cost_estimate" in equation.params] == [exponentials]


def test_a_launchs_cost_folds_a_bodys_64_bit_arithmetic_at_its_width():
    """A kernel whose body alone traces under x64 (its own `jax.enable_x64`), its warpgroup's index widened to
    int64 and 2**31 added: its launch's cost (`emission.costed`, folded outside that scope) folds the sum at
    int64, where it is no negative number at either warpgroup, and states no exponential, where a fold at
    the ambient width narrowed it to int32's wrap below zero at both (base review 802cf38a0 F1)."""
    def body(x_ref, o_ref, row):
        with jax.enable_x64(new_val=True):
            wg = lax.axis_index("wg").astype(jnp.int64)
            row[...] = lax.cond(wg + jnp.int64(2 ** 31) < 0, jnp.exp, lambda value: value, row[...])

    launch = plgpu.kernel(body, out_type=_ROW, num_threads=2, thread_name="wg",
                          scratch_types=[plgpu.SMEM((128,), jnp.float32)])
    traced = jax.make_jaxpr(emission.costed(launch, Stated(steps=1, bytes_accessed=0)))(_ROW)
    assert [equation.params["cost_estimate"].transcendentals for equation in programs.equations(traced.jaxpr)
            if "cost_estimate" in equation.params] == [0]


def _convert(dtype) -> dict:
    return {"new_dtype": np.dtype(dtype), "weak_type": False, "sharding": None}


@pytest.mark.parametrize("primitive, operands, params", [
    (lax.div_p, (np.int32(-7), np.int32(2)), {}), (lax.div_p, (np.int32(7), np.int32(-2)), {}),
    (lax.div_p, (np.int32(-7), np.int32(-2)), {}), (lax.rem_p, (np.int32(-7), np.int32(2)), {}),
    (lax.rem_p, (np.int32(7), np.int32(-2)), {}), (lax.add_p, (np.int32(2 ** 31 - 1), np.int32(1)), {}),
    (lax.sub_p, (np.uint32(0), np.uint32(1)), {}), (lax.neg_p, (np.int32(-2 ** 31),), {}),
    (lax.mul_p, (np.int8(16), np.int8(16)), {"out_dtype": np.dtype(np.int32)}),
    (lax.mul_p, (np.int32(2), np.int32(1)), {"out_dtype": np.dtype(np.bool_)}),
    (lax.and_p, (np.int32(6), np.int32(3)), {}), (lax.xor_p, (np.int32(6), np.int32(3)), {}),
    (lax.not_p, (np.int32(5),), {}), (lax.not_p, (np.bool_(1),), {}),
    (lax.or_p, (np.bool_(0), np.bool_(1)), {}),
    (lax.max_p, (np.int32(-3), np.int32(2)), {}), (lax.clamp_p, (np.int32(0), np.int32(9), np.int32(4)), {}),
    (lax.shift_left_p, (np.int32(3), np.int32(30)), {}), (lax.shift_left_p, (np.int32(3), np.int32(32)), {}),
    (lax.shift_right_arithmetic_p, (np.int32(-8), np.int32(1)), {}),
    (lax.shift_right_arithmetic_p, (np.int32(-8), np.int32(40)), {}),
    (lax.shift_right_arithmetic_p, (np.uint32(2 ** 31), np.uint32(1)), {}),
    (lax.shift_right_arithmetic_p, (np.uint32(2 ** 31), np.uint32(40)), {}),
    (lax.shift_right_logical_p, (np.int32(-1), np.int32(28)), {}),
    (lax.shift_right_logical_p, (np.int32(-1), np.int32(32)), {}),
    (lax.convert_element_type_p, (np.int32(10),), _convert(jnp.int4)),
    (lax.convert_element_type_p, (np.int32(200),), _convert(np.int8)),
    (lax.convert_element_type_p, (np.int32(-2),), _convert(np.bool_)),
    (lax.convert_element_type_p, (np.bool_(1),), _convert(np.int32)),
], ids=lambda value: getattr(value, "name", None))
def test_an_integer_scalar_folds_as_its_primitive_computes_it(primitive, operands, params):
    """A quotient and a remainder by either sign, a width's wrap (int32's, uint32's, int8's, int4's, a
    product at the output dtype it names, a boolean's among them), bitwise and boolean logic, an extremum and
    a clamp, shifts within and past the width (an unsigned value's arithmetic one reading it as signed) and
    conversions among integer and boolean dtypes fold as the primitive bound at its operands' dtypes computes
    them (`emission.scalar_evaluated` against `lax`'s own evaluation; base review 4608795fd F1, F2), and
    lower no program to fold: bound, each kernel's index arithmetic compiled a one-primitive program a process
    (18 beside a 4B step's first trace, row 8612)."""
    bound = np.asarray(primitive.bind(*map(jnp.asarray, operands), **params))
    lowered = []

    def heard(event: str, seconds: float, **_) -> None:
        if event == "/jax/core/compile/jaxpr_to_mlir_module_duration":
            lowered.append(event)

    # JAX's in-memory caches cleared, so a fold that binds the primitive lowers its program again.
    jax.clear_caches()
    jax.monitoring.register_event_duration_secs_listener(heard)
    try:
        folded = emission.scalar_evaluated(primitive, operands, params, bound.dtype)
    finally:
        jax.monitoring.unregister_event_duration_listener(heard)
    assert folded is not None and folded.dtype == bound.dtype and folded == bound[()], (folded, bound)
    assert not lowered, lowered


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


@pytest.mark.parametrize("reduced", [jnp.prod, jnp.all, jnp.any], ids=lambda f: f.__name__)
def test_a_product_or_a_masks_reduction_counts_one_operation_an_input_element_as_a_sum_does(reduced):
    """A row value of a [64, 128] tile reduced by a product, or a mask's all or any, counts one operation an
    input element, as a sum does (`emission._REDUCTIONS`): a producer scaling its rows by their product
    (`x * prod(x, axis=1)`, a fused prologue's row value) is counted, where it was refused unpriced (base
    review of 045f07e31, F1)."""
    tile = jax.ShapeDtypeStruct((64, 128), jnp.float32)

    def counted(reduce):
        masked = reduced in (jnp.all, jnp.any)
        return _emitted(lambda x: reduce(x > 0 if masked else x, axis=1), tile).roles[0]

    assert counted(reduced).fixed.operations == counted(jnp.sum).fixed.operations, counted(reduced).fixed


@pytest.mark.parametrize("move", [
    lambda x: lax.concatenate([x[:, 64:], x[:, :64]], 1), lambda x: lax.dynamic_slice(x, (0, 0), (1, 64)),
    lambda x: lax.dynamic_update_slice(x, x[:, :64], (0, 64)), lambda x: lax.split(x, [64, 64], axis=1)[1],
    lambda x: lax.stack([x], axis=0), lambda x: lax.tile(x, (1, 2)),
    lambda x: lax.unstack(x[None], axis=0)[0]],
    ids=["concatenate", "dynamic_slice", "dynamic_update_slice", "split", "stack", "tile", "unstack"])
def test_a_layout_move_counts_nothing_as_a_slice_does(move):
    """Each layout move the Pallas fuser tiles, alone in a body over a [1, 128] row (its slices free beside
    it), counts no operation of any unit, as a slice does (`emission._FREE`): a fused prologue's stage
    holding one is priced, never refused unpriced (rows 7260, 7261)."""
    role = _emitted(move, jax.ShapeDtypeStruct((1, 128), jnp.float32)).roles[0]
    assert role.fixed == Work(), role.fixed

