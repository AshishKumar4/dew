"""A launch's work counted from its kernel's emission (`templates.emission`, row 5895): a step of its walk
counted from its body, as each warpgroup runs it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from kernel_compile import lowered_for

from dew.nn import derivatives, programs
from dew.nn.kernels import generation
from dew.nn.kernels.templates import cost, emission
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


def _semantics(predicate) -> plgpu.CompilerParams:
    """The lowering a kernel holding `predicate` takes: Warpgroup's, Dew's own kernels', unless an operation
    of it states an output dtype, which Warpgroup refuses and Lane alone holds, typing it by rules of its own
    (`emission._typed_alike`)."""
    stated = any(equation.params.get("out_dtype") is not None
                 for equation in programs.equations(jax.make_jaxpr(predicate)(jnp.int32(0)).jaxpr))
    return plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane if stated
                                else plgpu.LoweringSemantics.Warpgroup)


def _kernel(body, predicate):
    """`body` as a kernel of two warpgroups along `wg`, lowered as one holding `predicate` is
    (`_semantics`)."""
    return plgpu.kernel(body, out_type=_ROW, num_threads=2, thread_name="wg",
                        scratch_types=[plgpu.SMEM((128,), jnp.float32)],
                        compiler_params=_semantics(predicate))


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


SCALAR_PREDICATES = [
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
    # A shift past the width is poison where Mosaic GPU computes it (MLIR's `shli` and `shrsi`), whatever
    # XLA's saturating fold would give: unknown, both branches at every warpgroup (base review bb25f261d F1).
    (lambda wg: lax.shift_left(wg + 1, 32) == 0, [128, 128]),
    (lambda wg: lax.shift_right_arithmetic(wg - 1, 40) == -1, [128, 128]),
    # A 4-bit integer's shift within its width folds at its own width (ml_dtypes' int4, which NumPy's
    # iinfo refuses): 1 << 1 is 2 at the first warpgroup, 2 << 1 is 4 at the second.
    (lambda wg: lax.shift_left((wg + 1).astype(jnp.int4), jnp.int4(1)) == jnp.int4(4), [0, 128]),
    # A float predicate is no warpgroup arithmetic the fold computes: both branches at every warpgroup.
    (lambda wg: wg.astype(jnp.float32) == jnp.float32(0), [128, 128]),
    # A quotient by zero is undefined on the GPU (LLVM's `sdiv`): unknown, both branches at every warpgroup,
    # where the host's XLA defines it -1 and would count neither.
    (lambda wg: lax.div(wg + 1, wg * 0) == 0, [128, 128]),
    # A remainder by zero, an unsigned quotient by zero and a signed least integer's quotient and remainder
    # by -1 are undefined too, each where the host's value would take neither branch.
    (lambda wg: lax.rem(wg + 1, wg * 0) == 0, [128, 128]),
    (lambda wg: lax.div(wg.astype(jnp.uint32) + 1, wg.astype(jnp.uint32) * 0) == 0, [128, 128]),
    (lambda wg: lax.div(wg * 0 + jnp.int32(-2 ** 31), wg * 0 - 1) == 0, [128, 128]),
    (lambda wg: lax.rem(wg * 0 + jnp.int32(-2 ** 31), wg * 0 - 1) == 1, [128, 128]),
    # A minimum stating a dtype apart from its operands' is one Mosaic GPU types by rules of its own: unknown,
    # both branches at every warpgroup.
    (lambda wg: lax.min_p.bind(wg, jnp.int32(-1), out_dtype=jnp.dtype(jnp.uint32)).astype(jnp.int32) >= 0,
     [128, 128]),
    # A maximum at the stated dtype its operands hold folds (Mosaic Lane's guard): 1 at both warpgroups.
    (lambda wg: lax.max_p.bind(wg, jnp.int32(1), out_dtype=jnp.dtype(jnp.int32)) == 0, [0, 0]),
    # A maximum at its operands' width converted to a narrower one: int16's 255 at the second warpgroup is
    # int8's -1.
    (lambda wg: lax.max((wg * 254).astype(jnp.int16) + jnp.int16(1), jnp.int16(1)).astype(jnp.int8) == -1,
     [0, 128]),
    # A clamp of a product stated unsigned of signed operands, between unsigned bounds: its product unknown,
    # both branches (Lane's own typing takes the upper bound at both warpgroups, base review 117cc9088 F1).
    (lambda wg: lax.clamp((wg - 5).astype(jnp.uint32),
                          lax.mul_p.bind(wg - 2, jnp.int32(1), out_dtype=jnp.dtype(jnp.uint32)),
                          (wg - 4).astype(jnp.uint32)) == (wg - 4).astype(jnp.uint32),
     [128, 128]),
    # A weak literal beside an int16 operand: Lane materializes it at int16, XLA's fold at int32. Unknown,
    # both branches (base review 7e470364e F1).
    (lambda wg: lax.mul_p.bind(256, (wg + 256).astype(jnp.int16), out_dtype=jnp.dtype(jnp.int16))
     == jnp.int16(0), [128, 128]),
    # A select of a known arm and a product stated unsigned of signed operands: the product unknown, so the
    # select is too, and both warpgroups take the exponential, as Lane's does (base review 7e470364e F2).
    (lambda wg: lax.select_n(wg != 0, jnp.uint32(3),
                             lax.mul_p.bind(wg - 1, jnp.int32(1), out_dtype=jnp.dtype(jnp.uint32)))
     >= jnp.uint32(2), [128, 128]),
    # Products stated boolean of integer operands are integers to Lane (2 and 3 times wg + 1, never equal),
    # not booleans: unknown, both branches, as the GPU's inequality takes it (base review 7e470364e F3).
    (lambda wg: lax.ne(lax.mul_p.bind(wg + 1, jnp.int32(2), out_dtype=jnp.dtype(jnp.bool_)),
                       lax.mul_p.bind(wg + 1, jnp.int32(3), out_dtype=jnp.dtype(jnp.bool_))), [128, 128]),
]
"""Integer and boolean predicates on a warpgroup's index, each with the exponentials each of two warpgroups'
programs prices where its row is exponentiated as it holds (`emission.emitted`)."""

SCALAR_IDS = ["truncated-quotient", "negative-divisor", "remainder-sign", "int8-wrap", "logical-shift",
        "shift-past-width", "arithmetic-shift-past-width", "int4-shift", "float-unknown",
        "quotient-by-zero-unknown",
        "remainder-by-zero-unknown", "unsigned-quotient-by-zero-unknown",
        "least-quotient-by-minus-one-unknown", "least-remainder-by-minus-one-unknown",
        "min-at-another-signedness", "max-at-its-own-dtype", "max-converted-narrower",
        "clamp-of-a-signed-product", "weak-literal-product", "select-of-a-stated-product",
        "stated-boolean-products"]


@pytest.mark.parametrize("predicate, exponentials", SCALAR_PREDICATES, ids=SCALAR_IDS)
def test_scalar_arithmetic_on_the_warpgroup_folds_as_the_program_computes_it(predicate, exponentials):
    """A conditional on integer arithmetic over the warpgroup's index (a quotient by either sign, a
    remainder, a narrower integer's wrap, shifts within the width) counts the branch each warpgroup's program
    takes, folded as `lax` computes it at its own dtype, where Python's arithmetic floored
    an integer quotient (base review 951e3f9e7 F3); a float predicate, which no warpgroup's index arithmetic
    is, counts both branches at every warpgroup, never one dropped: the fold computes integers and booleans
    alone, where IEEE semantics' edges (subnormals, signed zeros, float64, a complex value) were where its
    defects lived (base reviews 127d2d95c to edd99fc1d)."""
    def walk(row):
        wg = lax.axis_index("wg").astype(jnp.int32)
        return lax.cond(predicate(wg), jnp.exp, lambda row: row, row)

    def body(x_ref, o_ref, row):
        row[...] = walk(row[...])

    lowered_for(_kernel(body, predicate), (_ROW,), "sm90")
    assert [role.fixed.exponentials for role in _emitted(walk, _ROW).roles] == exponentials


def _known(predicate) -> list[bool | None]:
    """Each of two warpgroups' predicate as the fold knows it (`emission.emitted`): true where only its
    branch is priced, false where only the other's is, None where both are (an unknown predicate)."""
    def priced(*, taken: bool) -> list[bool]:
        def walk(row):
            wg = lax.axis_index("wg").astype(jnp.int32)
            exponentiated = (jnp.exp, lambda value: value) if taken else (lambda value: value, jnp.exp)
            return lax.cond(predicate(wg), *exponentiated, row)

        return [role.fixed.exponentials > 0 for role in _emitted(walk, _ROW).roles]

    return [None if held and other else held
            for held, other in zip(priced(taken=True), priced(taken=False), strict=True)]


@pytest.mark.gpu(reason="runs each predicate on the warpgroups of this GPU")
@pytest.mark.generations("sm90", "sm100")
@pytest.mark.parametrize("predicate, exponentials", SCALAR_PREDICATES, ids=SCALAR_IDS)
def test_a_folds_known_predicate_is_the_one_each_warpgroup_computes(predicate, exponentials):
    """Each predicate the fold knows at a warpgroup (`_known`: its branch alone priced, or the other's alone)
    is the one that warpgroup computes on this GPU, the kernel lowered by the semantics holding it
    (`_semantics`), each warpgroup writing its predicate to its own part of the output: the fold priced
    against the GPU, not against expectations stated by hand (base review 117cc9088 F4, row 8799). An
    unknown predicate, both branches priced, holds whatever the GPU computes."""
    del exponentials

    def body(out_gmem, staged):
        wg = lax.axis_index("wg").astype(jnp.int32)
        at = pl.ds(wg * 128, 128)
        staged[at] = jnp.full((128,), predicate(wg).astype(jnp.int32))
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(staged.at[at], out_gmem.at[at])
        plgpu.wait_smem_to_gmem(0)

    computed = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((256,), jnp.int32), num_threads=2,
                            thread_name="wg", scratch_types=[plgpu.SMEM((256,), jnp.int32)],
                            compiler_params=_semantics(predicate))()
    on_the_gpu = [bool(value) for value in np.asarray(computed).reshape(2, 128)[:, 0]]
    known = _known(predicate)
    assert all(held is None or held == gpu for held, gpu in zip(known, on_the_gpu, strict=True)), (
        known, on_the_gpu)


@pytest.mark.parametrize("predicate, exponentials", [
    # A product stated unsigned of signed operands is one Mosaic GPU types by rules of its own: unknown, both
    # branches at both warpgroups (Lane keeps its -1 signed, base review 68aa03836 F2).
    (lambda wg: lax.ge(lax.mul_p.bind(wg - 1, jnp.int32(1), out_dtype=jnp.dtype(jnp.uint32)), jnp.uint32(2)),
     256),
    # The second warpgroup's 10 wraps to int4's -6.
    (lambda wg: (wg * 10).astype(jnp.int4) == -6, 128),
    # A float predicate counts both branches at both warpgroups.
    (lambda wg: wg.astype(jnp.float32) == jnp.float32(2.0 ** -140), 256),
], ids=["product-at-another-signedness", "int4-wrap", "float-unknown"])
def test_a_launchs_cost_counts_a_scalar_predicate_as_the_gpu_computes_it(predicate, exponentials):
    """A kernel of two warpgroups exponentiating its row where a predicate on the warpgroup's index holds:
    its launch's cost (`emission.costed`, the estimate XLA and Dew's FLOP count read) states the
    exponentials of the warpgroups an integer predicate holds for, folded as the GPU computes it (a product
    at the output dtype it names, base review f9953e23a F1; int4's wrap, 0f72bd41c F1), and both branches of
    each warpgroup where the predicate is a float's, which the fold computes none of. Each kernel lowered for
    a GPU: a uint64's wrap (edd99fc1d F1) is no cell, as Mosaic GPU lowers no uint64 arithmetic of a
    warpgroup's index (std::bad_cast, Lane's and Warpgroup's alike)."""
    def body(x_ref, o_ref, row):
        wg = lax.axis_index("wg").astype(jnp.int32)
        row[...] = lax.cond(predicate(wg), jnp.exp, lambda value: value, row[...])

    launch = _kernel(body, predicate)
    lowered_for(launch, (_ROW,), "sm90")
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


def test_an_async_smem_to_tmem_copy_charges_shared_bytes_from_its_source_without_device_bytes():
    """`plgpu.async_copy_smem_to_tmem` charges `Work.shared_bytes` from its transformed SMEM source using
    packed byte sizing (`resources.stored_bytes`) and leaves `device_bytes` at 0 (rows 9106, 9476)."""
    def body(out_gmem, smem, tmem):
        del out_gmem
        plgpu.async_copy_smem_to_tmem(smem, tmem)

    def sliced_body(out_gmem, smem, tmem):
        del out_gmem
        plgpu.async_copy_smem_to_tmem(smem.at[:, pl.ds(0, 64)], tmem)

    full = plgpu.kernel(body, out_type=_ROW, num_threads=1, thread_name="wg",
                        scratch_types=[plgpu.SMEM((128, 256), jnp.bfloat16),
                                       plgpu.TMEM((128, 256), jnp.bfloat16, packed=True)])
    sliced = plgpu.kernel(sliced_body, out_type=_ROW, num_threads=1, thread_name="wg",
                          scratch_types=[plgpu.SMEM((128, 256), jnp.bfloat16),
                                         plgpu.TMEM((128, 64), jnp.bfloat16, packed=True)])
    packed = plgpu.kernel(body, out_type=_ROW, num_threads=1, thread_name="wg",
                          scratch_types=[plgpu.SMEM((128, 64), jnp.int4),
                                         plgpu.TMEM((128, 64), jnp.int4, packed=True)])
    full_work = emission.emitted(jax.make_jaxpr(full)(), 1, "wg").roles[0].fixed
    sliced_work = emission.emitted(jax.make_jaxpr(sliced)(), 1, "wg").roles[0].fixed
    packed_work = emission.emitted(jax.make_jaxpr(packed)(), 1, "wg").roles[0].fixed
    assert full_work.shared_bytes == 128 * 256 * 2 and full_work.device_bytes == 0
    assert sliced_work.shared_bytes == 128 * 64 * 2 and sliced_work.device_bytes == 0
    assert packed_work.shared_bytes == 128 * 64 // 2 and packed_work.device_bytes == 0



def test_a_walks_serial_loops_compose_their_paths_before_their_mean():
    """A walk's loops (its opening, steady and closing segments) run one after another through one ring: a
    step's path is their paths followed one by the other, then their mean, so two segments of one barrier
    round trip each take one a step (80.1 ns on a GH200), not half of one (base review 1e0a501f0 F2)."""
    target = generation.gpu_target("NVIDIA GH200 480GB")
    segment = cost.Work(chains=(cost.Chain(barriers=1),))
    role = emission.Role(cost.Work(), (segment, segment))
    assert role.step(lambda work: cost.chained(work, target)) == target.barrier_seconds


def test_a_launch_carries_each_warpgroups_paths_its_steps_times():
    """A launch's paths (`Emitted.paths`) are each warpgroup's work outside its walks followed by its walk's
    step its steps' times, the warpgroups side by side: a program waiting on four barriers outside its walk
    keeps them (base review 1e0a501f0 F3)."""
    waits = cost.Chain(barriers=4)
    stepping = cost.Chain(products=1, columns=128)
    roles = (emission.Role(cost.Work(chains=(waits,)), (cost.Work(chains=(stepping,)),)),)
    assert emission.Emitted(roles).paths(8) == (cost.Chain(products=8, columns=1024, barriers=4),)


_SWIZZLED = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))


def _waits(body, *scratch, warpgroups: int = 1) -> tuple:
    """The paths a kernel of `body` over `scratch` waits through, each warpgroup's work outside its walks."""
    kernel = plgpu.kernel(body, out_type=_ROW, num_threads=warpgroups, thread_name="wg",
                          scratch_types=list(scratch))
    return tuple(role.fixed.chains for role in emission.emitted(jax.make_jaxpr(kernel)(), warpgroups,
                                                                 "wg").roles)


@pytest.mark.parametrize("allowed, waited", [(0, (cost.Chain(products=1, columns=128),)), (1, ())])
def test_a_warpgroup_waits_on_the_products_it_drains(allowed, waited):
    """A 64 x 128 product of 16-bit operands 64 deep, waited down to `allowed` in flight: drained, the
    warpgroup's path waits on it, its 128 columns (`cost.Chain.columns`); let fly, a later trip waits on it,
    which this one's work hides (row 4123)."""
    def body(out_gmem, a, b):
        del out_gmem

        def scope(acc):
            plgpu.wgmma(acc, a, b)
            plgpu.wgmma_wait(allowed)
            return acc[...]

        pl.run_scoped(scope, plgpu.ACC((64, 128), jnp.float32))

    assert _waits(body, plgpu.SMEM((64, 64), jnp.bfloat16, transforms=_SWIZZLED),
                  plgpu.SMEM((64, 128), jnp.bfloat16, transforms=_SWIZZLED)) == (waited,)


def test_a_warpgroup_waits_on_a_copy_its_barrier_lands_or_on_another_warpgroups_arrival():
    """A barrier's wait the warpgroup's path runs through: a tile copy landing on it (its bytes), or, where
    none of its copies or products lands on it, another warpgroup's arrival, a round trip (row 4123)."""
    def copied(out_gmem, tile, ready):
        plgpu.copy_gmem_to_smem(out_gmem, tile, ready)
        plgpu.barrier_wait(ready)

    def handed(out_gmem, ready):
        del out_gmem
        plgpu.barrier_wait(ready)

    assert _waits(copied, plgpu.SMEM((128,), jnp.float32), plgpu.Barrier()) == (
        (cost.Chain(copies=1, copy_bytes=512),),)
    assert _waits(handed, plgpu.Barrier()) == ((cost.Chain(barriers=1),),)


def test_an_integer_operation_counts_on_the_integer_units_not_the_fp32_lanes():
    """A row's integer XOR, shift, add and comparison each count one integer operation an element
    (`Work.integer_operations`: the ALU's LOP3, SHF, IADD3 and ISETP, issued every 2 cycles a sub-partition,
    half the fp32 lanes' rate) and none of the fp32 lanes (row 5386)."""
    row = jax.ShapeDtypeStruct((1, 128), jnp.int32)
    operations = (lambda x: x ^ 7, lambda x: x >> 3, lambda x: x + 1, lambda x: (x < 5).astype(jnp.int32))
    for operation in operations:
        role = _emitted(operation, row).roles[0]
        assert (role.fixed.integer_operations, role.fixed.operations) == (128, 0), role.fixed


@pytest.mark.parametrize("longest, span", [(None, 10 / 3), (4, 4)])
def test_a_launchs_span_is_its_longest_programs_walk_where_its_programs_walk_unequal_shares(longest, span):
    """A launch of three programs walking ten steps among them: its span, the dependent steps on its critical
    path, is the most one program walks where the launch states it (`cost.Stated.longest`: a causal walk's
    tiles split 3, 3 and 4), else their mean, where the mean of unequal walks read 10 / 3 (base review
    14020e321 F3)."""
    def body(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    launch = plgpu.kernel(body, out_type=_ROW, grid=(3,), grid_names=("programs",))
    stated = Stated(steps=10, bytes_accessed=0, longest=longest)
    traced = jax.make_jaxpr(emission.costed(launch, stated))(_ROW)
    (kernel,) = [equation for equation in programs.equations(traced.jaxpr)
                 if "cost_estimate" in equation.params]
    assert emission.work_of(kernel).span == span


def test_identical_scale_smem_bodies_keep_independent_emission_work():
    """Row 9483: two distinct `across.ScaleSmem` bodies with identical fields register independent
    `emission.stating` entries (`eq=False`): binding the second into a jaxpr and releasing the first leaves
    the second's stated scaling work counted by emission."""
    import gc

    from dew.nn.kernels.templates import across

    elements = 128 * 64
    work = Work(operations=elements, shared_bytes=2 * elements * 2)
    first = emission.stating(lambda r: work)(across.ScaleSmem(elements // 2, 0x3F803F80, bf16=True))
    second = emission.stating(lambda r: work)(across.ScaleSmem(elements // 2, 0x3F803F80, bf16=True))

    def traced(inline_body):
        def body(out_gmem, smem):
            del out_gmem
            plgpu.inline_mgpu(arg_types=(plgpu.RefType(),))(inline_body)(smem)

        kernel = plgpu.kernel(body, out_type=_ROW, num_threads=1, thread_name="wg",
                              scratch_types=[plgpu.SMEM((128, 64), jnp.bfloat16)])
        return jax.make_jaxpr(kernel)()

    closed = traced(second)
    del first, second
    gc.collect()
    fixed = emission.emitted(closed, 1, "wg").roles[0].fixed
    assert fixed == work, fixed
