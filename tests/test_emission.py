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
from reference_error import ahead_arm

from dew.nn import programs
from dew.nn.kernels import generation
from dew.nn.kernels.templates import cost, emission, mosaic
from dew.nn.kernels.templates.cost import Stated, Work

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
        mosaic.step_index(index)
        return work(row)

    return lambda row, trips: lax.fori_loop(0, trips, step, row)


def _emitted(fn, *args) -> emission.Emitted:
    """`fn` counted as each of two warpgroups along `wg` runs it."""
    return emission.emitted(jax.make_jaxpr(fn, axis_env=[("wg", 2)])(*args), 2, "wg")


def _semantics(predicate) -> plgpu.CompilerParams:
    """The lowering a kernel holding `predicate` takes: Warpgroup's, Dew's own kernels', unless an operation
    of it states an output dtype, which Warpgroup refuses and Lane alone holds, typing it by rules of its own
    (`emission._typed_alike`)."""
    index = jax.ShapeDtypeStruct((), jnp.int32)
    stated = any(equation.params.get("out_dtype") is not None
                 for equation in programs.equations(jax.make_jaxpr(predicate)(index).jaxpr))
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


@pytest.mark.parametrize("operation", [jnp.exp, jnp.log, jnp.tanh, jax.nn.sigmoid], ids=lambda f: f.__name__)
def test_a_kernel_counts_a_special_function_in_the_form_its_approx_math_lowers_it_to(operation):
    """A kernel's special function counts as its compile lowers it: the hardware's approximate form where
    the kernel asks `approx_math` (an exp one MUFU beside its scaling multiply), and the accurate form
    elsewhere, every GEMM launch's (an exp the MUFU beside seven fp32 instructions, a saturating convert and
    a shift; a log a polynomial with no MUFU; row 10824, FS-APPROX-FORMS3, sm90 and sm100 alike). The
    accurate form issues more fp32 work an element than the approximate one."""
    def work(*, approx: bool):
        def body(row, out, staged):
            staged[...] = operation(row[...])
            out[...] = staged[...]

        launch = plgpu.kernel(body, out_type=_ROW, num_threads=1, thread_name="wg",
                              scratch_types=[plgpu.SMEM((128,), jnp.float32)],
                              compiler_params=plgpu.CompilerParams(approx_math=approx))
        traced = jax.make_jaxpr(launch)(_ROW).jaxpr
        (kernel,) = [equation for equation in programs.equations(traced)
                     if equation.primitive.name in ("mpmd_map", "pallas_call")]
        return emission.launch_work(kernel, Stated(steps=0, bytes_accessed=0))

    approximate, accurate = work(approx=True), work(approx=False)
    assert accurate.operations > approximate.operations, (accurate, approximate)
    assert approximate.unpriced_operations == accurate.unpriced_operations == 0, (accurate, approximate)


def test_a_step_counts_each_trip_of_a_loop_inside_it():
    """A walk step holding a loop of four trips counts its body four times."""
    walk = _walked(lambda row: lax.scan(lambda row, _: (jnp.exp(row), None), row, length=4)[0])
    assert [role.step(_exponentials) for role in _emitted(walk, _ROW, _TRIPS).roles] == [4 * 128] * 2


def test_a_walk_of_stated_trips_counts_each_trip_where_it_stands():
    """A walk whose trips its jaxpr states (a scan's) counts its body each trip, its work the program's own:
    no step a launch's statement multiplies."""
    def walk(row):
        def step(row, index):
            mosaic.step_index(index)
            return row + 1, None

        return lax.scan(step, row, jnp.arange(4))[0]

    counted = _emitted(walk, _ROW)
    assert [(role.fixed.operations, role.step(_operations)) for role in counted.roles] == [(512, None)] * 2


@pytest.mark.parametrize("device", [True, False], ids=["device", "scoped"])
def test_a_kernels_reads_and_writes_of_a_device_ref_count_their_bytes_as_device_traffic(device):
    """A kernel body's read of a `[4, 8]` fp32 ref and write of an `[8]` fp32 row (`state.get` and `swap`)
    count their 160 bytes as device traffic where its launch states the ref in device memory (`emitted`'s
    `device`: a Triton launch's blocks, the decode merge's), each a step of a walk of its own counted per
    step, and nothing where it states none (a scoped ref's, which its copies price): a block a narrower
    program rereads is traffic a launch's price takes (the merge P0's base review of 05f8cec4d, F8)."""
    def body(source, target, trips):
        def step(index, carry):
            mosaic.step_index(index)
            target[...] = jnp.sum(source[...], axis=0)
            return carry

        lax.fori_loop(0, trips[...], step, 0)

    shapes = (jax.ShapeDtypeStruct((4, 8), jnp.float32), jax.ShapeDtypeStruct((8,), jnp.float32),
              jax.ShapeDtypeStruct((), jnp.int32))
    (state,) = (equation for equation in jax.make_jaxpr(lambda *arrays: _with_refs(body, *arrays))(
        *shapes).jaxpr.eqns if equation.primitive.name == "run_state")
    (role,) = emission.emitted(state.params["jaxpr"], 1, None, device=(0, 1) if device else ()).roles
    assert role.step(lambda work: work.device_bytes) == (4 * 8 * 4 + 8 * 4 if device else 0)


@pytest.mark.parametrize("live", [True, False], ids=["live", "dropped"])
def test_a_swap_of_a_device_ref_reads_its_old_value_where_it_is_live(live):
    """A swap of eight fp32 elements of a device ref counts 64 bytes where its old value feeds an output
    (a load and a store, as Triton lowers it) and 32 where the old value is dropped (a store alone): the
    merge P0's base review of aa990591f, F6."""
    shapes = (jax.ShapeDtypeStruct((8,), jnp.float32),) * 2
    traced = jax.make_jaxpr(lambda *arrays: _with_refs(_swap_body(live), *arrays))(*shapes)
    (state,) = (equation for equation in traced.jaxpr.eqns if equation.primitive.name == "run_state")
    (role,) = emission.emitted(state.params["jaxpr"], 1, None, device=(0,)).roles
    assert role.fixed.device_bytes == (64 if live else 32)


def _swap_body(live: bool):
    """A body swapping eight ones into its first ref, its old value written to its second where `live`."""
    from jax._src.state import primitives as state_primitives

    def body(target, out):
        old = state_primitives.ref_swap(target, (), jnp.ones(8, jnp.float32))
        if live:
            out[...] = old
    return body


def _with_refs(body, *arrays):
    """`body` run over refs of `arrays` (`jax.experimental.pallas`' `run_state`), its jaxpr's operands
    refs as a kernel's are."""
    from jax._src.state.discharge import run_state
    return run_state(lambda refs: body(*refs))(arrays)


def _one(row, index):
    mosaic.step_index(index)
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
    mosaic.step_index(index)
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
    unknown predicate, both branches priced, holds whatever the GPU computes. Its program compiled in a
    tier's prepare (`ahead_arm`)."""
    del exponentials

    def body(out_gmem, staged):
        wg = lax.axis_index("wg").astype(jnp.int32)
        at = pl.ds(wg * 128, 128)
        staged[at] = jnp.full((128,), predicate(wg).astype(jnp.int32))
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(staged.at[at], out_gmem.at[at])
        plgpu.wait_smem_to_gmem(0)

    named = SCALAR_IDS[[each[0] for each in SCALAR_PREDICATES].index(predicate)]
    launch = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((256,), jnp.int32), num_threads=2,
                          thread_name="wg", scratch_types=[plgpu.SMEM((256,), jnp.int32)],
                          compiler_params=_semantics(predicate))
    computed = ahead_arm(jax.jit(launch), f"a warpgroup's predicate: {named}")()
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
    warpgroup's index (std::bad_cast, Lane's and Warpgroup's alike). Its cost, JAX's in-memory caches
    cleared before it, compiles no program (`Compiles`: no backend compile within its trace; row 8707, the
    fold proved through the launch that consumes it, not its helper alone)."""
    from dew.telemetry.compiles import Compiles

    def body(x_ref, o_ref, row):
        wg = lax.axis_index("wg").astype(jnp.int32)
        row[...] = lax.cond(predicate(wg), jnp.exp, lambda value: value, row[...])

    launch = _kernel(body, predicate)
    lowered_for(launch, (_ROW,), "sm90")
    jax.clear_caches()
    with Compiles(None) as compiles:
        traced = jax.make_jaxpr(emission.costed(launch, Stated(steps=1, bytes_accessed=0)))(_ROW)
    assert [equation.params["cost_estimate"].transcendentals for equation in programs.equations(traced.jaxpr)
            if "cost_estimate" in equation.params] == [exponentials]
    assert compiles.count == 0, compiles.programs


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


@pytest.mark.parametrize("start", ["operand", "indices"])
def test_a_walks_guards_on_its_steps_place_count_at_their_trips(start):
    """A walk over a range whose first step takes its own exponential and every step but its last the next
    one's (`index == lo`, `index + 1 < hi`, as a walk reading ahead guards them) counts one a row a step and
    none apart: each guard read at the trips it decides, not taken every step. Its start an operand, or
    computed of the kernel's indices alone (an iota's least), which the count holds by its atom as any
    unknown start (its invariance once hid that identity from the guard, which took both branches every
    step)."""
    def walk(row, lo, hi):
        lo = jnp.min(lax.iota(jnp.int32, 8)) if start == "indices" else lo

        def step(index, row):
            mosaic.step_index(index)
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
            mosaic.step_index(index)
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


def _ring_trips(ringed) -> tuple:
    """The paths of one warpgroup's kernel `ringed`, of a ring of three tile slots and their barriers, as a
    launch's work states them (`launch_work`)."""
    kernel = plgpu.kernel(ringed, out_type=_ROW,
                          scratch_types=[plgpu.SMEM((3, 128), jnp.float32), plgpu.Barrier(num_barriers=3)])
    (launched,) = (equation for equation in programs.equations(jax.make_jaxpr(kernel)().jaxpr)
                   if equation.primitive in programs.NATIVE_KERNELS)
    return emission.launch_work(launched, Stated(steps=0, bytes_accessed=0)).chains


@pytest.mark.parametrize("prefetched, stages", [(True, 3), (False, 1)], ids=["prefetched", "at-once"])
def test_a_copy_into_a_rings_slot_waits_as_the_ring_hides_it(prefetched, stages):
    """Each trip of a loop waiting on its own slot of a ring of three: a copy it issues two slots ahead, the
    ring's first two filled before the loop, a prefetch as deep as the ring, waits as its three stages hide
    it, where one copy landing on its own barrier hides nothing (row 4123: every copy priced waited whole, its
    ring's depth unread); a copy into its own slot, waited at once, waits whole, the ring's depth no proof of
    turns between (row 12369: an attention consumer's copy and wait on one slot priced a third; its review:
    the prefetch one transfer ahead of three stages, the at-once arm's slot filled twice before a wait)."""
    def ringed(out_gmem, tiles, ready):
        if prefetched:
            for filled in (0, 1):
                plgpu.copy_gmem_to_smem(out_gmem, tiles.at[filled], ready.at[filled])

        def trip(step):
            slot = step % 3
            copied = (step + 2) % 3 if prefetched else slot
            plgpu.copy_gmem_to_smem(out_gmem, tiles.at[copied], ready.at[copied])
            plgpu.barrier_wait(ready.at[slot])
            return step + 1

        lax.while_loop(lambda step: step < 4, trip, 0)

    assert _ring_trips(ringed) == (cost.Chain(copies=1, copy_bytes=512, stages=stages),)


def test_a_copy_waited_at_once_on_a_slot_a_captured_scalar_names_waits_whole():
    """A copy and its wait on a ring's slot a captured 0-d NumPy index names, a literal that hashes as none:
    the wait reads the slot the copy named, at once, and waits for the copy whole (row 12369's review: such a
    slot ended the count in a TypeError)."""
    slot = np.array(1, np.int32)

    def ringed(out_gmem, tiles, ready):
        plgpu.copy_gmem_to_smem(out_gmem, tiles.at[slot], ready.at[slot])
        plgpu.barrier_wait(ready.at[slot])

    assert _ring_trips(ringed) == (cost.Chain(copies=1, copy_bytes=512),)


def test_a_copy_one_warpgroup_issues_waits_in_the_warpgroup_that_reads_it():
    """A tile copy a producer warpgroup issues onto one of a ring's barriers each trip of a loop, which the
    consumer warpgroup waits on: the launch's path waits for the copy as the ring's three stages hide it,
    where the consumer read the wait as another warpgroup's arrival, one barrier round trip, and the copy
    waited on no path (row 4123, the pricing head's review: a GH200's recurrence reverse priced at 0.29 of its
    time, its producer's copies on none of its paths, M-FS-64)."""
    def split(out_gmem, tiles, ready):
        def trip(step):
            @pl.when(lax.axis_index("wg") == 1)
            def _():
                plgpu.copy_gmem_to_smem(out_gmem, tiles.at[1], ready.at[1])

            @pl.when(lax.axis_index("wg") == 0)
            def _():
                plgpu.barrier_wait(ready.at[1])

            return step + 1

        # Trips of a loop the program's walk runs (a while loop's, as a walk's own), the copy one trip's.
        lax.while_loop(lambda step: step < 4, trip, 0)

    kernel = plgpu.kernel(split, out_type=_ROW, num_threads=2, thread_name="wg",
                          scratch_types=[plgpu.SMEM((3, 128), jnp.float32), plgpu.Barrier(num_barriers=3)])
    (launched,) = (equation for equation in programs.equations(jax.make_jaxpr(kernel)().jaxpr)
                   if equation.primitive in programs.NATIVE_KERNELS)
    assert emission.launch_work(launched, Stated(steps=0, bytes_accessed=0)).chains == (
        cost.Chain(copies=1, copy_bytes=512, stages=3),)


def test_an_integer_operation_counts_on_the_integer_units_not_the_fp32_lanes():
    """A row's integer XOR, shift, add and comparison each count one integer operation an element
    (`Work.integer_operations`: the ALU's LOP3, SHF, IADD3 and ISETP, issued every 2 cycles a sub-partition,
    half the fp32 lanes' rate) and none of the fp32 lanes (row 5386)."""
    row = jax.ShapeDtypeStruct((1, 128), jnp.int32)
    operations = (lambda x: x ^ 7, lambda x: x >> 3, lambda x: x + 1, lambda x: (x < 5).astype(jnp.int32))
    for operation in operations:
        role = _emitted(operation, row).roles[0]
        assert (role.fixed.integer_operations, role.fixed.operations) == (128, 0), role.fixed


def test_a_steps_mask_of_its_tiles_indices_issues_nothing_and_one_of_its_place_counts():
    """A walk step's mask of its tile's indices alone (iotas and literals: the recurrence solve's blocks by a
    floor division and a remainder) issues no integer operation a step, the GPU's compile folding it or
    computing it once outside the loop (`emission._INVARIANT`, row 4123: the solve's masks priced at 6.1M
    integer operations a GH200 program against about a thousand integer instructions a thread a trip); a mask
    of the step's place counts its integer operations each step. Its select of the row's data counts on the
    fp32 lanes alike."""
    def walk(placed: bool):
        def step(index, row):
            mosaic.step_index(index)
            columns = lax.iota(jnp.int32, 128) + (index if placed else 0)
            return jnp.where((columns // 4) % 2 == 1, row, 0.0)

        return lambda row, trips: lax.fori_loop(0, trips, step, row)

    def counted(placed: bool) -> tuple:
        role = _emitted(walk(placed), _ROW, _TRIPS).roles[0]
        return role.step(lambda work: work.integer_operations), role.step(_operations)

    (indexed, selected), (placed, selected_placed) = counted(placed=False), counted(placed=True)
    assert indexed == 0 and placed >= 3 * 128, (indexed, placed)
    assert selected == selected_placed == 128, (selected, selected_placed)


def _static_multiply(factor: int) -> jax.Array:
    """`iota * factor` as a row's floats: an integer multiply of iotas and literals alone, 128 of them."""
    return (lax.iota(jnp.int32, 128) * factor).astype(jnp.float32)


def test_static_work_a_loop_reaches_is_hoisted_once_whichever_trips_reach_it():
    """Static work (iotas and literals) inside a loop is computed once outside it, as the compiler hoists
    it, whichever of its trips or arms reach it (row 12767): a walk whose parity picks `iota * 3` or
    `iota * 5` computes both, 256 multiplies; a two-trip scan whose second trip alone takes `iota * 3`
    computes it, 128; a two-trip scan whose arms its data decides computes both, 256."""
    def parity(row, trips):
        def step(index, row):
            mosaic.step_index(index)
            return row + lax.cond(index % 2 == 1, lambda: _static_multiply(3), lambda: _static_multiply(5))

        return lax.fori_loop(0, trips, step, row)

    def later(row):
        def step(index, row):
            return lax.cond(index == 1, lambda row: row + _static_multiply(3), lambda row: row, row)

        return lax.fori_loop(0, 2, step, row)

    def decided(row):
        def step(row, _):
            return lax.cond(row[0] > 0, lambda row: row + _static_multiply(3),
                            lambda row: row + _static_multiply(5), row), None

        return lax.scan(step, row, None, length=2)[0]

    walked = _emitted(parity, _ROW, _TRIPS).roles[0].fixed
    assert walked.integer_multiply_adds == 2 * 128, walked
    assert _emitted(later, _ROW).roles[0].fixed.integer_multiply_adds == 128
    assert _emitted(decided, _ROW).roles[0].fixed.integer_multiply_adds == 2 * 128


def test_static_work_outside_loops_runs_on_its_path_once_a_call():
    """Static work outside every loop runs where its path does: in a conditional its data decides, its arm's
    under the arms' envelope with the arm's other work (`iota * 3`'s multiply, conversion and add against
    one add: 384 issues; against three adds, 384 too, never 640), and once at each call of a cached jitted
    body (two calls, 256 multiplies) (row 12767)."""
    def chosen(first, other):
        def body(row):
            return lax.cond(row[0] > 0, first, other, row)

        return cost.issued(_emitted(body, _ROW).roles[0].fixed)

    def tripled_row(row):
        return row + _static_multiply(3)

    # The predicate and its branch index alone, which each conditional below issues beside its arms.
    baseline = chosen(lambda row: row, lambda row: row)
    against_one = chosen(tripled_row, lambda row: row + 1.0)
    against_three = chosen(tripled_row, lambda row: ((row + 1.0) + 2.0) + 3.0)
    assert against_one - baseline == against_three - baseline == 3 * 128, (baseline, against_one,
                                                                           against_three)

    tripled = jax.jit(lambda x: x * 3)

    def called(row):
        indices = lax.iota(jnp.int32, 128)
        return row + (tripled(indices) + tripled(indices + 1)).astype(jnp.float32)

    assert _emitted(called, _ROW).roles[0].fixed.integer_multiply_adds == 2 * 128


@pytest.mark.parametrize("held", ["step", "scope", "branch"])
def test_a_steps_mask_of_a_runtime_value_counts_its_integer_operations_each_step(held):
    """A walk step's mask of its tile's indices offset by its parity (`index % 2`, a value the count holds
    concretely at each trip it reads, `emission._Place`, none of the program's constants) is computed each
    step, its integer operations counted every step: in the step, in a scope's body that captures it
    (`pl.run_scoped`), whose constants are as its operands are, or by the literal offset a branch the parity
    selects returns, whose outputs vary as its selector does."""
    def masked(row, offset):
        columns = lax.iota(jnp.int32, 128) + offset
        return jnp.where((columns // 4) % 2 == 1, row, 0.0)

    def step(index, row):
        mosaic.step_index(index)
        parity = index % 2
        if held == "scope":
            return pl.run_scoped(lambda scratch: masked(row, parity), plgpu.SMEM((8,), jnp.float32))
        if held == "branch":
            return masked(row, lax.cond(parity == 0, lambda: 0, lambda: 1))
        return masked(row, parity)

    role = _emitted(lambda row, trips: lax.fori_loop(0, trips, step, row), _ROW, _TRIPS).roles[0]
    assert role.step(lambda work: work.integer_operations) >= 3 * 128, role


def test_a_calls_static_output_beside_a_runtime_one_issues_nothing_a_step():
    """A call returning a mask of its tile's indices beside a value of the step (`jax.jit`) gives each output
    the provenance its body computes: the mask stays static, so its use issues nothing a step, where the
    call's runtime operand once made every output a runtime value."""
    def step(index, row):
        mosaic.step_index(index)
        odd, _ = jax.jit(lambda at: (lax.iota(jnp.int32, 128) % 2 == 1, at + 1))(index)
        return jnp.where(odd & (lax.iota(jnp.int32, 128) < 64), row, 0.0)

    role = _emitted(lambda row, trips: lax.fori_loop(0, trips, step, row), _ROW, _TRIPS).roles[0]
    assert role.step(lambda work: work.integer_operations) < 128, role


def test_an_integer_multiply_counts_on_the_fma_pipe_at_its_own_rate_beside_the_fp32_lanes():
    """A row's integer multiply counts one integer multiply-add an element (`Work.integer_multiply_adds`:
    IMAD, the FMA pipe's, as a bit form's left shifts by a multiply are, `bits.BitForm.work`) and nothing of
    the ALU's or the fp32 lanes'; a launch of them is priced at the target's IMAD rate
    (`Architecture.integer_multiply_adds_per_clock`), its seconds the fp32 lanes' unit's, so an FFMA beside it
    adds to them: on sm90, 64 a clock against the lanes' 128 instructions, where read as fp32 instructions
    they cost half (the review of 3580, F3)."""
    row = jax.ShapeDtypeStruct((1, 128), jnp.int32)
    role = _emitted(lambda x: x * 3, row).roles[0]
    assert (role.fixed.integer_multiply_adds, role.fixed.integer_operations, role.fixed.operations) == (
        128, 0, 0), role.fixed
    target = generation.gpu_target("sm90")
    rates = cost.rates(target, device=False)
    assert rates.multiply_adds == 64 / 128 * rates.cuda
    alone, _ = cost.units(Work(integer_multiply_adds=128), rates)
    beside, _ = cost.units(Work(integer_multiply_adds=128, operations=256), rates)
    assert beside[1] == alone[1] + 256 / rates.cuda


def _read_twice(product, z):
    """`product + z` times `product`: one multiply's output read by an add and by a multiply."""
    return (product + z) * product


@pytest.mark.parametrize(("body", "fused"), [
    (lambda x, y, z: x * y + z, 128), (lambda x, y, z: jnp.maximum(x * y, z), 0),
    (lambda x, y, z: _read_twice(x * y, z), 0), (lambda x, y, z: lax.add(lax.mul(x[0, 0], y[0, 0]), z), 0)],
    ids=["multiply-add", "maximum", "product-read-twice", "scalar-product-into-a-row"])
def test_an_fp32_multiply_whose_one_use_is_an_add_counts_as_one_fused_instruction(body, fused):
    """An fp32 multiply whose result an add alone reads is one FFMA with it (ptxas's --fmad, its default):
    the body counts the pair fused (`Work.fused_multiply_adds`), an element each; a maximum of a product, a
    product read twice, or one scalar product added to a row's every element (the review of lane/toolkit
    5cbbe66c586, P1) keeps its multiply an instruction of its own."""
    row = jax.ShapeDtypeStruct((1, 128), jnp.float32)
    role = _emitted(body, row, row, row).roles[0]
    assert role.fixed.fused_multiply_adds == fused, role.fixed


def test_a_walk_issuing_one_warp_a_scheduler_waits_each_instructions_dependent_latency():
    """A GH200 program of one warpgroup, one resident a multiprocessor, issues one warp a scheduler: each of
    its instructions waits the one before it (`Multiprocessor.issue_interval`, 4.6 clocks: M-NA-21's decode
    walk at one warp a scheduler, 4.56, and M-FS-70's KDA intra at two, 57% not issued, 4.65), so its walk
    takes its issue slots 4.6 times over, beyond its ALU's 2; twice the programs of the same work, two
    resident, issue two warps a scheduler and take the same time (M-NA-21: K64 at two resident 309 us
    against K128 at one 346 us, the same bytes in flight). An RTX 3090's interval unmeasured, its walk is
    its units'."""
    hopper, unmeasured = generation.gpu_target("NVIDIA GH200 480GB"), generation.gpu_target("sm86")
    program = Work(integer_operations=128 * 1000)

    def launched(target, count: int, resident: int, warps: float) -> float:
        work = Work(integer_operations=program.integer_operations * count, kernels=1)
        return cost.launch_seconds(work, count, target, resident=resident, warps=warps)

    sms, issue = hopper.multiprocessors, cost.rates(hopper, device=False).issue
    assert sms is not None and issue is not None
    slots, launch = program.integer_operations / issue, hopper.launch_seconds or 0.0
    # One warp a scheduler waits the interval each instruction; two, each its half, a multiprocessor's two.
    assert launched(hopper, sms, 1, 4) == slots * 4.6 + launch
    assert launched(hopper, 2 * sms, 2, 4) == 2 * (slots * (4.6 / 2)) + launch
    # Three a multiprocessor at two resident: a wave of two, then one alone (the review of lane/toolkit
    # 61dd3c7d2).
    assert launched(hopper, 3 * sms, 2, 4) == 2 * (slots * (4.6 / 2)) + slots * 4.6 + launch
    filled = unmeasured.multiprocessors or 1
    assert launched(unmeasured, filled, 1, 4) == launched(unmeasured, filled, 1, 0)


def test_a_walks_issue_latency_follows_the_warps_its_steps_spread_over():
    """A GH200 program of two warpgroups walking a row doubled a step, its trips its data's (a while
    loop's), one program a multiprocessor (`emission.warps_of`, `cost.latency_bound`): one warpgroup walking
    S steps beside an idle one issues over 4 warps; two walking S steps each (S twice over, every consumer's
    steps stated, `cost.Stated.steps`) over 8; two splitting S over 8, each warp's chain half, the least
    bound."""
    target = generation.gpu_target("NVIDIA GH200 480GB")
    rates = cost.rates(target, device=False)
    row = jax.ShapeDtypeStruct((128,), jnp.float32)
    trips = jax.ShapeDtypeStruct((1,), jnp.int32)

    def priced(*, both: bool, steps: int) -> tuple[float, float]:
        def body(trips_ref, x_ref, o_ref):
            def step(index, row):
                mosaic.step_index(index)
                return row * 2.0

            def walk():
                o_ref[...] = lax.fori_loop(0, trips_ref[0], step, x_ref[...])

            if both:
                walk()
            else:
                pl.when(lax.axis_index("wg") == 0)(walk)

        launch = plgpu.kernel(body, out_type=row, num_threads=2, thread_name="wg")
        traced = jax.make_jaxpr(emission.costed(launch, Stated(steps=steps, bytes_accessed=0)))(trips, row)
        (kernel,) = [equation for equation in programs.equations(traced.jaxpr)
                     if "cost_estimate" in equation.params]
        warps = emission.warps_of(kernel)
        work = emission.work_of(kernel)
        assert warps is not None and work is not None
        return warps, cost.latency_bound(work, rates, target, held=warps)

    one, alone = priced(both=False, steps=64)
    both, twice = priced(both=True, steps=128)
    split, halved = priced(both=True, steps=64)
    assert (one, both, split) == (4, 8, 8), (one, both, split)
    # Each warpgroup's work outside its walk (a guard's cast) shares its program's warps, so the bounds
    # differ by it; their steps' chains order them (the exact arithmetic, `test_a_walk_issuing_one_warp_*`).
    assert halved < twice and halved < alone, (alone, twice, halved)


def test_a_launch_states_the_residency_its_plan_compiled_it_for():
    """A Mosaic GPU launch compiled for two programs a multiprocessor (`mosaic.compiler_params(resident=2)`,
    its `min_blocks_per_sm`) states it (`emission.resident_of`), and a program's launches price at it by
    default (`emission.launches_seconds`); one compiled for none states one (row 4123: the KDA recurrence's
    launches priced at one resident where their plans ran two, M-FS-73b's join median 0.65 against 0.98)."""
    from dew.nn.kernels.templates import mosaic as mosaic_templates

    target = generation.gpu_target("NVIDIA GH200 480GB")
    row = jax.ShapeDtypeStruct((128,), jnp.float32)

    def doubled(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    def launched(params):
        launch = plgpu.kernel(doubled, out_type=row, num_threads=1, thread_name="wg",
                              compiler_params=params)
        counted = emission.costed(launch, Stated(steps=0, bytes_accessed=1024))
        traced = jax.make_jaxpr(counted)(row).jaxpr
        (kernel,) = [equation for equation in programs.equations(traced)
                     if "cost_estimate" in equation.params]
        return traced, kernel

    traced, kernel = launched(mosaic_templates.compiler_params(resident=2))
    assert emission.resident_of(kernel) == 2
    work = emission.work_of(kernel)
    assert work is not None
    assert emission.launches_seconds(traced, target) == cost.launch_seconds(
        work, emission.launch_programs(kernel), target, resident=2, warps=emission.warps_of(kernel) or 0.0)
    assert emission.resident_of(launched(None)[1]) == 1


def test_a_programs_launches_price_in_sequence_and_an_uncounted_one_leaves_it_unknown():
    """A program of two costed launches (`emission.launches_seconds`) prices each at its own launch's seconds
    (`cost.launch_seconds` of its work, programs and warps), their sum, through a call alike; one launch its
    emission does not count, or launches under a scan (the review of lane/toolkit 61dd3c7d2), leave the
    program's price unknown, never 0."""
    target = generation.gpu_target("NVIDIA GH200 480GB")
    row = jax.ShapeDtypeStruct((128,), jnp.float32)

    def doubled(x_ref, o_ref):
        o_ref[...] = lax.fori_loop(0, 8, lambda index, value: value * 2.0, x_ref[...])

    launch = plgpu.kernel(doubled, out_type=row, num_threads=1, thread_name="wg")
    counted = emission.costed(launch, Stated(steps=8, bytes_accessed=1024))
    traced = jax.make_jaxpr(lambda x: counted(counted(x)))(row).jaxpr
    kernels = [equation for equation in programs.equations(traced) if "cost_estimate" in equation.params]
    each = [cost.launch_seconds(emission.work_of(kernel), emission.launch_programs(kernel), target,
                                warps=emission.warps_of(kernel) or 0.0) for kernel in kernels]
    assert len(kernels) == 2 and emission.launches_seconds(traced, target) == each[0] + each[1], each
    def halved(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 0.5

    # A launch of a body no costed launch traced (one that did answers with its count, `work_of`).
    alone = plgpu.kernel(halved, out_type=row, num_threads=1, thread_name="wg")
    uncounted = jax.make_jaxpr(lambda x: alone(counted(x)))(row).jaxpr
    assert emission.launches_seconds(uncounted, target) is None
    # Through a call, priced as its launches; under a scan, whose trips no sequence states, unknown.
    called = jax.make_jaxpr(jax.jit(lambda x: counted(counted(x))))(row).jaxpr
    assert emission.launches_seconds(called, target) == each[0] + each[1]
    scanned = jax.make_jaxpr(lambda x: lax.scan(lambda carry, _: (counted(carry), None), x, None,
                                                length=4)[0])(row).jaxpr
    assert emission.launches_seconds(scanned, target) is None


@pytest.mark.parametrize(("generation_name", "lanes"), [("sm80", 64), ("sm86", 128), ("sm89", 128),
                                                        ("sm90", 128), ("sm100", 128), ("sm120", 128)])
def test_an_fp32_instruction_takes_a_lane_a_clock_a_fused_pair_one(generation_name, lanes):
    """A CUDA multiprocessor's fp32 lanes each run one instruction a clock, an FFMA's two FLOPs or a lone
    add's, maximum's or select's one (`Architecture.vector_per_clock` its lanes' FLOPs, two a lane): 1024
    lone operations take 1024 lane-clocks, and 1024 multiplies with their 1024 fused adds the same, where the
    lanes' FLOP rate priced the lone ones at half (M-FS-70: the KDA intra launch's FMNMX, a quarter of its
    instructions, none fused)."""
    target = generation.gpu_target(generation_name)
    rates = cost.rates(target, device=False)
    alone = cost.units(Work(operations=1024), rates)[0][1]
    paired = cost.units(Work(operations=1024, fused_multiply_adds=1024), rates)[0][1]
    assert alone == paired == 1024 / (lanes * target.loaded_clock_hz), (alone, paired)


def test_a_tpus_vector_units_take_a_fused_pairs_add_as_an_operation_of_its_own():
    """A TPU's VPU rate is calibrated at two operations a multiply-add (`generation._TPUS`), so a pair's add
    (`Work.fused_multiply_adds`) is an operation there beside its multiply, where a CUDA multiprocessor's
    lanes run it within the multiply's FFMA (the review of lane/toolkit 5cbbe66c586, P1)."""
    rates = cost.rates(generation.TARGETS["v5e"], device=False)
    paired = cost.units(Work(operations=1024, fused_multiply_adds=1024), rates)[0][1]
    assert paired == 2048 / rates.cuda, paired


def _multiply_add_or_adds(row, index):
    """A conditional its data decides: a multiply-add an element, or two lone adds an element."""
    return lax.switch(index, [lambda row: row * row + row, lambda row: row + row + row], row)


def test_a_tpu_counts_a_bodys_multiply_adds_as_its_vector_units_run_them():
    """A TPU's VPU runs a multiply and its add as two operations (`generation._TPUS`' calibration), so its
    count fuses none: a conditional of 128 multiply-adds or 256 lone adds keeps 256 operations, where a
    fused branch's credit beside the other's adds priced 384 (the review of lane/toolkit c9307c11146, P1)."""
    with generation.assumed("v5e"):
        fixed = _emitted(_multiply_add_or_adds, _ROW, jax.ShapeDtypeStruct((), jnp.int32)).roles[0].fixed
    assert (fixed.operations, fixed.fused_multiply_adds) == (256, 0), fixed


def _issued_branches(*branches) -> float:
    """The issue slots of a conditional its data decides among `branches`, each of a row and 64 integers."""
    def branched(row, bits, index):
        return lax.switch(index, list(branches), row, bits)

    return cost.issued(_emitted(branched, _ROW, jax.ShapeDtypeStruct((64,), jnp.int32),
                                jax.ShapeDtypeStruct((), jnp.int32)).roles[0].fixed)


def test_an_unknown_branch_takes_the_issue_slots_its_costliest_branch_does():
    """A conditional its data decides, one branch 128 fp32 adds, the other 64 integer xors, issues one
    branch's instructions: the costlier's slots, as a conditional of that branch beside one doing nothing,
    where its units' most taken unit by unit counted both mixes (the review of lane/toolkit c9307c11146,
    P2)."""
    added, xored, kept = (lambda row, bits: (row + row, bits)), (lambda row, bits: (row, bits ^ bits)), (
        lambda row, bits: (row, bits))
    both = _issued_branches(added, xored)
    assert both == max(_issued_branches(added, kept), _issued_branches(kept, xored)), both


def test_an_unknown_branch_prices_the_most_lane_instructions_any_branch_issues():
    """A conditional its data decides, one branch 128 multiply-adds an FFMA each, the other 256 lone adds:
    its lanes run the costlier's 256 instructions, which a credit for the first's pairs, taken unit by unit
    over the branches' most, would cut to 128 (the review of lane/toolkit 5cbbe66c586, P1)."""
    fixed = _emitted(_multiply_add_or_adds, _ROW, jax.ShapeDtypeStruct((), jnp.int32)).roles[0].fixed
    rates = cost.rates(generation.gpu_target("sm90"), device=False)
    assert cost.units(fixed, rates)[0][1] == 256 / rates.cuda, fixed


def test_a_multiprocessor_issues_four_warps_instructions_a_clock_whichever_units_run_them():
    """An H100's four warp schedulers each issue one instruction a clock (CUDA C++ Programming Guide §16.8.1),
    128 threads' a clock: 1024 fp32 operations beside 512 integer ones take 1536 threads' issue slots, 12
    clocks, where each unit alone takes 8 (the lanes' 128 a clock, the ALU's 64); the whole device's schedule
    (XLA's fusion, its units in series) holds no issue slot of its own."""
    target = generation.gpu_target("sm90")
    work = Work(operations=1024, integer_operations=512)
    seconds = cost.units(work, cost.rates(target, device=False))[0]
    assert max(seconds) == 1536 / (128 * target.loaded_clock_hz), seconds
    device = sum(cost.units(work, cost.rates(target, device=True))[0])
    per_device = target.loaded_clock_hz * target.multiprocessors
    assert device == 1024 / (128 * per_device) + 512 / (64 * per_device), device


@pytest.mark.parametrize("generation_name", ["sm75", "sm80", "sm86", "sm89", "sm90", "sm100", "sm120"])
@pytest.mark.parametrize("field", ["integer_operations", "integer_multiply_adds"])
def test_a_gpus_integer_work_is_priced_on_every_cuda_generation(field, generation_name):
    """128 integer operations (the ALU's add, shift, logic and comparison) or integer multiply-adds (the FMA
    pipe's IMAD) cost a multiprocessor time on every CUDA generation from Turing to consumer Blackwell, none
    of them left at no rate: the CUDA C++ Best Practices Guide 13.5's §12.1.1 Table 5 states 64 a clock for
    shift, logic and mad.lo.s32 on compute capabilities 7.5 to 12.1 alike, where a B200's and an RTX PRO
    6000's were unread, their packed NVFP4 decode and integer multiplies priced at nothing (rows 11806,
    11843)."""
    rates = cost.rates(generation.gpu_target(generation_name), device=False)
    seconds, unpriced = cost.units(Work(**{field: 128}), rates)
    assert sum(seconds) > 0 and not any(unpriced), (seconds, unpriced)


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


def test_launch_seconds_shares_sm_units_overlaps_paths_and_prices_integer_work():
    """`cost.launch_seconds` (rows 10793, 10809, 10810):
    1. Integer-only `Work` at `programs = multiprocessors + 1` and `resident = 1` prices launch latency
       plus two waves of per-program integer work at one SM's integer rate.
    2. Co-resident programs (`resident = 2`) share one SM's throughput units (`ceil(n / SMs)` times
       per-program unit time, not divided by `resident`) while overlapping dependent paths (`ceil(n /
       (SMs * resident))` waves of `path`)."""
    target = generation.gpu_target("sm90")
    sms = target.multiprocessors
    assert sms is not None
    launch = target.launch_seconds or 0.0
    sm_rates = cost.rates(target, device=False)
    assert sm_rates.integer is not None and sm_rates.tensor is not None

    # 1. Integer-only work across sms + 1 programs (two waves on the busiest SM).
    total_int = 1e11
    programs = sms + 1
    expected_int = 2 * ((total_int / programs) / sm_rates.integer) + launch
    got_int = cost.launch_seconds(Work(integer_operations=total_int, kernels=1.0), programs, target,
                                  resident=1)
    assert got_int == expected_int, (got_int, expected_int)

    # 2. Throughput-bound work: 176 programs on 132 SMs takes 2 programs on the busiest SM even at
    #    resident=2 (co-resident programs share one SM's tensor cores, not 2x peak).
    total_flops = 1e12
    expected_tput = 2 * ((total_flops / 176) / sm_rates.tensor) + launch
    got_tput_r1 = cost.launch_seconds(Work(flops=total_flops, kernels=1.0), 176, target, resident=1)
    got_tput_r2 = cost.launch_seconds(Work(flops=total_flops, kernels=1.0), 176, target, resident=2)
    assert got_tput_r1 == expected_tput, (got_tput_r1, expected_tput)
    assert got_tput_r2 == expected_tput, (got_tput_r2, expected_tput)

    # 3. Latency/path-bound work: resident=2 overlaps the two programs on the busiest SM into 1 wave.
    path_work = Work(span=1e4, kernels=1.0)
    got_path_r1 = cost.launch_seconds(path_work, 176, target, resident=1)
    got_path_r2 = cost.launch_seconds(path_work, 176, target, resident=2)
    assert got_path_r2 < got_path_r1, (got_path_r2, got_path_r1)


@pytest.mark.parametrize("memory", ["global", "shared"])
def test_an_atomic_add_counts_its_bytes_reduced_where_global_and_an_a100_prices_them(memory):
    """A launch of `SMs + 1` programs, each adding a 64 x 64 fp32 tile into memory an element at a time
    (`plgpu.atomic_add`): into global memory (`red.global`) the launch counts every program's 16 KiB among its
    global reductions (`Work.global_reductions`), which an A100 prices at its fitted 3 bytes a multiprocessor
    a clock (M-NA-9), two programs' worth on its busiest multiprocessor whatever their residency, and an H100
    leaves unpriced; into shared memory (`red.shared::cta`) none, its bytes shared ones."""
    ampere, hopper = generation.gpu_target("sm80"), generation.gpu_target("sm90")
    assert ampere.multiprocessors is not None
    launched = ampere.multiprocessors + 1
    tile = jax.ShapeDtypeStruct((64, 64), jnp.float32)

    def body(x_ref, o_ref):
        if memory == "global":
            plgpu.atomic_add(o_ref, x_ref[...])
        else:
            def scoped(smem):
                plgpu.atomic_add(smem, x_ref[...])
                o_ref[...] = smem[...]

            pl.run_scoped(scoped, plgpu.SMEM((64, 64), jnp.float32))

    launch = plgpu.kernel(body, out_type=tile, grid=(launched,), grid_names=("programs",))
    traced = jax.make_jaxpr(emission.costed(launch, Stated(steps=0, bytes_accessed=0)))(tile)
    (kernel,) = [equation for equation in programs.equations(traced.jaxpr)
                 if "cost_estimate" in equation.params]
    work = emission.work_of(kernel)
    if memory == "shared":
        assert work.global_reductions == 0 and work.shared_bytes == launched * 64 * 64 * 4
        return
    assert work.global_reductions == launched * 64 * 64 * 4
    reductions = cost.rates(ampere, device=False).reductions
    assert reductions == 3.0 * ampere.loaded_clock_hz
    busiest = 2 * ((work.global_reductions / launched) / reductions) + (ampere.launch_seconds or 0.0)
    for resident in (1, 2):
        priced = cost.launch_seconds(work, launched, ampere, resident=resident)
        assert priced == busiest, (resident, priced, busiest)
    assert cost.rates(hopper, device=False).reductions is None
    assert work.global_reductions in cost.units(work, cost.rates(hopper, device=False))[1]
