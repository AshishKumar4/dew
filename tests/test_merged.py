"""A split decode step's slices merged where its launch wrote them (`merged.merged_padded`): its
derivatives the law's path's (`merged._value`), which its JVP rule takes; it and its state on par with the
plain merge from float64, here and on a GPU's one launch (tests/test_gpu_attention.py)."""
import functools
import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import plain_merged
from generation_support import COMPILES_ON_CPU
from kernel_compile import compile_kernels
from reference_error import FEWEST_SEEDS, as_given, assert_as_exact, dew_path

from dew.algebra import windows
from dew.nn import programs
from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_algebra as algebra, merged

SPECS = {"natural": algebra.finished_form(algebra.Forward("full", 3, ids=False, scale=1.0, natural=True)),
         "fold": algebra.finished_form(algebra.Forward("full", 3, ids=False, scale=0.125, natural=False,
                                                       subtracted=True))}
"""A decode merge's two forms: the accurate form's natural maxima, and the raw fold a decode step's split
selects (`attention_decode._spec`: its raw products' maxima, `subtracted`, in its fold's units), each
qualified against its own float64 reference."""


QUERIES, WIDTH = 4, 64
"""A drawn split's real rows and columns, within its launch's padding (`_drawn`)."""


def _drawn(seed):
    """Padded split slices `[2, 8, 8, 4, 64]` (8 slices, `QUERIES` of 8 rows real), their maxima and
    denominators stacked `[2, 2, 4, 8, 8]` with a slice of zero total in each row, a direction for each and a
    cotangent of the merged output."""
    rng = np.random.default_rng([7124, seed])
    padded = rng.standard_normal((2, 8, 8, 4, WIDTH)).astype(np.float32)
    stacked = np.stack([rng.standard_normal((2, 4, 8, 8)), np.abs(rng.standard_normal((2, 4, 8, 8)))])
    stacked = stacked.astype(np.float32)
    stacked[1, :, :, 3, :] = 0.0
    along = tuple(rng.standard_normal(x.shape).astype(np.float32) for x in (padded, stacked))
    return padded, stacked, *along, rng.standard_normal((2, QUERIES, 4, WIDTH)).astype(np.float32)


def _plain(spec):
    """The plain merge (`attention_oracle.plain_merged`) at a drawn split's rows and columns: check 1's
    reference, independent of the law."""
    return functools.partial(plain_merged, rate=_rate(spec), queries=QUERIES, width=WIDTH)


def _top(stacked):
    """Each row's largest maximum `[B, Q, H]` over its slices, held: the anchor a state's mass is read at."""
    return jax.lax.stop_gradient(jnp.max(jnp.moveaxis(stacked[0][..., :QUERIES], 1, 3), axis=1))


def _rate(spec) -> float:
    """The factor a maximum's units take to natural ones."""
    return 1.0 if spec.natural else spec.scale


def _plain_state(spec):
    """The plain merge's state: `_plain`'s output beside the slices' mass at their largest maximum (`_top`),
    check 1's reference for a merge's state, independent of the law."""
    output = _plain(spec)

    def state(padded, stacked):
        maxima, totals = (jnp.moveaxis(x[..., :QUERIES], 1, 3) for x in (stacked[0], stacked[1]))
        mass = jnp.sum(jnp.exp((maxima - _top(stacked)[:, None]) * _rate(spec)) * totals, axis=1)
        return output(padded, stacked), mass
    return state


def state_orders(state):
    """`state`'s output and mass, the output's gradient as a cotangent's pullback and that gradient's
    derivative along the draw's directions, of one draw's operands: a merge's returned state with the orders
    a training step and a second-order method take."""
    def orders(padded, stacked, along_padded, along_stacked, cotangent):
        def loss(padded, stacked):
            return jnp.vdot(cotangent, state(padded, stacked)[0])

        gradient = jax.grad(loss, argnums=(0, 1))
        return (*state(padded, stacked), *gradient(padded, stacked),
                *jax.jvp(gradient, (padded, stacked), (along_padded, along_stacked))[1])
    return orders


def finished_state(merge, spec):
    """`merge`'s state as `_plain_state` gives it: its output finished, normalized, and its total read at the
    slices' largest maximum, the meaning of its maximum and total together, whichever anchor it lands at."""
    finished = _finished(merge, spec)

    def state(padded, stacked):
        _, maximum, total = merge(padded, stacked)[:3]
        return finished(padded, stacked), total * jnp.exp((maximum - _top(stacked)) * _rate(spec))
    return state


def _finished(merge, spec):
    """`merge`'s finished output, normalized (`attention_algebra.finish`)."""
    def finished(padded, stacked):
        return algebra.finish(spec, algebra.Partial(*merge(padded, stacked)), initial=True,
                              normalized=True)[0]
    return finished


def _merges(form):
    spec = SPECS[form]
    return (spec, lambda padded, stacked: tuple(merged.merged_padded(spec, padded, stacked, queries=QUERIES,
                                                                     width=WIDTH)),
            functools.partial(merged._value, units=algebra.units_of(spec), queries=QUERIES, width=WIDTH))


@pytest.mark.parametrize("argnums", [0, 1], ids=["outputs", "states"])
@pytest.mark.parametrize("form", sorted(SPECS))
def test_a_padded_merges_gradient_along_one_operand_alone_is_on_par(form, argnums):
    """The padded merge's finished output's pullback along its outputs alone and along its states alone
    (the other operand's tangent a symbolic zero) on par with the plain merge's (`_plain`, JAX's
    differentiation of it) against float64 (check 1), jitted. Asked for one operand alone, its declared
    tangent instantiated the other's zero and the law's ratio refused it, a TypeError (the base review of
    9bc8a9da1, F1)."""
    spec, ours, _ = _merges(form)

    def alone(finished):
        def pullback(padded, stacked, along_padded, along_stacked, cotangent):
            del along_padded, along_stacked
            return jax.grad(lambda *operands: jnp.vdot(cotangent, finished(*operands)), argnums=argnums)(
                padded, stacked)
        return pullback

    assert_as_exact(jax.jit(alone(_finished(ours, spec))), as_given(alone(_plain(spec))), _drawn,
                    f"a padded merge's pullback along one operand, {form}", seeds=MERGE_SEEDS)


def _held_as_each_alone(program, each_alone, operands: tuple, label: str) -> None:
    """A merged state `program` of `operands` (on a mesh, or batched) held to each row's merge run alone
    (`each_alone`, one of Dew's own paths): its output and total on par from float64 (check 1; two programs,
    which XLA promises no common bits, the merge P0's base review of aeae18f07, F5), and its maxima, which a
    merge moves and never rounds, the same bit for bit."""
    def value(*args):
        return tuple(program(*args)[i] for i in (0, 2))

    assert_as_exact(value, dew_path(lambda *args: tuple(each_alone(*args)[i] for i in (0, 2))),
                    lambda seed: operands, label, seeds=FEWEST_SEEDS)
    maxima = (program(*operands)[1], jax.jit(each_alone)(*operands)[1])
    assert np.asarray(maxima[0]).tolist() == np.asarray(maxima[1]).tolist()


@pytest.mark.mesh(devices=2)
def test_a_padded_merge_of_states_varying_alone_carries_their_axes():
    """The padded merge in `jax.shard_map` over two devices whose outputs are replicated and whose states
    (maxima and totals) are each device's own: its results vary along the mesh's axis as the states do, so a
    scan whose carry starts as the law's output takes the merge's. Typed from the outputs alone, the merged
    output lost the states' varying axis (the base review of 7bf35ac55, F1: totals [1, 1] and [1, 3] beside
    outputs [0, 1])."""
    from jax.sharding import AxisType, PartitionSpec

    _, ours, law = _merges("natural")
    padded, stacked, *_ = _drawn(0)
    mesh = jax.make_mesh((2,), ("rows",), axis_types=(AxisType.Auto,), devices=jax.devices()[:2])

    def scanned(padded, stacked):
        # A carry that starts as a value of each device's own states, as the merge's results are: the
        # law's state plus nothing of the device's totals (varying along the mesh as they do).
        own = jnp.sum(stacked[1]) * 0
        start = tuple(value + own.astype(value.dtype) for value in law(padded, stacked)[:3])
        return jax.lax.scan(lambda carry, _: (tuple(ours(padded, stacked)[:3]), None), start, length=1)[0]

    # One row's outputs on both devices, each device its own row of states.
    replicated, states = PartitionSpec(), PartitionSpec(None, "rows")
    inner = jax.shard_map(scanned, mesh=mesh, in_specs=(replicated, states),
                          out_specs=(PartitionSpec("rows"),) * 3)

    # An automatic mesh's jit places the caller's arrays, whatever placement they have.
    mapped = jax.jit(inner)

    whole = padded[:1]

    def each_alone(whole, stacked):
        return tuple(jnp.concatenate(parts) for parts in zip(*(
            tuple(ours(whole, stacked[:, k:k + 1])[:3]) for k in range(2)), strict=True))

    _held_as_each_alone(mapped, each_alone, (whole, stacked), "a padded merge of states varying alone")


@pytest.mark.mesh(devices=2)
def test_a_padded_merge_on_two_devices_is_each_rows_own():
    """The padded merge on two devices, its operands split along their batch rows: in `jax.shard_map`, its
    results' types the law's (`merged._value`'s, each varying along the mesh's axis as its operands are), so
    a scan whose carry starts as the law's output and is the merge's output takes it; and in a jit of
    operands resharded to their specs; each row's state, both ways, the row's merge run alone. Its result
    types made fresh were invariant, and the scan was refused, its manual axes unmatched (the base review of
    9eb04fc04, F1)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    _, ours, law = _merges("natural")
    padded, stacked, *_ = _drawn(0)
    mesh = jax.make_mesh((2,), ("rows",), axis_types=(AxisType.Auto,), devices=jax.devices()[:2])

    def state(padded, stacked):
        return tuple(ours(padded, stacked)[:3])

    def scanned(padded, stacked):
        return jax.lax.scan(lambda carry, _: (state(padded, stacked), None), tuple(law(padded, stacked)[:3]),
                            length=1)[0]

    rows, columns = PartitionSpec("rows"), PartitionSpec(None, "rows")
    inner = jax.shard_map(scanned, mesh=mesh, in_specs=(rows, columns), out_specs=(rows,) * 3)

    # Each operand resharded to its spec in the program, whatever placement the caller's arrays have.
    # An automatic mesh's jit places the caller's arrays, whatever placement they have.
    mapped = jax.jit(inner)
    placed = jax.jit(state, in_shardings=(NamedSharding(mesh, rows), NamedSharding(mesh, columns)))

    def each_alone(padded, stacked):
        return tuple(jnp.concatenate(parts) for parts in zip(*(
            state(padded[k:k + 1], stacked[:, k:k + 1]) for k in range(len(padded))), strict=True))

    for name, program in (("shard_map", mapped), ("placed", placed)):
        _held_as_each_alone(program, each_alone, (padded, stacked), f"a padded merge on two devices, {name}")


def test_a_padded_merge_batched_is_each_batchs_own():
    """The padded merge mapped over a leading batch of splits (`jax.vmap`, its batching rule one launch of
    them), jitted: each split's state on par with its merge run alone (`_held_as_each_alone`)."""
    _, ours, _ = _merges("fold")

    def state(padded, stacked):
        return tuple(ours(padded, stacked)[:3])

    def each_alone(padded, stacked):
        return tuple(jnp.stack(parts) for parts in zip(*(
            state(padded[k], stacked[k]) for k in range(len(padded))), strict=True))

    splits = [_drawn(seed)[:2] for seed in range(3)]
    operands = tuple(np.stack(operand) for operand in zip(*splits, strict=True))
    _held_as_each_alone(jax.jit(jax.vmap(state)), each_alone, operands, "a padded merge batched")


def test_a_padded_merge_batched_over_no_splits_is_empty():
    """The padded merge mapped over a batch of no splits (`jax.vmap` at size 0), jitted: a state of no
    entries a split, each split's rows kept (the merge P0's base review of aeae18f07, F2: the batch's
    unfold named both its size, 0, and an inferred extent, and raised)."""
    _, ours, _ = _merges("fold")

    def state(padded, stacked):
        return tuple(ours(padded, stacked)[:3])

    padded, stacked = _drawn(0)[:2]
    empty = jax.jit(jax.vmap(state))(padded[None][:0], stacked[None][:0])
    alone = jax.eval_shape(state, padded, stacked)
    assert [got.shape for got in empty] == [(0, *split.shape) for split in alone]


MERGE_SEEDS = 32
"""The padded merge's check-1 draws, sized by `seeds_for` (DEW_CHECK1_QUALIFY)."""


def test_a_split_merged_with_a_subnormal_initial_mass_keeps_its_output():
    """A split's two slices of outputs 1 and 0, totals 2^-126 and 0, merged (`merged.merged_split`) with an
    initial state of total and sums 2^-127, all at one maximum: every state that holds mass has output 1, so
    the merge's is 1, where the initial state's output read as zero beside the mass the law weighs, its
    total compared on a lane that flushes it, and the merge was 2/3 (the merge P0's head review 3)."""
    spec = SPECS["natural"]
    padded = jnp.zeros((1, 2, 1, 1, 1), jnp.float32).at[0, 0].set(1.0)
    stacked = jnp.stack([jnp.zeros((1, 1, 2, 1), jnp.float32),
                         jnp.asarray([2.0 ** -126, 0.0], jnp.float32).reshape(1, 1, 2, 1)])
    tiny = 2.0 ** -127
    initial = template.Initial(jnp.zeros((1, 1, 1), jnp.float32), jnp.full((1, 1, 1), tiny, jnp.float32),
                               jnp.full((1, 1, 1, 1), tiny, jnp.float32))
    state = merged.merged_split(spec, padded, stacked, queries=1, width=1, initial=initial)
    out = algebra.finish(spec, state, initial=True, normalized=True)[0]
    assert float(out.reshape(())) == 1.0


CHECK1_SHARDS = {
    "test_a_padded_merge_and_its_derivatives_are_on_par[natural-op-by-op]": {"seeds": 32, "per_task": 8},
}
"""The cells whose draws run as CI's map tasks (`check1_shards`), by node id: the natural merge's 32 draws
op by op, each derivative order's every operation a dispatch of its own, took 123 s in one task in train
1191's CI, 8 a task (row 11921)."""


def _merge_cases() -> list:
    """Every form's merge jitted and op by op as parameters, each marked from `CHECK1_SHARDS` by its node
    id (`form-run`, its run's spaces hyphens, as a site's id holds none)."""
    test = "test_a_padded_merge_and_its_derivatives_are_on_par"
    marked = {name: [pytest.mark.check1_shards(**counts)] for name, counts in CHECK1_SHARDS.items()}
    cases = [(form, run, f"{form}-{run.replace(' ', '-')}") for form in sorted(SPECS)
             for run in ("jitted", "op by op")]
    return [pytest.param(form, run, id=named, marks=marked.get(f"{test}[{named}]", []))
            for form, run, named in cases]


@pytest.mark.parametrize(("form", "run"), _merge_cases())
def test_a_padded_merge_and_its_derivatives_are_on_par(form, run):
    """The padded merge's finished output, its state's mass, the output's gradient and that gradient's
    derivative along a direction on par with the plain merge's (`_plain_state`, JAX's differentiation of it,
    independent of the law) against float64 (check 1), jitted or op by op (`as_given`), each a case (both
    in one took 225 s over the two forms), the reference run op by op."""
    spec, ours, _ = _merges(form)
    reference = as_given(state_orders(_plain_state(spec)))
    orders = state_orders(finished_state(ours, spec))
    program = jax.jit(orders) if run == "jitted" else as_given(orders)
    assert_as_exact(program, reference, _drawn, f"a padded merge, {form}, {run}", seeds=MERGE_SEEDS)


@COMPILES_ON_CPU
@pytest.mark.parametrize("cases", ["widths-narrow", "widths-wide", "shapes-finished", "shapes-long"])
@pytest.mark.parametrize("name", ["sm80", "sm90", "sm100"])
def test_a_padded_merge_compiles_as_one_launch(name, cases):
    """The padded merge's one Mosaic GPU launch (`merged._launch_mosaic`, row 9795) compiles spill-free for
    compile-only Ampere, Hopper and Blackwell GPUs in one process per GPU across: value widths 80, 96, 100,
    128 and 192 (predicated coalesced GMEM loads/stores); a `jax.vmap` batch of three splits (beside nothing
    but the one transpose that folds it, every launch typed by `assembled.launched_thunks`); `merged_finished`
    with and without `Initial`; and long splits of 64 and 256 slices (row 8439)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    def one_case(label: str, *, slices: int, width: int) -> KernelCase:
        padded = jax.ShapeDtypeStruct((2, slices, 16, 8, width), jnp.float32)
        stacked = jax.ShapeDtypeStruct((2, 2, 8, slices, 16), jnp.float32)
        merge = functools.partial(merged.merged_padded, SPECS["natural"], queries=4, width=width)
        return KernelCase(
            label, lambda merge=merge, padded=padded, stacked=stacked: (merge, (padded, stacked)))
    batched_padded = jax.ShapeDtypeStruct((3, 2, 16, 16, 8, 64), jnp.float32)
    batched_stacked = jax.ShapeDtypeStruct((3, 2, 2, 8, 16, 16), jnp.float32)
    batched = jax.vmap(lambda padded, stacked: tuple(merged.merged_padded(SPECS["natural"], padded, stacked,
                                                                          queries=4, width=64)))
    finished_padded = jax.ShapeDtypeStruct((1, 16, 16, 8, 128), jnp.float32)
    finished_stacked = jax.ShapeDtypeStruct((2, 1, 8, 16, 16), jnp.float32)
    finished = functools.partial(merged.merged_finished, SPECS["natural"], queries=4, width=128,
                                 out_dtype=jnp.bfloat16, out_heads=32)
    init_m = jax.ShapeDtypeStruct((1, 8, 4), jnp.float32)
    init_t = jax.ShapeDtypeStruct((1, 8, 4), jnp.float32)
    init_acc = jax.ShapeDtypeStruct((1, 8, 4, 128), jnp.float32)

    def finished_with_initial(queries: int, width: int, out_dtype, out_heads: int):
        def _run(p, s, im, it, ia):
            return merged.merged_finished(
                SPECS["natural"], p, s, queries=queries, width=width, out_dtype=out_dtype,
                out_heads=out_heads, initial=template.Initial(im, it, ia))
        return _run

    # Row 11857 shape: 2 slices + Initial, width 16 (single chunk, sub-warp channel block).
    small_p = jax.ShapeDtypeStruct((1, 2, 4, 1, 16), jnp.float32)
    small_s = jax.ShapeDtypeStruct((2, 1, 1, 2, 4), jnp.float32)
    small_m = jax.ShapeDtypeStruct((1, 1, 2), jnp.float32)
    small_a = jax.ShapeDtypeStruct((1, 1, 2, 16), jnp.float32)

    named = {
        "widths-narrow": [one_case(f"width-{width}", slices=16, width=width) for width in (80, 96, 100)],
        "widths-wide": [one_case(f"width-{width}", slices=16, width=width) for width in (128, 192)],
        "shapes-finished": [
            KernelCase("batched", lambda: (batched, (batched_padded, batched_stacked))),
            KernelCase("finished", lambda: (finished, (finished_padded, finished_stacked))),
            KernelCase("finished-initial",
                       lambda: (finished_with_initial(4, 128, jnp.bfloat16, 32),
                                (finished_padded, finished_stacked, init_m, init_t, init_acc))),
        ],
        "shapes-long": [
            KernelCase("small-finished-initial",
                       lambda: (finished_with_initial(2, 16, jnp.float32, 2),
                                (small_p, small_s, small_m, small_m, small_a))),
            *(one_case(f"slices-{slices}", slices=slices, width=128) for slices in (64, 256)),
        ],
    }
    with generation.assumed(name):
        records = compile_mosaic_kernels(named[cases], name)
    assert not mosaic_failures(records), "\n".join(mosaic_failures(records))
    for record in records:
        assert "error" not in record, record
        expected = (["thunk", "wrapped_transpose"] if record["name"] == "batched" else ["thunk"])
        assert sorted(record["thunks"]) == expected, record
        assert record["modules"] and record["modules"].items() <= record["certified"].items(), record
        kernels = record["kernels"]
        assert len(kernels) == 1 and kernels[0]["kernel"] == "merged_padded", record
        assert kernels[0]["spilled"] == 0 and kernels[0]["shared"] == 0, record

@COMPILES_ON_CPU
def test_padded_merge_routes_nvidia_mosaic_and_keeps_triton_fallback():
    """Row 9795: `_uses_mosaic` selects Mosaic GPU on NVIDIA sm80+ CUDA families and keeps the Pallas Triton
    `_launch` fallback on pre-Ampere CUDA GPUs (`platform="cuda"`)."""
    from fla_gdn_support import KernelCase

    for gen in ("sm80", "sm90", "sm100", "sm120"):
        assert merged._uses_mosaic(generation.gpu_target(gen)), gen
    assert not merged._uses_mosaic(generation.gpu_target("sm75"))
    padded = jax.ShapeDtypeStruct((2, 16, 16, 8, 128), jnp.float32)
    stacked = jax.ShapeDtypeStruct((2, 2, 8, 16, 16), jnp.float32)
    triton_fn = functools.partial(
        merged._launch, units=algebra.units_of(SPECS["natural"]), queries=4, width=128, block=128)
    from jax._src.pallas.mosaic_gpu import core as plgpu_core, lowering as mosaic_lowering
    from jax.extend import core as jcore

    from dew.nn.kernels import launches

    launches._register_mosaic_gpu_lowerings()
    assert jcore.primitives.platform_index_p not in mosaic_lowering.mosaic_lowering_rules[
        plgpu_core.WGxWG_SEMANTICS]
    with generation.assumed("sm90"):
        records = compile_kernels([KernelCase("triton-fallback", lambda: (triton_fn, (padded, stacked)))],
                                  "sm90")
    assert len(records) == 1 and "error" not in records[0] and sorted(records[0]["thunks"]) == [
        "merged_padded"], records


def test_replicated_channel_merge_matches_value_with_subnormal_initial_mass():
    """Rows 9795, 11857: replicating row stats across `[chunk, block]` (the layout `_launch_mosaic` uses
    for zero-SMEM Mosaic GPU lowering) matches `merged._value` bit-for-bit on regular and subnormal
    (`2^-127`) initial-mass merges."""
    from dew.algebra import windows

    units = algebra.units_of(SPECS["natural"])
    rng = np.random.default_rng(7)
    slices, block, width = 6, 16, 12
    chunk = windows.chunk_width(slices + 1)
    chunks = -(-(slices + 1) // chunk)
    padded = rng.standard_normal((1, slices, 1, 1, width)).astype(np.float32)
    stacked = np.stack([rng.uniform(-4.0, 4.0, (1, 1, slices, 1)).astype(np.float32),
                        rng.uniform(0.1, 2.0, (1, 1, slices, 1)).astype(np.float32)])
    for label, subnormal in (("regular", False), ("subnormal-initial", True)):
        p = np.zeros_like(padded) if subnormal else padded
        s = np.stack([np.full((1, 1, slices, 1), -1e30, np.float32),
                      np.zeros((1, 1, slices, 1), np.float32)]) if subnormal else stacked
        init_m = np.full((1, 1, 1), 0.0, np.float32)
        init_t = np.full((1, 1, 1), np.float32(2.0 ** -127 if subnormal else 0.75), np.float32)
        init_acc = np.full((1, 1, 1, width), np.float32(2.0 ** -127 if subnormal else 1.25), np.float32)
        ref_acc, ref_m, ref_t = merged._value(
            jnp.asarray(p), jnp.asarray(s), jnp.asarray(init_m), jnp.asarray(init_t),
            jnp.asarray(init_acc), units=units, queries=1, width=width)
        raw_acc = jnp.pad(jnp.asarray(init_acc[0, 0, 0]), (0, block - width))
        raw_m = jnp.broadcast_to(jnp.asarray(init_m[0, 0, 0]), (block,))
        raw_t = jnp.broadcast_to(jnp.asarray(init_t[0, 0, 0]), (block,))
        init_st = algebra.initial_state(units, algebra.Partial(raw_acc, raw_m, raw_t))
        pad_row = jnp.pad(jnp.asarray(p[0, :, 0, 0, :]), ((0, chunks * chunk - slices), (0, block - width)))
        m_row = jnp.pad(jnp.asarray(s[0, 0, 0, :, 0]), (0, chunks * chunk - slices))
        t_row = jnp.pad(jnp.asarray(s[1, 0, 0, :, 0]), (0, chunks * chunk - slices))

        def load(c, pad_row=pad_row, m_row=m_row, t_row=t_row, init_st=init_st):
            start = jnp.int32(c) * chunk
            idx = start + jax.lax.broadcasted_iota(jnp.int32, (chunk, block), 0)
            o = jax.lax.dynamic_slice_in_dim(pad_row, start, chunk, 0)
            mx = jnp.broadcast_to(jax.lax.dynamic_slice_in_dim(m_row, start, chunk, 0)[:, None],
                                  (chunk, block))
            tot = jnp.broadcast_to(jax.lax.dynamic_slice_in_dim(t_row, start, chunk, 0)[:, None],
                                   (chunk, block))
            is_init = idx == slices
            return (jnp.where(is_init, init_st.acc, o), jnp.where(is_init, init_st.m, mx),
                    jnp.where(is_init, init_st.total, tot), idx < slices + 1)

        got = algebra.finished_merged_chunked(units, load, chunks)
        np.testing.assert_array_equal(np.asarray(got.acc[:width]), np.asarray(ref_acc[0, 0, 0]), label)
        np.testing.assert_array_equal(np.asarray(got.m[0]), np.asarray(ref_m[0, 0, 0]), label)
        np.testing.assert_array_equal(np.asarray(got.total[0]), np.asarray(ref_t[0, 0, 0]), label)
        if subnormal:
            np.testing.assert_array_equal(np.asarray(got.acc[:width]), np.ones(width, np.float32))

def _jaxpr_has_load(jaxpr) -> bool:
    """Whether `jaxpr` or any nested scan/while/cond executes a Pallas `get` load."""
    from jax import lax

    for eq in jaxpr.eqns:
        if eq.primitive.name == "get":
            return True
        if eq.primitive is lax.scan_p and _jaxpr_has_load(eq.params["jaxpr"].jaxpr):
            return True
        if eq.primitive is lax.while_p and _jaxpr_has_load(eq.params["body_jaxpr"].jaxpr):
            return True
        if eq.primitive is lax.cond_p and any(_jaxpr_has_load(b.jaxpr) for b in eq.params["branches"]):
            return True
    return False


def _eval_block_predicate(jaxpr, pred_var, block_id: int) -> bool | None:
    """Evaluate a scalar predicate depending on `program_id(axis=3)` at `block_id`, or None if dynamic."""
    from jax import lax

    env: dict[int, int | bool] = {}
    for eq in jaxpr.eqns:
        if eq.primitive.name == "program_id" and eq.params.get("axis") == 3:
            env[id(eq.outvars[0])] = block_id
            continue
        if eq.primitive in (lax.eq_p, lax.ne_p, lax.lt_p, lax.le_p, lax.gt_p, lax.ge_p):
            vals = [int(v.val) if hasattr(v, "val") else env.get(id(v)) for v in eq.invars]
            if all(val is not None for val in vals):
                a, b = vals
                ops = {lax.eq_p: a == b, lax.ne_p: a != b, lax.lt_p: a < b,
                       lax.le_p: a <= b, lax.gt_p: a > b, lax.ge_p: a >= b}
                env[id(eq.outvars[0])] = ops[eq.primitive]
        elif eq.primitive is lax.convert_element_type_p and len(eq.invars) == 1:
            val = env.get(id(eq.invars[0]))
            if val is not None:
                env[id(eq.outvars[0])] = int(val)
        if id(pred_var) in env:
            return bool(env[id(pred_var)])
    return None

def _live_vars(jaxpr) -> set[int]:
    """Variable ids that reach a side-effecting store or control-flow input, ignoring `sub(v, v)`
    type-carrying zero probes (which LLVM folds to 0 and DCEs)."""
    from jax import lax

    needed = {id(v) for v in jaxpr.outvars}
    for eq in reversed(jaxpr.eqns):
        if eq.primitive is lax.sub_p and len(eq.invars) == 2 and eq.invars[0] is eq.invars[1]:
            continue
        side_effect = (eq.primitive.name in ("swap", "run_scoped")
                       or (eq.primitive.name == "inline_mgpu" and not eq.outvars)
                       or eq.primitive in (lax.scan_p, lax.while_p, lax.cond_p))
        if side_effect or any(id(v) in needed for v in eq.outvars):
            needed.update(id(v) for v in eq.invars)
    return needed

def _sub_ref_kinds(parent_kinds: dict[int, str], sub_jaxpr, outer_invars) -> dict[int, str]:
    return {id(inner): parent_kinds[id(outer)]
            for inner, outer in zip(sub_jaxpr.invars, outer_invars, strict=False)
            if id(outer) in parent_kinds}


def _emitted_device_bytes(body, *, rows: int, blocks: int, slices: int, read_width: int,
                          width: int) -> float:
    """Device bytes transferred on the common one-window path of a traced Pallas `merged_padded` body,
    counted independently from its emitted `get` and `swap` equations (`body` is `kernel.params['jaxpr']`),
    evaluating `program_id(3)` guards per channel block and clipping masked tail accesses to physical
    ref bounds (`slices`, `read_width`, `width`)."""
    from jax import lax

    chunk = windows.chunk_width(slices)
    chunks = -(-slices // chunk)
    channels = 1 << (width - 1).bit_length()
    slice_frac = slices / (chunks * chunk)
    root_kinds = {
        id(body.invars[0]): "out",
        id(body.invars[1]): "stat",
        id(body.invars[2]): "stat",
        id(body.invars[3]): "merged",
    }
    scale_by_kind = {
        ("get", "out"): slice_frac * (read_width / channels),
        ("get", "stat"): slice_frac,
        ("swap", "merged"): width / channels,
    }

    def walk(jaxpr, kinds: dict[int, str]) -> float:
        live = _live_vars(jaxpr)
        total = 0.0
        for eq in jaxpr.eqns:
            if eq.primitive.name == "get" and id(eq.outvars[0]) in live:
                aval = eq.outvars[0].aval
                frac = scale_by_kind.get(("get", kinds.get(id(eq.invars[0]), "")), 1.0)
                total += math.prod(aval.shape) * aval.dtype.itemsize * frac
            elif eq.primitive.name == "swap":
                aval = eq.invars[1].aval
                frac = scale_by_kind.get(("swap", kinds.get(id(eq.invars[0]), "")), 1.0)
                total += math.prod(aval.shape) * aval.dtype.itemsize * frac
            elif eq.primitive is lax.scan_p:
                sub = eq.params["jaxpr"].jaxpr
                total += eq.params["length"] * walk(sub, _sub_ref_kinds(kinds, sub, eq.invars))
            elif eq.primitive is lax.while_p:
                sub = eq.params["body_jaxpr"].jaxpr
                total += walk(sub, _sub_ref_kinds(kinds, sub, eq.invars))
            elif eq.primitive is lax.cond_p:
                b0, b1 = (walk(b.jaxpr, _sub_ref_kinds(kinds, b.jaxpr, eq.invars[1:]))
                          for b in eq.params["branches"])
                taken = [_eval_block_predicate(jaxpr, eq.invars[0], bid) for bid in range(blocks)]
                if all(flag is not None for flag in taken):
                    true_count = sum(bool(flag) for flag in taken)
                    total += (true_count * b1 + (blocks - true_count) * b0) / blocks
                elif not _jaxpr_has_load(eq.params["branches"][1].jaxpr):
                    total += max(b0, b1)
                else:
                    total += min(b0, b1)
        return total

    return walk(body, root_kinds) * rows * blocks


def _traced_merge_bytes(spec, padded: jax.ShapeDtypeStruct, stacked: jax.ShapeDtypeStruct,
                        queries: int, width: int, block: int, *, out_dtype=None,
                        out_heads: int | None = None) -> float:
    """Independent device-byte count for `merged._launch` at `block` from its emitted Pallas body."""
    batch, slices, _, heads, read_width = padded.shape
    channels = 1 << (width - 1).bit_length()
    launch = functools.partial(merged._launch, units=algebra.units_of(spec), queries=queries, width=width,
                               block=block, out_dtype=out_dtype, out_heads=out_heads)
    (kernel,) = (eq for eq in jax.make_jaxpr(launch)(padded, stacked).jaxpr.eqns if programs.opaque(eq))
    return _emitted_device_bytes(kernel.params["jaxpr"], rows=batch * queries * heads,
                                 blocks=channels // block, slices=slices, read_width=read_width, width=width)


def _bind_mosaic_sub(sub, outer_invars, parent_kinds: dict[int, str],
                     parent_blocks: set[int]) -> tuple[dict[int, str], set[int]]:
    """Map ref kinds and `block` axis vars into `sub` (`run_scoped` binds `outer_invars` to `constvars`,
    while `scan`, `while` and `cond` bind them to `invars`)."""
    targets = (sub.constvars if sub.constvars and len(sub.constvars) == len(outer_invars)
               else sub.invars)
    kinds = {id(inner): parent_kinds[id(outer)]
             for inner, outer in zip(targets, outer_invars, strict=False) if id(outer) in parent_kinds}
    blocks = {id(inner) for inner, outer in zip(targets, outer_invars, strict=False)
              if id(outer) in parent_blocks}
    return kinds, blocks


def _jaxpr_has_inline_load(jaxpr, kinds: dict[int, str]) -> bool:
    """Whether `jaxpr` or any nested scope/scan/while/cond executes a Mosaic chunk-load `inline_mgpu`."""
    from jax import lax

    for eq in jaxpr.eqns:
        if (eq.primitive.name == "inline_mgpu" and len(eq.invars) >= 2
                and kinds.get(id(eq.invars[0])) == "out" and kinds.get(id(eq.invars[1])) == "stat"):
            return True
        if eq.primitive is lax.scan_p:
            sub = eq.params["jaxpr"].jaxpr
            if _jaxpr_has_inline_load(sub, _bind_mosaic_sub(sub, eq.invars, kinds, set())[0]):
                return True
        elif eq.primitive is lax.while_p:
            sub = eq.params["body_jaxpr"].jaxpr
            body_in = eq.invars[eq.params["cond_nconsts"]:]
            if _jaxpr_has_inline_load(sub, _bind_mosaic_sub(sub, body_in, kinds, set())[0]):
                return True
        elif eq.primitive is lax.cond_p:
            if any(_jaxpr_has_inline_load(b.jaxpr, _bind_mosaic_sub(b.jaxpr, eq.invars[1:], kinds, set())[0])
                   for b in eq.params["branches"]):
                return True
        elif eq.primitive.name == "run_scoped":
            sub = eq.params["jaxpr"]
            if _jaxpr_has_inline_load(sub, _bind_mosaic_sub(sub, eq.invars, kinds, set())[0]):
                return True
    return False


def _stats_store_gated_on_block_zero(mgpu_fn) -> bool:
    """Verify the lowered Mosaic MLIR body guards every `llvm.store` inside an `scf.if` whose condition
    conjoins `arith.cmpi eq(c_block, 0)`."""
    from jax.extend.mlir import ir as mgpu_ir
    from jax.extend.mlir.dialects import arith as mgpu_arith, func as mgpu_func
    from jax.interpreters import mlir as jax_mlir

    class _Arg:
        def __init__(self, regs: np.ndarray):
            self.registers = regs

    with jax_mlir.make_ir_context(), mgpu_ir.Location.unknown():
        f32 = mgpu_ir.F32Type.get()
        i32 = mgpu_ir.IntegerType.get_signless(32)
        ref_ty = mgpu_ir.MemRefType.get((1, 1, 1), f32, mgpu_ir.StridedLayoutAttr.get(0, [1, 1, 1]))
        vec_ty = mgpu_ir.VectorType.get((1,), f32)
        module = mgpu_ir.Module.create()
        with mgpu_ir.InsertionPoint(module.body):
            fn_op = mgpu_func.FuncOp("check", ([ref_ty, ref_ty, vec_ty, vec_ty, i32, i32, i32, i32], []))
            block = fn_op.add_entry_block()
            args = list(block.arguments)
            with mgpu_ir.InsertionPoint(block):
                mgpu_fn(None, args[0], args[1], _Arg(np.array([args[2]], dtype=object)),
                        _Arg(np.array([args[3]], dtype=object)),
                        *(_Arg(np.array(a, dtype=object)) for a in args[4:]))
        cb_arg = args[7]

        def has_cb_eq_zero(cond) -> bool:
            owner = cond.owner
            if not isinstance(owner, (mgpu_ir.Operation, mgpu_ir.OpView)):
                return False
            op_obj = owner.operation
            if op_obj.name == "arith.andi":
                return any(has_cb_eq_zero(operand) for operand in op_obj.operands)
            if op_obj.name != "arith.cmpi":
                return False
            pred = mgpu_ir.IntegerAttr(op_obj.attributes["predicate"]).value
            if pred != int(mgpu_arith.CmpIPredicate.eq):
                return False
            lhs, rhs = op_obj.operands[0], op_obj.operands[1]
            for candidate, other in ((lhs, rhs), (rhs, lhs)):
                other_owner = other.owner
                if (candidate == cb_arg
                        and isinstance(other_owner, (mgpu_ir.Operation, mgpu_ir.OpView))
                        and other_owner.operation.name == "arith.constant"
                        and mgpu_ir.IntegerAttr(other_owner.operation.attributes["value"]).value == 0):
                    return True
            return False

        guarded_stores = 0
        for op in block.operations:
            if op.operation.name == "llvm.store":
                return False
            if op.operation.name == "scf.if":
                then_ops = list(op.operation.regions[0].blocks[0].operations)
                stores = sum(1 for inner in then_ops if inner.operation.name == "llvm.store")
                if stores:
                    if not has_cb_eq_zero(op.operation.operands[0]):
                        return False
                    guarded_stores += stores
        return guarded_stores >= 2


def _emitted_mosaic_device_bytes(body, *, rows: int, blocks: int, slices: int, width: int) -> float:
    """Device bytes transferred on the common one-window path of a traced Mosaic `merged_padded` body
    (`kernel.params['jaxprs'][0]`), reading clipped channel bounds directly from each emitted `inline_mgpu`
    load/store and verifying row-stat stores are gated on `axis_index('block') == 0` in the lowered MLIR."""
    from jax import lax

    chunk = windows.chunk_width(slices)
    chunks = -(-slices // chunk)
    channels = 1 << (width - 1).bit_length()
    slice_frac = slices / (chunks * chunk)
    root_kinds = {id(body.invars[0]): "out", id(body.invars[1]): "stat", id(body.invars[2]): "merged"}
    if len(body.invars) >= 5:
        root_kinds[id(body.invars[3])] = "max"
        root_kinds[id(body.invars[4])] = "tot"

    def inline_bytes(eq, kinds: dict[int, str], live: set[int], block_vars: set[int]) -> float:
        nonlocals = inspect.getclosurevars(eq.params["mgpu_fn"]).nonlocals
        k0 = kinds.get(id(eq.invars[0]), "")
        k1 = kinds.get(id(eq.invars[1]), "") if len(eq.invars) > 1 else ""
        if k0 == "out" and k1 == "stat":
            if not any(id(v) in live for v in eq.outvars):
                return 0.0
            chunk_len = eq.outvars[0].aval.shape[0]
            return (chunk_len * nonlocals["block"] * eq.outvars[0].aval.dtype.itemsize
                    * slice_frac * (nonlocals["read_width"] / channels)
                    + chunk_len * 2 * eq.outvars[1].aval.dtype.itemsize * slice_frac)
        if k0 == "merged":
            return nonlocals["block"] * eq.invars[0].aval.dtype.itemsize * (nonlocals["width"] / channels)
        if k0 == "max" and k1 == "tot":
            gated = (id(eq.invars[-1]) in block_vars
                     and _stats_store_gated_on_block_zero(eq.params["mgpu_fn"]))
            stat_bytes = eq.invars[0].aval.dtype.itemsize + eq.invars[1].aval.dtype.itemsize
            return stat_bytes / (blocks if gated else 1)
        return 0.0

    def walk(jaxpr, kinds: dict[int, str], block_vars: set[int]) -> float:
        live = _live_vars(jaxpr)
        block_vars = set(block_vars)
        total = 0.0
        for eq in jaxpr.eqns:
            if eq.primitive.name == "axis_index" and eq.params.get("axis_name") == "block":
                block_vars.add(id(eq.outvars[0]))
            elif eq.primitive.name == "inline_mgpu":
                total += inline_bytes(eq, kinds, live, block_vars)
            elif eq.primitive is lax.scan_p:
                sub = eq.params["jaxpr"].jaxpr
                total += eq.params["length"] * walk(sub, *_bind_mosaic_sub(sub, eq.invars, kinds, block_vars))
            elif eq.primitive is lax.while_p:
                sub = eq.params["body_jaxpr"].jaxpr
                body_in = eq.invars[eq.params["cond_nconsts"]:]
                total += walk(sub, *_bind_mosaic_sub(sub, body_in, kinds, block_vars))
            elif eq.primitive is lax.cond_p:
                b0, b1 = (walk(b.jaxpr, *_bind_mosaic_sub(b.jaxpr, eq.invars[1:], kinds, block_vars))
                          for b in eq.params["branches"])
                b1_loads = _jaxpr_has_inline_load(
                    eq.params["branches"][1].jaxpr,
                    _bind_mosaic_sub(eq.params["branches"][1].jaxpr, eq.invars[1:], kinds, set())[0])
                total += max(b0, b1) if not b1_loads else min(b0, b1)
            elif eq.primitive.name == "run_scoped":
                sub = eq.params["jaxpr"]
                total += walk(sub, *_bind_mosaic_sub(sub, eq.invars, kinds, block_vars))
        return total
    return walk(body, root_kinds, set()) * rows * blocks


def _traced_mosaic_merge_bytes(spec, padded: jax.ShapeDtypeStruct, stacked: jax.ShapeDtypeStruct,
                               queries: int, width: int, block: int, *, target: generation.GpuTarget,
                               out_dtype=None, out_heads: int | None = None) -> float:
    """Independent device-byte count for `merged._launch_mosaic` at `block` from its emitted Mosaic body."""
    batch, slices, _, heads, _ = padded.shape
    channels = 1 << (width - 1).bit_length()
    launch = functools.partial(
        merged._launch_mosaic, units=algebra.units_of(spec), queries=queries, width=width, block=block,
        out_dtype=out_dtype, out_heads=out_heads, family=generation.family(target.generation) or "mma.sync")
    with generation.as_target(target):
        (kernel,) = (eq for eq in programs.equations(jax.make_jaxpr(launch)(padded, stacked).jaxpr)
                     if programs.opaque(eq))
    return _emitted_mosaic_device_bytes(kernel.params["jaxprs"][0], rows=batch * queries * heads,
                                        blocks=channels // block, slices=slices, width=width)


def test_a_padded_merges_launch_work_counts_common_passes_unmasked_bytes_and_span():
    """`merged.launch_work` prices a native `merged_padded` launch on the common one-window path (rows 8845,
    9795, 10533): one kernel, span `3 * chunks`, and exact agreement with the emitted Mosaic GPU (`sm90`)
    and Triton (`sm75`) launch bodies across unmasked, multi-block single-stats-writer, irregular masked
    tail, and padded-input (`read_width > width`) shapes."""
    sm90 = generation.gpu_target("sm90")
    sm75 = generation.gpu_target("sm75")
    spec = SPECS["fold"]

    cases = [
        (2, 8, 16, 8, 4, 128, 1),
        (2, 32, 16, 8, 4, 128, 4),
        (2, 9, 16, 8, 4, 96, 2),
        (1, 8, 8, 1, 1, 128, 1),
    ]
    for batch, slices, padded_q, heads, queries, width, chunks in cases:
        padded = jax.ShapeDtypeStruct((batch, slices, padded_q, heads, width), jnp.float32)
        stacked = jax.ShapeDtypeStruct((2, batch, heads, slices, padded_q), jnp.float32)
        block = merged._channel_block(spec, padded, stacked, queries, width, sm90)
        work = merged.launch_work(spec, padded, stacked, queries, width, sm90)
        assert work.kernels == 1.0
        assert work.span == float(3 * chunks)
        assert work.device_bytes == _traced_mosaic_merge_bytes(
            spec, padded, stacked, queries, width, block, target=sm90)
        triton_block = merged._channel_block(spec, padded, stacked, queries, width, sm75)
        triton_work = merged.launch_work(spec, padded, stacked, queries, width, sm75)
        assert triton_work.device_bytes == _traced_merge_bytes(
            spec, padded, stacked, queries, width, triton_block)
    # Padded input (`read_width=128 > width=80`): Mosaic predicates loads at `width`, while Triton reads
    # `read_width` physical columns; each matches its own emitted launch body directly (row 9795).
    pad_p = jax.ShapeDtypeStruct((2, 8, 16, 8, 128), jnp.float32)
    pad_s = jax.ShapeDtypeStruct((2, 2, 8, 8, 16), jnp.float32)
    mosaic_block = merged._channel_block(spec, pad_p, pad_s, 4, 80, sm90)
    mosaic_work = merged.launch_work(spec, pad_p, pad_s, 4, 80, sm90)
    assert mosaic_work.device_bytes == _traced_mosaic_merge_bytes(
        spec, pad_p, pad_s, 4, 80, mosaic_block, target=sm90)
    triton_block = merged._channel_block(spec, pad_p, pad_s, 4, 80, sm75)
    triton_work = merged.launch_work(spec, pad_p, pad_s, 4, 80, sm75)
    assert triton_work.device_bytes == _traced_merge_bytes(spec, pad_p, pad_s, 4, 80, triton_block)
    assert mosaic_work.device_bytes < triton_work.device_bytes, (mosaic_work, triton_work)
    # Explicit multi-block schedule (`block=64`, 2 blocks for `width=128`) vs single-block (`block=128`):
    # verifies single-stats-writer (`c_block == 0`) accounting directly on the emitted Mosaic MLIR body
    # without asserting a derived block choice.
    small_p = jax.ShapeDtypeStruct((1, 8, 8, 1, 128), jnp.float32)
    small_s = jax.ShapeDtypeStruct((2, 1, 1, 8, 8), jnp.float32)
    units = algebra.units_of(spec)
    work_1blk, _ = merged._launch_for_block(units, small_p, small_s, 1, 128, 128, sm90, family="wgmma")
    work_2blk, _ = merged._launch_for_block(units, small_p, small_s, 1, 128, 64, sm90, family="wgmma")
    assert work_1blk.device_bytes == _traced_mosaic_merge_bytes(spec, small_p, small_s, 1, 128, 128,
                                                                target=sm90)
    assert work_2blk.device_bytes == _traced_mosaic_merge_bytes(spec, small_p, small_s, 1, 128, 64,
                                                                target=sm90)
    # Second block adds only one extra pass of 3 * 8 * 2 * 4 stat reads (192 B), zero extra stat-store bytes.
    assert work_2blk.device_bytes - work_1blk.device_bytes == float(3 * 8 * 2 * 4)

def test_a_split_merge_with_initial_state_uses_one_merged_padded_launch():
    """A split decode merge with a sink or resumed `Initial` state (`merged.merged_split`, row 9804) lowers
    through `merged_padded_p` (one launch on GPU), matches `algebra.merged_state` on value and vmap, and
    prices its Triton launch body with the initial state."""
    spec = algebra.Forward("spans", 128, ids=False, scale=1.0, natural=True, initial=True)
    batch, slices, rows, heads, width = 2, 4, 8, 2, 16
    queries = 3
    key = jax.random.key(9804)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    padded = jax.random.normal(k1, (batch, slices, rows, heads, width), jnp.float32)
    stacked = jnp.stack([
        jax.random.normal(k2, (batch, heads, slices, rows), jnp.float32),
        jax.random.uniform(k3, (batch, heads, slices, rows), jnp.float32, minval=0.1, maxval=2.0),
    ])
    init_m = jax.random.normal(k4, (batch, heads, queries), jnp.float32)
    init_t = jax.random.uniform(k5, (batch, heads, queries), jnp.float32, minval=0.1, maxval=2.0)
    init_acc = jax.random.normal(k1, (batch, heads, queries, width), jnp.float32)
    initial = template.Initial(init_m, init_t, init_acc)

    def run(p, s, im, it, ia):
        res = merged.merged_split(spec, p, s, queries=queries, width=width,
                                  initial=template.Initial(im, it, ia))
        return res.acc, res.m, res.total

    traced = jax.make_jaxpr(run)(padded, stacked, init_m, init_t, init_acc)
    from dew.nn import programs
    merge_eqns = [e for e in programs.equations(traced.jaxpr) if e.primitive is merged.merged_padded_p]
    assert len(merge_eqns) == 1 and len(merge_eqns[0].invars) == 5

    got = run(padded, stacked, init_m, init_t, init_acc)
    trimmed = merged._trimmed(padded, stacked, queries=queries, width=width)
    ref = algebra.merged_state(spec, *trimmed, initial)
    for a, b in zip(got, ref[:3], strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    vmapped = jax.vmap(run)(padded[None], stacked[None], init_m[None], init_t[None], init_acc[None])
    for a, b in zip(vmapped, got, strict=True):
        np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b))

    with generation.assumed("sm90"):
        p_s = jax.ShapeDtypeStruct(padded.shape, padded.dtype)
        s_s = jax.ShapeDtypeStruct(stacked.shape, stacked.dtype)
        work = merged.launch_work(spec, p_s, s_s, queries, width, generation.gpu_target("sm90"),
                                  has_initial=True)
    assert work.kernels == 1 and work.device_bytes > 0
    # Empty batch (vmap over size 0) handles CUDA lowering without ZeroDivisionError.
    empty_p = jnp.zeros((0, slices, rows, heads, width), jnp.float32)
    empty_s = jnp.zeros((2, 0, heads, slices, rows), jnp.float32)
    with generation.assumed("sm90"):
        empty_out = merged._kernel(empty_p, empty_s, units=algebra.units_of(spec), queries=queries,
                                   width=width)
    assert empty_out[0].shape == (0, queries, heads, width)


def test_a_finished_split_merge_ungroups_and_casts_in_one_launch():
    """`merged.merged_finished` (row 7926) finishes, ungroups `[B, T*G, KV, Dv] -> [B, T, KV*G, Dv]` and
    casts to `out_dtype` inside the `merged_padded` launch, matching `algebra.finish` + `ungrouped_rows`
    bitwise with and without `initial` and under `vmap`, with fewer output device bytes priced."""
    spec = algebra.Forward("spans", 128, ids=False, scale=1.0, natural=True, initial=True)
    batch, slices, rows, kv_heads, group, tokens, width = 2, 4, 16, 2, 4, 2, 16
    queries, out_heads = tokens * group, kv_heads * group
    key = jax.random.key(7926)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    padded = jax.random.normal(k1, (batch, slices, rows, kv_heads, width), jnp.float32)
    totals = jax.random.uniform(k3, (batch, kv_heads, slices, rows), jnp.float32, minval=0.1, maxval=2.0)
    # Force query 0 on batch 0, kv_head 0 to have zero total across slices to exercise empty-row zeroing.
    totals = totals.at[0, 0, :, 0].set(0.0)
    stacked = jnp.stack([jax.random.normal(k2, (batch, kv_heads, slices, rows), jnp.float32), totals])
    init_m = jax.random.normal(k4, (batch, kv_heads, queries), jnp.float32)
    init_t = jax.random.uniform(k5, (batch, kv_heads, queries), jnp.float32, minval=0.1, maxval=2.0)
    init_acc = jax.random.normal(k1, (batch, kv_heads, queries, width), jnp.float32)
    initial = template.Initial(init_m, init_t, init_acc)

    for init in (None, initial):
        got = merged.merged_finished(spec, padded, stacked, queries=queries, width=width,
                                     out_dtype=jnp.bfloat16, out_heads=out_heads, initial=init)
        state = merged.merged_split(spec, padded, stacked, queries=queries, width=width, initial=init)
        ref = template.ungrouped_rows(
            algebra.finish(spec, state, initial=True, normalized=True)[0], out_heads).astype(jnp.bfloat16)
        assert got.shape == (batch, tokens, out_heads, width) and got.dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

    vmapped = jax.vmap(lambda p, s: merged.merged_finished(
        spec, p, s, queries=queries, width=width, out_dtype=jnp.bfloat16, out_heads=out_heads))(
        padded[None], stacked[None])
    ref_no_init = merged.merged_finished(spec, padded, stacked, queries=queries, width=width,
                                         out_dtype=jnp.bfloat16, out_heads=out_heads)
    np.testing.assert_array_equal(np.asarray(vmapped[0]), np.asarray(ref_no_init))

    with generation.assumed("sm90"):
        p_s = jax.ShapeDtypeStruct(padded.shape, padded.dtype)
        s_s = jax.ShapeDtypeStruct(stacked.shape, stacked.dtype)
        target = generation.gpu_target("sm90")
        work_plain = merged.launch_work(spec, p_s, s_s, queries, width, target)
        work_fin = merged.launch_work(spec, p_s, s_s, queries, width, target,
                                      out_dtype=jnp.bfloat16, out_heads=out_heads)
    assert work_fin.kernels == 1 and work_fin.device_bytes < work_plain.device_bytes


def test_a_finished_split_merge_preserves_subnormal_initial_mass():
    """When key slices are empty and `initial` carries subnormal mass (`2^-127`, row 10778), both
    `merged.merged_finished` (`_value`) and the Pallas merge kernel (`_launch`) preserve the merge law's
    normalized output `1.0` instead of flushing `state.total > 0` to `0.0`, while truly empty rows stay
    `0.0`."""
    batch, slices, rows, kv_heads, width, queries, out_heads = 1, 2, 4, 1, 16, 2, 2
    spec = algebra.finished_form(algebra.Forward(form="full", ids=False, keys=16, scale=1.0))

    def draw(_seed: int):
        padded = np.zeros((batch, slices, rows, kv_heads, width), np.float32)
        maxima = np.full((batch, kv_heads, slices, rows), -np.inf, np.float32)
        totals = np.zeros((batch, kv_heads, slices, rows), np.float32)
        stacked = np.stack([maxima, totals])
        sub = np.float32(2.0 ** -127)
        init_m = np.zeros((batch, kv_heads, queries), np.float32)
        init_t = np.array([[[sub, 0.0]]], np.float32)
        init_acc = np.stack([np.full((batch, kv_heads, width), sub, np.float32),
                             np.zeros((batch, kv_heads, width), np.float32)], axis=2)
        return padded, stacked, init_m, init_t, init_acc

    def ours(p, s, im, it, ia):
        return merged.merged_finished(spec, p, s, queries=queries, width=width, out_dtype=jnp.bfloat16,
                                      out_heads=out_heads, initial=template.Initial(im, it, ia))

    def reference(p, s, im, it, ia):
        del p, s, im, it, ia
        return np.concatenate([np.ones((1, 1, 1, width), np.float64),
                               np.zeros((1, 1, 1, width), np.float64)], axis=2)

    assert_as_exact(jax.jit(ours), as_given(reference), draw,
                    "finished split merge preserves subnormal initial mass", seeds=FEWEST_SEEDS)

def test_resumable_split_merge_preserves_subnormal_initial_mass():
    """Row 10818: `algebra.resumable` on a finished state carrying subnormal mass (`2^-127`) keeps its
    natural maximum instead of replacing it with `LEAST`, so a subsequent `merged.merged_finished` with a
    low-logit slice (`-140`) preserves the resumed output instead of letting the low-logit slice overwrite
    it."""
    batch, slices, rows, kv_heads, width, queries = 1, 2, 4, 1, 16, 2
    spec = algebra.finished_form(algebra.Forward(form="full", ids=False, keys=16, scale=1.0))
    padded = jnp.zeros((batch, slices, rows, kv_heads, width), jnp.float32)
    sub = jnp.float32(2.0 ** -127)
    scale = jnp.float32(2.0 ** 20)
    state_acc = jnp.stack([jnp.full((batch, kv_heads, width), scale, jnp.float32),
                           jnp.zeros((batch, kv_heads, width), jnp.float32)], axis=1)
    state_m = jnp.zeros((batch, queries, kv_heads), jnp.float32)
    state_total = jnp.array([[[sub], [0.0]]], jnp.float32)
    state = algebra.Partial(state_acc, state_m, state_total)

    low_m = jnp.full((batch, kv_heads, slices, rows), -160.0, jnp.float32)
    low_t = jnp.ones((batch, kv_heads, slices, rows), jnp.float32)
    low_stacked = jnp.stack([low_m, low_t])

    def step(p_in, st_in, low_s):
        resumed = algebra.resumable(spec, st_in)
        next_init = template.Initial(jnp.swapaxes(resumed.m, 1, 2), jnp.swapaxes(resumed.total, 1, 2),
                                     jnp.swapaxes(resumed.acc, 1, 2))
        out = merged.merged_finished(spec, p_in, low_s, queries=queries, width=width,
                                     out_dtype=jnp.float32, out_heads=queries, initial=next_init)
        return resumed.m, out

    resumed_m, got = jax.jit(step)(padded, state, low_stacked)
    assert float(resumed_m[0, 0, 0]) == 0.0, resumed_m
    np.testing.assert_array_equal(np.asarray(got[0, 0, 0]), np.full(width, float(scale), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(got[0, 0, 1]), np.zeros(width, dtype=np.float32))
