"""A split decode step's slices merged where its launch wrote them (`merged.merged_padded`): its
derivatives the law's path's (`merged._value`), which its JVP rule takes; it and its state on par with the
plain merge from float64, here and on a GPU's one launch (tests/test_gpu_attention.py)."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import plain_merged
from generation_support import COMPILES_ON_CPU
from kernel_compile import compile_kernels
from reference_error import FEWEST_SEEDS, as_given, assert_as_exact, dew_path

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
            functools.partial(merged._value, spec=spec, queries=QUERIES, width=WIDTH))


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


@pytest.mark.parametrize("form", sorted(SPECS))
def test_a_padded_merge_and_its_derivatives_are_on_par(form):
    """The padded merge's finished output, its state's mass, the output's gradient and that gradient's
    derivative along a direction on par with the plain merge's (`_plain_state`, JAX's differentiation of it,
    independent of the law) against float64 (check 1), jitted and op by op (`as_given`), the reference run
    op by op."""
    spec, ours, _ = _merges(form)
    reference = as_given(state_orders(_plain_state(spec)))
    orders = state_orders(finished_state(ours, spec))
    for program, run in ((jax.jit(orders), "jitted"), (as_given(orders), "op by op")):
        assert_as_exact(program, reference, _drawn, f"a padded merge, {form}, {run}", seeds=MERGE_SEEDS)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [80, 96, 100, 128, 192])
@pytest.mark.parametrize("name", ["sm90", "sm100"])
def test_a_padded_merge_lowers_and_compiles_as_one_launch_at_any_value_width(name, width):
    """The padded merge's one launch (`merged._kernel`, a GPU's lowering) lowers and compiles for a
    compile-only H100 and B200 at value widths 80, 96, 100 and 192 as at 128: a row's channels one block of
    the least power of two past the width (Triton's tiles), its loads and stores past the columns masked."""
    _one_launch(name, slices=16, width=width)


@COMPILES_ON_CPU
@pytest.mark.parametrize("name", ["sm90", "sm100"])
def test_a_batch_of_splits_merges_as_one_launch(name):
    """The padded merge mapped over a batch of three splits (`jax.vmap`, its batching rule) compiles for a
    compile-only H100 and B200 to one launch of `merged_padded`, the batch folded into its rows, beside
    nothing but the one transpose that folds it (the stacked states' batch moved past their two kinds), its
    every launch typed (`report.launched_thunks`): mapped by the law's path, a batched split merge ran XLA's
    law, no launch (the merge P0's base review of aa990591f, F9; its head reviews of e97ed082d and
    fb1f405be, F6: custom kernels alone counted, then an instruction count, neither a launch count)."""
    from fla_gdn_support import KernelCase

    padded = jax.ShapeDtypeStruct((3, 2, 16, 16, 8, 64), jnp.float32)
    stacked = jax.ShapeDtypeStruct((3, 2, 2, 8, 16, 16), jnp.float32)
    merge = jax.vmap(lambda padded, stacked: tuple(merged.merged_padded(SPECS["natural"], padded, stacked,
                                                                        queries=4, width=64)))
    with generation.assumed(name):
        (record,) = compile_kernels([KernelCase("batched", lambda: (merge, (padded, stacked)))], name)
    assert "error" not in record, record
    assert sorted(record["thunks"]) == ["merged_padded", "wrapped_transpose"], record["thunks"]


@COMPILES_ON_CPU
@pytest.mark.parametrize("slices", [64, 256])
@pytest.mark.parametrize("name", ["sm90", "sm100"])
def test_a_long_splits_merge_compiles_as_one_launch(name, slices):
    """The padded merge of a long split's 64 and 256 slices (row 8439's: its law's states unrolled made
    the one launch's program grow with them, 236 s and past 15 min to compile for an H100) compiles as one
    launch for a compile-only H100 and B200, its program a chunk of the law's at a time (`merged._chunks`)."""
    _one_launch(name, slices=slices, width=128)


def _one_launch(name: str, *, slices: int, width: int) -> None:
    """The padded merge of `slices` slices at value `width`, compiled for a compile-only `name` GPU
    (`kernel_compile.compile_kernels`): its executable launches `merged_padded` once and no other custom
    kernel, a lowering of XLA's alone none, and launches nothing else (`report.launched_thunks`)."""
    from fla_gdn_support import KernelCase

    padded = jax.ShapeDtypeStruct((2, slices, 16, 8, width), jnp.float32)
    stacked = jax.ShapeDtypeStruct((2, 2, 8, slices, 16), jnp.float32)
    merge = functools.partial(merged.merged_padded, SPECS["natural"], queries=4, width=width)
    with generation.assumed(name):
        (record,) = compile_kernels([KernelCase("merge", lambda: (merge, (padded, stacked)))], name)
    assert "error" not in record and record["thunks"] == ["merged_padded"], record
