"""A split decode step's slices merged where its launch wrote them (`merged.merged_padded`): its
derivatives the law's path's (`merged._value`), which its JVP rule takes; it and its state on par with the
plain merge from float64, here and on a GPU's one launch (tests/test_gpu_attention.py)."""
import functools
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


@pytest.mark.parametrize("run", ["jitted", "op by op"])
@pytest.mark.parametrize("form", sorted(SPECS))
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
@pytest.mark.parametrize("name", ["sm90", "sm100"])
def test_a_padded_merge_compiles_as_one_launch(name):
    """The padded merge's one launch (`merged._kernel`, a GPU's lowering) compiles for a compile-only H100
    and B200 in one process per GPU across: value widths 80, 96, 100, 128 and 192 (a row's channels one block
    of the least power of two past the width, its loads and stores past the columns masked); a `jax.vmap`
    batch of three splits (the batch folded into its rows, beside nothing but the one transpose that folds it,
    every launch typed by `report.launched_thunks`); and long splits of 64 and 256 slices (row 8439: its
    program a chunk of the law's at a time, `merged._chunks`)."""
    from fla_gdn_support import KernelCase

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
    cases = [*(one_case(f"width-{width}", slices=16, width=width) for width in (80, 96, 100, 128, 192)),
             KernelCase("batched", lambda: (batched, (batched_padded, batched_stacked))),
             KernelCase("finished", lambda: (finished, (finished_padded, finished_stacked))),
             *(one_case(f"slices-{slices}", slices=slices, width=128) for slices in (64, 256))]
    with generation.assumed(name):
        records = compile_kernels(cases, name)
    for record in records:
        assert "error" not in record, record
        expected = (["merged_padded", "wrapped_transpose"] if record["name"] == "batched"
                    else ["merged_padded"])
        assert sorted(record["thunks"]) == expected, record

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
        if (eq.primitive.name == "swap" or eq.primitive in (lax.scan_p, lax.while_p, lax.cond_p)
                or any(id(v) in needed for v in eq.outvars)):
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

def test_a_padded_merges_launch_work_counts_common_passes_unmasked_bytes_and_span():
    """`merged.launch_work` prices a native `merged_padded` launch on the common one-window path (rows 8845,
    10533): one kernel, span `3 * chunks`, and exact agreement with the emitted Pallas `get`/`swap` byte
    count across unmasked, multi-block (row-stat stores evaluated per `program_id(3)`), and irregular masked
    tail shapes."""
    target = generation.gpu_target("sm90")
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
        block = merged._channel_block(spec, padded, stacked, queries, width, target)
        work = merged.launch_work(spec, padded, stacked, queries, width, target)
        assert work.kernels == 1.0
        assert work.span == float(3 * chunks)
        assert work.device_bytes == _traced_merge_bytes(spec, padded, stacked, queries, width, block)
    # The 1-row 128-wide case splits across multiple channel blocks on an H100 (row 8439), exercising the
    # single-stat-store-per-row accounting directly.
    small_p = jax.ShapeDtypeStruct((1, 8, 8, 1, 128), jnp.float32)
    small_s = jax.ShapeDtypeStruct((2, 1, 1, 8, 8), jnp.float32)
    assert merged._channel_block(spec, small_p, small_s, 1, 128, target) < 128


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
    padded = jnp.zeros((batch, slices, rows, kv_heads, width), jnp.float32)
    maxima = jnp.full((batch, kv_heads, slices, rows), -jnp.inf, jnp.float32)
    totals = jnp.zeros((batch, kv_heads, slices, rows), jnp.float32)
    stacked = jnp.stack([maxima, totals])
    sub = jnp.float32(2.0 ** -127)
    init_m = jnp.zeros((batch, kv_heads, queries), jnp.float32)
    init_t = jnp.array([[[sub, 0.0]]], jnp.float32)
    init_acc = jnp.stack([jnp.full((batch, kv_heads, width), sub, jnp.float32),
                          jnp.zeros((batch, kv_heads, width), jnp.float32)], axis=2)
    initial = template.Initial(init_m, init_t, init_acc)

    got = jax.jit(lambda p, s, i: merged.merged_finished(
        spec, p, s, queries=queries, width=width, out_dtype=jnp.bfloat16, out_heads=out_heads,
        initial=i))(padded, stacked, initial)
    expected = np.concatenate([np.ones((1, 1, 1, width), dtype=np.float32),
                               np.zeros((1, 1, 1, width), dtype=np.float32)], axis=2)
    np.testing.assert_array_equal(np.asarray(got, dtype=np.float32), expected)


def test_resumable_split_merge_preserves_subnormal_initial_mass():
    """Row 10818: a non-finished split merge (`merged.merged_split` -> `algebra.resumable`) carrying
    subnormal initial mass (`2^-127`) keeps its natural maximum instead of replacing it with `LEAST`, so a
    subsequent merge with a low-logit key (`-88`) preserves the initial output `1.0` instead of `0.0`."""
    batch, slices, rows, kv_heads, width, queries = 1, 2, 4, 1, 16, 2
    spec = algebra.finished_form(algebra.Forward(form="full", ids=False, keys=16, scale=1.0))
    padded = jnp.zeros((batch, slices, rows, kv_heads, width), jnp.float32)
    maxima = jnp.full((batch, kv_heads, slices, rows), -jnp.inf, jnp.float32)
    totals = jnp.zeros((batch, kv_heads, slices, rows), jnp.float32)
    stacked = jnp.stack([maxima, totals])
    sub = jnp.float32(2.0 ** -127)
    init_m = jnp.zeros((batch, kv_heads, queries), jnp.float32)
    init_t = jnp.array([[[sub, 0.0]]], jnp.float32)
    init_acc = jnp.stack([jnp.full((batch, kv_heads, width), sub, jnp.float32),
                          jnp.zeros((batch, kv_heads, width), jnp.float32)], axis=2)
    initial = template.Initial(init_m, init_t, init_acc)

    low_m = jnp.full((batch, kv_heads, slices, rows), -140.0, jnp.float32)
    low_t = jnp.ones((batch, kv_heads, slices, rows), jnp.float32)
    low_stacked = jnp.stack([low_m, low_t])

    def step(p_in, s_in, init_in, low_s):
        state = merged.merged_split(spec, p_in, s_in, queries=queries, width=width, initial=init_in)
        resumed = algebra.resumable(spec, state)
        next_init = template.Initial(jnp.swapaxes(resumed.m, 1, 2), jnp.swapaxes(resumed.total, 1, 2),
                                     jnp.swapaxes(resumed.acc, 1, 2))
        return merged.merged_finished(spec, p_in, low_s, queries=queries, width=width,
                                      out_dtype=jnp.float32, out_heads=queries, initial=next_init)

    # Under jitted XLA:CPU FTZ, `state.total > 0` would replace m=0.0 with LEAST so the -140 slice wins (0.0);
    # with `means.holding`, m=0.0 survives and 2^-127 dominates 2 * 2^-140 (output ~1.0).
    got = jax.jit(step)(padded, stacked, initial, low_stacked)
    np.testing.assert_allclose(np.asarray(got[0, 0, 0]), np.ones(width, dtype=np.float32), atol=1e-3)
    np.testing.assert_array_equal(np.asarray(got[0, 0, 1]), np.zeros(width, dtype=np.float32))
