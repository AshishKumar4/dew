"""float32 dense products on a CPU, summed no worse than a vectorized dot sums them (row 4863).

XLA's CPU dot sums each output's float32 products as one sequential FMA chain (all 15 of a llama
decoder's dense sites matched it output for output, row 4735's receipts), so its rounding path grows
with the contraction: 2.0 to 2.7 times libggml's local error at 64- and 128-long sums, and a decoder's
logits 1.77 times llama.cpp's distance from float64. A vectorized dot (libggml's, a BLAS kernel's)
keeps one partial sum a lane of each FMA it holds in flight and sums them as a tree at the end, a path
of `depth / partials` FMAs and `log2(partials)` adds. Here the contraction is split into blocks, each
block a dot of its own (XLA's chain, `block` FMAs long), and the blocks' sums are added as a pairwise
tree (`log2(blocks)` adds): the block is the longest power of two whose path is no longer than the
CPU's own vectorized accumulation (`contraction_block`), read from the CPU's vector lanes and its FMA
latency. Its partial sums move `depth / block` outputs' bytes: on armada's EPYC, NS4863-PROF2, 0.55-0.67
of XLA's time at 1-16 rows of 1024-4096 deep, 5-11 times it at 256 rows.

A product's bits are the host's (testing.md, row 3177: XLA promises no common bits across programs). Each
block's sum is XLA's dot, whose algorithm XLA picks by the program's shapes and the host's instruction
level (a vectorized product for a few rows, a YNN library fusion or a plain dot for more), so two programs
summing one row (the dense, ragged and grouped forms, a call of one row and one of many, a decode step and
its prefill) agree to the blocked tree's accuracy, held by check 1, and bit for bit only where they are one
program: one group's products summed by the grouped form and by the dense one differed by 1 to 4 ulp on an
x86-64-v3 EPYC 7713 and agreed on a v4 EPYC 9684X (the host-level verdicts of 2026-11-10; a fixed order of
Dew's own, at 4.9 to 23 times XLA's dot, is row 5383's). A product's gradients are products too, each summed
over its own contraction (`pairwise_dot_general`'s reverse pass). On another platform the product is XLA's own
(`accumulated`'s `platform_dependent`).
"""

import functools
import itertools
import math
import os
import platform
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp

from dew.nn.derivatives import Protected, contracted
from dew.nn.programs import RaggedLayout, ragged_layout

FMA_LATENCY = 4
"""Cycles from an FMA's issue to its result on the cores Dew runs on (Intel Skylake-SP through Sapphire
Rapids, AMD Zen 2 through Zen 4, Arm Neoverse N1, V1 and V2: Agner Fog's instruction tables, Arm's
software optimization guides): the FMAs a dot keeps in flight to issue one a cycle, each with a vector of
partial sums of its own."""


@functools.cache
def vector_lanes() -> int:
    """The float32 lanes of this CPU's widest vector register: 16 with AVX-512 (`avx512f` among its flags),
    8 with AVX or AVX2, 4 for Arm's NEON (`asimd`) and any CPU naming none of them."""
    flags: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith(("flags", "Features")):
                flags = set(line.split(":", 1)[1].split())
                break
    if "avx512f" in flags:
        return 16
    if flags & {"avx", "avx2"} or (not flags and platform.machine() in ("x86_64", "AMD64")):
        return 8
    return 4


def contraction_block(depth: int, lanes: int | None = None) -> int:
    """The products a block sums in one chain, for a contraction `depth` long: the longest power of two
    whose rounding path, `block` FMAs and `ceil(log2(blocks))` adds, is no longer than a vectorized dot's
    on this CPU, `ceil(depth / partials)` FMAs and `ceil(log2(partials))` adds over its `partials` (its
    `vector_lanes` times `FMA_LATENCY`). A depth one block holds, or none, is the plain dot's."""
    if depth <= 1:
        return max(depth, 1)
    partials = min(depth, (lanes or vector_lanes()) * FMA_LATENCY)
    budget = -(-depth // partials) + math.ceil(math.log2(partials))
    block = 1 << (depth.bit_length() - 1)
    while block > 1 and block + math.ceil(math.log2(-(-depth // block))) > budget:
        block //= 2
    return block


@functools.cache
def workspace_bytes() -> int:
    """The bytes a product's blocks' sums may hold at once (`_blocked`): a 64th of this machine's memory,
    or 256 MiB where it states none."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 64
    except (ValueError, OSError):
        return 256 << 20


def _tree(parts: jax.Array) -> jax.Array:
    """`parts`' sum over its axis 1, pairwise: each level adds neighbours, an odd one out carried up."""
    while parts.shape[1] > 1:
        carried = parts[:, -1:] if parts.shape[1] % 2 else None
        even = parts[:, :parts.shape[1] - (carried is not None)]
        parts = even[:, 0::2] + even[:, 1::2]
        if carried is not None:
            parts = jnp.concatenate([parts, carried], axis=1)
    return parts[:, 0]


def _summed(a: jax.Array, b: jax.Array, precision) -> jax.Array:
    """`_blocked`'s sum of `a @ b`, `[batch, rows, depth]` and `[batch, depth, columns]` float32 arrays, over
    the shares of the depth an Explicit mesh axis splits across devices (a projection's rows, which its
    matrix's gradient contracts, split over a data axis): each device's share summed as one product
    (`_blocked`), the shares then added across the devices, as the plain product's partials are; a depth no
    axis splits, one share."""
    sharding = jax.typeof(a).sharding
    names = sharding.spec[2] if len(sharding.spec) > 2 else None
    names = () if names is None else names if isinstance(names, tuple) else (names,)
    shares = math.prod(sharding.mesh.shape[name] for name in names)
    if shares == 1:
        return _blocked(a, b, precision)
    batch, rows, depth = a.shape
    columns = b.shape[2]
    # The shares a batch of their own, leading, so each device's lies whole on it.
    a = jnp.moveaxis(a.reshape(batch, rows, shares, depth // shares), 2, 0).reshape(
        shares * batch, rows, depth // shares)
    b = jnp.moveaxis(b.reshape(batch, shares, depth // shares, columns), 1, 0).reshape(
        shares * batch, depth // shares, columns)
    return _blocked(a, b, precision).reshape(shares, batch, rows, columns).sum(axis=0)


def _blocked(a: jax.Array, b: jax.Array, precision) -> jax.Array:
    """`a @ b` of `[batch, rows, depth]` and `[batch, depth, columns]` float32 arrays, the depth in blocks
    (`contraction_block`), each block's dot XLA's own, their sums added as a tree (`_tree`); a zero
    product pads the last block. Where the blocks' sums would hold more than `workspace_bytes`, a run of the
    columns at a time (`lax.map`), the fewest runs that fit: a CPU vocabulary head's 64 blocks at once
    (512 tokens of 151,936 columns) would hold 18.5 GiB. An empty operand's product is the dot's own."""
    depth, columns = a.shape[2], b.shape[2]
    block = contraction_block(depth)
    if block >= depth or 0 in a.shape + b.shape:
        return jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,))), precision=precision,
                                   preferred_element_type=jnp.float32)
    count = -(-depth // block)
    pad = count * block - depth
    a = jnp.pad(a, ((0, 0), (0, 0), (0, pad))).reshape(a.shape[0], a.shape[1], count, block)
    held = a.shape[0] * count * a.shape[1] * columns * 4
    runs = min(columns, max(1, -(-held // workspace_bytes())))
    width = -(-columns // runs)
    b = jnp.pad(b, ((0, 0), (0, pad), (0, runs * width - columns)))
    b = b.reshape(b.shape[0], count, block, runs, width).transpose(3, 0, 1, 2, 4)

    def run(columns_run):
        return _tree(jax.lax.dot_general(a, columns_run, (((3,), (2,)), ((0, 2), (0, 1))),
                                         precision=precision, preferred_element_type=jnp.float32))

    summed = jax.lax.map(run, b) if runs > 1 else run(b[0])[None]
    return jnp.moveaxis(summed, 0, 2).reshape(a.shape[0], a.shape[1], runs * width)[..., :columns]


def pairwise_dot_general(lhs: jax.Array, rhs: jax.Array, dimension_numbers, precision=None) -> jax.Array:
    """`jax.lax.dot_general` of float32 `lhs` and `rhs` into float32, summed by `_blocked`: each operand laid
    out as `[batch, free, contracted]` and `[batch, contracted, free]`, the result in `dot_general`'s order
    (batch, `lhs`'s free axes, `rhs`'s). Its reverse pass is the plain contraction's, its operands alone
    saved, each cotangent a product of its own blocked over its own contraction (the rows' over the columns,
    the matrix's over the rows; `generation.linear_tangent`, Protected: a reassociation would undo the tree
    it sums by); forward mode and a higher order through the forward's own tangent."""
    from dew.nn.kernels.generation import linear_tangent

    (contracted_axes, against), (batched, batched_rhs) = (tuple(map(tuple, pair))
                                                          for pair in dimension_numbers)
    free = tuple(axis for axis in range(lhs.ndim) if axis not in contracted_axes + batched)
    free_rhs = tuple(axis for axis in range(rhs.ndim) if axis not in against + batched_rhs)
    order, order_rhs = batched + free + contracted_axes, batched_rhs + against + free_rhs

    def shape(array, axes):
        return tuple(array.shape[axis] for axis in axes)

    def laid(left, right):
        # Each operand as `[batch, free, contracted]` and `[batch, contracted, free]`.
        groups, groups_rhs = (batched, free, contracted_axes), (batched_rhs, against, free_rhs)
        return (jax.lax.reshape(jnp.transpose(left, order), [math.prod(shape(lhs, axes)) for axes in groups],
                                out_sharding=_placed((left, groups))),
                jax.lax.reshape(jnp.transpose(right, order_rhs),
                                [math.prod(shape(rhs, axes)) for axes in groups_rhs],
                                out_sharding=_placed((right, groups_rhs))))

    def summed(left, right):
        out = _summed(*laid(left, right), precision)
        return jax.lax.reshape(out, (*shape(lhs, batched + free), *shape(rhs, free_rhs)),
                               out_sharding=_placed((left, _each(batched + free)), (right, _each(free_rhs))))

    def pulled(saved, operands, cotangent, live):
        left, right = laid(*operands)
        # The cotangent's dimensions are the result's: the batch's, `lhs`'s free ones, `rhs`'s.
        ends = itertools.accumulate((0, len(batched), len(free), len(free_rhs)))
        groups = tuple(tuple(range(start, stop)) for start, stop in itertools.pairwise(ends))
        flat = jax.lax.reshape(cotangent, (left.shape[0], left.shape[1], right.shape[2]),
                               out_sharding=_placed((cotangent, groups)))
        gradients = {
            0: lambda: jnp.transpose(jax.lax.reshape(
                _summed(flat, jnp.swapaxes(right, 1, 2), precision), shape(lhs, order),
                out_sharding=_placed((operands[0], _each(order)))), _inverse(order)),
            1: lambda: jnp.transpose(jax.lax.reshape(
                _summed(jnp.swapaxes(left, 1, 2), flat, precision), shape(rhs, order_rhs),
                out_sharding=_placed((operands[1], _each(order_rhs)))), _inverse(order_rhs)),
        }
        return tuple(gradients[index]() for index in live)

    product = contracted("test_a_cpu_pairwise_dense_product_holds_every_order")(
        linear_tangent(lambda *operands: (summed(*operands), ()), pulled, summed, live_cotangents=True,
                       kind=Protected()))
    return product(lhs, rhs)


def _placed(*parts: tuple[jax.Array, tuple[tuple[int, ...], ...]]) -> jax.sharding.NamedSharding | None:
    """The placement of a reshape whose dimensions are `parts`' groups of dimensions, each part an operand
    and its groups in order, a group's dimensions merged into one, where an Explicit axis splits an
    operand: each dimension split by the mesh axes that split its group's, where those are its leading
    ones (a split dimension merged with whole minor ones, or split off alone), else whole. JAX does not
    place a reshape that splits a split dimension, or merges one into a new shape (row 5422), which takes
    it as its `out_sharding` (`sharding.explicit_spec`); None where no Explicit axis splits an operand."""
    from dew.mesh import mesh_axes
    from dew.nn.sharding import explicit_spec

    placed = [(explicit_spec(array), groups) for array, groups in parts]
    mesh = next((found.mesh for found, _ in placed if found is not None), None)
    if mesh is None:
        return None
    entries = []
    for found, groups in placed:
        for axes in groups:
            specs = [None if found is None else found.spec[axis] for axis in axes]
            split = [entry for entry in specs if entry is not None]
            names = tuple(name for entry in split for name in mesh_axes(entry))
            leading = all(entry is not None for entry in specs[:len(split)])
            entries.append((names[0] if len(names) == 1 else names) if names and leading else None)
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*entries))


def _each(axes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """`axes`, each a group of its own (`_placed`): a reshape that keeps them apart."""
    return tuple((axis,) for axis in axes)


def _inverse(permutation: tuple) -> tuple:
    """The permutation undoing `permutation`."""
    return tuple(sorted(range(len(permutation)), key=permutation.__getitem__))


def _rows_blocked(rows: jax.Array, matrices: jax.Array, group_sizes: jax.Array, precision) -> jax.Array:
    """Each row of float32 `rows` (`[rows, depth]`) times its group's matrix of `matrices` (`[groups, depth,
    columns]`, the groups `group_sizes`' runs of rows, the rows past them zero), summed as `_blocked` sums a
    dense product: the depth in blocks (`contraction_block`), each block's ragged dot XLA's own, their sums
    added as a tree (`_tree`), a run of the columns at a time where the blocks' sums would hold more than
    `workspace_bytes`."""
    depth, columns = rows.shape[1], matrices.shape[2]
    block = contraction_block(depth)

    def plain(rows, matrices):
        # Placed as XLA's ragged product places it on a mesh with Explicit axes, where JAX infers no
        # placement for a ragged dot (`grouped_matmul.ragged_placement`): this branch is traced on every
        # platform (`precision.projection_product`'s `platform_dependent`).
        from dew.nn.kernels.grouped_matmul import ragged_placement

        return jax.lax.ragged_dot(rows, matrices, group_sizes, precision=precision,
                                  preferred_element_type=jnp.float32,
                                  out_sharding=ragged_placement(((rows, 0), (matrices, matrices.ndim - 1))))

    if block >= depth or 0 in rows.shape + matrices.shape:
        return plain(rows, matrices)
    count = -(-depth // block)
    pad = count * block - depth
    rows = jnp.pad(rows, ((0, 0), (0, pad))).reshape(rows.shape[0], count, block).transpose(1, 0, 2)
    held = count * rows.shape[1] * columns * 4
    runs = min(columns, max(1, -(-held // workspace_bytes())))
    width = -(-columns // runs)
    matrices = jnp.pad(matrices, ((0, 0), (0, pad), (0, runs * width - columns)))
    matrices = matrices.reshape(matrices.shape[0], count, block, runs, width).transpose(3, 1, 0, 2, 4)

    def run(columns_run):
        parts = jax.lax.map(lambda pair: plain(*pair), (rows, columns_run))
        return _tree(jnp.moveaxis(parts, 0, 1))

    summed = jax.lax.map(run, matrices) if runs > 1 else run(matrices[0])[None]
    return jnp.moveaxis(summed, 0, 1).reshape(rows.shape[1], runs * width)[:, :columns]


def _groups_blocked(rows: jax.Array, cotangent: jax.Array, group_sizes: jax.Array, precision) -> jax.Array:
    """Each group's float32 `rows` (`[rows, depth]`) transposed times its rows of `cotangent` (`[rows,
    width]`), `[groups, depth, width]`, summed as `_blocked` sums a dense contraction over its rows: each
    group's rows in blocks from its first (`contraction_block` of the rows), a block of every group a batched
    dot of XLA's own, the blocks' sums added as `_tree` adds them, one block's into a binary counter of
    partial sums at a time, so the steps are the largest group's blocks and the sums held a level of the tree
    each (a group of no rows zero). The steps walk to the most blocks any group may hold, a static count
    reverse mode differentiates, each past the largest group's blocks skipped (`lax.cond`): adding a zero
    block to the counter changes no sum. Of no rows at all, every group's contraction is the empty sum, zero,
    as XLA's own empty product is, and there is no row to gather."""
    total, depth, width = rows.shape[0], rows.shape[1], cotangent.shape[1]
    groups = group_sizes.shape[0]
    if total == 0:
        return jnp.zeros((groups, depth, width), jnp.float32)
    # Each group's rows gathered from the whole rows: a gather along a dimension a mesh's Explicit axis splits
    # takes no placement JAX infers, so the rows are first made whole on the device (`_whole_rows`).
    rows, cotangent = _whole_rows(rows), _whole_rows(cotangent)
    block = contraction_block(total)
    starts = jnp.cumsum(group_sizes) - group_sizes
    steps = (jnp.max(group_sizes, initial=0) + block - 1) // block
    most = -(-total // block)
    levels = most.bit_length()
    offsets = jnp.arange(block)

    def gathered(operand, step):
        place = step * block + offsets[None, :]
        valid = place < group_sizes[:, None]
        taken = operand[jnp.clip(starts[:, None] + place, 0, total - 1)]
        return jnp.where(valid[..., None], taken, 0)

    def counted(held, step):
        return jax.lax.cond(step < steps, functools.partial(added, step), lambda held: held, held), None

    def added(step, held):
        value = jax.lax.dot_general(gathered(rows, step), gathered(cotangent, step),
                                    (((1,), (1,)), ((0,), (0,))), precision=precision,
                                    preferred_element_type=jnp.float32)
        done = jnp.zeros((), jnp.bool_)
        for level in range(levels):
            bit = (step >> level) & 1 == 1
            value = jnp.where(~done & bit, held[level] + value, value)
            held = held.at[level].set(jnp.where(~done & ~bit, value, held[level]))
            done = done | ~bit
        return held

    held, _ = jax.lax.scan(counted, jnp.zeros((levels, groups, depth, width), jnp.float32), jnp.arange(most))
    # The counter's levels added from the lowest, as `_tree` carries its odd part up to the last add.
    summed, have = jnp.zeros((groups, depth, width), jnp.float32), jnp.zeros((), jnp.bool_)
    for level in range(levels):
        bit = (steps >> level) & 1 == 1
        summed = jnp.where(bit & ~have, held[level], jnp.where(bit, held[level] + summed, summed))
        have = have | bit
    return summed


def _whole_rows(operand: jax.Array) -> jax.Array:
    """`operand` with its first dimension placed whole where a mesh's Explicit axis splits it (its other
    dimensions placed as they are); `operand` itself elsewhere."""
    sharding = jax.typeof(operand).sharding
    if not (isinstance(sharding, jax.sharding.NamedSharding) and sharding.mesh.explicit_axes
            and len(sharding.spec) and sharding.spec[0] is not None):
        return operand
    return jax.sharding.reshard(operand, jax.sharding.PartitionSpec(None, *sharding.spec[1:]))


def pairwise_ragged_dot_general(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array,
                                ragged_dot_dimension_numbers, precision=None, out_sharding=None) -> jax.Array:
    """`jax.lax.ragged_dot_general` of float32 `lhs` and `rhs` into float32 in a layout
    `programs.ragged_layout` names,
    summed blocked and pairwise as `pairwise_dot_general` sums a dense one: rows grouped and their depth
    contracted by `_rows_blocked`, each group's rows contracted by `_groups_blocked`. Its reverse pass is the
    plain product's, each cotangent a product of its own summed so over its own contraction (Protected, as
    the dense form's). Its result placed at `out_sharding` where given, the ragged product's own."""
    from dew.nn.kernels.generation import linear_tangent

    layout = ragged_layout(jnp.ndim(lhs), jnp.ndim(rhs), ragged_dot_dimension_numbers)
    if layout is None:
        raise ValueError(f"no pairwise ragged product takes {ragged_dot_dimension_numbers}")
    summed, pulled = (_ragged_groups(precision) if layout is RaggedLayout.GROUPS
                      else _ragged_rows(layout is RaggedLayout.ROWS_TRANSPOSED, precision))
    product = contracted("test_a_cpu_pairwise_grouped_product_holds_every_order")(
        linear_tangent(lambda *operands: (summed(*operands), ()), pulled, summed, live_cotangents=True,
                       kind=Protected()))
    out = product(lhs, rhs, group_sizes)
    return out if out_sharding is None else jax.sharding.reshard(out, out_sharding)


def _ragged_rows(transposed: bool, precision) -> tuple[Callable, Callable]:
    """`pairwise_ragged_dot_general`'s product and pullback where rows are grouped and their depth contracted,
    each group's matrix `[depth, width]`, or `[width, depth]` where `transposed`."""
    def summed(rows, matrices, sizes):
        matrices = jnp.swapaxes(matrices, 1, 2) if transposed else matrices
        return _rows_blocked(rows, matrices, sizes, precision)

    def pulled(saved, operands, cotangent, live):
        del saved
        rows, matrices, sizes = operands
        laid = jnp.swapaxes(matrices, 1, 2) if transposed else matrices
        gradients = {0: lambda: _rows_blocked(cotangent, jnp.swapaxes(laid, 1, 2), sizes, precision),
                     1: lambda: (lambda grads: jnp.swapaxes(grads, 1, 2) if transposed else grads)(
                         _groups_blocked(rows, cotangent, sizes, precision))}
        return tuple(gradients[index]() for index in live)

    return summed, pulled


def _ragged_groups(precision) -> tuple[Callable, Callable]:
    """`pairwise_ragged_dot_general`'s product and pullback where each group's rows are contracted."""
    def summed(rows, cotangent, sizes):
        return _groups_blocked(rows, cotangent, sizes, precision)

    def pulled(saved, operands, cotangent, live):
        del saved
        rows, rows_cotangent, sizes = operands
        gradients = {0: lambda: _rows_blocked(rows_cotangent, jnp.swapaxes(cotangent, 1, 2), sizes,
                                              precision),
                     1: lambda: _rows_blocked(rows, cotangent, sizes, precision)}
        return tuple(gradients[index]() for index in live)

    return summed, pulled
