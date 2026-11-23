# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dew: products summed over their contractions joined end to end (`mosaic_gpu.gmm`'s `chained`): a
multi-matrix input gradient's products (q, k and v's) one launch, one accumulator a tile (`Chained`, its
kernel refs `ChainedRefs`), its tiles that end non-finite flagged (`flags_declared`) and walked again
with their sums bounded (`guarded_walk`, row 8282)."""

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels.templates import mosaic, rings

if TYPE_CHECKING:
    from dew.nn.kernels.templates.gemm.tiling import Tiles


@dataclasses.dataclass(frozen=True)
class Partition:
    """Dew: extents end to end, each whole tiles (`starts`: each one's first tile): a launch's tiles over
    several arrays in order, tile t the one `which(t)`'s, its own tile `t - starts[which(t)]`. Matrices side
    by side partition a launch's column tiles (`mosaic_gpu.Beside`); chained products its steps
    (`Chained`)."""

    starts: tuple[int, ...]

    @staticmethod
    def started(extents: tuple[int, ...], tile: int) -> tuple[int, ...]:
        """`extents`' first tiles, each whole tiles of `tile`, end to end."""
        tiled = [extent // tile for extent in extents]
        return tuple(sum(tiled[:index]) for index in range(len(tiled)))

    def which(self, index) -> jax.Array:
        """The extent tile `index` falls in: how many of the others start at or before it."""
        later = (jnp.asarray(index >= start, jnp.int32) for start in self.starts[1:])
        return functools.reduce(jnp.add, later, jnp.zeros((), jnp.int32))


@dataclasses.dataclass(frozen=True)
class Chained(Partition):
    """Dew: products summed over their contractions joined end to end (`gmm`'s `rhs` and its `chained`): each
    of `depths`, whole steps of the launch's `block_k`, its steps after the ones before it (`starts`, each
    one's first), step j reading its rows' and its matrix's blocks from the product it falls in (`which`)
    at its own step `j - starts[which]`, one accumulator summing them all: a multi-matrix input gradient's
    products (q, k and v's; a gate's and an up projection's) in the order the plain program adds them, as
    upstream's fused QKV computes it (row 8282). A launch's static configuration, its arrays the launch's
    operands, in the kernel its refs (`read`, `ChainedRefs`). Its one running sum can pass float32's largest
    where every product's own and the total do not; the launch flags each tile that ends non-finite and a
    second walk takes those tiles alone, each product's sum its own (`guarded_walk`). Its contract is a
    GEMM's over the joined depth: check 1 against float64, and finiteness where float64 and the plain float32
    program are finite (`guarded_walk`'s), not each product's float32 sum combined in order."""

    depths: tuple[int, ...]

    @classmethod
    def taken(cls, lhs, rhs, chained: tuple, tiles: "Tiles", family: mosaic.Family, *, transpose_rhs: bool,
              parts: tuple) -> "Chained | None":
        """`lhs @ rhs` and `chained`'s `(rows, matrix)` products joined end to end where there are more than
        one, refused as `chained_refusal` says; None for one."""
        if not chained:
            return None
        refusal = chained_refusal(lhs, rhs, chained, tiles, family, transpose_rhs=transpose_rhs, parts=parts)
        if refusal is not None:
            raise ValueError(refusal)
        depths = (lhs.shape[1], *(rows.shape[1] for rows, _ in chained))
        return cls(cls.started(depths, tiles.block_k), depths)

    @property
    def depth(self) -> int:
        """Their depths together, the launch's contraction."""
        return sum(self.depths)

    @property
    def narrowed(self) -> int:
        """The factor `guarded_walk`'s column tile is narrower than the flagged walk's (`narrowing`)."""
        return narrowing(len(self.depths))

    def read(self, lhs_gmem, matrix_gmem, refs: tuple) -> tuple:
        """In the kernel, the rows and the matrix as the steps read them (`ChainedRefs`: the first's ref and
        the others', the next of `refs`, a product's rows then its matrix), and the refs past them."""
        count = 2 * (len(self.starts) - 1)
        others, refs = refs[:count], refs[count:]
        rows = ChainedRefs(self, (lhs_gmem, *others[0::2]))
        return (rows, ChainedRefs(self, (matrix_gmem, *others[1::2]))), refs


def replay_tile(tiles: "Tiles", products: int) -> "Tiles":
    """Dew: the tile a chained launch's replay walks (`guarded_walk`) for `products` products at `tiles`: its
    columns narrowed `narrowing` ways, each product its accumulator. The one replay plan the launch
    (`chained_refusal`, `replay_refusal`), its selection (`grouped_matmul.MosaicGmm._replays`) and its replay
    walk read. A stated output staging's chunk narrows with the columns, to the widest dividing both."""
    columns = tiles.block_n // narrowing(products)
    staging = None if tiles.staging is None else (math.gcd(tiles.staging[0], columns), tiles.staging[1])
    return dataclasses.replace(tiles, block_n=columns, staging=staging)


def replay_refusal(tiles: "Tiles", products: int, family: mosaic.Family, dtype) -> str | None:
    """Why a chained launch at `tiles` of `products` products on `family` with operands of `dtype` has no
    replay tile (`replay_tile`): its columns narrowed to no whole tile of them, or to no whole column unit of
    the family (`Family.column_unit`, the stores' and products' granularity: nine products narrowed a
    64-column Hopper tile to four, base review f86cb5b03 F1); None where it has one."""
    narrowed = narrowing(products)
    columns, unit = tiles.block_n // narrowed, family.column_unit(dtype)
    if tiles.block_n % narrowed or not columns or columns % unit:
        return (f"a chained launch's replay walks its tile's {tiles.block_n} columns {narrowed} ways, an "
                f"accumulator a product (`guarded_walk`): {columns} columns, no whole {unit}-column units")
    if tiles.staging is not None:
        # Its narrowed staging's chunk within the pieces its issuer emits of the narrowed columns.
        chunk = math.gcd(tiles.staging[0], columns)
        piece = family.chunk_columns(columns * (tiles.cluster if family.pairs_product else 1))
        if piece % chunk:
            return (f"a chained launch's replay stages its {columns} columns in chunks of {chunk}, which its "
                    f"issuer's {piece}-column pieces do not hold")
    return None


def narrowing(products: int) -> int:
    """Dew: the factor a chained launch's replay tile (`guarded_walk`) is narrower than its flagged walk's,
    for `products` products: their count's next power of two, so its accumulators, one a product, hold no
    more registers than the flagged walk's one, and a flagged tile is whole column tiles of it. A tile it
    leaves no columns of, or one the launch's own legality refuses (`grouped_matmul.MosaicGmm.legal`), takes
    no chain (c3b97e20f F2: nine products narrowed a 64-column tile to four, 65 to none)."""
    return pl.next_power_of_2(products)


@dataclasses.dataclass(frozen=True)
class ChainedRefs:
    """Dew: a chained launch's rows or matrices in a kernel (`Chained.read`), `gmems` its products' refs in
    order: step j's block read from the product it falls in, at that product's own step."""

    chained: Chained
    gmems: tuple

    def view(self, sliced: Callable, index_map: Callable, first=0) -> rings.Among:
        """The stream's view (`rings.Among`) of a walk from step `first` (a balanced tail's program's): each
        product's ref as `sliced` takes it (the tile's rows, or its column tile of the group's matrix), its
        step's block at `index_map` of its own step past `first`."""
        chained = self.chained
        return rings.Among(lambda j: chained.which(first + j), tuple(
            (sliced(gmem), lambda j, start=start: index_map(j - start))
            for gmem, start in zip(self.gmems, chained.starts, strict=True)))


def chained_refusal(lhs, rhs, chained: tuple, tiles: "Tiles", family: mosaic.Family, *, transpose_rhs: bool,
                    parts: tuple) -> str | None:
    """Dew: why a launch at `tiles` on `family` cannot join `lhs @ rhs` and `chained`'s products end to end
    (`Chained`), None where it can: each product's rows of the call's count and dtype, its matrix of the
    call's groups, width and dtype, each depth whole steps; no hooks, pair, joined rows, scales, tables,
    transposed or packed rows, banked matrices, exact sums, matrices side by side or depth split (`parts`,
    each None or False where the call has none), whose walks read one product a tile; a family whose step
    views choose their source at run time (`selects_sources`), whose products are no pair's."""
    if not family.selects_sources or family.pairs_product:
        return f"chained products choose each step's source at run time, which {family.name}'s walk does not"
    if any(part not in (None, False, ()) for part in parts):
        return "chained products take no hooks, pair, scales, tables, exact sums, beside or packed rows"
    if tiles.splits != 1:
        return f"chained products walk their depth whole, not split {tiles.splits} ways"
    replay = replay_refusal(tiles, 1 + len(chained), family, lhs.dtype)
    if replay is not None:
        return replay
    width_axis = 1 if transpose_rhs else 2
    for rows, matrix in ((lhs, rhs), *chained):
        depth = matrix.shape[2 if transpose_rhs else 1]
        if (rows.shape[0] != lhs.shape[0] or rows.dtype != lhs.dtype or matrix.dtype != rhs.dtype
                or matrix.shape[0] != rhs.shape[0] or matrix.shape[width_axis] != rhs.shape[width_axis]
                or rows.shape[1] != depth or depth % tiles.block_k or not depth):
            return (f"chained products are each the call's rows by its matrix's groups and width, a depth "
                    f"of whole {tiles.block_k}-steps, none empty: {rows.shape} {rows.dtype} by "
                    f"{matrix.shape} {matrix.dtype}")
    return None


def guarded_walk(walk: Callable, outs: tuple, tiles: "Tiles", products: int, programs: int) -> tuple:
    """Dew: a chained launch's outputs (`outs`, its flags of each tile's share last) with each flagged tile
    walked again: its one running sum went non-finite (an overflowing float32 sum stays so: an infinity added
    to anything finite is one, two of opposite signs NaN), so where the inputs are finite each product's own
    sums and their total may not be. The launch's contract (the 2026-11-19 ruling on base review c3b97e20f
    F1): its one accumulator reassociates the products' sum as any GEMM's walk reassociates its depth's, held
    by check 1 against float64 and finite wherever float64 and the plain float32 program are; not each
    product's float32 sum combined in order, which a finite cancellation across a product's boundary
    (1, then 2^30, then -2^30) leaves apart, as a single GEMM over the joined depth would. The second
    launch walks the flagged tiles alone (`when`), the rest idle,
    each product's steps into an accumulator of its own (`mosaic_gpu._gmm_step`'s `chosen`) and its epilogue
    adding them in the walk's order, the plain program's: each product's float32 sum over its own depth,
    their total rounded once, its operands as given (base review 0711d5f57 F2; 237e8e558 F2: rows scaled in
    their own dtype lost their least values). Its tile is `replay_tile`'s, so its
    accumulators hold the flagged walk's registers. Its stores land in the first launch's outputs in place
    (`into`, a mutable ref: no copy). Where no tile is flagged it walks nothing."""
    outs, flags = outs[:-1], outs[-1]

    def summed(*accs):
        return functools.reduce(jnp.add, accs)

    into = jax.new_ref(outs[0])
    walk(tiles=replay_tile(tiles, products), multiprocessors=programs, hooks=(None, (), summed, ()),
         sides=((), ()), tail=None, when=flags, into=into)
    return (jax.freeze(into),)


def flags_declared(grid: tuple, tiles: "Tiles") -> jax.ShapeDtypeStruct:
    """Dew: a flagged launch's flags (`_gmm_mainloop`'s `flagged`): int32, a row share of each tile its
    walk's `grid` schedules one (`[row tiles, column tiles, row shares]`, `Tiles.sharing`: each consumer's of
    a split tile, a ping-pong tile's one, as its one consumer finishes it): a row its table's entry's, a
    pair's two multiprocessors' each its own (`flag_row`), so no two tiles of a row tile (two groups', or an
    unused entry's: 237e8e558 F1) share one; 1 where its accumulators end non-finite (`_finished`), each
    written by its share's finish, none left unwritten."""
    return jax.ShapeDtypeStruct((grid[0] * tiles.cluster, grid[1], tiles.sharing), jnp.int32)


def flag_row(entry, pair: int, member):
    """Dew: the flags' row (`flags_declared`) of a pair's `member` at its walk's table `entry`."""
    return entry * pair + member


def step_unit(dtype, target) -> int | None:
    """Step depth of `target`'s primary source-selecting Mosaic bundle for `dtype`, or None if unsupported."""
    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import bundles

    if generation.tpu(target):
        return None
    for group in bundles.groups(target.generation, dtype, stated=True, tpu=False):
        for family in group:
            if family.selects_sources and not family.pairs_product:
                depths = family.depths(dtype)
                if depths:
                    return max(depths)
    return None


def padded_spec(rows, matrix, step: int, *, transpose_rhs: bool, dense: bool = True
                ) -> tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]:
    """Abstract padded `(rows, grouped_matrix)` shapes for route probing without materializing array pads."""
    depth = rows.shape[1] + (-rows.shape[1]) % step
    if dense:
        m_shape = (1, matrix.shape[0], depth) if transpose_rhs else (1, depth, matrix.shape[1])
    else:
        m_shape = ((matrix.shape[0], matrix.shape[1], depth) if transpose_rhs
                   else (matrix.shape[0], depth, matrix.shape[2]))
    return (jax.ShapeDtypeStruct((rows.shape[0], depth), rows.dtype),
            jax.ShapeDtypeStruct(m_shape, matrix.dtype))


def padded_pair(rows: jax.Array, matrix: jax.Array, step: int, *, transpose_rhs: bool, dense: bool = True
                ) -> tuple[jax.Array, jax.Array]:
    """Pad `(rows, matrix)` along depth to a positive multiple of `step` (adding a leading group axis when
    `dense`) so small LoRA/adapter terms join a base product in place via `chained` (row 1304)."""
    pad = (-rows.shape[1]) % step
    if pad:
        rows = jnp.pad(rows, ((0, 0), (0, pad)))
        if dense:
            matrix = jnp.pad(matrix, ((0, 0), (0, pad)) if transpose_rhs else ((0, pad), (0, 0)))
        else:
            matrix = jnp.pad(matrix, ((0, 0), (0, 0), (0, pad)) if transpose_rhs
                             else ((0, 0), (0, pad), (0, 0)))
    return rows, (matrix[None] if dense else matrix)


def joined_in_place(pairs: list[tuple[jax.Array, jax.Array]], *, group_sizes=None, transpose_rhs: bool,
                    out_dtype, precision, interpret_on_cpu: bool, launch_route: Callable, run_gemm: Callable,
                    gemm_fn: Callable, gemm_t_fn: Callable) -> jax.Array | None:
    """Walk dense or grouped `pairs` in place through `mosaic_gpu.gmm`'s `chained` launch without
    concatenating base operands, padding only small unaligned trailing adapters to `step` (row 1304)."""
    from jax.custom_derivatives import zero_from_primal

    from dew.nn.derivatives import KernelGlue, contracted
    from dew.nn.kernels import generation, xla_products
    from dew.nn.precision import at_least_fp32
    from dew.nn.sharding import automatic_axes

    dense = group_sizes is None
    dtype, out_d = pairs[0][0].dtype, jnp.dtype(out_dtype)
    target = (generation.interpreted_target()
              if xla_products.interpreted(interpret_on_cpu) else generation.gpu_target())
    step = step_unit(dtype, target)
    if step is None or pairs[0][0].shape[1] % step != 0:
        return None
    r0, m0 = pairs[0]
    r0_spec = jax.ShapeDtypeStruct(r0.shape, r0.dtype)
    m0_spec = jax.ShapeDtypeStruct((1, *m0.shape) if dense else m0.shape, m0.dtype)
    rest_spec = tuple(padded_spec(r, m, step, transpose_rhs=transpose_rhs, dense=dense)
                      for r, m in pairs[1:])
    sizes_spec = jax.ShapeDtypeStruct((1,) if group_sizes is None else group_sizes.shape, jnp.int32)
    options = {'interpret_on_cpu': interpret_on_cpu, 'precision': precision, 'transpose_rhs': transpose_rhs,
               'dense': dense, 'first': None, 'gather': None}
    route = launch_route(r0_spec, m0_spec, {'chained': rest_spec}, sizes_spec, out_d, None, options,
                         whole=not automatic_axes())
    if not route:
        return None
    sizes_arg = (jnp.full((1,), r0.shape[0], jnp.int32) if group_sizes is None
                 else group_sizes.astype(jnp.int32))
    wide = at_least_fp32(dtype)
    count = len(pairs)

    def forward(*flat):
        sizes = flat[-1]
        terms = [(flat[2 * i], flat[2 * i + 1]) for i in range(count)]
        padded = tuple(padded_pair(r, m, step, transpose_rhs=transpose_rhs, dense=dense)
                       for r, m in terms[1:])
        m0_in = terms[0][1][None] if dense else terms[0][1]
        return run_gemm(terms[0][0], m0_in, sizes, None, (), (), None, (), (), (out_d,),
                        several=False, transpose_rhs=transpose_rhs, interpret_on_cpu=interpret_on_cpu,
                        precision=precision, dense=dense, chained=padded, selected=route), ()

    def portable(*flat):
        sizes = flat[-1]
        if dense:
            terms = [xla_products.xla_dot(flat[2 * i],
                                          flat[2 * i + 1].T if transpose_rhs else flat[2 * i + 1],
                                          precision=precision, preferred_element_type=wide)
                     for i in range(count)]
        else:
            terms = [xla_products.xla_ragged_dot(
                         flat[2 * i],
                         jnp.swapaxes(flat[2 * i + 1], 1, 2) if transpose_rhs else flat[2 * i + 1],
                         sizes, precision=precision, preferred_element_type=wide)
                     for i in range(count)]
        return functools.reduce(jnp.add, terms).astype(out_d)

    def backward(saved, operands, cotangent, live):
        del saved
        g_sizes = None if dense else operands[-1]
        work = dtype if jnp.promote_types(cotangent.dtype, dtype) == dtype else at_least_fp32(cotangent.dtype)
        cot = cotangent.astype(work)
        wanted = set(live)
        leaves = []
        for i in range(count):
            ri, mi = operands[2 * i], operands[2 * i + 1]
            dr = (None if 2 * i not in wanted else
                  gemm_fn(cot, mi.astype(work), g_sizes, transpose_rhs=not transpose_rhs, out_dtype=ri.dtype,
                          precision=precision, interpret_on_cpu=interpret_on_cpu))
            if 2 * i + 1 not in wanted:
                dm = None
            else:
                dm = gemm_t_fn(ri.astype(work), cot, g_sizes, out_dtype=mi.dtype, precision=precision,
                               interpret_on_cpu=interpret_on_cpu)
                dm = (dm[0].T if transpose_rhs else dm[0]) if dense else (
                    jnp.swapaxes(dm, 1, 2) if transpose_rhs else dm)
            leaves.extend((dr, dm))
        leaves.append(zero_from_primal(operands[-1]))
        return tuple(leaves[index] for index in live)

    flat = (*(operand for pair in pairs for operand in pair), sizes_arg)
    fn = contracted("test_an_adapted_product_holds_every_reverse_order")(
        generation.linear_tangent(forward, backward, portable, live_cotangents=True, kind=KernelGlue()))
    return fn(*flat)
