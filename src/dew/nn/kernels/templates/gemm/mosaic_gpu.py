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

"""JAX's Pallas Mosaic GPU grouped matmuls for Hopper, vendored: `gmm`, `[rows, depth] x [groups, depth,
width]` over rows sorted by group (with `transpose_rhs`, `[groups, width, depth]`), and `tgmm`, each
group's rows transposed times their cotangent rows, `[groups, depth, width]`.

They are `ragged_dot` of `pallas/ops/gpu/ragged_dot_mgpu.py` and `transposed_ragged_dot` of
`pallas/ops/gpu/transposed_ragged_dot_mgpu.py` in the jax-v0.11.2 source tree (tag jax-v0.11.2,
32544801; license header above), which ship as examples: wgmma over pipelined TMA loads, persistent
over the multiprocessors, a planar snake over the output tiles. A tile is a block-aligned row block of
one group, a block two groups share running once for each, and a tile stores only its group's rows, in
a power-of-two ladder of TMA stores. Changed for Dew, each marked `# Dew:`:

- the rows past the routed ones are one more group, whose tiles multiply nothing, so each output there
  is the epilogue of a zero accumulator (zero without one), and a grid slot past the last tile, which
  upstream ran on group 0's matrix and stored nothing of,
  multiplies nothing;
- each writes the output dtype it is asked for (fp32 for a kernel gradient), not the operands';
- `tgmm` zeroes the neighbouring groups' rows of both operands by selection, where upstream
  multiplied one operand by a mask, which kept a neighbour's NaN or infinity (NaN * 0);
- the rows are padded to a multiple of 8, which a TMA descriptor's tiling needs (upstream's examples
  took row counts that were);
- the program count is the device's multiprocessors, where `ragged_dot` fixed an H100 SXM's 132;
- `tgmm`'s shared-memory tiling and swizzle are the warpgroup lowering's inference, as `gmm`'s are
  upstream, where upstream stated them with transforms whose type JAX leaves abstract;
- both take a cooperative tile (`Tiles.warpgroups`: two consumer warpgroups beside a producer, CUTLASS's
  cooperative sm90 schedule, as JAX's `hopper_matmul_mgpu`; `tgmm`'s operand stages stream across its
  persistent loop's tiles), `gmm` a pair of multiprocessors sharing one operand's blocks
  (`Tiles.cluster`), and the template's hooks (`gmm`, `tgmm`).

Each sums exact products of its 16-bit operands in fp32 (wgmma's accumulator) and rounds once to the
output dtype.
"""

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Any

import jax
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    axes,
    bodies,
    cost as unit_cost,
    mosaic,
    mosaic_sm100,
    outputs,
    resources,
    rings,
    stores,
    swizzling,
    tail as tail_of,
)
from dew.nn.kernels.templates.gemm import (
    packed as packing,
    quantized,
    row_tiles,
    rows_extended,
    rows_padded,
    scattered,
    sides_extended,
    sides_padded,
)
from dew.nn.kernels.templates.gemm.chained import Chained, Partition, flag_row, flags_declared, guarded_walk
from dew.nn.kernels.templates.gemm.pricing import dense_tail, family_of, tile_columns, tile_rows
from dew.nn.kernels.templates.gemm.row_tiles import TMA_ROWS
from dew.nn.kernels.templates.gemm.tiling import GMM, GUARDED_GMM, TGMM, Tiles
from dew.nn.kernels.templates.product import SCALE_DEPTH, block_format

JOINED = 'joined'
"""The stream of a joined product's second rows (`gmm`'s `joined`), each step's block multiplied by the paired
matrix's into the one accumulator."""


def row_unit(family: mosaic.Family, dtype) -> int:
    """Dew: the rows a tile's first row is a multiple of (`tile_groups`): a TMA descriptor's 8 (`TMA_ROWS`),
    or where a family's copies start at whole tiles of its layout (`mosaic.Transport.copies_tiled_rows`: a
    TPU core's 8 sublanes of 32-bit words, 16 rows of bf16), the rows of `dtype` they hold."""
    return 8 * 4 // jnp.dtype(dtype).itemsize if family.copies_tiled_rows else TMA_ROWS


@dataclasses.dataclass(frozen=True)
class Allocations:
    """Dew: a GEMM launch's shared memory at its tile, as its kernel scopes it (`gmm_allocations`,
    `tgmm_allocations`): its streams' blocks `(shape, dtype)` by name (`mosaic.Family.streams`), what it
    scopes beside them in the kernel's order (`staged`: side operands' staging, `mosaic.Transport.staging`,
    a meeting's barriers, `mosaic.Transport.meeting`), and its outputs' tiles `(shape, dtype)`, which the
    family stages (`mosaic.Family.persist`)."""
    blocks: dict
    staged: tuple
    outputs: tuple
    shared: tuple = ()
    """The streams a cluster's programs read alike, which they load once between them (`multicast`): `gmm`'s
    matrices, each row tile's the same column block; `tgmm`'s cotangent, each depth tile's the same rows and
    columns."""
    gathered: tuple = ()
    """The streams a producer gathers (`GATHERED_STREAMS`, a kernel gradient's `_tgmm_rings`;
    `mosaic.Family.streams`' `gathered`): their load barriers its family's own (`gathered.ring`'s 8-byte
    words at a 1024-byte alignment, which the planner counted as plain barriers: 976 bytes short of a B200's
    gathered input gradient's, row 5358, and of its gathered kernel gradients')."""

    def scope(self, family: mosaic.Family, tiles: "Tiles", steps: int | None = None,
              rewritten: bool = False, *, reduces: bool = False,
              reducing_warpgroups: int = 0) -> resources.Scope:
        """Its allocations by lifetime (`resources.shared_bytes`): its streams where its family places them
        (`mosaic.Family.lifetimes`, in no more slots than a walk's `steps` where they are known), and each
        output staged as its family stages it (`mosaic.Family.output_staging`, the allocations `persist`
        makes), after the walks where the family scopes them apart, its pipeline's memory reused; with
        `rewritten`, its streams' slots rewritten by its family's own role (`persist`'s `prepared`), whose
        publish barriers `persist` scopes too."""
        consumers = tiles.warpgroups
        staging = tuple(family.output_staging(self.outputs, consumers, tiles.staging))
        rewrites = tuple(tiles.stages for _ in self.blocks) if rewritten else ()
        persisted, addresses = family.persist_scratch(consumers, rewrites, pingpong=tiles.pingpong)
        streams = family.streams(self.blocks, tiles.stages, tiles.sharing,
                                 multicast=multicast(tiles, family, self.shared), gathered=self.gathered,
                                 joined=operand_streams(rewritten))
        held, walks = family.lifetimes(streams, steps)
        # A core holding its values in scratch memory holds them beside (`mosaic.Issuer.value_scratch`).
        values = family.value_scratch(self.blocks, self.outputs)
        if walks:
            return resources.Scope(program=(*held, self.staged, persisted, values), walks=walks,
                                   after=tuple((buffer.shape, buffer.dtype) for buffer in staging),
                                   addresses=addresses, reduces=reduces,
                                   reducing_warpgroups=reducing_warpgroups)
        return resources.Scope(program=(*held, self.staged, persisted, staging, values),
                               addresses=addresses, reduces=reduces,
                               reducing_warpgroups=reducing_warpgroups)

def _typed(dtype) -> jax.ShapeDtypeStruct:
    """A stand-in for an operand of `dtype`, as `mosaic.Transport.staging` reads an operand's dtype."""
    return jax.ShapeDtypeStruct((), dtype)


def gmm_allocations(tiles: "Tiles", family: mosaic.Family, *, dtype, rhs_dtype, transpose_rhs: bool,
                    prologue_axes: tuple, prologue_dtypes: tuple, epilogue_axes: tuple,
                    epilogue_dtypes: tuple, dtypes: tuple, paired: bool = False,
                    transpose_lhs: bool = False, packed: bool = False, step_scaled: bool = False,
                    block_scaled: bool = False, interpret: bool = False, gathered: bool = False,
                    packed_rows: bool = False, joined: bool = False,
                    packed_format: str | None = None) -> Allocations:
    """Dew: `gmm`'s allocations at `tiles` (`Allocations`), for rows of `dtype` and a matrix of `rhs_dtype`
    (two with `paired`), its side operands' axes and dtypes and its outputs' dtypes: a stream for the rows,
    each matrix and each prologue side operand that follows the steps (`_stepped`), each epilogue side
    operand's staging at a consumer's share of the tile, each prologue side operand without depth's at the
    tile's rows; a `transpose_lhs` call's rows stream their `[depth, rows]` blocks. A `packed` matrix (an
    NVFP4 one's, `scales`' in `gmm`, or a word-packed matrix's, `packed_format`) streams each group's
    sixteen uint32 words (`words_{q}`, a 64-byte row, which cp.async's two-dimensional copies take and a
    register tile loads whole) and (for NVFP4) its block scales K-major, decoded a stage at a time into
    bfloat16 buffers (`packed.stage_decoded` / `packed.stage_decoded_words`)."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    # A consumer's accumulators: its share of the rows by the tile's columns, a pair's both where its products
    # span it (`tile_columns`); each stream's slot a multiprocessor's block.
    part = (tiles.share, tile_columns(tiles, family))
    stepped, _, fixed = _stepped(family, prologue_axes, 'depth')
    extent = {'rows': block_m, 'depth': block_k}
    rhs_block = (block_n, block_k) if transpose_rhs else (block_k, block_n)
    lhs_block = (block_k, block_m) if transpose_lhs else (block_m, block_k)

    def words(across: int) -> dict:
        if packed_format is not None:
            group = packing.packed_group_depth(packed_format)
            return {f'words_{q}': ((across, 16), jnp.uint32) for q in range(block_k // group)}
        return {**{f'words_{q}': ((across, 16), jnp.uint32) for q in range(block_k // 128)},
                'scales': ((across, block_k // 16), jnp.float8_e4m3fn)}

    is_packed = packed or packed_format is not None
    matrix = words(block_n) if is_packed and not packed_rows else {'matrix': (rhs_block, rhs_dtype)}
    blocks = {**(words(block_m) if packed_rows else {'rows': (lhs_block, dtype)}), **matrix,
              **({'paired': (rhs_block, rhs_dtype)} if paired else {}),
              **({JOINED: (lhs_block, dtype)} if joined else {}),
              **(_scale_blocks(block_m, block_n) if step_scaled else {}),
              # A block-scaled product's rows' and matrix's scales, a step's tiles of each (`gmm`'s
              # `mx_scales`), four scales a tile's row (`product.block_format`).
              **(_scale_tiles(block_m, block_n, block_k, dtype) if block_scaled else {}),
              **{f'side_{i}': (axes.named(axes.held(prologue_axes[i]), extent), prologue_dtypes[i])
                 for i in stepped}}
    sides = family.staging('side', [(axes.named(axes.held(side_axes), {'rows': part[0], 'columns': part[1]}),
                                     _typed(side))
                                    for side_axes, side in zip(epilogue_axes, epilogue_dtypes, strict=True)])
    held = family.staging('fixed', [(axes.named(axes.held(prologue_axes[i]), extent),
                                     _typed(prologue_dtypes[i])) for i in fixed])
    shared = packing.shared_decode(tiles, interpret=interpret)
    decoded = ([] if not is_packed or packed_rows
               else [plgpu.SMEM((family.in_flight + 2, block_n, block_k), jnp.bfloat16)]
               if shared else [plgpu.SMEM((tiles.warpgroups, family.in_flight + 1, block_n, block_k),
                                          jnp.bfloat16)])
    return Allocations(blocks, (sides['side_smems'], sides['side_barriers'], held['fixed_smems'],
                                held['fixed_barriers'], decoded),
                       tuple((part, jnp.dtype(out)) for out in dtypes),
                       tuple(name for name in ('matrix', 'paired') if name in blocks),
                       GATHERED_STREAMS if gathered and not interpret else ())


def _scale_tiles(block_m: int, block_n: int, block_k: int, dtype) -> dict:
    """Dew: a block-scaled step's scale streams (`gmm_allocations`'): its rows' and its matrix's columns'
    scales of its `block_k` depths, one a block (`product.block_format`), in their tiles
    (`mosaic_sm100.scale_tiles`)."""
    scale, per = block_format(dtype)
    return {'a_scales': (mosaic_sm100.scale_tiles(block_m, block_k // per), scale),
            'b_scales': (mosaic_sm100.scale_tiles(block_n, block_k // per), scale)}


def _scale_blocks(block_m: int, block_n: int) -> dict:
    """Dew: a step-scaled call's scale streams (`gmm`'s and `tgmm`'s `step_scales`): a step's row of its
    rows' scales at the tile's rows and of its columns' at the tile's columns (a row of the matrix's coarser
    ones holding the tile's, `_column_scales`), unswizzled (`SCALE_SWIZZLES`)."""
    return {'row_scales': ((1, block_m), jnp.float32), 'column_scales': ((1, block_n), jnp.float32)}


SCALE_SWIZZLES = {'row_scales': swizzling.UNSWIZZLED, 'column_scales': swizzling.UNSWIZZLED}
"""Dew: the scale streams' swizzles (`_scale_blocks`): rows of fp32 a step, read a row at a time."""


def tgmm_allocations(tiles: "Tiles", family: mosaic.Family, *, lhs_dtype, rhs_dtype,
                     prologue_axes: tuple, prologue_dtypes: tuple, out_dtype, k_major: bool = False,
                     step_scaled: bool = False, unstaged: bool = False,
                     gathered: tuple[bool, ...] = (False, False), interpret: bool = False) -> Allocations:
    """Dew: `tgmm`'s allocations at `tiles` (`Allocations`): a ring for the rows, the cotangent and each
    prologue side operand with rows that follows the steps (`_stepped`), each side operand without rows'
    staging, the consumers' meeting where they prepare their stages (not `unstaged`, `rows_unstaged`), and
    its output's share of the tile; a cluster's depth tiles reading the cotangent's blocks alike, which they
    load once between them (`Allocations.shared`), but where its rows are gathered; the rings a producer
    gathers compiled (`gathered`: whether the rows' and the cotangent's are a table's, `_tgmm_rings`)."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    stepped, _, fixed = _stepped(family, prologue_axes, 'rows')
    extent = {'rows': block_k, 'depth': block_m}
    # A rows-contiguous call's blocks lie depth (or width) by rows, the rows their minor axis.
    blocks = {'ring_0': ((block_m, block_k) if k_major else (block_k, block_m), lhs_dtype),
              'ring_1': ((block_n, block_k) if k_major else (block_k, block_n), rhs_dtype),
              **(_scale_blocks(block_m, block_n) if step_scaled else {}),
              **{f'ring_{2 + position}': (axes.named(axes.held(prologue_axes[i]), extent),
                                          prologue_dtypes[i]) for position, i in enumerate(stepped)}}
    held = family.staging('fixed', [(axes.named(axes.held(prologue_axes[i]), extent),
                                     _typed(prologue_dtypes[i])) for i in fixed])
    # Two consumers meet at their halves' boundaries (`_staged_halves`); one meets none, nor do two reading
    # their stages as loaded.
    meeting = (family.meeting(tiles.warpgroups),) if tiles.warpgroups > 1 and not unstaged else ()
    # A consumer's share of the tile's rows by the columns its accumulators hold: a pair's both
    # (`tile_columns`), each stream's slot a multiprocessor's block.
    return Allocations(blocks, (held['fixed_smems'], held['fixed_barriers'], *meeting),
                       (((tiles.share, tile_columns(tiles, family)), jnp.dtype(out_dtype)),),
                       () if gathered[1] else ('ring_1',), () if interpret else _tgmm_rings(gathered))


def _stepped(family: mosaic.Family, axes_of: tuple, stepping: str) -> tuple[tuple, tuple, tuple]:
    """Dew: which side operands a launch streams with its steps (each with the `stepping` axis, but a vector a
    family does not copy, `mosaic.Family.copies_vectors`, its tile's axes those it holds, `axes.held`), which
    its step reads from global memory (such a vector), and which it loads once a tile (none with the
    `stepping` axis)."""
    stepped = tuple(i for i, side_axes in enumerate(axes_of)
                    if stepping in side_axes and (len(axes.held(side_axes)) > 1 or family.copies_vectors))
    read = tuple(i for i, side_axes in enumerate(axes_of) if stepping in side_axes and i not in stepped)
    fixed = tuple(i for i, side_axes in enumerate(axes_of) if stepping not in side_axes)
    return stepped, read, fixed


APPROX_MATH = False
"""Whether a GEMM launch compiles its special functions in the hardware's approximate forms
(`mosaic.Family.kernel`'s `approx_math`): never, a product's prologue and epilogue held to their accurate
forms (`emission.ACCURATE_FORMS`, which a fused region's price counts them in, `fused_gemm.priced`)."""

MOVES = True
"""Dew: whether a GEMM launch's schedule moves registers between its roles (`mosaic.Family.kernel`'s
`moves`): always, until its register models match ptxas at no move (row 5704: at the entry share ptxas
spilled the wgmma paired forward at 128 x 128 x 64, three stages, a pair, priced 160 against its 176; the
tcgen05 consumers' 112 to 192 against 64 to 112), where the move rule (`resources.moves_registers`) would
have its models decide."""


def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, transpose_rhs: bool,
        out_dtype, multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False,
        prologue=None, prologue_operands: tuple = (), prologue_axes: tuple = (), epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (),
        out_dtypes: tuple | None = None, dense: bool = False,
        paired: jax.Array | None = None, joined: jax.Array | None = None, exact_sums: bool = False,
        transpose_lhs: bool = False, first: jax.Array | None = None,
        scales: jax.Array | None = None, step_scales: tuple | None = None, mx_scales: tuple | None = None,
        gather: jax.Array | None = None, scatter: jax.Array | None = None, packed_rows: bool = False,
        registers: int | None = None, shared: int | None = None, balanced: bool | tail_of.Tail = True,
        beside: tuple = (), chained: tuple = (), packed_format: str | None = None
        ) -> jax.Array | tuple[jax.Array, ...]:
    """Each group's rows of `lhs` `[rows, depth]` times its matrix of `rhs`, `[groups, depth, width]` or with
    `transpose_rhs` `[groups, width, depth]`: `[rows, width]` in `out_dtype`, unrouted tail rows zeroed.
    With `dense`, one matrix multiplies every row without a grouped schedule. With `paired`, a second matrix
    of `rhs`'s shape multiplies into a second accumulator passed to `epilogue` or returned in `out_dtypes`;
    with `joined`, `lhs @ rhs + joined @ paired` (`_joined_refused`).
    With `transpose_lhs` (a dense call without a prologue), `lhs` is `[depth, rows]`, its blocks read
    transposed by the products: a product whose rows are the call's weight (its swapped orientation,
    `Tiles.swapped`). With `first`, an int32 scalar, `rhs` is a stack of more matrices than `group_sizes` has
    groups, group `g` reading matrix `first + g` where it lies (a banked layer's experts); the grid and the
    tile table are the groups'. With `scales`, `rhs` is an NVFP4 matrix's (`formats.NvFp4`), read transposed:
    its E2M1 codes eight a uint32 in the staged order (`[groups, width, depth / 8]`) and `scales` its blocks'
    E4M3 scales (`[groups, width, depth / 16]`); each step decodes its stage exactly into bfloat16 (each code
    times its block's scale over `bits.NVFP4_OWED`, `packed.stage_decoded`), so the product is the rows times
    those values, the deferred factor the caller's. Its depth step is whole 256s and its width tile whole 64s,
    with no hooks, pair, exact sums, cluster or split. With `packed_rows` too, the NVFP4 matrix is `lhs`
    (`[width, depth / 8]` words, `scales` `[width, depth / 16]`), the products' rows, and `rhs` the call's
    rows (`[1, rows, depth]`, read transposed): a decode-sized product swapped (row 5387), each step's rows
    decoded in registers into the products' left operand (`packed.rows_decoded`, `packed.rows_taken`). With
    `step_scales` (`row_scales` `[depth / 128, rows]`, `column_scales` `[groups, depth / 128, width / c]`,
    fp32), each 128-deep step's exactly summed partial is multiplied by its rows' scales and its group's
    scales of its columns, `c` columns a scale, before it is added (`xla_products.step_scaled`: DeepSeek-V3's
    blockwise FP8, row 4518); its step is 128 deep, with no hooks, pair, packed matrix or split. With
    `mx_scales`, an MX product (`_mx_refused`; row 4519). With `gather` (int32 `[rows]`), row `i` is
    `lhs[gather[i]]`, read where it lies by the producer (`gathered.Ring`; interpreted, `Ring.parts`); with
    `scatter` (int32 `[rows]`, a permutation), row `i` lands at the outputs' row `scatter[i]`, once
    (`stores.Scattered`): a mixture of experts' sort and its undoing fused into the loads and the stores (row
    1376), each refused as `_tables_refused` says.

Dew: `prologue(tile, *side_tiles)` transforms each `[block_m, block_k]` tile of `lhs` (a cooperative
    tile's consumer, half its rows) in its shared memory stage before the warpgroup MMA reads it;
    `epilogue(acc, *side_tiles)` each fp32 accumulator tile (a cooperative consumer's, half its rows),
    returning one output or a tuple of them (`out_dtypes`). A side operand's dimensions follow the launch
    axes its `*_axes` entry names, 'rows' and 'depth' for the prologue's, 'rows' and 'columns' for the
    epilogue's, each brought to shared memory by TMA. On a family whose products span a pair of
    multiprocessors (`mosaic.Transport.pairs_product`, tcgen05's pair), a tile is the pair's: twice
    `tiles.block_m` rows by twice `tiles.block_n` columns, each multiprocessor storing its own rows.
    `registers` and `shared` state its launch's register model (`mosaic.Family.kernel`'s;
    `grouped_matmul._Mosaic`'s prices), and `balanced` whether its registers hold a balanced last wave's
    merge (`grouped_matmul._Mosaic.balanced`), its tail unbalanced where they do not. With `beside`, further
    matrices whose products the launch takes with `rhs`'s side by side, each its own output (`Beside`: q, k
    and v's projections, row 6794). With `chained`, `(rows, matrix)` products summed with `lhs @ rhs` over
    their depths end to end, in order (`Chained`: a multi-matrix input gradient, row 8282; its overflowing
    tiles again, `guarded_walk`)."""
    fam = family_of(tiles, dtype := _products_dtype(lhs, {'matrix': rhs}, packed_rows))
    block_n = tile_columns(tiles, fam)
    if transpose_lhs and (not dense or prologue is not None):
        raise ValueError("a transposed lhs is a dense call's without a prologue")
    (k, rows), (matrices, k2, n) = lhs.shape if transpose_lhs else lhs.shape[::-1], rhs.shape
    g = matrices if group_sizes is None else group_sizes.shape[0]
    k2, n = (n, k2) if transpose_rhs else (k2, n)
    side_by_side = Beside.taken((rhs, *beside), tiles, fam, transpose_rhs=transpose_rhs,
                                parts=(prologue, epilogue, paired, joined, scales, step_scales, mx_scales,
                                       gather, scatter, transpose_lhs or None, packed_rows or None))
    n = n if side_by_side is None else side_by_side.width
    linked = Chained.taken(lhs, rhs, chained, tiles, fam, transpose_rhs=transpose_rhs,
                           parts=(prologue, epilogue, paired, joined, scales, step_scales, mx_scales, gather,
                                  scatter, transpose_lhs, packed_rows, first, exact_sums, side_by_side))
    k, k2 = packing.packed_depth(k, k2, n, rhs, tiles, packed_rows=packed_rows, scales=scales,
                                 packed_format=packed_format, transpose_rhs=transpose_rhs, prologue=prologue,
                                 paired=paired, transpose_lhs=transpose_lhs, lhs=lhs, family=fam, dense=dense)
    if packed_format is None:
        packing.refused(lhs, rhs, scales, tiles, fam, packed_rows=packed_rows, dense=dense,
                        transpose_rhs=transpose_rhs, transpose_lhs=transpose_lhs,
                        hooked=(prologue, epilogue, paired), depth=k2, width=n)
    packed_flag = scales if scales is not None else packed_format
    _gmm_shape_refused(lhs, rhs, tiles, (rows, k, matrices, k2, n), interpret, dtype=dtype, columns=block_n,
                       dense=dense, first=first, paired=paired, transpose_rhs=transpose_rhs,
                       transpose_lhs=transpose_lhs, exact_sums=exact_sums,
                       parts=(prologue, packed_flag, step_scales, mx_scales, gather, scatter))
    epilogue, epilogue_axes, epilogue_operands = _joined(
        joined, lhs, paired, (epilogue, epilogue_axes, epilogue_operands), transpose_lhs=transpose_lhs,
        parts=(prologue, packed_flag, step_scales, mx_scales, gather), packed_rows=packed_rows)
    k = k if linked is None else linked.depth
    rows = rows if gather is None else gather.shape[0]
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    if 0 in (k, rows):
        return _gmm_of_no_depth(side_by_side, (rows, n), group_sizes, paired, scatter, dtypes, out_dtypes,
                                (epilogue, epilogue_operands, epilogue_axes))
    _gmm_forms_refused(tiles, family_of(tiles, dtype), k, n, mx_scales, step_scales,
                       parts=(prologue, epilogue, paired, scales), transpose_rhs=transpose_rhs,
                       transpose_lhs=transpose_lhs, exact_sums=exact_sums)
    (lhs, gather, scatter, joined, m, extent), unit = _gmm_walked(lhs, gather, scatter, joined, tiles, dtype,
                                                                  transpose_lhs, dense, interpret)
    row_tile = tiles.block_m * tiles.cluster
    table = None if dense else row_tiles.grouped_table(group_sizes, m, row_tile, first, unit)
    step_scales, per = _step_scales_walked(step_scales, n, block_n, extent)
    prologue_operands = sides_extended(prologue_operands, prologue_axes, extent)
    epilogue_operands = sides_extended(epilogue_operands, epilogue_axes, extent)
    can_balance = dense and tiles.splits == 1 and not (beside or packed_format) and paired is scales is None
    tail = _dense_tail(tiles, dtype, (m, n, k), (lhs.dtype.itemsize, rhs.dtype.itemsize), multiprocessors,
                       balanced=balanced if can_balance else False)
    walked = (-(-m // row_tile) + (0 if dense else g)) * -(-n // block_n) * tiles.cluster * tiles.splits
    walk = functools.partial(
        _gmm_mainloop, lhs, rhs, table, (m, n, k, g), tiles=tiles, transpose_rhs=transpose_rhs,
        dtypes=dtypes if tiles.splits == 1 else (jnp.dtype(jnp.float32),), interpret=interpret, dense=dense,
        paired=paired, joined=joined, exact_sums=exact_sums, transpose_lhs=transpose_lhs, scales=scales,
        step_scales=step_scales, scale_per=per, mx_scales=mx_scales, gather=gather, scatter=scatter,
        stored=None if scatter is None else rows, packed_rows=packed_rows, registers=registers, shared=shared,
        beside=side_by_side, beside_matrices=beside, chained=linked, packed_format=packed_format,
        chained_operands=_chained_walked(chained, tiles, dtype, transpose_lhs, dense, interpret))
    outs = walk(multiprocessors=launched(fam, multiprocessors, walked, tail),
                hooks=(prologue, prologue_axes, epilogue, epilogue_axes),
                sides=(prologue_operands, epilogue_operands), tail=tail, flagged=linked is not None)
    if linked is not None:
        outs = guarded_walk(walk, outs, tiles, len(linked.depths),
                            launched(fam, multiprocessors, walked * linked.narrowed))
    outs = _gmm_outputs(outs, dtypes, rows, tiles.splits)
    return outs if out_dtypes or beside else outs[0]


def _gmm_outputs(outs: tuple, dtypes: tuple, rows: int, splits: int) -> tuple:
    """`gmm`'s outputs from its walk's, each its first `rows`: a split's runs' fp32 partials summed in their
    order (the same sums every call) and rounded once to each of `dtypes`."""
    if splits > 1:
        total = functools.reduce(jnp.add, tuple(outs[0]))
        outs = tuple(total.astype(dtype) for dtype in dtypes)
    return tuple(out[:rows] for out in outs)


def _joined(joined, lhs, paired, epilogue: tuple, *, parts: tuple, transpose_lhs: bool,
            packed_rows: bool) -> tuple:
    """Dew: `gmm`'s `epilogue` (the hook, its axes and side operands) as a call with `joined` rows runs it:
    refused but as `_joined_refused` takes them, with no epilogue of its own, its two sums added at the
    finish (`joined_sum`); the call's own where it joins none."""
    if joined is None:
        return epilogue
    _joined_refused(joined, lhs, paired, parts=(*parts, epilogue[0]), transpose_lhs=transpose_lhs,
                    packed_rows=packed_rows)
    return joined_sum, (), ()


def joined_sum(acc, paired_acc):
    """Dew: a joined pair's finish (`gmm`'s `joined`): its two fp32 accumulators added, once."""
    return acc + paired_acc


def _joined_refused(joined, lhs, paired, *, parts: tuple, transpose_lhs: bool, packed_rows: bool) -> None:
    """Dew: `gmm`'s refusal of `joined` rows but a pair's (`paired`) of `lhs`'s shape and dtype, with no
    prologue, scales, step or block scales, gather or epilogue (`parts`, in that order), transposed or packed
    rows.
    Joined, each step multiplies `lhs`'s block by `rhs`'s and `joined`'s by `paired`'s, each into its own
    accumulator, the two added at the finish and rounded once (`joined_sum`): no partial of one product
    meets the other's before either is whole, so each sum stays as finite as the product's alone (base
    review F2), and each product's promotion window is its own: a paired product's input gradient, the
    gate's and the up projection's cotangents through their matrices, one launch (row 6783)."""

    if joined is None:
        return
    if paired is None or joined.shape != lhs.shape or joined.dtype != lhs.dtype:
        raise ValueError(f"joined rows are a pair's, of the rows' shape and dtype: {joined.shape} "
                         f"{joined.dtype} beside {lhs.shape} {lhs.dtype}, paired {paired is not None}")
    if any(part is not None for part in parts) or transpose_lhs or packed_rows:
        raise ValueError("joined rows take no prologue, scales, gather, epilogue, transposed or packed rows")


def _dense_tail(tiles: Tiles, dtype, extents: tuple, itemsizes: tuple, multiprocessors: int, *,
                balanced: bool | tail_of.Tail):
    """Dew: a dense `gmm` or `tgmm` launch's last wave balanced over the programs (`pricing.dense_tail`),
    reusing the planner's `Tail` when passed so planner and emitter make one decision (row 11945)."""
    if isinstance(balanced, tail_of.Tail):
        return balanced
    if not balanced:
        return None
    return dense_tail(tiles, dtype, extents, itemsizes, generation.gpu_target(), multiprocessors, dense=True)


def _of_no_depth(shape: tuple, group_sizes, accumulators: int, epilogue, epilogue_operands: tuple,
                 epilogue_axes: tuple, dtypes: tuple) -> tuple:
    """Dew: `gmm`'s outputs of a product of no depth, which sums nothing, or of no rows, which has no tile to
    through the epilogue on whole arrays (its hooks are a tile's elementwise arithmetic), with no launch (a
    walk of no steps has no block of the matrix to view: JAX 0.12 refuses a zero-extent slice; a block of no
    rows none to tile: the interpreter's padding divided by it); its side operands by group read at each row
    as XLA's program reads them (`axes.at_rows`, row 4489)."""
    rows = shape[0]
    zeros = (jnp.zeros(shape, jnp.float32),) * accumulators
    by_row = tuple(axes.at_rows(side, side_axes, group_sizes, rows) if axes.GROUPS in side_axes else side
                   for side, side_axes in zip(epilogue_operands, epilogue_axes, strict=True))
    values = zeros if epilogue is None else epilogue(*zeros, *by_row)
    values = values if isinstance(values, tuple) else (values,)
    return tuple(value.astype(dtype) for value, dtype in zip(values, dtypes, strict=True))


def split_runs(steps: int, splits: int) -> bool:
    """Dew: whether `steps` split into `splits` runs of a share each (the last's the rest, `_gmm_scheduled`)
    leaves every run some steps."""
    return (splits - 1) * -(-steps // splits) < steps


def launched(family: mosaic.Family, multiprocessors: int, walked: int,
             tail: tail_of.Tail | None = None) -> int:
    """Dew: the programs a launch walking `walked` tiles runs: the `multiprocessors` its residency holds at
    once, persistent, where its family streams across a program's tiles (a producer's ring loading the next
    tile's steps under the last one's stores) or a balanced tail splits its last wave (`tail`); a program a
    tile where each walk scopes its streams apart (`mosaic.Family.walks_scoped`: cp.async's pipeline,
    filled and drained a tile), which the block scheduler starts in the walk's order as multiprocessors free
    up. A static persistent walk's programs drift apart over many rounds, where the scheduler keeps the
    running tiles a window of the walk's order: on an A100, a program a tile ran 0.37x and 0.43x the
    persistent walk's time on Mixtral-8x7B's down and DeepSeek-V3's gate input gradients, 0.80x and 0.90x
    on Mixtral's and Qwen3-30B-A3B's gate forwards, 0.88-0.95x on four grouped kernel gradients and 0.63x on
    Qwen3-32B's gate_up input gradient (a 51200-deep walk), and 0.99-1.03x on three shorter dense rows, the
    outputs bitwise equal (NS-P8-WALK-a100, NS-P8-WALK2-a100)."""
    return multiprocessors if tail is not None or not family.walks_scoped else max(walked, 1)


@tuning.lowered_once(('shape', 'tiles', 'transpose_rhs', 'dtypes', 'multiprocessors', 'interpret', 'hooks',
                      'dense', 'tail', 'exact_sums', 'transpose_lhs', 'scale_per', 'stored', 'packed_rows',
                      'registers', 'shared', 'beside', 'chained', 'flagged', 'packed_format'))
def _gmm_mainloop(lhs, rhs, table, shape: tuple, *, tiles: Tiles, transpose_rhs: bool, dtypes: tuple,
                  multiprocessors: int, interpret, hooks: tuple, sides: tuple, dense: bool = False,
                  transpose_lhs: bool = False, paired: jax.Array | None = None,
                  joined: jax.Array | None = None, tail=None, exact_sums: bool = False,
                  scales: jax.Array | None = None, step_scales: tuple | None = None, scale_per: int = 1,
                  stored: int | None = None, packed_rows: bool = False, registers: int | None = None,
                  mx_scales: tuple | None = None, gather=None, scatter=None, shared: int | None = None,
                  beside: "Beside | None" = None, beside_matrices: tuple = (), when=None, into=None,
                  chained=None, chained_operands: tuple = (), flagged: bool = False,
                  packed_format: str | None = None) -> tuple:
    """Dew: `gmm` on its instruction family's bundle (`family_of`): `tiles.warpgroups` consumer roles, each
    its share of the rows by all the columns, one MMA a step (two: CUTLASS's cooperative split, 2-7% faster
    the family owning the operands' streams, their staging and the stores (`mosaic.Family.persist`). Each
    consumer's prologue rewrites its share of the rows' slot, its epilogue takes its share of each side
    tile. `lhs` is padded; `table` is `tile_groups`' at the pair's rows (`gmm`). At `tiles.cluster` 2 a pair
    of multiprocessors runs two row tiles of one group, loading its matrix block once (`Family.streams`'
    multicast), both walking each pair's steps; products spanning the pair take its blocks (`_gmm_views`).
    A `paired` matrix streams beside the first, its blocks times the same rows' (or `joined` rows') into a
    second accumulator. With a balanced `tail` (a dense product's), the tiles past the whole waves share
    their steps over the programs, the program walking a tile's first step adding its partials and storing
    it (`tail.Tail`, `mosaic.Family.fixed_up`), once a configuration; `chained` on: `guarded_walk`'s."""
    family = family_of(tiles, _products_dtype(lhs, {'matrix': rhs}, packed_rows))
    (prologue, prologue_axes, epilogue, epilogue_axes), (prologue_operands, epilogue_operands) = hooks, sides
    (m, n, k, g), block_m, block_n, block_k = shape, tiles.block_m, tile_columns(tiles, family), tiles.block_k
    consumers, pair, part = tiles.warpgroups, tiles.cluster, (tiles.share, block_n)
    pairs = multiprocessors // pair
    stepped, read, fixed = _stepped(family, prologue_axes, 'depth')
    rewritten = prologue_rewritten(family, prologue)
    allocations = _gmm_allocated(tiles, family, (lhs, rhs, paired, scales, step_scales, mx_scales, gather),
                                 hooks, sides, dtypes, transpose_rhs=transpose_rhs, packed_rows=packed_rows,
                                 transpose_lhs=transpose_lhs, interpret=bool(interpret), joined=joined,
                                packed_format=packed_format)
    bounded = None if when is None else chained  # each product its own accumulator (`guarded_walk`)
    matrices, streamed, accumulators = _gmm_streamed(paired, scales, joined, bounded)
    # A pair's tiles: g groups' boundaries, and the unrouted rows.
    grid = (pl.cdiv(m, block_m * pair) + (0 if dense else g), pl.cdiv(n, block_n))
    def body(*gmems):
        (table_gmem, lhs_gmem, matrix_gmems), (scale_gmems, mx_gmems, gather_gmem), refs = _gmm_refs(
            gmems, (step_scales, mx_scales, gather, scatter), (len(streamed), len(prologue_operands),
                                                                len(epilogue_operands)),
            dense=dense, beside=beside, tail=tail, interpret=bool(interpret), chained=chained,
            guarded=(when is not None, into is not None, flagged))
        prologue_gmems, side_gmems, workspace, out, (chained_gmems, when_gmem, flags_gmem) = refs
        ready = None if tail is None else family.partials_ready((pair, pairs, consumers), bool(interpret))

        schedule = functools.partial(
            _gmm_scheduled, table_gmem=table_gmem, lhs_gmem=lhs_gmem, splits=tiles.splits, grid=grid,
            matrix_gmems=dict(zip(streamed, matrix_gmems, strict=True)), prologue_gmems=prologue_gmems,
            shape=(m, pl.cdiv(k, block_k), g), tiles=tiles, dense=dense, tail=tail, programs=pairs,
            transpose_rhs=transpose_rhs, axes_of={i: prologue_axes[i] for i in stepped}, traced=exact_sums,
            interpreted=bool(interpret), transpose_lhs=transpose_lhs, packed_rows=packed_rows,
            scale_gmems=scale_gmems, scale_per=scale_per, mx_gmems=mx_gmems, gather_gmem=gather_gmem,
            when_gmem=when_gmem, chained_gmems=chained_gmems, bounded=bounded, packed_format=packed_format)
        stepped_at = functools.partial(
            _gmm_stepped_at, family=family, tiles=tiles, dtype=lhs.dtype, prologue=(prologue, prologue_axes),
            matrices=matrices, transpose=(transpose_lhs, transpose_rhs), interpret=bool(interpret),
            mx=mx_scales is not None, packed_rows=packed_rows, packed_format=packed_format, dense=dense,
            chosen=None if when is None else chained,
            scaled=(step_scales is not None, scale_per, block_n), exact_sums=exact_sums)
        def scope(side_smems, side_barriers, fixed_smems, fixed_barriers, decoded, *, streams):
            def issue(wg, tile, walk, *, active):
                group, row_start, *_ = tile
                if tail is not None and not active:
                    return None  # past its program's range: another program walks the tile
                fixed_refs = [smem.at[wg] for smem in fixed_smems]
                if active and not rewritten:
                    _fixed_loaded(family, prologue_gmems, prologue_axes, fixed, fixed_refs, fixed_barriers,
                                  wg=wg, rows=pl.ds(row_start, block_m))
                step, prefetch = stepped_at(
                    wg, tile, fixed_tiles=dict(zip(fixed, fixed_refs, strict=True)),
                    read_tiles=_reads(prologue_gmems, prologue_axes, read, group, g),
                    decoded=packing.decoding(decoded, tiles, wg, bool(interpret), weight_fmt=packed_format))
                return _walked_products(family, walk, step, part, accumulators, exact_sums, prefetch)
            def finish(wg, tile, accs, store, *, active):
                if (tail is not None or when is not None) and not active:
                    return  # another program's tile, or one a bounded walk's flags name not (`guarded_walk`)
                finished = functools.partial(
                    _gmm_finished, accs, wg=wg, store=store, family=family, part=part,
                    share_row=tiles.share_row(wg), m=m, g=g, dense=dense, row_tile=block_m * pair,
                    epilogue=epilogue, flags=(flags_gmem, pair, tiles.sharing),
                    sides=tuple(zip(side_gmems, epilogue_axes, side_smems, side_barriers, strict=True)),
                    partials=None if tail is None else (workspace, ready))
                _stored(finished, tile, out, beside)
            walks = (tiles.splits * grid[0] * grid[1],) if tail is None else (tail.segments, pairs)
            family.persist(streams, walks, schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=allocations.outputs, interpret=bool(interpret),
                           in_flight=family.in_flight, unrolled=_unrolled(prologue, packed_rows, exact_sums),
                           accumulators=(part,) * accumulators, **_summed(family, tiles, exact_sums),
                           producer=tiles.producers.of(GMM),
                           **({'pingpong': True} if tiles.pingpong else {}),
                           chunks=tiles.stated(allocations.outputs, family),
                           **_prologue_rewrite(rewritten, family, prologue, (prologue_axes, prologue_gmems),
                                               (stepped, read, fixed), (fixed_smems, fixed_barriers), g))
        family.scoped(scope, *allocations.staged,
                      streams=family.streams(allocations.blocks, tiles.stages, tiles.sharing,
                                             multicast=multicast(tiles, family, allocations.shared),
                                             swizzles=SCALE_SWIZZLES if step_scales is not None else None,
                                             joined=operand_streams(rewritten),
                                             **_gathered(gather, tiles, lhs.dtype, interpret)))
    out_type = () if into is not None else (
        *_declared(lhs, rhs, gather, (m, n, stored), tiles.splits, dtypes, transpose_lhs, transpose_rhs,
                   beside, beside_matrices), *((flags_declared(grid, tiles),) if flagged else ()))
    operands = _gmm_operands(lhs, rhs, paired, scales, step_scales, mx_scales, gather, scatter, joined,
                             beside_matrices, (chained_operands, when, into))
    kernel = family.kernel(body, out_type=out_type, grid=(multiprocessors // pair,), grid_names=("programs",),
                           consumers=consumers + rewritten, interpret=interpret, approx_math=APPROX_MATH,
                           name=GMM if when is None else GUARDED_GMM, cluster=pair, resident=tiles.programs,
                           scalars=0 if dense else 1, model_flops=None if when is None else 0,  # replay: 0
                           registers=registers, producer=tiles.producers.of(GMM), shared=shared, moves=MOVES,
                           cost=unit_cost.estimate(2 * m * n * k * len(matrices), *operands,
                                                   *prologue_operands, *epilogue_operands, *out_type),
                           **_prefetched(tiles, grid, k // block_k, transpose_rhs, interpret, dense=dense,
                                         parts=(tail, scales if scales is not None else packed_format,
                                                mx_scales, gather, chained), beside=beside))
    return kernel(*(operands if dense else (table, *operands)), *prologue_operands, *epilogue_operands,
                  *_workspace(tail, (pair, pairs, 2, consumers, *part), interpret))


def _gmm_of_no_depth(beside: "Beside | None", shape: tuple, group_sizes, paired, scatter, dtypes: tuple,
                     out_dtypes, epilogue: tuple):
    """Dew: `gmm`'s outputs of a call of no depth or no rows: each matrix side by side's zeros, or its
    epilogue's value on zero accumulators (`_of_no_depth`), a scattered product's rows where its table says,
    as its stores put them."""
    rows, n = shape
    if beside is not None:
        return tuple(jnp.zeros((rows, width), dtypes[0]) for width in beside.widths)
    hook, operands, axes_of = epilogue
    outs = scattered(_of_no_depth((rows, n), group_sizes, 1 if paired is None else 2, hook, operands, axes_of,
                                  dtypes), scatter)
    return outs if out_dtypes else outs[0]


def _gmm_forms_refused(tiles: Tiles, family: mosaic.Family, k: int, n: int, mx_scales, step_scales, *,
                       parts: tuple, transpose_rhs: bool, transpose_lhs: bool, exact_sums: bool) -> None:
    """Dew: `gmm`'s refusal of an MX product's or a depth split's form it cannot run (`_mx_refused`,
    `_gmm_split_refused`), `parts` its prologue, epilogue, pair and scales."""
    hooked = any(each is not None for each in parts[:3])
    _mx_refused(mx_scales, tiles, family, transpose_rhs=transpose_rhs,
                plain=all(each is None for each in (*parts, step_scales))
                and not (transpose_lhs or exact_sums))
    _gmm_split_refused(tiles, k, n, step_scales, exact_sums=exact_sums, hooked=hooked,
                       plain=all(each is None for each in parts) and not transpose_lhs)


def _chained_walked(chained: tuple, tiles: Tiles, dtype, transpose_lhs: bool, dense: bool,
                    interpret) -> tuple:
    """Dew: a chained call's further products' operands as its walk reads them, each product's rows then its
    matrix: the rows padded and extended as the call's (`_gmm_walked`)."""
    return tuple(array for rows, matrix in chained for array in (
        _gmm_walked(rows, None, None, None, tiles, dtype, transpose_lhs, dense, interpret)[0][0], matrix))


def _gmm_stepped_at(wg, tile, *, family: mosaic.Family, tiles: Tiles, dtype, prologue: tuple, matrices: dict,
                    transpose: tuple, interpret: bool, mx: bool, packed_rows: bool, scaled: tuple,
                    exact_sums: bool, dense: bool = False, chosen=None, packed_format: str | None = None,
                    **refs) -> tuple[Callable, Callable | None]:
    """Dew: consumer `wg`'s step of `tile` (`_gmm_step`), idleness, row bound (`_row_bound`), and prefetch."""
    rows = pl.ds(tiles.share_row(wg), tiles.share)
    (hook, hook_axes), (transpose_lhs, transpose_rhs) = prologue, transpose
    idle = _idle(tiles, tile[0], tile[1], wg)
    within = _row_bound(family, tiles, dtype, tile[0], tile[1], dense)
    step_family = family if within is None else family.bounded(within)
    step = functools.partial(
        _gmm_step, family=step_family, prologue=hook, prologue_axes=hook_axes, mine_rows=rows, idle=idle,
        transpose_rhs=transpose_rhs, matrices=matrices, interpret=interpret, first=tile[3],
        transpose_lhs=transpose_lhs, block_scaled=mx, packed_rows=packed_rows, packed_format=packed_format,
        block_k=tiles.block_k, scaling=_gmm_scaling(scaled[0], scaled[1], scaled[2], tile[2], rows),
        chosen=chosen, **refs)
    prefetching = idle is None and prefetches(
        family, matrices=len(matrices), prologue=hook, scaled=refs.get('decoded') is not None or mx,
        chained=chosen is not None, interpret=interpret,
        transposed=(transpose_lhs or packed_rows, transpose_rhs),
        summed=exact_sums and not family.sums_in_persist)
    return step, _gmm_prefetch(step_family, matrices, rows) if prefetching else None


def prefetches(family: mosaic.Family, *, matrices: int, prologue, scaled: bool, chained: bool,
               interpret: bool, transposed: tuple[bool, bool], summed: bool) -> bool:
    """Dew: whether a `gmm` walk reads a step's first fragments under the previous step's last product
    (`_gmm_prefetch`, `Transport.prefetches_fragments`): plain, unscaled, unchained, compiled, untransposed
    (transposed-RHS prefetch regressed A100 deep-K gate_up dX by +32-34% in M-FS-53-a100 & M-NG-2-a100), and
    tensor-core summed."""
    return (family.prefetches_fragments and matrices == 1 and prologue is None and not scaled and not chained
            and not interpret and not any(transposed) and not summed)
# `tgmm` does not prefetch fragments across steps on `mma.sync` (`M-NG-45-a100` 20261123T015937Z: dW
# geomean 1.019x, 9/13 slower, 4B key +11.6%, confirming `FS-DENSE-AB1`).


def _gmm_streamed(paired, scales, joined, bounded: "Chained | None" = None) -> tuple[dict, tuple, int]:
    """Dew: `_gmm_mainloop`'s matrices (its own, a paired one), each to the stream of the rows it multiplies
    (the rows', the joined rows' for a joined pair's paired matrix), the streams beyond the rows in its
    operands' order (`_gmm_operands`: a packed matrix's scales beside it, the joined rows beside the paired
    matrix) and its accumulators, one a matrix, or a `bounded` walk's chained products' each one
    (`chained.guarded_walk`)."""
    second = {} if paired is None else {'paired': 'rows' if joined is None else JOINED}
    matrices = {'matrix': 'rows', **second}
    streamed = (*matrices, *(() if scales is None else ('scales',)), *(() if joined is None else (JOINED,)))
    return matrices, streamed, len(matrices) if bounded is None else len(bounded.depths)


def _gmm_operands(lhs, rhs, paired, scales, step_scales, mx_scales, gather, scatter, joined=None,
                  beside: tuple = (), guarded: tuple = ()) -> tuple:
    """Dew: `gmm`'s kernel operands in the order its body reads them: the rows, each streamed operand (its
    matrix, a paired one, a packed one's scales, joined rows), the step-wise and block scales, the gather and
    scatter tables, the matrices side by side past the first (`Beside.read`), each where the call has it,
    and `guarded`'s: chained products' operands, a bounded walk's flags and the ref it stores into
    (`chained.guarded_walk`)."""
    present = (*(() if paired is None else (paired,)), *(() if scales is None else (scales,)),
               *(() if joined is None else (joined,)), *(step_scales or ()), *(mx_scales or ()),
               *(() if gather is None else (gather,)), *(() if scatter is None else (scatter,)),
               *beside, *(guarded[0] if guarded else ()),
               *(each for each in guarded[1:] if each is not None))
    return lhs, rhs, *present


def _prefetched(tiles: Tiles, grid: tuple, steps: int, transpose_rhs: bool, interpret, *, dense: bool,
                parts: tuple, beside: "Beside | None" = None) -> dict:
    """Dew: a dense `gmm`'s matrix prefetched into L2 before its launch's wait for the one before it
    (`bodies.dependent`'s `prefetched`, `bodies.waits`): each program's first tile's matrix blocks of its
    first stages, so its first loads hit L2 rather than wait on memory under the previous launch's tail
    (M-FS-12: a decode GEMM's prologue exposed 3-5%; row 3601), of its `steps` whose block lies whole in the
    matrix (a partial last step's, `partial_runs`, would read past its depth). A plain dense product's alone
    (no tail, scales or gathered rows: `parts` all None), whose first tile a program's index states (its
    persistent walk's first, `persistent.tiled`); a grouped one's needs its routing, written before the
    wait. A prefetch reads nothing into the program, so a matrix the launch before it still writes is
    prefetched early, never read stale. Matrices side by side (`beside`, the operands after the first's)
    prefetch the first tile's own matrix's blocks (`BesideRefs.columns`)."""
    plain = dense and all(part is None for part in parts)
    if not (plain and tiles.splits == 1 and tiles.cluster == 1
            and bodies.waits(interpret)):
        return {}
    block = (tiles.block_n, tiles.block_k) if transpose_rhs else (tiles.block_k, tiles.block_n)

    def prefetched(lhs_gmem, matrix_gmem, *refs):
        # The matrix is the launch's second operand, after the rows (`gmm`'s operands: a dense call's
        # take no table), the matrices side by side next.
        del lhs_gmem
        _, column_tile = tail_of.planar_snake(lax.axis_index("programs"), grid, 1, tiles.grid_block_n)
        if beside is None:
            view = (_matrix_columns(matrix_gmem.at[0], column_tile, block_n=tiles.block_n,
                                    transpose_rhs=transpose_rhs), _rhs_index(transpose_rhs))
        else:
            (side_by_side,), _ = beside.read((matrix_gmem,), refs)  # its refs bound as the body binds them
            view = side_by_side.columns(0, column_tile, _rhs_index(transpose_rhs))
        return [rings.viewed(view, block, step) for step in range(min(tiles.stages, steps))]

    return {'prefetched': prefetched}


def _declared(lhs, rhs, gather, shape: tuple[int, int, int | None], splits: int, dtypes: tuple,
              transpose_lhs: bool, transpose_rhs: bool, beside: "Beside | None" = None,
              beside_matrices: tuple = ()) -> tuple:
    """Dew: a `gmm` launch's outputs, `[rows, n]` each, a depth split's each run's partial a slice of its own
    (`[splits, rows, n]`), stated as its operands' axes (`outputs.output`): its rows `lhs`'s (a gathered
    product's table's, `gather`), its columns `rhs`'s width (each as its transpose lays it). A
    scattered product declares its stored rows alone, which its table's entries land on, each once: its padded
    rows land nowhere, so rows past them would be declared and never written (NSNAN-2); that count is its
    call's, no operand axis of the launch, so its outputs state their shapes alone. Matrices side by side
    declare each one's own (`Beside.declared`), `beside_matrices` those past `rhs`."""
    m, n, stored = shape
    if beside is not None:
        return beside.declared(lhs, (rhs, *beside_matrices), dtypes[0], m)
    if stored is not None:
        return tuple(jax.ShapeDtypeStruct((stored, n) if splits == 1 else (splits, stored, n), dtype)
                     for dtype in dtypes)
    # Its rows the walked rows' less their extension past the last tile (`_walked_rows`), sized by the launch.
    rows = outputs.axis(lhs, 1 if transpose_lhs else 0) if gather is None else outputs.axis(gather, 0)
    rows = rows.sized(m)
    width = outputs.axis(rhs, 1 if transpose_rhs else 2)
    if width.size != n:
        raise ValueError(f"a gmm's outputs are its matrix's width, {n}, not {width.size}")
    dims = (rows, width) if splits == 1 else (splits, rows, width)
    return tuple(outputs.output(dtype, *dims) for dtype in dtypes)


def _gmm_finished(accs: tuple, tile: tuple, wg, store, *, family: mosaic.Family, part: tuple, share_row,
                  m: int, g: int, dense: bool, row_tile: int, epilogue, o_gmems: tuple, sides: tuple,
                  partials, flags: tuple = (None, 1, 1)) -> None:
    """Dew: a consumer's share of a `gmm` tile stored (`_finished`) from its accumulators: at once, or,
    a balanced tail's tile (`partials`: its workspace and readiness flags), once its program's partial and
    the other programs' are added (`mosaic.Family.fixed_up`); with `flags` (`chained.flags_declared`'s, the
    pair and the consumers sharing a tile's rows, `Tiles.sharing`), its row share's written at its table
    entry (`chained.flag_row`): a ping-pong consumer's whole tile's the one slot, as it alone finishes the
    tile (`persistent`'s ping-pong walk; 3b97e20f F5)."""
    group, row_start, ni, _, gathered, program, split, entry = tile
    flags_gmem, pair, sharing = flags
    nonfinite = None if flags_gmem is None else flags_gmem.at[
        flag_row(entry, pair, mosaic.cluster_index(pair)), ni, pl.ds(wg if sharing > 1 else 0, 1)]
    # A depth split's run stores its partial in its own slice of each output.
    o_gmems = o_gmems if split is None else tuple(
        dataclasses.replace(o_gmem, ref=o_gmem.ref.at[split]) if isinstance(o_gmem, stores.Scattered)
        else o_gmem.at[split] for o_gmem in o_gmems)
    # This consumer's share of the tile: its first row, and its group's rows there.
    part_row, column = row_start + share_row, ni * part[1]
    first, stored = _stored_rows(group, part_row, part[0], m, dense=dense, row_tile=row_tile)
    finished = functools.partial(
        _finished, family=family, wg=wg, epilogue=epilogue, o_gmems=o_gmems, store=store, sides=sides,
        origins={'rows': pl.ds(part_row, part[0]), 'columns': pl.ds(column, part[1])},
        group=_side_group(group, g), nonfinite=nonfinite,
        place=(part_row, first, stored, column, min(part[0], m), TMA_ROWS if dense else 1))
    if partials is None or partials[0] is None:
        finished(*accs)
        return
    workspace, ready = partials
    first, gathered = gathered
    cta = mosaic.cluster_index(pair)
    family.fixed_up(accs[0], workspace.at[cta, :, 0, wg], None if ready is None else ready.at[cta, :, wg],
                    program, first, gathered, finished, firsts=workspace.at[cta, :, 1, wg])


def _stored_rows(group, part_row, rows: int, m: int, *, dense: bool, row_tile: int) -> tuple:
    """Dew: a consumer's share of a tile, `rows` from `part_row`: the first of them its group's, and how many
    (a dense product's every row to the padded rows `m`, whole where they fill `row_tile`s)."""
    if dense:
        return 0, (rows if m % row_tile == 0 else jnp.clip(m - part_row, 0, rows))
    first = jnp.clip(group.actual_start - part_row, 0, rows)
    return first, jnp.clip(group.actual_end - part_row, 0, rows) - first


def _prologue_rewrite(rewritten: bool, family: mosaic.Family, prologue, operands: tuple, sides: tuple,
                      held: tuple, groups: int) -> dict:
    """Dew: `gmm`'s persist argument where its family's own role rewrites its stage by the prologue
    (`rewritten`, `prologue_rewritten`; the consumers' steps then read the stage as rewritten): each step's
    rows stage rewritten (`_prologue_stage`), the prologue's axes and side operands `operands`, its
    side tiles without depth loaded into the first of each of the `held` staging's refs and barriers; none
    elsewhere."""
    if not rewritten:
        return {}
    (prologue_axes, prologue_gmems), (smems, barriers) = operands, held
    return {'prepared': functools.partial(_prologue_stage, family, prologue, prologue_axes, prologue_gmems,
                                          sides, ([smem.at[0] for smem in smems], barriers), groups=groups)}


def _prologue_stage(family: mosaic.Family, prologue, prologue_axes: tuple, prologue_gmems, sides: tuple,
                    held: tuple, tile, index, slots, *, groups: int) -> None:
    """Dew: a `gmm` step's rows stage rewritten whole by its prologue in its family's own role
    (`prologue_rewritten`, `mosaic.Family.persist`'s `prepared`), in the layouts that role reads it in
    (`mosaic.Family.rewriting`, `_rewritten`): its side tiles the step's slots of those streamed
    (`sides`' first), the step's depth of those read from global memory (its second, at the tile's group,
    `_reads`), and those without depth (its third) loaded once a tile, at its first step, into the role's
    `held` refs and barriers (`_fixed_loaded`, a consumer's own where its consumers take the prologue); a
    role's rows at a time (`Family.role_rows` of the layouts it reads in), so its registers hold that many
    rows' tiles whatever the stage's (a gated prologue's two tiles of 128 rows widened would hold 160)."""
    stepped, read, fixed = sides
    refs, barriers = held
    group, row_start, _, first = tile[:4]
    stage = slots['rows']
    block_m, depth = stage.shape

    @pl.when(index == 0)
    def _():
        _fixed_loaded(family, prologue_gmems, prologue_axes, fixed, refs, barriers, wg=0,
                      rows=pl.ds(row_start, block_m))

    reads = _reads(prologue_gmems, prologue_axes, read, group, groups)
    tiles_of = {**{i: slots[f'side_{i}'] for i in stepped},
                **{i: reads[i].at[pl.ds((first + index) * depth, depth)] for i in read},
                **dict(zip(fixed, refs, strict=True))}
    rows = family.rewriting.role_rows
    for first_row in range(0, block_m, rows):
        part = pl.ds(first_row, rows)
        _rewritten(family.rewriting, prologue, prologue_axes, stage.at[part], tiles_of, 'rows', part)


def _fixed_loaded(family: mosaic.Family, prologue_gmems, prologue_axes: tuple, fixed, refs, barriers, *, wg,
                  rows) -> None:
    """Dew: a consumer's prologue side tiles without depth (`fixed`: loaded once a tile, at its `rows`) into
    its `refs`, each in the layout of its own axes (`Family.loaded`)."""
    family.loaded([(prologue_gmems[i].at[rows], ref, barrier.at[wg])
                   for i, ref, barrier in zip(fixed, refs, barriers, strict=True)],
                  axes=tuple(axes.held(prologue_axes[i]) for i in fixed))


def _finished(*accs, family: mosaic.Family, wg, epilogue, o_gmems: tuple, store, sides: tuple, origins: dict,
              group, place: tuple, nonfinite=None) -> None:
    """Dew: a consumer's share of a `gmm` tile from its accumulators (`accs`), read out a chunk of columns
    at a time (`mosaic.Issuer.chunks`: all of them at once where its registers hold them): its epilogue on
    its side tiles' same columns (`sides`, each its global ref, axes, staging and barrier, loaded at
    `origins`, a side operand by group at the tile's `group`, the unrouted rows' the last's, as XLA's
    program reads them, and handed the epilogue at each of the share's rows), then each output stored at
    `place`, its column the chunk's (`mosaic.Family.persist`'s `store`), in its output's dtype; with
    `nonfinite` (a one-entry int32 ref, `flags_declared`), whether any value the share stores is non-finite
    written there: the stored value's, converted once, as a finite fp32 sum can overflow its output's dtype
    (base review b1ab251ac F1: 65536, the joined sum where the products' was 65504, an fp16 infinity), of its
    stored rows alone: another group's, by this entry's matrix, may overflow (base review d0bc7dee6 F3)."""
    family.loaded([(gmem.at[axes.named(side_axes, origins | axes.at_group(side_axes, gmem.shape, group))],
                    smem.at[wg], barrier.at[wg])
                   for gmem, side_axes, smem, barrier in sides],
                  axes=tuple(axes.held(side_axes) for _, side_axes, _, _ in sides))
    rows = origins['rows'].size
    row_start, first, stored, column, *ladder = place
    seen = jnp.zeros((), jnp.bool_)
    layout = family.layout if family.budgets_registers else None
    for start, chunk in family.chunks(accs):
        columns = {'columns': slice(start, start + chunk[0].shape[-1])}
        with quantized.epilogue_layout(layout):
            values = chunk if epilogue is None else epilogue(*chunk, *(
                axes.by_rows(family.side_tile(smem.at[wg].at[axes.placed(axes.held(side_axes), columns)],
                                              axes.held(side_axes)), side_axes, rows)
                for _, side_axes, smem, _ in sides))
        values = values if isinstance(values, tuple) else (values,)
        for output, (o_gmem, value) in enumerate(zip(o_gmems, values, strict=True)):
            value = value.astype(_stored_dtype(o_gmem))
            if nonfinite is not None:
                # An entry times zero is NaN where it is infinite or NaN, zero elsewhere, of the rows the
                # share stores: a predicate reduced over the warpgroup by a barrier, no scratch.
                kept = family.iota(value.shape, 0) - first
                stored_rows = (kept >= 0) & (kept < stored)
                seen = seen | jnp.any(stored_rows & jnp.isnan(value * 0))
            store(output, value, o_gmem, (row_start, first, stored, column + start, *ladder))
    if nonfinite is not None:
        nonfinite[...] = jnp.broadcast_to(seen.astype(jnp.int32), (1,))


def _stored_dtype(o_gmem):
    """Dew: the dtype an output ref stores (a scattered output's, `stores.Scattered`, its ref's)."""
    return (o_gmem.ref if isinstance(o_gmem, stores.Scattered) else o_gmem).dtype


def _gmm_scheduled(index, *, table_gmem, lhs_gmem, matrix_gmems: dict, prologue_gmems, shape: tuple,
                   grid: tuple, tiles: Tiles, dense: bool, tail: tail_of.Tail | None, programs: int,
                   transpose_rhs: bool, axes_of: dict, interpreted: bool,
                   transpose_lhs: bool = False, splits: int = 1, traced: bool = False,
                   scale_gmems=None, scale_per: int = 1, mx_gmems=None, gather_gmem=None,
                   packed_rows: bool = False, when_gmem=None, chained_gmems: tuple | None = None,
                   bounded: "Chained | None" = None, packed_format: str | None = None) -> tuple:
    """Dew: `gmm`'s tile at the persistent walk's `index` (`mosaic.Family.persist`'s schedule; its first row
    a multiple of its family's `row_unit`): whether it
    tile: its pair's but for this multiprocessor's row block; in a balanced tail, the first of the tile's
    steps its program walks and how many, its partials' gatherer and its program (`tail.Tail.segment`);
    with `splits`, its depth's run: the walk every tile's first run, then every
    tile's second (as many launches of a run each would walk them), each run a share of the steps, the
    last's the rest; `traced`,
    its steps a count the compiler cannot see (`rings.opaque`), which a walk takes as a loop's bound (an
    exactly summed walk's: cp.async's pipeline unrolled a static walk's drain, each drain step's fresh
    product its own, and at five stages the A100 forward's 128 x 128 x 64 tile spilled 40 bytes beside the
    running sum; an opaque count's drain, a loop, spilled none, and a merely traced constant left ptxas
    spilling 32 bytes at the input gradient's 64 x 160 x 32 at five stages, NSDBG-5, NSDBG-DR1).
    `shape` is the padded rows, the steps and the groups. A `bounded` walk (`guarded_walk`'s) walks a tile
    where any share of its pair's row tiles at the flagged walk's column tile holding it (`Chained.narrowed`)
    is flagged (`when_gmem`)."""
    m, steps, groups = shape
    dtype = _products_dtype(lhs_gmem, matrix_gmems, packed_rows)
    pair, unit = tiles.cluster, row_unit(family_of(tiles, dtype), dtype)
    index = tail_of.ordered(tail, index, programs, interpreted)
    split = None
    if tail is not None:
        snaked, start, count, gathered = tail.segment(index, programs, grid[0] * grid[1])
    elif splits == 1:
        snaked, start, count, gathered = index[0], 0, rings.opaque(steps) if traced else steps, None
    else:
        share = -(-steps // splits)
        split, snaked = index[0] // (grid[0] * grid[1]), index[0] % (grid[0] * grid[1])
        start, gathered = split * share, None
        count = jnp.minimum(share, steps - start)
    mi, ni = tail_of.planar_snake(snaked, grid, 1, tiles.grid_block_n)
    group = (row_tiles.GroupInfo.dense(tiles.block_m * pair, mi, m) if dense
             else row_tiles.GroupInfo.create(table_gmem, tiles.block_m * pair, mi, unit))
    # This multiprocessor's row tile of its pair, from its group's first: a multiple of the rows a TMA
    # descriptor tiles by, as Mosaic's layout inference reads a slice's start.
    row_start = pl.multiple_of(group.block_start + mosaic.cluster_index(pair) * tiles.block_m, unit)
    views = _gmm_views(lhs_gmem, matrix_gmems, prologue_gmems, group, row_start, ni, tiles=tiles,
                       groups=groups, transpose_rhs=transpose_rhs, axes_of=axes_of, first=start,
                       transpose_lhs=transpose_lhs, scale_gmems=scale_gmems, scale_per=scale_per,
                       mx_gmems=mx_gmems, gather_gmem=gather_gmem, packed_rows=packed_rows,
                       chained_gmems=chained_gmems, packed_format=packed_format)
    active = (count > 0 if tail is not None else jnp.ones((), jnp.bool_) if dense
              else (group.group_id < groups) & (group.actual_size > 0))
    if when_gmem is not None:
        # Both multiprocessors of a pair walk its steps or neither (`_gmm_mainloop`): the pair's flags at its
        # table entry, of the flagged walk's column tile holding this one.
        flagged_column = ni // (1 if bounded is None else bounded.narrowed)
        flagged = (when_gmem[flag_row(mi, pair, member), flagged_column, share] for member in range(pair)
                   for share in range(tiles.sharing))
        active = active & (functools.reduce(jnp.maximum, flagged) > 0)
    return active, count, views, (group, row_start, ni, start, gathered, index[-1], split, mi)


def _gmm_split_refused(tiles: Tiles, k: int, n: int, step_scales, *, exact_sums: bool, hooked: bool,
                       plain: bool) -> None:
    """Dew: `gmm`'s refusal of a depth split but a product's of no hooks or pair, each run some steps (a
    grouped one's runs over each tile its group's rows, the rows past every group's zeros in every run, so
    their partials sum to the product: a decode step's experts, row 3622; an exact sum's runs summed exactly,
    their partials by IEEE adds), of step scales but on an
    exactly summed product of no hook, pair, packed matrix or split (`_step_scaled_takes`)."""
    if tiles.splits > 1 and (hooked or not split_runs(-(-k // tiles.block_k), tiles.splits)):
        raise ValueError(f"a depth split is a product's of no hooks or pair, each run some steps, not "
                         f"{tiles} of a {k}-deep one")
    if step_scales is not None and not _step_scaled_takes(step_scales, tiles, k, n, exact_sums=exact_sums,
                                                          plain=plain):
        raise ValueError(f"a step-scaled product sums each {SCALE_DEPTH}-deep step exactly with no "
                         f"hook, pair, packed matrix or split, its scales covering its steps and columns, "
                         f"not {tiles}")


def _gmm_allocated(tiles: Tiles, family: mosaic.Family, operands: tuple, hooks: tuple, sides: tuple,
                   dtypes: tuple, *, transpose_rhs: bool, transpose_lhs: bool,
                   interpret: bool = False, packed_rows: bool = False, joined=None,
                   packed_format: str | None = None) -> Allocations:
    """Dew: `gmm_allocations` for `_gmm_mainloop`'s call: its `operands` (rows, matrix, paired matrix, packed
    scales, step and block scales, gather table), its hooks' axes and side operands' dtypes, and its outputs'
    `dtypes`; with `packed_rows`, the packed matrix its rows; with `joined` rows, their stream."""
    lhs, rhs, paired, scales, step_scales, mx_scales, gather = operands
    _, prologue_axes, _, epilogue_axes = hooks
    prologue_operands, epilogue_operands = sides
    return gmm_allocations(tiles, family, dtype=rhs.dtype if packed_rows else lhs.dtype, rhs_dtype=rhs.dtype,
                           transpose_rhs=transpose_rhs,
                           prologue_axes=prologue_axes,
                           prologue_dtypes=tuple(side.dtype for side in prologue_operands),
                           epilogue_axes=epilogue_axes,
                           epilogue_dtypes=tuple(side.dtype for side in epilogue_operands), dtypes=dtypes,
                           paired=paired is not None, transpose_lhs=transpose_lhs, packed=scales is not None,
                           step_scaled=step_scales is not None, block_scaled=mx_scales is not None,
                           interpret=interpret, gathered=gather is not None, packed_rows=packed_rows,
                           joined=joined is not None, packed_format=packed_format)

def _mx_refused(mx_scales, tiles: Tiles, family: mosaic.Family, *, transpose_rhs: bool, plain: bool) -> None:
    """Dew: `gmm`'s refusal of a block-scaled product (`mx_scales`: the rows' and the matrices' scales, one
    every block of depths, `product.block_format`'s, tiled as `tcgen05.scales_layout` lays them out, `[rows /
    128, scale tiles, 32, 16]` and `[groups, width / 128, scale tiles, 32, 16]`, the rows' in the product's
    row order, a gathered product's its tokens' at the table (the caller's gather, `gemm._block_scaled`, row
    4851); FP8 or E2M1 rows times a K-major matrix of theirs, each block of each scaled in the products,
    `mosaic.Issuer.scaled_products`) it does not take: but on a family whose products take block scales, its
    matrix K-major, with no hook, pair, packed matrix, step scales, transposed rows, exact sums or split, its
    tile whole 128s of rows, columns and depth (the scales' tiles), each group's rows whole 128-row tiles (the
    caller's)."""
    if mx_scales is None:
        return
    if not (family.block_formats and transpose_rhs and plain and tiles.splits == 1
            and tiles.block_m % 128 == 0 and tiles.block_n % 128 == 0 and tiles.block_k % 128 == 0):
        raise ValueError(f"a block-scaled product runs on a family whose products take block scales "
                         f"({family.name}'s do not, or its call or {tiles} is not one it takes)")


def _gmm_refs(gmems: tuple, tables: tuple, counts: tuple[int, int, int], *, dense: bool,
              beside: "Beside | None", tail, interpret: bool, chained: "Chained | None" = None,
              guarded: tuple = (False, False, False)) -> tuple:
    """Dew: a `gmm` kernel's refs in its operands' order (`_gmm_operands`): its tile table (None where
    `dense`), the rows and the matrix streams (`counts`' first; the first reading matrices side by side,
    `Beside.read`); the step scales', MX scales' and gather table's (`_scale_refs` of `tables`); the
    chained products' (`Chained.read`: the rows' and the matrices' refs, None unchained), the flags its
    tiles are walked by and the ref it stores into
    (`guarded`'s first two: `guarded_walk`'s); and the prologue's and the epilogue's side operands
    (`counts`' others), a balanced tail's workspace and the outputs (`_outputs`, through the scatter table;
    the `into` ref where given), its flags last where `guarded`'s third (`flags_declared`)."""
    streamed, *sides = counts
    walked_by, stored_into, flagged = guarded
    table_gmem, gmems = (None, gmems) if dense else (gmems[0], gmems[1:])
    (lhs_gmem, *matrix_gmems), refs = gmems[:1 + streamed], gmems[1 + streamed:]
    (scale_gmems, mx_gmems, gather_gmem, scatter_gmem), refs = _scale_refs(refs, *tables)
    matrix_gmems = tuple(matrix_gmems)
    matrix_gmems, refs = (matrix_gmems, refs) if beside is None else beside.read(matrix_gmems, refs)
    chained_gmems, refs = (None, refs) if chained is None else chained.read(lhs_gmem, matrix_gmems[0], refs)
    when_gmem, refs = (refs[0], refs[1:]) if walked_by else (None, refs)
    into_gmem, refs = (refs[0], refs[1:]) if stored_into else (None, refs)
    prologue_gmems, side_gmems, o_gmems = _split(refs, *sides)
    workspace, o_gmems = (None, o_gmems) if tail is None else (o_gmems[0], o_gmems[1:])
    flags_gmem, o_gmems = (o_gmems[-1], o_gmems[:-1]) if flagged else (None, o_gmems)
    out = (into_gmem,) if stored_into else _outputs(tuple(o_gmems), scatter_gmem, interpret)
    return ((table_gmem, lhs_gmem, matrix_gmems), (scale_gmems, mx_gmems, gather_gmem),
            (prologue_gmems, side_gmems, workspace, out, (chained_gmems, when_gmem, flags_gmem)))


def _outputs(o_gmems: tuple, scatter_gmem, interpret: bool) -> tuple:
    """Dew: a `gmm` kernel's outputs, each `stores.Scattered` through `scatter_gmem` where the call has
    one."""
    if scatter_gmem is not None:
        return tuple(stores.Scattered(o_gmem, scatter_gmem, interpret) for o_gmem in o_gmems)
    return o_gmems


def _workspace(tail, shape: tuple, interpret) -> tuple:
    """Dew: a balanced tail's partials (`tail.Tail`): each program's, a consumer's share of a tile each, of
    the tile its range starts inside and of the one it walks the first step of (`Family.fixed_up`'s
    `firsts`), a ref the kernel writes in place, the last of its operands; none without a tail. Its slots
    are written only where programs share a tile, so it is no declared output, which would leave the rest
    unwritten (NSNAN-2), and it starts uninitialized (`mosaic.workspace`)."""
    return () if tail is None else (mosaic.workspace(shape, jnp.float32, interpret=interpret),)


def _scale_refs(refs: tuple, step_scales, mx_scales, gather, scatter) -> tuple:
    """Dew: a `gmm` kernel's step scales', MX scales' (each two), gather table's and scatter table's
    references, leading the rest in that order (None where the call has none), and the references after
    them."""
    step, refs = (refs[:2], refs[2:]) if step_scales is not None else (None, refs)
    mx, refs = (refs[:2], refs[2:]) if mx_scales is not None else (None, refs)
    gathered, refs = (refs[0], refs[1:]) if gather is not None else (None, refs)
    scattered, refs = (refs[0], refs[1:]) if scatter is not None else (None, refs)
    return (step, mx, gathered, scattered), refs


def _gathered(gather, tiles: Tiles, dtype, interpret) -> dict:
    """Dew: a gathered product's rows stream (`family.streams`' arguments): its producer's gather
    (`gathered.Ring`), or interpreted, where no lane-level copy runs, each row's copies (`Ring.parts`, its
    slot staged where layout inference lays it out, at no stated swizzle: `rings.piece_copies`), that many
    arrivals at its load barrier; none ungathered."""
    if gather is None:
        return {}
    if interpret:
        copies = rings.piece_copies((tiles.block_m, tiles.block_k), dtype, None, 1)
        return {'copies': {'rows': tiles.block_m * copies}}
    return {'gathered': GATHERED_STREAMS}


def operand_streams(rewritten: bool) -> tuple[str, ...]:
    """Dew: the streams of `gmm`'s product operands, one barrier pair between them (`Family.streams`'
    `joined`, CUTLASS's PipelineTmaAsync for A and B): a step's loads of the rows and the matrices awaited
    once and their slots given back once, where two pairs a step took the consumers two waits and two
    arrivals (row 5666: a GH200's grouped forward issued 391M instructions to CUTLASS's 265M at equal
    HGMMA); none where a role of the family's own `rewritten` rewrites a slot, its release its own."""
    return () if rewritten else ('rows', JOINED, 'matrix', 'paired')


GATHERED_STREAMS = ('rows',)
"""The streams a gathered product's producer gathers (`gmm`'s `gather`): its rows'."""

def _gmm_walked(lhs, gather, scatter, joined, tiles: Tiles, dtype, transpose_lhs: bool, dense: bool,
                interpret) -> tuple[tuple, int]:
    """Dew: `gmm`'s rows as its schedule walks them (`_walked_rows`), whole tiles where its family stores
    whole tiles, `joined` rows (of `lhs`'s shape) padded as `lhs`: `(lhs, gather, scatter, joined, m,
    extent)` and the rows' unit (`row_unit`)."""
    family = family_of(tiles, dtype)
    unit = row_unit(family, dtype)
    walked_rows = functools.partial(_walked_rows, tiles=tiles, transpose_lhs=transpose_lhs, dense=dense,
                                    bounded=family.bounded_loads and not interpret, unit=unit,
                                    least=tiles.block_m * tiles.cluster if family.copies_tiled_rows else 0)
    lhs, gather, scatter, m, extent = walked_rows(lhs, gather, scatter)
    joined = None if joined is None else walked_rows(joined, None, None)[0]
    return (lhs, gather, scatter, joined, m, extent), unit


def _step_scales_walked(step_scales: tuple | None, n: int, block_n: int, extent: int) -> tuple:
    """Dew: a blockwise-scaled call's `step_scales` as its walk reads them: its rows' scales padded and
    extended as the rows (`_walked_rows`), its columns' a tile's (`_column_scales`), and the columns a scale
    of a tile (`per`, 1 without them)."""
    if step_scales is None:
        return None, 1
    column_scales, per = _column_scales(step_scales[1], n, block_n)
    return (rows_extended(rows_padded(step_scales[0], 1, TMA_ROWS), 1, extent), column_scales), per


def _walked_rows(lhs, gather, scatter, tiles: Tiles, *, transpose_lhs: bool, dense: bool,
                 bounded: bool, unit: int = TMA_ROWS, least: int = 0) -> tuple:
    """Dew: `gmm`'s rows as its schedule walks them: `lhs`, or a gathered product's table (`gather`, a pad
    reading row 0 of the tokens, which stay as they are), padded to whole `unit`s of rows and to `least`
    (`m` of them; `row_unit`: a TMA descriptor's 8; a tile's rows where its family stores whole tiles) and
    extended to the schedule's extent where the family's loads past
    the rows are not zeros (`bounded`), and a scattered product's table to the extent, its padded and
    extended rows stored nowhere (`stores.Scattered`): `(lhs, gather, scatter, m, extent)`. The rows past
    the routed ones, the padding among them, are a last group, whose tiles store zeros; a group's tiles
    start at its first row (`tile_groups`), so a tile reads up to a tile's rows past the last group's first.
    Where the loads read zeros past an operand's rows (TMA's) the operands stay as they are; elsewhere every
    row-bearing operand is allocated to the extent, zeros past its rows, which a group's stores never reach
    (a 128-row lhs of groups [120, 8] at 64-row tiles read rows 120 to 183 unpredicated on an A100,
    cp.async's loads), as the interpreter's are, whose slices past an array clamp rather than read zeros. On
    an H100 that copy was a kernel of its own each call, 0.035 to 0.33 ms beside a dense product of 0.34 to
    2.4 (BREAKDOWN1-h100). A dense call's tiles start at whole row tiles, so it reads to its last tile's end
    alone, no copy where its rows fill whole tiles: on an A100 that copy was 0.094 ms of Llama 3 8B o's
    1.52 ms forward (BC-A100-EXTENT-AB1). A table is always extended: its threads read a tile's entries by
    plain loads, which no TMA bounds (a gathered product's last tile read past an unextended table on a
    GH200, CUDA_ERROR_ILLEGAL_ADDRESS, FS-4346-base-sm90)."""
    row_axis = 1 if transpose_lhs else 0
    if gather is not None:
        gather = _normalized_gather(gather, lhs.shape[row_axis])
    walked, axis = (lhs, row_axis) if gather is None else (gather, 0)
    walked = rows_padded(walked, axis, unit)
    walked = rows_extended(walked, axis, max(walked.shape[axis], least))
    m = walked.shape[axis]
    row_tile = tiles.block_m * tiles.cluster
    tabled = gather is not None or scatter is not None
    extent = (m if bounded and not tabled else -(-m // row_tile) * row_tile if dense
              else row_tiles.row_extent(m, row_tile, unit))
    walked = rows_extended(walked, axis, extent)
    if scatter is not None:
        scatter = jnp.pad(scatter.astype(jnp.int32), (0, extent - scatter.shape[0]), constant_values=-1)
    return (walked, None, scatter, m, extent) if gather is None else (lhs, walked, scatter, m, extent)


def _normalized_gather(table: jax.Array, rows: int) -> jax.Array:
    """Normalize signed gather indices to `[0, rows - 1]` (`lhs[gather]` semantics) before native
    row loads (`gathered.row_copies` sign-extends int32 indices to int64, row 5391)."""
    indices = table.astype(jnp.int32)
    return jnp.clip(jnp.where(indices < 0, indices + rows, indices), 0, max(rows - 1, 0)).astype(jnp.int32)


def _tables_refused(gather, scatter, tiles: Tiles, family: mosaic.Family, *, dense: bool,
                    transpose_lhs: bool) -> None:
    """Dew: `gmm`'s refusal of a gathered product (`gather`) but on a family that gathers rows beside the
    matrix's whole loads (`Family.gathers_rows`, `gathers_alone`), of no transposed rows or dense call (a
    prologue runs fused on the gathered rows in the mainloop, row 5391), and of a scattered one (`scatter`)
    but a grouped call's on such a family (its stores `stores.staged_rows`')."""
    gathers = family.gathers_rows and family.gathers_alone
    if gather is not None and (dense or transpose_lhs or not gathers):
        raise ValueError(f"a gathered product's rows are a ring family's gather, with no "
                         f"transposed rows or dense call, not {tiles}'s")
    if scatter is not None and (dense or not gathers):
        raise ValueError(f"scattered rows are a grouped call's on a ring family, not {tiles}'s")


def _tgmm_tables_refused(tables: tuple, tiles: Tiles, family: mosaic.Family, *, dense: bool,
                         k_major: bool, step_scaled: bool) -> None:
    """Dew: `tgmm`'s refusal of a gathered operand (`tables`, each operand's) but on a family that gathers
    rows, of no rows-contiguous layout, step scales or dense call."""
    if any(table is not None for table in tables) and (
            dense or k_major or step_scaled or not (family.gathers_rows and family.gathers_alone)):
        raise ValueError(f"a gathered kernel gradient's rows are a ring family's gather, with no "
                         f"rows-contiguous layout, step scales or dense call, not {tiles}'s")


def depth_masked(prologue, tile, *sides):
    """Dew: `prologue` of a stage `tile` and its side tiles, the last a vector over the depth of ones where
    the call's depth runs and zeros past it (`grouped_matmul.MosaicGmm._aligned`'s, its padding), its result
    zeroed past it: the padding's zeros go into the product as its own, where a prologue of them (a
    reciprocal's infinities) would make NaNs of them."""
    *own, valid = sides
    transformed = prologue(tile, *own)
    # The vector broadcast straight to the tile's shape, as a side vector's arithmetic broadcasts it: a
    # `[1, depth]` reshape between took no layout Mosaic GPU infers ("Tiling rank different than broadcast
    # result rank, (8,) vs (1, 64)", the GEMM compile gate's unaligned projection on sm90).
    return jnp.where(valid > 0, transformed, jnp.zeros_like(transformed))


def multicast(tiles: Tiles, family: mosaic.Family, matrices: tuple[str, ...]) -> tuple[str, ...]:
    """Dew: the streams a cluster's programs share (`mosaic.Family.streams`' multicast): its `matrices`, the
    group's blocks its row tiles load once between them, where its cluster's programs each run a tile of
    their own; none on a family whose products span the pair (`mosaic.Transport.pairs_product`, each CTA
    loading its half of the pair's block) or of one program a cluster. The one statement the kernel's
    streams and the planner's (`Allocations.scope`) both read: the planner's multicast a pair's streams
    refused, so no pair's tile was ever legal."""
    return matrices if tiles.cluster > 1 and not family.pairs_product else ()


def pair_takes(tiles: Tiles, family: mosaic.Family) -> bool:
    """Dew: whether `tiles` is a tile of a family whose products span a pair
    (`mosaic.Transport.pairs_product`) where it is one: its cluster the pair, each CTA's columns at most its
    half of one product (`mosaic.Issuer.widest_product`, tcgen05's 256 over a pair); any tile on another
    family."""
    if not family.pairs_product:
        return True
    widest = family.widest_product
    return tiles.cluster == family.cluster and (widest is None or tiles.block_n <= widest)


def pair_refuses(tiles: Tiles, family: mosaic.Family, *, transpose_lhs: bool, exact_sums: bool,
                 scaled: bool, parts: tuple, rewritten: bool = False) -> bool:
    """Dew: whether a pair's product (`mosaic.Transport.pairs_product`) refuses a call at `tiles`: any but
    a plain one's at a pair's tile (`pair_takes`), its step views its operands' blocks alone, each matrix's
    the pair's (`_gmm_views`: a paired product's second matrix too, its accumulators beside the first's in
    tensor memory; `_tgmm_scheduled`), so none of `parts` (a prologue, gathered rows), no
    scales (`scaled`: a packed matrix's, step or block scales), transposed rows, exact sums, a split or a
    stage its family's role rewrites (`rewritten`, `rewrites`: the pair's persist sums and prepares
    nothing, `mosaic_sm100`). The one statement `gmm`'s and `tgmm`'s refusals (`_pair_refused`) and a
    launch's candidate (`grouped_matmul._Mosaic.candidate`) read, so a derivation never takes a tile its
    call refuses."""
    return family.pairs_product and (not pair_takes(tiles, family) or transpose_lhs or exact_sums or scaled
                                     or rewritten or tiles.splits > 1
                                     or any(part is not None for part in parts))


def _pair_refused(tiles: Tiles, family: mosaic.Family, *, transpose_lhs: bool, exact_sums: bool,
                  scaled: bool, parts: tuple, rewritten: bool = False) -> None:
    """Dew: `gmm`'s and `tgmm`'s refusal of a pair's product a pair refuses (`pair_refuses`)."""
    if pair_refuses(tiles, family, transpose_lhs=transpose_lhs, exact_sums=exact_sums, scaled=scaled,
                    parts=parts, rewritten=rewritten):
        raise ValueError(f"a pair's product is a pair's tile's, of no prologue, transposed or gathered rows, "
                         f"second matrix, scales, exact sums, split or rewritten stage, not {tiles}")


def _gmm_shape_refused(lhs, rhs, tiles: Tiles, extents: tuple, interpret, *, dtype, columns: int,
                       dense: bool, first, paired, transpose_rhs: bool, transpose_lhs: bool, exact_sums: bool,
                       parts: tuple) -> None:
    """Dew: `gmm`'s refusals of its call's shape at `tiles` (`_gmm_refused`, its last step and column tile
    partial where `partial_runs`), of a cooperative pair (`_pair_refused`), of its tables
    (`_tables_refused`); `parts` the call's prologue, packed matrix's scales, step and block scales, gather
    and scatter tables; `dtype` the products' operands'."""
    prologue, scales, step_scales, mx_scales, gather, scatter = parts
    family = family_of(tiles, dtype)
    plain = all(each is None for each in (scales, step_scales, mx_scales))
    _gmm_refused(rhs, tiles, extents, columns=columns, dense=dense, first=first, paired=paired,
                 partial=partial_runs(tiles, dtype, interpret, transpose_rhs=transpose_rhs, plain=plain,
                                      gathered=gather is not None, scattered=scatter is not None))
    _pair_refused(tiles, family, transpose_lhs=transpose_lhs, exact_sums=exact_sums, scaled=not plain,
                  parts=(prologue, gather))
    _tables_refused(gather, scatter, tiles, family, dense=dense, transpose_lhs=transpose_lhs)


def _gmm_refused(rhs, tiles: Tiles, extents: tuple, *, columns: int, dense: bool, first, paired,
                 partial: tuple[bool, bool] = (False, False)) -> None:
    """Dew: `gmm`'s refusal of a call its launch does not take: a dense product of several matrices, a stack
    read from its first matrix but a grouped product's (a pair's second stack read alike: each matrix stream
    reads its tile's `TileGroup.matrix`, the stack's from `first`), a paired matrix not its product's shape,
    or extents (`rows`, `depth`, `matrices`, the matrix's depth, `width`) its tile's step and `columns`
    (`tile_columns`) do not divide but where its last step and column tile run `partial` (`partial_runs`), no
    deeper or wider than the call."""
    rows, k, matrices, k2, n = extents
    if dense and matrices != 1:
        raise ValueError("a dense product takes one matrix")
    if first is not None and dense:
        raise ValueError("a stack read from its first matrix is a grouped product's")
    if paired is not None and paired.shape != rhs.shape:
        raise ValueError(f"a paired matrix {paired.shape} is not its product's {rhs.shape}")
    depth = k % tiles.block_k == 0 or (partial[0] and tiles.block_k <= k)
    width = n % columns == 0 or (partial[1] and columns <= n)
    if k != k2 or not (depth and width):
        raise ValueError(f"a [{rows}, {k}] x [{matrices}, {k2}, {n}] product does not take {tiles}")


def partial_step(dtype) -> int:
    """Dew: the step a depth no step divides takes with its last step partial (`partial_runs`): one swizzled
    tile of `dtype`, a copy of the rows' block one tile along its depth."""
    return swizzling.swizzle_columns(dtype, swizzling.SWIZZLE_BYTES)


def partial_runs(tiles: Tiles, dtype, interpret, *, transpose_rhs: bool, plain: bool = True,
                 gathered: bool = False, scattered: bool = False) -> tuple[bool, bool]:
    """Dew: whether `gmm` at `tiles` runs a depth, and a width, its tile does not divide as it is, its last
    step and column tile partial, no padded copy (row 5299): where its family's loads past an operand read
    zeros and its stores past it are dropped (`Family.bounded_loads`: a TMA descriptor's bounds), a plain
    matrix (no packed, step- or block-scaled one), never interpreted (an interpreted slice clamps), and each
    TMA copy of a partial dimension one tile along every tiled dimension but its first, which Mosaic GPU's
    view then describes untiled with the array's true extents (the fork's `launch_context`): a depth where the
    step is one swizzled tile (`partial_step`) and the rows are not gathered (a producer's gather copies by
    cp.async, unbounded); a width where the stores are not scattered (piecewise, unbounded) and the matrix
    lies `[width, depth]` (`transpose_rhs`) or its copy is one swizzled tile wide (`tile_columns`: a CTA
    pair's both halves; each CTA's copy of its half of the pair's block partitioned under a partial extent by
    the fork from post84, row 5727); on a family whose products read the staged blocks by descriptor
    (`Issuer.operands_in_registers` False: wgmma's, tcgen05's), staged at the widest swizzle, a step's copy
    one tile along its depth, where a family reading them into registers is staged narrower by layout
    inference (mma.sync.fed's 32-byte rows on an sm120: a 64-deep step four 16-element tiles), and a partial
    depth's rows no more than one copy moves along a dimension (`Family.copy_extent`). Elsewhere the
    launch's call is padded to whole tiles (`grouped_matmul._Mosaic. _aligned`), as it is where a stride is no
    whole 16 bytes, which no TMA copy takes."""
    family = family_of(tiles, dtype)
    bounded = family.bounded_loads and not family.operands_in_registers and not interpret and plain
    one = partial_step(dtype)
    # A partial depth's untiled copy of the rows' block is a CTA's whole rows tall, one dimension of the
    # copy's box, which TMA bounds (`Family.copy_extent`: an H100's 384 x 64 forward of a 200-deep call failed
    # to lower); a partial width's copies are the matrix's one-tile blocks and each consumer's output share.
    tall = family.copy_extent is not None and tiles.block_m > family.copy_extent
    return (bounded and not gathered and tiles.block_k == one and not tall,
            bounded and not scattered and (transpose_rhs or tile_columns(tiles, family) == one))


def _step_scaled_takes(step_scales: tuple, tiles: Tiles, depth: int, width: int, *, exact_sums: bool,
                       plain: bool) -> bool:
    """Dew: whether `gmm` takes `step_scales` at `tiles` (`_step_scaling`): an exactly summed product of no
    hook, pair, packed matrix, transposed rows or split, a step 128 deep, a row scale each step, and a
    column scale each column or each multiple of the tile's width."""
    rows, columns = step_scales
    per = width // columns.shape[-1] if columns.shape[-1] and width % columns.shape[-1] == 0 else 0
    return (exact_sums and plain and tiles.splits == 1 and tiles.block_k == SCALE_DEPTH
            and rows.shape[0] * SCALE_DEPTH == depth
            and (per == 1 or (per > 0 and per % tiles.block_n == 0)))


def _unrolled(prologue, packed_rows: bool, exact_sums: bool) -> bool:
    """Dew: whether a `gmm` walk takes a rotation of its ring a trip (`mosaic.stepping`): where its step is
    its products and its waits, small enough that a rotation unrolled takes fewer instructions an HGMMA;
    not a step that rewrites its rows (a `prologue`'s, a packed matrix's decode, `packed_rows`), nor an
    exactly
    summed step (its product apart, its IEEE add beside the running
    sum), a rotation of which spilled as the ring deepened (row 3620: FP8 at 128 x 128 x 128 spilled 152
    bytes at five stages and 424 at six, 34% slower than at four, FS-3620-SERIAL2, FS-3620-STAGES1-h100)."""
    return prologue is None and not packed_rows and not exact_sums


def _zeroed_atoms(family: mosaic.Family, dtype, block_n: int, consumers: int) -> tuple[int, int]:
    """Dew: the columns a consumer zeroes of the cotangent's stage at a time (`zeroing_columns`) and how many
    such atoms each consumer zeroes (`_zeroed_share`): as many each as the stage's divide among them, the
    last's share short where they do not divide evenly."""
    atom = zeroing_columns(family, dtype)
    return atom, -(-(block_n // atom) // consumers)


def zeroing_columns(family: mosaic.Family, dtype) -> int:
    """Dew: the columns of a stage a consumer zeroes at a time (`_zero_neighbours_of`), so its registers hold
    a few whatever the tile's width: a swizzled row, never narrower than its family's column unit (a TPU
    core's slice of a VMEM block is whole lanes, 128 columns)."""
    return max(family.column_unit(dtype), swizzling.swizzle_columns(dtype, swizzling.SWIZZLE_BYTES))


def _dtypes_of(arrays) -> tuple:
    """Dew: each of `arrays`' dtype."""
    return tuple(array.dtype for array in arrays)


def _scales_apart(refs: list, scaled: bool) -> tuple:
    """Dew: a step-scaled kernel's two scale references, which lead its side operands' (`tgmm`'s
    `step_scales`), and the rest; None and all of them where it has no step scales."""
    return (tuple(refs[:2]), refs[2:]) if scaled else (None, refs)


def _column_scales(column_scales: jax.Array, width: int, block_n: int) -> tuple[jax.Array, int]:
    """Dew: a step-scaled `gmm`'s column scales (`[groups, depth / 128, width / per]`) as its stream reads
    them (`_scale_blocks`), and the columns `per` a scale covers: as they are where each covers a column;
    else each step's row of them padded to whole `block_n`s, a block of which holds a tile's scale
    (`_column_block`)."""
    per = width // column_scales.shape[-1]
    if per == 1:
        return column_scales, per
    return jnp.pad(column_scales, ((0, 0), (0, 0), (0, -column_scales.shape[-1] % block_n))), per


def _column_block(column_tile, block_n: int, per: int) -> tuple:
    """Dew: the block of a step's column scales `column_tile`'s stream reads (`_column_scales`'), and where
    in it the tile's own scale lies: its columns' block, at its first, where a scale covers a column; else
    the block holding the tile's one scale, `per` columns (whole tiles of them) a scale."""
    if per == 1:
        return column_tile, 0
    scale = column_tile * block_n // per
    return scale // block_n, scale % block_n


def _tgmm_scaling(scaled: bool, mine_depth) -> Callable | None:
    """Dew: a step-scaled `tgmm` tile's `_step_scaling` (`tgmm`'s `step_scales`), its depth rows' and its
    columns' scales streamed, a scale a column; none where the call has no step scales."""
    if not scaled:
        return None
    return functools.partial(_step_scaling, per=1, offset=0, mine_rows=mine_depth)


def _gmm_scaling(scaled: bool, per: int, block_n: int, column_tile, mine_rows) -> Callable | None:
    """Dew: a step-scaled `gmm` tile's `_step_scaling` for a consumer's share of the rows (`mine_rows`), its
    matrix's column scales a scale every `per` columns (`_column_block`); none where the call has no step
    scales."""
    if not scaled:
        return None
    return functools.partial(_step_scaling, per=per, offset=_column_block(column_tile, block_n, per)[1],
                             mine_rows=mine_rows)


def _step_scaling(index, slots, *, per: int, offset, mine_rows) -> Callable:
    """Dew: a step-scaled walk's multiplier of a step's exactly summed partial (`gmm`'s and `tgmm`'s
    `step_scales`, `mosaic.Issuer.summed_apart`'s `scale`): this consumer's rows' scales (the step's row of
    them streamed, `row_scales`) times the tile's columns' (streamed beside them, `column_scales`: the step's
    row of them at the tile's columns, one a column, or the block holding the tile's one, at `offset`, `per`
    columns a scale), in shared memory as the step's operands land: read from global memory, a step's
    scaling waited on its load (M-NS-13: the blockwise FP8 template 1.24-1.42x its unscaled exact product).
    `index` is the step's, whose scales are its streams'."""
    del index
    rows = slots['row_scales'].at[0, mine_rows][...]
    column_scales = slots['column_scales']
    if per == 1:
        return lambda start, columns: rows[:, None] * column_scales.at[0, pl.ds(start, columns)][...][None, :]
    # A weight block's one scale scales the rows' vector before it is broadcast over the columns: a multiply
    # a row, not one an element of the partial beside its own (the same products, so the same bits).
    scaled = rows * column_scales[0, offset]
    return lambda start, columns: scaled[:, None]


def _gmm_views(lhs_gmem, matrix_gmems: dict, prologue_gmems, group, row_start, column_tile, *,
               tiles: Tiles, groups: int, transpose_rhs: bool, axes_of: dict, first=0,
               transpose_lhs: bool = False,
               scale_gmems=None, scale_per: int = 1, mx_gmems=None, gather_gmem=None,
               packed_rows: bool = False, chained_gmems: tuple | None = None,
               packed_format: str | None = None) -> dict:
    """Dew: `gmm`'s streams' views at a tile (`group`'s rows from `row_start`, `column_tile`;
    `rings.viewed`): its rows', its group's matrix's (each of `matrix_gmems`, by stream: a paired
    `j` each one's `first + j`-th depth block (`first` a balanced tail's program's first step); a
    `transpose_lhs` call's rows as `[depth, rows]` blocks. On a family whose products span a pair
    (`mosaic.Transport.pairs_product`), the rows' and the matrix's views are the pair's (`rings.Paired`):
    its two row tiles from the group's block and its columns (`tile_columns`), each multiprocessor loading
    its half. With `packed_rows`, the rows are a packed matrix's (`gmm`'s): its words' streams and its
    scales' at the tile's rows."""
    rows = pl.ds(row_start, tiles.block_m)
    family = family_of(tiles, _products_dtype(lhs_gmem, matrix_gmems, packed_rows))
    rhs_index = _rhs_index(transpose_rhs)
    # The joined rows (`gmm`'s `joined`) are viewed as the rows are.
    joined = matrix_gmems.get(JOINED)
    matrix_gmems = {name: gmem for name, gmem in matrix_gmems.items() if name != JOINED}
    if family.pairs_product:
        pair_rows = pl.ds(group.block_start, tiles.block_m * tiles.cluster)
        columns = tile_columns(tiles, family)
        rows_of = {'rows': lhs_gmem, **({} if joined is None else {JOINED: joined})}
        return {**{name: (gmem.at[pair_rows], lambda j: rings.Paired((0, first + j), 0))
                   for name, gmem in rows_of.items()},
                **{name: (_matrix_columns(gmem.at[group.matrix], column_tile, block_n=columns,
                                          transpose_rhs=transpose_rhs),
                          lambda j: rings.Paired(rhs_index(first + j), 0 if transpose_rhs else 1))
                   for name, gmem in matrix_gmems.items()}}
    # A product's rows and matrix: each its block of a ref and its step's index, ordinary or chained.
    row_block, row_step = (lambda gmem: gmem.at[rows]), (lambda j: (0, first + j))
    matrix_of = functools.partial(_matrix_columns, column_tile=column_tile, block_n=tiles.block_n,
                                  transpose_rhs=transpose_rhs)
    matrix_block, matrix_step = ((lambda gmem: matrix_of(gmem.at[group.matrix])),
                                 (lambda j: rhs_index(first + j)))

    # Chained products' rows and matrices the same views, each step's from the product it falls in
    # (`Chained.view`, its step translated to the product's own).
    chained_rows, chained_matrices = (None, None) if chained_gmems is None else chained_gmems
    lhs_view = ((lhs_gmem.at[:, rows], lambda j: (first + j, 0)) if transpose_lhs
                else chained_rows.view(row_block, row_step, first) if chained_rows is not None
                else (row_block(lhs_gmem), row_step) if gather_gmem is None
                # A gathered product's rows where its table says (a row a run), the tile's at every step.
                else (lhs_gmem, lambda j: rings.Rows(gather_gmem, row_start, (first + j,), 1, tiled=True)))
    # Each stream's view and its step's block index (`(row, column)` of a matrix, four of a scale tile's).
    matrices: dict[str, Any] = {
        name: (gmem.columns(group.matrix, column_tile, matrix_step) if isinstance(gmem, BesideRefs)
               else chained_matrices.view(matrix_block, matrix_step, first)
               if name == 'matrix' and chained_matrices is not None
               else (matrix_block(gmem), matrix_step))
        for name, gmem in matrix_gmems.items() if not (packed_rows and name == 'scales')}
    groups_of = tiles.block_k // (packing.packed_group_depth(packed_format)
                                  if packed_format is not None else 128)
    rows_views = {'rows': lhs_view, **({} if joined is None else {JOINED: (row_block(joined), row_step)})}
    if packed_rows:
        scales_view = ({} if packed_format is not None
                       else {'scales': (matrix_gmems['scales'].at[rows], lambda j: (0, first + j))})
        rows_views = {**{f'words_{q}': (lhs_gmem.at[rows], lambda j, q=q: (0, groups_of * (first + j) + q))
                         for q in range(groups_of)}, **scales_view}
    elif 'scales' in matrix_gmems or packed_format is not None:
        # A packed matrix's words, a stream for each group of depths of a step (128 NVFP4, 256 ternary):
        # step j's group q, sixteen words a column at the tile's columns.
        step_groups = tiles.block_k // (packing.packed_group_depth(packed_format)
                                        if packed_format is not None
                                        else 128)
        words = matrices.pop('matrix')[0]
        matrices |= {f'words_{q}': (words, lambda j, q=q: (0, step_groups * (first + j) + q))
                     for q in range(step_groups)}
    if scale_gmems is not None:
        # Its rows' scales and its matrix's columns', a step's row of each (`step_scales`): the rows' at the
        row_scales, column_scales = scale_gmems
        block, _ = _column_block(column_tile, tiles.block_n, scale_per)
        matrices['row_scales'] = (row_scales.at[:, rows], lambda j: (first + j, 0))
        matrices['column_scales'] = (column_scales.at[group.matrix], lambda j: (first + j, block))
    if mx_gmems is not None:
        # An MX product's scale tiles at step j: its rows' 128-row tiles from the tile's (whole tiles: a
        # group's rows are whole 128s) and its matrix's at the tile's columns.
        row_scales, matrix_scales = mx_gmems
        matrices['a_scales'] = (row_scales.at[pl.ds(row_start // 128, tiles.block_m // 128)],
                                lambda j: (0, first + j, 0, 0))
        matrices['b_scales'] = (matrix_scales.at[group.matrix, pl.ds(column_tile * (tiles.block_n // 128),
                                                                tiles.block_n // 128)],
                                lambda j: (0, first + j, 0, 0))
    # A side operand by group at the tile's group (`axes.at_group`), its groups' axis indexed away.
    return {**rows_views, **matrices,
            **{f'side_{i}': (prologue_gmems[i].at[axes.placed(side_axes, {'rows': rows} | axes.at_group(
                                 side_axes, prologue_gmems[i].shape, _side_group(group, groups)))],
                             lambda j, side_axes=side_axes: axes.placed(axes.held(side_axes), {'rows': 0},
                                                                        first + j))
               for i, side_axes in axes_of.items()}}


def _products_dtype(lhs, matrices: dict, packed_rows: bool):
    """Dew: the dtype `gmm`'s products take, which names its family: its rows', or its matrix's where the
    rows are a packed matrix's words (`packed_rows`)."""
    return matrices['matrix'].dtype if packed_rows else lhs.dtype


def _reads(prologue_gmems, prologue_axes: tuple, read: tuple, group, groups: int) -> dict:
    """Dew: the prologue side vectors a step reads from global memory (`read`, by operand), one by group at
    the tile's `group` of `groups` (`_side_group`, `axes.at_group`)."""
    return {i: prologue_gmems[i].at[axes.placed(prologue_axes[i], axes.at_group(
        prologue_axes[i], prologue_gmems[i].shape, _side_group(group, groups)))] for i in read}


def _side_group(group, groups: int):
    """Dew: the group a tile reads its side operands by group at (`axes.at_group`): its own, the unrouted
    rows' the last's, as XLA's program reads them (`axes.at_rows`)."""
    return jnp.minimum(group.group_id, groups - 1)


def _summed(family: mosaic.Family, tiles: Tiles, exact_sums: bool) -> dict:
    """Dew: the persist arguments of an exactly summed launch on a family that sums its windows itself
    (`mosaic.Family.sums_in_persist`): a window's steps (`summed_window`); none on any other."""
    return {'summed': family.summed_window(tiles.block_k)} if exact_sums and family.sums_in_persist else {}


def _walked_products(family: mosaic.Family, walk, step, shape: tuple, count: int, exact_sums: bool,
                     prefetch: Callable | None = None) -> tuple:
    """Walk `count` products (`_gmm_step`), summed in fp32 or via `exact_sums` IEEE running sums."""
    if not exact_sums or family.sums_in_persist:
        _, accs = walk(step, None, (shape,) * count, **({} if prefetch is None else {'prefetch': prefetch}))
        return accs

    def exact_step(index, slots, running, accumulators):
        _, totals = step.func(index, slots, None, running, **step.keywords, summed=True)
        return totals, accumulators

    running, _ = walk(exact_step, tuple(family.zeros(shape) for _ in range(count)), ())
    return running


def _idle(tiles: Tiles, group: "row_tiles.GroupInfo", row_start, consumer) -> jax.Array | None:
    """Whether `consumer`'s share lies outside its group's rows (`issued_unless`); None if sharing == 1."""
    if tiles.sharing == 1:
        return None
    first = row_start + tiles.share_row(consumer)
    return (first >= group.actual_end) | (first + tiles.share <= group.actual_start)


def uses_row_bound(family: mosaic.Family, tiles: Tiles, dtype, *, dense: bool,
                   staged_rows: bool = False) -> bool:
    """Single bounded-MMA eligibility rule shared by `_row_bound` and `pricing.bounded_registers` (rows 5623,
    7091): one-consumer grouped non-fp32 `gmm` on an issuer overriding `Issuer.bounded`."""
    return (not staged_rows and not dense and tiles.sharing == 1
            and jnp.dtype(dtype) != jnp.float32 and type(family).bounded is not mosaic.Issuer.bounded)


def _row_bound(family: mosaic.Family, tiles: Tiles, dtype, group: "row_tiles.GroupInfo", row_start,
               dense: bool) -> jax.Array | None:
    """Live rows `[0, rows)` of a one-consumer grouped tile before its group's end (`uses_row_bound`)."""
    if not uses_row_bound(family, tiles, dtype, dense=dense):
        return None
    return jnp.clip(group.actual_end - row_start, 0, tiles.block_m).astype(jnp.int32)


def _gmm_step(index, slots, nothing, accumulators, *, family: mosaic.Family, prologue, prologue_axes: tuple,
              fixed_tiles: dict, read_tiles: dict, mine_rows, transpose_rhs: bool, matrices: dict,
              interpret: bool, first=0, summed: bool = False, transpose_lhs: bool = False,
              decoded=None, scaling=None, block_scaled: bool = False, packed_rows: bool = False, idle=None,
              chosen=None, prefetched=None, boundary: Callable | None = None,
              packed_format: str | None = None, block_k: int = 0):
    """Dew: a consumer's step of the cooperative `gmm` (`mosaic.Family.walk`): its half of the rows' slot,
    transformed by the prologue on its side tiles (the streamed ones' slots, the fixed ones' `fixed_tiles`, a
    depth vector the family does not copy read at step `index` from global memory: `read_tiles`), times each
    of `matrices`' slot into its accumulator (a paired product's two); with `summed`, each product summed
    apart into its accumulator, the running sum (`_walked_products`' exact sums,
    `mosaic.Issuer.summed_apart`). The prologue's value stays in registers, where the products take it as
    their left operand (warpgroup MMA's register operand A, mma.sync's fragments): rewritten into the stage
    and read back, the gated prologue of a Llama-3-8B gate_up ran at 8.3x its plain product on an H100 (row
    1336). With `decoded` (a packed matrix's, `packed.Decoding`), the step decodes its stage of words and
    scales, or its share of them, into the stage's buffer (`packed.stage_decoded`), published and met before
    its products read it as the matrix's stage. With `packed_rows`, its rows' slots hold a packed matrix's
    words and scales, its half of them decoded in registers into the products' left operand
    (`packed.rows_decoded`), as a prologue's value. `matrices` maps each matrix's stream to its rows' (the
    rows', or the joined rows' for a joined pair's paired matrix: `gmm`'s `joined`). With `idle` (`_idle`),
    its products issued only where its rows meet its group's. With `chosen` (a bounded walk's `Chained`),
    its one matrix's products issued into the accumulator of the product the step falls in alone, from that
    product's first step on (`Issuer.added_onto`), each product's sum its own (`chained.guarded_walk`). With
    `boundary` (the walk's prefetch, `_gmm_prefetch`: a plain step's one product), its product issued from
    its `prefetched` first fragments, the next step's read under its last (`Issuer.products_across`),
    returning its carry beside them."""
    read = {name: slots[name] for name in matrices if decoded is None}
    if decoded is not None:
        decode_layout = None if interpret or not family.budgets_registers else family.layout
        read['matrix'] = decoded.decode(index, slots, interpret, layout=decode_layout)
        family.published()
        decoded.met()
    oriented = _oriented(read, matrices, transpose_rhs)
    # This warpgroup's half of the rows, held in registers where a step rewrites them: a packed matrix's
    # decoded, or the prologue's transform of its slot.
    left = None
    if packed_rows:
        rows_layout = None if interpret or not family.budgets_registers else family.layout
        if packed_format is not None:
            words = [slots[f'words_{q}'].at[mine_rows]
                     for q in range(block_k // packing.packed_group_depth(packed_format))]
            left = packing.rows_decoded(words, None, interpret, layout=rows_layout, fmt=packed_format)
        else:
            words = [slots[f'words_{q}'].at[mine_rows] for q in range(slots['scales'].shape[1] // 8)]
            left = packing.rows_decoded(words, slots['scales'].at[mine_rows], interpret, layout=rows_layout)
    elif prologue is not None and not prologue_rewritten(family, prologue):
        # A prologue its family's own role rewrites the stage by (`_prologue_stage`) leaves it read as staged.
        stage = slots['rows']
        depth = stage.shape[1]
        tiles_of = {i: fixed_tiles[i] if i in fixed_tiles
                    else read_tiles[i].at[pl.ds((first + index) * depth, depth)] if i in read_tiles
                    else slots[f'side_{i}'] for i in range(len(prologue_axes))}
        left = _transformed(prologue, prologue_axes, stage.at[mine_rows][...], tiles_of, 'rows', mine_rows)
    # A consumer whose rows lie outside its group issues none, whatever its products' form (`_idle`).
    def issued(acc, issue):
        return issue() if idle is None else family.issued_unless(idle, acc, issue)

    if left is not None:
        if summed:
            return nothing, tuple(
                issued(total, lambda total=total, name=name, left=left:
                       total + family.registers_product(left, oriented[name]))
                for total, name in zip(accumulators, matrices, strict=True))
        return nothing, tuple(issued(acc, lambda acc=acc, name=name, left=left: family.register_products(
            acc, (left,), oriented[name])) for acc, name in zip(accumulators, matrices, strict=True))
    rows_of = _rows_of(slots, matrices, mine_rows, transpose_lhs)
    if summed:
        scale = None if scaling is None else scaling(index, slots)
        return nothing, tuple(
            issued(total, lambda total=total, name=name:
                   family.summed_apart(total, rows_of[name], oriented[name], scale))
            for total, name in zip(accumulators, matrices, strict=True))
    if chosen is not None:
        (name,) = matrices
        step = first + index
        which = chosen.which(step)

        def chosen_issue(product, acc):
            # The product's sum from its own first step on, issued at its steps alone (`issued_one_of`).
            acc = family.added_onto(acc, step > chosen.starts[product])
            return issued(acc, lambda: family.products(acc, rows_of[name], oriented[name]))

        return nothing, family.issued_one_of(which, accumulators, chosen_issue)
    if block_scaled:
        return nothing, tuple(family.scaled_products(acc, rows_of['matrix'], oriented[name],
                                                     (slots['a_scales'], slots['b_scales']))
                              for acc, name in zip(accumulators, matrices, strict=True))
    if boundary is not None:
        ((acc,), (name,)) = accumulators, matrices
        acc, following = family.products_across(acc, rows_of[name], oriented[name], prefetched, boundary)
        return (nothing, (acc,)), following
    return nothing, tuple(
        issued(acc, lambda acc=acc, name=name: family.products(acc, rows_of[name], oriented[name]))
        for acc, name in zip(accumulators, matrices, strict=True))


def _oriented(read, matrices: dict, transpose_rhs: bool) -> dict:
    """Dew: a `_gmm_step`'s matrices as its products read them, `read` their slots (a packed matrix's
    decoded)."""
    return {name: read[name].transpose((1, 0)) if transpose_rhs else read[name] for name in matrices}


def _rows_of(slots, matrices: dict, mine_rows, transpose_lhs: bool) -> dict:
    """Dew: each of a `_gmm_step`'s matrices' rows: this warpgroup's half of the rows' stage (the only half
    its MMA reads, a `[depth, rows]` stage's transposed), or the joined rows' for the paired matrix
    (`matrices`' stream)."""
    stage = slots['rows']
    mine = stage.at[:, mine_rows].transpose((1, 0)) if transpose_lhs else stage.at[mine_rows]
    return {name: mine if stream == 'rows' else slots[stream].at[mine_rows]
            for name, stream in matrices.items()}


def _gmm_prefetch(family: mosaic.Family, matrices: dict, rows) -> Callable:
    """Dew: the walk's prefetch of a plain `_gmm_step`'s slots (`prefetches_fragments`): untransposed rows
    and matrix only (`prefetches`)."""
    (name,) = matrices

    def prefetch(slots):
        rows_of = _rows_of(slots, matrices, rows, transpose_lhs=False)
        oriented = _oriented(slots, matrices, transpose_rhs=False)
        return family.first_fragments(rows_of[name], oriented[name])

    return prefetch

def _transformed(prologue, axes_of: tuple, mine: jax.Array, tiles_of: dict, halved: str, half) -> jax.Array:
    """Dew: `mine`, this warpgroup's half of a stage read into registers, transformed by the prologue on its
    side tiles (`tiles_of`, by operand) cut to the same half along the `halved` launch axis, a side operand by
    group's at each of its rows (`axes.by_rows`), in the stage's dtype."""
    rows = mine.shape[0]
    cut = [axes.by_rows(tiles_of[i].at[axes.placed(axes.held(side_axes), {halved: half})][...], side_axes,
                        rows) for i, side_axes in enumerate(axes_of)]
    return prologue(mine, *cut).astype(mine.dtype)


def _rewritten(family: mosaic.Issuer, prologue, axes_of: tuple, mine, tiles_of: dict, halved: str,
               half) -> None:
    """Dew: `mine` rewritten in place by the prologue (`_transformed`), read in its family's accumulator
    layout (`Family.tile_of`); its reader publishes the writes (`mosaic.Family.published`). Read
    unannotated, Mosaic's layout inference seeds a shared-memory load from the warpgroup MMAs' 64- and
    128-row layouts, none of which tiles a 32-row step of `tgmm`'s transposed stage on mma.sync, and
    exhausts its search (a normed kernel gradient's 64 x 128 x 32 tile on sm120 and sm100)."""
    mine[...] = _transformed(prologue, axes_of, family.tile_of(mine), tiles_of, halved, half)


def _split(refs: tuple, prologue: int, epilogue: int) -> tuple:
    """Dew: a kernel's refs after its operands: the prologue's side operands, the epilogue's, the outputs."""
    return refs[:prologue], refs[prologue:prologue + epilogue], refs[prologue + epilogue:]


@dataclasses.dataclass(frozen=True)
class Beside(Partition):
    """Dew: matrices side by side (`gmm`'s `rhs` and its `beside`), each of the call's groups, depth and
    orientation and a width of its own (`widths`), whole tiles of `block_n` columns: one launch's column tiles
    over their columns in order, column tile t the matrix `which(t)`'s, its column tile `t - starts[which]`
    (`starts`, each one's first), its accumulators stored into that one's output alone. A launch's static
    configuration, no array of its own: its matrices are the launch's operands, in the kernel its refs
    (`read`, `BesideRefs`)."""

    widths: tuple[int, ...]
    block_n: int
    transpose_rhs: bool

    @classmethod
    def taken(cls, matrices: tuple, tiles: Tiles, family: mosaic.Family, *, transpose_rhs: bool,
              parts: tuple) -> "Beside | None":
        """`matrices` (`rhs` and `gmm`'s `beside`) side by side where there are more than one, refused as
        `beside_refusal` says; None for one."""
        if len(matrices) == 1:
            return None
        widths = tuple(matrix.shape[1 if transpose_rhs else 2] for matrix in matrices)
        refusal = beside_refusal(tiles, family, widths, parts=parts)
        if refusal is not None:
            raise ValueError(refusal)
        block_n = tile_columns(tiles, family)
        return cls(cls.started(widths, block_n), widths, block_n, transpose_rhs)

    @property
    def width(self) -> int:
        """Their columns together, the launch's."""
        return sum(self.widths)

    def stored(self, finished: Callable, tile: tuple, outputs: tuple) -> None:
        """A column tile's accumulators stored (`finished`) into its own matrix's output (`outputs`), at its
        own columns."""
        which = self.which(tile[2])
        for index, start in enumerate(self.starts):
            local = (*tile[:2], tile[2] - start, *tile[3:])
            pl.when(which == index)(functools.partial(finished, local, o_gmems=(outputs[index],)))

    def read(self, matrix_gmems: tuple, refs: tuple) -> tuple[tuple, tuple]:
        """In the kernel, the launch's matrix streams (`matrix_gmems`), the first reading each column tile's
        own matrix (`BesideRefs`: the first's ref and the others', the next of `refs`), and the refs past
        them."""
        others, refs = refs[:len(self.starts) - 1], refs[len(self.starts) - 1:]
        return (BesideRefs(self, (matrix_gmems[0], *others)), *matrix_gmems[1:]), refs

    def declared(self, lhs, matrices: tuple, dtype, m: int) -> tuple:
        """Each of `matrices`' output, `[rows, its width]` in `dtype`, its rows `lhs`'s sized by the
        launch."""
        rows = outputs.axis(lhs, 0).sized(m)
        return tuple(outputs.output(dtype, rows, outputs.axis(matrix, 1 if self.transpose_rhs else 2))
                     for matrix in matrices)


@dataclasses.dataclass(frozen=True)
class BesideRefs:
    """Dew: matrices side by side (`beside`) in a kernel, `gmems` their refs in order: a step's block read
    from the tile's matrix (`columns`)."""

    beside: Beside
    gmems: tuple

    def columns(self, group_matrix, column_tile, index_map: Callable) -> rings.Among:
        """A group's view at a launch's column tile: each matrix's at its own column tile, its step's block at
        `index_map`'s indices, the one the tile reads (at every step) chosen as a ring loads it
        (`rings.Among`)."""
        beside = self.beside
        which = beside.which(column_tile)
        return rings.Among(lambda step: which, tuple(
            (_matrix_columns(gmem.at[group_matrix], column_tile - start, block_n=beside.block_n,
                             transpose_rhs=beside.transpose_rhs), index_map)
            for gmem, start in zip(self.gmems, beside.starts, strict=True)))


def _stored(finished: Callable, tile: tuple, out: tuple, beside: Beside | None) -> None:
    """Dew: a tile's accumulators stored (`finished`) into `out`, or where its columns are matrices side by
    side, into its own matrix's output (`Beside.stored`)."""
    if beside is None:
        finished(tile, o_gmems=out)
    else:
        beside.stored(finished, tile, out)


def beside_refusal(tiles: Tiles, family: mosaic.Family, widths: tuple[int, ...], *,
                   parts: tuple) -> str | None:
    """Dew: why a launch at `tiles` on `family` cannot run matrices of `widths` side by side (`Beside`), None
    where it can: one would read or store a column tile as more than its own matrix's with hooks, a pair,
    scales, tables, transposed or packed rows or a swapped launch (`parts`, None where the call has none), a
    depth split, on a family whose products span a pair (each CTA its half of the pair's columns, which may
    lie in two matrices), or one whose transport selects no step's source among arrays
    (`Transport.selects_sources`: cp.async's pipeline takes each stream's one array, `mosaic_sm80`); and each
    width whole column tiles, one at least (a matrix of no columns is no column tile of the launch, its
    products apart). Exact sums are a tile's accumulation alone, whichever matrix it reads, so they are taken.
    The launch's refusal (`gmm`) and the router's eligibility read it alike."""
    if any(part is not None for part in parts):
        return ("matrices side by side take no hooks, pair, scales, tables, transposed or packed rows, "
                "swapped launch")
    if tiles.splits > 1 or family.pairs_product or not family.selects_sources:
        return "matrices side by side take no depth split, a pair's products or a pipeline's walk"
    columns = tile_columns(tiles, family)
    if any(width % columns or not width for width in widths):
        return f"matrices side by side are whole tiles of {columns} columns each, one at least: {widths}"
    return None


def _matrix_columns(gmem, column_tile, *, block_n: int, transpose_rhs: bool):
    """Dew: a group's matrix at a column tile: its rows (`transpose_rhs`'s `[width, depth]`) or columns."""
    columns = pl.ds(column_tile * block_n, block_n)
    return gmem.at[columns] if transpose_rhs else gmem.at[:, columns]


def _rhs_index(transpose_rhs: bool):
    """A pipeline step's block of the matrix's column tile: its depth block `step`."""
    return (lambda step: (0, step)) if transpose_rhs else (lambda step: (step, 0))


def tgmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, out_dtype,
         multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False, prologue=None,
         prologue_operands: tuple = (), prologue_axes: tuple = (), exact_sums: bool = False,
         dense: bool = False, k_major: bool = False, step_scales: tuple | None = None,
         gather: tuple | None = None, whole_steps: bool = False, registers: int | None = None,
         shared: int | None = None, balanced: bool | tail_of.Tail = True) -> jax.Array:
    """Each group's rows of `lhs` `[rows, depth]` transposed times its rows of `rhs` `[rows, width]`:
    `[groups, depth, width]` in `out_dtype`, an empty group's zero. The depth must divide by
    `tiles.block_m` and the width by `tiles.block_n`, and the tile takes one multiprocessor;
    `tiles.block_k` is the rows a step takes.

    Dew: `prologue(rows, *side_tiles)` transforms each `[block_k, block_m]` stage of `lhs`'s rows (a
    cooperative tile's consumer, half its depth) in place before the group masking and the MMA; side
    operands follow 'rows' and 'depth' as `gmm`'s.
    `exact_sums` adds each step's product into the fp32 sum by an IEEE add, which holds two
    accumulators in registers. A `dense` call (one group, every row) on a family that fixes up a tile
    across programs (`balanced`) runs its output tiles' last wave balanced over the programs; with no
    prologue, its stages are read as loaded (`rows_unstaged`), and so are the runs of its split (each a group
    of whole steps from the call's first row, `sliced`: `whole_steps`, which no other call states).
    With `k_major`, `lhs` is `[depth, rows]` and `rhs` `[width, rows]`, the rows contiguous (an FP8
    product's operands, as Hopper's FP8 tensor cores read both), each group's rows whole steps of
    `tiles.block_k` (the caller's layout: no step straddles groups, so no stage is rewritten), with no
    prologue or split; with `step_scales` besides (`row_scales` `[rows / 128, depth]`, `column_scales`
    `[rows / 128, width]`, fp32; exact sums, 128-row steps), each step's partial times its depth rows' and
    columns' scales before its add (`gmm`'s `step_scales`: a blockwise FP8 recipe's kernel gradient, row
    4518). `gather`, a table for either operand (`(lhs_table, rhs_table)`, int32 `[rows]` or None), makes
    the rows that operand's rows at the table's entries, read where they lie (`gmm`'s `gather`: a mixture of
    experts' tokens, and a scattered product's cotangent at its slots, row 1376), on a ring family's
    producer, with no rows-contiguous layout, step scales or dense call. `registers`, `shared` and
    `balanced` as `gmm`'s, its split's runs at the tile unsplit, as the launch prices them."""
    block_m, block_k = tiles.block_m, tiles.block_k
    family = family_of(tiles, lhs.dtype)
    if k_major and (prologue is not None or tiles.splits > 1 or (step_scales is not None and (
            not exact_sums or block_k != SCALE_DEPTH))):
        raise ValueError(f"a rows-contiguous kernel gradient takes no prologue or split, and its step "
                         f"scales an exact {SCALE_DEPTH}-row step, not {tiles}")
    if prologue is not None and not family.issues_in_consumers:
        raise ValueError(f"a kernel gradient's prologue requires consumer-issued products, not {tiles}'s")
    tables = (None, None) if gather is None else tuple(gather)
    _tgmm_tables_refused(tables, tiles, family, dense=dense, k_major=k_major,
                         step_scaled=step_scales is not None)
    m, k = lhs.shape if k_major else lhs.shape[::-1]
    n, k2 = rhs.shape if k_major else rhs.shape[::-1]
    k, k2 = (rows if table is None else table.shape[0] for rows, table in zip((k, k2), tables, strict=True))
    g = group_sizes.shape[0]
    # Dew: a depth the tile's rows do not divide takes a partial last row tile (`ragged_rows`); a cluster's
    # tiles (`tile_rows`, `tile_columns`: a pair's products spanning two CTAs, or a multicast's row tiles
    # side by side) are whole, as `gmm`'s (`partial_runs`).
    ragged = bool(m % tile_rows(tiles, family))
    if (k != k2 or n % tile_columns(tiles, family) or tiles.cluster > family.cluster
            or (ragged and (tiles.cluster > 1 or not ragged_rows(m, prologue=prologue, k_major=k_major)))):
        raise ValueError(f"a [{k}, {m}] by [{k2}, {n}] kernel gradient does not take {tiles}")
    # A stage read as loaded (`rows_unstaged`), or rewritten by the family's own role (`rewrites`).
    unstaged = not k_major and rows_unstaged(dense=dense or whole_steps, prologue=prologue,
                                             tabled=any(table is not None for table in tables))
    _pair_refused(tiles, family, transpose_lhs=False, exact_sums=exact_sums, scaled=step_scales is not None,
                  parts=(prologue, gather), rewritten=rewrites(family, k_major, unstaged))
    group_sizes = group_sizes.astype(jnp.int32)
    out_dtype = jnp.dtype(out_dtype)
    if tiles.splits > 1:
        return _tgmm_split(lhs, rhs, group_sizes, (g, m, n), tiles=tiles, out_dtype=out_dtype,
                           multiprocessors=multiprocessors, interpret=interpret, prologue=prologue,
                           prologue_operands=prologue_operands, prologue_axes=prologue_axes,
                           exact_sums=exact_sums, gather=gather, dense=dense, registers=registers,
                           shared=shared)
    # Dew: the rows padded to a multiple of 8, as `gmm`'s, past every group; to whole row blocks where the
    # family's loads read past an operand unpredicated (its last group's last block). A table is padded to
    # whole row blocks always, a pad reading row 0 of its operand, which stays as it is: its producer reads
    # a step's entries by plain loads, which no TMA bounds (`_walked_rows`).
    # A rows-contiguous call's stages are read whole as ever (`_whole_stages`).
    # A stage read as loaded (`rows_unstaged`) zeroes no rows past the call's, so its rows are padded to whole
    # steps wherever its loads past them do not read zeros (an interpreted slice clamps).
    bounds = family_of(tiles, lhs.dtype).bounded_loads and not (unstaged and interpret)
    multiple = 8 if bounds else math.lcm(8, block_k)
    row_axis = 1 if k_major else 0
    lhs, rhs = (rows_padded(operand, row_axis, multiple) if table is None else operand
                for operand, table in zip((lhs, rhs), tables, strict=True))
    tables = tuple(
        None if table is None else rows_padded(_normalized_gather(table, operand.shape[row_axis]),
                                               0, math.lcm(8, block_k))
        for operand, table in zip((lhs, rhs), tables, strict=True))
    k = -(-k // multiple) * multiple
    prologue_operands = sides_padded(prologue_operands, prologue_axes, multiple)
    bounded = family_of(tiles, lhs.dtype).bounded_loads and not interpret and tables[0] is None
    if ragged and not bounded:
        # A partial row tile's depth past the rows read as zeros: TMA's loads past an operand read them; an
        # interpreted slice clamps, and cp.async reads past it (a gathered operand's producer's too,
        # `gathered.Ring`), so their depth is padded (`ragged_rows`).
        lhs = rows_padded(lhs, 1, block_m)
    group_starts, group_ends, group_num_blocks, group_block_starts = row_tiles.group_blocks(group_sizes,
                                                                                          block_k)
    # The groups walked longest first (`_segmented`), a group's tiles their steps (ties in their order).
    order = jnp.argsort(-group_num_blocks).astype(jnp.int32)
    # A tile summed across programs adds their exact partials by IEEE adds (`Family.fixed_up`), as a split's.
    tail = _dense_tail(tiles, lhs.dtype, (m, n, k), (lhs.dtype.itemsize, rhs.dtype.itemsize), multiprocessors,
                       balanced=balanced if dense else False)
    walked = g * -(-m // tile_rows(tiles, family)) * (n // tile_columns(tiles, family))
    programs = launched(family, multiprocessors, walked, tail)
    return _tgmm_streamed(
        (lhs, rhs, *tables), (group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts,
                              order),
        (k, m, n, g), tiles=tiles, out_dtype=out_dtype, multiprocessors=programs,
        interpret=interpret, prologue=prologue, prologue_operands=prologue_operands,
        prologue_axes=prologue_axes, exact_sums=exact_sums, tail=tail, k_major=k_major,
        step_scales=step_scales, unstaged=unstaged, registers=registers, shared=shared)


def _tgmm_split(lhs, rhs, group_sizes, shape: tuple, *, tiles: Tiles, out_dtype, multiprocessors: int,
                interpret, prologue, prologue_operands: tuple, prologue_axes: tuple, exact_sums: bool, gather,
                dense: bool, registers: int | None, shared: int | None) -> jax.Array:
    """Dew: `tgmm` at a split tile (`Tiles.splits`, `shape` its groups, depth and width): each group's rows
    in `splits` runs of whole steps from the group's first block, the first and last clipped to its rows,
    each run a group of its own, its fp32 partial summed after. The operands go as they came, the run's call
    padding them (a padded table and its ungathered operand would meet at two row counts). An exact sum's
    runs sum exactly too, and their partials by IEEE adds in the runs' order: a reassociation of IEEE sums,
    which the exact-sum contract (`templates.product.promoted`) allows, where the tensor cores' truncating
    adds are what it rules out."""
    g, m, n = shape
    sliced = row_tiles.sliced(group_sizes, tiles.block_k, tiles.splits)
    # A side operand by group follows its group into each of its runs: its value repeated a run each, in
    # the runs' group-major order, so a run's group reads its own group's (`axes.at_group`, modulo the
    # side's groups, as a stacked layer's call numbers them).
    prologue_operands = tuple(
        side if axes.GROUPS not in side_axes
        else jnp.repeat(side, tiles.splits, axis=list(side_axes).index(axes.GROUPS))
        for side, side_axes in zip(prologue_operands, prologue_axes, strict=True))
    # A dense call's runs are whole steps from its first row, so no step holds two runs' rows: read as
    # loaded, as the call's own planning prices them (`grouped_matmul._Mosaic.unstaged`), not rewritten by
    # a role zeroing neighbours no step has (em35d: the 4B's up kernel gradient issued 6.7x cuBLAS's
    # instructions at 384 threads, its rewriting role's predicated LDSM, PRMT and SEL each step).
    partials = tgmm(lhs, rhs, sliced, tiles=dataclasses.replace(tiles, splits=1), out_dtype=jnp.float32,
                    multiprocessors=multiprocessors, interpret=interpret, prologue=prologue,
                    prologue_operands=prologue_operands, prologue_axes=prologue_axes,
                    exact_sums=exact_sums, gather=gather, whole_steps=dense, registers=registers,
                    shared=shared)
    return partials.reshape(g, tiles.splits, m, n).sum(axis=1).astype(out_dtype)


def rows_unstaged(*, dense: bool, prologue, tabled: bool) -> bool:
    """Dew: whether a `tgmm` stage is read as loaded, no consumer or role rewriting it: a dense call's (one
    group, every row: no step holds a neighbouring group's rows, and its rows past the call's are zeros,
    `tgmm`'s padding; its split's runs alike, whole steps from its first row), with no prologue transforming
    it and no table gathering its rows (`tabled`). Its step is then its products alone
    (`_gradient_unrolled`)."""
    return dense and prologue is None and not tabled


def prologue_rewritten(family: mosaic.Family, prologue) -> bool:
    """Dew: whether `gmm`'s prologue rewrites each step's rows stage in its family's own role
    (`mosaic.Family.rewrites_in_persist`, tcgen05's rewriting warpgroup, `persist`'s `prepared`), the
    products reading the stage as rewritten, where its consumers issue no products to take the prologue's
    value in registers (warpgroup MMA's register operand, mma.sync's fragments). The one decision: the
    kernel's role and its planning's (`grouped_matmul._Mosaic.rewritten`)."""
    return prologue is not None and family.rewrites_in_persist


def rewrites(family: mosaic.Family, k_major: bool, unstaged: bool) -> bool:
    """Dew: whether `tgmm`'s stage is rewritten by its family's own role (`mosaic.Family.rewrites_in_persist`,
    a role beside its consumers): its loads' rows, gathered by its load warp's tables or not
    (`gathered.Committed`), never a k-major launch's whole steps of one group or a stage read as loaded
    (`rows_unstaged`); its products' warp reads no tile, so a gathered stage's neighbouring groups' rows are
    masked there (row 1376). The one decision: the kernel's role and its planning's (`_Mosaic.rewritten`)."""
    return family.rewrites_in_persist and not k_major and not unstaged


def _gradient_unrolled(family: mosaic.Family, *, unstaged: bool, exact_sums: bool, scaled: bool) -> bool:
    """Dew: whether a `tgmm` walk takes a rotation of its ring a trip (`mosaic.stepping`, `_unrolled`'s
    reason): where its step is its products, a stage read as loaded (`rows_unstaged`) summed by the tensor
    cores or by its family's persist (`mosaic.Family.sums_in_persist`). Not a stage its consumers zero
    (`_staged_halves`), nor a step's partial summed apart or a window's (`_walked_gradient`), nor a
    step-scaled one (`_unrolled`): wgmma's exact gradient, its windows' ends known at an unrolled rotation's
    slots, spilled 1028 to 2892 bytes at 128 x 128 x 64 (six stages), 128 x 192 x 64 and 64 x 192 x 64 (four),
    dense or grouped, where walked a step a trip it spills none (the GEMM compile gate, EM-G1, EM-G1-r2)."""
    return unstaged and not scaled and (not exact_sums or family.sums_in_persist)


def _whole_stages(slots: dict, mine_depth) -> tuple:
    """Dew: a `tgmm` step's two stages read whole as its products take them: rewritten by its family's own
    role (`mine_depth` None, `_rewritten_stage`), or a k-major launch's depth rows of the rows' block and
    the cotangent's block read transposed (whole steps of one group, nothing to rewrite)."""
    if mine_depth is None:
        return slots['ring_0'].transpose((1, 0)), slots['ring_1']
    return slots['ring_0'].at[mine_depth], slots['ring_1'].transpose((1, 0))


def _stage_rewrite(rewrites: bool, family: mosaic.Family, starts_gmem, ends_gmem, block_k: int) -> dict:
    """Dew: `tgmm`'s persist argument where its family rewrites its stage (`rewrites`): each step's
    neighbouring groups' rows zeroed by the family's own role (`_rewritten_stage`); none elsewhere."""
    if not rewrites:
        return {}
    return {'prepared': functools.partial(_rewritten_stage, family, starts_gmem, ends_gmem, block_k)}


def _rewritten_stage(family: mosaic.Family, starts_gmem, ends_gmem, block_k: int, tile, index, slots) -> None:
    """Dew: a `tgmm` step's neighbouring groups' rows zeroed in both stages, whole (`_zero_neighbours_of`),
    by its family's rewriting role (`mosaic.Family.persist`'s `prepared`), in the layouts it reads them in
    (`rewriting`)."""
    g_i, _, _, steps, start, *_ = tile
    widths = [(stage, zeroing_columns(family, stage.dtype)) for stage in (slots['ring_0'], slots['ring_1'])]
    shares = [(stage.at[:, pl.ds(column, width)], None) for stage, width in widths
              for column in range(0, stage.shape[1], width)]
    _zero_neighbours_of(family.rewriting, shares, index + start, starts_gmem[g_i], ends_gmem[g_i], steps,
                        block_k, aligned_skip=True)


def _stored_chunks(family: mosaic.Family, summed, store: Callable, place: tuple) -> None:
    """Dew: a `tgmm` tile's columns stored as its family reads them out (`chunks`: a register home's whole,
    tensor memory's a chunk at a time), each at its column (`store(value, place)`)."""
    row_start, first_row, stored, column, *ladder = place
    for start_column, (chunk,) in family.chunks((summed,)):
        at = column + start_column if start_column else column
        store(chunk, (row_start, first_row, stored, at, *ladder))


def _tgmm_declared(lhs, rhs, sizes, shape: tuple[int, int, int], dtype, k_major: bool):
    """Dew: a `tgmm` launch's output, each group's depth by width, stated as its operands' axes
    (`outputs.output`): its groups `sizes`', its depth `lhs`'s and its width `rhs`'s, each where `k_major`
    lays it, at the sizes the launch computes of them (its operands padded past them)."""
    g, m, n = shape
    out_type = outputs.output(dtype, outputs.axis(sizes, 0), outputs.axis(lhs, 0 if k_major else 1).sized(m),
                              outputs.axis(rhs, 0 if k_major else 1).sized(n))
    if out_type.shape[0] != g:
        raise ValueError(f"a tgmm's outputs are its {g} groups, not its sizes' {out_type.shape[0]}")
    return out_type


@tuning.lowered_once(('shape', 'tiles', 'out_dtype', 'multiprocessors', 'interpret', 'prologue',
                      'prologue_axes', 'exact_sums', 'tail', 'k_major', 'unstaged', 'registers',
                      'shared'))
def _tgmm_streamed(operands: tuple, groups: tuple, shape: tuple, *, tiles: Tiles, out_dtype,
                   multiprocessors: int, interpret, prologue, prologue_operands: tuple, prologue_axes: tuple,
                   exact_sums: bool, tail=None, k_major: bool = False, step_scales=None,
                   unstaged: bool = False, registers: int | None = None, shared: int | None = None
                   ) -> jax.Array:
    """Dew: `tgmm`'s product at a tile `tiles.warpgroups` consumer roles compute, each its share of the depth
    by all the columns, one MMA a step (two: CUTLASS's cooperative sm90 schedule), on its instruction
    family's bundle (`family_of`), its streams running across the persistent loop's tiles where the family
    streams so (the next tile's first row blocks load while the consumers store this one, which a group's
    few rows, a tile's few steps, make most of the time). Each consumer reads its depth share of the rows'
    stage, which it prepares (the prologue, the neighbouring groups' rows zeroed), and the whole
    cotangent's, a column share of which each zeroes, waiting for the other's where a group boundary falls
    in the step; an `unstaged` stage (`rows_unstaged`) is read as loaded, prepared by none and met at no
    boundary. A side operand with rows streams beside the operands; one without is loaded once a tile.
    One multiprocessor a tile, but where a dense call's last wave is balanced (`tail`): a tile then walked
    by several programs in turn, each but its first program's leaving its fp32 partial in a workspace and
    signalling a semaphore, which the first waits on and adds before storing the tile. Interpreted, one
    program walks every program's tiles in turn (the interpreter runs its programs one after another, each
    paying its own setup), a balanced tail's programs ordered so no semaphore is needed (`tail.ordered`).
    `operands` are padded, then each one's gather table (or None); `groups` `group_blocks`' (`tgmm`)."""
    lhs, rhs, tables = operands[0], operands[1], tuple(operands[2:])
    family = family_of(tiles, lhs.dtype)
    rewritten = rewrites(family, k_major, unstaged)
    k, m, n, g = shape
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    consumers, depth_half = tiles.warpgroups, block_m // tiles.warpgroups
    atom, atoms_each = _zeroed_atoms(family, rhs.dtype, block_n, consumers)
    # A consumer's accumulators: its share of the rows by its columns (`tile_columns`); a cluster's tiles.
    part = (depth_half, tile_columns(tiles, family))
    grid = (-(-m // tile_rows(tiles, family)), n // tile_columns(tiles, family))
    pair, pairs = tiles.cluster, multiprocessors // tiles.cluster
    stepped, read, fixed = _stepped(family, prologue_axes, 'rows')
    allocations = tgmm_allocations(tiles, family, lhs_dtype=lhs.dtype, rhs_dtype=rhs.dtype,
                                   gathered=tuple(table is not None for table in tables),
                                   interpret=bool(interpret), prologue_axes=prologue_axes, unstaged=unstaged,
                                   prologue_dtypes=_dtypes_of(prologue_operands), out_dtype=out_dtype,
                                   k_major=k_major, step_scaled=step_scales is not None)
    unrolled = _gradient_unrolled(family, unstaged=unstaged, exact_sums=exact_sums,
                                  scaled=step_scales is not None)

    def body(sizes_gmem, starts_gmem, ends_gmem, blocks_gmem, block_starts_gmem, order_gmem, lhs_gmem,
             rhs_gmem, *refs):
        table_gmems, refs = _tables_placed(tables, refs)
        *side_gmems, o_gmem = refs
        side_gmems, workspace = (side_gmems, None) if tail is None else (side_gmems[:-1], side_gmems[-1])
        scale_gmems, side_gmems = _scales_apart(side_gmems, step_scales is not None)
        ready = None if tail is None else family.partials_ready((pair, pairs, consumers), bool(interpret))
        schedule = functools.partial(
            _tgmm_scheduled, groups=(sizes_gmem, blocks_gmem, block_starts_gmem, order_gmem),
            operands=(lhs_gmem, rhs_gmem, side_gmems), tail=tail, programs=pairs, grid=grid,
            tiles=tiles, stepped=stepped, prologue_axes=prologue_axes, interpreted=bool(interpret),
            k_major=k_major, scale_gmems=scale_gmems, table_gmems=table_gmems)

        def scope(fixed_smems, fixed_barriers, *meeting, streams):
            halves = meeting[0] if meeting else None
            names = tuple(allocations.blocks)

            def issue(wg, tile, walk, *, active):
                g_i, depth, _, steps, start, *_ = tile
                if tail is not None and not active:
                    return None  # past its program's range: another program walks the tile
                mine_depth = pl.ds(wg * depth_half, depth_half)
                mine_columns = (wg * atoms_each, atoms_each, atoms_each * consumers > block_n // atom)
                fixed_refs = [smem.at[wg] for smem in fixed_smems]
                group = (starts_gmem[g_i], ends_gmem[g_i], steps, block_k)
                first = pl.multiple_of(block_starts_gmem[g_i], block_k) // block_k
                if active:
                    # A side operand by group at the tile's (`axes.at_group`).
                    family.loaded([(side_gmems[i].at[axes.named(prologue_axes[i], {'depth': depth} | (
                                        axes.at_group(prologue_axes[i], side_gmems[i].shape, g_i)))],
                                    ref, barrier.at[wg])
                                   for i, ref, barrier in zip(fixed, fixed_refs, fixed_barriers,
                                                              strict=True)])

                def staged_at(index, slots):
                    # The step of its group's the program walks, from its first (`start`).
                    index = index + start
                    if rewritten or k_major:
                        return _whole_stages(slots, None if rewritten else mine_depth)
                    if unstaged:
                        return slots['ring_0'].at[:, mine_depth].transpose((1, 0)), slots['ring_1']
                    # A vector read from global memory at the step's row block (`read`).
                    read_refs = {i: side_gmems[i].at[pl.ds((first + index) * block_k, block_k)] for i in read}
                    hook = (None if prologue is None
                            else (prologue, prologue_axes, stepped, fixed, fixed_refs, read_refs))
                    return _staged_halves(family, [slots[name] for name in names], (index, *group),
                                          (mine_depth, mine_columns), hook, halves)

                scaling = _tgmm_scaling(step_scales is not None, mine_depth)
                return _walked_gradient(family, walk, staged_at, part, exact_sums, bool(interpret), scaling,
                                        family.summed_window(tiles.block_k))

            finish = functools.partial(_tgmm_finished, family=family, tiles=tiles, m=m, o_gmem=o_gmem,
                                       partials=None if tail is None else (workspace, ready))
            walks = (g * grid[0] * grid[1],) if tail is None else (tail.segments, pairs)
            family.persist(streams, walks, schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=allocations.outputs, interpret=bool(interpret),
                           in_flight=family.in_flight, unrolled=unrolled, accumulators=(part,),
                           producer=tiles.producers.of(TGMM),
                           chunks=tiles.stated(allocations.outputs, family),
                           **_summed(family, tiles, exact_sums),
                           **_stage_rewrite(rewritten, family, starts_gmem, ends_gmem, block_k))

        family.scoped(scope, *allocations.staged,
                      streams=family.streams(allocations.blocks, tiles.stages, consumers,
                                             multicast=multicast(tiles, family, allocations.shared),
                                             swizzles=SCALE_SWIZZLES if step_scales is not None else None,
                                             **_tgmm_gathered(tables, allocations.blocks, bool(interpret))))

    out_type = _tgmm_declared(lhs, rhs, groups[0], (g, m, n), out_dtype, k_major)
    kernel = family.kernel(body, out_type=out_type, grid=(1 if interpret else pairs,),
                           grid_names=("programs",), consumers=consumers + rewritten, interpret=interpret,
                           approx_math=APPROX_MATH, name=TGMM, cluster=pair, registers=registers,
                           producer=tiles.producers.of(TGMM), shared=shared, moves=MOVES,
                           resident=tiles.programs, scalars=len(groups),
                           cost=unit_cost.estimate(2 * k * m * n, lhs, rhs, *prologue_operands, out_type))
    return kernel(*groups, lhs, rhs, *(table for table in tables if table is not None), *(step_scales or ()),
                  *prologue_operands, *_workspace(tail, (pair, pairs, 2, consumers, *part), interpret))

def _tgmm_finished(wg, tile: tuple, summed, store, *, active, family: mosaic.Family, tiles: Tiles, m: int,
                   o_gmem, partials) -> None:
    """Dew: consumer `wg`'s share of a `tgmm` tile stored (`mosaic.Family.persist`'s `finish`): at once, or,
    a balanced tail's tile (`partials`: its workspace and readiness flags), its program's partial and the
    other programs' added first (`mosaic.Family.fixed_up`); nothing where its program walks none of it."""
    g_i, depth, columns, _, _, gathered, program = tile
    if partials is not None and not active:
        return
    place = _tgmm_place(m, depth.start + wg * (tiles.block_m // tiles.warpgroups), columns.start, tiles)
    out_gmem = o_gmem.at[g_i]
    if partials is None:
        _stored_chunks(family, summed, lambda value, at: store(0, value, out_gmem, at), place)
        return
    workspace, ready = partials
    first, gathered = gathered
    cta = mosaic.cluster_index(tiles.cluster)
    family.fixed_up(summed, workspace.at[cta, :, 0, wg], None if ready is None else ready.at[cta, :, wg],
                    program, first, gathered, lambda value: store(0, value, out_gmem, place),
                    firsts=workspace.at[cta, :, 1, wg])


def ragged_rows(m: int, *, prologue, k_major: bool) -> bool:
    """Dew: whether `tgmm` takes a depth `m` (its output's rows) that its tile's rows do not divide, its
    last row tile partial (row 2295: GPT-OSS's 2880 divides by no cooperative wgmma tile's 128 rows, so its
    kernel gradient ran one consumer's 64 x 192 tile, 1.79x Triton's time): its depth's loads past the
    rows reading zeros (or padded), each consumer storing only its rows within the depth (`_tgmm_place`),
    in whole multiples of 8 rows a store takes; no prologue (whose side operands follow the depth) and no
    rows-contiguous layout (whose depth is its operands' outer axis, the step-scaled gradient's)."""
    return prologue is None and not k_major and m % TMA_ROWS == 0


def _tgmm_place(m: int, start, column, tiles: Tiles) -> tuple:
    """Dew: where a consumer's share of a kernel gradient's tile from row `start` and `column` is stored
    (`Family.persist`'s `store`): its rows within the depth `m` (`ragged_rows`: a partial last tile's), in
    `stores.store_rows`' ladder of the share's rows down to their least multiple the depth leaves; the share
    whole where the tile divides the depth."""
    share = tiles.block_m // tiles.warpgroups
    if m % tiles.block_m == 0:
        return start, 0, share, column, share
    # The ladder's rungs are powers of two: the least the largest dividing every count it stores.
    least = math.gcd(m, share)
    return start, 0, jnp.clip(m - start, 0, share), column, share, least & -least


def _tables_placed(tables: tuple, refs: tuple) -> tuple:
    """Dew: a `tgmm` kernel's gather tables' references, each operand's (None where it has none), leading
    the rest, and the references after them."""
    placed = []
    for table in tables:
        placed.append(None if table is None else refs[0])
        refs = refs if table is None else refs[1:]
    return tuple(placed), refs


def _tgmm_gathered(tables: tuple, blocks: dict, interpret: bool) -> dict:
    """Dew: a gathered kernel gradient's streams (`family.streams`' arguments, `_gathered`'s for `gmm`): each
    gathered operand's ring (`ring_0` the rows', `ring_1` the cotangent's) its producer's gather, or
    interpreted each of a step's rows its copies (`rings.piece_copies`, at no stated swizzle)."""
    names = _tgmm_rings(tuple(table is not None for table in tables))
    if not names:
        return {}
    if interpret:
        return {'copies': {name: blocks[name][0][0] * rings.piece_copies(blocks[name][0], blocks[name][1],
                                                                          None, 1) for name in names}}
    return {'gathered': names}


def _tgmm_rings(gathered: tuple[bool, ...]) -> tuple[str, ...]:
    """Dew: a kernel gradient's gathered rings, `ring_0` the rows', `ring_1` the cotangent's, of whether each
    is a table's."""
    return tuple(name for name, held in zip(('ring_0', 'ring_1'), gathered, strict=True) if held)


def _tgmm_scheduled(index, *, groups: tuple, operands: tuple, tail: tail_of.Tail | None, programs: int,
                    grid: tuple, tiles: Tiles, stepped: tuple, prologue_axes: tuple,
                    interpreted: bool, k_major: bool = False, scale_gmems=None,
                    table_gmems: tuple = (None, None)) -> tuple:
    """Dew: a streamed kernel gradient's tile at the persistent walk's `index` (`mosaic.Family.persist`'s
    schedule): whether its program walks any of its rows, how many steps, its streams' views (step `j` its
    group's `j`-th row block from the program's first, at the tile's depth and columns), and its group,
    depth and columns, its group's steps, the program's first of them, its partials' gatherer (`walked`)
    and its program. `groups` are the group sizes', block counts', first blocks' and walking order's
    references, `operands` the rows', cotangent's and side operands', and `table_gmems` each operand's
    gather table (a gathered operand's step its rows at the table's entries, `rings.Rows` of one row)."""
    sizes_gmem, blocks_gmem, block_starts_gmem, order_gmem = groups
    lhs_gmem, rhs_gmem, side_gmems = operands
    index = tail_of.ordered(tail, index, programs, interpreted)
    g_i, snaked, start, count, gathered = _segmented(tail, index, programs, grid[0] * grid[1], order_gmem,
                                                     sizes_gmem.shape[0])
    m_i, n_i = tail_of.planar_snake(snaked, grid, 1, tiles.grid_block_n)
    family = family_of(tiles, lhs_gmem.dtype)
    rows, width = tile_rows(tiles, family), tile_columns(tiles, family)
    # This multiprocessor's rows of its tile (its pair's, `tile_rows`) and the tile's columns, its
    # accumulators' (a pair's both, `tile_columns`).
    depth = pl.ds(m_i * rows + mosaic.cluster_index(tiles.cluster) * tiles.block_m, tiles.block_m)
    columns = pl.ds(n_i * width, width)
    first = pl.multiple_of(block_starts_gmem[g_i], tiles.block_k) // tiles.block_k + start
    steps = jnp.where(sizes_gmem[g_i] > 0, blocks_gmem[g_i], 0)
    walks = steps if count is None else count

    def row_block(j):
        return first + j, 0

    def rows_block(j):
        return 0, first + j

    def gathered_rows(table, block):
        return lambda j: rings.Rows(table, (first + j) * tiles.block_k, (block,), 1)

    # A pair's step: both CTAs' blocks of an operand, each CTA loading its own along the features
    # (`rings.Paired`, as `gmm`'s pair views its operands): their minor axis, a rows-contiguous call's major.
    features = 0 if k_major else 1

    def pair_block(j):
        return rings.Paired(rows_block(j) if k_major else row_block(j), features)

    pair_depth = pl.ds(m_i * rows, rows)
    views: dict[str, tuple[Any, Callable[..., Any]]] = (
             {'ring_0': (lhs_gmem.at[pair_depth] if k_major else lhs_gmem.at[:, pair_depth], pair_block),
              'ring_1': (rhs_gmem.at[columns] if k_major else rhs_gmem.at[:, columns], pair_block)}
             if family.pairs_product else
             {'ring_0': (lhs_gmem.at[depth], rows_block), 'ring_1': (rhs_gmem.at[columns], rows_block)}
             if k_major else
             # A gathered operand's block this program's own, as its view's extent (`depth`, `columns`: a
             # pair's program's half of the pair's rows).
             {name: (gmem.at[:, extent], row_block) if table is None
              else (gmem, gathered_rows(table, extent.start // width_of))
              for name, gmem, table, extent, width_of in zip(('ring_0', 'ring_1'), (lhs_gmem, rhs_gmem),
                                                             table_gmems, (depth, columns),
                                                             (tiles.block_m, width), strict=True)})
    if scale_gmems is not None:
        # Its depth rows' scales and its columns', a step's row of each at the tile's (`step_scales`).
        views['row_scales'] = (scale_gmems[0].at[:, depth], row_block)
        views['column_scales'] = (scale_gmems[1].at[:, columns], row_block)
    views = {**views,
             **{f'ring_{2 + position}': (
                 side_gmems[i].at[axes.placed(prologue_axes[i], {'rows': axes.WHOLE}, depth)],
                 lambda j, side_axes=prologue_axes[i]: axes.placed(side_axes, {'rows': first + j}, 0))
                for position, i in enumerate(stepped)}}
    return walks > 0, walks, views, (g_i, depth, columns, steps, start, gathered, index[-1])


def _segmented(tail: tail_of.Tail | None, index, programs: int, tiles: int, order, groups: int) -> tuple:
    """Dew: a streamed kernel gradient's walk at `index`: its group, its tile's place in the snake, the
    first of the tile's steps the program walks and how many (None: its group's), and its partials'
    gatherer's end (`Tail.segment`). Where nothing balances, a tile a program a round of the programs:
    the groups' tiles longest first (`order`, a group's tiles its steps each) and every other round taken
    by the programs in reverse, so each program's tiles sum to its share of the steps however the rows are
    routed: under qwen3-30b-a3b's gate gradient, groups as skewed as a gamma of CV 0.6 ran 1.143 of a
    uniform routing's time walked group by group in index order, 1.060 longest first (row 3614,
    FS-3614-SKEW2-h100), the alternating rounds bringing the slowest program's predicted share from 1.034
    of an even split to 1.014. A group's tiles stay a run of the walk, so a wave shares its rows in the
    L2."""
    if tail is not None:
        return 0, *tail.segment(index, programs, tiles)
    (step,) = index
    rounds, place = step // programs, step % programs
    length = jnp.minimum(programs, groups * tiles - rounds * programs)
    walked = rounds * programs + jnp.where(rounds % 2 == 0, place, length - 1 - place)
    return order[walked // tiles], walked % tiles, 0, None, None


def _staged_halves(family: mosaic.Family, stages: list, group: tuple, mine: tuple, hook, halves) -> tuple:
    """Dew: a streamed kernel gradient step's operands for one consumer warpgroup, from its `stages` (the
    rows', the cotangent's, then each side operand's with rows): its depth half of the rows, rewritten by
    the prologue (`hook`: it, its operands' axes, which are stepped and which fixed, the fixed ones' tiles
    and the vectors read from global memory, by operand), and the whole cotangent, the neighbouring
    groups' rows zeroed in its shares of both (`group`: the step, the group's first and last row, its
    steps, the rows a step takes; `mine`: its depth half, and its first swizzled row of the cotangent's,
    how many, and whether the last consumer's share runs short of them), the other's share awaited at
    `halves`
    where it zeroed any. Both consumers' MMAs read the whole cotangent stage, so each zeroes its share of
    it and they meet where a group boundary falls in the step, the kernel's one exchange between them."""
    rows_stage, cotangent_stage, *side_stages = stages
    mine_depth, (first_atom, atoms_each, short) = mine
    if hook is not None:
        prologue, axes_of, stepped, fixed, fixed_refs, read_refs = hook
        tiles_of = (dict(zip(stepped, side_stages, strict=True)) | dict(zip(fixed, fixed_refs, strict=True))
                    | read_refs)
        _rewritten(family, prologue, axes_of, rows_stage.at[:, mine_depth], tiles_of, 'depth', mine_depth)
    atom = zeroing_columns(family, cotangent_stage.dtype)
    atoms = cotangent_stage.shape[1] // atom
    # Its depth half's swizzled rows, each its own, and its share of the cotangent's, the last consumer's
    # past the stage's end held (`atoms_each` rounds the share up).
    shares = [(rows_stage.at[:, pl.ds(mine_depth.start + column, atom)], None)
              for column in range(0, mine_depth.size, atom)]
    shares += [(cotangent_stage.at[:, pl.ds((first_atom + j) * atom, atom)],
                first_atom + j < atoms if short else None) for j in range(atoms_each)]
    boundary = _zero_neighbours_of(family, shares, *group, aligned_skip=halves is None)
    family.published()
    if halves is not None:
        pl.when(boundary)(lambda: family.meet(halves))
    return rows_stage.at[:, mine_depth].transpose((1, 0)), cotangent_stage


def _walked_gradient(family: mosaic.Family, walk, staged, shape: tuple, exact_sums: bool, interpret: bool,
                     scaling=None, window: int = 1):
    """Dew: a streamed kernel gradient tile's steps (`walk`, a tile's of `mosaic.Family.persist`;
    `staged(index, slots)` their operands), summed into an fp32 tile of `shape`: as the tensor cores add, or
    with `exact_sums` each step's product summed apart and added into the running sum by an IEEE add (a
    tensor core's accumulation truncates each add), times `scaling(index, slots)`'s multiplier where given
    (`tgmm`'s `step_scales`); with a `window` of several steps (`mosaic.Issuer.summed_window`), the
    tensor cores' sum of each window's steps added at its end, the last window's however short. A tile of
    no steps is zero."""
    if exact_sums and family.sums_in_persist:
        # Its persist sums the windows itself (`_summed`): the steps walked plainly.
        exact_sums = False
    if exact_sums and scaling is None and window > 1:
        def windowed_step(index, slots, running, accumulators):
            (fresh,) = accumulators
            lhs, rhs = staged(index, slots)
            fresh = family.products(fresh, lhs, rhs)
            # A window's end adds its sum into the running sum and starts the next from zeros: a branch on
            # the step, the warpgroup's tiles not selected elementwise by a scalar.
            return lax.cond(lax.rem(index + 1, window) == 0,
                            lambda: (running + fresh, (family.zeros(shape),)),
                            lambda: (running, (fresh,)))

        running, (fresh,) = walk(windowed_step, family.zeros(shape), (shape,))
        return running + fresh
    if exact_sums:
        def exact_step(index, slots, running, accumulators):
            lhs, rhs = staged(index, slots)
            scale = None if scaling is None else scaling(index, slots)
            return family.summed_apart(running, lhs, rhs, scale), accumulators

        running, _ = walk(exact_step, family.zeros(shape), ())
        return running

    def step(index, slots, nothing, accumulators):
        (acc,) = accumulators
        lhs, rhs = staged(index, slots)
        return nothing, (family.products(acc, lhs, rhs),)

    _, (summed,) = walk(step, None, (shape,))
    return summed


def _zero_neighbours_of(family: mosaic.Issuer, refs: list, step, start, end, steps, block_k: int, *,
                        aligned_skip: bool = False):
    """Dew: in a group's first and last row blocks, zero each of `refs` (`[block_k, columns]` stages, each
    one swizzled row wide, so one holds a thread's few registers whatever the tile's width; each with
    whether it is held, None where always) on the rows a neighbouring group holds, by selection; whether
    `step` is either block. With `aligned_skip` (a one-consumer tile), a block the group starts or ends at
    a step's edge holds no neighbour's rows and is left as loaded: GPT-OSS-120B's 256-row groups at 64-row
    steps zeroed two of their four (3.18 ms against 2.52, DBG2295-BAND1). Two consumers zero as before:
    their zeroing is where they meet, and skipping it at aligned edges ran slower on DeepSeek-V3's and Kimi
    Linear's kernel gradients (DBG2295-FACTOR1), for a reason not established."""
    first, last = step == 0, step == steps - 1
    if aligned_skip:
        first = first & (lax.rem(start, block_k) != 0)
        last = last & (lax.rem(end, block_k) != 0)

    def zero(kept):
        for ref, held in refs:
            def zeroed(ref=ref):
                values = family.tile_of(ref)
                rows = family.iota((values.shape[0], values.shape[1]), 0)
                ref[...] = jnp.where(kept(rows), values, jnp.zeros_like(values))

            if held is None:
                zeroed()
            else:
                pl.when(held)(zeroed)

    pl.when(first)(lambda: zero(lambda rows: rows >= lax.rem(start, block_k)))
    pl.when(last)(lambda: zero(lambda rows: rows <= lax.rem(end - 1, block_k)))
    return first | last
