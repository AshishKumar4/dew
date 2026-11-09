"""A shared-memory block's layout as the Mosaic GPU templates stage it: its swizzle, the bytes a row of it
spans before the pattern repeats (`SWIZZLE_BYTES` at widest, `swizzle_for` a block's contractions), its tiles
(`swizzle_transforms`), plain where a step reads it a row at a time (`UNSWIZZLED`, `plain`), declared on a
block (`staged`) and read back from a ref (`swizzle_of`); and the copies between shared memory and registers
or global memory that read that layout (`copied`, `coalesced`, `copied_swizzle`)."""

import math
from collections.abc import Callable

from jax import numpy as jnp
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import resources

SWIZZLE_BYTES = 128
"""The widest swizzle a shared-memory block's rows are staged at, which an MMA reads conflict-free."""

NARROWEST_SWIZZLE_BYTES = 32
"""The narrowest swizzle Mosaic GPU stages a block's rows at (`plgpu.find_swizzle`'s least)."""

UNSWIZZLED = 0
"""A stream's swizzle (`Family.streams`' `swizzles`) where a step reads its tile a row at a time, at a traced
row (a scan's tokens): staged plain, no tiling or swizzle, which Mosaic GPU indexes at any row."""


def swizzle_columns(dtype, swizzle: int) -> int:
    """The elements of `dtype` a row of a `swizzle`-byte swizzle spans: a swizzled tile's width, so the least
    slice a load of a block swizzled along that axis takes (a packed dtype's two a byte)."""
    return swizzle * 8 // resources.element_bits(dtype)


def swizzle_for(dtype, *contractions: int) -> int:
    """The widest swizzle, in bytes, a shared-memory block of `dtype` takes whose group of elements (the
    swizzle over the itemsize: Mosaic's wgmma issues a product's contraction in groups of it, one swizzle
    for both its shared operands, wgmma.py's `k_group_elems`) divides each of `contractions`, the elements
    the products it takes part in contract over a step. A product contracting over 32 rows of 16-bit
    operands so stages them at 64 bytes, at which its partner block must stage too (K32SWZ3: S = K Qᵀ
    over a 128-wide head lowers with both at 64 and both at 128, not mixed; dV += P dO over 32 rows lowers
    at 64, not 128)."""
    return next((swizzle for swizzle in (SWIZZLE_BYTES, SWIZZLE_BYTES // 2, NARROWEST_SWIZZLE_BYTES)
                 if all(contraction % swizzle_columns(dtype, swizzle) == 0 for contraction in contractions)),
                NARROWEST_SWIZZLE_BYTES)


TILING: Callable[[tuple[int, int]], object] = plgpu.TilingTransform
SWIZZLE: Callable[[int], object] = plgpu.SwizzleTransform
"""Mosaic GPU's tile transforms, which it declares as protocol classes (a type checker takes them for
uninstantiable)."""


TILE_ROWS = 8
"""The rows of a swizzled shared-memory tile (`swizzle_transforms`): a block's rows are sliced in whole
multiples of them (a ring slot's `Pieces`)."""


def swizzle_transforms(dtype, swizzle: int) -> tuple:
    """A shared-memory tile's transforms at `swizzle` bytes: `TILE_ROWS`-row tiles a swizzle's row wide,
    swizzled; a row narrower than the narrowest swizzle (16 bytes: an NVFP4 stage's block scales, row 3621)
    tiled alone, a 16-byte swizzle being no permutation, which Mosaic GPU's swizzle transform refuses."""
    tiling = TILING((TILE_ROWS, swizzle_columns(dtype, swizzle)))
    return (tiling,) if swizzle < NARROWEST_SWIZZLE_BYTES else (tiling, SWIZZLE(swizzle))


def plain(shape: tuple, dtype):
    """A shared-memory block laid out plainly, neither tiled nor swizzled, stated so (`plgpu.PlainTransform`):
    one read at row offsets no tiling divides, which layout inference would otherwise search every tiling
    and swizzle for, each subview of it apart (a Kimi delta attention intra launch's out of its fuel at 32-
    and 64-key blocks on an H100)."""
    return plgpu.SMEM(shape, dtype, transforms=(plgpu.PlainTransform(),))


def swizzle_of(ref) -> int:
    """The swizzle, in bytes, a shared-memory ref's rows are staged at (`staged`): 0 where it is staged plain
    (`UNSWIZZLED`)."""
    if not isinstance(ref, state_types.TransformedRef):
        return UNSWIZZLED
    return max((transform.swizzle for transform in ref.transforms
                if isinstance(transform, plgpu_core.UnswizzleRef)), default=UNSWIZZLED)


def staged(shape: tuple, dtype, swizzle: int | None = None):
    """A shared-memory block of `shape` whose rows are staged at `swizzle` bytes: as layout inference stages
    them where unstated (the widest that fits); otherwise in 8-row tiles of the swizzle's row, explicitly,
    which inference does not take for a block whose products contract over fewer elements than the widest
    swizzle's row (K32SWZ5: a register operand at a 32-row contraction lowered only so); plain where
    `UNSWIZZLED`."""
    if swizzle == UNSWIZZLED:
        return plain(shape, dtype)
    if swizzle is None:
        return plgpu.SMEM(shape, dtype)
    return plgpu.SMEM(shape, dtype, transforms=swizzle_transforms(dtype, swizzle))


def copied(source, destination) -> None:
    """`source` (in shared memory) to `destination` (in global memory) through registers: a GPU without
    TMA's `store_rows` copy, explicitly unoptimized. JAX 0.12 takes a transfer with no `optimized` flag as
    optimized-only (0.11 downgraded it), which a ladder rung of a row or two cannot be, so its layout
    inference searched every rung's layouts to exhaustion (ns-gate1: the sm80 forward failed after about
    1000 s; without the ladder, or with these transfers unoptimized, it lowers in seconds:
    ns-noladder2, ns-abort1). A block whose elements no warpgroup's strided layout holds, a multiple of its
    threads (a ragged store's one row of an odd multiple of 64 columns, as every tile of gpt-oss's 2880-wide
    outputs: row 4474), is copied in a row's layout (`_row_layout`)."""
    strided = math.prod(source.shape) % resources.THREADS == 0
    layout = None if strided else _row_layout(source.shape, source.dtype)
    plgpu.store(destination, plgpu.load(source, layout=layout, optimized=False), optimized=False)


def _row_layout(shape: tuple[int, ...], dtype) -> plgpu_core.SomeLayout:
    """A `[rows, columns]` block's layout a row at a time (`copied`'s): each row's columns over a warp's
    lanes, as many adjacent ones a lane as a whole number of them takes up to 16 bytes, and every warp of
    the warpgroup holding the same; their stores race, writing the same values, which the CUDA memory model
    allows (`fragmented_array`'s tiled store). `columns` a multiple of a warp's lanes."""
    lanes = mgpu_fragments.WARP_SIZE
    vector = math.gcd(shape[-1] // lanes, 16 // jnp.dtype(dtype).itemsize)
    return plgpu.Layout.TILED(mgpu_fragments.Tiling(((1, lanes * vector), (vector,))),
                              warp_dims=(mgpu_fragments.Replicated(mgpu_fragments.WARPS_IN_WARPGROUP),),
                              lane_dims=(-2,), vector_dim=-1)


def coalesced(source, destination) -> None:
    """`copied`, through registers in Mosaic GPU's shared-to-global copy layout (`Layout.SMEM_GMEM_COPY`)
    where `source` tiles it: each lane 16 bytes of a row, a warp's lanes along tiles of 8 rows and the
    widest swizzle a row's bytes are a multiple of, so its stores write whole sectors. The layout
    inference's own choice was the MMA fragment's, a lane's 4 bytes a row, so a warp wrote half sectors: an
    A100's forward of Qwen3.5-27B's gated delta net took 1.162 ms against 0.941 with every store so, its
    states launch 0.422 against 0.314 and its output launch 0.299 against 0.258, the port's 0.984
    (JS-a100copy-a100-0b4eb412b). A walk's 16-bit stores alone (`Mma.stored`): the attention template's
    forward, whose `persist` stores its output so, took 19% longer (JS-2075cells-a100-491c05798), and an fp32
    tile's copy so spilled 48 bytes in the training query/key launch (the recurrence resource gate, sm80)."""
    layout = _copy_layout(source.shape, source.dtype)
    if layout is None:
        copied(source, destination)
    else:
        plgpu.store(destination, plgpu.load(source, layout=layout, optimized=False), optimized=False)


def _copy_layout(shape: tuple[int, ...], dtype) -> plgpu_core.SomeLayout | None:
    """`coalesced`'s layout for a 16-bit tile of `shape`, None where the copy layout tiles none of it."""
    if len(shape) != 2 or jnp.dtype(dtype).itemsize != 2:
        return None
    swizzle = next((swizzle for swizzle in (128, 64, 32, 16) if copy_tiles(shape, dtype, swizzle)), None)
    return None if swizzle is None else plgpu.Layout.SMEM_GMEM_COPY(tuple(shape), jnp.dtype(dtype), swizzle)


def copy_tiles(shape: tuple[int, ...], dtype, swizzle: int) -> bool:
    """Whether Mosaic GPU's tiled copy layout (cp.async's into a swizzled block, and `coalesced`'s) tiles a
    `[rows, columns]` block of `dtype` staged at `swizzle` bytes: its 8-row tiles as wide as the swizzle,
    a warpgroup's four warps' tiles four down the rows, two by two, or four across
    (fragmented_array.tiled_copy_smem_gmem_layout, which refuses any other)."""
    rows, width = shape[-2], shape[-1] * jnp.dtype(dtype).itemsize
    if rows % 8 or width % swizzle:
        return False
    row_tiles, column_tiles = rows // 8, width // swizzle
    return row_tiles % 4 == 0 or (row_tiles % 2 == 0 and column_tiles % 2 == 0) or column_tiles % 4 == 0


def copied_swizzle(shape: tuple[int, ...], dtype) -> int:
    """The swizzle, in bytes, a `[rows, columns]` block of `dtype` a program copies in with cp.async stages
    at: its rows' widest (`resources.widest_swizzle`), or the widest narrower one the copy layout tiles the
    block at (`copy_tiles`) where it tiles none at that, as an A100's key memory backward's 8 rows of 64 bf16
    columns, one tile at 128 bytes, four at 32 (a decay a key dimension 16 wide, the recurrence stability
    sites)."""
    widest = resources.widest_swizzle(dtype, shape[-1])
    narrower = (swizzle for swizzle in (widest, widest // 2, widest // 4)
                if swizzle >= NARROWEST_SWIZZLE_BYTES)
    return next((swizzle for swizzle in narrower if copy_tiles(shape, dtype, swizzle)), widest)
