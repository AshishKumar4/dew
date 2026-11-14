"""An NVFP4 matrix's stages decoded in the template's mainloop (`mosaic_gpu.gmm`'s `scales`; rows 3621, 3733,
5387): each 128 depths' sixteen words a column, in the staged order (`formats.staged_rows`), decoded code by
code by the form's PTX inline (`bits.NVFP4_BFLOAT16_FORM`, `inline_forms`), each code times its block's E4M3
scale, exactly, into bfloat16: into shared-memory buffers its products read as their matrix (`Decoding`,
`stage_decoded`), or, the matrix the products' rows (a decode-sized product swapped, `rows_decoded`), into
the left operand they take from registers, the staged order the warpgroup MMA's fragment, so no decoded stage
reaches shared memory."""

import dataclasses
from typing import Any

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.algebra import bits
from dew.nn.kernels.templates import inline_forms, product as product_formats, resources

NVFP4_GROUP = 128
"""Dew: the depths an NVFP4 matrix's staged order groups (`nvfp4_staged_depth`): a group's sixteen word rows
decode, code by code, into its eight sixteen-depth slices."""


def nvfp4_staged_depth(group, word, code):
    """Dew: the depth the staged order (`formats.staged_rows`, the stage decode's `stage_decoded`) gives word
    `word`'s code `code` of group `group`: each code's sixteen-depth slice (an NVFP4 block's,
    `product.NVFP4`) holds the group's words in order, the one layout the packer writes and the decode
    reads."""
    return NVFP4_GROUP * group + product_formats.NVFP4.depth * code + word


NVFP4_STEP = 256
"""The depth an NVFP4 matrix's step decodes (`gmm`'s `scales`): two 128-depth groups of its staged order, so
each column's block scales are a 16-byte row, a copy's least."""


def taken(rhs, scales, tiles, *, transpose_rhs: bool, hooked: tuple, transpose_lhs: bool, depth: int,
          width: int) -> None:
    """Dew: refuses a packed matrix's call (`gmm`'s `scales`) its decode does not take: its words read
    transposed, its scales its blocks', in `NVFP4_STEP`-deep steps and 64-column tiles, with no hook, pair,
    transposed rows, cluster, split or swap (its exact sums, each step's product apart, read the decoded
    stage as any)."""
    if (not transpose_rhs or rhs.dtype != jnp.uint32 or scales.shape != (rhs.shape[0], width, depth // 16)
            or tiles.block_k % NVFP4_STEP or tiles.block_n % 64 or any(hook is not None for hook in hooked)
            or transpose_lhs or tiles.cluster > 1 or tiles.splits > 1 or tiles.swapped):
        raise ValueError(f"an NVFP4 matrix {rhs.shape} with scales {scales.shape} is read transposed in "
                         f"256-deep steps and 64-column tiles with no hooks, pair, cluster or split, not by "
                         f"{tiles}")


def refused(lhs, rhs, scales, tiles, family, *, packed_rows: bool, dense: bool, transpose_rhs: bool,
            transpose_lhs: bool, hooked: tuple, depth: int, width: int) -> None:
    """Dew: `gmm`'s refusal of a packed call (`scales`) its decode does not take: as its matrix (`taken`), or
    with `packed_rows` as its rows (`rows_taken`); none where it multiplies no packed matrix."""
    if packed_rows:
        rows_taken(lhs, scales, tiles, family, dense=dense, transpose_rhs=transpose_rhs, hooked=hooked)
    elif scales is not None:
        taken(rhs, scales, tiles, transpose_rhs=transpose_rhs, hooked=hooked, transpose_lhs=transpose_lhs,
              depth=depth, width=width)


def rows_taken(lhs, scales, tiles, family, *, dense: bool, transpose_rhs: bool, hooked: tuple) -> None:
    """Dew: refuses a packed matrix's call as its products' rows (`gmm`'s `packed_rows`, `rows_decoded`) its
    decode does not take: one matrix (dense), its words `[rows, depth / 8]` and its scales its blocks', the
    call's rows a matrix read transposed, in `NVFP4_STEP`-deep steps and a consumer's 64 rows, on a family
    whose products take their left operand from registers in the decode's layout (the warpgroup MMA's
    fragment), with no hook, pair, cluster or split (its exact sums, each step's product apart, read the
    decoded operand as any)."""
    if (not dense or lhs.dtype != jnp.uint32 or scales.shape != (lhs.shape[0], 8 * lhs.shape[1] // 16)
            or not transpose_rhs or tiles.block_k % NVFP4_STEP or tiles.block_m // tiles.warpgroups != 64
            or family.layout != plgpu.Layout.WGMMA or any(hook is not None for hook in hooked)
            or tiles.cluster > 1 or tiles.splits > 1 or tiles.swapped):
        raise ValueError(f"an NVFP4 matrix's rows {lhs.shape} with scales {scales.shape} are a dense "
                         f"product's, its matrix read transposed, in 256-deep steps and 64 rows a "
                         f"consumer on warpgroup MMA's register operand, with no hooks, pair, cluster or "
                         f"split, not by {tiles}")


def decoding(decoded: list, tiles, wg, interpret: bool) -> "Decoding | None":
    """Dew: consumer `wg`'s decode of a packed matrix's stages into `decoded`'s buffers (`gmm_allocations`'),
    shared or its own (`shared_decode`); None for a matrix of no words."""
    if not decoded:
        return None
    if shared_decode(tiles, interpret=interpret):
        return Decoding(decoded[0], wg, tiles.warpgroups)
    return Decoding(decoded[0].at[wg], 0, 1)


def shared_decode(tiles, *, interpret) -> bool:
    """Dew: whether a packed matrix's consumers share its decoded stages, each decoding its share of the
    columns (`Decoding`): compiled (they meet at a named barrier, lane-level code the interpreter does not
    run), on two consumers or more, each share whole 64-column fragments (the decode's register layout,
    `stage_decoded`'s). Each decoding the whole stage, an H100 ran the packed kernel 5.2-5.3x the matrix
    decoded once a call at 4096 rows (W7-gh200-8-M-NS-8-h100), its decode unhidden under the products."""
    share = tiles.block_n // tiles.warpgroups
    return not interpret and tiles.warpgroups > 1 and share % 64 == 0


@dataclasses.dataclass(frozen=True)
class Decoding:
    """Dew: a consumer's decode of a packed matrix's stages (`_gmm_step`'s): its `buffers` (`[stages,
    columns, depth]` bfloat16), and with `consumers` sharing them, its own share of the columns (`wg`'s),
    after which they meet (`met`)."""

    buffers: Any
    wg: Any
    consumers: int

    def decode(self, index, slots, interpret: bool):
        """Step `index`'s stage of words and scales (`slots`'), this consumer's share of it, decoded into
        its buffer, which it returns."""
        if self.consumers > 1:
            # A tile's first step restarts the buffers' rotation (`index` is the tile's own): every consumer
            # has drained its last tile's products, which read them, before any decodes into them.
            pl.when(index == 0)(self.met)
        buffer = self.buffers.at[index % self.buffers.shape[0]]
        columns = buffer.shape[0] // self.consumers
        mine = pl.ds(self.wg * columns, columns) if self.consumers > 1 else pl.ds(0, buffer.shape[0])
        words = [slots[f'words_{q}'].at[mine] for q in range(buffer.shape[-1] // 128)]
        stage_decoded(words, slots['scales'].at[mine], buffer.at[mine], interpret)
        return buffer

    def met(self) -> None:
        """The consumers' meeting once each has decoded and published its share: a named barrier of all
        their threads (`DECODE_BARRIER`), none where one consumer decodes its own."""
        if self.consumers > 1:
            threads = self.consumers * resources.THREADS
            plgpu.inline_mgpu()(lambda ctx: mgpu_utils.inline_ptx(
                f"bar.sync {DECODE_BARRIER}, {threads};", has_side_effects=True))()


DECODE_BARRIER = 15
"""Dew: the named barrier a packed matrix's consumers meet at (`Decoding.met`): the last of a block's 16,
past Mosaic GPU's own (0, the block's; 1 on, a warpgroup's)."""


def stage_decoded(words, scales, out, interpret: bool) -> None:
    """Dew: a stage of an NVFP4 matrix (`words`, each 128 depths' `[columns, 16]` uint32 in the staged order,
    `formats.staged_rows`; `scales` `[columns, depth / 16]` E4M3) decoded into `out` (`[columns, depth]`
    bfloat16), each code's sixteen-depth slice where the staged order puts it (`lanes`)."""
    for group, code, lane in lanes(words, scales, interpret):
        out[:, pl.ds(nvfp4_staged_depth(group, 0, code), product_formats.NVFP4.depth)] = lane


def rows_decoded(words, scales, interpret: bool) -> jax.Array:
    """Dew: a stage of an NVFP4 matrix's rows (`stage_decoded`'s, its columns the products' rows: a swapped
    product's left operand, row 5387) decoded into registers, `[rows, depth]` bfloat16 in the warpgroup MMA's
    fragment layout: each code's slice concatenated in the depths' own order, which is the staged order's
    (`nvfp4_staged_depth`: a group's code `c` holds its depths `16c` to `16c + 15`, a word's place among
    them), and in which a thread holds of each sixteen depths the words it loaded (a fragment's two adjacent
    columns of each eight), so no value moves between threads."""
    return jnp.concatenate([lane for _, _, lane in lanes(words, scales, interpret)], axis=1)


def lanes(words, scales, interpret: bool):
    """Dew: each code's sixteen-depth slice of a stage (`[columns, 16]` bfloat16), with its group and code,
    group by group: each 128 depths' sixteen words a column give, code by code, its eight slices
    (`bits.NVFP4_BFLOAT16_FORM`'s registers, each split into its two bfloat16 lanes, `_nvfp4_lanes`), each
    times its block's scale over `bits.NVFP4_OWED`, exactly: every value of an E2M1 code times a power-of-two
    scale within bfloat16's range. Its registers lie in the warpgroup MMA's fragment layout, in which a
    column's scale is a column vector and a lane's slice a contiguous store (the strided layout found no
    assignment for this decode); interpreted, arrays."""
    layout = None if interpret else plgpu.Layout.WGMMA

    def loaded(ref):
        return ref[...] if layout is None else plgpu.load(ref, layout=layout, optimized=False)
    owed = e4m3_values(loaded(scales)) * jnp.float32(1 / bits.NVFP4_OWED)
    which = jax.lax.broadcasted_iota(jnp.int32, owed.shape, 1)
    for group, group_words in enumerate(words):
        for code, lane in enumerate(_nvfp4_lanes(loaded(group_words), layout)):
            # The block's scale a column: its group's code-th, picked from the loaded row by a mask (a
            # column slice of a register tile has no layout).
            scale = jnp.sum(jnp.where(which == 8 * group + code, owed, 0.0), axis=1)
            yield group, code, (lane * scale[:, None]).astype(jnp.bfloat16)


def e4m3_values(codes: jax.Array) -> jax.Array:
    """Dew: E4M3 codes (float8_e4m3fn) as float32 by the format's integer decode (`bits.decoded`), exactly,
    with no FP8 conversion: an A100's ptxas refused the packed NVFP4 kernel's scales cast so (W8-a100,
    M-NS-8). The NaN code (no block scale holds it) reads as 480."""
    return bits.decoded(jax.lax.bitcast_convert_type(codes, jnp.uint8).astype(jnp.uint32), bits.E4M3)


def _nvfp4_lanes(words: jax.Array, layout) -> list[jax.Array]:
    """Dew: the eight codes of each of `words` (uint32, a register tile in `layout`), each code's E2M1 value
    in float32, code by code (`bits.NVFP4_BFLOAT16_FORM`'s four bfloat16x2 registers, each split low half
    first), each restored by the form's one decoded-value rule (`bits.restored`), so `0.0`, `0.5` and each
    normal code come out exact. The registers are the form's PTX inline (`inline_forms.registers`, row
    3733), its `jax.numpy` reading where interpreted (`layout` None), where no inline assembly runs."""
    form = bits.NVFP4_BFLOAT16_FORM
    decoded = form.traced(words) if layout is None else inline_forms.registers(form, words, layout)
    return [bits.restored(half) for register in decoded
            for half in (register & jnp.uint32(0xFFFF), register >> jnp.uint32(16))]
