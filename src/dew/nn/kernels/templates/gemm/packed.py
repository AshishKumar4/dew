"""An NVFP4 matrix's stages decoded in the template's mainloop (`mosaic_gpu.gmm`'s `scales`; rows 3621, 3733,
5387): each 128 depths' sixteen words a column, in the staged order (`formats.staged_rows`), decoded code by
code by the form's PTX inline (`bits.NVFP4_BFLOAT16_FORM`, `inline_forms`), each code times its block's E4M3
scale, exactly, into bfloat16: into shared-memory buffers its products read as their matrix (`Decoding`,
`stage_decoded`), or, the matrix the products' rows (a decode-sized product swapped, `rows_decoded`), into
the left operand they take from registers, the staged order the warpgroup MMA's fragment, so no decoded stage
reaches shared memory."""

import dataclasses
from collections.abc import Callable
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

TERNARY_GROUP = 256
"""Dew: the depths a BitNet b1.58 ternary weight matrix's staged order groups (`ternary_staged_depth`,
row 1464): sixteen uint32 words a column (a 64-byte row), each word's sixteen 2-bit codes decoding into
sixteen-depth slices."""

TERNARY_STEP = 256
"""Depth step for inline BitNet b1.58 ternary weight decode (`mosaic_gpu.gmm`'s `packed_format='ternary'`)."""


def ternary_staged_depth(group, word, code):
    """Dew: the depth `formats.ternary_staged_rows` gives word `word`'s 2-bit `code` (`0..15`) of `group`."""
    return TERNARY_GROUP * group + 16 * code + word


def _decode_ternary_lanes(loaded: jax.Array, active) -> list[jax.Array]:
    from dew.nn.kernels import formats

    del active
    return [formats.ternary_code_value((loaded >> jnp.uint32(2 * code)) & jnp.uint32(3), jnp.bfloat16)
            for code in range(16)]


@dataclasses.dataclass(frozen=True)
class PackedWeightFormat:
    """Descriptor for inline-decoded scale-free packed weights (`ternary`, rows 1464, 5387): word geometry,
    steps, byte cost, live-register cost, and lane decoder."""
    name: str
    codes_per_word: int
    group_depth: int
    weight_bytes: float
    steps: tuple[int, ...]
    decode_live: int
    decode_lanes: Callable[[jax.Array, Any], list[jax.Array]]


def weight_format(name: str) -> PackedWeightFormat:
    """Resolve a scale-free packed weight format name (`'ternary'`) to its descriptor."""
    if name == "ternary":
        return PackedWeightFormat(
            name="ternary", codes_per_word=16, group_depth=TERNARY_GROUP, weight_bytes=0.25,
            steps=(TERNARY_STEP,), decode_live=4, decode_lanes=_decode_ternary_lanes)
    raise ValueError(f"unsupported packed weight format {name!r}; expected 'ternary'")


def packed_group_depth(fmt: str) -> int:
    """Depth covered by one 16-word group of packed format `fmt` (`'ternary'`)."""
    return weight_format(fmt).group_depth


def packed_codes_per_word(fmt: str) -> int:
    """Number of weight codes packed in one uint32 word of packed format `fmt`."""
    return weight_format(fmt).codes_per_word
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


def word_taken(rhs, fmt: str, tiles, *, transpose_rhs: bool, prologue, paired, transpose_lhs: bool,
               depth: int, width: int, lhs=None, family=None, dense: bool = False,
               packed_rows: bool = False) -> None:
    """Refuse a scale-free word-packed weight call (`gmm`'s `packed_format`, rows 8548, 1464, 5387) its
    decode cannot take: staged as matrix (`rhs` `[G, width, depth // per]`) or swapped directly into `wgmma`
    register operand A (`packed_rows=True`, `lhs` `[rows, depth // per]`, dense, 64 rows/consumer)."""
    desc = weight_format(fmt)
    if packed_rows:
        if (not dense or lhs is None or lhs.dtype != jnp.uint32 or not transpose_rhs
                or tiles.block_k % desc.group_depth or tiles.block_m // tiles.warpgroups != 64
                or family is None or not rows_supported(family)
                or prologue is not None or paired is not None or transpose_lhs
                or tiles.cluster > 1 or tiles.splits > 1 or tiles.swapped):
            raise ValueError(f"packed({fmt}) swapped rows require dense uint32 lhs, transpose_rhs, "
                             f"{desc.group_depth}-deep steps, 64 rows/consumer on wgmma, not {tiles}")
        return
    if (not transpose_rhs or rhs.dtype != jnp.uint32
            or rhs.shape[1:] != (width, depth // desc.codes_per_word)
            or depth % desc.group_depth or tiles.block_k % desc.group_depth or tiles.block_n % 64
            or prologue is not None or paired is not None or transpose_lhs
            or tiles.cluster > 1 or tiles.splits > 1 or tiles.swapped):
        raise ValueError(f"a packed({fmt}) matrix {rhs.shape} is read transposed in {desc.group_depth}-deep "
                         f"steps and 64-col tiles with no prologue, pair, cluster or split, not {tiles}")


def packed_depth(k: int, k2: int, n: int, rhs, tiles, *, packed_rows: bool, scales,
                 packed_format: str | None, transpose_rhs: bool, prologue, paired,
                 transpose_lhs: bool, lhs=None, family=None, dense: bool = False) -> tuple[int, int]:
    """Resolve logical contraction depth `(k, k2)` for packed NVFP4/ternary calls and validate."""
    per = packed_codes_per_word(packed_format) if packed_format is not None else 8
    k, k2 = ((per * k, k2) if packed_rows
             else (k, 8 * k2 if scales is not None else (per if packed_format is not None else 1) * k2))
    if packed_format is not None:
        if scales is not None:
            raise ValueError("packed_format cannot combine with NVFP4 scales")
        word_taken(rhs, packed_format, tiles, transpose_rhs=transpose_rhs, prologue=prologue,
                   paired=paired, transpose_lhs=transpose_lhs, depth=k2, width=n,
                   lhs=lhs, family=family, dense=dense, packed_rows=packed_rows)
    return k, k2


def rows_supported(family) -> bool:
    """Whether `family`'s products read an NVFP4 matrix decoded in registers (`rows_decoded`): its register
    family prefix is declared in `bits.NVFP4_BFLOAT16.fragments` (row 12038)."""
    return family.name.split(".")[0] in bits.NVFP4_BFLOAT16.fragments
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
            or not rows_supported(family) or any(hook is not None for hook in hooked)
            or tiles.cluster > 1 or tiles.splits > 1 or tiles.swapped):
        raise ValueError(f"an NVFP4 matrix's rows {lhs.shape} with scales {scales.shape} are a dense "
                         f"product's, its matrix read transposed, in 256-deep steps and 64 rows a "
                         f"consumer on warpgroup MMA's register operand, with no hooks, pair, cluster or "
                         f"split, not by {tiles}")


def decoding(decoded: list, tiles, wg, interpret: bool, *,
             weight_fmt: str | None = None) -> "Decoding | None":
    """Dew: consumer `wg`'s decode of a packed matrix's stages into `decoded`'s buffers (`gmm_allocations`'),
    shared or its own (`shared_decode`); None for a matrix of no words."""
    if not decoded:
        return None
    if shared_decode(tiles, interpret=interpret):
        return Decoding(decoded[0], wg, tiles.warpgroups, weight_fmt=weight_fmt)
    return Decoding(decoded[0].at[wg], 0, 1, weight_fmt=weight_fmt)

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
    after which they meet (`met`). Decodes NVFP4 when `weight_fmt` is None, or a word-packed format
    (`weight_format`: BitNet b1.58 ternary) when set."""

    buffers: Any
    wg: Any
    consumers: int
    weight_fmt: str | None = None

    def decode(self, index, slots, interpret: bool, *, layout=None):
        """Step `index`'s stage of words (and NVFP4 scales) (`slots`'), this consumer's share of it, decoded
        into its buffer, which it returns."""
        if self.consumers > 1:
            # A tile's first step restarts the buffers' rotation (`index` is the tile's own): every consumer
            # has drained its last tile's products, which read them, before any decodes into them.
            pl.when(index == 0)(self.met)
        buffer = self.buffers.at[index % self.buffers.shape[0]]
        columns = buffer.shape[0] // self.consumers
        mine = pl.ds(self.wg * columns, columns) if self.consumers > 1 else pl.ds(0, buffer.shape[0])
        if self.weight_fmt is not None:
            desc = weight_format(self.weight_fmt)
            words = [slots[f'words_{q}'].at[mine] for q in range(buffer.shape[-1] // desc.group_depth)]
            stage_decoded_words(words, buffer.at[mine], desc, interpret, layout=layout)
        else:
            words = [slots[f'words_{q}'].at[mine] for q in range(buffer.shape[-1] // NVFP4_GROUP)]
            stage_decoded(words, slots['scales'].at[mine], buffer.at[mine], interpret, layout=layout)
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


def stage_decoded(words, scales, out, interpret: bool, *, layout=None) -> None:
    """Dew: a stage of an NVFP4 matrix (`words`, each 128 depths' `[columns, 16]` uint32 in the staged order,
    `formats.staged_rows`; `scales` `[columns, depth / 16]` E4M3) decoded into `out` (`[columns, depth]`
    bfloat16), each code's sixteen-depth slice where the staged order puts it (`lanes`)."""
    for group, code, lane in lanes(words, scales, interpret, layout=layout):
        out[:, pl.ds(nvfp4_staged_depth(group, 0, code), product_formats.NVFP4.depth)] = lane


def rows_decoded(words, scales, interpret: bool, *, layout=None, fmt: str | None = None) -> jax.Array:
    """Dew: a stage of packed matrix rows (NVFP4 with `scales`, or BitNet b1.58 ternary when `fmt` is
    set; row 5387) decoded directly into register operand A (`[rows, depth]` bfloat16 in `wgmma`'s fragment
    layout, zero SMEM staging)."""
    stream = (packed_lanes(words, fmt, interpret, layout=layout) if fmt is not None
              else lanes(words, scales, interpret, layout=layout))
    return jnp.concatenate([lane for _, _, lane in stream], axis=1)


def lanes(words, scales, interpret: bool, *, layout=None):
    """Dew: each code's sixteen-depth slice of a stage (`[columns, 16]` bfloat16), with its group and code,
    group by group: each 128 depths' sixteen words a column give, code by code, its eight slices
    (`bits.NVFP4_BFLOAT16_FORM`'s registers, each split into its two bfloat16 lanes, `_nvfp4_lanes`), each
    times its block's scale over `bits.NVFP4_OWED`, exactly: every value of an E2M1 code times a power-of-two
    scale within bfloat16's range. Its registers lie in the issuing family's fragment `layout`, in which a
    column's scale is a column vector and a lane's slice a contiguous store (the strided layout found no
    assignment for this decode); interpreted, arrays."""
    layout = None if interpret else layout

    def loaded(ref):
        return ref[...] if layout is None else plgpu.load(ref, layout=layout, optimized=False)
    owed = bits.scales_decoded(jax.lax.bitcast_convert_type(loaded(scales), jnp.uint8)) * jnp.float32(
        1 / bits.NVFP4_OWED)
    which = jax.lax.broadcasted_iota(jnp.int32, owed.shape, 1)
    for group, group_words in enumerate(words):
        for code, lane in enumerate(_nvfp4_lanes(loaded(group_words), layout)):
            # The block's scale a column: its group's code-th, picked from the loaded row by a mask (a
            # column slice of a register tile has no layout).
            scale = jnp.sum(jnp.where(which == 8 * group + code, owed, 0.0), axis=1)
            yield group, code, (lane * scale[:, None]).astype(jnp.bfloat16)


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


def packed_lanes(words, desc: PackedWeightFormat | str, interpret: bool, *, layout=None):
    """Yield `(group, code, lane)` (`[columns, 16]` bfloat16) for a scale-free word-packed weight format
    (`PackedWeightFormat` or its name), shared by staged and direct in-register decoders (rows 1464,
    5387)."""
    resolved = weight_format(desc) if isinstance(desc, str) else desc
    active = None if interpret else layout
    for group, group_words in enumerate(words):
        loaded = group_words[...] if active is None else plgpu.load(group_words, layout=active,
                                                                    optimized=False)
        for code, lane in enumerate(resolved.decode_lanes(loaded, active)):
            yield group, code, lane


def stage_decoded_words(words, out, desc: PackedWeightFormat, interpret: bool, *, layout=None) -> None:
    """Decode a stage of word-packed weight groups (`desc`) into `out` (`[columns, depth]` bfloat16)."""
    for group, code, lane in packed_lanes(words, desc, interpret, layout=layout):
        out[:, pl.ds(desc.group_depth * group + 16 * code, 16)] = lane


def stage_decoded_ternary(words, out, interpret: bool, *, layout=None) -> None:
    """Decode a stage of 2-bit packed BitNet b1.58 ternary words into `out` (row 1464)."""
    stage_decoded_words(words, out, weight_format("ternary"), interpret, layout=layout)
