"""An NVFP4 matrix's stages decoded in the template's mainloop (`mosaic_gpu.gmm`'s `scales`; rows 3621, 3733,
5387): each 128 depths' sixteen words a column, in the staged order (`formats.staged_rows`), decoded code by
code by the form's traced reading (`bits.NVFP4_BFLOAT16_FORM`), each code times its block's E4M3
scale, exactly, into bfloat16: into shared-memory buffers its products read as their matrix (`Decoding`,
`stage_decoded`), or, the matrix the products' rows (a decode-sized product swapped, `rows_decoded`), into
the left operand they take from registers, the staged order the warpgroup MMA's fragment, so no decoded stage
reaches shared memory."""

import dataclasses
import functools
import itertools
from collections.abc import Callable
from typing import Any

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.algebra import bits
from dew.nn.kernels.templates import product as product_formats, resources, swizzling

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

@dataclasses.dataclass(frozen=True)
class WindowLayout:
    """Dew: the one layout of a lossless bfloat16 window matrix (`formats.Bf16Window`, row 5387) its encoder,
    its decoders (the XLA one, a stage's lanes), its stream, its price and its form digest read (row 13707).
    Its coding is fixed: a weight's byte (its sign and 7 mantissa bits, or an escape's index) and 4-bit
    offset, packed into uint32 words a stage streams in blocks of sixteen, so a group (one nibble block's
    depths) holds two byte blocks of 64 depths and a window of 15 binades, and its escapes two 16-bit values
    a meta word, low half first. Its meta's placement is its choice: `meta_words` words a group, its base at
    `base_word`, its escapes at `escape_words` (`window_bits` takes the two words), the rest zero. Every
    field is the stored form's (`fingerprint`): the depths a stage decodes are the kernel's schedule, its
    derivation's (`weight_format`'s steps)."""
    meta_words: int = 4
    base_word: int = 0
    escape_words: tuple[int, ...] = (1, 2)

    def __post_init__(self):
        placed = (self.base_word, *self.escape_words)
        if (len(self.escape_words) != 2 or len(set(placed)) != len(placed)
                or not all(0 <= word < self.meta_words for word in placed)):
            raise ValueError(f"a window's meta places its base and its four escapes on distinct words of its "
                             f"{self.meta_words}, not {placed}")

    block_words = 16
    """The uint32 words a block holds a column (a 64-byte row, a stage copy's): each `block_words`-deep slice
    of a block one code of each word (`byte_depth`, `nibble_depth`, `unpacked`)."""
    group = block_words * 8
    """The depths a group shares a base and escapes over: a nibble block's (its words' 8 offsets each)."""
    byte_block = block_words * 4
    """The depths a byte block holds: its words' 4 bytes each."""
    binades = 2 ** 4 - 1
    """The binades a group's 4-bit offsets code (offsets 1 to 15, offset 0 a signed zero or an escape)."""

    @property
    def byte_blocks(self) -> int:
        """The byte blocks a group holds."""
        return self.group // self.byte_block

    @property
    def escapes(self) -> int:
        """The exact values below its window a group holds, two a meta word."""
        return 2 * len(self.escape_words)

    @property
    def byte_words(self) -> int:
        """The uint32 words a group's bytes hold a column."""
        return self.block_words * self.byte_blocks

    @property
    def words(self) -> int:
        """The uint32 words a group holds a column: its bytes' then its nibbles' (one block)."""
        return self.byte_words + self.block_words

    @property
    def weight_bytes(self) -> float:
        """The bytes a weight costs, its group's words and meta over its depths."""
        return 4 * (self.words + self.meta_words) / self.group

    @property
    def base_half(self) -> int:
        """The 16-bit half of a group's meta (`Bf16Window.meta_halves`) its base is."""
        return 2 * self.base_word

    @property
    def escape_halves(self) -> tuple[int, ...]:
        """The 16-bit halves of a group's meta its escapes are, in their slots' order."""
        return tuple(2 * word + half for word in self.escape_words for half in (0, 1))

    def byte_depth(self, group, word, code):
        """The depth the byte plane gives word `word`'s byte `code` (`0..3`) of its `byte_block`-deep
        `group`: each byte's sixteen-depth slice holds the group's words in order, as a stage decode reads
        them a lane at a time (`ternary_staged_depth`'s order)."""
        return self.byte_block * group + self.block_words * code + word

    def nibble_depth(self, group, word, code):
        """The depth the nibble plane gives word `word`'s nibble `code` (`0..7`) of its `group`."""
        return self.group * group + self.block_words * code + word

    def unpacked(self, words: jax.Array, bits: int) -> jax.Array:
        """A plane's codes in their depths' order (`byte_depth`'s or `nibble_depth`'s inverse) from its
        words (`[..., blocks * block_words]` uint32, `32 // bits` codes of `bits` bits a word): a block's
        word `w` holds its code `c` at depth `block_words * c + w` of the block, so each code's shift of a
        block's words is its next `block_words` depths, by shifts and reshapes alone (no gather by the
        staged order, `formats.window_byte_rows`)."""
        *lead, count = words.shape
        blocks, codes = count // self.block_words, 32 // bits
        shifts = (bits * jnp.arange(codes, dtype=jnp.uint32))[:, None]
        held = (words.reshape(*lead, blocks, 1, self.block_words) >> shifts) & ((1 << bits) - 1)
        return held.reshape(*lead, blocks * codes * self.block_words)

    def fingerprint(self) -> bytes:
        """The stored form's fields as bytes, which the form digest folds in (`formats.window_form_version`):
        its meta's placement (an encoding of another reads as another's)."""
        stored = (self.group, self.byte_block, self.meta_words, self.base_word, self.escape_words)
        return repr(stored).encode()


WINDOW = WindowLayout()
"""Dew: the bfloat16 window's layout (`WindowLayout`)."""


TERNARY_STEP = 256
"""Depth step for inline BitNet b1.58 ternary weight decode (`mosaic_gpu.gmm`'s `packed_format='ternary'`)."""


def ternary_staged_depth(group, word, code):
    """Dew: the depth `formats.ternary_staged_rows` gives word `word`'s 2-bit `code` (`0..15`) of `group`."""
    return TERNARY_GROUP * group + 16 * code + word


def _decode_ternary_lanes(blocks: list[jax.Array], meta, group: int, active) -> list[jax.Array]:
    from dew.nn.kernels import formats

    del meta, group, active
    [loaded] = blocks
    return [formats.ternary_code_value((loaded >> jnp.uint32(2 * code)) & jnp.uint32(3), jnp.bfloat16)
            for code in range(16)]


def _decode_window_lanes(blocks: list[jax.Array], meta: jax.Array, group: int, active,
                         layout: WindowLayout = WINDOW) -> list[jax.Array]:
    """A `formats.Bf16Window` group's eight sixteen-depth slices (`[columns, 16]` bfloat16): slice `s`'s bytes
    the `s % 4`-th of its `s // 4`-th byte block's words (`WindowLayout.byte_depth`), its offsets the `s`-th
    nibble of its nibble block's (`WindowLayout.nibble_depth`), its group's base and escapes a column each,
    picked from the stage's meta halves (`meta`, `[columns, depth / 16]`, a group's `2 * meta_words`) by a
    mask (a column slice of a register tile has no layout), its escapes' halves joined into their two words
    (`window_bits`' escape words), each weight's bits `formats.window_bits`' and the lane their bfloat16 (a
    16-bit integer's bits, no arithmetic)."""
    from dew.nn.kernels import formats

    del active
    *bytes_, nibbles = blocks
    which = jax.lax.broadcasted_iota(jnp.int32, meta.shape, 1)
    halves = meta.astype(jnp.int32)

    def column(half: int) -> jax.Array:
        picked = jnp.sum(jnp.where(which == 2 * layout.meta_words * group + half, halves, 0), axis=1)
        return picked.astype(jnp.uint32)[:, None]

    first, second, third, fourth = (column(half) for half in layout.escape_halves)
    base, escapes = column(layout.base_half), (first | (second << 16), third | (fourth << 16))
    lanes = []
    for slice_ in range(packed_depth_slices(layout.group)):
        word = bytes_[slice_ // 4]
        byte = (word >> jnp.uint32(8 * (slice_ % 4))) & jnp.uint32(0xFF)
        offset = (nibbles >> jnp.uint32(4 * slice_)) & jnp.uint32(0xF)
        held = formats.window_bits(byte, offset, base, escapes)
        lanes.append(jax.lax.bitcast_convert_type(held.astype(jnp.uint16), jnp.bfloat16))
    return lanes


def packed_depth_slices(group_depth: int) -> int:
    """The sixteen-depth slices a packed group of `group_depth` depths decodes into, a lane each."""
    return group_depth // 16


@dataclasses.dataclass(frozen=True)
class PackedWeightFormat:
    """Descriptor for inline-decoded scale-free packed weights (`ternary`, `window`; rows 1464, 5387): a
    group of `group_depth` depths a column streams `blocks` blocks of sixteen uint32 words (a 64-byte row
    each, `words_{q}`), beside each step's meta halves where `meta` (`[columns, depth / 16]` uint16, `gmm`'s
    `scales` stream: a window's bases and escapes); its steps, byte cost, live-register cost and lane decoder
    (`decode_lanes(blocks, meta, group, layout)`: the group's sixteen-depth slices in depth order)."""
    name: str
    group_depth: int
    weight_bytes: float
    steps: tuple[int, ...]
    decode_live: int
    decode_lanes: Callable[[list[jax.Array], Any, int, Any], list[jax.Array]]
    blocks: int = 1
    meta_halves: int = 0
    """The 16-bit meta halves a group streams a column beside its words (`gmm`'s `scales` stream), none
    where its weights take no meta (a window's `2 * WindowLayout.meta_words`)."""

    @property
    def meta(self) -> bool:
        """Whether its groups stream meta halves beside their words."""
        return self.meta_halves > 0

    @property
    def words_per_group(self) -> int:
        """The uint32 words a group holds a column."""
        return 16 * self.blocks

    def meta_columns(self, depth: int) -> int:
        """The meta halves a column of `depth` depths streams (`depth` whole groups)."""
        return depth // self.group_depth * self.meta_halves


def weight_format(name: str, layout: WindowLayout = WINDOW) -> PackedWeightFormat:
    """Resolve a scale-free packed weight format name (`'ternary'`, `'window'`) to its descriptor, a window's
    at `layout` (this Dew's, the one served; another's a geometry's witness)."""
    if name == "ternary":
        return PackedWeightFormat(
            name="ternary", group_depth=TERNARY_GROUP, weight_bytes=0.25,
            steps=(TERNARY_STEP,), decode_live=4, decode_lanes=_decode_ternary_lanes)
    if name == "window":
        # A weight's 13 bits; a slice's decode holds its byte and nibble words, its group's base and four
        # escapes a column, and a weight's bits in flight. A stage decodes two groups: their meta halves fill
        # the sixteen columns a register tile of them takes in mma.sync's fragment layout, where one group's
        # eight (or its four words) do not lower ("Cannot cast to layout", compile-only sm80 and sm90).
        return PackedWeightFormat(
            name="window", group_depth=layout.group, weight_bytes=layout.weight_bytes,
            steps=(2 * layout.group,), decode_live=10,
            decode_lanes=functools.partial(_decode_window_lanes, layout=layout),
            blocks=layout.words // 16, meta_halves=2 * layout.meta_words)
    raise ValueError(f"unsupported packed weight format {name!r}; expected 'ternary' or 'window'")


def packed_group_depth(fmt: str) -> int:
    """Depth covered by one group of packed format `fmt`."""
    return weight_format(fmt).group_depth


def packed_blocks(fmt: str, depth: int) -> int:
    """The sixteen-word blocks `depth` depths of packed format `fmt` stream a column (`words_{q}`)."""
    desc = weight_format(fmt)
    return depth // desc.group_depth * desc.blocks
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
    if packed_rows and desc.meta:
        raise ValueError(f"packed({fmt}) rows decode only as a staged matrix: its meta halves are its "
                         "columns', which no swapped call's rows stream")
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
            or rhs.shape[1:] != (width, depth // desc.group_depth * desc.words_per_group)
            or depth % desc.group_depth or tiles.block_k % desc.group_depth or tiles.block_n % 64
            or prologue is not None or paired is not None or transpose_lhs
            or tiles.cluster > 1 or tiles.splits > 1 or tiles.swapped):
        raise ValueError(f"a packed({fmt}) matrix {rhs.shape} is read transposed in {desc.group_depth}-deep "
                         f"steps and 64-col tiles with no prologue, pair, cluster or split, not {tiles}")


def packed_depth(k: int, k2: int, n: int, rhs, tiles, *, packed_rows: bool, scales,
                 packed_format: str | None, transpose_rhs: bool, prologue, paired,
                 transpose_lhs: bool, lhs=None, family=None, dense: bool = False) -> tuple[int, int]:
    """Resolve logical contraction depth `(k, k2)` for packed NVFP4/word-packed calls and validate."""
    desc = weight_format(packed_format) if packed_format is not None else None

    def depth(words: int) -> int:
        if desc is None:
            return 8 * words
        if words % desc.words_per_group:
            raise ValueError(f"packed({desc.name}) words come {desc.words_per_group} a group, not {words}")
        return words // desc.words_per_group * desc.group_depth

    k, k2 = ((depth(k), k2) if packed_rows
             else (k, depth(k2) if scales is not None or desc is not None else k2))
    if desc is not None:
        if (scales is not None) != desc.meta:
            raise ValueError(f"packed({packed_format}) takes {'its meta halves as' if desc.meta else 'no'} "
                             "scales")
        halves = (rhs.shape[0], n, desc.meta_columns(k2))
        if scales is not None and (scales.dtype != jnp.uint16 or tuple(scales.shape) != halves):
            raise ValueError(f"packed({packed_format})'s meta halves are uint16 {halves}, not "
                             f"{scales.dtype} {tuple(scales.shape)}")
        word_taken(rhs, desc.name, tiles, transpose_rhs=transpose_rhs, prologue=prologue,
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
    if weight_fmt != ROW_CODES and shared_decode(tiles, interpret=interpret):
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


CODES = "codes"
"""Dew: the decode of a matrix of 8-bit codes beside 16-bit rows (`Decoding`'s `weight_fmt`), each stage
converted exactly (`mosaic_gpu.gmm`'s, row 8548), where no format's words are packed."""


ROW_CODES = "row codes"
"""Dew: the decode of rows of 8-bit codes beside a 16-bit matrix (`Decoding`'s `weight_fmt`): a swapped
call's, its codes matrix the products' rows (row 5387), each consumer converting its own rows' stage."""


def decode_of(rows, matrix) -> str | None:
    """Dew: a stage decode a call of `rows` beside `matrix` takes where no packed format names one (`gmm`'s):
    its matrix's codes converted (`CODES`), or its rows' (`ROW_CODES`); None where neither is codes."""
    return CODES if converted(rows, matrix) else ROW_CODES if converted(matrix, rows) else None


def converted(rows, matrix) -> bool:
    """Dew: whether a matrix of `matrix` dtype beside rows of `rows` dtype is a matrix of codes each stage of
    which converts exactly into the rows' dtype (`CODES`): 8-bit codes a device converts in registers
    (`mosaic.converts`) beside 16-bit rows, the one decision of its stages' buffers and their decode
    (`mosaic_gpu.gmm_allocations`, `_gmm_mainloop`)."""
    from dew.nn.kernels.templates import mosaic

    return mosaic.converts(matrix) and jnp.dtype(matrix).itemsize == 1 and jnp.dtype(rows).itemsize == 2


@dataclasses.dataclass(frozen=True)
class Decoding:
    """Dew: a consumer's decode of a packed matrix's stages (`_gmm_step`'s): its `buffers` (`[stages,
    columns, depth]` bfloat16), and with `consumers` sharing them, its own share of the columns (`wg`'s),
    after which they meet (`met`). Decodes NVFP4 when `weight_fmt` is None, or a word-packed format
    (`weight_format`: BitNet b1.58 ternary) when set; a stage of 8-bit codes (`CODES`: `gmm`'s FP8 matrix
    beside 16-bit rows, row 8548) it converts exactly (`mosaic.Family.converted`)."""

    buffers: Any
    wg: Any
    consumers: int
    weight_fmt: str | None = None

    @property
    def converts_rows(self) -> bool:
        """Whether it converts the rows' stage, a swapped call's codes (`ROW_CODES`), not the matrix's."""
        return self.weight_fmt == ROW_CODES

    def decode(self, index, slots, interpret: bool, *, layout=None, family=None, rows=None):
        """Step `index`'s stage of words (and NVFP4 scales) or of 8-bit codes (`slots`'), this consumer's
        share of it (its own `rows` of a rows' stage), decoded into its buffer, which it returns."""
        if self.consumers > 1:
            # A tile's first step restarts the buffers' rotation (`index` is the tile's own): every consumer
            # has drained its last tile's products, which read them, before any decodes into them.
            pl.when(index == 0)(self.met)
        buffer = self.buffers.at[index % self.buffers.shape[0]]
        columns = buffer.shape[0] // self.consumers
        mine = pl.ds(self.wg * columns, columns) if self.consumers > 1 else pl.ds(0, buffer.shape[0])
        if self.weight_fmt in (CODES, ROW_CODES):
            # Codes (`gmm_allocations`' `converted`): the warpgroup met where it converts alone, its threads
            # reading each other's writes (`mosaic.synced`); consumers sharing them meet (`met`).
            from dew.nn.kernels.templates import mosaic

            assert family is not None, "a stage of codes converts on its products' family"
            codes = slots['rows'].at[rows] if self.converts_rows else slots['matrix'].at[mine]
            family.converted(codes, buffer if self.converts_rows else buffer.at[mine], interpret)
            if self.consumers == 1:
                mosaic.synced(interpret)
        elif self.weight_fmt is not None:
            desc = weight_format(self.weight_fmt)
            words = [slots[f'words_{q}'].at[mine]
                     for q in range(buffer.shape[-1] // desc.group_depth * desc.blocks)]
            meta = slots['scales'].at[mine] if desc.meta else None
            stage_decoded_words(words, buffer.at[mine], desc, interpret, layout=layout, meta=meta)
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
    _stored(out, ((nvfp4_staged_depth(group, 0, code), lane)
                  for group, code, lane in lanes(words, scales, interpret, layout=layout)))


def _stored(out, placed) -> None:
    """Dew: the decoded lanes of `placed` (`(depth, lane)`: lanes `[columns, 16]` of consecutive depths,
    their stage's in order, a group's 8 or 16 of them) stored into `out` (`[columns, depth]`) a swizzled row
    at a time (`swizzling.SWIZZLE_BYTES` of the lanes' dtype: four bfloat16 lanes), each run stored as it
    fills: a stage's buffer so stages at the widest swizzle, the one a gathered ring's rows stage at
    (`gathered.ring`), which its product reads beside it under one transform. Stored a lane at a time, its
    16-value columns held the buffer to a 32-byte swizzle the gathered rows' 128-byte one could not share,
    and Mosaic's layout inference found no assignment (row 14111: the packed NVFP4 product of gathered rows
    on a GH200)."""
    placed = iter(placed)
    first = next(placed)
    _, lane = first
    run = swizzling.swizzle_columns(lane.dtype, swizzling.SWIZZLE_BYTES) // lane.shape[1]
    for held in itertools.batched(itertools.chain((first,), placed), run):
        joined = jnp.concatenate([each for _, each in held], axis=1)
        out[:, pl.ds(held[0][0], joined.shape[1])] = joined


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
        for code, lane in enumerate(_nvfp4_lanes(loaded(group_words))):
            # The block's scale a column: its group's code-th, picked from the loaded row by a mask (a
            # column slice of a register tile has no layout).
            scale = jnp.sum(jnp.where(which == 8 * group + code, owed, 0.0), axis=1)
            yield group, code, (lane * scale[:, None]).astype(jnp.bfloat16)


def _nvfp4_lanes(words: jax.Array) -> list[jax.Array]:
    """Dew: the eight codes of each of `words` (uint32, a register tile or an array), each code's E2M1 value
    in float32, code by code (`bits.NVFP4_BFLOAT16_FORM`'s four bfloat16x2 registers, each split low half
    first), each restored by the form's one decoded-value rule (`bits.restored`), so `0.0`, `0.5` and each
    normal code come out exact. The registers are the form's traced reading (`BitForm.traced`), compiled
    and interpreted alike: its PTX inline (`inline_forms.registers`) ran slower on both GPUs measured, each
    form its own process and the output bits equal (row 3733): Llama-3-8B's up projection 1.138x at 64 rows
    and 1.069x at 4096 on a GH200 (M-FS-88), 1.006x to 1.037x on an A100 (M-EM-17), the inline form issuing
    more of the ALU's instructions at the same stalls."""
    decoded = bits.NVFP4_BFLOAT16_FORM.traced(words)
    return [bits.restored(half) for register in decoded
            for half in (register & jnp.uint32(0xFFFF), register >> jnp.uint32(16))]


def packed_lanes(words, desc: PackedWeightFormat | str, interpret: bool, *, layout=None, meta=None):
    """Yield `(group, code, lane)` (`[columns, 16]` bfloat16) for a scale-free word-packed weight format
    (`PackedWeightFormat` or its name): `words` its stage's sixteen-word blocks in order, `desc.blocks` a
    group, beside its step's `meta` halves where the format reads them; shared by staged and direct
    in-register decoders (rows 1464, 5387)."""
    resolved = weight_format(desc) if isinstance(desc, str) else desc
    active = None if interpret else layout

    def loaded(ref):
        return ref[...] if active is None else plgpu.load(ref, layout=active, optimized=False)

    halves = None if meta is None else loaded(meta)
    for group in range(len(words) // resolved.blocks):
        blocks = [loaded(block) for block in words[group * resolved.blocks:(group + 1) * resolved.blocks]]
        for code, lane in enumerate(resolved.decode_lanes(blocks, halves, group, active)):
            yield group, code, lane


def stage_decoded_words(words, out, desc: PackedWeightFormat, interpret: bool, *, layout=None,
                        meta=None) -> None:
    """Decode a stage of word-packed weight groups (`desc`) into `out` (`[columns, depth]` bfloat16)."""
    _stored(out, ((desc.group_depth * group + 16 * code, lane)
                  for group, code, lane in packed_lanes(words, desc, interpret, layout=layout, meta=meta)))


def stage_decoded_ternary(words, out, interpret: bool, *, layout=None) -> None:
    """Decode a stage of 2-bit packed BitNet b1.58 ternary words into `out` (row 1464)."""
    stage_decoded_words(words, out, weight_format("ternary"), interpret, layout=layout)
