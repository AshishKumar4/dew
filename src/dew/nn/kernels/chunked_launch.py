# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Translated from flash-linear-attention v0.5.2 (fla/ops/utils/index.py, fla/ops/common), which is
# licensed under the MIT license: see licenses/flash-linear-attention-LICENSE.

"""What the chunked recurrences' launches share: the token spans of a call's sequences, packed
documents each its own (`Sequences`, fla's `cu_seqlens`, `chunk_indices` and `chunk_offsets`), a
chunk's place and its masked loads and stores, the tiles a launch takes (`Tiles`, `_Launch`), and a forked
call's documents, each token's part in their walk (`Forked.tokens`) and their place on a packed call's
chunks (`ForkedChunks`). The recurrence template's stages
(`dew.nn.kernels.recurrence`), the ported kernels (`fla_gdn`, `fla_kda`, `mamba_ssd`) and the portable walk
read them alike.
"""

import dataclasses
from collections.abc import Mapping
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from dew.batch import document_firsts
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.tables import Column, column, packed, unpacked
from dew.nn.precision import aligned_sum
from dew.nn.sharding import rows_like


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's tiles: its key and value column blocks (fla's `BK`, `BV`; 0 where a kernel blocks
    neither), its warps, its pipeline stages, and the value heads a program takes (fla's `BH`, Kimi
    delta attention's diagonal kernel's; 0 where a program takes one)."""

    bk: int
    bv: int
    warps: int
    stages: int
    bh: int = 0


def _even(batch: int, length: int, count: int) -> jax.Array:
    """`cu_seqlens` of `count` sequences over `batch` rows of `length` tokens, at least one a row as
    every packed call's, none crossing a row: each row cut as evenly as its share, the first rows'
    shares one larger."""
    shares = [count // batch + (row < count % batch) for row in range(batch)]
    cuts = [row * length + piece * length // share
            for row, share in enumerate(shares) for piece in range(share)]
    return jnp.asarray([*cuts, batch * length], jnp.int32)


@dataclasses.dataclass(frozen=True)
class _Launch(tuning.KernelOp):
    """One of the port's launches, its `Tiles` by activation dtype and GPU block `PICKS`. A subclass names
    its kernel, the arguments whose widths clip its blocks, and its tiles (`_picks`)."""

    config_cls: ClassVar[type[Tiles]] = Tiles

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Normal floats, and chunk and sequence descriptors (`Sequences`) a timed launch can run: the
        rows' own where the descriptors count them, else as many sequences within the rows (`_even`)."""
        arrays = super().operands(specs, seed)
        batch, length = specs[0].shape[:2]
        rows = {spec.shape[-1]: spec.shape[0] for spec in specs if spec.dtype == jnp.int32 and spec.ndim == 2}
        if not rows:
            return arrays
        # A launch reads chunk descriptors `[NT, 4]` (`ChunkSpan`), sequence ones `[N, 8]` (`Span`) or both:
        # a packed call's NT is the tokens' chunks and one more for each of its N sequences (`sequences`).
        chunk_width, span_width = len(ChunkSpan._fields), len(Span._fields)
        chunks, count = rows.get(chunk_width), rows.get(span_width)
        if chunks in (None, batch * -(-length // 64)) and count in (None, batch):
            drawn = sequences(batch, length)
        else:
            count = chunks - -(-batch * length // 64) if count is None else count
            drawn = sequences(batch, length, _even(batch, length, count))
        return [jnp.asarray(drawn.chunks if spec.shape[-1] == chunk_width else drawn.spans)
                if spec.dtype == jnp.int32 and spec.ndim == 2 else array
                for spec, array in zip(specs, arrays, strict=True)]
    NAME: ClassVar[str]
    KEYS: ClassVar[str | None] = None
    VALUES: ClassVar[str | None] = None
    PICKS: ClassVar[Mapping[tuple[str, bool], Tiles]]

    def _tiles(self, ba, tiles: Tiles) -> Tiles:
        """`tiles` with its blocks clipped to the call's key and value widths."""
        bk, bv = (block if name is None else min(block, ba.arguments[name].shape[-1])
                  for block, name in ((tiles.bk, self.KEYS), (tiles.bv, self.VALUES)))
        return dataclasses.replace(tiles, bk=bk, bv=bv)

    def _get_heuristics_config(self, ba) -> Tiles:
        return self._tiles(ba, self.PICKS[_dtype(ba.arguments[self.KEYS or self.VALUES or "g"]), _wide()])


def _picks(tiles: Tiles, *, small: Tiles | None = None, wide_bf16: Tiles | None = None,
           wide_fp32: Tiles | None = None,
           small_fp32: Tiles | None = None) -> Mapping[tuple[str, bool], Tiles]:
    """A launch's tiles by activation dtype and whether the GPU's block holds an A100's shared memory:
    `tiles`, fla's A100 choice, where no other is named; `small`, fla's narrower pick for 99 KB devices.
    The picks are compiled by Triton on a CPU at the training shapes and the widest keys
    (tests/test_kernel_configs.py)."""
    small = tiles if small is None else small
    return {("bfloat16", True): wide_bf16 or tiles, ("bfloat16", False): small,
            ("float32", True): wide_fp32 or tiles, ("float32", False): small_fp32 or small}


def _dtype(x) -> str:
    """The activation dtype a call's tiles are chosen for, fp16 taking bf16's."""
    return "float32" if jnp.dtype(x.dtype).itemsize == 4 else "bfloat16"


def _wide() -> bool:
    """Whether the GPU's block (its assumed generation's off a GPU) holds an A100's shared memory: a GPU
    the targets do not list takes the 99 KB picks, which every Triton GPU's block holds, as a CPU does that
    states no GPU it emulates."""
    target = generation.gpu_target()
    return (target.shared_limit or 0) >= (generation.TARGETS["sm80"].shared_limit or 0)


def zero_padded(x, axis: int, size: int):
    """`x` padded with zeros on the end of `axis` to `size` (as it is where `axis` is that long already):
    the recurrence's routes' and its template's operands widened to whole tiles."""
    if x.shape[axis] == size:
        return x
    return jnp.pad(x, [(0, size - x.shape[axis]) if i == axis else (0, 0) for i in range(x.ndim)])


class Span(NamedTuple):
    """A sequence's row of `Sequences.spans`, its one declaration: its first token and its tokens, the slot
    its second chunk enters less one (`base`), the slots its first chunk enters and its last leaves, the
    call's memory it takes and gives back (`h0`'s and the final's index), and whether it takes it and gives
    it back (1 or 0)."""

    first: Column
    tokens: Column
    base: Column
    entered: Column
    left: Column
    memory: Column
    takes: Column
    gives: Column


class ChunkSpan(NamedTuple):
    """A chunk's row of `Sequences.chunks`, its one declaration: its first token and count (`taken`), and
    the slots of the stored memories it enters and leaves."""

    start: Column
    taken: Column
    entered: Column
    left: Column


class Sequences(NamedTuple):
    """The token spans fla's varlen kernels read (its `cu_seqlens`, `chunk_indices` and `chunk_offsets`,
    fla/ops/utils/index.py) over the `[B * T]` tokens of a call's rows. `chunks` `[NT, 4]`
    (`ChunkSpan`) is each chunk's first token and count, in order, a chunk never crossing a sequence and
    those past the last empty, then the slots of the stored memories (`slots`) it enters and leaves;
    `spans` `[N, 8]` (`Span`) is each sequence's first token and count, the slot its second chunk enters
    less one, the slots its first chunk enters and its last leaves, the call's memory it takes and gives
    back (`h0`'s and the final's index), and whether it takes it and gives it back. Sequences never cross
    rows, so a row's first sequence takes its initial memory and its last gives its final one.

    The memories are stored by where they pass, not by chunk: one slot for each that passes between
    two chunks of a sequence, then a slot a memory, its initial one, which its first chunk enters, and
    its final one's gradient, which its last leaves; and, where documents are packed and enter none of
    their own, one of zeros, which every other document's first chunk enters and last leaves. However
    many documents a row holds, the slots are no more than the tokens' chunks, a slot a memory and one
    (`slots`); an unpacked call's are its chunks, as many as main stored."""
    chunks: jax.Array | np.ndarray
    spans: jax.Array | np.ndarray


def _passing(batch: int, length: int, chunk: int, packed: bool) -> int:
    """The slots of the memories passing between two chunks of a sequence (`Sequences`): each row's
    chunks less its first where the sequences are the rows, else at most every token's chunks."""
    return -(-batch * length // chunk) if packed else batch * (-(-length // chunk) - 1)


def slots(spans: Sequences, batch: int, length: int, chunk: int = 64) -> int:
    """How many memories the kernels store for `spans` over `batch` rows of `length` tokens
    (`Sequences`): those passing between chunks, a row's each, and a slot of zeros where documents are
    packed, whose sequences take more chunks than the rows' own."""
    packed = spans.chunks.shape[0] != batch * -(-length // chunk)
    return _passing(batch, length, chunk, packed) + batch + packed


def packed_sequences(starts, firsts, bound: int, *, chunk: int) -> Sequences:
    """The spans (`sequences`) of a packed call's documents, `starts` `[B, T]` each one's first token,
    `firsts` `[B, D]` where given, at most `bound` a row: the one form a packed call's chunks
    (`templates.recurrence.chunk_forward`) and a forked call's (`ForkedChunks.spans`) take."""
    from dew.nn.inputs import cu_seqlens

    batch, length = starts.shape
    return sequences(batch, length, cu_seqlens(starts, batch * (bound + 1), firsts), chunk=chunk)


def entry_terms(key, value, weight, *, values_first: bool = False) -> list[tuple]:
    """A forked document's entry write's terms (`entry_write`'s) as the write lays them: from `key`
    `[..., R, Dk]` and `value` `[..., R, Dv]` each `(weight, key_r [..., Dk, 1], value_r [..., 1, Dv])`, or
    with `values_first` from `value` `[..., Dv, R]` each `(weight, value_r [..., Dv, 1], key_r [..., 1,
    Dk])` (the TPU forked scan's layout): the one layout the write's sum (`entry_write`) and its transpose
    (`dew.nn.aligned.aligned_cotangents`, the TPU forked scan's backward) take (row 8666). Of rank zero,
    none."""
    rank = key.shape[-2]
    if values_first:
        return [(weight, value[..., :, r:r + 1], key[..., r:r + 1, :]) for r in range(rank)]
    return [(weight, key[..., r, :, None], value[..., r, None, :]) for r in range(rank)]


def entry_write(key, value, weight, *, values_first: bool = False) -> jax.Array:
    """A forked document's entry write `weight sum_r key_r value_r^T` (`recurrence_chunk.Chunk.entry_key`):
    the write `[..., Dk, Dv]`, or with `values_first` `[..., Dv, Dk]` (`entry_terms`' layouts); `weight`
    broadcast against the write. Its terms summed aligned to the largest (`precision.aligned_sum`), so a sum
    in range is its value though its terms leave fp32's range: the one form the portable walk's opening, the
    template's chunks (`ForkedChunks.entries`) and the TPU forked scan take. Of rank zero it is zero, the
    memory it enters unchanged, as the einsum before it gave (row 8967: an aligned sum of no terms raised)."""
    if not key.shape[-2]:
        width = (value.shape[-2], key.shape[-1]) if values_first else (key.shape[-1], value.shape[-1])
        shape = jnp.broadcast_shapes(jnp.shape(weight), key.shape[:-2] + width)
        return jnp.zeros(shape, jnp.result_type(jnp.float32, weight, key, value))
    return aligned_sum(entry_terms(key, value, weight, values_first=values_first))


def sequences(batch: int, length: int, cu_seqlens=None, *, chunk: int = 64) -> Sequences:
    """The spans of `cu_seqlens` (`[N + 1]` int32 over the `batch * length` tokens, each row's
    boundaries among them), or of the rows themselves where None. A traced `cu_seqlens` takes the most
    chunks its N sequences can split `batch * length` tokens into."""
    # The rows' own spans are constants; a traced `cu_seqlens` takes the same arithmetic in XLA.
    xp = np if cu_seqlens is None else jnp
    documents = cu_seqlens is not None
    if cu_seqlens is None:
        cu_seqlens = np.arange(batch + 1, dtype=np.int32) * length
        count = batch * -(-length // chunk)
    else:
        count = -(-batch * length // chunk) + cu_seqlens.shape[0] - 1
    first, lengths = cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1]
    sizes = -(-lengths // chunk)
    offsets = xp.concatenate([xp.zeros(1, np.int32), xp.cumsum(sizes)]).astype(np.int32)
    # A sequence's n-th chunk enters the memory stored at `base + n`, for n past its first: the chunks
    # before it, less each non-empty sequence's first.
    base = offsets[:-1] - xp.cumsum(lengths > 0)
    # A memory's slots follow the passing ones, and the zeros theirs.
    passing = _passing(batch, length, chunk, documents)
    row = first // length
    memory, takes, gives = row, first % length == 0, (first + lengths) % length == 0
    zeros = passing + batch
    entered = xp.where(takes, passing + memory, zeros)
    left = xp.where(gives, passing + memory, zeros)
    index = xp.arange(count, dtype=np.int32)
    owner = xp.clip(xp.searchsorted(offsets, index, side="right") - 1, 0, lengths.shape[0] - 1)
    within = index - offsets[owner]
    real = index < offsets[-1]
    start = first[owner] + within * chunk
    taken = xp.where(real, xp.minimum(chunk, first[owner] + lengths[owner] - start), 0)
    entering = xp.where(within > 0, base[owner] + within, entered[owner])
    leaving = xp.where(within + 1 < sizes[owner], base[owner] + within + 1, left[owner])
    return Sequences(packed(ChunkSpan(start, taken, entering, leaving), xp),
                     packed(Span(first, lengths, base, entered, left, memory, takes, gives), xp))


def spanned(launch, *operands, chunk: int, spans: Sequences | None, **options):
    """`launch` over `operands`, the first `[B, T, ...]`, and the sequences `spans`, the rows' own
    (`sequences`) where None, in chunks of `chunk` tokens."""
    spans = sequences(*operands[0].shape[:2], chunk=chunk) if spans is None else spans
    return launch(*operands, *spans, chunk=chunk, **options)


def _aligned(spans: Sequences, batch: int, length: int, chunk: int) -> bool:
    """Whether every chunk starts on a multiple of `chunk`: the rows' own spans over whole chunks."""
    return (length % chunk == 0 and spans.spans.shape[0] == batch
            and spans.chunks.shape[0] == batch * length // chunk)


class Laid(NamedTuple):
    """A per-chunk launch's call as it lays its tokens out (`laid`): its sequences (`Sequences`), its rows'
    `batch` and `length`, the keys' heads and `width`, the values' heads and `columns`, the value heads a key
    head serves (`group`), the rows' tokens, and whether every chunk is whole and aligned (`_aligned`)."""

    spans: Sequences
    batch: int
    length: int
    key_heads: int
    width: int
    value_heads: int
    columns: int
    group: int
    tokens: int
    aligned: bool


def laid(keys, values, chunk_spans, sequence_spans, chunk: int) -> Laid:
    """The launch geometry (`Laid`) of `keys` `[B, T, H, K]` and `values` `[B, T, HV, V]` over the
    sequences `chunk_spans` and `sequence_spans` (`Sequences`) in chunks of `chunk` tokens."""
    spans = Sequences(chunk_spans, sequence_spans)
    batch, length, key_heads, width = keys.shape
    value_heads, columns = values.shape[2:]
    return Laid(spans, batch, length, key_heads, width, value_heads, columns, value_heads // key_heads,
                batch * length, _aligned(spans, batch, length, chunk))


def _chunk_of(chunk_ref, chunk: int, aligned: bool):
    """A chunk's first token and count (`Sequences.chunks`). Where `aligned` every chunk is whole and
    starts on a multiple of `chunk`, as fla's fixed-length chunks do: the count is the constant
    `chunk`, so its loads and stores take no mask (`read_rows`), and Triton is told the alignment. A
    packed document's chunks start anywhere and end where it does, as fla's varlen ones."""
    start, taken = chunk_ref[column(ChunkSpan, "start")], chunk_ref[column(ChunkSpan, "taken")]
    return (pl.multiple_of(start, chunk), chunk) if aligned else (start, taken)


def _rows(ref, start, size, *rest):
    """The `size` rows from `start` of `ref`, with whatever trailing index `rest` takes."""
    return ref.at[(pl.ds(start, size), *rest)]


def _row_mask(view, count):
    mask = (jnp.arange(view.shape[0]) < count).reshape((-1,) + (1,) * (len(view.shape) - 1))
    return jnp.broadcast_to(mask, view.shape)


def read_rows(view, count):
    """`view`'s rows below `count`, zeros past them: fla's boundary-checked loads, none where `count`
    is a constant covering the view."""
    from jax.experimental.pallas import triton as plgpu

    if isinstance(count, int) and count >= view.shape[0]:
        return view[...]
    return plgpu.load(view, mask=_row_mask(view, count), other=0)


def _gates(g_ref, start, taken, size, offset=0, *rest):
    """A chunk's cumulated decays, fp32, its `size` rows from `offset`, a row past `taken` holding the
    last real row's: it decays nothing further, so every causal pair's `exp2(g_t - g_s)` stays at most
    1 where fla's zero fill, which it masks, could overflow."""
    view = _rows(g_ref, start + offset, size, *rest)
    values = read_rows(view, taken - offset).astype(jnp.float32)
    if isinstance(taken, int) and taken - offset >= size:
        return values
    last = g_ref[(start + taken - 1, *rest)].astype(jnp.float32)
    return jnp.where(_row_mask(view, taken - offset), values, last)


def _row(g_ref, index):
    """One row of a cumulated decay, fp32."""
    return g_ref[index].astype(jnp.float32)


def write_rows(view, count, value):
    """`value` into `view`'s rows below `count`, none past them."""
    from jax.experimental.pallas import triton as plgpu

    if isinstance(count, int) and count >= view.shape[0]:
        view[...] = value.astype(view.dtype)
        return
    plgpu.store(view, value.astype(view.dtype), mask=_row_mask(view, count))


_CHUNK = pl.BlockSpec((None, 4), lambda i_t, *_: (i_t, 0))
"""A per-chunk kernel's own chunk: its first token and count, its memories' slots (`Sequences.chunks`)."""


def _headed(tokens: int, size: int):
    """A per-chunk kernel's value head's `[B * T, size]` rows, its grid's second axis the head."""
    return pl.BlockSpec((tokens, None, size), lambda i_t, h: (0, h, 0))


def _flat(*arrays):
    """`[B, T, ...]` arrays as `[B * T, ...]`, the token axis fla's varlen kernels address."""
    return tuple(x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]) for x in arrays)


def _span(span_ref, length: int, chunk: int, aligned: bool) -> Span:
    """A sequence's row of `Sequences.spans` (`Span`), whether it takes its memory (its row's initial one,
    or its document's) and gives it back (its row's final one, or its document's) as booleans. `aligned`
    as `_chunk_of`."""
    span = unpacked(Span, span_ref.__getitem__)
    if aligned:
        # The rows' own spans: each `length` long, whole chunks from a multiple of `chunk`.
        span = span._replace(first=pl.multiple_of(span.first, chunk), tokens=length)
    return span._replace(takes=span.takes == 1, gives=span.gives == 1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Forked:
    """A forked call's documents as its walks read them (`recurrence_chunk.Chunk.loads`): each document's
    first slot (`firsts` `[B, D]`, `S` past a row's last), the table slot it enters and the stored slot it
    leaves its final in (`loads`, `stores` `[B, D]`), the stored slots' count (`held`), its entry write,
    where the call gives one (`entry`, its `[B, D, Hv, Dk, Dv]` never formed: each walk forms a chunk's
    own), the second slot each document adds to the one it enters, where the call gives them (`adds` `[B, D]`,
    `Chunk.adds`), and where a turned call gives it, the rotation `(cos, sin)` `[B, D, Hv, A]` each document
    turns the memory it takes up by as its walk loads it, its first `A` key pairs (`turning`,
    `templates.recurrence.turned`: minus the document's own turns)."""

    firsts: jax.Array
    loads: jax.Array
    stores: jax.Array
    entry: tuple | None
    held: int = dataclasses.field(metadata={'static': True})
    adds: jax.Array | None = None
    turning: tuple | None = None

    @classmethod
    def of(cls, operands, starts, firsts, work) -> "Forked":
        """`operands`' forked documents (a `recurrence_chunk.Chunk`'s, its `loads`), the entry write's
        operands in `work`."""
        if operands.state is None or operands.state.ndim != 5:
            raise ValueError("a forked call (`recurrence_chunk.Chunk.loads`) gives its table's given "
                             "memories, `state` `[B, S', Hv, Dk, Dv]`")
        loads = jnp.asarray(operands.loads, jnp.int32)
        documents = loads.shape[1]
        begins = document_firsts(starts, documents) if firsts is None else firsts[:, :documents]
        entry = None
        if operands.entry_key is not None:
            entry = tuple(jnp.asarray(t, work) for t in
                          (operands.entry_key, operands.entry_value, operands.entry_weight))
        adds = None if operands.adds is None else jnp.asarray(operands.adds, jnp.int32)
        return cls(begins, loads, jnp.asarray(operands.stores, jnp.int32), entry, operands.held, adds)

    @property
    def walked(self) -> tuple:
        """The operands a walk takes beside the call's (`carrying`, `specs`): its arrays in their fields'
        order, the documents' slots, entry write, added slots and load rotation."""
        return tuple(jax.tree.leaves(self))

    def carrying(self, *walked) -> "Forked":
        """This call's documents from the operands a product's walk was handed (`walked`)."""
        return jax.tree.unflatten(jax.tree.structure(self), walked)

    @staticmethod
    def entry_specs(rows, heads) -> tuple:
        """The entry write's placements on a shard map (`manual_map`), over the mesh axes that split the
        call's `rows` and value `heads`: its keys and values `[B, D, R, H, .]` and weights `[B, D, H]`
        (`specs`, `recurrence_chunk._tpu_specs`)."""
        P = jax.sharding.PartitionSpec
        return (P(rows, None, None, heads, None),) * 2 + (P(rows, None, heads),)

    def specs(self, rows, heads) -> tuple:
        """`walked`'s placements on a launch's shard map (`manual_map`), in its order, over the mesh axes that
        split the call's `rows` and value `heads`: the documents' slots and their added slots a row's, the
        entry write's keys and values `[B, D, R, H, .]` and weights `[B, D, H]` (`entry_specs`), and a
        turned call's load rotation `[B, D, H, A]` its rows' and heads'. Each field's own: a field without one
        raises, so no launch's specs fall out of step with its operands (row 7973: a launch's copy of this
        order took the load rotation's none)."""
        P = jax.sharding.PartitionSpec
        placed = {'firsts': P(rows, None), 'loads': P(rows, None), 'stores': P(rows, None),
                  'adds': P(rows, None),
                  'entry': self.entry_specs(rows, heads),
                  'turning': (P(rows, None, heads, None),) * 2}
        return tuple(placed[path[0].name] if len(path) == 1 else placed[path[0].name][path[1].idx]
                     for path, _ in jax.tree_util.tree_flatten_with_path(self)[0])

    def tokens(self, length: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Each of a call's `length` tokens' part in its walk, `[B, S]` int each, the documents walked in
        place (a document's memory taken up at its first token, mid-chunk wherever it falls): the table slot
        a document's first token takes its memory up from (`Chunk.loads`' one-based, zero a zero memory; -1
        at every other token), the stored slot its last token leaves its memory in (`Chunk.stores`'; zero
        elsewhere), and the document a first token begins (its index into `loads`, the entry write's;
        zero elsewhere). Its sizes are the tokens', none the documents' bound's."""
        firsts = jnp.asarray(self.firsts, jnp.int32)
        held = firsts < length
        lasts = jnp.concatenate([firsts[:, 1:], jnp.full_like(firsts[:, :1], length)], axis=1)
        lasts = jnp.minimum(lasts, length) - 1
        began = jnp.where(held, firsts, length)
        # A row at a time, each row's buffer placed as `firsts`' rows: a row's tokens are its documents'
        # alone, so the maps are placed as the rows are, where scattered by a row index into whole buffers,
        # JAX refused indices and updates split on an Explicit axis (row 9690).
        def laid(fill: int, at, values):
            base = rows_like(jnp.full((firsts.shape[0], length), fill, jnp.int32), firsts)
            return jax.vmap(lambda row, where, value: row.at[where].set(value, mode="drop"))(base, at, values)

        numbers = jnp.zeros_like(firsts) + jnp.arange(firsts.shape[1], dtype=jnp.int32)
        return (laid(-1, began, jnp.asarray(self.loads, jnp.int32)),
                laid(0, jnp.where(held, lasts, length), jnp.asarray(self.stores, jnp.int32)),
                laid(0, began, numbers))


@dataclasses.dataclass(frozen=True)
class AlignedDocuments:
    """A forked call's documents laid from chunks' edges (`Forked`, its walks — Mamba-1's portable scan and
    the turned recurrence's adjoint): each document's tokens moved past the tokens before it to the next
    multiple of the chunk, the slots between tokens zero (leaving the memory as it was), so a walk takes up
    each document's memory at its first chunk's edge and leaves its final at its last chunk's. At most
    `chunk - 1` slots a document, against a row of the call's length were each document laid out apart."""

    slots: jax.Array
    """Each token's slot on the aligned grid `[B, S]`."""
    loads: jax.Array
    """The table slot each aligned chunk takes up `[B, chunks]` (`recurrence_chunk.Chunk.loads`: one-based,
    zero a zero memory), -1 for none: the chunk continues the memory before it."""
    stores: jax.Array
    """The stored slot each aligned chunk leaves its memory in `[B, chunks]`
    (`recurrence_chunk.Chunk.stores`), zero for none."""
    length: int
    """The aligned grid's slots, the chunks' a row."""
    document: jax.Array
    """Each aligned chunk's document index `[B, chunks]`, clipped to the last past a row's live chunks."""
    alive: jax.Array
    """Whether each aligned chunk belongs to a live document `[B, chunks]`."""
    last: jax.Array
    """Whether each aligned chunk is its document's last chunk `[B, chunks]`."""
    final: jax.Array
    """Whether each aligned chunk is the row's last live document's last chunk `[B, chunks]`."""
    within: jax.Array
    """Each aligned slot's offset within its document `[B, length]` (zero on padding slots)."""

    @classmethod
    def of(cls, forked: Forked, length: int, chunk: int) -> "AlignedDocuments":
        """`forked`'s documents (`Forked.firsts`, `S` past a row's last) of a call of `length` tokens on
        `chunk`."""
        firsts = jnp.asarray(forked.firsts, jnp.int32)
        rows, documents = firsts.shape
        ends = jnp.concatenate([firsts[:, 1:], jnp.full_like(firsts[:, :1], length)], axis=1)
        sizes = jnp.clip(jnp.minimum(ends, length) - jnp.minimum(firsts, length), 0)
        held = sizes > 0
        # Each document's aligned first slot: the chunks every document before it fills, whole.
        chunks = -(-sizes // chunk)
        chunk_ends = jnp.cumsum(chunks, axis=1)
        chunk_starts = chunk_ends - chunks
        opened = chunk_starts * chunk
        total = -(-length // chunk) + documents
        positions = jnp.arange(length, dtype=jnp.int32)
        # Each slot's document: the last whose first slot is at or before it, its row's firsts sorted.
        found = jax.vmap(lambda row: jnp.searchsorted(row, positions, side="right"))(firsts)
        tok_doc = jnp.clip(found - 1, 0, documents - 1)
        tok_within = positions[None, :] - jnp.take_along_axis(firsts, tok_doc, axis=1)
        aligned = (jnp.take_along_axis(opened, tok_doc, axis=1) + tok_within).astype(jnp.int32)
        lines = jnp.arange(rows)[:, None]
        first = jnp.where(held, chunk_starts, total)
        last_at = jnp.where(held, chunk_ends - 1, total)
        loads = jnp.full((rows, total), -1, jnp.int32).at[lines, first].set(forked.loads, mode="drop")
        stores = jnp.zeros((rows, total), jnp.int32).at[lines, last_at].set(forked.stores, mode="drop")
        chunk_idx = jnp.arange(total, dtype=jnp.int32)
        chunk_doc = jax.vmap(lambda row: jnp.searchsorted(row, chunk_idx, side="right"))(chunk_ends)
        alive = chunk_doc < documents
        chunk_doc = jnp.minimum(chunk_doc, documents - 1)
        last = jnp.zeros((rows, total), jnp.bool_).at[lines, last_at].set(True, mode="drop")
        final = last & (chunk_doc == tok_doc[:, -1:])
        within = jax.vmap(lambda row, at, vals: row.at[at].set(vals))(
            jnp.zeros((rows, total * chunk), jnp.int32), aligned, tok_within)
        return cls(aligned, loads, stores, total * chunk, chunk_doc, alive, last, final, within)

    def laid(self, *operands):
        """The operands `[B, S, ...]` on the aligned grid `[B, length, ...]`, zero between documents."""
        def moved(x):
            out = jnp.zeros((x.shape[0], self.length, *x.shape[2:]), x.dtype)
            return jax.vmap(lambda row, at, values: row.at[at].set(values))(out, self.slots, x)
        return tuple(moved(x) for x in operands)

    def unlaid(self, out):
        """`out` `[B, length, ...]` back at the call's slots `[B, S, ...]`."""
        return jax.vmap(lambda row, at: row[at])(out, self.slots)


@dataclasses.dataclass(frozen=True)
class ForkedChunks:
    """A forked call's documents on the chunks a packed call's are laid on (`sequences` over its documents'
    spans, fla's `chunk_indices`: each document from a chunk's edge, `ceil(B * S / C) + N` chunks), each row
    walking its documents' chunks in turn: the table slot each chunk takes its memory up from (`loads` `[NC]`,
    `Chunk.loads`' one-based, zero a zero memory; -1 but at a document's first chunk), the stored slot it
    leaves its memory in (`stores` `[NC]`, zero but at a storing document's last chunk), the document a first
    chunk begins (`taken` `[NC]`, `row * D + d`; -1 elsewhere), which its entry write reads, and the row each
    chunk walks (`rows` `[NC]`; the last row's past every row's chunks)."""

    loads: jax.Array
    stores: jax.Array
    taken: jax.Array
    rows: jax.Array

    @staticmethod
    def spans(forked: Forked, batch: int, length: int, chunk: int) -> "Sequences":
        """`forked`'s documents' spans, as a packed call of its documents' first tokens and `firsts` lays
        them at a documents bound of its own (`Chunk.loads`' `D`; `packed_sequences`)."""
        return packed_sequences(forked.tokens(length)[0] >= 0, forked.firsts, forked.loads.shape[1],
                                chunk=chunk)

    @classmethod
    def of(cls, forked: Forked, batch: int, length: int, chunk: int) -> "ForkedChunks":
        """`forked`'s documents on the chunks of `batch` rows of `length` tokens (`spans`)."""
        spans = cls.spans(forked, batch, length, chunk)
        span = unpacked(Span, lambda i: jnp.asarray(spans.spans)[:, i])
        count = spans.chunks.shape[0]
        live = span.tokens > 0
        sizes = -(-span.tokens // chunk)
        firsts = jnp.cumsum(sizes) - sizes
        # Each sequence's document: the sequences are each row's documents in turn (`cu_seqlens`' order).
        number = jnp.arange(span.first.shape[0], dtype=jnp.int32)
        opening = jax.ops.segment_min(jnp.where(live, number, number.shape[0]), span.memory,
                                      num_segments=batch)
        documents = forked.loads.shape[1]
        document = jnp.clip(number - opening[jnp.clip(span.memory, 0, batch - 1)], 0, documents - 1)
        row = jnp.clip(span.memory, 0, batch - 1)
        drop = jnp.where(live, 0, count)
        loads = jnp.full((count,), -1, jnp.int32).at[firsts + drop].set(
            jnp.asarray(forked.loads, jnp.int32)[row, document], mode="drop")
        stores = jnp.zeros((count,), jnp.int32).at[firsts + sizes - 1 + drop].set(
            jnp.asarray(forked.stores, jnp.int32)[row, document], mode="drop")
        taken = jnp.full((count,), -1, jnp.int32).at[firsts + drop].set(row * documents + document,
                                                                         mode="drop")
        # Each chunk's row: its sequence's, a sequence's chunks following its first.
        owner = jnp.clip(jnp.searchsorted(firsts + sizes, jnp.arange(count), side="right"), 0,
                         row.shape[0] - 1)
        return cls(loads, stores, taken, row[owner])

    def turning(self, forked: Forked):
        """Each chunk's load rotation (`Forked.turning`), the document it takes up's, a unit turn for none:
        `(cos, sin)` `[NC, Hv, A]`; None where the call gives none."""
        if forked.turning is None:
            return None
        live = (self.taken >= 0)[:, None, None]
        taken = jnp.maximum(self.taken, 0)
        cos, sin = (part.reshape(-1, *part.shape[2:])[taken] for part in forked.turning)
        return jnp.where(live, cos, 1.0), jnp.where(live, sin, 0.0)
    def added(self, forked: Forked) -> jax.Array | None:
        """Each chunk's added slot (`recurrence_chunk.Chunk.adds`), the document it takes up's, zero for none:
        `[NC]`; None where the call adds none."""
        if forked.adds is None:
            return None
        return jnp.where(self.taken >= 0, jnp.asarray(forked.adds, jnp.int32).reshape(-1)[
            jnp.maximum(self.taken, 0)], 0)

    def entries(self, forked: Forked):
        """Each chunk's entry write (`recurrence_chunk.Chunk.entry_key`), the document it takes up's, zero for
        none: `[NC, Hv, Dk, Dv]` summed over its terms and weighed; None where the call gives none. A chunk's
        own, formed once a chunk."""
        if forked.entry is None:
            return None
        # Each row's documents flat, by their count: a write of no terms leaves `-1` nothing to infer.
        key, value, weight = _flat(*forked.entry)
        taken = jnp.maximum(self.taken, 0)
        key, value, weight = (x[taken] for x in (key, value, weight))
        group = value.shape[-2] // key.shape[-2]
        if group > 1:
            key = jnp.repeat(key, group, axis=-2)
        # The weight inside the contraction, so its gradient is one too (`dot_general`'s accumulation), not
        # an elementwise product summed after: the SSD port's weight gradient took 1.6 to 1.9 times the
        # error each document alone takes from float64 summed after (armada, 11-11).
        weighed = jnp.where((self.taken >= 0)[:, None], weight, 0)
        return entry_write(jnp.moveaxis(key, 1, 2), jnp.moveaxis(value, 1, 2), weighed[..., None, None])
