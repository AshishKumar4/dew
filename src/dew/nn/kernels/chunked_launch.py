# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Translated from flash-linear-attention v0.5.2 (fla/ops/utils/index.py, fla/ops/common), which is
# licensed under the MIT license: see licenses/flash-linear-attention-LICENSE.

"""What the chunked recurrences' launches share: the token spans of a call's sequences, packed
documents each its own (`Sequences`, fla's `cu_seqlens`, `chunk_indices` and `chunk_offsets`), a
chunk's place and its masked loads and stores, and the tiles a launch takes (`Tiles`, `_Launch`). The
recurrence template's stages (`dew.nn.kernels.recurrence`) and the ported kernels (`fla_gdn`,
`fla_kda`) read them alike.
"""

import dataclasses
from collections.abc import Mapping
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from dew.nn.kernels import generation, tuning


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
        # A launch reads chunk descriptors `[NT, 4]`, sequence ones `[N, 5]` or both: a packed call's
        # NT is the tokens' chunks and one more for each of its N sequences (`sequences`).
        chunks, count = rows.get(4), rows.get(5)
        if chunks in (None, batch * -(-length // 64)) and count in (None, batch):
            drawn = sequences(batch, length)
        else:
            count = chunks - -(-batch * length // 64) if count is None else count
            drawn = sequences(batch, length, _even(batch, length, count))
        return [jnp.asarray(drawn.chunks if spec.shape[-1] == 4 else drawn.spans)
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
    the targets do not list takes the 99 KB picks, which every Triton GPU's block holds, and a CPU's
    interpretation an A100's."""
    target = generation.effective(generation.gpu_target())
    return (target.shared_limit or 0) >= (generation.TARGETS["sm80"].shared_limit or 0)


class Sequences(NamedTuple):
    """The token spans fla's varlen kernels read (its `cu_seqlens`, `chunk_indices` and `chunk_offsets`,
    fla/ops/utils/index.py) over the `[B * T]` tokens of a call's rows. `chunks` `[NT, 4]` is each
    chunk's first token and count, in order, a chunk never crossing a sequence and those past the
    last empty, then the slots of the stored memories (`slots`) it enters and leaves; `spans` `[N, 5]`
    is each sequence's first token and count, the slot its second chunk enters less one, and the
    slots its first chunk enters and its last leaves. Sequences never cross rows, so a row's first
    sequence takes its initial memory and its last gives its final one.

    The memories are stored by where they pass, not by chunk: one slot for each that passes between
    two chunks of a sequence, then a slot a row for its initial memory, which its first chunk enters,
    and its final one's gradient, which its last leaves; and, where documents are packed, one of
    zeros, which every other document's first chunk enters and last leaves. However many documents a
    row holds, the slots are no more than the tokens' chunks, a slot a row and one (`slots`); an
    unpacked call's are its chunks, as many as main stored."""
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


def sequences(batch: int, length: int, cu_seqlens=None, *, chunk: int = 64) -> Sequences:
    """The spans of `cu_seqlens` (`[N + 1]` int32 over the `batch * length` tokens, each row's
    boundaries among them), or of the rows themselves where None. A traced `cu_seqlens` takes the most
    chunks its N sequences can split `batch * length` tokens into."""
    # The rows' own spans are constants; a traced `cu_seqlens` takes the same arithmetic in XLA.
    xp = np if cu_seqlens is None else jnp
    packed = cu_seqlens is not None
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
    # A row's own slots follow the passing ones, and the zeros theirs.
    passing = _passing(batch, length, chunk, packed)
    zeros = passing + batch
    entered = xp.where(first % length == 0, passing + first // length, zeros)
    left = xp.where((first + lengths) % length == 0, passing + (first + lengths - 1) // length, zeros)
    index = xp.arange(count, dtype=np.int32)
    owner = xp.clip(xp.searchsorted(offsets, index, side="right") - 1, 0, lengths.shape[0] - 1)
    within = index - offsets[owner]
    real = index < offsets[-1]
    start = first[owner] + within * chunk
    taken = xp.where(real, xp.minimum(chunk, first[owner] + lengths[owner] - start), 0)
    entering = xp.where(within > 0, base[owner] + within, entered[owner])
    leaving = xp.where(within + 1 < sizes[owner], base[owner] + within + 1, left[owner])
    return Sequences(xp.stack([start, taken, entering, leaving], axis=1).astype(np.int32),
                     xp.stack([first, lengths, base, entered, left], axis=1).astype(np.int32))


def _aligned(spans: Sequences, batch: int, length: int, chunk: int) -> bool:
    """Whether every chunk starts on a multiple of `chunk`: the rows' own spans over whole chunks."""
    return (length % chunk == 0 and spans.spans.shape[0] == batch
            and spans.chunks.shape[0] == batch * length // chunk)


def _chunk_of(chunk_ref, chunk: int, aligned: bool):
    """A chunk's first token and count (`Sequences.chunks`). Where `aligned` every chunk is whole and
    starts on a multiple of `chunk`, as fla's fixed-length chunks do: the count is the constant
    `chunk`, so its loads and stores take no mask (`read_rows`), and Triton is told the alignment. A
    packed document's chunks start anywhere and end where it does, as fla's varlen ones."""
    start, taken = chunk_ref[0], chunk_ref[1]
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


def _span(span_ref, length: int, chunk: int, aligned: bool):
    """A sequence's first token, count, base slot and the slots its first chunk enters and its last
    leaves (`Sequences.spans`), its row in a call of rows `length` long, and whether it starts its row
    (taking the row's initial memory) and ends it (giving the row's final memory). `aligned` as
    `_chunk_of`."""
    start, count, base, entered, left = (span_ref[i] for i in range(5))
    if aligned:
        # The rows' own spans: each `length` long, whole chunks from a multiple of `chunk`.
        start, count = pl.multiple_of(start, chunk), length
    return (start, count, base, entered, left, start // length, start % length == 0,
            (start + count) % length == 0)
