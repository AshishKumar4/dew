# Copyright (c) 2024, Tri Dao, Albert Gu.
#
# Translated from mamba_ssm 2.3.2.post1 (mamba_ssm/ops/triton/ssd_bmm.py, ssd_chunk_state.py,
# ssd_state_passing.py, ssd_chunk_scan.py, ssd_combined.py), which is licensed under the Apache License
# 2.0: see licenses/mamba-LICENSE.

"""mamba_ssm's chunked SSD scan, ported kernel for kernel to Pallas reading Dew's [B, T, H, P].

The port follows state-spaces/mamba's Triton SSD (`mamba_chunk_scan_combined`) without redesign: the same
four forward kernels (each chunk's own state, the states passed across chunks, each chunk's `C B^T`,
then the output), the same eight backward ones, the same blocks over each kernel's grid, and the numerics
mamba_ssm runs: operands in their own dtype into fp32 accumulation, the states the chunks enter stored in
the operands' dtype, the decay between two steps of a chunk `exp(min(acs_i - acs_j, 0))` of the log decay
`acs` cumulated within the chunk, and each gradient of the decay summed from its products as mamba_ssm's
stable kernels sum it. mamba_ssm's deterministic mode is the only one: a sum over a kernel's blocks lands
in a workspace that XLA sums, where mamba_ssm's default adds atomically. Each kernel names its origin.

What mamba_ssm folds in and Dew computes outside stays outside: the step's bias, softplus and limit
(`dew.nn.mixers.mamba2._ssd`), the output gate `z` (Dew's gated norm), and the cumulative sum of `A dt`
within each chunk, an XLA cumsum (`_per_chunk`). Packed documents are mamba_ssm's `seq_idx`, each token's
document counted from its row's `starts`. Products of fp32 operands take TF32x3 (`fla_gdn._dot`), where
mamba_ssm's take one TF32.

Operands stay in Dew's layout: `x` `[B, T, H, P]`, `B` and `C` `[B, T, G, N]`, the step and its
cumulated decay `[B, T, H]` padded to whole chunks, the states the chunks enter `[B, T // chunk, H, P,
N]`. A program reads its rows of its head where they sit, as mamba_ssm's pointer arithmetic does.

Each kernel is a launch the tuner times (`dew.nn.kernels.tuning.KernelOp`) over the configs mamba_ssm's
own autotuner lists for it (`_space`), mamba_ssm's pick at Nemotron 3's shapes on an H100 its heuristic
(`HEURISTICS`).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels import tuning
from dew.nn.kernels.fla_gdn import _dot, _read, _write


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's config: mamba_ssm's BLOCK_SIZE_M, _N and _K in each kernel's own reading (0 where it
    takes none), its warps and pipeline stages, and for the gradients a program sums over a group's
    heads the heads it takes (mamba_ssm's `nheads_per_program`, 0 elsewhere)."""

    bm: int
    bn: int
    bk: int
    warps: int
    stages: int
    heads: int = 0


@dataclasses.dataclass(frozen=True)
class _Launch(tuning.KernelOp):
    """One of the port's twelve launches, its `Tiles` measured per GPU model and shape over its
    kernel's space (`_space`) by the tuner, mamba_ssm's own pick its heuristic (`HEURISTICS`). A
    subclass names its kernel and what each block spans (`SPANS`)."""

    config_cls: ClassVar[type[Tiles]] = Tiles
    NAME: ClassVar[str]
    SPANS: ClassVar[tuple[tuple[str, int] | str | None, ...]] = (None, None, None)
    """What each of `bm`, `bn` and `bk` spans, clipping it: an argument's axis, 'chunk' the chunk
    length, 'state' the flattened state `P * N` of an argument `[..., P, N]` named 'states'."""

    def _size(self, ba, span) -> int:
        if span == "chunk":
            return ba.arguments["chunk"]
        if span == "state":
            return ba.arguments["states"].shape[-1]
        name, axis = span
        return ba.arguments[name].shape[axis]

    def _fitted(self, ba, tiles: Tiles) -> Tiles:
        """`tiles` with each block clipped to what it spans and `heads` to a group's."""
        blocks = [block if span is None else min(block, self._size(ba, span))
                  for block, span in zip((tiles.bm, tiles.bn, tiles.bk), self.SPANS, strict=True)]
        heads = tiles.heads and min(tiles.heads, _operand(ba).shape[2] // ba.arguments["groups"])
        return dataclasses.replace(tiles, bm=blocks[0], bn=blocks[1], bk=blocks[2], heads=heads)

    def _get_heuristics_config(self, ba) -> Tiles:
        dtype = "float32" if jnp.dtype(_operand(ba).dtype).itemsize == 4 else "bfloat16"
        return self._fitted(ba, HEURISTICS[dtype][self.NAME])

    def _get_autotuning_configs(self, ba) -> set[Tiles]:
        return {self._fitted(ba, tiles) for tiles in _space(self.NAME)}


def _operand(ba):
    """A call's first `[B, T, H, P]`-like operand: its dtype picks the heuristic (fp16 taking bf16's),
    its third axis counts the heads."""
    return next(value for value in ba.arguments.values()
                if isinstance(value, jax.Array | jax.ShapeDtypeStruct)
                and jnp.issubdtype(value.dtype, jnp.floating) and value.ndim == 4)


def _tiles(blocks, warps: int, stages: int, heads=(0,)) -> frozenset[Tiles]:
    return frozenset(Tiles(*block, warps, stages, h) for block in blocks for h in heads)


_MATMUL = frozenset({Tiles(128, 256, 64, 8, 3), Tiles(64, 256, 32, 4, 4), Tiles(128, 128, 32, 4, 4),
                     Tiles(128, 64, 32, 4, 4), Tiles(64, 128, 32, 4, 4), Tiles(128, 32, 32, 4, 4),
                     Tiles(64, 32, 32, 2, 5), Tiles(32, 64, 32, 2, 5), Tiles(64, 64, 32, 2, 4)})
"""mamba_ssm's nine-config matmul list (ssd_chunk_state.py:177-188, ssd_bmm.py:22-33 and 96-107,
ssd_chunk_scan.py:434-445); the chunk scan's adds two (ssd_chunk_scan.py:32-45) and the combined
`dx`'s runs its two-warp configs at four (ssd_combined.py:78-89)."""

_HEADED = (2, 4, 8, 16, 32, 64)
"""The heads of a group one program of a summing kernel takes (mamba_ssm's `nheads_per_program`,
which it derives from the SM count), clipped to the group's (`_Launch._fitted`)."""

_SUMMING = [(32, 128), (128, 32), (64, 128), (128, 64), (64, 64), (64, 32), (32, 64), (32, 32)]


def _space(kernel: str) -> frozenset[Tiles]:
    """`kernel`'s configs to measure: mamba_ssm's own autotune list for it, `heads` over `_HEADED` for
    the kernels that sum a group's heads, every config Triton's default four warps and three stages
    where mamba_ssm's names none."""
    return {
        "chunk_state": _MATMUL,
        "state_passing": _tiles([(size, 0, 0) for size in (64, 128, 256, 512, 1024, 2048)], 4, 3),
        "bmm_chunk": _MATMUL,
        "chunk_scan": _MATMUL | {Tiles(128, 64, 64, 4, 4), Tiles(64, 128, 64, 4, 4)},
        "chunk_scan_bwd_dstates": _MATMUL,
        "state_passing_bwd": _tiles([(size, 0, 0) for size in (64, 128, 256, 512, 1024, 2048)], 4, 3),
        "chunk_scan_chunk_state_bwd_dx": frozenset(dataclasses.replace(tiles, warps=max(tiles.warps, 4))
                                                   for tiles in _MATMUL),
        "chunk_state_bwd_db": _tiles([(m, n, 0) for m, n in _SUMMING], 4, 3, _HEADED),
        "chunk_scan_bwd_dc": _tiles([(m, n, 0) for m, n in _SUMMING], 4, 3, _HEADED),
        "chunk_scan_bwd_dcb": _tiles([(32, 128, 0), (128, 32, 0), (64, 64, 0), (64, 32, 0), (32, 64, 0),
                                      (32, 32, 0)], 4, 3, _HEADED),
        "bmm_chunk_bwd": _MATMUL,
        "chunk_scan_bwd_ddacs_stable": _tiles([(32, 32, 0), (64, 32, 0), (64, 64, 0), (128, 32, 0),
                                               (128, 64, 0), (128, 128, 0)], 4, 3),
    }[kernel]


HEURISTICS = {
    "bfloat16": {
        "chunk_state": Tiles(64, 64, 32, 2, 4),
        "state_passing": Tiles(256, 0, 0, 4, 3),
        "bmm_chunk": Tiles(128, 64, 32, 4, 4),
        "chunk_scan": Tiles(32, 64, 32, 2, 5),
        "chunk_scan_bwd_dstates": Tiles(64, 128, 32, 4, 4),
        "state_passing_bwd": Tiles(512, 0, 0, 4, 3),
        "chunk_scan_chunk_state_bwd_dx": Tiles(32, 64, 32, 4, 5),
        "chunk_state_bwd_db": Tiles(64, 128, 0, 4, 3, 64),
        "chunk_scan_bwd_dc": Tiles(64, 128, 0, 4, 3, 64),
        "chunk_scan_bwd_dcb": Tiles(64, 64, 0, 4, 3, 64),
        "bmm_chunk_bwd": Tiles(32, 64, 32, 2, 5),
        "chunk_scan_bwd_ddacs_stable": Tiles(64, 32, 0, 4, 3),
    },
}
HEURISTICS["float32"] = HEURISTICS["bfloat16"]
"""Each launch's config without a measurement, by operand dtype: mamba_ssm's own autotuner's pick on
an H100 at Nemotron 3 Ultra's shapes (one row of 4096 tokens, 256 heads of 64 over 8 groups, a state of
128, chunks of 128), each summing kernel taking a group's every head as mamba_ssm's SM-count rule
gives there."""


def _limit(c, length: int, chunk: int):
    """The tokens chunk `c` of a row holds (mamba_ssm's `chunk_size_limit`): the constant `chunk` where
    whole chunks tile the rows, so that no load or store takes a mask."""
    return chunk if length % chunk == 0 else jnp.minimum(chunk, length - c * chunk)


def _from(limit, offset):
    """The rows a block from `offset` of a chunk of `limit` tokens holds: all of any block where `limit`
    is the constant whole chunk."""
    return limit if isinstance(limit, int) else limit - offset


def _earlier(seq_ref, start, c):
    """The document the state a chunk enters belongs to: the previous chunk's last token's, the
    initial state's 0 for the first (mamba_ssm's `seq_idx_prev`)."""
    return jnp.where(c > 0, seq_ref[jnp.maximum(start - 1, 0)], 0)


def _causal(row0, rows: int, col0, cols: int, *, strict: bool = False):
    """`[rows, cols]` mask of the pairs whose row (from `row0`) is at or past (`strict`: past) its
    column (from `col0`)."""
    row = row0 + jnp.arange(rows)[:, None]
    col = col0 + jnp.arange(cols)[None, :]
    return row > col if strict else row >= col


def _compiled(warps: int, stages: int):
    from jax.experimental.pallas import triton as plgpu

    return plgpu.CompilerParams(num_warps=warps, num_stages=stages)


def _token(batch: int, padded: int, *, heads: int | None = None):
    """A per-token `[B, Tp, H]` array's block for a grid whose second axis counts `(chunk, row)`: its
    row's `[Tp]` of the grid's head (third axis), or all `heads` `[Tp, H]`."""
    if heads is None:
        return pl.BlockSpec((None, padded, None), lambda i, bc, h: (bc % batch, 0, h))
    return pl.BlockSpec((None, padded, heads), lambda i, bc, s: (bc % batch, 0, 0))


def _documents(batch: int, padded: int):
    return pl.BlockSpec((None, padded), lambda i, bc, h: (bc % batch, 0))


def _chunk_state_kernel(x_ref, b_ref, dt_ref, da_ref, *refs, batch: int, length: int, chunk: int,
                        bk: int, exact: bool):
    """mamba_ssm's `_chunk_state_fwd_kernel` (ssd_chunk_state.py:191-268) for one chunk of one head and
    a `[BM, BN]` block of its state: the chunk's own write `sum_k x_k (B_k exp(min(acs_last - acs_k,
    0)) dt_k)^T`, `B_k` scaled in fp32 and rounded to `x`'s dtype before its product, a token of an
    earlier document than the chunk's last excluded."""
    *seq, states_ref = refs
    c = pl.program_id(1) // batch
    start = c * chunk
    limit = _limit(c, length, chunk)
    last = da_ref[start + chunk - 1]

    def step(k, acc):
        offset = k * bk
        rows = pl.ds(start + offset, bk)
        x = _read(x_ref.at[rows], _from(limit, offset))
        b = _read(b_ref.at[rows], _from(limit, offset)).astype(jnp.float32)
        scale = jnp.exp(jnp.minimum(last - da_ref[rows], 0.0)) * dt_ref[rows]
        if seq:
            scale = jnp.where(seq[0][rows] == seq[0][start + chunk - 1], scale, 0.0)
        return acc + _dot(x.T, (b * scale[:, None]).astype(x.dtype), exact=exact)

    states_ref[...] = jax.lax.fori_loop(0, chunk // bk, step, jnp.zeros(states_ref.shape, jnp.float32))


@dataclasses.dataclass(frozen=True)
class ChunkState(_Launch):
    """`chunk_state`'s launch (tuning.KernelOp)."""

    NAME = "chunk_state"
    SPANS = (("x", 3), ("b", 3), "chunk")

    def _fwd(self, x, b, dt, da, seq, *, chunk: int, groups: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, heads, width = x.shape
        state = b.shape[-1]
        padded = dt.shape[1]
        ratio = heads // groups
        bm, bn = config.bm, config.bn
        columns = state // bn
        states = pl.pallas_call(
            functools.partial(_chunk_state_kernel, batch=batch, length=length, chunk=chunk, bk=config.bk,
                              exact=interpret),
            grid=(width // bm * columns, batch * (padded // chunk), heads),
            in_specs=[pl.BlockSpec((None, length, None, bm),
                                   lambda mn, bc, h: (bc % batch, 0, h, mn // columns)),
                      pl.BlockSpec((None, length, None, bn),
                                   lambda mn, bc, h: (bc % batch, 0, h // ratio, mn % columns)),
                      _token(batch, padded), _token(batch, padded),
                      *([] if seq is None else [_documents(batch, padded)])],
            out_specs=pl.BlockSpec((None, None, None, bm, bn),
                                   lambda mn, bc, h: (bc % batch, bc // batch, h, mn // columns,
                                                      mn % columns)),
            out_shape=jax.ShapeDtypeStruct((batch, padded // chunk, heads, width, state), jnp.float32),
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_chunk_state_fwd",
        )(x, b, dt, da, *([] if seq is None else [seq]))
        return states, None


def _state_passing_kernel(states_ref, da_ref, h0_ref, *refs, chunk: int):
    """mamba_ssm's `_state_passing_fwd_kernel` (ssd_state_passing.py:29-87) for one row's head and a
    block of its flattened `[P * N]` state: chunk by chunk, store the state the chunk enters in the
    output's dtype, then decay it by the chunk's total and add the chunk's own write, from zero where
    the chunk ends in another document than its predecessor did. What leaves the last chunk is the
    final state, fp32."""
    *seq, out_ref, final_ref = refs

    def step(c, carry):
        state, document = carry
        out_ref[c] = state.astype(out_ref.dtype)
        end = c * chunk + chunk - 1
        scale = jnp.exp(da_ref[end])
        if seq:
            current = seq[0][end]
            scale = jnp.where(current == document, scale, 0.0)
            document = current
        return scale * state + states_ref[c], document

    final, _ = jax.lax.fori_loop(0, out_ref.shape[0], step, (h0_ref[...], jnp.int32(0)))
    final_ref[...] = final


@dataclasses.dataclass(frozen=True)
class StatePassing(_Launch):
    """`state_passing`'s launch (tuning.KernelOp)."""

    NAME = "state_passing"
    SPANS = ("state", None, None)

    def _fwd(self, states, da, h0, seq, *, chunk: int, dtype: str, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, chunks, heads, flat = states.shape
        padded = da.shape[1]
        size = config.bm
        rows = pl.BlockSpec((None, chunks, None, size), lambda i, b, h: (b, 0, h, i))
        final = pl.BlockSpec((None, None, size), lambda i, b, h: (b, h, i))
        out, last = pl.pallas_call(
            functools.partial(_state_passing_kernel, chunk=chunk),
            grid=(flat // size, batch, heads),
            in_specs=[rows, pl.BlockSpec((None, padded, None), lambda i, b, h: (b, 0, h)), final,
                      *([] if seq is None else [pl.BlockSpec((None, padded), lambda i, b, h: (b, 0))])],
            out_specs=[rows, final],
            out_shape=[jax.ShapeDtypeStruct(states.shape, jnp.dtype(dtype)),
                       jax.ShapeDtypeStruct((batch, heads, flat), jnp.float32)],
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_state_passing_fwd",
        )(states, da, h0, *([] if seq is None else [seq]))
        return (out, last), None


def _bmm_kernel(a_ref, b_ref, *refs, length: int, chunk: int, groups: int, bm: int, bn: int, bk: int,
                exact: bool):
    """mamba_ssm's `_bmm_chunk_fwd_kernel` (ssd_bmm.py:36-93) for one chunk of one group and a `[BM, BN]`
    block of its `C B^T`, fp32, zero between tokens of two documents."""
    *seq, out_ref = refs
    columns = chunk // bn
    m, n = pl.program_id(0) // columns, pl.program_id(0) % columns
    c = pl.program_id(2) // groups
    start = c * chunk
    limit = _limit(c, length, chunk)
    rows_m, rows_n = pl.ds(start + m * bm, bm), pl.ds(start + n * bn, bn)

    def step(k, acc):
        width = pl.ds(k * bk, bk)
        a = _read(a_ref.at[rows_m, width], _from(limit, m * bm))
        b = _read(b_ref.at[rows_n, width], _from(limit, n * bn))
        return acc + _dot(a, b.T, exact=exact)

    acc = jax.lax.fori_loop(0, a_ref.shape[-1] // bk, step, jnp.zeros((bm, bn), jnp.float32))
    if seq:
        acc = jnp.where(seq[0][rows_m][:, None] == seq[0][rows_n][None, :], acc, 0.0)
    out_ref[...] = acc


@dataclasses.dataclass(frozen=True)
class BmmChunk(_Launch):
    """`bmm_chunk`'s launch (tuning.KernelOp)."""

    NAME = "bmm_chunk"
    SPANS = ("chunk", "chunk", ("a", 3))

    def _fwd(self, a, b, seq, *, chunk: int, padded: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, groups, state = a.shape
        chunks = padded // chunk
        bm, bn = config.bm, config.bn
        columns = chunk // bn
        grouped = pl.BlockSpec((None, length, None, state), lambda mn, row, cg: (row, 0, cg % groups, 0))
        cb = pl.pallas_call(
            functools.partial(_bmm_kernel, length=length, chunk=chunk, groups=groups, bm=bm, bn=bn,
                              bk=config.bk, exact=interpret),
            grid=(chunk // bm * columns, batch, chunks * groups),
            in_specs=[grouped, grouped,
                      *([] if seq is None else [pl.BlockSpec((None, padded), lambda mn, row, cg: (row, 0))])],
            out_specs=pl.BlockSpec((None, None, None, bm, bn),
                                   lambda mn, row, cg: (row, cg // groups, cg % groups, mn // columns,
                                                        mn % columns)),
            out_shape=jax.ShapeDtypeStruct((batch, chunks, groups, chunk, chunk), jnp.float32),
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_bmm_chunk_fwd",
        )(a, b, *([] if seq is None else [seq]))
        return cb, None


def _chunk_scan_kernel(cb_ref, x_ref, dt_ref, da_ref, c_ref, states_ref, d_ref, *refs, batch: int,
                       length: int, chunk: int, width: int, bm: int, bn: int, bk: int, exact: bool):
    """mamba_ssm's `_chunk_scan_fwd_kernel` (ssd_chunk_scan.py:48-180) without `z`, for one chunk of one
    head and a `[BM, BN]` block of its output: the state the chunk entered read through `C`, rounded
    to `C`'s dtype and decayed to each row by `exp(acs)` (zero for a row of a later document); then the
    causal `sum_k CB_mk exp(min(acs_m - acs_k, 0)) dt_k x_k` over the chunk's rows, each `CB` block
    weighted in fp32 and rounded to `x`'s dtype before its product; then `D x`."""
    *seq, out_ref = refs
    columns = width // bn
    m, h = pl.program_id(0) // columns, pl.program_id(2)
    c = pl.program_id(1) // batch
    start = c * chunk
    limit = _limit(c, length, chunk)
    rows = pl.ds(start + m * bm, bm)
    da_m = da_ref[rows]
    scale = jnp.exp(da_m)
    if seq:
        scale = jnp.where(seq[0][rows] == _earlier(seq[0], start, c), scale, 0.0)
    state = c_ref.shape[-1]
    # One product over the whole state up to 128 wide, mamba_ssm's BLOCK_SIZE_DSTATE; `bk` at a time past it.
    span = state if state <= 128 else bk
    acc = jnp.zeros((bm, bn), jnp.float32)
    for k in range(state // span):
        part = pl.ds(k * span, span)
        cm = _read(c_ref.at[rows, part], _from(limit, m * bm))
        acc = acc + _dot(cm, states_ref[:, part].astype(cm.dtype).T, exact=exact)
    acc = acc * scale[:, None]

    def step(k, acc):
        offset = k * bk
        steps = pl.ds(start + offset, bk)
        cb = cb_ref[pl.ds(m * bm, bm), pl.ds(offset, bk)]
        cb = cb * jnp.exp(jnp.minimum(da_m[:, None] - da_ref[steps][None, :], 0.0))
        cb = cb * dt_ref[steps][None, :]
        cb = jnp.where(_causal(m * bm, bm, offset, bk), cb, 0.0).astype(x_ref.dtype)
        return acc + _dot(cb, _read(x_ref.at[steps], _from(limit, offset)), exact=exact)

    acc = jax.lax.fori_loop(0, ((m + 1) * bm + bk - 1) // bk, step, acc)
    taken = _from(limit, m * bm)
    acc = acc + _read(x_ref.at[rows], taken).astype(jnp.float32) * d_ref[h]
    _write(out_ref.at[rows], taken, acc)


@dataclasses.dataclass(frozen=True)
class ChunkScan(_Launch):
    """`chunk_scan`'s launch (tuning.KernelOp)."""

    NAME = "chunk_scan"
    SPANS = ("chunk", ("x", 3), "chunk")

    def _fwd(self, cb, x, dt, da, c, states, d, seq, *, chunk: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, heads, width = x.shape
        groups, state = c.shape[2:]
        padded = dt.shape[1]
        ratio = heads // groups
        bm, bn = config.bm, config.bn
        columns = width // bn
        headed = pl.BlockSpec((None, length, None, bn), lambda mn, bc, h: (bc % batch, 0, h, mn % columns))
        out = pl.pallas_call(
            functools.partial(_chunk_scan_kernel, batch=batch, length=length, chunk=chunk, width=width,
                              bm=bm, bn=bn, bk=config.bk, exact=interpret),
            grid=(chunk // bm * columns, batch * (padded // chunk), heads),
            in_specs=[pl.BlockSpec((None, None, None, chunk, chunk),
                                   lambda mn, bc, h: (bc % batch, bc // batch, h // ratio, 0, 0)),
                      headed, _token(batch, padded), _token(batch, padded),
                      pl.BlockSpec((None, length, None, state),
                                   lambda mn, bc, h: (bc % batch, 0, h // ratio, 0)),
                      pl.BlockSpec((None, None, None, bn, state),
                                   lambda mn, bc, h: (bc % batch, bc // batch, h, mn % columns, 0)),
                      pl.BlockSpec((heads,), lambda mn, bc, h: (0,)),
                      *([] if seq is None else [_documents(batch, padded)])],
            out_specs=headed,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_chunk_scan_fwd",
        )(cb, x, dt, da, c, states, d, *([] if seq is None else [seq]))
        return out, None


def _dstates_kernel(dout_ref, c_ref, da_ref, *refs, batch: int, length: int, chunk: int, bk: int,
                    exact: bool):
    """mamba_ssm's `_chunk_scan_bwd_dstates_kernel` (ssd_chunk_scan.py:448-513) for one chunk of one head
    and a `[BM, BN]` block: the gradient of the state the chunk entered through its output, `sum_k
    (dout_k exp(acs_k))^T C_k`, each `dout_k` scaled in fp32 and rounded to its dtype, a token of a
    later document than the state's excluded."""
    *seq, out_ref = refs
    c = pl.program_id(1) // batch
    start = c * chunk
    limit = _limit(c, length, chunk)
    earlier = _earlier(seq[0], start, c) if seq else None

    def step(k, acc):
        offset = k * bk
        rows = pl.ds(start + offset, bk)
        dout = _read(dout_ref.at[rows], _from(limit, offset))
        scale = jnp.exp(da_ref[rows])
        if seq:
            scale = jnp.where(seq[0][rows] == earlier, scale, 0.0)
        dout = (dout.astype(jnp.float32) * scale[:, None]).astype(dout.dtype)
        cm = _read(c_ref.at[rows], _from(limit, offset)).astype(dout.dtype)
        return acc + _dot(dout.T, cm, exact=exact)

    out_ref[...] = jax.lax.fori_loop(0, chunk // bk, step, jnp.zeros(out_ref.shape, jnp.float32))


@dataclasses.dataclass(frozen=True)
class ChunkScanBwdDstates(_Launch):
    """`chunk_scan_bwd_dstates`'s launch (tuning.KernelOp)."""

    NAME = "chunk_scan_bwd_dstates"
    SPANS = (("dout", 3), ("c", 3), "chunk")

    def _fwd(self, dout, c, da, seq, *, chunk: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, heads, width = dout.shape
        groups, state = c.shape[2:]
        padded = da.shape[1]
        ratio = heads // groups
        bm, bn = config.bm, config.bn
        columns = state // bn
        dstates = pl.pallas_call(
            functools.partial(_dstates_kernel, batch=batch, length=length, chunk=chunk, bk=config.bk,
                              exact=interpret),
            grid=(width // bm * columns, batch * (padded // chunk), heads),
            in_specs=[pl.BlockSpec((None, length, None, bm),
                                   lambda mn, bc, h: (bc % batch, 0, h, mn // columns)),
                      pl.BlockSpec((None, length, None, bn),
                                   lambda mn, bc, h: (bc % batch, 0, h // ratio, mn % columns)),
                      _token(batch, padded), *([] if seq is None else [_documents(batch, padded)])],
            out_specs=pl.BlockSpec((None, None, None, bm, bn),
                                   lambda mn, bc, h: (bc % batch, bc // batch, h, mn // columns,
                                                      mn % columns)),
            out_shape=jax.ShapeDtypeStruct((batch, padded // chunk, heads, width, state), jnp.float32),
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_chunk_scan_bwd_dstates",
        )(dout, c, da, *([] if seq is None else [seq]))
        return dstates, None


def _state_passing_bwd_kernel(dout_ref, states_ref, da_ref, dfinal_ref, *refs, chunk: int):
    """mamba_ssm's `_state_passing_bwd_kernel` (ssd_state_passing.py:101-193) with an initial state and
    the states converted, for one row's head and a block of its flattened state: chunk by chunk from
    the last, store the gradient of the state the chunk leaves in the output's dtype and the state it
    entered in the same, the gradient of the chunk's total decay `exp(total) (entered . dleave)`, then
    `dleave <- exp(total) dleave + dentered`, with no decay where the chunk ends in another document
    than its predecessor did. What leaves the first chunk is the initial state's gradient."""
    *seq, dstates_ref, dtotal_ref, dinit_ref, converted_ref = refs
    chunks = dstates_ref.shape[0]
    document = seq[0][seq[0].shape[0] - 1] if seq else jnp.int32(0)

    def step(i, carry):
        dstate, document = carry
        c = chunks - 1 - i
        dstates_ref[c] = dstate.astype(dstates_ref.dtype)
        entered = states_ref[c]
        converted_ref[c] = entered.astype(converted_ref.dtype)
        scale = jnp.exp(da_ref[c * chunk + chunk - 1])
        if seq:
            earlier = _earlier(seq[0], c * chunk, c)
            scale = jnp.where(earlier == document, scale, 0.0)
            document = earlier
        dtotal_ref[c] = jnp.sum(entered * dstate) * scale
        return scale * dstate + dout_ref[c], document

    dinit, _ = jax.lax.fori_loop(0, chunks, step, (dfinal_ref[...], document))
    dinit_ref[...] = dinit


@dataclasses.dataclass(frozen=True)
class StatePassingBwd(_Launch):
    """`state_passing_bwd`'s launch (tuning.KernelOp)."""

    NAME = "state_passing_bwd"
    SPANS = ("state", None, None)

    def _fwd(self, dout, states, da, dfinal, seq, *, chunk: int, dtype: str, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, chunks, heads, flat = states.shape
        padded = da.shape[1]
        size = config.bm
        rows = pl.BlockSpec((None, chunks, None, size), lambda i, b, h: (b, 0, h, i))
        whole = pl.BlockSpec((None, None, size), lambda i, b, h: (b, h, i))
        dstates, dtotal, dinit, converted = pl.pallas_call(
            functools.partial(_state_passing_bwd_kernel, chunk=chunk),
            grid=(flat // size, batch, heads),
            in_specs=[rows, rows, pl.BlockSpec((None, padded, None), lambda i, b, h: (b, 0, h)), whole,
                      *([] if seq is None else [pl.BlockSpec((None, padded), lambda i, b, h: (b, 0))])],
            out_specs=[rows, pl.BlockSpec((None, None, chunks, None), lambda i, b, h: (b, h, 0, i)), whole,
                       rows],
            out_shape=[jax.ShapeDtypeStruct(states.shape, jnp.dtype(dtype)),
                       jax.ShapeDtypeStruct((batch, heads, chunks, flat // size), jnp.float32),
                       jax.ShapeDtypeStruct((batch, heads, flat), jnp.float32),
                       jax.ShapeDtypeStruct(states.shape, jnp.dtype(dtype))],
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_state_passing_bwd",
        )(dout, states, da, dfinal, *([] if seq is None else [seq]))
        return (dstates, jnp.sum(dtotal, axis=-1), dinit, converted), None


def _dx_kernel(x_ref, cb_ref, dout_ref, dt_ref, da_ref, d_ref, b_ref, dstates_ref, *refs, batch: int,
               length: int, chunk: int, width: int, bm: int, bn: int, bk: int, exact: bool):
    """mamba_ssm's `_chunk_scan_chunk_state_bwd_dx_kernel` (ssd_combined.py:92-254) for one chunk of one
    head and a `[BM, BN]` block of `x`: the gradient through the state the chunk leaves, `B_m dleave^T`
    decayed by `exp(min(acs_last - acs_m, 0))`, plus the causal `sum_k CB_km exp(min(acs_k - acs_m, 0))
    dout_k` over the chunk's later rows (`CB` weighted in fp32, rounded to `dout`'s dtype), times
    `dt_m`, plus `D dout_m`; `dt`'s gradient `sum_p (...) x_m` and `D`'s `sum dout x` per block for
    XLA to sum."""
    *seq, dx_ref, ddt_ref, dd_ref = refs
    columns = width // bn
    m, n, h = pl.program_id(0) // columns, pl.program_id(0) % columns, pl.program_id(2)
    c = pl.program_id(1) // batch
    start = c * chunk
    limit = _limit(c, length, chunk)
    rows = pl.ds(start + m * bm, bm)
    taken = _from(limit, m * bm)
    da_m = da_ref[rows]
    scale = jnp.exp(jnp.minimum(da_ref[start + chunk - 1] - da_m, 0.0))
    if seq:
        scale = jnp.where(seq[0][rows] == seq[0][start + chunk - 1], scale, 0.0)
    state = b_ref.shape[-1]
    span = state if state <= 128 else bk
    acc = jnp.zeros((bm, bn), jnp.float32)
    for k in range(state // span):
        part = pl.ds(k * span, span)
        bmat = _read(b_ref.at[rows, part], taken)
        acc = acc + _dot(bmat, dstates_ref[:, part].astype(bmat.dtype).T, exact=exact)
    acc = acc * scale[:, None]

    def step(k, acc):
        offset = m * bm + k * bk
        steps = pl.ds(start + offset, bk)
        cb = cb_ref[pl.ds(offset, bk), pl.ds(m * bm, bm)].T
        cb = cb * jnp.exp(jnp.minimum(da_ref[steps][None, :] - da_m[:, None], 0.0))
        cb = jnp.where(_causal(offset, bk, m * bm, bm).T, cb, 0.0).astype(dout_ref.dtype)
        return acc + _dot(cb, _read(dout_ref.at[steps], _from(limit, offset)), exact=exact)

    acc = jax.lax.fori_loop(0, (chunk - m * bm + bk - 1) // bk, step, acc)
    dx = acc * dt_ref[rows][:, None]
    dout = _read(dout_ref.at[rows], taken).astype(jnp.float32)
    _write(dx_ref.at[rows], taken, dx + dout * d_ref[h])
    x = _read(x_ref.at[rows], taken).astype(jnp.float32)
    dd_ref[m, n] = jnp.sum(dout * x)
    ddt_ref[rows] = jnp.sum(acc * x, axis=1)


@dataclasses.dataclass(frozen=True)
class ChunkScanChunkStateBwdDx(_Launch):
    """`chunk_scan_chunk_state_bwd_dx`'s launch (tuning.KernelOp)."""

    NAME = "chunk_scan_chunk_state_bwd_dx"
    SPANS = ("chunk", ("x", 3), "chunk")

    def _fwd(self, x, cb, dout, dt, da, d, b, dstates, seq, *, chunk: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, heads, width = x.shape
        groups, state = b.shape[2:]
        padded = dt.shape[1]
        chunks = padded // chunk
        ratio = heads // groups
        bm, bn = config.bm, config.bn
        columns = width // bn
        headed = pl.BlockSpec((None, length, None, bn), lambda mn, bc, h: (bc % batch, 0, h, mn % columns))
        dx, ddt, dd = pl.pallas_call(
            functools.partial(_dx_kernel, batch=batch, length=length, chunk=chunk, width=width, bm=bm, bn=bn,
                              bk=config.bk, exact=interpret),
            grid=(chunk // bm * columns, batch * chunks, heads),
            in_specs=[headed,
                      pl.BlockSpec((None, None, None, chunk, chunk),
                                   lambda mn, bc, h: (bc % batch, bc // batch, h // ratio, 0, 0)),
                      headed, _token(batch, padded), _token(batch, padded),
                      pl.BlockSpec((heads,), lambda mn, bc, h: (0,)),
                      pl.BlockSpec((None, length, None, state),
                                   lambda mn, bc, h: (bc % batch, 0, h // ratio, 0)),
                      pl.BlockSpec((None, None, None, bn, state),
                                   lambda mn, bc, h: (bc % batch, bc // batch, h, mn % columns, 0)),
                      *([] if seq is None else [_documents(batch, padded)])],
            out_specs=[headed,
                       pl.BlockSpec((None, padded, None, None),
                                    lambda mn, bc, h: (bc % batch, 0, h, mn % columns)),
                       pl.BlockSpec((None, None, None, chunk // bm, columns),
                                    lambda mn, bc, h: (bc % batch, bc // batch, h, 0, 0))],
            out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype),
                       jax.ShapeDtypeStruct((batch, padded, heads, columns), jnp.float32),
                       jax.ShapeDtypeStruct((batch, chunks, heads, chunk // bm, columns), jnp.float32)],
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_chunk_scan_chunk_state_bwd_dx",
        )(x, cb, dout, dt, da, d, b, dstates, *([] if seq is None else [seq]))
        return (dx, jnp.sum(ddt, axis=-1), jnp.sum(dd, axis=(0, 1, 3, 4))), None


def _heads_of(s, g, *, ratio: int, heads: int):
    """The first head and the count of heads program `(s, g)` of a summing kernel takes: split `s` of
    group `g`'s `ratio` heads, `heads` a split, the last split the rest."""
    count = heads if ratio % heads == 0 else jnp.minimum(heads, ratio - s * heads)
    return g * ratio + s * heads, count


def _db_kernel(x_ref, dstates_ref, b_ref, dt_ref, da_ref, *refs, batch: int, length: int, chunk: int,
               groups: int, ratio: int, heads: int, state: int, bm: int, bn: int, exact: bool):
    """mamba_ssm's `_chunk_state_bwd_db_kernel` (ssd_chunk_state.py:397-497) with `B`, for one chunk of
    one split of a group's heads and a `[BM, BN]` block of `B`'s gradient: over the split's heads, `x_m
    dleave` scaled by `exp(min(acs_last - acs_m, 0)) dt_m` (zero for a token of an earlier document
    than the chunk's last), summed; and per head the gradient of `acs_last - acs_m`, its row's sum with
    `B_m`, unshifted (XLA cumulates it, `_decay_gradient`)."""
    *seq, db_ref, dacs_ref = refs
    columns = state // bn
    m = pl.program_id(0) // columns
    c = pl.program_id(1) // batch
    split, group = pl.program_id(2) // groups, pl.program_id(2) % groups
    first, count = _heads_of(split, group, ratio=ratio, heads=heads)
    start = c * chunk
    limit = _limit(c, length, chunk)
    rows = pl.ds(start + m * bm, bm)
    taken = _from(limit, m * bm)
    bmat = _read(b_ref.at[rows], taken).astype(jnp.float32)
    same = seq[0][rows] == seq[0][start + chunk - 1] if seq else None

    def head(i, acc):
        h = first + i
        x = _read(x_ref.at[rows, h], taken)
        db = _dot(x, dstates_ref[h].astype(x.dtype), exact=exact)
        scale = jnp.exp(jnp.minimum(da_ref[start + chunk - 1, h] - da_ref[rows, h], 0.0))
        if seq:
            scale = jnp.where(same, scale, 0.0)
        db = db * (scale * dt_ref[rows, h])[:, None]
        dacs_ref[rows, h] = jnp.sum(db * bmat, axis=1)
        return acc + db

    _write(db_ref.at[rows], taken, jax.lax.fori_loop(0, count, head, jnp.zeros((bm, bn), jnp.float32)))


@dataclasses.dataclass(frozen=True)
class ChunkStateBwdDb(_Launch):
    """`chunk_state_bwd_db`'s launch (tuning.KernelOp)."""

    NAME = "chunk_state_bwd_db"
    SPANS = ("chunk", ("b", 3), None)

    def _fwd(self, x, dstates, b, dt, da, seq, *, chunk: int, groups: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, heads, width = x.shape
        state = b.shape[-1]
        padded = dt.shape[1]
        ratio = heads // groups
        bm, bn = config.bm, config.bn
        columns = state // bn
        splits = -(-ratio // config.heads)
        db, dacs = pl.pallas_call(
            functools.partial(_db_kernel, batch=batch, length=length, chunk=chunk, groups=groups, ratio=ratio,
                              heads=config.heads, state=state, bm=bm, bn=bn, exact=interpret),
            grid=(chunk // bm * columns, batch * (padded // chunk), splits * groups),
            in_specs=[pl.BlockSpec((None, length, heads, width), lambda mn, bc, sg: (bc % batch, 0, 0, 0)),
                      pl.BlockSpec((None, None, heads, width, bn),
                                   lambda mn, bc, sg: (bc % batch, bc // batch, 0, 0, mn % columns)),
                      pl.BlockSpec((None, length, None, bn),
                                   lambda mn, bc, sg: (bc % batch, 0, sg % groups, mn % columns)),
                      _token(batch, padded, heads=heads), _token(batch, padded, heads=heads),
                      *([] if seq is None else [_documents(batch, padded)])],
            out_specs=[pl.BlockSpec((None, length, None, None, bn),
                                    lambda mn, bc, sg: (bc % batch, 0, sg // groups, sg % groups,
                                                        mn % columns)),
                       pl.BlockSpec((None, padded, heads, None),
                                    lambda mn, bc, sg: (bc % batch, 0, 0, mn % columns))],
            out_shape=[jax.ShapeDtypeStruct((batch, length, splits, groups, state), jnp.float32),
                       jax.ShapeDtypeStruct((batch, padded, heads, columns), jnp.float32)],
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_chunk_state_bwd_db",
        )(x, dstates, b, dt, da, *([] if seq is None else [seq]))
        return (jnp.sum(db, axis=2), jnp.sum(dacs, axis=-1)), None


def _dc_kernel(dout_ref, states_ref, c_ref, da_ref, *refs, batch: int, length: int, chunk: int,
               groups: int, ratio: int, heads: int, state: int, bm: int, bn: int, exact: bool):
    """mamba_ssm's `_chunk_scan_bwd_dc_kernel` (ssd_chunk_scan.py:529-618) with `C`, for one chunk of one
    split of a group's heads and a `[BM, BN]` block of `C`'s gradient: over the split's heads, `dout_m`
    times the state the chunk entered (in `dout`'s dtype), scaled by `exp(acs_m)` (zero for a token of a
    later document than the state's), summed; and per head the gradient of `acs_m`, its row's sum with
    `C_m`."""
    *seq, dc_ref, dacs_ref = refs
    columns = state // bn
    m = pl.program_id(0) // columns
    c = pl.program_id(1) // batch
    split, group = pl.program_id(2) // groups, pl.program_id(2) % groups
    first, count = _heads_of(split, group, ratio=ratio, heads=heads)
    start = c * chunk
    limit = _limit(c, length, chunk)
    rows = pl.ds(start + m * bm, bm)
    taken = _from(limit, m * bm)
    cmat = _read(c_ref.at[rows], taken).astype(jnp.float32)
    same = seq[0][rows] == _earlier(seq[0], start, c) if seq else None

    def head(i, acc):
        h = first + i
        dout = _read(dout_ref.at[rows, h], taken)
        dc = _dot(dout, states_ref[h].astype(dout.dtype), exact=exact)
        scale = jnp.exp(da_ref[rows, h])
        if seq:
            scale = jnp.where(same, scale, 0.0)
        dc = dc * scale[:, None]
        dacs_ref[rows, h] = jnp.sum(dc * cmat, axis=1)
        return acc + dc

    _write(dc_ref.at[rows], taken, jax.lax.fori_loop(0, count, head, jnp.zeros((bm, bn), jnp.float32)))


@dataclasses.dataclass(frozen=True)
class ChunkScanBwdDc(_Launch):
    """`chunk_scan_bwd_dc`'s launch (tuning.KernelOp)."""

    NAME = "chunk_scan_bwd_dc"
    SPANS = ("chunk", ("c", 3), None)

    def _fwd(self, dout, states, c, da, seq, *, chunk: int, groups: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, heads, width = dout.shape
        state = c.shape[-1]
        padded = da.shape[1]
        ratio = heads // groups
        bm, bn = config.bm, config.bn
        columns = state // bn
        splits = -(-ratio // config.heads)
        dc, dacs = pl.pallas_call(
            functools.partial(_dc_kernel, batch=batch, length=length, chunk=chunk, groups=groups, ratio=ratio,
                              heads=config.heads, state=state, bm=bm, bn=bn, exact=interpret),
            grid=(chunk // bm * columns, batch * (padded // chunk), splits * groups),
            in_specs=[pl.BlockSpec((None, length, heads, width), lambda mn, bc, sg: (bc % batch, 0, 0, 0)),
                      pl.BlockSpec((None, None, heads, width, bn),
                                   lambda mn, bc, sg: (bc % batch, bc // batch, 0, 0, mn % columns)),
                      pl.BlockSpec((None, length, None, bn),
                                   lambda mn, bc, sg: (bc % batch, 0, sg % groups, mn % columns)),
                      _token(batch, padded, heads=heads),
                      *([] if seq is None else [_documents(batch, padded)])],
            out_specs=[pl.BlockSpec((None, length, None, None, bn),
                                    lambda mn, bc, sg: (bc % batch, 0, sg // groups, sg % groups,
                                                        mn % columns)),
                       pl.BlockSpec((None, padded, heads, None),
                                    lambda mn, bc, sg: (bc % batch, 0, 0, mn % columns))],
            out_shape=[jax.ShapeDtypeStruct((batch, length, splits, groups, state), jnp.float32),
                       jax.ShapeDtypeStruct((batch, padded, heads, columns), jnp.float32)],
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_chunk_scan_bwd_dc",
        )(dout, states, c, da, *([] if seq is None else [seq]))
        return (jnp.sum(dc, axis=2), jnp.sum(dacs, axis=-1)), None


def _dcb_kernel(x_ref, dout_ref, dt_ref, da_ref, *refs, batch: int, length: int, chunk: int, groups: int,
                ratio: int, heads: int, bm: int, bn: int, exact: bool):
    """mamba_ssm's `_chunk_scan_bwd_dcb_kernel` (ssd_chunk_scan.py:773-875) without the decay's gradient,
    for one chunk of one split of a group's heads and a `[BM, BN]` block of `C B^T`'s gradient: over the
    split's heads, `dout_m . x_n` times `dt_n exp(min(acs_m - acs_n, 0))`, summed, kept on and below
    the diagonal within a document; a block wholly above the diagonal stores zeros."""
    *seq, dcb_ref = refs
    columns = chunk // bn
    m, n = pl.program_id(0) // columns, pl.program_id(0) % columns
    c = pl.program_id(1) // batch
    split, group = pl.program_id(2) // groups, pl.program_id(2) % groups

    @pl.when(n * bn >= (m + 1) * bm)
    def _():
        dcb_ref[...] = jnp.zeros(dcb_ref.shape, dcb_ref.dtype)

    @pl.when(n * bn < (m + 1) * bm)
    def _():
        first, count = _heads_of(split, group, ratio=ratio, heads=heads)
        start = c * chunk
        limit = _limit(c, length, chunk)
        rows_m, rows_n = pl.ds(start + m * bm, bm), pl.ds(start + n * bn, bn)

        def head(i, acc):
            h = first + i
            dout = _read(dout_ref.at[rows_m, h], _from(limit, m * bm))
            x = _read(x_ref.at[rows_n, h], _from(limit, n * bn))
            dcb = _dot(dout, x.T, exact=exact) * dt_ref[rows_n, h][None, :]
            decay = jnp.exp(jnp.minimum(da_ref[rows_m, h][:, None] - da_ref[rows_n, h][None, :], 0.0))
            return acc + dcb * decay

        acc = jax.lax.fori_loop(0, count, head, jnp.zeros((bm, bn), jnp.float32))
        if seq:
            acc = jnp.where(seq[0][rows_m][:, None] == seq[0][rows_n][None, :], acc, 0.0)
        dcb_ref[...] = jnp.where(_causal(m * bm, bm, n * bn, bn), acc, 0.0)


@dataclasses.dataclass(frozen=True)
class ChunkScanBwdDcb(_Launch):
    """`chunk_scan_bwd_dcb`'s launch (tuning.KernelOp)."""

    NAME = "chunk_scan_bwd_dcb"
    SPANS = ("chunk", "chunk", None)

    def _fwd(self, x, dout, dt, da, seq, *, chunk: int, groups: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, heads, width = x.shape
        padded = dt.shape[1]
        chunks = padded // chunk
        ratio = heads // groups
        bm, bn = config.bm, config.bn
        columns = chunk // bn
        splits = -(-ratio // config.heads)
        rows = pl.BlockSpec((None, length, heads, width), lambda mn, bc, sg: (bc % batch, 0, 0, 0))
        dcb = pl.pallas_call(
            functools.partial(_dcb_kernel, batch=batch, length=length, chunk=chunk, groups=groups,
                              ratio=ratio, heads=config.heads, bm=bm, bn=bn, exact=interpret),
            grid=(chunk // bm * columns, batch * chunks, splits * groups),
            in_specs=[rows, rows, _token(batch, padded, heads=heads), _token(batch, padded, heads=heads),
                      *([] if seq is None else [_documents(batch, padded)])],
            out_specs=pl.BlockSpec((None, None, None, None, bm, bn),
                                   lambda mn, bc, sg: (bc % batch, bc // batch, sg // groups, sg % groups,
                                                       mn // columns, mn % columns)),
            out_shape=jax.ShapeDtypeStruct((batch, chunks, splits, groups, chunk, chunk), jnp.float32),
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_chunk_scan_bwd_dcb",
        )(x, dout, dt, da, *([] if seq is None else [seq]))
        return jnp.sum(dcb, axis=2), None


def _bmm_bwd_kernel(a_ref, dout_ref, res_ref, out_ref, *, length: int, chunk: int, groups: int, state: int,
                    transposed: bool, bm: int, bn: int, bk: int, exact: bool):
    """mamba_ssm's `_bmm_chunk_bwd_kernel` (ssd_bmm.py:110-162) with its residual, for one chunk of one
    group and a `[BM, BN]` block: `sum_k G_km a_k` plus the residual, in `a`'s dtype, `G` the gradient
    of `C B^T` (B's gradient from `C`) or its transpose (`transposed`, C's from `B`), each block rounded
    to `a`'s dtype before its product."""
    columns = state // bn
    m = pl.program_id(0) // columns
    c = pl.program_id(2) // groups
    start = c * chunk
    limit = _limit(c, length, chunk)
    rows = pl.ds(start + m * bm, bm)

    def step(k, acc):
        offset = k * bk
        if transposed:
            g = dout_ref[pl.ds(m * bm, bm), pl.ds(offset, bk)]
        else:
            g = dout_ref[pl.ds(offset, bk), pl.ds(m * bm, bm)].T
        a = _read(a_ref.at[pl.ds(start + offset, bk)], _from(limit, offset))
        return acc + _dot(g.astype(a.dtype), a, exact=exact)

    acc = jax.lax.fori_loop(0, chunk // bk, step, jnp.zeros((bm, bn), jnp.float32))
    taken = _from(limit, m * bm)
    _write(out_ref.at[rows], taken, acc + _read(res_ref.at[rows], taken))


@dataclasses.dataclass(frozen=True)
class BmmChunkBwd(_Launch):
    """`bmm_chunk_bwd`'s launch (tuning.KernelOp)."""

    NAME = "bmm_chunk_bwd"
    SPANS = ("chunk", ("a", 3), "chunk")

    def _fwd(self, a, dout, residual, *, chunk: int, transposed: bool, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, groups, state = a.shape
        chunks = dout.shape[1]
        bm, bn = config.bm, config.bn
        columns = state // bn
        grouped = pl.BlockSpec((None, length, None, bn),
                               lambda mn, row, cg: (row, 0, cg % groups, mn % columns))
        out = pl.pallas_call(
            functools.partial(_bmm_bwd_kernel, length=length, chunk=chunk, groups=groups, state=state,
                              transposed=transposed, bm=bm, bn=bn, bk=config.bk, exact=interpret),
            grid=(chunk // bm * columns, batch, chunks * groups),
            in_specs=[grouped,
                      pl.BlockSpec((None, None, None, chunk, chunk),
                                   lambda mn, row, cg: (row, cg // groups, cg % groups, 0, 0)),
                      grouped],
            out_specs=grouped,
            out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_bmm_chunk_bwd",
        )(a, dout, residual)
        return out, None


def _ddacs_kernel(x_ref, dout_ref, dt_ref, da_ref, cb_ref, out_ref, *, batch: int, length: int, chunk: int,
                  bm: int, bn: int, exact: bool):
    """mamba_ssm's `_chunk_scan_bwd_ddAcs_stable_kernel` (ssd_chunk_scan.py:1097-1183) for one chunk of one
    head and `BM` output rows: the gradient of each step's log decay within the chunk, `sum` over the
    pairs `n < k <= m` of `dout_m . x_n CB_mn dt_n exp(min(acs_m - acs_n, 0))`, as each row's running
    sum over `n` summed over the rows; unshifted, the value at `n` the gradient of step `n + 1`
    (`_decay_gradient`). Blocks of `n` past the rows store zeros."""
    m = pl.program_id(0)
    c = pl.program_id(1) // batch
    start = c * chunk
    limit = _limit(c, length, chunk)
    rows = pl.ds(start + m * bm, bm)
    dout = _read(dout_ref.at[rows], _from(limit, m * bm))
    da_m = da_ref[rows]

    def step(k, rowsum):
        offset = k * bn
        steps = pl.ds(start + offset, bn)
        acc = _dot(dout, _read(x_ref.at[steps], _from(limit, offset)).T, exact=exact)
        acc = acc * dt_ref[steps][None, :] * cb_ref[pl.ds(m * bm, bm), pl.ds(offset, bn)]
        acc = acc * jnp.exp(jnp.minimum(da_m[:, None] - da_ref[steps][None, :], 0.0))
        strict = _causal(m * bm, bm, offset, bn, strict=True)
        acc = jnp.where(strict, acc, 0.0)
        total = rowsum + jnp.sum(acc, axis=1)
        acc = jnp.where(strict, rowsum[:, None] + jnp.cumsum(acc, axis=1), 0.0)
        out_ref[pl.ds(offset, bn)] = jnp.sum(acc, axis=0)
        return total

    jax.lax.fori_loop(0, (m + 1) * bm // bn, step, jnp.zeros((bm,), jnp.float32))

    def zero(k, carry):
        out_ref[pl.ds(k * bn, bn)] = jnp.zeros((bn,), jnp.float32)
        return carry

    jax.lax.fori_loop((m + 1) * bm // bn, chunk // bn, zero, 0)


@dataclasses.dataclass(frozen=True)
class ChunkScanBwdDdacsStable(_Launch):
    """`chunk_scan_bwd_ddacs_stable`'s launch (tuning.KernelOp)."""

    NAME = "chunk_scan_bwd_ddacs_stable"
    SPANS = ("chunk", "chunk", None)

    def _fwd(self, x, dout, dt, da, cb, *, chunk: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        batch, length, heads, width = x.shape
        groups = cb.shape[2]
        padded = dt.shape[1]
        chunks = padded // chunk
        ratio = heads // groups
        bm = config.bm
        headed = pl.BlockSpec((None, length, None, width), lambda m, bc, h: (bc % batch, 0, h, 0))
        dacs = pl.pallas_call(
            functools.partial(_ddacs_kernel, batch=batch, length=length, chunk=chunk, bm=bm, bn=config.bn,
                              exact=interpret),
            grid=(chunk // bm, batch * chunks, heads),
            in_specs=[headed, headed, _token(batch, padded), _token(batch, padded),
                      pl.BlockSpec((None, None, None, chunk, chunk),
                                   lambda m, bc, h: (bc % batch, bc // batch, h // ratio, 0, 0))],
            out_specs=pl.BlockSpec((None, None, None, None, chunk),
                                   lambda m, bc, h: (bc % batch, bc // batch, h, m, 0)),
            out_shape=jax.ShapeDtypeStruct((batch, chunks, heads, chunk // bm, chunk), jnp.float32),
            compiler_params=_compiled(config.warps, config.stages), interpret=interpret,
            name="mamba_chunk_scan_bwd_ddacs_stable",
        )(x, dout, dt, da, cb)
        return jnp.sum(dacs, axis=3), None


_CHUNK_STATE = ChunkState()
_STATE_PASSING = StatePassing()
_BMM_CHUNK = BmmChunk()
_CHUNK_SCAN = ChunkScan()
_DSTATES = ChunkScanBwdDstates()
_STATE_PASSING_BWD = StatePassingBwd()
_DX = ChunkScanChunkStateBwdDx()
_DB = ChunkStateBwdDb()
_DC = ChunkScanBwdDc()
_DCB = ChunkScanBwdDcb()
_BMM_CHUNK_BWD = BmmChunkBwd()
_DDACS = ChunkScanBwdDdacsStable()


def _per_chunk(dt, A, seq, chunk: int):
    """The step `[B, T, H]` in fp32 padded with zeros to whole chunks, its log decay `A dt` cumulated
    within each chunk (mamba_ssm's `_chunk_cumsum_fwd`), and the documents padded with each row's
    last: a padded token neither decays nor writes, and the chunk it ends reads its last real token's
    total and document at its own last row."""
    batch, length, heads = dt.shape
    pad = -length % chunk
    dt = dt.astype(jnp.float32)
    if pad:
        dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
        seq = None if seq is None else jnp.pad(seq, ((0, 0), (0, pad)), mode="edge")
    da = jnp.cumsum((dt * A.astype(jnp.float32)).reshape(batch, -1, chunk, heads), axis=2).reshape(dt.shape)
    return dt, da, seq


def _decay_gradient(dnext, dprev, dtotal, dintra, chunk: int):
    """Each step's log decay gradient `[B, Tp, H]` from the kernels' parts (mamba_ssm's
    `_mamba_chunk_scan_combined_bwd`, ssd_combined.py:494-507): the chunk state's gradient of `acs_last
    - acs_m` cumulated over the steps before (`_chunk_state_bwd_db`'s cumsum of its shifted part); the
    carried state's gradient of `acs_m`, the chunk total's added at the last step, cumulated from the
    end; and the intra-chunk gradient, `dintra` `[B, NC, H, C]` shifted one step later."""
    batch, padded, heads = dnext.shape
    blocked = (batch, padded // chunk, chunk, heads)
    shift = ((0, 0), (0, 0), (1, 0), (0, 0))
    dnext = jnp.pad(jnp.cumsum(dnext.reshape(blocked), axis=2)[:, :, :-1], shift)
    dprev = dprev.reshape(blocked).at[:, :, -1].add(jnp.swapaxes(dtotal, 1, 2))
    dprev = jax.lax.cumsum(dprev, axis=2, reverse=True)
    dintra = jnp.pad(jnp.swapaxes(dintra, 2, 3)[:, :, :-1], shift)
    return (dintra + (dnext + dprev)).reshape(dnext.shape[0], padded, heads)


def _flat(states):
    """States `[..., P, N]` as the state passing's flattened `[..., P * N]`."""
    return states.reshape(*states.shape[:-2], -1)


def _scan(x, dt, A, b, c, d, h0, seq, chunk: int, interpret: bool):
    return _scan_fwd(x, dt, A, b, c, d, h0, seq, chunk, interpret)[0]


def _scan_fwd(x, dt, A, b, c, d, h0, seq, chunk: int, interpret: bool):
    """mamba_ssm's `_mamba_chunk_scan_combined_fwd` (ssd_combined.py:343-394): each chunk's own state,
    the states across chunks (stored in `C`'s dtype), each chunk's `C B^T`, then the output. Saves the
    inputs alone, as mamba_ssm does."""
    groups = b.shape[2]
    dtp, da, seqp = _per_chunk(dt, A, seq, chunk)
    launch = {"chunk": chunk, "interpret": interpret}
    states = _CHUNK_STATE(x, b, dtp, da, seqp, groups=groups, **launch)
    states, final = _STATE_PASSING(_flat(states), da, _flat(h0).astype(jnp.float32), seqp,
                                   dtype=jnp.dtype(c.dtype).name, **launch)
    cb = _BMM_CHUNK(c, b, seqp, padded=dtp.shape[1], **launch)
    y = _CHUNK_SCAN(cb, x, dtp, da, c, states.reshape(*states.shape[:3], *h0.shape[2:]),
                    d.astype(jnp.float32), seqp, **launch)
    return (y, final.reshape(h0.shape)), (x, dt, A, b, c, d, h0, seq)


def _scan_bwd(chunk: int, interpret: bool, residual, cotangent):
    """mamba_ssm's `_mamba_chunk_scan_combined_bwd` (ssd_combined.py:397-516): the states and `C B^T`
    recomputed, the entered states' gradient, the reverse state passing, then `x`'s, `B`'s, `C`'s and
    `C B^T`'s gradients, `C B^T`'s carried into `B` and `C`, and the decay's from its three parts,
    which gives `dt`'s and `A`'s."""
    x, dt, A, b, c, d, h0, seq = residual
    dy, dfinal = cotangent
    groups = b.shape[2]
    shape = h0.shape
    dtp, da, seqp = _per_chunk(dt, A, seq, chunk)
    launch = {"chunk": chunk, "interpret": interpret}
    cb = _BMM_CHUNK(c, b, seqp, padded=dtp.shape[1], **launch)
    states = _CHUNK_STATE(x, b, dtp, da, seqp, groups=groups, **launch)
    states, _ = _STATE_PASSING(_flat(states), da, _flat(h0).astype(jnp.float32), seqp, dtype="float32",
                               **launch)
    dy = dy.astype(x.dtype)
    dentered = _DSTATES(dy, c, da, seqp, **launch)
    dstates, dtotal, dh0, states = _STATE_PASSING_BWD(_flat(dentered), states, da,
                                                      _flat(dfinal).astype(jnp.float32), seqp,
                                                      dtype=jnp.dtype(x.dtype).name,
                                                      **launch)
    dstates, states = (s.reshape(*s.shape[:3], *shape[2:]) for s in (dstates, states))
    dx, ddt, dd = _DX(x, cb, dy, dtp, da, d.astype(jnp.float32), b, dstates, seqp, **launch)
    db, dnext = _DB(x, dstates, b, dtp, da, seqp, groups=groups, **launch)
    dc, dprev = _DC(dy, states, c, da, seqp, groups=groups, **launch)
    dcb = _DCB(x, dy, dtp, da, seqp, groups=groups, **launch)
    db = _BMM_CHUNK_BWD(c, dcb, db, transposed=False, **launch)
    dc = _BMM_CHUNK_BWD(b, dcb, dc, transposed=True, **launch)
    dacs = _decay_gradient(dnext, dprev, dtotal, _DDACS(x, dy, dtp, da, cb, **launch), chunk)
    length = dt.shape[1]
    ddt = (dacs * A.astype(jnp.float32) + ddt)[:, :length]
    dA = jnp.sum(dacs * dtp, axis=(0, 1))
    return (dx, ddt.astype(dt.dtype), dA.astype(A.dtype), db, dc, dd.astype(d.dtype),
            dh0.reshape(shape).astype(h0.dtype), None)


ssd_scan = jax.custom_vjp(_scan, nondiff_argnums=(8, 9))
"""mamba_ssm's chunked SSD scan on the ported kernels: `(y, final)` for `x` `[B, T, H, P]`, the step
`dt` `[B, T, H]` through its bias, softplus and limit, `A` `[H]`, `B` and `C` `[B, T, G, N]` in `x`'s
dtype, `D` `[H]`, each row's initial state `h0` `[B, H, P, N]` and each token's document `seq` `[B, T]`
int32 (mamba_ssm's `seq_idx`, counted from 0 at the initial state's) or None, over chunks of `chunk`
tokens. `y` is in `x`'s dtype, the final state fp32. P, N and the chunk are powers of two of at least
16. Its numbers are mamba_ssm's: operands in their own dtype into fp32 accumulation, the states the
chunks enter in the operands' dtype, the decay and the states across chunks in fp32."""
ssd_scan.defvjp(_scan_fwd, _scan_bwd)
