# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Translated from flash-linear-attention v0.5.2 (fla/ops/kda, fla/ops/gla, fla/ops/utils), which is
# licensed under the MIT license: see licenses/flash-linear-attention-LICENSE.

"""fla's chunked Kimi delta attention, ported kernel for kernel to Pallas reading Dew's [B, T, H, D].

The port follows fla-org/flash-linear-attention v0.5.2 (fla/ops/kda/chunk.py) on the path Dew's layer
takes: the queries and keys normalised before the rule, the log decay per key dimension computed
before it, beta given, fla's varlen sequences, 64-token chunks and fla's default `safe_gate=False`. It
is fla's gated delta rule (`fla_gdn`) with a decay per key dimension, and shares that port's memory
kernels, whose USE_GK branch it takes (`fla_gdn.chunk_fwd_h`, `fla_gdn.chunk_bwd_dhu`), its forward
substitution and its dot helpers, the recurrence template's decay prefix (`recurrence.cumulated`),
and the chunked launches' token spans (`Sequences`) and base (`chunked_launch._Launch`, a
`tuning.KernelOp`); the kernels here are KDA's own. Each names its fla origin.
`dew.nn.kernels.recurrence_chunk.chunk_gated_delta_rule` runs it for a decay per key dimension.

fla's grids are kept: a program per place of each chunk for the intra-chunk diagonal blocks (fla's
per token, which searches the token's sequence), per chunk for the solve, per key block and
sub-chunk for the backward intra-chunk gradients.

Operands keep their dtypes as fla's do: bf16 queries, keys and values store `A`, `w`, `u`, the
gated keys and the memory each chunk enters in bf16 with fp32 accumulation; the decay is fp32 and
cumulated in base 2 (`recurrence.cumulated`); gradients of q, k, the decay and beta accumulate
in fp32.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl

from dew.nn.kernels import fla_gdn, recurrence, recurrence_algebra as algebra
from dew.nn.kernels.chunked_launch import (
    _CHUNK,
    Sequences,
    Tiles,
    _aligned,
    _chunk_of,
    _flat,
    _gates,
    _headed,
    _Launch,
    _picks,
    _row,
    _rows,
    laid,
    read_rows,
    sequences,
    spanned,
    write_rows,
)
from dew.nn.kernels.fla_gdn import _dot, _weighted_values
from dew.nn.precision import compensated_difference

SUB = 16
"""fla's BC: the sub-chunk the intra-chunk kernels build their 16 x 16 blocks over."""


def _intra_rows_kernel(chunk_ref, q_ref, k_ref, g_ref, beta_ref, *refs, chunk: int, aligned: bool,
                       scale: float, compensated: bool = False):
    """One token's row of its chunk's scores and keys' products for `BH` value heads, every pair's
    decay its own: `Aqk_ij = scale sum_k q_ik k_jk exp2(g_ik - g_jk)` for `j <= i` and `Akk_ij = sum_k
    beta_i k_ik k_jk exp2(g_ik - g_jk)` for `j < i`, in fp32, `Aqk` stored in its dtype and `Akk` fp32.
    fla's `chunk_kda_fwd_kernel_intra_token_parallel` (chunk_intra_token_parallel.py:33-118, its
    `safe_gate=False` path) takes the diagonal 16-token blocks so, and factors every block below them
    through its sub-chunk's first row (`(q exp2(g_i - g_n)) (k exp2(g_n - g_j))`, chunk_intra.py:43-391):
    two exponentials rounded and one more product a term, against one exponential here (Dew's portable
    math's, `dew.nn.kernels.recurrence_algebra.decays`'). This port's earlier factored blocks,
    which moved decayed keys from sub-chunk to sub-chunk where fla recomputes each factor, put 11 of a
    packed row's 16384 scores and 5 of its inverse's past twice the pairwise form's distance from
    float64; fla's own factored form was
    not measured. A program per place of a chunk, where fla's searches the token's sequence; a place
    past the chunk's end does nothing. Where `compensated`, each pair's decay is of both prefix halves
    (`algebra.apart`): from the rounded prefix alone a steep token's rounding took every mild one after it."""
    lo_ref, (aqk_ref, akk_ref) = (refs[0], refs[1:]) if compensated else (None, refs)
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    place = pl.program_id(0) % chunk

    @pl.when(place < taken)
    def _():
        token = start + place
        index = jnp.arange(chunk)
        query = q_ref[token].astype(jnp.float32)
        weighted = k_ref[token].astype(jnp.float32) * beta_ref[token].astype(jnp.float32)[:, None]
        gate = _row(g_ref, token)
        low = None if lo_ref is None else _row(lo_ref, token)

        def column(j, sums):
            earlier = start + j
            if lo_ref is None:
                decay = jnp.exp2(gate - _row(g_ref, earlier))
            else:
                before = (_row(g_ref, earlier), _row(lo_ref, earlier))
                decay = jnp.exp2(compensated_difference((gate, low), before))
            gated = k_ref[earlier].astype(jnp.float32) * decay
            here = (index == j)[None, :]
            return (jnp.where(here, (jnp.sum(query * gated, axis=1) * scale)[:, None], sums[0]),
                    jnp.where(here & (j < place), jnp.sum(weighted * gated, axis=1)[:, None], sums[1]))

        empty = jnp.zeros((query.shape[0], chunk), jnp.float32)
        qk, kk = jax.lax.fori_loop(0, place + 1, column, (empty, empty))
        aqk_ref[token] = qk.astype(aqk_ref.dtype)
        akk_ref[token] = kk


@dataclasses.dataclass(frozen=True)
class IntraRows(_Launch):
    """`intra_rows`'s launch (tuning.KernelOp): its value heads a program, warps and stages."""

    NAME = "intra_rows"
    KEYS, VALUES = "k", None
    PICKS = _picks(Tiles(0, 0, 4, 1, 4))

    def _tiles(self, ba, tiles: Tiles) -> Tiles:
        """A block of value heads that divides them, one where value heads share a key head, as fla's
        ceil(HV / BH) grid masks its tail."""
        value_heads, key_heads = ba.arguments["beta"].shape[2], ba.arguments["k"].shape[2]
        bh = math.gcd(tiles.bh, value_heads) if value_heads == key_heads else 1
        return dataclasses.replace(tiles, bh=bh)

    def _fwd(self, q: jax.Array, k: jax.Array, g: jax.Array, g_lo: jax.Array | None, beta: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, scale: float, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, key_heads, width = k.shape
        value_heads = beta.shape[2]
        group, bh, tokens = value_heads // key_heads, config.bh, batch * length
        keys = pl.BlockSpec((tokens, bh, width), lambda p, b: (0, b // group, 0))
        out = pl.BlockSpec((tokens, bh, chunk), lambda p, b: (0, b, 0))
        found = pl.pallas_call(
            functools.partial(_intra_rows_kernel, chunk=chunk,
                              aligned=_aligned(spans, batch, length, chunk), scale=scale,
                              compensated=g_lo is not None),
            grid=(spans.chunks.shape[0] * chunk, value_heads // bh),
            in_specs=[pl.BlockSpec((None, 4), lambda p, b: (p // chunk, 0)), keys, keys,
                      pl.BlockSpec((tokens, bh, width), lambda p, b: (0, b, 0)),
                      pl.BlockSpec((tokens, bh), lambda p, b: (0, b)),
                      *([pl.BlockSpec((tokens, bh, width), lambda p, b: (0, b, 0))]
                        if g_lo is not None else [])],
            out_specs=[out, out],
            out_shape=[jax.ShapeDtypeStruct((tokens, value_heads, chunk), k.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads, chunk), jnp.float32)],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_intra_rows",
        )(spans.chunks, *_flat(q, k, g, beta, *([] if g_lo is None else [g_lo])))
        return tuple(x.reshape(batch, length, value_heads, chunk) for x in found), None


_INTRA_ROWS = IntraRows()


def intra_rows(q, k, g, beta, *, scale: float, g_lo=None, chunk: int = 64, spans: Sequences | None = None,
               interpret: bool = False):
    """`(Aqk, Akk)` `[B, T, HV, BT]`, each token's row of its chunk, `Aqk` in `k`'s dtype and `Akk` fp32,
    every pair's decay its own (`_intra_rows_kernel`): a program per place of each chunk (`spans`, the
    rows' own where None) and `BH` value heads."""
    return spanned(_INTRA_ROWS, q, k, g, g_lo, beta, chunk=chunk, spans=spans, scale=scale,
                   interpret=interpret)


def _intra_solve_kernel(chunk_ref, aqkd_ref, akkd_ref, aqk_ref, akk_ref, *, chunk: int, aligned: bool,
                        exact: bool, solve):
    """fla's `chunk_kda_fwd_kernel_inter_solve_fused` (chunk_intra.py:43-391), safe_gate off, from the
    rows `intra_rows` computed, for one chunk of one head in 16-row blocks: `Aqk` stored in the keys'
    dtype, and `I + Akk`'s diagonal blocks inverted by forward substitution and merged in fla's
    order, `X_IJ = -(X_II A_IJ) X_JJ` for neighbours and `-X_II (sum_M A_IM X_MJ)` beyond, at the
    call's solve precision (`fla_gdn._dot`'s `solve`), the inverse stored in the keys' dtype; both
    zero above the diagonal. A chunk's rows past its end are none of them stored."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        count = chunk // SUB
        zero = jnp.zeros((SUB, SUB), jnp.float32)

        def part(ref, i, j):
            return read_rows(_rows(ref, start + i * SUB, SUB, pl.ds(j * SUB, SUB)), taken - i * SUB).astype(
                jnp.float32)

        def store(ref, i, j, value):
            write_rows(_rows(ref, start + i * SUB, SUB, pl.ds(j * SUB, SUB)), taken - i * SUB, value)

        for i in range(count):
            for j in range(count):
                store(aqk_ref, i, j, part(aqkd_ref, i, j) if j <= i else zero)
        kks = {(i, j): part(akkd_ref, i, j) for i in range(count) for j in range(i + 1)}
        inverse = algebra.inverted(kks, lambda name, a, b: _dot(a, b, exact=exact, solve=solve),
                                   neighbours_first=True)
        for i in range(count):
            for j in range(count):
                store(akk_ref, i, j, inverse[i, j] if j <= i else zero)


@dataclasses.dataclass(frozen=True)
class IntraSolve(_Launch):
    """`intra_solve`'s launch (tuning.KernelOp)."""

    revision: int = 2

    NAME = "intra_solve"
    KEYS, VALUES = "aqk", None
    PICKS = _picks(Tiles(0, 0, 1, 1), small=Tiles(0, 0, 2, 1))

    def _fwd(self, aqk: jax.Array, akk: jax.Array, chunk_spans: jax.Array, sequence_spans: jax.Array, *,
             chunk: int, interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, value_heads = aqk.shape[:3]
        tokens = batch * length
        headed = functools.partial(_headed, tokens)
        shape = jax.ShapeDtypeStruct((tokens, value_heads, chunk), aqk.dtype)
        found = pl.pallas_call(
            functools.partial(_intra_solve_kernel, chunk=chunk, aligned=_aligned(spans, batch, length, chunk),
                              exact=interpret, solve=aqk.dtype),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, headed(chunk), headed(chunk)],
            out_specs=[headed(chunk), headed(chunk)], out_shape=[shape, shape],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_intra_solve",
        )(spans.chunks, *_flat(aqk, akk))
        return tuple(x.reshape(batch, length, value_heads, chunk) for x in found), None


_INTRA_SOLVE = IntraSolve()


def intra_solve(q, k, g, beta, *, scale: float, g_lo=None, chunk: int = 64,
                spans: Sequences | None = None, interpret: bool = False):
    """`(Aqk, Akk)` `[B, T, HV, BT]` in `k`'s dtype: each chunk's `scale (q exp2(g)) (k exp2(-g))^T`,
    causal, and the inverse `(I + tril(beta (k exp2(g)) (k exp2(-g))^T, -1))^-1`: fla's
    `chunk_kda_fwd_intra` before its `recompute_w_u_fwd` (chunk_intra.py:794-894), each pair's decay
    its own (`intra_rows`), then the solve chunk by chunk (`spans`, the rows' own where None). `q`, `k`
    `[B, T, H, K]`, `g` the base-2 cumulated decay (`recurrence.cumulated`) `[B, T, HV, K]` fp32,
    `beta` `[B, T, HV]`, `g_lo` its remainder where compensated."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    aqk, akk = intra_rows(q, k, g, beta, scale=scale, g_lo=g_lo, chunk=chunk, spans=spans,
                          interpret=interpret)
    return _INTRA_SOLVE(aqk, akk, *spans, chunk=chunk, interpret=interpret)


def _w_u_kernel(chunk_ref, k_ref, v_ref, beta_ref, a_ref, g_ref, *refs, chunk: int, aligned: bool,
                bk: int, bv: int, queries: bool, compensated: bool, exact: bool):
    """fla's `recompute_w_u_fwd_kda_kernel` (fla/ops/kda/wy_fast.py:34-140) for one chunk of one head:
    `u = A (v beta)` and `w = A ((k beta) exp2(g))`, each operand rounded to its input dtype before the
    product; the gated keys `k exp2(g_last - g)`; with `queries`, the gated queries `q exp2(g)`. Where
    `compensated`, the gated keys' range is the halves' (`algebra.apart`): from the rounded prefix alone,
    Kimi delta attention's final memory was 1.57 times fla's recorded torch's distance from float64 on an
    H100, 1.27 from both halves (row 1328)."""
    lo_ref, refs = (refs[0], refs[1:]) if compensated else (None, refs)
    if queries:
        q_ref, w_ref, u_ref, kg_ref, qg_ref = refs
    else:
        (w_ref, u_ref, kg_ref), q_ref, qg_ref = refs, None, None
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        def write(ref, columns, value):
            write_rows(_rows(ref, start, chunk, columns), taken, value)

        beta = read(beta_ref)
        inverse = read(a_ref)
        _weighted_values(v_ref, u_ref, beta, inverse, start, taken, chunk=chunk, bv=bv, exact=exact,
                         ieee=False)
        for block in range(k_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            key = read(k_ref, columns)
            gate = _gates(g_ref, start, taken, chunk, 0, columns)
            raised = jnp.exp2(gate) if lo_ref is None else algebra.raised(
                gate, _gates(lo_ref, start, taken, chunk, 0, columns))
            scaled = (key * beta[:, None]) * raised
            if q_ref is not None and qg_ref is not None:
                write(qg_ref, columns, read(q_ref, columns) * raised)
            write(kg_ref, columns, key * jnp.exp2(_range(g_ref, lo_ref, start, taken, chunk, columns, gate)))
            write(w_ref, columns, _dot(inverse, scaled.astype(key.dtype), exact=exact))


def _range(g_ref, lo_ref, start, taken, chunk: int, columns, gate):
    """Each of a chunk's rows' decay to its last real row in base 2, `g_last - g` `[chunk, columns]` (`gate`
    the rows' prefix, read): where the prefix's remainder `lo_ref` is given, the difference of both halves
    (`precision.compensated_difference`), else of the rounded prefix alone."""
    last = _row(g_ref, (start + taken - 1, columns))[None, :]
    if lo_ref is None:
        return last - gate
    return compensated_difference((last, _row(lo_ref, (start + taken - 1, columns))[None, :]),
                                  (gate, _gates(lo_ref, start, taken, chunk, 0, columns)))


@dataclasses.dataclass(frozen=True)
class RecomputeWU(_Launch):
    """`recompute_w_u`'s launch (tuning.KernelOp)."""

    NAME = "recompute_w_u"
    KEYS, VALUES = "k", "v"
    PICKS = _picks(Tiles(64, 64, 4, 2))

    def _fwd(self, k: jax.Array, v: jax.Array, beta: jax.Array, a: jax.Array, g: jax.Array,
             g_lo: jax.Array | None, q: jax.Array | None, chunk_spans: jax.Array, sequence_spans: jax.Array,
             *, chunk: int, interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans, batch, length, _, width, value_heads, columns, group, tokens, aligned = laid(
            k, v, chunk_spans, sequence_spans, chunk)
        keys = pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0))

        headed = functools.partial(_headed, tokens)

        keyed = jax.ShapeDtypeStruct((tokens, value_heads, width), k.dtype)
        queries, compensated = q is not None, g_lo is not None
        found = pl.pallas_call(
            functools.partial(_w_u_kernel, chunk=chunk, aligned=aligned,
                              bk=min(config.bk, width), bv=min(config.bv, columns), queries=queries,
                              compensated=compensated, exact=interpret),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, keys, headed(columns), pl.BlockSpec((tokens, None), lambda i_t, h: (0, h)),
                      headed(chunk), headed(width), *([headed(width)] if compensated else []),
                      *([keys] if queries else [])],
            out_specs=[headed(width), headed(columns), headed(width), *([headed(width)] if queries else [])],
            out_shape=[keyed, jax.ShapeDtypeStruct((tokens, value_heads, columns), v.dtype), keyed,
                       *([keyed] if queries else [])],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_recompute_w_u",
        )(spans.chunks, *_flat(k, v, beta, a, g, *([g_lo] if compensated else []),
                               *([q] if queries else [])))
        w, u, kg, *qg = found
        shaped = (w.reshape(batch, length, value_heads, width), u.reshape(v.shape),
                  kg.reshape(batch, length, value_heads, width))
        return (*shaped, qg[0].reshape(batch, length, value_heads, width) if queries else None), None


_RECOMPUTE_W_U = RecomputeWU()


def recompute_w_u(k, v, beta, a, g, q=None, *, g_lo=None, chunk: int = 64, spans: Sequences | None = None,
                  interpret: bool = False):
    """`(w, u, kg, qg)`: fla's `recompute_w_u_fwd` for KDA (wy_fast.py:264-311). `w`, `kg` and `qg`
    `[B, T, HV, K]` in `k`'s dtype, `u` in `v`'s; `qg` only for given queries `q` `[B, T, H, K]`, else
    None. `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `beta` `[B, T, HV]`, `a` the inverse per chunk `[B,
    T, HV, BT]`, `g` the base-2 cumulated decay `[B, T, HV, K]` fp32 and `g_lo` its remainder, where
    given (`recurrence.cumulated(compensated=True)`), which the gated keys' range takes too."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _RECOMPUTE_W_U(k, v, beta, a, g, g_lo, q, *spans, chunk=chunk, interpret=interpret)


def _o_kernel(chunk_ref, q_ref, v_ref, g_ref, h_ref, a_ref, *refs, chunk: int, aligned: bool, scale: float,
              bk: int, exact: bool, compensated: bool = False):
    """fla's `chunk_gla_fwd_kernel_o` (fla/ops/gla/chunk.py:345-432) for one chunk of one head and `BV`
    value columns: `scale (q exp2(g))_narrow h + tril(Aqk) v`, the gated queries rounded to their
    dtype and `Aqk` to the values', fp32 products IEEE as the gated delta rule's output kernel takes
    them (`fla_gdn._o_kernel`), the memory the chunk enters read at its slot (`Sequences`); the queries'
    decay of both prefix halves where `compensated` (`algebra.raised`)."""
    lo_ref, (o_ref,) = (refs[0], refs[1:]) if compensated else (None, refs)
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    entered = chunk_ref[2]

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        out = jnp.zeros((chunk, v_ref.shape[-1]), jnp.float32)
        for block in range(q_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            query, gate = read(q_ref, columns), _gates(g_ref, start, taken, chunk, 0, columns)
            raised = jnp.exp2(gate) if lo_ref is None else algebra.raised(
                gate, _gates(lo_ref, start, taken, chunk, 0, columns))
            gated = (query * raised).astype(query.dtype)
            out = out + _dot(gated, h_ref[entered, columns, :].astype(gated.dtype), exact=exact, ieee=True)
        out = out * scale
        index = jnp.arange(chunk)
        value = read(v_ref)
        scores = jnp.where(index[:, None] >= index[None, :], read(a_ref), 0.0).astype(value.dtype)
        write_rows(_rows(o_ref, start, chunk), taken, out + _dot(scores, value, exact=exact, ieee=True))


@dataclasses.dataclass(frozen=True)
class ChunkFwdO(_Launch):
    """KDA's `chunk_fwd_o`'s launch (tuning.KernelOp)."""

    NAME = "chunk_fwd_o"
    KEYS, VALUES = "q", "v"
    # Measured on an A100-40GB at Kimi Linear's 1 x 4096 x 32 x 128 bf16: 0.25 -> 0.23 ms.
    PICKS = _picks(Tiles(64, 128, 4, 2), small=Tiles(64, 64, 4, 2))

    def _fwd(self, q: jax.Array, v: jax.Array, g: jax.Array, g_lo: jax.Array | None, a: jax.Array,
             h: jax.Array, chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, scale: float,
             interpret: bool,
             config: Tiles, wide: bool = False, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans, _, _, _, width, value_heads, columns, group, tokens, aligned = laid(
            q, v, chunk_spans, sequence_spans, chunk)
        bv = min(config.bv, columns)
        dtype = jnp.float32 if wide else v.dtype
        values = pl.BlockSpec((tokens, None, bv), lambda i_v, i_t, i_h: (0, i_h, i_v))
        out = pl.pallas_call(
            functools.partial(_o_kernel, chunk=chunk, aligned=aligned, scale=scale, bk=min(config.bk, width),
                              exact=interpret, compensated=g_lo is not None),
            grid=(columns // bv, spans.chunks.shape[0], value_heads),
            in_specs=[pl.BlockSpec((None, 4), lambda i_v, i_t, i_h: (i_t, 0)),
                      pl.BlockSpec((tokens, None, width), lambda i_v, i_t, i_h: (0, i_h // group, 0)), values,
                      pl.BlockSpec((tokens, None, width), lambda i_v, i_t, i_h: (0, i_h, 0)),
                      pl.BlockSpec((h.shape[0], None, width, bv), lambda i_v, i_t, i_h: (0, i_h, 0, i_v)),
                      pl.BlockSpec((tokens, None, chunk), lambda i_v, i_t, i_h: (0, i_h, 0)),
                      *([pl.BlockSpec((tokens, None, width), lambda i_v, i_t, i_h: (0, i_h, 0))]
                        if g_lo is not None else [])],
            out_specs=values, out_shape=jax.ShapeDtypeStruct((tokens, value_heads, columns), dtype),
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_chunk_fwd_o",
        )(spans.chunks, *_flat(q, v, g), h, *_flat(a, *([] if g_lo is None else [g_lo])))
        return out.reshape(v.shape), None


_CHUNK_FWD_O = ChunkFwdO()


def chunk_fwd_o(q, v, g, a, h, *, scale: float, g_lo=None, chunk: int = 64, spans: Sequences | None = None,
                interpret: bool = False, wide: bool = False):
    """The output `[B, T, HV, V]` in `v`'s dtype, fp32 where `wide` (`fla_gdn.chunk_fwd_o`'s): fla's
    `chunk_gla_fwd_o_gk` (gla/chunk.py:962-997). `q`
    `[B, T, H, K]`, `v` the corrected values `[B, T, HV, V]`, `g` the base-2 cumulated decay `[B, T, HV,
    K]`, `a` `Aqk` `[B, T, HV, BT]`, `h` the memories the chunks enter at their slots
    (`fla_gdn.chunk_fwd_h`'s), `g_lo` the decay's remainder where compensated."""
    spans = sequences(*q.shape[:2], chunk=chunk) if spans is None else spans
    return _CHUNK_FWD_O(q, v, g, g_lo, a, h, *spans, chunk=chunk, scale=scale, interpret=interpret, wide=wide)


def _dav_kernel(chunk_ref, v_ref, do_ref, a_ref, dv_ref, da_ref, *, chunk: int, aligned: bool, scale: float,
                bv: int, exact: bool):
    """fla's `chunk_kda_bwd_kernel_dAv` (fla/ops/kda/chunk_bwd.py:42-129) for one chunk of one head:
    `dAqk = scale tril(do v^T)` in fp32 and `dv = tril(Aqk)^T do`, `Aqk` in the gradient's dtype."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        index = jnp.arange(chunk)
        out_grad_dtype = do_ref.dtype
        transposed = jnp.where(index[:, None] <= index[None, :], read(a_ref).T, 0.0).astype(out_grad_dtype)
        scores = jnp.zeros((chunk, chunk), jnp.float32)
        for block in range(v_ref.shape[-1] // bv):
            columns = pl.ds(block * bv, bv)
            out_grad = read(do_ref, columns)
            scores = scores + _dot(out_grad, read(v_ref, columns).T, exact=exact)
            write_rows(_rows(dv_ref, start, chunk, columns), taken, _dot(transposed, out_grad, exact=exact))
        causal = index[:, None] >= index[None, :]
        write_rows(_rows(da_ref, start, chunk), taken, jnp.where(causal, scores * scale, 0.0))


@dataclasses.dataclass(frozen=True)
class BwdDav(_Launch):
    """`bwd_dav`'s launch (tuning.KernelOp)."""

    NAME = "bwd_dav"
    KEYS, VALUES = None, "v"
    PICKS = _picks(Tiles(0, 64, 4, 2))

    def _fwd(self, v: jax.Array, do: jax.Array, a: jax.Array, chunk_spans: jax.Array,
             sequence_spans: jax.Array, *, chunk: int, scale: float, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, value_heads, columns = v.shape
        tokens = batch * length

        headed = functools.partial(_headed, tokens)

        da, dv = pl.pallas_call(
            functools.partial(_dav_kernel, chunk=chunk, aligned=_aligned(spans, batch, length, chunk),
                              scale=scale, bv=min(config.bv, columns), exact=interpret),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, headed(columns), headed(columns), headed(chunk)],
            out_specs=[headed(columns), headed(chunk)],
            out_shape=[jax.ShapeDtypeStruct((tokens, value_heads, columns), do.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads, chunk), jnp.float32)],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_bwd_dav",
        )(spans.chunks, *_flat(v, do, a))[::-1]
        return (da.reshape(batch, length, value_heads, chunk), dv.reshape(do.shape)), None


_BWD_DAV = BwdDav()


def bwd_dav(v, do, a, *, scale: float, chunk: int = 64, spans: Sequences | None = None,
            interpret: bool = False):
    """`(dAqk, dv)`: fla's `chunk_kda_bwd_dAv` (chunk_bwd.py:314-365), `dAqk` `[B, T, HV, BT]` fp32 and
    `dv` `[B, T, HV, V]` in `do`'s dtype. `v` the corrected values, `do` the output's gradient, `a`
    `Aqk`."""
    spans = sequences(*v.shape[:2], chunk=chunk) if spans is None else spans
    return _BWD_DAV(v, do, a, *spans, chunk=chunk, scale=scale, interpret=interpret)


def _wy_dqkg_kernel(chunk_ref, q_ref, k_ref, v_ref, vn_ref, g_ref, beta_ref, a_ref, h_ref, do_ref, dh_ref,
                    dv_ref, *refs, chunk: int, aligned: bool, scale: float, bk: int, bv: int, exact: bool,
                    compensated: bool):
    """fla's `chunk_kda_bwd_kernel_wy_dqkg_fused` (fla/ops/kda/chunk_bwd.py:132-311) for one chunk of
    one head: the queries', keys', decay's and beta's gradients through the output's memory term and
    the WY representation, the values' through `A`, and the gradient of `Akk`, with fla's casts and
    sums in its order. `A` is read transposed, as fla loads it; fla's first key block's value terms
    run first, on their own loop. The memory the chunk enters and its gradient where it leaves are
    read at their slots (`Sequences`). Where `compensated`, the keys' gradient takes the gated keys' range
    from both prefix halves, as the forward wrote them (`_w_u_kernel`, `_range`): from the rounded prefix
    alone, a chunk whose first decay is steep differentiated a write the forward did not make (row 2103)."""
    lo_ref, refs = (refs[0], refs[1:]) if compensated else (None, refs)
    dq_ref, dk_ref, dv2_ref, dg_ref, db_ref, da_ref = refs
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    entered, left = chunk_ref[2], chunk_ref[3]

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        def write(ref, value, *rest):
            write_rows(_rows(ref, start, chunk, *rest), taken, value)

        width, columns = k_ref.shape[-1], v_ref.shape[-1]
        index = jnp.arange(chunk)
        beta = read(beta_ref)
        transposed = read(a_ref).T

        def first(i_v, carry):
            scores, beta_grad = carry
            values = pl.ds(i_v * bv, bv)
            value_grad, value = read(dv_ref, values), read(v_ref, values)
            weighted = _dot(transposed, value_grad, exact=exact)
            write(dv2_ref, weighted * beta[:, None], values)
            return (scores + _dot(value_grad, value.T, exact=exact),
                    beta_grad + jnp.sum(weighted * value, axis=1))

        carry = (jnp.zeros((chunk, chunk), jnp.float32), jnp.zeros((chunk,), jnp.float32))
        carry = jax.lax.fori_loop(0, columns // bv, first, carry)

        def key_block(i_k, carry):
            scores, beta_grad = carry
            keys = pl.ds(i_k * bk, bk)

            def value_block(i_v, sums):
                dq, dk, dw, decay_grad = sums
                values = pl.ds(i_v * bv, bv)
                corrected, out_grad = read(vn_ref, values), read(do_ref, values)
                memory, memory_grad = h_ref[entered, keys, values], dh_ref[left, keys, values]
                return (dq + _dot(out_grad, memory.T.astype(out_grad.dtype), exact=exact),
                        dk + _dot(corrected, memory_grad.T.astype(corrected.dtype), exact=exact),
                        dw + _dot(read(dv_ref, values).astype(corrected.dtype),
                                  memory.T.astype(corrected.dtype), exact=exact),
                        decay_grad + jnp.sum((memory * memory_grad).astype(jnp.float32), axis=1))

            empty = jnp.zeros((chunk, bk), jnp.float32)
            dq, dk, dw, decay_grad = jax.lax.fori_loop(0, columns // bv, value_block,
                                                       (empty, empty, empty, jnp.zeros((bk,), jnp.float32)))
            key = read(k_ref, keys)
            gate = _gates(g_ref, start, taken, chunk, 0, keys)
            final = _row(g_ref, (start + taken - 1, keys))
            gate_exp = jnp.exp2(gate)
            decay_grad = decay_grad * jnp.exp2(final)
            dq = dq * gate_exp * scale
            dk = dk * jnp.exp2(_range(g_ref, lo_ref, start, taken, chunk, keys, gate))
            gated_key = key * gate_exp
            dw = -dw.astype(transposed.dtype)
            scores = scores + _dot(dw, gated_key.T.astype(transposed.dtype), exact=exact)
            dkgb = _dot(transposed, dw, exact=exact)
            beta_grad = beta_grad + jnp.sum(dkgb * gated_key, axis=1)
            key_dk = key * dk
            decay_grad = decay_grad + jnp.sum(key_dk, axis=0)
            dg = (read(q_ref, keys) * dq - key_dk
                  + jnp.where((index == taken - 1)[:, None], decay_grad[None, :], 0.0)
                  + gated_key * dkgb * beta[:, None])
            write(dq_ref, dq, keys)
            write(dk_ref, dk + dkgb * (gate_exp * beta[:, None]), keys)
            write(dg_ref, dg, keys)
            return scores, beta_grad

        scores, beta_grad = jax.lax.fori_loop(0, width // bk, key_block, carry)
        strict = index[:, None] > index[None, :]
        scores = jnp.where(strict, scores * beta[None, :], 0.0)
        scores = _dot(scores.astype(transposed.dtype), transposed, exact=exact)
        scores = _dot(transposed, scores.astype(transposed.dtype), exact=exact)
        write(da_ref, jnp.where(strict, -scores, 0.0))
        write(db_ref, beta_grad)


@dataclasses.dataclass(frozen=True)
class BwdWyDqkg(_Launch):
    """`bwd_wy_dqkg`'s launch (tuning.KernelOp)."""

    NAME = "bwd_wy_dqkg"
    KEYS, VALUES = "k", "v"
    PICKS = _picks(Tiles(32, 64, 4, 1), small=Tiles(32, 32, 4, 1))

    def _fwd(self, q: jax.Array, k: jax.Array, v: jax.Array, corrected: jax.Array, g: jax.Array,
             g_lo: jax.Array | None, beta: jax.Array, a: jax.Array, h: jax.Array, do: jax.Array,
             dh: jax.Array, dv: jax.Array, chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int,
             scale: float, interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans, batch, length, _, width, value_heads, columns, group, tokens, aligned = laid(
            k, v, chunk_spans, sequence_spans, chunk)
        keys = pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0))

        headed = functools.partial(_headed, tokens)

        gate = pl.BlockSpec((tokens, None), lambda i_t, h: (0, h))
        states = pl.BlockSpec((h.shape[0], None, width, columns), lambda i_t, head: (0, head, 0, 0))
        wide = jax.ShapeDtypeStruct((tokens, value_heads, width), jnp.float32)
        compensated = g_lo is not None
        dq, dk, dv2, dg, db, da = pl.pallas_call(
            functools.partial(_wy_dqkg_kernel, chunk=chunk, aligned=aligned,
                              scale=scale, bk=min(config.bk, width), bv=min(config.bv, columns),
                              exact=interpret, compensated=compensated),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, keys, keys, headed(columns), headed(columns), headed(width), gate,
                      headed(chunk), states, headed(columns), states, headed(columns),
                      *([headed(width)] if compensated else [])],
            out_specs=[headed(width), headed(width), headed(columns), headed(width), gate, headed(chunk)],
            out_shape=[wide, wide, jax.ShapeDtypeStruct((tokens, value_heads, columns), v.dtype), wide,
                       jax.ShapeDtypeStruct((tokens, value_heads), jnp.float32),
                       jax.ShapeDtypeStruct((tokens, value_heads, chunk), jnp.float32)],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_bwd_wy_dqkg",
        )(spans.chunks, *_flat(q, k, v, corrected, g, beta, a), h, *_flat(do), dh, *_flat(dv),
          *(_flat(g_lo) if compensated else ()))
        keyed = (batch, length, value_heads, width)
        return (dq.reshape(keyed), dk.reshape(keyed), dv2.reshape(v.shape), db.reshape(beta.shape),
                dg.reshape(keyed), da.reshape(batch, length, value_heads, chunk)), None


_BWD_WY_DQKG = BwdWyDqkg()


def bwd_wy_dqkg(q, k, v, corrected, g, beta, a, h, do, dh, dv, *, scale: float, g_lo=None, chunk: int = 64,
                spans: Sequences | None = None, interpret: bool = False):
    """`(dq, dk, dv, db, dg, dAkk)`: fla's `chunk_kda_bwd_wy_dqkg_fused` (chunk_bwd.py:367-431), `dq`,
    `dk`, `dg` `[B, T, HV, K]`, `db` `[B, T, HV]` and `dAkk` `[B, T, HV, BT]` fp32, `dv` in `v`'s dtype.
    `h` and `dh` the memories the chunks enter and their gradients at their slots
    (`fla_gdn.chunk_fwd_h`'s and `chunk_bwd_dhu`'s), `dv` the corrected values' full gradient, `a`
    `Akk`'s inverse; `g_lo` the decay prefix's remainder where given (`recompute_w_u`'s), which the keys'
    gradient's range takes as the gated keys' does."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _BWD_WY_DQKG(q, k, v, corrected, g, g_lo, beta, a, h, do, dh, dv, *spans, chunk=chunk, scale=scale,
                        interpret=interpret)


def _bwd_intra_kernel(chunk_ref, q_ref, k_ref, g_ref, beta_ref, daqk_ref, dakk_ref, dq_ref, dk_ref,
                      dg_ref, dq2_ref, dk2_ref, db2_ref, dg2_ref, *, chunk: int, aligned: bool, exact: bool):
    """fla's `chunk_kda_bwd_kernel_intra` (fla/ops/kda/chunk_intra.py:395-679), safe_gate off, for one
    sub-chunk `i` of a chunk, one head and one block of key columns, as fla's grid runs it. The
    gradients of `Aqk` and `Akk` reach the sub-chunk's queries and keys from the sub-chunks before
    (`dq2`, `dk2`, the decay factored at its first row) and after (`dkt`, factored at its last row),
    fp32 products at the call's solve precision (`fla_gdn._dot`'s `solve`), and from its own tokens
    one by one in fp32; then beta's
    and the decay's gradients, added to the ones `bwd_wy_dqkg` gave. Beta's is this key block's share. In
    an fp32 call the decay's gradient is each pair's own (`_paired_decay`), where fla recovers it from the
    summed gradients, `q dq + (dk - dk^T) k`: that cancellation, its decays factored at a sub-chunk's rows,
    put the decay's gradient 1.6 times fla's recorded torch's distance from float64 on an A100, 1.4 paired
    (SE-dbg-kda-pairwise32-5480feaf3); a 16-bit call's operands' rounding stays above it.
    A chunk's rows past its end read as zeros under its last row's decay, and its own tokens stop
    there, as fla's varlen loads stop at the sequence's end."""
    count = chunk // SUB
    i = pl.program_id(1) % count
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    first = i * SUB

    @pl.when(first < taken)
    def _():
        index = jnp.arange(SUB)
        width = k_ref.shape[-1]
        own_tokens = SUB if aligned else jnp.minimum(SUB, taken - first)
        aligned_tokens = chunk if aligned else taken

        def left(at):
            """The chunk's rows from `at`: a whole sub-chunk's where every chunk is whole."""
            return SUB if aligned else taken - at

        def tile(ref, at):
            return read_rows(_rows(ref, start + at, SUB), left(at))

        def gates(at):
            return _gates(g_ref, start + at, left(at), SUB)

        def scores(ref, row, column):
            return read_rows(_rows(ref, start + row, SUB, pl.ds(column, SUB)), left(row)).astype(jnp.float32)

        def row(ref, at):
            return ref[start + at].astype(jnp.float32)

        def decayed(at):
            """The cumulated decay at the chunk's row `at`, `[1, K]`."""
            return _row(g_ref, start + at)[None, :]

        gate = gates(first)
        key = tile(k_ref, first).astype(jnp.float32)
        beta = tile(beta_ref, first).astype(jnp.float32)
        origin = decayed(first)

        def before(j, sums):
            decay = jnp.exp2(origin - gates(j * SUB))
            gated = tile(k_ref, j * SUB).astype(jnp.float32) * decay
            return (sums[0] + _dot(scores(daqk_ref, first, j * SUB), gated, exact=exact, solve=k_ref.dtype),
                    sums[1] + _dot(scores(dakk_ref, first, j * SUB), gated, exact=exact, solve=k_ref.dtype))

        empty = jnp.zeros((SUB, width), jnp.float32)
        dq2, dk2 = jax.lax.fori_loop(0, i, before, (empty, empty))
        ahead = jnp.exp2(gate - origin)
        dq2, dk2 = dq2 * ahead, dk2 * ahead
        def diagonal(j, sums):
            token = first + j
            decay = jnp.exp2(gate - decayed(token))
            causal = (index >= j)[:, None]
            own = row(k_ref, token)[None, :] * decay
            column = pl.ds(token, 1)
            qk = read_rows(_rows(daqk_ref, start + first, SUB, column), left(first))
            kk = read_rows(_rows(dakk_ref, start + first, SUB, column), left(first))
            return (sums[0] + jnp.where(causal, qk * own, 0.0), sums[1] + jnp.where(causal, kk * own, 0.0))

        dq2, dk2 = jax.lax.fori_loop(0, own_tokens, diagonal, (dq2, dk2))
        beta_grad = jnp.sum(dk2 * key, axis=1)
        dk2 = dk2 * beta[:, None]
        dg2 = tile(q_ref, first).astype(jnp.float32) * dq2
        write_rows(_rows(dq2_ref, start + first, SUB), left(first), dq2 + tile(dq_ref, first))
        closing = decayed(first + own_tokens - 1)

        def after(j, dkt):
            behind = jnp.exp2(gates(j * SUB) - closing)
            # k beta in the operands' dtype, as fla's b_kb, then widened by the decay.
            weighted = (tile(k_ref, j * SUB) * tile(beta_ref, j * SUB)[:, None]).astype(jnp.float32)
            dkt = dkt + _dot(scores(daqk_ref, j * SUB, first).T,
                             tile(q_ref, j * SUB).astype(jnp.float32) * behind, exact=exact,
                             solve=k_ref.dtype)
            return dkt + _dot(scores(dakk_ref, j * SUB, first).T, weighted * behind, exact=exact,
                              solve=k_ref.dtype)

        dkt = jax.lax.fori_loop(i + 1, count, after, empty) * jnp.exp2(closing - gate)

        def own(j, dkt):
            token = first + j
            query = row(q_ref, token)
            weighted = row(k_ref, token) * row(beta_ref, token)
            decay = jnp.exp2(decayed(token) - gate)
            causal = (index <= j)[:, None]
            qk = daqk_ref[start + token, pl.ds(first, SUB)]
            kk = dakk_ref[start + token, pl.ds(first, SUB)]
            dkt = dkt + jnp.where(causal, qk[:, None] * query[None, :] * decay, 0.0)
            return dkt + jnp.where(causal, kk[:, None] * weighted[None, :] * decay, 0.0)

        dkt = jax.lax.fori_loop(0, own_tokens, own, dkt)
        if k_ref.dtype == jnp.float32:
            dg2 = _paired_decay(q_ref, k_ref, g_ref, beta_ref, daqk_ref, dakk_ref, start, first,
                                aligned_tokens, gate, tile(q_ref, first).astype(jnp.float32), key, beta,
                                left(first))
        else:
            dg2 = dg2 + (dk2 - dkt) * key
        dg2 = dg2 + tile(dg_ref, first)
        write_rows(_rows(dk2_ref, start + first, SUB), left(first), dk2 + tile(dk_ref, first) + dkt)
        write_rows(_rows(dg2_ref, start + first, SUB), left(first), dg2)
        write_rows(_rows(db2_ref, start + first, SUB), left(first), beta_grad)


def _paired_decay(q_ref, k_ref, g_ref, beta_ref, daqk_ref, dakk_ref, start, first, tokens, gate, query, key,
                  beta, taken):
    """A sub-chunk's rows' decay gradient within their chunk, each pair's own term, fp32: for each of the
    chunk's `tokens` `s` and each row `t` from `first`, the term `(dAqk[t, s] q_t + dAkk[t, s] beta_t k_t)
    k_s exp2(g_t - g_s)` through the range `(s, t]` adds to `t`'s gradient where `s < t`, and the pair
    `(s, t)`'s subtracts where `s > t`. `gate`, `query`, `key` and `beta` are the rows' cumulated decay and
    operands, `taken` the rows a read takes."""
    rows = first + jnp.arange(SUB)
    weighted = beta[:, None] * key

    def token(ref, s):
        return ref[start + s].astype(jnp.float32)

    def paired(s, total):
        g_s, k_s = _row(g_ref, start + s)[None, :], token(k_ref, s)[None, :]
        column = pl.ds(s, 1)
        qk = read_rows(_rows(daqk_ref, start + first, SUB, column), taken).astype(jnp.float32)
        kk = read_rows(_rows(dakk_ref, start + first, SUB, column), taken).astype(jnp.float32)
        toward = (qk * query + kk * weighted) * k_s * jnp.exp2(gate - g_s)
        qk_s = daqk_ref[start + s, pl.ds(first, SUB)].astype(jnp.float32)[:, None]
        kk_s = dakk_ref[start + s, pl.ds(first, SUB)].astype(jnp.float32)[:, None]
        back = ((qk_s * token(q_ref, s)[None, :] + kk_s * (token(k_ref, s) * token(beta_ref, s))[None, :])
                * key * jnp.exp2(g_s - gate))
        return (total + jnp.where((rows > s)[:, None], toward, 0.0)
                - jnp.where((rows < s)[:, None], back, 0.0))

    return jax.lax.fori_loop(0, tokens, paired, jnp.zeros(gate.shape, jnp.float32))


@dataclasses.dataclass(frozen=True)
class BwdIntra(_Launch):
    """`bwd_intra`'s launch (tuning.KernelOp)."""

    NAME = "bwd_intra"
    KEYS, VALUES = "k", None
    # Measured on an A100-40GB at Kimi Linear's 1 x 4096 x 32 x 128 bf16: 2.49 -> 1.98 ms.
    PICKS = _picks(Tiles(64, 0, 4, 1), small=Tiles(32, 0, 4, 1))

    def _fwd(self, q: jax.Array, k: jax.Array, g: jax.Array, beta: jax.Array,
             daqk: jax.Array, dakk: jax.Array, dq: jax.Array, dk: jax.Array, db: jax.Array, dg: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, key_heads, width = k.shape
        value_heads = beta.shape[2]
        group, tokens, count = value_heads // key_heads, batch * length, chunk // SUB
        bk = min(config.bk, width)
        blocks = width // bk
        keys = pl.BlockSpec((tokens, None, bk), lambda i_k, i_s, h: (0, h // group, i_k))
        headed = pl.BlockSpec((tokens, None, bk), lambda i_k, i_s, h: (0, h, i_k))
        gate = pl.BlockSpec((tokens, None), lambda i_k, i_s, h: (0, h))
        scores = pl.BlockSpec((tokens, None, chunk), lambda i_k, i_s, h: (0, h, 0))
        wide = jax.ShapeDtypeStruct((tokens, value_heads, width), jnp.float32)
        dq, dk, partial, dg = pl.pallas_call(
            functools.partial(_bwd_intra_kernel, chunk=chunk, aligned=_aligned(spans, batch, length, chunk),
                              exact=interpret),
            grid=(blocks, spans.chunks.shape[0] * count, value_heads),
            in_specs=[pl.BlockSpec((None, 4), lambda i_k, i_s, h: (i_s // count, 0)), keys, keys, headed,
                      gate, scores, scores, headed, headed, headed],
            out_specs=[headed, headed, pl.BlockSpec((None, tokens, None), lambda i_k, i_s, h: (i_k, 0, h)),
                       headed],
            out_shape=[wide, wide, jax.ShapeDtypeStruct((blocks, tokens, value_heads), jnp.float32), wide],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_bwd_intra",
        )(spans.chunks, *_flat(q, k, g, beta, daqk, dakk, dq, dk, dg))
        keyed = (batch, length, value_heads, width)
        return (dq.reshape(keyed), dk.reshape(keyed), db + jnp.sum(partial, axis=0).reshape(beta.shape),
                dg.reshape(keyed)), None


_BWD_INTRA = BwdIntra()


def bwd_intra(q, k, g, beta, daqk, dakk, dq, dk, db, dg, *, chunk: int = 64,
              spans: Sequences | None = None, interpret: bool = False):
    """`(dq, dk, db, dg)` fp32, `bwd_wy_dqkg`'s plus the intra-chunk terms of `dAqk` and `dAkk`: fla's
    `chunk_kda_bwd_intra` (chunk_intra.py:897-960), its grid over key blocks, sub-chunks and heads,
    beta's gradient summed over the key blocks after. `dq`, `dk`, `dg` `[B, T, HV, K]`, `db` `[B, T,
    HV]`."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _BWD_INTRA(q, k, g, beta, daqk, dakk, dq, dk, db, dg, *spans, chunk=chunk,
                      interpret=interpret)


def _rule(q, k, v, g, beta, h0, cu_seqlens, scale: float, chunk: int, interpret: bool, keep: bool = True,
          wide: bool = False):
    return _rule_fwd(q, k, v, g, beta, h0, cu_seqlens, scale, chunk, interpret, keep, wide)[0]


def _rule_fwd(q, k, v, g, beta, h0, cu_seqlens, scale: float, chunk: int, interpret: bool, keep: bool = True,
              wide: bool = False):
    """fla's `chunk_kda_fwd` (fla/ops/kda/chunk_fwd.py:20-134): the decay cumulated in base 2, the
    intra-chunk scores and inverse, `w`, `u` and the gated keys, the memory across chunks
    (`fla_gdn.chunk_fwd_h`'s USE_GK branch), then the output. Saves what fla saves, and where `keep`
    its memory: `w`, the gated queries and keys, the memory each chunk enters and the corrected
    values, which fla recomputes in its backward ('delta_memory', as `fla_gdn._rule_fwd` keeps it)."""
    return fla_gdn.prefixed_fwd(cumulated_fwd, q, k, v, g, beta, h0, cu_seqlens, scale, chunk, interpret,
                                keep, wide, compensated=True)


def cumulated_fwd(q, k, v, g, beta, h0, spans: Sequences, scale: float, chunk: int, interpret: bool,
                  keep: bool = True, wide: bool = False):
    """`_rule_fwd` from the decay cumulated in base 2 within each chunk (`recurrence.cumulated`), its
    `(hi, lo)` halves `g`, which is all the rest of the rule reads of it, as `fla_gdn.cumulated_fwd`: a
    node of the port's graph and the port's arithmetic on it. Every decay takes both halves: its pairs'
    (`intra_rows`), its ranges' and its absolute ones (`_w_u_kernel`, `_o_kernel`, the memory's)."""
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    g, g_lo = g
    aqk, akk = intra_solve(q, k, g, beta, scale=scale, g_lo=g_lo, **kernels)
    akk = checkpoint_name(akk, 'delta_inverse')
    w, u, kg, qg = recompute_w_u(k, v, beta, akk, g, q if keep else None, g_lo=g_lo, **kernels)
    h, corrected, final = fla_gdn.chunk_fwd_h(kg, w, u, g, h0, g_lo=g_lo, **kernels)
    o = chunk_fwd_o(q, corrected, g, aqk, h, scale=scale, g_lo=g_lo, wide=wide, **kernels)
    memory = checkpoint_name((w, kg, qg, h, corrected), 'delta_memory') if keep else None
    return (o, final), (q, k, v, (g, g_lo), beta, aqk, akk, h0, memory, spans)


def first_decay_grad(dg, h0, dh0, spans: Sequences):
    """`dg` `[B, T, HV, K]`, the decay's gradient summed back over each chunk, with each sequence's
    first token's taken from the memory its decay scales: `<h0, dh0>` over the values for a row's
    first sequence, and exactly zero for any other, which starts from no memory. Summed back, that
    token's gradient is every term of its chunk, which cancel to fp32's residue where the memory is
    zero (2.7e-11 on a zero initial memory, as a float64 recurrence and the portable math hold
    exactly). The gated delta rule's port sums each token's own terms instead
    (`fla_gdn._summed_back`), which give it exactly."""
    batch, length = dg.shape[:2]
    first, count = spans.spans[:, 0], spans.spans[:, 1]
    shared = jnp.sum(h0.astype(jnp.float32) * dh0, axis=-1)
    value = jnp.where((first % length == 0)[:, None, None], shared[first // length], 0.0)
    flat = dg.reshape(batch * length, *dg.shape[2:])
    index = jnp.where(count > 0, first, batch * length)
    return flat.at[index].set(value.astype(dg.dtype), mode='drop').reshape(dg.shape)


def _rule_bwd(scale: float, chunk: int, interpret: bool, keep: bool, wide: bool, residual, cotangent):
    """fla's `chunk_kda_bwd` (fla/ops/kda/chunk_bwd.py:434-615): `w`, the gated queries and keys, the
    memory and corrected values the forward kept, or recomputed as fla does, then `dAqk` and the values'
    local gradient, the reverse recurrence
    (`fla_gdn.chunk_bwd_dhu`'s USE_GK branch), the fused WY and memory gradients, the intra-chunk
    terms, a key head's value heads summed (fla's GVA), and the decay's gradient summed back over
    each chunk, each sequence's first token's from its memory (`first_decay_grad`). A `wide` output's
    cotangent is taken in the values' dtype (`fla_gdn._rule_bwd`)."""
    q, k, v, (g, g_lo), beta, aqk, akk, h0, memory, spans = residual
    do, dht = cotangent
    do = do.astype(v.dtype) if wide else do
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    if keep:
        w, kg, qg, h, corrected = memory
    else:
        w, u, kg, qg = recompute_w_u(k, v, beta, akk, g, q, g_lo=g_lo, **kernels)
        h, corrected, _ = fla_gdn.chunk_fwd_h(kg, w, u, g, h0, g_lo=g_lo, **kernels)
    daqk, dv = bwd_dav(corrected, do, aqk, scale=scale, **kernels)
    dh, dh0, dv = fla_gdn.chunk_bwd_dhu(qg, kg, w, g, dht.astype(jnp.float32), do, dv, scale=scale,
                                        g_lo=g_lo, **kernels)
    dq, dk, dv, db, dg, dakk = bwd_wy_dqkg(q, k, v, corrected, g, beta, akk, h, do, dh, dv, g_lo=g_lo,
                                           scale=scale, **kernels)
    dq, dk, db, dg = bwd_intra(q, k, g, beta, daqk, dakk, dq, dk, db, dg, **kernels)
    group = v.shape[2] // k.shape[2]
    if group > 1:
        dq, dk = (x.reshape(*x.shape[:2], k.shape[2], group, x.shape[-1]).sum(axis=3) for x in (dq, dk))
    dg = first_decay_grad(recurrence.cumulated(dg, reverse=True, **kernels), h0, dh0, spans)
    return (dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), dg, db.astype(beta.dtype), dh0, None)


chunk_kda = jax.custom_vjp(_rule, nondiff_argnums=(7, 8, 9, 10, 11))
"""fla's chunked Kimi delta attention on the ported kernels: `(o, final)` for `q`, `k` `[B, T, H, K]`
(normalised as the caller wants them), `v` `[B, T, HV, V]`, the per-token natural log decay `g` `[B,
T, HV, K]` fp32, `beta` `[B, T, HV]` and each row's initial memory `h0` `[B, HV, K, V]` fp32, the
final one each row's. `cu_seqlens` (`[N + 1]` int32 over the `B * T` tokens, fla's varlen) splits the
rows into sequences, the memory starting from zero at each one a row does not begin with; None runs
each row as one. Its numbers are fla's: bf16 operands store `A`, `w`, `u`, the gated keys and the
memory each chunk enters in bf16, with fp32 accumulation, fp32 decay and the memory fp32 across
chunks; the output in the values' dtype, or fp32 where `wide`, and the final memory fp32."""
chunk_kda.defvjp(_rule_fwd, _rule_bwd)
