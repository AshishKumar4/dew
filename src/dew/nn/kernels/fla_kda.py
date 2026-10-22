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
and the chunked launches' token spans (`Sequences`) and base (`chunked_launch._Launch`, a tuner
KernelOp); the kernels here are KDA's own. Each names its fla origin.
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
    _halves,
    _headed,
    _Launch,
    _picks,
    _row,
    _rows,
    _spaces,
    _tiles,
    read_rows,
    sequences,
    write_rows,
)
from dew.nn.kernels.fla_gdn import _dot, _issuer, _weighted_values

SUB = 16
"""fla's BC: the sub-chunk the intra-chunk kernels build their 16 x 16 blocks over."""

DECAY_FLOOR = -5.0
"""The steepest per-token log decay the port takes, Kimi Linear's bound (`dew.nn.kda`'s `lower_bound`):
its backward factors each block below a sub-chunk's diagonal through the sub-chunk's first row
(`_bwd_intra_kernel`), `exp2(+-SUB g log2(e))`, past fp32's range beyond -5.5 a token; at -30 its decay
gradient sat 5.0e4 times the token-by-token recurrence's error from float64. A
decay with no declared floor at or above it runs the portable math (`recurrence_chunk.port_refusal`)."""


def _intra_rows_kernel(chunk_ref, q_ref, k_ref, g_ref, gl_ref, beta_ref, aqk_ref, akk_ref, *,
                       chunk: int, aligned: bool, scale: float):
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
    past the chunk's end does nothing."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    place = pl.program_id(0) % chunk

    @pl.when(place < taken)
    def _():
        token = start + place
        query = q_ref[token].astype(jnp.float32)
        weighted = k_ref[token].astype(jnp.float32) * beta_ref[token].astype(jnp.float32)[:, None]

        def earlier(j):
            return _row(g_ref, start + j), _row(gl_ref, start + j), k_ref[start + j].astype(jnp.float32)

        qk, kk = algebra.key_decayed_row(query, weighted, _row(g_ref, token), _row(gl_ref, token), earlier,
                                         place, chunk, scale)
        aqk_ref[token] = qk.astype(aqk_ref.dtype)
        akk_ref[token] = kk


@dataclasses.dataclass(frozen=True)
class IntraRows(_Launch):
    """`intra_rows`'s launch (tuning.KernelOp): its value heads a program, warps and stages."""

    NAME = "intra_rows"
    KEYS, VALUES = "k", None
    # fla autotunes BH over 1, 2, 4, 8 and warps over 1, 2, 4, 8 (chunk_intra_token_parallel.py:13-20).
    revision: int = 2
    SPACES = _spaces(frozenset(Tiles(0, 0, warps, 1, bh) for bh in (1, 2, 4, 8) for warps in (1, 2, 4, 8)))
    PICKS = _picks(Tiles(0, 0, 4, 1, 4))

    def _tiles(self, ba, tiles: Tiles) -> Tiles:
        """A block of value heads that divides them, one where value heads share a key head, as fla's
        ceil(HV / BH) grid masks its tail."""
        value_heads, key_heads = ba.arguments["beta"].shape[2], ba.arguments["k"].shape[2]
        bh = math.gcd(tiles.bh, value_heads) if value_heads == key_heads else 1
        return dataclasses.replace(tiles, bh=bh)

    def _fwd(self, q: jax.Array, k: jax.Array, g: jax.Array, gl: jax.Array, beta: jax.Array,
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
                              aligned=_aligned(spans, batch, length, chunk), scale=scale),
            grid=(spans.chunks.shape[0] * chunk, value_heads // bh),
            in_specs=[pl.BlockSpec((None, 4), lambda p, b: (p // chunk, 0)), keys, keys,
                      pl.BlockSpec((tokens, bh, width), lambda p, b: (0, b, 0)),
                      pl.BlockSpec((tokens, bh, width), lambda p, b: (0, b, 0)),
                      pl.BlockSpec((tokens, bh), lambda p, b: (0, b))],
            out_specs=[out, out],
            out_shape=[jax.ShapeDtypeStruct((tokens, value_heads, chunk), k.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads, chunk), jnp.float32)],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_intra_rows",
        )(spans.chunks, *_flat(q, k, g, gl, beta))
        return tuple(x.reshape(batch, length, value_heads, chunk) for x in found), None


_INTRA_ROWS = IntraRows()


def intra_rows(q, k, g, beta, *, scale: float, chunk: int = 64, spans: Sequences | None = None,
               interpret: bool = False):
    """`(Aqk, Akk)` `[B, T, HV, BT]`, each token's row of its chunk, `Aqk` in `k`'s dtype and `Akk` fp32,
    every pair's decay its own (`_intra_rows_kernel`): a program per place of each chunk (`spans`, the
    rows' own where None) and `BH` value heads."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _INTRA_ROWS(q, k, *_halves(g), beta, *spans, chunk=chunk, scale=scale, interpret=interpret)


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
        inverse = algebra.inverted(kks, _issuer(exact, solve))
        for i in range(count):
            for j in range(count):
                store(akk_ref, i, j, inverse[i, j] if j <= i else zero)


@dataclasses.dataclass(frozen=True)
class IntraSolve(_Launch):
    """`intra_solve`'s launch (tuning.KernelOp)."""

    NAME = "intra_solve"
    KEYS, VALUES = "aqk", None
    SPACES = _spaces(_tiles([(0, 0)], (1, 2, 4), (1,)))
    PICKS = _picks(Tiles(0, 0, 1, 1), small=Tiles(0, 0, 2, 1))
    revision: int = 2

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


def intra_solve(q, k, g, beta, *, scale: float, chunk: int = 64, spans: Sequences | None = None,
                interpret: bool = False):
    """`(Aqk, Akk)` `[B, T, HV, BT]` in `k`'s dtype: each chunk's `scale (q exp2(g)) (k exp2(-g))^T`,
    causal, and the inverse `(I + tril(beta (k exp2(g)) (k exp2(-g))^T, -1))^-1`: fla's
    `chunk_kda_fwd_intra` before its `recompute_w_u_fwd` (chunk_intra.py:794-894), each pair's decay
    its own (`intra_rows`), then the solve chunk by chunk (`spans`, the rows' own where None). `q`, `k`
    `[B, T, H, K]`, `g` the base-2 cumulated decay (`recurrence.cumulated`) `[B, T, HV, K]` fp32,
    `beta` `[B, T, HV]`."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    aqk, akk = intra_rows(q, k, g, beta, scale=scale, chunk=chunk, spans=spans, interpret=interpret)
    return _INTRA_SOLVE(aqk, akk, *spans, chunk=chunk, interpret=interpret)


def _w_u_kernel(chunk_ref, k_ref, v_ref, beta_ref, a_ref, g_ref, gl_ref, *refs, chunk: int, aligned: bool,
                bk: int, bv: int, queries: bool, exact: bool):
    """fla's `recompute_w_u_fwd_kda_kernel` (fla/ops/kda/wy_fast.py:34-140) for one chunk of one head:
    `u = A (v beta)` and `w = A ((k beta) exp2(g))`, each operand rounded to its input dtype before the
    product; the gated keys `k exp2(g_last - g)`; with `queries`, the gated queries `q exp2(g)`."""
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
            if q_ref is not None and qg_ref is not None:
                write(qg_ref, columns, read(q_ref, columns) * jnp.exp2(gate))
            write(kg_ref, columns, algebra.key_decayed_write(
                key, gate, _gates(gl_ref, start, taken, chunk, 0, columns),
                _row(g_ref, (start + taken - 1, columns))[None, :],
                _row(gl_ref, (start + taken - 1, columns))[None, :]))
            write(w_ref, columns,
                  algebra.weighted("apply_w", inverse, key, beta, _issuer(exact), jnp.exp2(gate)))


@dataclasses.dataclass(frozen=True)
class RecomputeWU(_Launch):
    """`recompute_w_u`'s launch (tuning.KernelOp)."""

    NAME = "recompute_w_u"
    KEYS, VALUES = "k", "v"
    revision: int = 2
    SPACES = _spaces(_tiles([(64, 64)], (2, 4), (2, 3, 4)))
    PICKS = _picks(Tiles(64, 64, 4, 2))

    def _fwd(self, k: jax.Array, v: jax.Array, beta: jax.Array, a: jax.Array, g: jax.Array, gl: jax.Array,
             q: jax.Array | None, chunk_spans: jax.Array, sequence_spans: jax.Array, *,
             chunk: int, interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, key_heads, width = k.shape
        value_heads, columns = v.shape[2:]
        group, tokens = value_heads // key_heads, batch * length
        keys = pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0))

        headed = functools.partial(_headed, tokens)

        keyed = jax.ShapeDtypeStruct((tokens, value_heads, width), k.dtype)
        queries = q is not None
        found = pl.pallas_call(
            functools.partial(_w_u_kernel, chunk=chunk, aligned=_aligned(spans, batch, length, chunk),
                              bk=min(config.bk, width), bv=min(config.bv, columns), queries=queries,
                              exact=interpret),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, keys, headed(columns), pl.BlockSpec((tokens, None), lambda i_t, h: (0, h)),
                      headed(chunk), headed(width), headed(width), *([keys] if queries else [])],
            out_specs=[headed(width), headed(columns), headed(width), *([headed(width)] if queries else [])],
            out_shape=[keyed, jax.ShapeDtypeStruct((tokens, value_heads, columns), v.dtype), keyed,
                       *([keyed] if queries else [])],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_recompute_w_u",
        )(spans.chunks, *_flat(k, v, beta, a, g, gl, *([q] if queries else [])))
        w, u, kg, *qg = found
        shaped = (w.reshape(batch, length, value_heads, width), u.reshape(v.shape),
                  kg.reshape(batch, length, value_heads, width))
        return (*shaped, qg[0].reshape(batch, length, value_heads, width) if queries else None), None


_RECOMPUTE_W_U = RecomputeWU()


def recompute_w_u(k, v, beta, a, g, q=None, *, chunk: int = 64, spans: Sequences | None = None,
                  interpret: bool = False):
    """`(w, u, kg, qg)`: fla's `recompute_w_u_fwd` for KDA (wy_fast.py:264-311). `w`, `kg` and `qg`
    `[B, T, HV, K]` in `k`'s dtype, `u` in `v`'s; `qg` only for given queries `q` `[B, T, H, K]`, else
    None. `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `beta` `[B, T, HV]`, `a` the inverse per chunk `[B,
    T, HV, BT]`, `g` the base-2 cumulated decay `[B, T, HV, K]` fp32."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _RECOMPUTE_W_U(k, v, beta, a, *_halves(g), q, *spans, chunk=chunk, interpret=interpret)


def _o_kernel(chunk_ref, q_ref, v_ref, g_ref, h_ref, a_ref, o_ref, *, chunk: int, aligned: bool, scale: float,
              bk: int, exact: bool):
    """fla's `chunk_gla_fwd_kernel_o` (fla/ops/gla/chunk.py:345-432) for one chunk of one head and `BV`
    value columns: `scale (q exp2(g))_narrow h + tril(Aqk) v`, the gated queries rounded to their
    dtype and `Aqk` to the values', fp32 products IEEE as the gated delta rule's output kernel takes
    them (`fla_gdn._o_kernel`), the memory the chunk enters read at its slot (`Sequences`)."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    entered = chunk_ref[2]

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        def columns(c):
            return pl.ds(c * bk, bk)

        out = algebra.key_decayed_output(lambda c: read(q_ref, columns(c)),
                                         lambda c: _gates(g_ref, start, taken, chunk, 0, columns(c)),
                                         lambda c: h_ref[entered, columns(c), :], q_ref.shape[-1] // bk,
                                         read(a_ref), read(v_ref), scale, _issuer(exact))
        write_rows(_rows(o_ref, start, chunk), taken, out)


@dataclasses.dataclass(frozen=True)
class ChunkFwdO(_Launch):
    """KDA's `chunk_fwd_o`'s launch (tuning.KernelOp)."""

    NAME = "chunk_fwd_o"
    KEYS, VALUES = "q", "v"
    SPACES = _spaces(_tiles([(32, 64), (64, 64), (32, 128), (64, 128)], (2, 4, 8), (2, 3, 4)),
                     _tiles([(32, 64), (64, 64)], (2, 4, 8), (2,)))
    # Measured on an A100-40GB at Kimi Linear's 1 x 4096 x 32 x 128 bf16: 0.25 -> 0.23 ms.
    PICKS = _picks(Tiles(64, 128, 4, 2), small=Tiles(64, 64, 4, 2))

    def _fwd(self, q: jax.Array, v: jax.Array, g: jax.Array, a: jax.Array, h: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, scale: float, interpret: bool,
             config: Tiles, wide: bool = False, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, key_heads, width = q.shape
        value_heads, columns = v.shape[2:]
        group, tokens = value_heads // key_heads, batch * length
        bv = min(config.bv, columns)
        dtype = jnp.float32 if wide else v.dtype
        values = pl.BlockSpec((tokens, None, bv), lambda i_v, i_t, i_h: (0, i_h, i_v))
        out = pl.pallas_call(
            functools.partial(_o_kernel, chunk=chunk, aligned=_aligned(spans, batch, length, chunk),
                              scale=scale, bk=min(config.bk, width), exact=interpret),
            grid=(columns // bv, spans.chunks.shape[0], value_heads),
            in_specs=[pl.BlockSpec((None, 4), lambda i_v, i_t, i_h: (i_t, 0)),
                      pl.BlockSpec((tokens, None, width), lambda i_v, i_t, i_h: (0, i_h // group, 0)), values,
                      pl.BlockSpec((tokens, None, width), lambda i_v, i_t, i_h: (0, i_h, 0)),
                      pl.BlockSpec((h.shape[0], None, width, bv), lambda i_v, i_t, i_h: (0, i_h, 0, i_v)),
                      pl.BlockSpec((tokens, None, chunk), lambda i_v, i_t, i_h: (0, i_h, 0))],
            out_specs=values, out_shape=jax.ShapeDtypeStruct((tokens, value_heads, columns), dtype),
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_chunk_fwd_o",
        )(spans.chunks, *_flat(q, v, g), h, *_flat(a))
        return out.reshape(v.shape), None


_CHUNK_FWD_O = ChunkFwdO()


def chunk_fwd_o(q, v, g, a, h, *, scale: float, chunk: int = 64, spans: Sequences | None = None,
                interpret: bool = False, wide: bool = False):
    """The output `[B, T, HV, V]` in `v`'s dtype, fp32 where `wide` (`fla_gdn.chunk_fwd_o`'s): fla's
    `chunk_gla_fwd_o_gk` (gla/chunk.py:962-997). `q`
    `[B, T, H, K]`, `v` the corrected values `[B, T, HV, V]`, `g` the base-2 cumulated decay `[B, T, HV,
    K]`, `a` `Aqk` `[B, T, HV, BT]`, `h` the memories the chunks enter at their slots
    (`fla_gdn.chunk_fwd_h`'s)."""
    spans = sequences(*q.shape[:2], chunk=chunk) if spans is None else spans
    return _CHUNK_FWD_O(q, v, g, a, h, *spans, chunk=chunk, scale=scale, interpret=interpret, wide=wide)


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
    SPACES = _spaces(_tiles([(0, 64)], (2, 4, 8), (2, 3, 4)), _tiles([(0, 64)], (2, 4, 8), (2,)))
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


def _wy_dqkg_kernel(chunk_ref, q_ref, k_ref, v_ref, vn_ref, g_ref, gl_ref, beta_ref, a_ref, h_ref, do_ref,
                    dh_ref, dv_ref, dq_ref, dk_ref, dv2_ref, dg_ref, db_ref, da_ref, *, chunk: int,
                    aligned: bool, scale: float, bk: int, bv: int, exact: bool):
    """fla's `chunk_kda_bwd_kernel_wy_dqkg_fused` (fla/ops/kda/chunk_bwd.py:132-311) for one chunk of
    one head: the queries', keys', decay's and beta's gradients through the output's memory term and
    the WY representation, the values' through `A`, and the gradient of `Akk`, with fla's casts and
    sums in its order. `A` is read transposed, as fla loads it; fla's first key block's value terms
    run first, on their own loop. The memory the chunk enters and its gradient where it leaves are
    read at their slots (`Sequences`)."""
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
            final_low = _row(gl_ref, (start + taken - 1, keys))
            dk = dk * jnp.exp2((final[None, :] - gate)
                               + (final_low[None, :] - _gates(gl_ref, start, taken, chunk, 0, keys)))
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
    revision: int = 2
    KEYS, VALUES = "k", "v"
    SPACES = _spaces(_tiles([(32, 32), (32, 64)], (2, 4), (1,)), _tiles([(32, 32)], (2, 4), (1,)))
    PICKS = _picks(Tiles(32, 64, 4, 1), small=Tiles(32, 32, 4, 1))

    def _fwd(self, q: jax.Array, k: jax.Array, v: jax.Array, corrected: jax.Array, g: jax.Array,
             gl: jax.Array, beta: jax.Array, a: jax.Array, h: jax.Array, do: jax.Array, dh: jax.Array,
             dv: jax.Array, chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, scale: float,
             interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, key_heads, width = k.shape
        value_heads, columns = v.shape[2:]
        group, tokens = value_heads // key_heads, batch * length
        keys = pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0))

        headed = functools.partial(_headed, tokens)

        gate = pl.BlockSpec((tokens, None), lambda i_t, h: (0, h))
        states = pl.BlockSpec((h.shape[0], None, width, columns), lambda i_t, head: (0, head, 0, 0))
        wide = jax.ShapeDtypeStruct((tokens, value_heads, width), jnp.float32)
        dq, dk, dv2, dg, db, da = pl.pallas_call(
            functools.partial(_wy_dqkg_kernel, chunk=chunk, aligned=_aligned(spans, batch, length, chunk),
                              scale=scale, bk=min(config.bk, width), bv=min(config.bv, columns),
                              exact=interpret),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, keys, keys, headed(columns), headed(columns), headed(width), headed(width),
                      gate, headed(chunk), states, headed(columns), states, headed(columns)],
            out_specs=[headed(width), headed(width), headed(columns), headed(width), gate, headed(chunk)],
            out_shape=[wide, wide, jax.ShapeDtypeStruct((tokens, value_heads, columns), v.dtype), wide,
                       jax.ShapeDtypeStruct((tokens, value_heads), jnp.float32),
                       jax.ShapeDtypeStruct((tokens, value_heads, chunk), jnp.float32)],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_bwd_wy_dqkg",
        )(spans.chunks, *_flat(q, k, v, corrected, g, gl, beta, a), h, *_flat(do), dh, *_flat(dv))
        keyed = (batch, length, value_heads, width)
        return (dq.reshape(keyed), dk.reshape(keyed), dv2.reshape(v.shape), db.reshape(beta.shape),
                dg.reshape(keyed), da.reshape(batch, length, value_heads, chunk)), None


_BWD_WY_DQKG = BwdWyDqkg()


def bwd_wy_dqkg(q, k, v, corrected, g, beta, a, h, do, dh, dv, *, scale: float, chunk: int = 64,
                spans: Sequences | None = None, interpret: bool = False):
    """`(dq, dk, dv, db, dg, dAkk)`: fla's `chunk_kda_bwd_wy_dqkg_fused` (chunk_bwd.py:367-431), `dq`,
    `dk`, `dg` `[B, T, HV, K]`, `db` `[B, T, HV]` and `dAkk` `[B, T, HV, BT]` fp32, `dv` in `v`'s dtype.
    `h` and `dh` the memories the chunks enter and their gradients at their slots
    (`fla_gdn.chunk_fwd_h`'s and `chunk_bwd_dhu`'s), `dv` the corrected values' full gradient, `a`
    `Akk`'s inverse."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _BWD_WY_DQKG(q, k, v, corrected, *_halves(g), beta, a, h, do, dh, dv, *spans, chunk=chunk,
                        scale=scale, interpret=interpret)


def _bwd_intra_kernel(chunk_ref, q_ref, k_ref, g_ref, gl_ref, beta_ref, daqk_ref, dakk_ref, dq_ref, dk_ref,
                      dg_ref, dq2_ref, dk2_ref, db2_ref, dg2_ref, *, chunk: int, aligned: bool, exact: bool):
    """fla's `chunk_kda_bwd_kernel_intra` (fla/ops/kda/chunk_intra.py:395-679), safe_gate off, for one
    sub-chunk `i` of a chunk, one head and one block of key columns, as fla's grid runs it. The
    gradients of `Aqk` and `Akk` reach the sub-chunk's queries and keys from the sub-chunks before
    (`dq2`, `dk2`, the decay factored at its first row) and after (`dkt`, factored at its last row),
    fp32 products at the call's solve precision (`fla_gdn._dot`'s `solve`), and from its own tokens
    one by one in fp32; then beta's
    and the decay's gradients, added to the ones `bwd_wy_dqkg` gave. Beta's is this key block's share.
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

        def left(at):
            """The chunk's rows from `at`: a whole sub-chunk's where every chunk is whole."""
            return SUB if aligned else taken - at

        def tile(ref, at):
            return read_rows(_rows(ref, start + at, SUB), left(at))

        def gates(at):
            return _gates(g_ref, start + at, left(at), SUB)

        def lows(at):
            return _gates(gl_ref, start + at, left(at), SUB)

        def scores(ref, row, column):
            return read_rows(_rows(ref, start + row, SUB, pl.ds(column, SUB)), left(row)).astype(jnp.float32)

        def row(ref, at):
            return ref[start + at].astype(jnp.float32)

        def decayed(at):
            """The cumulated decay at the chunk's row `at`, `[1, K]`, and its low half."""
            return _row(g_ref, start + at)[None, :], _row(gl_ref, start + at)[None, :]

        def apart(rows, columns):
            """`rows - columns` over both halves of each, `(hi, lo)` pairs."""
            return (rows[0] - columns[0]) + (rows[1] - columns[1])

        gate = gates(first), lows(first)
        key = tile(k_ref, first).astype(jnp.float32)
        beta = tile(beta_ref, first).astype(jnp.float32)
        origin = decayed(first)

        def before(j, sums):
            decay = jnp.exp2(apart(origin, (gates(j * SUB), lows(j * SUB))))
            gated = tile(k_ref, j * SUB).astype(jnp.float32) * decay
            return (sums[0] + _dot(scores(daqk_ref, first, j * SUB), gated, exact=exact, solve=k_ref.dtype),
                    sums[1] + _dot(scores(dakk_ref, first, j * SUB), gated, exact=exact, solve=k_ref.dtype))

        empty = jnp.zeros((SUB, width), jnp.float32)
        dq2, dk2 = jax.lax.fori_loop(0, i, before, (empty, empty))
        ahead = jnp.exp2(apart(gate, origin))
        dq2, dk2 = dq2 * ahead, dk2 * ahead
        def diagonal(j, sums):
            token = first + j
            decay = jnp.exp2(apart(gate, decayed(token)))
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
            behind = jnp.exp2(apart((gates(j * SUB), lows(j * SUB)), closing))
            # k beta in the operands' dtype, as fla's b_kb, then widened by the decay.
            weighted = (tile(k_ref, j * SUB) * tile(beta_ref, j * SUB)[:, None]).astype(jnp.float32)
            dkt = dkt + _dot(scores(daqk_ref, j * SUB, first).T,
                             tile(q_ref, j * SUB).astype(jnp.float32) * behind, exact=exact,
                             solve=k_ref.dtype)
            return dkt + _dot(scores(dakk_ref, j * SUB, first).T, weighted * behind, exact=exact,
                              solve=k_ref.dtype)

        dkt = jax.lax.fori_loop(i + 1, count, after, empty) * jnp.exp2(apart(closing, gate))

        def own(j, dkt):
            token = first + j
            query = row(q_ref, token)
            weighted = row(k_ref, token) * row(beta_ref, token)
            decay = jnp.exp2(apart(decayed(token), gate))
            causal = (index <= j)[:, None]
            qk = daqk_ref[start + token, pl.ds(first, SUB)]
            kk = dakk_ref[start + token, pl.ds(first, SUB)]
            dkt = dkt + jnp.where(causal, qk[:, None] * query[None, :] * decay, 0.0)
            return dkt + jnp.where(causal, kk[:, None] * weighted[None, :] * decay, 0.0)

        dkt = jax.lax.fori_loop(0, own_tokens, own, dkt)
        dg2 = dg2 + (dk2 - dkt) * key + tile(dg_ref, first)
        write_rows(_rows(dk2_ref, start + first, SUB), left(first), dk2 + tile(dk_ref, first) + dkt)
        write_rows(_rows(dg2_ref, start + first, SUB), left(first), dg2)
        write_rows(_rows(db2_ref, start + first, SUB), left(first), beta_grad)


@dataclasses.dataclass(frozen=True)
class BwdIntra(_Launch):
    """`bwd_intra`'s launch (tuning.KernelOp)."""

    NAME = "bwd_intra"
    revision: int = 2
    KEYS, VALUES = "k", None
    SPACES = _spaces(_tiles([(32, 0), (64, 0)], (2, 4, 8), (1,)))
    # Measured on an A100-40GB at Kimi Linear's 1 x 4096 x 32 x 128 bf16: 2.49 -> 1.98 ms.
    PICKS = _picks(Tiles(64, 0, 4, 1), small=Tiles(32, 0, 4, 1))

    def _fwd(self, q: jax.Array, k: jax.Array, g: jax.Array, gl: jax.Array, beta: jax.Array,
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
                      headed, gate, scores, scores, headed, headed, headed],
            out_specs=[headed, headed, pl.BlockSpec((None, tokens, None), lambda i_k, i_s, h: (i_k, 0, h)),
                       headed],
            out_shape=[wide, wide, jax.ShapeDtypeStruct((blocks, tokens, value_heads), jnp.float32), wide],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_bwd_intra",
        )(spans.chunks, *_flat(q, k, g, gl, beta, daqk, dakk, dq, dk, dg))
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
    return _BWD_INTRA(q, k, *_halves(g), beta, daqk, dakk, dq, dk, db, dg, *spans, chunk=chunk,
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
                                keep, wide)


def cumulated_fwd(q, k, v, g, beta, h0, spans: Sequences, scale: float, chunk: int, interpret: bool,
                  keep: bool = True, wide: bool = False):
    """`_rule_fwd` from the decay cumulated in base 2 within each chunk (`recurrence.cumulated`),
    which is all the rest of the rule reads of it, as `fla_gdn.cumulated_fwd`: a node of the port's
    graph and the port's arithmetic on it."""
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    aqk, akk = intra_solve(q, k, g, beta, scale=scale, **kernels)
    akk = checkpoint_name(akk, 'delta_inverse')
    w, u, kg, qg = recompute_w_u(k, v, beta, akk, g, q if keep else None, **kernels)
    h, corrected, final = fla_gdn.chunk_fwd_h(kg, w, u, g, h0, **kernels)
    o = chunk_fwd_o(q, corrected, _halves(g)[0], aqk, h, scale=scale, wide=wide, **kernels)
    memory = checkpoint_name((w, kg, qg, h, corrected), 'delta_memory') if keep else None
    return (o, final), (q, k, v, g, beta, aqk, akk, h0, memory, spans)


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
    q, k, v, prefix, beta, aqk, akk, h0, memory, spans = residual
    do, dht = cotangent
    do = do.astype(v.dtype) if wide else do
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    if keep:
        w, kg, qg, h, corrected = memory
    else:
        w, u, kg, qg = recompute_w_u(k, v, beta, akk, prefix, q, **kernels)
        h, corrected, _ = fla_gdn.chunk_fwd_h(kg, w, u, prefix, h0, **kernels)
    daqk, dv = bwd_dav(corrected, do, aqk, scale=scale, **kernels)
    dh, dh0, dv = fla_gdn.chunk_bwd_dhu(qg, kg, w, prefix, dht.astype(jnp.float32), do, dv, scale=scale,
                                        **kernels)
    dq, dk, dv, db, dg, dakk = bwd_wy_dqkg(q, k, v, corrected, prefix, beta, akk, h, do, dh, dv, scale=scale,
                                           **kernels)
    dq, dk, db, dg = bwd_intra(q, k, prefix, beta, daqk, dakk, dq, dk, db, dg, **kernels)
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
