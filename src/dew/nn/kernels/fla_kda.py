# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Translated from flash-linear-attention v0.5.2 (fla/ops/kda, fla/ops/gla, fla/ops/utils), which is
# licensed under the MIT license: see licenses/flash-linear-attention-LICENSE.

"""fla's chunked Kimi delta attention, ported kernel for kernel to Pallas reading Dew's [B, T, H, D].

The port follows fla-org/flash-linear-attention v0.5.2 (fla/ops/kda/chunk.py) on the path Dew's layer
takes: the queries and keys normalised before the rule, the log decay per key dimension computed
before it, beta given, fla's varlen sequences, 64-token chunks and fla's default `safe_gate=False`. It
is fla's gated delta rule (`fla_gdn`) with a decay per key dimension, and shares that port's memory
kernels, whose USE_GK branch it takes (`fla_gdn.chunk_fwd_h`, `fla_gdn.chunk_bwd_dhu`), its chunk-local
cumsum, its forward substitution, its dot helpers, its token spans (`fla_gdn.Sequences`) and its
launches' base (`fla_gdn._Launch`, a tuner KernelOp); the kernels here are KDA's own. Each names its fla
origin. `linear.chunk_gated_delta_rule` runs it for a decay per key dimension.

fla's grids are kept: a program per place of each chunk for the intra-chunk diagonal blocks (fla's
per token, which searches the token's sequence), per chunk for the solve, per key block and
sub-chunk for the backward intra-chunk gradients.

Operands keep their dtypes as fla's do: bf16 queries, keys and values store `A`, `w`, `u`, the
gated keys and the memory each chunk enters in bf16 with fp32 accumulation; the decay is fp32 and
cumulated in base 2 as compensated halves (`fla_gdn.chunk_local_cumsum`), every range's exponent
taken from both (`compensated_difference`); gradients of q, k, the decay and beta accumulate in fp32.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels import fla_gdn
from dew.nn.kernels.fla_gdn import (
    _CHUNK,
    RCP_LN2,
    Sequences,
    Tiles,
    _aligned,
    _chunk_of,
    _chunk_states,
    _dot,
    _flat,
    _gates,
    _halves,
    _headed,
    _Launch,
    _launch,
    _read,
    _row,
    _rows,
    _tf32_dot,
    _tiles,
    _tril_inverse,
    _weighted_values,
    _write,
    sequences,
)
from dew.nn.precision import compensated_difference

SUB = 16
"""fla's BC: the sub-chunk the intra-chunk kernels build their 16 x 16 blocks over."""


def _space(kernel: str, wide: bool) -> frozenset[Tiles]:
    """`kernel`'s 16-bit tiles to measure, fla's own autotune lists for it (chunk_intra.py:33-39,
    wy_fast.py:247-256, gla/chunk.py:335-344, chunk_bwd.py:331), narrower value blocks where a GPU's
    block holds 99 KB."""
    return {
        # fla autotunes BH over 1, 2, 4, 8 and warps over 1, 2, 4, 8 (chunk_intra_token_parallel.py:13-20).
        "intra_diagonal": frozenset(Tiles(0, 0, warps, 1, bh) for bh in (1, 2, 4, 8)
                                    for warps in (1, 2, 4, 8)),
        "intra_solve": _tiles([(32, 0), (64, 0)], (1, 2, 4), (1,)),
        "recompute_w_u": _tiles([(64, 64)], (2, 4), (2, 3, 4)),
        "chunk_fwd_o": _tiles([(32, 64), (64, 64)] + ([(32, 128), (64, 128)] if wide else []), (2, 4, 8),
                              (2, 3, 4) if wide else (2,)),
        "bwd_dav": _tiles([(0, 64)], (2, 4, 8), (2, 3, 4) if wide else (2,)),
        "bwd_wy_dqkg": _tiles([(32, 32), (32, 64)] if wide else [(32, 32)], (2, 4), (1,)),
        "bwd_intra": _tiles([(32, 0), (64, 0)], (2, 4, 8), (1,)),
    }[kernel]


_A100 = {
    "intra_diagonal": Tiles(0, 0, 4, 1, 4),
    # Measured on an A100-40GB at Kimi Linear's 1 x 4096 x 32 x 128 bf16: the solve 0.98 -> 0.80 ms,
    # the backward intra kernel 2.49 -> 1.98 ms, the output 0.25 -> 0.23 ms.
    "intra_solve": Tiles(32, 0, 1, 1),
    "recompute_w_u": Tiles(64, 64, 4, 2),
    "chunk_fwd_o": Tiles(64, 128, 4, 2),
    "bwd_dav": Tiles(0, 64, 4, 2),
    "bwd_wy_dqkg": Tiles(32, 64, 4, 1),
    "bwd_intra": Tiles(64, 0, 4, 1),
}
_SMALL = {**_A100, "intra_solve": Tiles(32, 0, 2, 1), "chunk_fwd_o": Tiles(64, 64, 4, 2),
          "bwd_wy_dqkg": Tiles(32, 32, 4, 1), "bwd_intra": Tiles(32, 0, 4, 1)}
HEURISTICS = {True: _A100, False: _SMALL}
"""Each launch's tiles without a measurement, by whether the GPU's block holds an A100's shared
memory: an A100's measured choices, fla's narrower ones for 99 KB devices, for every dtype."""


class _KdaLaunch(_Launch):
    """One of KDA's own launches (`fla_gdn._Launch`), over its own tiles (`_space`, `HEURISTICS`)."""

    def heuristic(self, dtype: str, wide: bool) -> Tiles:
        return HEURISTICS[wide][self.NAME]

    def space(self, wide: bool) -> frozenset[Tiles]:
        return _space(self.NAME, wide)


def _intra_diagonal_kernel(chunk_ref, q_ref, k_ref, g_ref, lo_ref, beta_ref, aqk_ref, akk_ref, *,
                           chunk: int, aligned: bool, scale: float):
    """fla's `chunk_kda_fwd_kernel_intra_token_parallel` (chunk_intra_token_parallel.py:33-118) for one
    token and `BH` value heads: its row of its 16-token sub-chunk's diagonal block, `Aqk_ij = scale
    sum_k q_ik k_jk exp2(g_ik - g_jk)` for `j <= i` and `Akk_ij = sum_k beta_i k_ik k_jk exp2(g_ik -
    g_jk)` for `j < i`, in fp32, the exponents from the decay's two halves
    (`compensated_difference`); `Aqk` stored in its dtype and `Akk` fp32. A program per place of a
    chunk, where fla's searches the token's sequence; a place past the chunk's end does nothing."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    place = pl.program_id(0) % chunk

    @pl.when(place < taken)
    def _():
        row = place % SUB
        token = start + place
        first = token - row
        index = jnp.arange(SUB)
        query = q_ref[token].astype(jnp.float32)
        weighted = k_ref[token].astype(jnp.float32) * beta_ref[token].astype(jnp.float32)[:, None]
        gate = _row(g_ref, lo_ref, token)

        def column(j, sums):
            earlier = first + j
            decay = jnp.exp2(compensated_difference(gate, _row(g_ref, lo_ref, earlier)))
            gated = k_ref[earlier].astype(jnp.float32) * decay
            here = (index == j)[None, :]
            return (jnp.where(here, (jnp.sum(query * gated, axis=1) * scale)[:, None], sums[0]),
                    jnp.where(here & (j < row), jnp.sum(weighted * gated, axis=1)[:, None], sums[1]))

        empty = jnp.zeros((query.shape[0], SUB), jnp.float32)
        qk, kk = jax.lax.fori_loop(0, row + 1, column, (empty, empty))
        aqk_ref[token] = qk.astype(aqk_ref.dtype)
        akk_ref[token] = kk


@dataclasses.dataclass(frozen=True)
class IntraDiagonal(_KdaLaunch):
    """`intra_diagonal`'s launch (tuning.KernelOp): its value heads a program, warps and stages."""

    NAME = "intra_diagonal"
    KEYS, VALUES = "k", None

    def _tiles(self, ba, tiles: Tiles) -> Tiles:
        """A block of value heads that divides them, one where value heads share a key head, as fla's
        ceil(HV / BH) grid masks its tail."""
        value_heads, key_heads = ba.arguments["beta"].shape[2], ba.arguments["k"].shape[2]
        bh = math.gcd(tiles.bh, value_heads) if value_heads == key_heads else 1
        return dataclasses.replace(tiles, bh=bh)

    def _fwd(self, q: jax.Array, k: jax.Array, g: jax.Array, g_lo: jax.Array, beta: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, scale: float, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, key_heads, width = k.shape
        value_heads = beta.shape[2]
        group, bh, tokens = value_heads // key_heads, config.bh, batch * length
        keys = pl.BlockSpec((tokens, bh, width), lambda p, b: (0, b // group, 0))
        out = pl.BlockSpec((tokens, bh, SUB), lambda p, b: (0, b, 0))
        found = pl.pallas_call(
            functools.partial(_intra_diagonal_kernel, chunk=chunk,
                              aligned=_aligned(spans, batch, length, chunk), scale=scale),
            grid=(spans.chunks.shape[0] * chunk, value_heads // bh),
            in_specs=[pl.BlockSpec((None, 2), lambda p, b: (p // chunk, 0)), keys, keys,
                      pl.BlockSpec((tokens, bh, width), lambda p, b: (0, b, 0)),
                      pl.BlockSpec((tokens, bh, width), lambda p, b: (0, b, 0)),
                      pl.BlockSpec((tokens, bh), lambda p, b: (0, b))],
            out_specs=[out, out],
            out_shape=[jax.ShapeDtypeStruct((tokens, value_heads, SUB), k.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads, SUB), jnp.float32)],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_intra_diagonal",
        )(spans.chunks, *_flat(q, k, g, g_lo, beta))
        return tuple(x.reshape(batch, length, value_heads, SUB) for x in found), None


_INTRA_DIAGONAL = IntraDiagonal()


def intra_diagonal(q, k, g, g_lo, beta, *, scale: float, chunk: int = 64, spans: Sequences | None = None,
                   bh: int | None = None, warps: int | None = None, stages: int | None = None,
                   interpret: bool = False):
    """`(Aqk, Akk)` `[B, T, HV, 16]`, each token's row of its sub-chunk's diagonal block, `Aqk` in `k`'s
    dtype and `Akk` fp32: fla's `chunk_kda_fwd_intra_token_parallel` (chunk_intra_token_parallel.py:121-
    180), a program per place of each chunk (`spans`, the rows' own where None) and `BH` value heads."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _launch(_INTRA_DIAGONAL, {"bh": bh, "warps": warps, "stages": stages},
                   q, k, g, g_lo, beta, *spans, chunk=chunk, scale=scale, interpret=interpret)


def _intra_solve_kernel(chunk_ref, q_ref, k_ref, g_ref, lo_ref, beta_ref, aqkd_ref, akkd_ref, aqk_ref,
                        akk_ref, *, chunk: int, aligned: bool, scale: float, bk: int):
    """fla's `chunk_kda_fwd_kernel_inter_solve_fused` (chunk_intra.py:43-391), safe_gate off, for one
    chunk of one head, in 16-row blocks, the key dimensions in blocks of `bk`. The diagonal blocks are
    `intra_diagonal`'s. A block below the diagonal, rows `i` and columns `j` of sub-chunks `I > J`,
    factors the decay through `g_n`, the first row of `I`: `(x_I exp2(g_I - g_n)) (k_J exp2(g_n -
    g_J))^T`, each exponent from the decay's two halves (`compensated_difference`), fp32 products at
    fla's TF32 (`_tf32_dot`), then scaled by `scale` or by beta. `k_J exp2(g_n - g_J)` is held from
    one `I` to the next and moved on by the row between their first rows, `exp2(g_n' - g_n)`, so
    each earlier sub-chunk's keys are decayed once, where fla decays them again for every `I`. The
    diagonal blocks of `I + Akk` invert by forward substitution and merge in fla's order, `X_IJ =
    -(X_II A_IJ) X_JJ` for neighbours and `-X_II (sum_M A_IM X_MJ)` beyond, at TF32. `Aqk` and the
    inverse are stored in the keys' dtype, zero above the diagonal; a chunk's rows past its end read
    as zero keys under its last row's decay, and none is stored."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        width = k_ref.shape[-1]
        count = chunk // SUB
        index = jnp.arange(SUB)
        zero = jnp.zeros((SUB, SUB), jnp.float32)
        pairs = [(i, j) for i in range(count) for j in range(i)]

        def part(ref, i, *rest):
            return _read(_rows(ref, start + i * SUB, SUB, *rest), taken - i * SUB)

        def block(c, sums):
            """One block of key columns for every pair in fla's order: each row's sub-chunk loaded after
            the rows before it are summed, and the earlier sub-chunks' keys held, decayed to its first
            row."""
            columns = pl.ds(c * bk, bk)

            def load(ref, i):
                return part(ref, i, columns).astype(jnp.float32)

            def gates(i):
                return _halves(g_ref, lo_ref, start, taken, SUB, i * SUB, columns)

            def origin(i):
                """The decay's halves at sub-chunk `i`'s first row, or the chunk's last past its end."""
                at = jnp.minimum(start + i * SUB, start + taken - 1)
                return tuple(x[None, :] for x in _row(g_ref, lo_ref, (at, columns)))

            found = dict(zip(pairs, sums, strict=True))
            key, gate, held, before = load(k_ref, 0), gates(0), [], None
            for i in range(1, count):
                # The earlier sub-chunks' keys, decayed to this sub-chunk's first row.
                here = origin(i)
                if before is not None:
                    moved = jnp.exp2(compensated_difference(here, before))
                    held = [earlier * moved for earlier in held]
                held.append(key * jnp.exp2(compensated_difference(here, gate)))
                key, gate, before = load(k_ref, i), gates(i), here
                ahead = jnp.exp2(compensated_difference(gate, here))
                read, written = load(q_ref, i) * ahead, key * ahead
                for j, earlier in enumerate(held):
                    qk, kk = found[i, j]
                    found[i, j] = (qk + _tf32_dot(read, earlier.T), kk + _tf32_dot(written, earlier.T))
            return [found[pair] for pair in pairs]

        found = jax.lax.fori_loop(0, width // bk, block, [(zero, zero)] * len(pairs))

        def store(ref, i, j, value):
            _write(_rows(ref, start + i * SUB, SUB, pl.ds(j * SUB, SUB)), taken - i * SUB, value)

        # Aqk is stored before the solve, so none of it is held through the solve.
        kks = {}
        for (i, j), (qk, kk) in zip(pairs, found, strict=True):
            store(aqk_ref, i, j, qk * scale)
            store(aqk_ref, j, i, zero)
            kks[i, j] = kk * part(beta_ref, i).astype(jnp.float32)[:, None]
        for i in range(count):
            store(aqk_ref, i, i, part(aqkd_ref, i))
            kks[i, i] = part(akkd_ref, i)
        inverse = {(i, i): _tril_inverse(kks[i, i], index) for i in range(count)}
        for distance in range(1, count):
            for i in range(distance, count):
                j = i - distance
                if distance == 1:
                    inverse[i, j] = -_tf32_dot(_tf32_dot(inverse[i, i], kks[i, j]), inverse[j, j])
                else:
                    inner = sum(_tf32_dot(kks[i, m], inverse[m, j]) for m in range(j, i))
                    inverse[i, j] = -_tf32_dot(inverse[i, i], inner)
        for i in range(count):
            for j in range(count):
                store(akk_ref, i, j, inverse[i, j] if j <= i else zero)


@dataclasses.dataclass(frozen=True)
class IntraSolve(_KdaLaunch):
    """`intra_solve`'s launch (tuning.KernelOp)."""

    NAME = "intra_solve"
    KEYS, VALUES = "k", None

    def _fwd(self, q: jax.Array, k: jax.Array, g: jax.Array, g_lo: jax.Array, beta: jax.Array,
             aqkd: jax.Array, akkd: jax.Array, chunk_spans: jax.Array, sequence_spans: jax.Array, *,
             chunk: int, scale: float, interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, key_heads, width = k.shape
        value_heads = beta.shape[2]
        group, tokens = value_heads // key_heads, batch * length
        keys = pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0))

        headed = functools.partial(_headed, tokens)

        shape = jax.ShapeDtypeStruct((tokens, value_heads, chunk), k.dtype)
        found = pl.pallas_call(
            functools.partial(_intra_solve_kernel, chunk=chunk, aligned=_aligned(spans, batch, length, chunk),
                              scale=scale, bk=min(config.bk, width)),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, keys, keys, headed(width), headed(width),
                      pl.BlockSpec((tokens, None), lambda i_t, h: (0, h)), headed(SUB), headed(SUB)],
            out_specs=[headed(chunk), headed(chunk)], out_shape=[shape, shape],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_intra_solve",
        )(spans.chunks, *_flat(q, k, g, g_lo, beta, aqkd, akkd))
        return tuple(x.reshape(batch, length, value_heads, chunk) for x in found), None


_INTRA_SOLVE = IntraSolve()


def intra_solve(q, k, g, g_lo, beta, *, scale: float, chunk: int = 64, spans: Sequences | None = None,
                bk: int | None = None, warps: int | None = None, stages: int | None = None,
                interpret: bool = False):
    """`(Aqk, Akk)` `[B, T, HV, BT]` in `k`'s dtype: each chunk's `scale (q exp2(g)) (k exp2(-g))^T`,
    causal, and the inverse `(I + tril(beta (k exp2(g)) (k exp2(-g))^T, -1))^-1`: fla's
    `chunk_kda_fwd_intra` before its `recompute_w_u_fwd` (chunk_intra.py:794-894): the diagonal blocks
    token by token (`intra_diagonal`), the rest and the solve chunk by chunk (`spans`, the rows' own
    where None). `q`, `k` `[B, T, H, K]`, `g` and `g_lo` the base-2 cumulated decay's halves
    (`fla_gdn.chunk_local_cumsum`) `[B, T, HV, K]` fp32, `beta` `[B, T, HV]`."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    aqkd, akkd = intra_diagonal(q, k, g, g_lo, beta, scale=scale, chunk=chunk, spans=spans,
                                interpret=interpret)
    return _launch(_INTRA_SOLVE, {"bk": bk, "warps": warps, "stages": stages},
                   q, k, g, g_lo, beta, aqkd, akkd, *spans, chunk=chunk, scale=scale, interpret=interpret)


def _w_u_kernel(chunk_ref, k_ref, v_ref, beta_ref, a_ref, g_ref, lo_ref, *refs, chunk: int, aligned: bool,
                bk: int, bv: int, queries: bool, exact: bool):
    """fla's `recompute_w_u_fwd_kda_kernel` (fla/ops/kda/wy_fast.py:34-140) for one chunk of one head:
    `u = A (v beta)` and `w = A ((k beta) exp2(g))`, each operand rounded to its input dtype before the
    product; the gated keys `k exp2(g_last - g)`, the exponent from the decay's two halves
    (`compensated_difference`); with `queries`, the gated queries `q exp2(g)`."""
    if queries:
        q_ref, w_ref, u_ref, kg_ref, qg_ref = refs
    else:
        (w_ref, u_ref, kg_ref), q_ref, qg_ref = refs, None, None
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return _read(_rows(ref, start, chunk, *rest), taken)

        def write(ref, columns, value):
            _write(_rows(ref, start, chunk, columns), taken, value)

        beta = read(beta_ref)
        inverse = read(a_ref)
        _weighted_values(v_ref, u_ref, beta, inverse, start, taken, chunk=chunk, bv=bv, exact=exact,
                         ieee=False)
        for block in range(k_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            key = read(k_ref, columns)
            gate = _halves(g_ref, lo_ref, start, taken, chunk, 0, columns)
            scaled = (key * beta[:, None]) * jnp.exp2(gate[0])
            if q_ref is not None and qg_ref is not None:
                write(qg_ref, columns, read(q_ref, columns) * jnp.exp2(gate[0]))
            last = tuple(x[None, :] for x in _row(g_ref, lo_ref, (start + taken - 1, columns)))
            write(kg_ref, columns, key * jnp.exp2(compensated_difference(last, gate)))
            write(w_ref, columns, _dot(inverse, scaled.astype(key.dtype), exact=exact))


@dataclasses.dataclass(frozen=True)
class RecomputeWU(_KdaLaunch):
    """`recompute_w_u`'s launch (tuning.KernelOp)."""

    NAME = "recompute_w_u"
    KEYS, VALUES = "k", "v"

    def _fwd(self, k: jax.Array, v: jax.Array, beta: jax.Array, a: jax.Array, g: jax.Array,
             g_lo: jax.Array, q: jax.Array | None, chunk_spans: jax.Array, sequence_spans: jax.Array, *,
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
        )(spans.chunks, *_flat(k, v, beta, a, g, g_lo, *([q] if queries else [])))
        w, u, kg, *qg = found
        shaped = (w.reshape(batch, length, value_heads, width), u.reshape(v.shape),
                  kg.reshape(batch, length, value_heads, width))
        return (*shaped, qg[0].reshape(batch, length, value_heads, width) if queries else None), None


_RECOMPUTE_W_U = RecomputeWU()


def recompute_w_u(k, v, beta, a, g, g_lo, q=None, *, chunk: int = 64, spans: Sequences | None = None,
                  bk: int | None = None, bv: int | None = None, warps: int | None = None,
                  stages: int | None = None, interpret: bool = False):
    """`(w, u, kg, qg)`: fla's `recompute_w_u_fwd` for KDA (wy_fast.py:264-311). `w`, `kg` and `qg`
    `[B, T, HV, K]` in `k`'s dtype, `u` in `v`'s; `qg` only for given queries `q` `[B, T, H, K]`, else
    None. `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `beta` `[B, T, HV]`, `a` the inverse per chunk `[B,
    T, HV, BT]`, `g` and `g_lo` the base-2 cumulated decay's halves `[B, T, HV, K]` fp32."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _launch(_RECOMPUTE_W_U, {"bk": bk, "bv": bv, "warps": warps, "stages": stages},
                   k, v, beta, a, g, g_lo, q, *spans, chunk=chunk, interpret=interpret)


def _o_kernel(chunk_ref, q_ref, v_ref, g_ref, h_ref, a_ref, o_ref, *, chunk: int, aligned: bool, scale: float,
              bk: int, exact: bool):
    """fla's `chunk_gla_fwd_kernel_o` (fla/ops/gla/chunk.py:345-432) for one chunk of one head and `BV`
    value columns: `scale (q exp2(g))_narrow h + tril(Aqk) v`, the gated queries rounded to their
    dtype and `Aqk` to the values'."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return _read(_rows(ref, start, chunk, *rest), taken)

        out = jnp.zeros((chunk, v_ref.shape[-1]), jnp.float32)
        for block in range(q_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            query = read(q_ref, columns)
            gated = (query * jnp.exp2(_gates(g_ref, start, taken, chunk, 0, columns))).astype(query.dtype)
            out = out + _dot(gated, h_ref[columns, :].astype(gated.dtype), exact=exact)
        out = out * scale
        index = jnp.arange(chunk)
        value = read(v_ref)
        scores = jnp.where(index[:, None] >= index[None, :], read(a_ref), 0.0).astype(value.dtype)
        _write(_rows(o_ref, start, chunk), taken, out + _dot(scores, value, exact=exact))


@dataclasses.dataclass(frozen=True)
class ChunkFwdO(_KdaLaunch):
    """KDA's `chunk_fwd_o`'s launch (tuning.KernelOp)."""

    NAME = "chunk_fwd_o"
    KEYS, VALUES = "q", "v"

    def _fwd(self, q: jax.Array, v: jax.Array, g: jax.Array, a: jax.Array, h: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, scale: float, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, key_heads, width = q.shape
        value_heads, columns = v.shape[2:]
        group, tokens = value_heads // key_heads, batch * length
        bv = min(config.bv, columns)
        values = pl.BlockSpec((tokens, None, bv), lambda i_v, i_t, i_h: (0, i_h, i_v))
        out = pl.pallas_call(
            functools.partial(_o_kernel, chunk=chunk, aligned=_aligned(spans, batch, length, chunk),
                              scale=scale, bk=min(config.bk, width), exact=interpret),
            grid=(columns // bv, spans.chunks.shape[0], value_heads),
            in_specs=[pl.BlockSpec((None, 2), lambda i_v, i_t, i_h: (i_t, 0)),
                      pl.BlockSpec((tokens, None, width), lambda i_v, i_t, i_h: (0, i_h // group, 0)), values,
                      pl.BlockSpec((tokens, None, width), lambda i_v, i_t, i_h: (0, i_h, 0)),
                      pl.BlockSpec((None, None, width, bv), lambda i_v, i_t, i_h: (i_t, i_h, 0, i_v)),
                      pl.BlockSpec((tokens, None, chunk), lambda i_v, i_t, i_h: (0, i_h, 0))],
            out_specs=values, out_shape=jax.ShapeDtypeStruct((tokens, value_heads, columns), v.dtype),
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_chunk_fwd_o",
        )(spans.chunks, *_flat(q, v, g), _chunk_states(h), *_flat(a))
        return out.reshape(v.shape), None


_CHUNK_FWD_O = ChunkFwdO()


def chunk_fwd_o(q, v, g, a, h, *, scale: float, chunk: int = 64, spans: Sequences | None = None,
                bk: int | None = None, bv: int | None = None, warps: int | None = None,
                stages: int | None = None, interpret: bool = False):
    """The output `[B, T, HV, V]` in `v`'s dtype: fla's `chunk_gla_fwd_o_gk` (gla/chunk.py:962-997). `q`
    `[B, T, H, K]`, `v` the corrected values `[B, T, HV, V]`, `g` the base-2 cumulated decay `[B, T, HV,
    K]`, `a` `Aqk` `[B, T, HV, BT]`, `h` the memory each chunk enters (`fla_gdn.chunk_fwd_h`'s)."""
    spans = sequences(*q.shape[:2], chunk=chunk) if spans is None else spans
    return _launch(_CHUNK_FWD_O, {"bk": bk, "bv": bv, "warps": warps, "stages": stages},
                   q, v, g, a, h, *spans, chunk=chunk, scale=scale, interpret=interpret)


def _dav_kernel(chunk_ref, v_ref, do_ref, a_ref, dv_ref, da_ref, *, chunk: int, aligned: bool, scale: float,
                bv: int, exact: bool):
    """fla's `chunk_kda_bwd_kernel_dAv` (fla/ops/kda/chunk_bwd.py:42-129) for one chunk of one head:
    `dAqk = scale tril(do v^T)` in fp32 and `dv = tril(Aqk)^T do`, `Aqk` in the gradient's dtype."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return _read(_rows(ref, start, chunk, *rest), taken)

        index = jnp.arange(chunk)
        out_grad_dtype = do_ref.dtype
        transposed = jnp.where(index[:, None] <= index[None, :], read(a_ref).T, 0.0).astype(out_grad_dtype)
        scores = jnp.zeros((chunk, chunk), jnp.float32)
        for block in range(v_ref.shape[-1] // bv):
            columns = pl.ds(block * bv, bv)
            out_grad = read(do_ref, columns)
            scores = scores + _dot(out_grad, read(v_ref, columns).T, exact=exact)
            _write(_rows(dv_ref, start, chunk, columns), taken, _dot(transposed, out_grad, exact=exact))
        causal = index[:, None] >= index[None, :]
        _write(_rows(da_ref, start, chunk), taken, jnp.where(causal, scores * scale, 0.0))


@dataclasses.dataclass(frozen=True)
class BwdDav(_KdaLaunch):
    """`bwd_dav`'s launch (tuning.KernelOp)."""

    NAME = "bwd_dav"
    KEYS, VALUES = None, "v"

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


def bwd_dav(v, do, a, *, scale: float, chunk: int = 64, spans: Sequences | None = None, bv: int | None = None,
            warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`(dAqk, dv)`: fla's `chunk_kda_bwd_dAv` (chunk_bwd.py:314-365), `dAqk` `[B, T, HV, BT]` fp32 and
    `dv` `[B, T, HV, V]` in `do`'s dtype. `v` the corrected values, `do` the output's gradient, `a`
    `Aqk`."""
    spans = sequences(*v.shape[:2], chunk=chunk) if spans is None else spans
    return _launch(_BWD_DAV, {"bv": bv, "warps": warps, "stages": stages},
                   v, do, a, *spans, chunk=chunk, scale=scale, interpret=interpret)


def _wy_dqkg_kernel(chunk_ref, q_ref, k_ref, v_ref, vn_ref, g_ref, lo_ref, beta_ref, a_ref, h_ref, do_ref,
                    dh_ref, dv_ref, dq_ref, dk_ref, dv2_ref, dg_ref, db_ref, da_ref, *, chunk: int,
                    aligned: bool, scale: float, bk: int, bv: int, exact: bool):
    """fla's `chunk_kda_bwd_kernel_wy_dqkg_fused` (fla/ops/kda/chunk_bwd.py:132-311) for one chunk of
    one head: the queries', keys', decay's and beta's gradients through the output's memory term and
    the WY representation, the values' through `A`, and the gradient of `Akk`, with fla's casts and
    sums in its order. `A` is read transposed, as fla loads it; fla's first key block's value terms
    run first, on their own loop."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return _read(_rows(ref, start, chunk, *rest), taken)

        def write(ref, value, *rest):
            _write(_rows(ref, start, chunk, *rest), taken, value)

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
                memory, memory_grad = h_ref[keys, values], dh_ref[keys, values]
                return (dq + _dot(out_grad, memory.T.astype(out_grad.dtype), exact=exact),
                        dk + _dot(corrected, memory_grad.T.astype(corrected.dtype), exact=exact),
                        dw + _dot(read(dv_ref, values).astype(corrected.dtype),
                                  memory.T.astype(corrected.dtype), exact=exact),
                        decay_grad + jnp.sum((memory * memory_grad).astype(jnp.float32), axis=1))

            empty = jnp.zeros((chunk, bk), jnp.float32)
            dq, dk, dw, decay_grad = jax.lax.fori_loop(0, columns // bv, value_block,
                                                       (empty, empty, empty, jnp.zeros((bk,), jnp.float32)))
            key = read(k_ref, keys)
            gate = _halves(g_ref, lo_ref, start, taken, chunk, 0, keys)
            final = _row(g_ref, lo_ref, (start + taken - 1, keys))
            gate_exp = jnp.exp2(gate[0])
            decay_grad = decay_grad * jnp.exp2(final[0])
            dq = dq * gate_exp * scale
            dk = dk * jnp.exp2(compensated_difference(tuple(x[None, :] for x in final), gate))
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
class BwdWyDqkg(_KdaLaunch):
    """`bwd_wy_dqkg`'s launch (tuning.KernelOp)."""

    NAME = "bwd_wy_dqkg"
    KEYS, VALUES = "k", "v"

    def _fwd(self, q: jax.Array, k: jax.Array, v: jax.Array, corrected: jax.Array, g: jax.Array,
             g_lo: jax.Array, beta: jax.Array, a: jax.Array, h: jax.Array, do: jax.Array, dh: jax.Array,
             dv: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, scale: float, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        batch, length, key_heads, width = k.shape
        value_heads, columns = v.shape[2:]
        group, tokens = value_heads // key_heads, batch * length
        keys = pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0))

        headed = functools.partial(_headed, tokens)

        gate = pl.BlockSpec((tokens, None), lambda i_t, h: (0, h))
        states = pl.BlockSpec((None, None, width, columns), lambda i_t, h: (i_t, h, 0, 0))
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
        )(spans.chunks, *_flat(q, k, v, corrected, g, g_lo, beta, a), _chunk_states(h), *_flat(do),
          _chunk_states(dh), *_flat(dv))
        keyed = (batch, length, value_heads, width)
        return (dq.reshape(keyed), dk.reshape(keyed), dv2.reshape(v.shape), db.reshape(beta.shape),
                dg.reshape(keyed), da.reshape(batch, length, value_heads, chunk)), None


_BWD_WY_DQKG = BwdWyDqkg()


def bwd_wy_dqkg(q, k, v, corrected, g, g_lo, beta, a, h, do, dh, dv, *, scale: float, chunk: int = 64,
                spans: Sequences | None = None, bk: int | None = None, bv: int | None = None,
                warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`(dq, dk, dv, db, dg, dAkk)`: fla's `chunk_kda_bwd_wy_dqkg_fused` (chunk_bwd.py:367-431), `dq`,
    `dk`, `dg` `[B, T, HV, K]`, `db` `[B, T, HV]` and `dAkk` `[B, T, HV, BT]` fp32, `dv` in `v`'s dtype.
    `h` and `dh` the memory each chunk enters and its gradient (`fla_gdn.chunk_fwd_h`'s and
    `chunk_bwd_dhu`'s), `dv` the corrected values' full gradient, `a` `Akk`'s inverse."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _launch(_BWD_WY_DQKG, {"bk": bk, "bv": bv, "warps": warps, "stages": stages},
                   q, k, v, corrected, g, g_lo, beta, a, h, do, dh, dv, *spans, chunk=chunk, scale=scale,
                   interpret=interpret)


def _bwd_intra_kernel(chunk_ref, q_ref, k_ref, g_ref, lo_ref, beta_ref, daqk_ref, dakk_ref, dq_ref, dk_ref,
                      dg_ref, dq2_ref, dk2_ref, db2_ref, dg2_ref, *, chunk: int, aligned: bool):
    """fla's `chunk_kda_bwd_kernel_intra` (fla/ops/kda/chunk_intra.py:395-679), safe_gate off, for one
    sub-chunk `i` of a chunk, one head and one block of key columns, as fla's grid runs it. The
    gradients of `Aqk` and `Akk` reach the sub-chunk's queries and keys from the sub-chunks before
    (`dq2`, `dk2`, the decay factored at its first row) and after (`dkt`, factored at its last row),
    every exponent from the decay's two halves (`compensated_difference`),
    fp32 products at fla's TF32 (`_tf32_dot`), and from its own tokens one by one in fp32; then beta's
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
            return _read(_rows(ref, start + at, SUB), left(at))

        def gates(at):
            return _halves(g_ref, lo_ref, start + at, left(at), SUB)

        def scores(ref, row, column):
            return _read(_rows(ref, start + row, SUB, pl.ds(column, SUB)), left(row))

        def row(ref, at):
            return ref[start + at].astype(jnp.float32)

        def decayed(at):
            """The decay's halves at the chunk's row `at`, `[1, K]` each."""
            return tuple(x[None, :] for x in _row(g_ref, lo_ref, start + at))

        gate = gates(first)
        key = tile(k_ref, first).astype(jnp.float32)
        beta = tile(beta_ref, first).astype(jnp.float32)
        origin = decayed(first)

        def before(j, sums):
            decay = jnp.exp2(compensated_difference(origin, gates(j * SUB)))
            gated = tile(k_ref, j * SUB).astype(jnp.float32) * decay
            return (sums[0] + _tf32_dot(scores(daqk_ref, first, j * SUB), gated),
                    sums[1] + _tf32_dot(scores(dakk_ref, first, j * SUB), gated))

        empty = jnp.zeros((SUB, width), jnp.float32)
        dq2, dk2 = jax.lax.fori_loop(0, i, before, (empty, empty))
        ahead = jnp.exp2(compensated_difference(gate, origin))
        dq2, dk2 = dq2 * ahead, dk2 * ahead
        def diagonal(j, sums):
            token = first + j
            decay = jnp.exp2(compensated_difference(gate, decayed(token)))
            causal = (index >= j)[:, None]
            own = row(k_ref, token)[None, :] * decay
            column = pl.ds(token, 1)
            qk = _read(_rows(daqk_ref, start + first, SUB, column), left(first))
            kk = _read(_rows(dakk_ref, start + first, SUB, column), left(first))
            return (sums[0] + jnp.where(causal, qk * own, 0.0), sums[1] + jnp.where(causal, kk * own, 0.0))

        dq2, dk2 = jax.lax.fori_loop(0, own_tokens, diagonal, (dq2, dk2))
        beta_grad = jnp.sum(dk2 * key, axis=1)
        dk2 = dk2 * beta[:, None]
        dg2 = tile(q_ref, first).astype(jnp.float32) * dq2
        _write(_rows(dq2_ref, start + first, SUB), left(first), dq2 + tile(dq_ref, first))
        closing = decayed(first + own_tokens - 1)

        def after(j, dkt):
            behind = jnp.exp2(compensated_difference(gates(j * SUB), closing))
            # k beta in the operands' dtype, as fla's b_kb, then widened by the decay.
            weighted = (tile(k_ref, j * SUB) * tile(beta_ref, j * SUB)[:, None]).astype(jnp.float32)
            dkt = dkt + _tf32_dot(scores(daqk_ref, j * SUB, first).T,
                                  tile(q_ref, j * SUB).astype(jnp.float32) * behind)
            return dkt + _tf32_dot(scores(dakk_ref, j * SUB, first).T, weighted * behind)

        dkt = jax.lax.fori_loop(i + 1, count, after, empty) * jnp.exp2(compensated_difference(closing, gate))

        def own(j, dkt):
            token = first + j
            query = row(q_ref, token)
            weighted = row(k_ref, token) * row(beta_ref, token)
            decay = jnp.exp2(compensated_difference(decayed(token), gate))
            causal = (index <= j)[:, None]
            qk = daqk_ref[start + token, pl.ds(first, SUB)]
            kk = dakk_ref[start + token, pl.ds(first, SUB)]
            dkt = dkt + jnp.where(causal, qk[:, None] * query[None, :] * decay, 0.0)
            return dkt + jnp.where(causal, kk[:, None] * weighted[None, :] * decay, 0.0)

        dkt = jax.lax.fori_loop(0, own_tokens, own, dkt)
        dg2 = dg2 + (dk2 - dkt) * key + tile(dg_ref, first)
        _write(_rows(dk2_ref, start + first, SUB), left(first), dk2 + tile(dk_ref, first) + dkt)
        _write(_rows(dg2_ref, start + first, SUB), left(first), dg2)
        _write(_rows(db2_ref, start + first, SUB), left(first), beta_grad)


@dataclasses.dataclass(frozen=True)
class BwdIntra(_KdaLaunch):
    """`bwd_intra`'s launch (tuning.KernelOp)."""

    NAME = "bwd_intra"
    KEYS, VALUES = "k", None

    def _fwd(self, q: jax.Array, k: jax.Array, g: jax.Array, g_lo: jax.Array, beta: jax.Array,
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
            functools.partial(_bwd_intra_kernel, chunk=chunk, aligned=_aligned(spans, batch, length, chunk)),
            grid=(blocks, spans.chunks.shape[0] * count, value_heads),
            in_specs=[pl.BlockSpec((None, 2), lambda i_k, i_s, h: (i_s // count, 0)), keys, keys, headed,
                      headed, gate, scores, scores, headed, headed, headed],
            out_specs=[headed, headed, pl.BlockSpec((None, tokens, None), lambda i_k, i_s, h: (i_k, 0, h)),
                       headed],
            out_shape=[wide, wide, jax.ShapeDtypeStruct((blocks, tokens, value_heads), jnp.float32), wide],
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=config.stages),
            interpret=interpret, name="fla_kda_bwd_intra",
        )(spans.chunks, *_flat(q, k, g, g_lo, beta, daqk, dakk, dq, dk, dg))
        keyed = (batch, length, value_heads, width)
        return (dq.reshape(keyed), dk.reshape(keyed), db + jnp.sum(partial, axis=0).reshape(beta.shape),
                dg.reshape(keyed)), None


_BWD_INTRA = BwdIntra()


def bwd_intra(q, k, g, g_lo, beta, daqk, dakk, dq, dk, db, dg, *, chunk: int = 64,
              spans: Sequences | None = None,
              bk: int | None = None, warps: int | None = None, stages: int | None = None,
              interpret: bool = False):
    """`(dq, dk, db, dg)` fp32, `bwd_wy_dqkg`'s plus the intra-chunk terms of `dAqk` and `dAkk`: fla's
    `chunk_kda_bwd_intra` (chunk_intra.py:897-960), its grid over key blocks, sub-chunks and heads,
    beta's gradient summed over the key blocks after. `dq`, `dk`, `dg` `[B, T, HV, K]`, `db` `[B, T,
    HV]`."""
    spans = sequences(*k.shape[:2], chunk=chunk) if spans is None else spans
    return _launch(_BWD_INTRA, {"bk": bk, "warps": warps, "stages": stages},
                   q, k, g, g_lo, beta, daqk, dakk, dq, dk, db, dg, *spans, chunk=chunk, interpret=interpret)


def _rule(q, k, v, g, beta, h0, cu_seqlens, scale: float, chunk: int, interpret: bool):
    return _rule_fwd(q, k, v, g, beta, h0, cu_seqlens, scale, chunk, interpret)[0]


def _rule_fwd(q, k, v, g, beta, h0, cu_seqlens, scale: float, chunk: int, interpret: bool):
    """fla's `chunk_kda_fwd` (fla/ops/kda/chunk_fwd.py:20-134): the decay cumulated in base 2, the
    intra-chunk scores and inverse, `w`, `u` and the gated keys, the memory across chunks
    (`fla_gdn.chunk_fwd_h`'s USE_GK branch), then the output. Saves what fla saves."""
    spans = sequences(q.shape[0], q.shape[1], cu_seqlens, chunk=chunk)
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    g_hi, g_lo = fla_gdn.chunk_local_cumsum(g, scale=RCP_LN2, **kernels)
    aqk, akk = intra_solve(q, k, g_hi, g_lo, beta, scale=scale, **kernels)
    w, u, kg, _ = recompute_w_u(k, v, beta, akk, g_hi, g_lo, **kernels)
    h, corrected, final = fla_gdn.chunk_fwd_h(kg, w, u, g_hi, g_lo, h0, **kernels)
    o = chunk_fwd_o(q, corrected, g_hi, aqk, h, scale=scale, **kernels)
    return (o, final), (q, k, v, g_hi, g_lo, beta, aqk, akk, h0, spans)


def _rule_bwd(scale: float, chunk: int, interpret: bool, residual, cotangent):
    """fla's `chunk_kda_bwd` (fla/ops/kda/chunk_bwd.py:434-615): `w`, `u`, the gated queries and keys
    and the memory recomputed, then `dAqk` and the values' local gradient, the reverse recurrence
    (`fla_gdn.chunk_bwd_dhu`'s USE_GK branch), the fused WY and memory gradients, the intra-chunk
    terms, a key head's value heads summed (fla's GVA), and the decay's gradient summed back over
    each chunk."""
    q, k, v, g_hi, g_lo, beta, aqk, akk, h0, spans = residual
    do, dht = cotangent
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    w, u, kg, qg = recompute_w_u(k, v, beta, akk, g_hi, g_lo, q, **kernels)
    h, corrected, _ = fla_gdn.chunk_fwd_h(kg, w, u, g_hi, g_lo, h0, **kernels)
    daqk, dv = bwd_dav(corrected, do, aqk, scale=scale, **kernels)
    dh, dh0, dv = fla_gdn.chunk_bwd_dhu(qg, kg, w, g_hi, g_lo, dht.astype(jnp.float32), do, dv, scale=scale,
                                        **kernels)
    dq, dk, dv, db, dg, dakk = bwd_wy_dqkg(q, k, v, corrected, g_hi, g_lo, beta, akk, h, do, dh, dv,
                                           scale=scale, **kernels)
    dq, dk, db, dg = bwd_intra(q, k, g_hi, g_lo, beta, daqk, dakk, dq, dk, db, dg, **kernels)
    group = v.shape[2] // k.shape[2]
    if group > 1:
        dq, dk = (x.reshape(*x.shape[:2], k.shape[2], group, x.shape[-1]).sum(axis=3) for x in (dq, dk))
    dg = fla_gdn.chunk_local_cumsum(dg, reverse=True, **kernels)
    return (dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), dg, db.astype(beta.dtype), dh0, None)


chunk_kda = jax.custom_vjp(_rule, nondiff_argnums=(7, 8, 9))
"""fla's chunked Kimi delta attention on the ported kernels: `(o, final)` for `q`, `k` `[B, T, H, K]`
(normalised as the caller wants them), `v` `[B, T, HV, V]`, the per-token natural log decay `g` `[B,
T, HV, K]` fp32, `beta` `[B, T, HV]` and each row's initial memory `h0` `[B, HV, K, V]` fp32, the
final one each row's. `cu_seqlens` (`[N + 1]` int32 over the `B * T` tokens, fla's varlen) splits the
rows into sequences, the memory starting from zero at each one a row does not begin with; None runs
each row as one. Its numbers are fla's: bf16 operands store `A`, `w`, `u`, the gated keys and the
memory each chunk enters in bf16, with fp32 accumulation, fp32 decay and the memory fp32 across
chunks; the output in the values' dtype and the final memory fp32."""
chunk_kda.defvjp(_rule_fwd, _rule_bwd)
