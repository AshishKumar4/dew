# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Translated from flash-linear-attention v0.5.2 (fla/ops/kda, fla/ops/gla, fla/ops/utils), which is
# licensed under the MIT license: see licenses/flash-linear-attention-LICENSE.

"""fla's chunked Kimi delta attention, ported kernel for kernel to Pallas reading Dew's [B, T, H, D].

The port follows fla-org/flash-linear-attention v0.5.2 (fla/ops/kda/chunk.py) on the path Dew's layer
takes: the queries and keys normalised before the rule, the log decay per key dimension computed
before it, beta given, no variable lengths, 64-token chunks and fla's default `safe_gate=False`. It
is fla's gated delta rule (`fla_gdn`) with a decay per key dimension, and shares that port's memory
kernels, whose USE_GK branch it takes (`fla_gdn.chunk_fwd_h`, `fla_gdn.chunk_bwd_dhu`), its chunk-local
cumsum, its forward substitution, its dot helpers and its config lookup; the kernels here are KDA's
own. Each names its fla origin. `linear.chunk_gated_delta_rule` runs it for a decay per key dimension.

fla's grids are kept: a program per token for the intra-chunk diagonal blocks, per chunk for the
solve, per key block and sub-chunk for the backward intra-chunk gradients.

Operands keep their dtypes as fla's do: bf16 queries, keys and values store `A`, `w`, `u`, the
gated keys and the memory each chunk enters in bf16 with fp32 accumulation; the decay is fp32 and
cumulated in base 2; gradients of q, k, the decay and beta accumulate in fp32.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels import fla_gdn, generation
from dew.nn.kernels.fla_gdn import RCP_LN2, _dot, _tf32_dot, _tril_inverse

SUB = 16
"""fla's BC: the sub-chunk the intra-chunk kernels build their 16 x 16 blocks over."""

_SM80 = {
    # fla autotunes BH over 1, 2, 4, 8 and warps over 1, 2, 4, 8 (chunk_intra_token_parallel.py:13-20).
    "intra_diagonal": {"bh": 4, "warps": 4, "stages": 1},
    # fla autotunes BK over 32, 64 and warps over 1, 2, 4 (chunk_intra.py:33-39).
    "intra_solve": {"bk": 32, "warps": 2, "stages": 1},
    # fla fixes BK = BV = 64 and autotunes warps over 2, 4 and stages over 2, 3, 4 (wy_fast.py:247-256).
    "recompute_w_u": {"bk": 64, "bv": 64, "warps": 4, "stages": 2},
    # fla autotunes BK 32/64, BV 64/128, warps 2/4/8, stages 2/3/4 (gla/chunk.py:335-344).
    "chunk_fwd_o": {"bk": 64, "bv": 64, "warps": 4, "stages": 2},
    # fla's tiling: BV the next power of two of the values' width up to 64, 128 on Hopper (chunk_bwd.py:331).
    "bwd_dav": {"bv": 64, "warps": 4, "stages": 2},
    "bwd_wy_dqkg": {"bk": 32, "bv": 64, "warps": 4, "stages": 1},
    # fla's BK = min(32, K) (chunk_intra.py:918).
    "bwd_intra": {"bk": 32, "warps": 4, "stages": 1},
}
_SMALL = {**_SM80, "bwd_wy_dqkg": {"bk": 32, "bv": 32, "warps": 4, "stages": 1}}

_A100 = {**_SM80,
         # Measured on an A100-40GB at Kimi Linear's 1 x 4096 x 32 x 128 bf16: the solve 0.98 -> 0.80 ms,
         # the backward intra kernel 2.49 -> 1.98 ms, the output 0.25 -> 0.23 ms.
         "intra_solve": {"bk": 32, "warps": 1, "stages": 1},
         "chunk_fwd_o": {"bk": 64, "bv": 128, "warps": 4, "stages": 2},
         "bwd_intra": {"bk": 64, "warps": 4, "stages": 1}}

_A100_SHARED = generation.TARGETS["sm80"].shared_limit or 0

CONFIGS = {(name, dtype): _A100 if name == "sm80" else
           _SM80 if (generation.TARGETS[name].shared_limit or 0) >= _A100_SHARED else _SMALL
           for name, dtype in fla_gdn.CONFIGS}
"""Each kernel's blocks, warps and stages, keyed as `fla_gdn.CONFIGS` and read through its
`kernel_config`: an A100's measured choices; fla's elsewhere wherever a block holds as much shared memory
as an A100's, narrower value blocks for the 99 KB ones."""

def kernel_config(kernel: str, width: int, dtype, *, bk: int | None = None, bv: int | None = None,
                  warps: int | None = None, stages: int | None = None) -> dict[str, int]:
    """`kernel`'s config from `CONFIGS` (`fla_gdn.kernel_config`), blocks clipped to `width`."""
    return fla_gdn.kernel_config(kernel, {"width": width, "columns": width}, dtype, table=CONFIGS,
                                 bk=bk, bv=bv, warps=warps, stages=stages)


def _params(config: Mapping[str, int]):
    from jax.experimental.pallas import triton as plgpu

    return plgpu.CompilerParams(num_warps=config["warps"], num_stages=config["stages"])


def _heads(value_heads: int, group: int):
    """Index maps over a `(chunk, batch * value_heads)` grid: a chunk of a key head's `[B, T, H, K]`,
    of a value head's `[B, T, HV, X]`, of its `[B, T, HV]`, and the chunk's memory `[B, NT, HV, K, V]`."""
    def keyed(i_t, bh):
        return bh // value_heads, i_t, (bh % value_heads) // group, 0

    def headed(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads, 0

    def gated(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads

    def memory(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads, 0, 0

    return keyed, headed, gated, memory


def _intra_diagonal_kernel(q_ref, k_ref, g_ref, beta_ref, aqk_ref, akk_ref, *, scale: float):
    """fla's `chunk_kda_fwd_kernel_intra_token_parallel` (chunk_intra_token_parallel.py:33-118) for one
    token and `BH` value heads: its row of its 16-token sub-chunk's diagonal block, `Aqk_ij = scale
    sum_k q_ik k_jk exp2(g_ik - g_jk)` for `j <= i` and `Akk_ij = sum_k beta_i k_ik k_jk exp2(g_ik -
    g_jk)` for `j < i`, in fp32; `Aqk` stored in its dtype and `Akk` fp32."""
    row = pl.program_id(0) % SUB
    index = jnp.arange(SUB)
    query = q_ref[row].astype(jnp.float32)
    weighted = k_ref[row].astype(jnp.float32) * beta_ref[row].astype(jnp.float32)[:, None]
    gate = g_ref[row].astype(jnp.float32)

    def column(j, sums):
        gated = k_ref[j].astype(jnp.float32) * jnp.exp2(gate - g_ref[j].astype(jnp.float32))
        here = (index == j)[None, :]
        return (jnp.where(here, (jnp.sum(query * gated, axis=1) * scale)[:, None], sums[0]),
                jnp.where(here & (j < row), jnp.sum(weighted * gated, axis=1)[:, None], sums[1]))

    empty = jnp.zeros((query.shape[0], SUB), jnp.float32)
    qk, kk = jax.lax.fori_loop(0, row + 1, column, (empty, empty))
    aqk_ref[...] = qk.astype(aqk_ref.dtype)
    akk_ref[...] = kk


def intra_diagonal(q, k, g, beta, *, scale: float, bh: int | None = None, warps: int | None = None,
                   stages: int | None = None, interpret: bool = False):
    """`(Aqk, Akk)` `[B, T, HV, 16]`, each token's row of its sub-chunk's diagonal block, `Aqk` in `k`'s
    dtype and `Akk` fp32: fla's `chunk_kda_fwd_intra_token_parallel` (chunk_intra_token_parallel.py:121-
    180), a program per token and `BH` value heads (one where value heads share a key head)."""
    config = kernel_config("intra_diagonal", k.shape[-1], k.dtype, warps=warps, stages=stages)
    batch, length, key_heads, width = k.shape
    value_heads = beta.shape[2]
    group = value_heads // key_heads
    # A block of value heads that divides them, as fla's ceil(HV / BH) grid masks its tail.
    bh = math.gcd(config["bh"] if bh is None else bh, value_heads) if group == 1 else 1
    blocks = value_heads // bh

    def keyed(i_t, b_h):
        return b_h // blocks, i_t // SUB, (b_h % blocks) // group, 0

    def headed(i_t, b_h):
        return b_h // blocks, i_t // SUB, b_h % blocks, 0

    def gated(i_t, b_h):
        return b_h // blocks, i_t // SUB, b_h % blocks

    def row(i_t, b_h):
        return b_h // blocks, i_t, b_h % blocks, 0

    keys = pl.BlockSpec((None, SUB, bh, width), keyed)
    out = pl.BlockSpec((None, None, bh, SUB), row)
    return pl.pallas_call(
        functools.partial(_intra_diagonal_kernel, scale=scale),
        grid=(length, batch * blocks),
        in_specs=[keys, keys, pl.BlockSpec((None, SUB, bh, width), headed),
                  pl.BlockSpec((None, SUB, bh), gated)],
        out_specs=[out, out],
        out_shape=[jax.ShapeDtypeStruct((batch, length, value_heads, SUB), k.dtype),
                   jax.ShapeDtypeStruct((batch, length, value_heads, SUB), jnp.float32)],
        compiler_params=_params(config), interpret=interpret, name="fla_kda_intra_diagonal",
    )(q, k, g, beta)


def _intra_solve_kernel(q_ref, k_ref, g_ref, beta_ref, aqkd_ref, akkd_ref, aqk_ref, akk_ref, *,
                        scale: float, bk: int, exact: bool):
    """fla's `chunk_kda_fwd_kernel_inter_solve_fused` (chunk_intra.py:43-391), safe_gate off, for one
    chunk of one head, in 16-row blocks, the key dimensions in blocks of `bk`. The diagonal blocks are
    `intra_diagonal`'s. A block below the diagonal, rows `i` and columns `j` of sub-chunks `I > J`,
    factors the decay through `g_n`, the first row of `I`: `(x_I exp2(g_I - g_n)) (k_J exp2(g_n -
    g_J))^T`, fp32 products at fla's TF32 (`_tf32_dot`), then scaled by `scale` or by beta. The
    diagonal blocks of `I + Akk` invert by forward substitution and merge in fla's order, `X_IJ =
    -(X_II A_IJ) X_JJ` for neighbours and `-X_II (sum_M A_IM X_MJ)` beyond, at TF32. `Aqk` and the
    inverse are stored in the keys' dtype, zero above the diagonal."""
    size, width = k_ref.shape
    count = size // SUB
    blocks = width // bk
    index = jnp.arange(SUB)
    zero = jnp.zeros((SUB, SUB), jnp.float32)

    pairs = [(i, j) for i in range(count) for j in range(i)]

    def block(c, sums):
        """One block of key columns for every pair in fla's order: each row's sub-chunk loaded after the
        rows before it are summed, and the earlier sub-chunks' keys and decays held."""
        columns = pl.ds(c * bk, bk)

        def load(ref, i):
            return ref[pl.ds(i * SUB, SUB), columns].astype(jnp.float32)

        found = dict(zip(pairs, sums, strict=True))
        held = [(load(k_ref, 0), load(g_ref, 0))]
        for i in range(1, count):
            key, gate = load(k_ref, i), load(g_ref, i)
            origin = g_ref[i * SUB, columns].astype(jnp.float32)[None, :]
            ahead = jnp.exp2(gate - origin)
            read, written = load(q_ref, i) * ahead, key * ahead
            for j, (earlier, decay) in enumerate(held):
                behind = (earlier * jnp.exp2(origin - decay)).T
                qk, kk = found[i, j]
                found[i, j] = (qk + _tf32_dot(read, behind), kk + _tf32_dot(written, behind))
            held.append((key, gate))
        return [found[pair] for pair in pairs]

    def window(i, j):
        return pl.ds(i * SUB, SUB), pl.ds(j * SUB, SUB)

    found = jax.lax.fori_loop(0, blocks, block, [(zero, zero)] * len(pairs))
    # Aqk is stored before the solve, so none of it is held through the solve.
    kks = {}
    for (i, j), (qk, kk) in zip(pairs, found, strict=True):
        aqk_ref[window(i, j)] = (qk * scale).astype(aqk_ref.dtype)
        aqk_ref[window(j, i)] = zero.astype(aqk_ref.dtype)
        kks[i, j] = kk * beta_ref[pl.ds(i * SUB, SUB)].astype(jnp.float32)[:, None]
    for i in range(count):
        aqk_ref[window(i, i)] = aqkd_ref[pl.ds(i * SUB, SUB), :]
        kks[i, i] = akkd_ref[pl.ds(i * SUB, SUB), :]
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
            akk_ref[window(i, j)] = (inverse[i, j] if j <= i else zero).astype(akk_ref.dtype)


def intra_solve(q, k, g, beta, *, scale: float, chunk: int = 64, bk: int | None = None,
                warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`(Aqk, Akk)` `[B, T, HV, BT]` in `k`'s dtype: each chunk's `scale (q exp2(g)) (k exp2(-g))^T`,
    causal, and the inverse `(I + tril(beta (k exp2(g)) (k exp2(-g))^T, -1))^-1`: fla's
    `chunk_kda_fwd_intra` before its `recompute_w_u_fwd` (chunk_intra.py:794-894): the diagonal blocks
    token by token (`intra_diagonal`), the rest and the solve chunk by chunk. `q`, `k` `[B, T, H,
    K]`, `g` the base-2 cumulated decay `[B, T, HV, K]` fp32, `beta` `[B, T, HV]`."""
    config = kernel_config("intra_solve", k.shape[-1], k.dtype, bk=bk, warps=warps, stages=stages)
    aqkd, akkd = intra_diagonal(q, k, g, beta, scale=scale, interpret=interpret)
    batch, length, key_heads, width = k.shape
    value_heads = beta.shape[2]
    keyed, headed, gated, _ = _heads(value_heads, value_heads // key_heads)
    keys = pl.BlockSpec((None, chunk, None, width), keyed)
    diagonal = pl.BlockSpec((None, chunk, None, SUB), headed)
    solved = pl.BlockSpec((None, chunk, None, chunk), headed)
    shape = jax.ShapeDtypeStruct((batch, length, value_heads, chunk), k.dtype)
    return pl.pallas_call(
        functools.partial(_intra_solve_kernel, scale=scale, bk=config["bk"], exact=interpret),
        grid=(length // chunk, batch * value_heads),
        in_specs=[keys, keys, pl.BlockSpec((None, chunk, None, width), headed),
                  pl.BlockSpec((None, chunk, None), gated), diagonal, diagonal],
        out_specs=[solved, solved], out_shape=[shape, shape],
        compiler_params=_params(config), interpret=interpret, name="fla_kda_intra_solve",
    )(q, k, g, beta, aqkd, akkd)


def _w_u_kernel(k_ref, v_ref, beta_ref, a_ref, g_ref, *refs, bk: int, bv: int, queries: bool, exact: bool):
    """fla's `recompute_w_u_fwd_kda_kernel` (fla/ops/kda/wy_fast.py:34-140) for one chunk of one head:
    `u = A (v beta)` and `w = A ((k beta) exp2(g))`, each operand rounded to its input dtype before the
    product; the gated keys `k exp2(g_last - g)`; with `queries`, the gated queries `q exp2(g)`."""
    if queries:
        q_ref, w_ref, u_ref, kg_ref, qg_ref = refs
    else:
        (w_ref, u_ref, kg_ref), q_ref, qg_ref = refs, None, None
    beta = beta_ref[...]
    inverse = a_ref[...]
    size = k_ref.shape[0]
    for block in range(v_ref.shape[-1] // bv):
        columns = pl.ds(block * bv, bv)
        value = v_ref[:, columns]
        scaled = (value * beta[:, None]).astype(value.dtype)
        u_ref[:, columns] = _dot(inverse, scaled, exact=exact).astype(u_ref.dtype)
    for block in range(k_ref.shape[-1] // bk):
        columns = pl.ds(block * bk, bk)
        key = k_ref[:, columns]
        gate = g_ref[:, columns].astype(jnp.float32)
        scaled = (key * beta[:, None]) * jnp.exp2(gate)
        if q_ref is not None and qg_ref is not None:
            qg_ref[:, columns] = (q_ref[:, columns] * jnp.exp2(gate)).astype(qg_ref.dtype)
        last = g_ref[size - 1, columns].astype(jnp.float32)
        kg_ref[:, columns] = (key * jnp.exp2(last[None, :] - gate)).astype(kg_ref.dtype)
        w_ref[:, columns] = _dot(inverse, scaled.astype(key.dtype), exact=exact).astype(w_ref.dtype)


def recompute_w_u(k, v, beta, a, g, q=None, *, chunk: int = 64, bk: int | None = None, bv: int | None = None,
                  warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`(w, u, kg, qg)`: fla's `recompute_w_u_fwd` for KDA (wy_fast.py:264-311). `w`, `kg` and `qg`
    `[B, T, HV, K]` in `k`'s dtype, `u` in `v`'s; `qg` only for given queries `q` `[B, T, H, K]`, else
    None. `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `beta` `[B, T, HV]`, `a` the inverse per chunk `[B,
    T, HV, BT]`, `g` the base-2 cumulated decay `[B, T, HV, K]` fp32."""
    config = kernel_config("recompute_w_u", min(k.shape[-1], v.shape[-1]), k.dtype, bk=bk, bv=bv,
                           warps=warps, stages=stages)
    batch, length, key_heads, width = k.shape
    value_heads, columns = v.shape[2:]
    keyed, headed, gated, _ = _heads(value_heads, value_heads // key_heads)
    keys = pl.BlockSpec((None, chunk, None, width), keyed)
    headed_keys = pl.BlockSpec((None, chunk, None, width), headed)
    gate = pl.BlockSpec((None, chunk, None), gated)
    keyed_shape = jax.ShapeDtypeStruct((batch, length, value_heads, width), k.dtype)
    queries = q is not None
    found = pl.pallas_call(
        functools.partial(_w_u_kernel, bk=min(config["bk"], width), bv=min(config["bv"], columns),
                          queries=queries, exact=interpret),
        grid=(length // chunk, batch * value_heads),
        in_specs=[keys, pl.BlockSpec((None, chunk, None, columns), headed), gate,
                  pl.BlockSpec((None, chunk, None, chunk), headed), headed_keys,
                  *([keys] if queries else [])],
        out_specs=[headed_keys, pl.BlockSpec((None, chunk, None, columns), headed),
                   headed_keys, *([headed_keys] if queries else [])],
        out_shape=[keyed_shape, jax.ShapeDtypeStruct(v.shape, v.dtype), keyed_shape,
                   *([keyed_shape] if queries else [])],
        compiler_params=_params(config), interpret=interpret, name="fla_kda_recompute_w_u",
    )(k, v, beta, a, g, *([q] if queries else []))
    return (*found[:3], found[3] if queries else None)


def _o_kernel(q_ref, v_ref, g_ref, h_ref, a_ref, o_ref, *, scale: float, bk: int, exact: bool):
    """fla's `chunk_gla_fwd_kernel_o` (fla/ops/gla/chunk.py:345-432) for one chunk of one head and `BV`
    value columns: `scale (q exp2(g))_narrow h + tril(Aqk) v`, the gated queries rounded to their
    dtype and `Aqk` to the values'."""
    size = q_ref.shape[0]
    out = jnp.zeros((size, v_ref.shape[-1]), jnp.float32)
    for block in range(q_ref.shape[-1] // bk):
        columns = pl.ds(block * bk, bk)
        query = q_ref[:, columns]
        gated = (query * jnp.exp2(g_ref[:, columns].astype(jnp.float32))).astype(query.dtype)
        out = out + _dot(gated, h_ref[columns, :].astype(gated.dtype), exact=exact)
    out = out * scale
    index = jnp.arange(size)
    value = v_ref[...]
    scores = jnp.where(index[:, None] >= index[None, :], a_ref[...], 0.0).astype(value.dtype)
    o_ref[...] = (out + _dot(scores, value, exact=exact)).astype(o_ref.dtype)


def chunk_fwd_o(q, v, g, a, h, *, scale: float, chunk: int = 64, bk: int | None = None, bv: int | None = None,
                warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """The output `[B, T, HV, V]` in `v`'s dtype: fla's `chunk_gla_fwd_o_gk` (gla/chunk.py:962-997). `q`
    `[B, T, H, K]`, `v` the corrected values `[B, T, HV, V]`, `g` the base-2 cumulated decay `[B, T, HV,
    K]`, `a` `Aqk` `[B, T, HV, BT]`, `h` the memory each chunk enters `[B, T // BT, HV, K, V]`."""
    config = kernel_config("chunk_fwd_o", min(q.shape[-1], v.shape[-1]), q.dtype, bk=bk, bv=bv,
                           warps=warps, stages=stages)
    batch, length, key_heads, width = q.shape
    value_heads, columns = v.shape[2:]
    group = value_heads // key_heads
    bv = min(config["bv"], columns)

    def keyed(i_v, i_t, bh):
        return bh // value_heads, i_t, (bh % value_heads) // group, 0

    def headed(i_v, i_t, bh):
        return bh // value_heads, i_t, bh % value_heads, 0

    def valued(i_v, i_t, bh):
        return bh // value_heads, i_t, bh % value_heads, i_v

    def memory(i_v, i_t, bh):
        return bh // value_heads, i_t, bh % value_heads, 0, i_v

    values = pl.BlockSpec((None, chunk, None, bv), valued)
    return pl.pallas_call(
        functools.partial(_o_kernel, scale=scale, bk=min(config["bk"], width), exact=interpret),
        grid=(columns // bv, length // chunk, batch * value_heads),
        in_specs=[pl.BlockSpec((None, chunk, None, width), keyed), values,
                  pl.BlockSpec((None, chunk, None, width), headed),
                  pl.BlockSpec((None, None, None, width, bv), memory),
                  pl.BlockSpec((None, chunk, None, chunk), headed)],
        out_specs=values, out_shape=jax.ShapeDtypeStruct(v.shape, v.dtype),
        compiler_params=_params(config), interpret=interpret, name="fla_kda_chunk_fwd_o",
    )(q, v, g, h, a)


def _dav_kernel(v_ref, do_ref, a_ref, dv_ref, da_ref, *, scale: float, bv: int, exact: bool):
    """fla's `chunk_kda_bwd_kernel_dAv` (fla/ops/kda/chunk_bwd.py:42-129) for one chunk of one head:
    `dAqk = scale tril(do v^T)` in fp32 and `dv = tril(Aqk)^T do`, `Aqk` in the gradient's dtype."""
    size = a_ref.shape[0]
    index = jnp.arange(size)
    out_grad_dtype = do_ref.dtype
    transposed = jnp.where(index[:, None] <= index[None, :], a_ref[...].T, 0.0).astype(out_grad_dtype)
    scores = jnp.zeros((size, size), jnp.float32)
    for block in range(v_ref.shape[-1] // bv):
        columns = pl.ds(block * bv, bv)
        out_grad = do_ref[:, columns]
        scores = scores + _dot(out_grad, v_ref[:, columns].T, exact=exact)
        dv_ref[:, columns] = _dot(transposed, out_grad, exact=exact).astype(dv_ref.dtype)
    da_ref[...] = jnp.where(index[:, None] >= index[None, :], scores * scale, 0.0).astype(da_ref.dtype)


def bwd_dav(v, do, a, *, scale: float, chunk: int = 64, bv: int | None = None, warps: int | None = None,
            stages: int | None = None, interpret: bool = False):
    """`(dAqk, dv)`: fla's `chunk_kda_bwd_dAv` (chunk_bwd.py:314-365), `dAqk` `[B, T, HV, BT]` fp32 and
    `dv` `[B, T, HV, V]` in `do`'s dtype. `v` the corrected values, `do` the output's gradient, `a`
    `Aqk`."""
    config = kernel_config("bwd_dav", v.shape[-1], v.dtype, bv=bv, warps=warps, stages=stages)
    batch, length, value_heads, columns = v.shape
    _, headed, _, _ = _heads(value_heads, 1)
    values = pl.BlockSpec((None, chunk, None, columns), headed)
    scores = pl.BlockSpec((None, chunk, None, chunk), headed)
    return pl.pallas_call(
        functools.partial(_dav_kernel, scale=scale, bv=min(config["bv"], columns), exact=interpret),
        grid=(length // chunk, batch * value_heads),
        in_specs=[values, values, scores], out_specs=[values, scores],
        out_shape=[jax.ShapeDtypeStruct(do.shape, do.dtype),
                   jax.ShapeDtypeStruct((batch, length, value_heads, chunk), jnp.float32)],
        compiler_params=_params(config), interpret=interpret, name="fla_kda_bwd_dav",
    )(v, do, a)[::-1]


def _wy_dqkg_kernel(q_ref, k_ref, v_ref, vn_ref, g_ref, beta_ref, a_ref, h_ref, do_ref, dh_ref, dv_ref,
                    dq_ref, dk_ref, dv2_ref, dg_ref, db_ref, da_ref, *, scale: float, bk: int, bv: int,
                    exact: bool):
    """fla's `chunk_kda_bwd_kernel_wy_dqkg_fused` (fla/ops/kda/chunk_bwd.py:132-311) for one chunk of
    one head: the queries', keys', decay's and beta's gradients through the output's memory term and
    the WY representation, the values' through `A`, and the gradient of `Akk`, with fla's casts and
    sums in its order. `A` is read transposed, as fla loads it; fla's first key block's value terms
    run first, on their own loop."""
    size, width = k_ref.shape
    columns = v_ref.shape[-1]
    index = jnp.arange(size)
    beta = beta_ref[...]
    transposed = a_ref[...].T

    def first(i_v, carry):
        scores, beta_grad = carry
        values = pl.ds(i_v * bv, bv)
        value_grad, value = dv_ref[:, values], v_ref[:, values]
        weighted = _dot(transposed, value_grad, exact=exact)
        dv2_ref[:, values] = (weighted * beta[:, None]).astype(dv2_ref.dtype)
        return (scores + _dot(value_grad, value.T, exact=exact),
                beta_grad + jnp.sum(weighted * value, axis=1))

    carry = (jnp.zeros((size, size), jnp.float32), jnp.zeros((size,), jnp.float32))
    carry = jax.lax.fori_loop(0, columns // bv, first, carry)

    def key_block(i_k, carry):
        scores, beta_grad = carry
        keys = pl.ds(i_k * bk, bk)

        def value_block(i_v, sums):
            dq, dk, dw, decay_grad = sums
            values = pl.ds(i_v * bv, bv)
            corrected, out_grad = vn_ref[:, values], do_ref[:, values]
            memory, memory_grad = h_ref[keys, values], dh_ref[keys, values]
            return (dq + _dot(out_grad, memory.T.astype(out_grad.dtype), exact=exact),
                    dk + _dot(corrected, memory_grad.T.astype(corrected.dtype), exact=exact),
                    dw + _dot(dv_ref[:, values].astype(corrected.dtype), memory.T.astype(corrected.dtype),
                              exact=exact),
                    decay_grad + jnp.sum((memory * memory_grad).astype(jnp.float32), axis=1))

        empty = jnp.zeros((size, bk), jnp.float32)
        dq, dk, dw, decay_grad = jax.lax.fori_loop(0, columns // bv, value_block,
                                                   (empty, empty, empty, jnp.zeros((bk,), jnp.float32)))
        key = k_ref[:, keys]
        gate = g_ref[:, keys].astype(jnp.float32)
        final = g_ref[size - 1, keys].astype(jnp.float32)
        gate_exp = jnp.exp2(gate)
        decay_grad = decay_grad * jnp.exp2(final)
        dq = dq * gate_exp * scale
        dk = dk * jnp.exp2(final[None, :] - gate)
        gated_key = key * gate_exp
        dw = -dw.astype(transposed.dtype)
        scores = scores + _dot(dw, gated_key.T.astype(transposed.dtype), exact=exact)
        dkgb = _dot(transposed, dw, exact=exact)
        beta_grad = beta_grad + jnp.sum(dkgb * gated_key, axis=1)
        key_dk = key * dk
        decay_grad = decay_grad + jnp.sum(key_dk, axis=0)
        dg = (q_ref[:, keys] * dq - key_dk + jnp.where((index == size - 1)[:, None], decay_grad[None, :], 0.0)
              + gated_key * dkgb * beta[:, None])
        dq_ref[:, keys] = dq.astype(dq_ref.dtype)
        dk_ref[:, keys] = (dk + dkgb * (gate_exp * beta[:, None])).astype(dk_ref.dtype)
        dg_ref[:, keys] = dg.astype(dg_ref.dtype)
        return scores, beta_grad

    scores, beta_grad = jax.lax.fori_loop(0, width // bk, key_block, carry)
    strict = index[:, None] > index[None, :]
    scores = jnp.where(strict, scores * beta[None, :], 0.0)
    scores = _dot(scores.astype(transposed.dtype), transposed, exact=exact)
    scores = _dot(transposed, scores.astype(transposed.dtype), exact=exact)
    da_ref[...] = jnp.where(strict, -scores, 0.0).astype(da_ref.dtype)
    db_ref[...] = beta_grad.astype(db_ref.dtype)


def bwd_wy_dqkg(q, k, v, corrected, g, beta, a, h, do, dh, dv, *, scale: float, chunk: int = 64,
                bk: int | None = None, bv: int | None = None, warps: int | None = None,
                stages: int | None = None, interpret: bool = False):
    """`(dq, dk, dv, db, dg, dAkk)`: fla's `chunk_kda_bwd_wy_dqkg_fused` (chunk_bwd.py:367-431), `dq`,
    `dk`, `dg` `[B, T, HV, K]`, `db` `[B, T, HV]` and `dAkk` `[B, T, HV, BT]` fp32, `dv` in `v`'s dtype.
    `h` and `dh` `[B, T // BT, HV, K, V]` the memory each chunk enters and its gradient, `dv` the
    corrected values' full gradient (`fla_gdn.chunk_bwd_dhu`'s), `a` `Akk`'s inverse."""
    config = kernel_config("bwd_wy_dqkg", min(k.shape[-1], v.shape[-1]), k.dtype, bk=bk, bv=bv,
                           warps=warps, stages=stages)
    batch, length, key_heads, width = k.shape
    value_heads, columns = v.shape[2:]
    keyed, headed, gated, memory = _heads(value_heads, value_heads // key_heads)
    keys = pl.BlockSpec((None, chunk, None, width), keyed)
    head_keys = pl.BlockSpec((None, chunk, None, width), headed)
    values = pl.BlockSpec((None, chunk, None, columns), headed)
    gate = pl.BlockSpec((None, chunk, None), gated)
    scores = pl.BlockSpec((None, chunk, None, chunk), headed)
    states = pl.BlockSpec((None, None, None, width, columns), memory)
    wide = jax.ShapeDtypeStruct((batch, length, value_heads, width), jnp.float32)
    found = pl.pallas_call(
        functools.partial(_wy_dqkg_kernel, scale=scale, bk=min(config["bk"], width),
                          bv=min(config["bv"], columns), exact=interpret),
        grid=(length // chunk, batch * value_heads),
        in_specs=[keys, keys, values, values, head_keys, gate, scores, states, values, states, values],
        out_specs=[head_keys, head_keys, values, head_keys, gate, scores],
        out_shape=[wide, wide, jax.ShapeDtypeStruct(v.shape, v.dtype), wide,
                   jax.ShapeDtypeStruct(beta.shape, jnp.float32),
                   jax.ShapeDtypeStruct((batch, length, value_heads, chunk), jnp.float32)],
        compiler_params=_params(config), interpret=interpret, name="fla_kda_bwd_wy_dqkg",
    )(q, k, v, corrected, g, beta, a, h, do, dh, dv)
    dq, dk, dv2, dg, db, da = found
    return dq, dk, dv2, db, dg, da


def _bwd_intra_kernel(q_ref, k_ref, g_ref, beta_ref, daqk_ref, dakk_ref, dq_ref, dk_ref, dg_ref,
                      dq2_ref, dk2_ref, db2_ref, dg2_ref, *, count: int, exact: bool):
    """fla's `chunk_kda_bwd_kernel_intra` (fla/ops/kda/chunk_intra.py:395-679), safe_gate off, for one
    sub-chunk `i` of a chunk, one head and one block of key columns, as fla's grid runs it. The
    gradients of `Aqk` and `Akk` reach the sub-chunk's queries and keys from the sub-chunks before
    (`dq2`, `dk2`, the decay factored at its first row) and after (`dkt`, factored at its last row),
    fp32 products at fla's TF32 (`_tf32_dot`), and from its own tokens one by one in fp32; then beta's
    and the decay's gradients, added to the ones `bwd_wy_dqkg` gave. Beta's is this key block's share."""
    i = pl.program_id(1) % count
    first = i * SUB
    end = first + SUB - 1
    rows = pl.ds(first, SUB)
    index = jnp.arange(SUB)
    width = k_ref.shape[-1]

    def tile(ref, start):
        return ref[pl.ds(start, SUB), :].astype(jnp.float32)

    def scores(ref, row, column):
        return ref[pl.ds(row, SUB), pl.ds(column, SUB)]

    gate = tile(g_ref, first)
    key = tile(k_ref, first)
    beta = beta_ref[rows].astype(jnp.float32)
    origin = g_ref[first, :].astype(jnp.float32)[None, :]

    def before(j, sums):
        gated = tile(k_ref, j * SUB) * jnp.exp2(origin - tile(g_ref, j * SUB))
        return (sums[0] + _tf32_dot(scores(daqk_ref, first, j * SUB), gated),
                sums[1] + _tf32_dot(scores(dakk_ref, first, j * SUB), gated))

    empty = jnp.zeros((SUB, width), jnp.float32)
    dq2, dk2 = jax.lax.fori_loop(0, i, before, (empty, empty))
    ahead = jnp.exp2(gate - origin)
    dq2, dk2 = dq2 * ahead, dk2 * ahead

    def diagonal(j, sums):
        token = first + j
        decay = jnp.exp2(gate - g_ref[token, :].astype(jnp.float32)[None, :])
        causal = (index >= j)[:, None]
        own = k_ref[token, :].astype(jnp.float32)[None, :] * decay
        return (sums[0] + jnp.where(causal, daqk_ref[rows, token][:, None] * own, 0.0),
                sums[1] + jnp.where(causal, dakk_ref[rows, token][:, None] * own, 0.0))

    dq2, dk2 = jax.lax.fori_loop(0, SUB, diagonal, (dq2, dk2))
    beta_grad = jnp.sum(dk2 * key, axis=1)
    dk2 = dk2 * beta[:, None]
    dg2 = tile(q_ref, first) * dq2
    dq2_ref[...] = (dq2 + dq_ref[rows, :]).astype(dq2_ref.dtype)
    closing = g_ref[end, :].astype(jnp.float32)[None, :]

    def after(j, dkt):
        behind = jnp.exp2(tile(g_ref, j * SUB) - closing)
        # k beta in the operands' dtype, as fla's b_kb, then widened by the decay.
        later = pl.ds(j * SUB, SUB)
        weighted = (k_ref[later, :] * beta_ref[later][:, None]).astype(jnp.float32)
        dkt = dkt + _tf32_dot(scores(daqk_ref, j * SUB, first).T, tile(q_ref, j * SUB) * behind)
        return dkt + _tf32_dot(scores(dakk_ref, j * SUB, first).T, weighted * behind)

    dkt = jax.lax.fori_loop(i + 1, count, after, empty) * jnp.exp2(closing - gate)

    def own(j, dkt):
        token = first + j
        query = q_ref[token, :].astype(jnp.float32)
        weighted = k_ref[token, :].astype(jnp.float32) * beta_ref[token].astype(jnp.float32)
        decay = jnp.exp2(g_ref[token, :].astype(jnp.float32)[None, :] - gate)
        causal = (index <= j)[:, None]
        dkt = dkt + jnp.where(causal, daqk_ref[token, rows][:, None] * query[None, :] * decay, 0.0)
        return dkt + jnp.where(causal, dakk_ref[token, rows][:, None] * weighted[None, :] * decay, 0.0)

    dkt = jax.lax.fori_loop(0, SUB, own, dkt)
    dg2 = dg2 + (dk2 - dkt) * key + dg_ref[rows, :]
    dk2_ref[...] = (dk2 + dk_ref[rows, :] + dkt).astype(dk2_ref.dtype)
    dg2_ref[...] = dg2.astype(dg2_ref.dtype)
    db2_ref[...] = beta_grad.astype(db2_ref.dtype)


def bwd_intra(q, k, g, beta, daqk, dakk, dq, dk, db, dg, *, chunk: int = 64, bk: int | None = None,
              warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`(dq, dk, db, dg)` fp32, `bwd_wy_dqkg`'s plus the intra-chunk terms of `dAqk` and `dAkk`: fla's
    `chunk_kda_bwd_intra` (chunk_intra.py:897-960), its grid over key blocks, sub-chunks and heads,
    beta's gradient summed over the key blocks after. `dq`, `dk`, `dg` `[B, T, HV, K]`, `db` `[B, T,
    HV]`."""
    config = kernel_config("bwd_intra", k.shape[-1], k.dtype, bk=bk, warps=warps, stages=stages)
    batch, length, key_heads, width = k.shape
    value_heads = beta.shape[2]
    group = value_heads // key_heads
    count = chunk // SUB
    bk = config["bk"]

    def keyed(i_k, i_s, bh):
        return bh // value_heads, i_s // count, (bh % value_heads) // group, i_k

    def headed(i_k, i_s, bh):
        return bh // value_heads, i_s // count, bh % value_heads, i_k

    def gated(i_k, i_s, bh):
        return bh // value_heads, i_s // count, bh % value_heads

    def scored(i_k, i_s, bh):
        return bh // value_heads, i_s // count, bh % value_heads, 0

    def sub(i_k, i_s, bh):
        return bh // value_heads, i_s, bh % value_heads, i_k

    def shared(i_k, i_s, bh):
        return i_k, bh // value_heads, i_s, bh % value_heads

    keys = pl.BlockSpec((None, chunk, None, bk), keyed)
    head_keys = pl.BlockSpec((None, chunk, None, bk), headed)
    scores = pl.BlockSpec((None, chunk, None, chunk), scored)
    out = pl.BlockSpec((None, SUB, None, bk), sub)
    wide = jax.ShapeDtypeStruct((batch, length, value_heads, width), jnp.float32)
    blocks = width // bk
    dq, dk, partial, dg = pl.pallas_call(
        functools.partial(_bwd_intra_kernel, count=count, exact=interpret),
        grid=(blocks, length // SUB, batch * value_heads),
        in_specs=[keys, keys, head_keys, pl.BlockSpec((None, chunk, None), gated), scores, scores,
                  head_keys, head_keys, head_keys],
        out_specs=[out, out, pl.BlockSpec((None, None, SUB, None), shared), out],
        out_shape=[wide, wide, jax.ShapeDtypeStruct((blocks, *beta.shape), jnp.float32), wide],
        compiler_params=_params(config), interpret=interpret, name="fla_kda_bwd_intra",
    )(q, k, g, beta, daqk, dakk, dq, dk, dg)
    return dq, dk, db + jnp.sum(partial, axis=0), dg


def _rule(q, k, v, g, beta, h0, scale: float, chunk: int, interpret: bool):
    return _rule_fwd(q, k, v, g, beta, h0, scale, chunk, interpret)[0]


def _rule_fwd(q, k, v, g, beta, h0, scale: float, chunk: int, interpret: bool):
    """fla's `chunk_kda_fwd` (fla/ops/kda/chunk_fwd.py:20-134): the decay cumulated in base 2, the
    intra-chunk scores and inverse, `w`, `u` and the gated keys, the memory across chunks
    (`fla_gdn.chunk_fwd_h`'s USE_GK branch), then the output. Saves what fla saves."""
    gc = fla_gdn.chunk_local_cumsum(g, chunk=chunk, scale=RCP_LN2, interpret=interpret)
    aqk, akk = intra_solve(q, k, gc, beta, scale=scale, chunk=chunk, interpret=interpret)
    w, u, kg, _ = recompute_w_u(k, v, beta, akk, gc, chunk=chunk, interpret=interpret)
    h, corrected, final = fla_gdn.chunk_fwd_h(kg, w, u, gc, h0, chunk=chunk, interpret=interpret)
    o = chunk_fwd_o(q, corrected, gc, aqk, h, scale=scale, chunk=chunk, interpret=interpret)
    return (o, final), (q, k, v, gc, beta, aqk, akk, h0)


def _rule_bwd(scale: float, chunk: int, interpret: bool, residual, cotangent):
    """fla's `chunk_kda_bwd` (fla/ops/kda/chunk_bwd.py:434-615): `w`, `u`, the gated queries and keys
    and the memory recomputed, then `dAqk` and the values' local gradient, the reverse recurrence
    (`fla_gdn.chunk_bwd_dhu`'s USE_GK branch), the fused WY and memory gradients, the intra-chunk
    terms, a key head's value heads summed (fla's GVA), and the decay's gradient summed back over
    each chunk."""
    q, k, v, gc, beta, aqk, akk, h0 = residual
    do, dht = cotangent
    w, u, kg, qg = recompute_w_u(k, v, beta, akk, gc, q, chunk=chunk, interpret=interpret)
    h, corrected, _ = fla_gdn.chunk_fwd_h(kg, w, u, gc, h0, chunk=chunk, interpret=interpret)
    daqk, dv = bwd_dav(corrected, do, aqk, scale=scale, chunk=chunk, interpret=interpret)
    dh, dh0, dv = fla_gdn.chunk_bwd_dhu(qg, kg, w, gc, dht.astype(jnp.float32), do, dv, scale=scale,
                                        chunk=chunk, interpret=interpret)
    dq, dk, dv, db, dg, dakk = bwd_wy_dqkg(q, k, v, corrected, gc, beta, akk, h, do, dh, dv, scale=scale,
                                           chunk=chunk, interpret=interpret)
    dq, dk, db, dg = bwd_intra(q, k, gc, beta, daqk, dakk, dq, dk, db, dg, chunk=chunk, interpret=interpret)
    group = v.shape[2] // k.shape[2]
    if group > 1:
        dq, dk = (x.reshape(*x.shape[:2], k.shape[2], group, x.shape[-1]).sum(axis=3) for x in (dq, dk))
    dg = fla_gdn.chunk_local_cumsum(dg, chunk=chunk, reverse=True, interpret=interpret)
    return (dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), dg, db.astype(beta.dtype), dh0)


chunk_kda = jax.custom_vjp(_rule, nondiff_argnums=(6, 7, 8))
"""fla's chunked Kimi delta attention on the ported kernels: `(o, final)` for `q`, `k` `[B, T, H, K]`
(normalised as the caller wants them), `v` `[B, T, HV, V]`, the per-token natural log decay `g` `[B,
T, HV, K]` fp32, `beta` `[B, T, HV]` and the initial memory `h0` `[B, HV, K, V]` fp32; `T` a whole
number of chunks. Its numbers are fla's: bf16 operands store `A`, `w`, `u`, the gated keys and the
memory each chunk enters in bf16, with fp32 accumulation, fp32 decay and the memory fp32 across
chunks; the output in the values' dtype and the final memory fp32."""
chunk_kda.defvjp(_rule_fwd, _rule_bwd)
