# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Translated from flash-linear-attention v0.5.2 (fla/ops/gated_delta_rule, fla/ops/common,
# fla/ops/utils), which is licensed under the MIT license: see licenses/flash-linear-attention-LICENSE.

"""fla's chunked gated delta rule, ported kernel for kernel to Pallas reading Dew's [B, T, H, D].

The port follows fla-org/flash-linear-attention v0.5.2 (fla/ops/gated_delta_rule/chunk.py) without
redesign: the same decomposition into kernels, the same blocks over each kernel's grid, and the
numerics fla runs: bf16 operands into fp32 accumulation where fla stores bf16, the log decay
cumulated in base 2 (`g * log2(e)`, fla's RCP_LN2) and fp32 state. Each kernel names its fla origin.
Every operand stays in Dew's activation layout: queries and keys `[B, T, H, K]`, values `[B, T, HV,
V]`, per-token gates `[B, T, HV]`; a program reads its chunk of its head where it sits, as fla's
pointer arithmetic does. The memories the chunks enter are stored where they pass, `[slots, HV, K, V]`
(`Sequences`), so however many documents a row packs, they take no more slots than its chunks, a
row's own and one of zeros.

Each kernel is a launch (`dew.nn.kernels.tuning.KernelOp`) whose tiles are fla's own picks, for an
A100's block and for 99 KB GPUs' (its `PICKS`); `dew.nn.kernels.recurrence_chunk.fla_rule` shapes any
other call to them.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl

from dew.nn.kernels import recurrence, recurrence_algebra as algebra
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
    _span,
    laid,
    read_rows,
    sequences,
    slots,
    spanned,
    write_rows,
)

RCP_LN2 = 1 / math.log(2)
"""1 / ln 2, fla's RCP_LN2 (whose 1.4426950216 rounds to the same fp32): its kernels take the cumulated
decay in base 2 and exponentiate with exp2; the prefix converts in float64 (`recurrence.cumulated`)."""

WIDEST = 128
"""The widest values each of the port's calls takes: its tiles are proven at the 0.8B's and 9B's
128-wide heads; wider values split into more heads."""

WIDEST_KEYS = 256
"""The widest 16-bit keys the port takes, fla's own bound (chunk_delta_h.py's four 64-wide blocks of
the memory); keys cannot split. Its picks run there (`ChunkBwdDhu.WIDE_KEYS_PICK` the one narrower
than the 128-wide pick). fp32 keys take at most WIDEST: at 256 an A100's fp32 forward h needs 172,292
shared bytes at its narrowest tile (CPU compile)."""


def _pairs(rows, columns):
    """The decay's exponent from each column position to each row position, `[R, C]`: `g_r - g_c`."""
    return rows[:, None] - columns[None, :]


def _weighted_values(v_ref, u_ref, beta, inverse, start, taken, *, chunk: int, bv: int, exact: bool,
                     ieee: bool):
    """`u = A (v beta)` for a chunk, `BV` value columns at a time, `v beta` rounded to the values' dtype
    before the product (fla's recompute_w_u kernels' `b_u`), at IEEE fp32 where `ieee`."""
    for block in range(v_ref.shape[-1] // bv):
        columns = pl.ds(block * bv, bv)
        value = read_rows(_rows(v_ref, start, chunk, columns), taken)
        scaled = (value * beta[:, None]).astype(value.dtype)
        write_rows(_rows(u_ref, start, chunk, columns), taken, _dot(inverse, scaled, exact=exact, ieee=ieee))


def _dot(a, b, *, exact: bool, ieee: bool = False, solve: jnp.dtype | None = None):
    """`a @ b`, same-dtype operands, with fp32 accumulation; reject mismatches before any interpret-mode
    widening, so CPU tests catch dots Triton rejects. Exact interpretation widens bf16/fp16 inputs
    for CPUs without narrow-input/fp32-accumulating dots.

    Each operand dtype names its products, so no global `jax_default_matmul_precision` changes the
    kernel (the suite's 'highest' made every fp32 product IEEE):
    - bf16 or fp16: exact products summed in fp32 (fla's `tl.dot` of narrow operands); no precision.
    - fp32 with `ieee`: IEEE products, fla's `allow_tf32=False` dots (`recompute_w_u`'s `A @ v`).
    - fp32 otherwise: BF16x6 (`DotAlgorithmPreset.BF16_BF16_F32_X6`), six bf16 products, an fp32
      product's accuracy where fla's own dots take one TF32 (Triton's default `input_precision` for fp32,
      and `SOLVE_TRIL_DOT_PRECISION`, chunk_fwd.py:21). TF32x3 drops the low parts' product: an
      A100 gave (1 + 2^-11)^2 as 1.0009765625, its fp32 1.0009768 (SE-dbg-kda-dot-redgreen-3f3c64bb2),
      which put Kimi delta attention's fp32 rule past check 1 against fla's recorded torch. jax 0.11's
      Pallas lowers `Precision.HIGH` as plain TF32 (pallas/triton/lowering.py `_TF32_PRECISIONS`), so
      the preset is named. IEEE there lowers on sm80 to FMA loops ptxas took 3.5 to over 15 minutes to
      compile for the WY backward; the presets compile in seconds.
    - fp32 with `solve`, the call's dtype, at the solves' products of fp32 intermediates (the kkt and
      KDA sub-chunk merges and products, fla's SOLVE_TRIL_DOT_PRECISION): BF16x6 in an fp32 call,
      where one TF32 put a Kimi linear layer's logits 100 times the reference's distance from
      float64 on an H100; one TF32, fla's, in a 16-bit call, whose operands' rounding it stays under
      and where an fp32 product's three-part form cost Kimi Linear's bf16 layer 17% on an H100 and 19%
      on an A100.
    - interpreted: no precision, the CPU's fp32."""
    if jnp.dtype(a.dtype) != jnp.dtype(b.dtype):
        raise TypeError(f"Pallas dot operands must have the same dtype, got {a.dtype} and {b.dtype}")
    wide = jnp.dtype(a.dtype) == jnp.float32
    if exact:
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        precision = jax.lax.Precision.HIGHEST if ieee and wide else None
    elif wide:
        narrow_call = solve is not None and jnp.dtype(solve).itemsize < 4
        precision = (jax.lax.Precision.HIGHEST if ieee else jax.lax.Precision.DEFAULT if narrow_call
                     else jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6)
    else:
        precision = None
    return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=precision)


def _kkt_solve_kernel(chunk_ref, k_ref, g_ref, beta_ref, a_ref, *, chunk: int, aligned: bool,
                      sub: int, bk: int, exact: bool):
    """fla's `chunk_gated_delta_rule_fwd_kkt_solve_kernel` (fla/ops/gated_delta_rule/chunk_fwd.py:40-
    321) for one chunk of one head: `(I + A)^-1` for `A = tril(beta_s (k_s . k_t) exp2(g_s - g_t), -1)`,
    in `sub`-row blocks (fla's BC = 16 of BT = 64). Every lower block of `k k^T` accumulates over the
    key columns `bk` at a time; the gate and beta scale them; forward substitution inverts each
    diagonal block; and neighbouring inverses merge as `-X_i A_ij X_j`, the merge's products at
    the call's solve precision (`_dot`'s `solve`)."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        _kkt_solve_chunk(k_ref, g_ref, beta_ref, a_ref, start, taken, chunk=chunk, sub=sub, bk=bk,
                         exact=exact)


def _kkt_solve_chunk(k_ref, g_ref, beta_ref, a_ref, start, taken, *, chunk: int, sub: int, bk: int,
                     exact: bool):
    # Triton lowers no slice of a loaded value, so every block is loaded from its ref.
    width = k_ref.shape[-1]
    count = chunk // sub

    def part(ref, i, *rest):
        return read_rows(_rows(ref, start + i * sub, sub, *rest), taken - i * sub)

    gates = [_gates(g_ref, start, taken, sub, i * sub) for i in range(count)]
    betas = [part(beta_ref, i).astype(jnp.float32) for i in range(count)]

    def keys(i, c):
        return part(k_ref, i, pl.ds(c * bk, bk))

    def issue(name, a, b):
        # The keys' products at the call's precision, the merges at its solve precision (`_dot`).
        return _dot(a, b, exact=exact, solve=k_ref.dtype if name == "merge" else None)

    blocks = algebra.system(keys, lambda i, j: jnp.exp2(_pairs(gates[i], gates[j])), betas, count,
                            width // bk, issue)
    inverse = algebra.inverted(blocks, issue)
    # fla zero-fills A before its kernel stores the lower blocks; here the blocks above are stored as zeros.
    for i in range(count):
        for j in range(count):
            found = inverse[i, j] if j <= i else jnp.zeros((sub, sub), jnp.float32)
            write_rows(_rows(a_ref, start + i * sub, sub, pl.ds(j * sub, sub)), taken - i * sub, found)


@dataclasses.dataclass(frozen=True)
class KktSolve(_Launch):
    """`kkt_solve`'s launch (tuning.KernelOp)."""

    NAME = "kkt_solve"
    KEYS, VALUES = 'k', None
    PICKS = _picks(Tiles(64, 0, 1, 3))

    def _fwd(self, k: jax.Array, g: jax.Array, beta: jax.Array, chunk_spans: jax.Array,
             sequence_spans: jax.Array, *, chunk: int, sub: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        bk, warps, stages = config.bk, config.warps, config.stages
        batch, length, key_heads, width = k.shape
        value_heads = beta.shape[2]
        group = value_heads // key_heads
        tokens = batch * length
        aligned = _aligned(spans, batch, length, chunk)
        gated = pl.BlockSpec((tokens, None), lambda i_t, h: (0, h))
        solved = pl.pallas_call(
            functools.partial(_kkt_solve_kernel, chunk=chunk, aligned=aligned, sub=sub, bk=min(bk, width),
                              exact=interpret),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0)), gated,
                      gated],
            out_specs=pl.BlockSpec((tokens, None, chunk), lambda i_t, h: (0, h, 0)),
            out_shape=jax.ShapeDtypeStruct((tokens, value_heads, chunk), k.dtype),
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="fla_kkt_solve",
        )(spans.chunks, *_flat(k, g, beta))
        return (solved.reshape(batch, length, value_heads, chunk)), None


_KKT_SOLVE = KktSolve()


def kkt_solve(k, g, beta, *, chunk: int = 64, sub: int = 16, spans: Sequences | None = None,
              interpret: bool = False):
    """`(I + A)^-1` per chunk, `[B, T, HV, BT]` in `k`'s dtype and zero above the diagonal: fla's
    fused kkt and solve for 64-token chunks (chunk_fwd.py:367-396). `k` `[B, T, H, K]`, `g`
    the base-2 cumulated decay (`recurrence.cumulated`) and `beta` `[B, T, HV]`. fla autotunes
    `BK` over 32 and 64 and warps over 1, 2 and 4."""
    return spanned(_KKT_SOLVE, k, g, beta, chunk=chunk, spans=spans, sub=sub, interpret=interpret)


def _w_u_kernel(chunk_ref, k_ref, v_ref, beta_ref, a_ref, g_ref, w_ref, u_ref, *, chunk: int,
                aligned: bool, bk: int,
                bv: int, exact: bool):
    """fla's `recompute_w_u_fwd_kernel` (fla/ops/gated_delta_rule/wy_fast.py:39-103) for one chunk
    of one head: `u = A (v * beta)` and `w = A (k * beta * exp2(g))`, each operand rounded to its
    input dtype before the product and `A`'s product with `v` at IEEE fp32 (`allow_tf32=False`)."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        beta = read(beta_ref)
        inverse = read(a_ref)
        _weighted_values(v_ref, u_ref, beta, inverse, start, taken, chunk=chunk, bv=bv, exact=exact,
                         ieee=True)
        decay = jnp.exp2(_gates(g_ref, start, taken, chunk))
        for block in range(k_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            key = read(k_ref, columns)
            scaled = key * beta[:, None] * decay[:, None]
            product = _dot(inverse, scaled.astype(key.dtype), exact=exact)
            write_rows(_rows(w_ref, start, chunk, columns), taken, product)


@dataclasses.dataclass(frozen=True)
class RecomputeWU(_Launch):
    """`recompute_w_u`'s launch (tuning.KernelOp)."""

    NAME = "recompute_w_u"
    KEYS, VALUES = 'k', 'v'
    PICKS = _picks(Tiles(64, 64, 2, 4))

    def _fwd(self, k: jax.Array, v: jax.Array, beta: jax.Array, a: jax.Array, g: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        bk, bv, warps, stages = config.bk, config.bv, config.warps, config.stages
        spans, batch, length, _, width, value_heads, columns, group, tokens, aligned = laid(
            k, v, chunk_spans, sequence_spans, chunk)

        headed = functools.partial(_headed, tokens)

        gate = pl.BlockSpec((tokens, None), lambda i_t, h: (0, h))
        w, u = pl.pallas_call(
            functools.partial(_w_u_kernel, chunk=chunk, aligned=aligned, bk=min(bk, width),
                              bv=min(bv, columns), exact=interpret),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0)),
                      headed(columns), gate, headed(chunk), gate],
            out_specs=[headed(width), headed(columns)],
            out_shape=[jax.ShapeDtypeStruct((tokens, value_heads, width), k.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads, columns), v.dtype)],
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="fla_recompute_w_u",
        )(spans.chunks, *_flat(k, v, beta, a, g))
        return (w.reshape(batch, length, value_heads, width), u.reshape(v.shape)), None


_RECOMPUTE_W_U = RecomputeWU()


def recompute_w_u(k, v, beta, a, g, *, chunk: int = 64, spans: Sequences | None = None,
                  interpret: bool = False):
    """`(w, u)`, in `k`'s and `v`'s dtypes: fla's `recompute_w_u_fwd` (wy_fast.py:247-296).

    `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `beta` `[B, T, HV]` in the inputs' dtype, `a` the
    inverse `(I + beta K K^T)^-1` per chunk `[B, T, HV, BT]` (fla's `A`, rows by token), `g` the
    base-2 cumulated decay `[B, T, HV]` fp32. fla fixes `BK = BV = 64` and autotunes warps over
    2, 4, 8 and stages over 2, 3, 4."""
    return spanned(_RECOMPUTE_W_U, k, v, beta, a, g, chunk=chunk, spans=spans, interpret=interpret)


def _stored(ref, row, memory) -> None:
    """A row's memory, its 64-key slices in order, into `ref`'s row `row` `[K, V]`."""
    for i, block in enumerate(memory):
        ref[row, pl.ds(i * 64, 64), :] = block.astype(ref.dtype)

def _h_kernel(span_ref, k_ref, u_ref, w_ref, g_ref, h0_ref, h_ref, corrected_ref, ht_ref, *,
              chunk: int, aligned: bool, length: int, exact: bool, per_key: bool = False):
    """fla's `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (fla/ops/common/chunk_delta_h.py:50-
    343) for one sequence's head and `BV` value columns, its chunks in order. The memory is fp32 in
    64-key slices; each chunk stores the memory it enters in `h`'s dtype at its slot (`Sequences`),
    then the corrected values
    `u - w h` (fla's `v_new`; `h` cast to `w`'s dtype for the product), gates them by
    `exp2(g_last - g)` and the memory by `exp2(g_last)`,
    casts the gated values to `k`'s dtype and adds `k^T v` to the memory: h remains fp32, and h and
    the corrected values keep fla's storage.
    `per_key` is fla's USE_GK branch: `g_ref` holds a decay per key dimension `[T, K]`, which gates
    the memory's rows by `exp2(g_last)` and not the values, whose keys carry their own decay. A row's
    first sequence starts from its initial memory and its last stores the final one."""
    start, count, base, entered, _, row, first, last_in_row = _span(span_ref, length, chunk, aligned)
    slices = k_ref.shape[-1] // 64

    @pl.when(count > 0)
    def _():
        state = [jnp.where(first, h0_ref[row, pl.ds(i * 64, 64), :].astype(jnp.float32), 0.0)
                 for i in range(slices)]

        def step(n, state):
            begin = start + n * chunk
            taken = chunk if aligned else jnp.minimum(chunk, count - n * chunk)

            def read(ref, *rest):
                return read_rows(_rows(ref, begin, chunk, *rest), taken)

            slot = jnp.where(n > 0, base + n, entered)
            for i in range(slices):
                h_ref[slot, pl.ds(i * 64, 64), :] = state[i].astype(h_ref.dtype)
            predicted = sum(_dot(read(w_ref, pl.ds(i * 64, 64)), state[i].astype(w_ref.dtype), exact=exact)
                            for i in range(slices))
            corrected = read(u_ref).astype(jnp.float32) - predicted
            write_rows(_rows(corrected_ref, begin, chunk), taken, corrected)
            final = begin + taken - 1

            def written(i, values):
                return _dot(read(k_ref, pl.ds(i * 64, 64)).T, values.astype(k_ref.dtype), exact=exact)

            if per_key:
                return [state[i] * jnp.exp2(g_ref[final, pl.ds(i * 64, 64)].astype(jnp.float32))[:, None]
                        + written(i, corrected) for i in range(slices)]
            last = _row(g_ref, final)
            gate = _gates(g_ref, begin, taken, chunk)
            gated = corrected * jnp.exp2(last - gate)[:, None]
            return [state[i] * jnp.exp2(last) + written(i, gated) for i in range(slices)]

        state = jax.lax.fori_loop(0, -(-count // chunk), step, state)

        pl.when(last_in_row)(lambda: _stored(ht_ref, row, state))


@dataclasses.dataclass(frozen=True)
class ChunkFwdH(_Launch):
    """`chunk_fwd_h`'s launch (tuning.KernelOp)."""

    NAME = "chunk_fwd_h"
    KEYS, VALUES = 'k', 'u'
    # fla's stages for devices under an A100's shared memory (chunk_delta_h.py:43-44), with two warps.
    PICKS = _picks(Tiles(0, 32, 4, 3), small=Tiles(0, 32, 2, 2), wide_fp32=Tiles(0, 64, 4, 2),
                   small_fp32=Tiles(0, 64, 2, 1))

    def _fwd(self, k: jax.Array, w: jax.Array, u: jax.Array, g: jax.Array, h0: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        bv, warps, stages = config.bv, config.warps, config.stages
        spans, batch, length, _, width, value_heads, columns, group, tokens, aligned = laid(
            k, u, chunk_spans, sequence_spans, chunk)

        def head(i_v, sh):
            return sh % value_heads

        per_key = g.ndim == 4
        rows = pl.BlockSpec((batch, None, width, bv), lambda i_v, sh: (0, head(i_v, sh), 0, i_v))
        gated = (pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh), 0)) if per_key
                 else pl.BlockSpec((tokens, None), lambda i_v, sh: (0, head(i_v, sh))))
        held = slots(spans, batch, length, chunk)
        h, corrected, final = pl.pallas_call(
            functools.partial(_h_kernel, chunk=chunk, aligned=aligned, length=length, exact=interpret,
                              **({"per_key": True} if per_key else {})),
            grid=(columns // bv, spans.spans.shape[0] * value_heads),
            in_specs=[pl.BlockSpec((None, 5), lambda i_v, sh: (sh // value_heads, 0)),
                      pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh) // group, 0)),
                      pl.BlockSpec((tokens, None, bv), lambda i_v, sh: (0, head(i_v, sh), i_v)),
                      pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh), 0)),
                      gated, rows],
            out_specs=[pl.BlockSpec((held, None, width, bv), lambda i_v, sh: (0, head(i_v, sh), 0, i_v)),
                       pl.BlockSpec((tokens, None, bv), lambda i_v, sh: (0, head(i_v, sh), i_v)),
                       rows],
            out_shape=[jax.ShapeDtypeStruct((held, value_heads, width, columns), k.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads, columns), u.dtype),
                       jax.ShapeDtypeStruct((batch, value_heads, width, columns), jnp.float32)],
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="fla_chunk_fwd_h",
        )(spans.spans, *_flat(k, u, w, g), h0)
        return (h, corrected.reshape(u.shape), final), None


_CHUNK_FWD_H = ChunkFwdH()


def chunk_fwd_h(k, w, u, g, h0, *, chunk: int = 64, spans: Sequences | None = None,
                interpret: bool = False):
    """`(h, corrected, final)`: the memory each chunk enters at its slot `[slots, HV, K, V]`
    (`Sequences`) in `k`'s dtype, the corrected values `[B, T, HV, V]` in `u`'s and each row's final
    memory `[B, HV, K, V]` fp32: fla's `chunk_gated_delta_rule_fwd_h` (chunk_delta_h.py:672-737)
    with an initial state. `k` `[B, T, H, K]`, `w` `[B, T, HV, K]`, `u` `[B, T, HV, V]`, `g`
    the base-2 cumulated decay (`recurrence.cumulated`) `[B, T, HV]`, or per key dimension `[B,
    T, HV, K]` (fla's `gk`, Kimi's delta attention), `h0` `[B, HV, K, V]` fp32, each row's. `K` a
    multiple of 64 (fla's blockdim64 kernel takes up to 256). On an A100 fla's autotuner chose BV 32,
    4 warps, 3 stages (c53)."""
    return spanned(_CHUNK_FWD_H, k, w, u, g, h0, chunk=chunk, spans=spans, interpret=interpret)


def _dhu_kernel(span_ref, q_ref, k_ref, w_ref, g_ref, dht_ref, do_ref, dv_ref, dh_ref, dh0_ref,
                dv2_ref, *, chunk: int, aligned: bool, length: int, scale: float, exact: bool,
                per_key: bool = False):
    """fla's `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` (chunk_delta_h.py:359-669) for one
    sequence's head and `BV` value columns, chunks in reverse. Each stores the gradient of the memory
    it leaves into, `dh`, at that memory's slot (`Sequences`), then
    `dv2 = dv + (k dh) exp2(g_last - g)` (dh cast to k's dtype), then
    `dh = dh exp2(g_last) + scale (q exp2(g))^T do - w^T dv2`: the gated queries fp32 as two parts
    of `do`'s dtype (`_split_dot`), where fla takes one TF32 product at the same cost, its terms summed
    from the chunk's last token, and dv2 cast to w's dtype for its product, as fla casts it.
    `per_key` is fla's USE_GK branch: `g_ref` `[T, K]` decays the gradient's rows by `exp2(g_last)`,
    and neither `dv2` nor the queries, which carry their own decay, are gated; the query product
    stays in q's dtype. A row's last sequence starts from the final
    memory's gradient and its first stores the initial memory's."""
    start, count, base, _, left, row, first, last_in_row = _span(span_ref, length, chunk, aligned)
    slices = k_ref.shape[-1] // 64
    chunks = -(-count // chunk)

    @pl.when(count > 0)
    def _():
        grad = [jnp.where(last_in_row, dht_ref[row, pl.ds(i * 64, 64), :].astype(jnp.float32), 0.0)
                for i in range(slices)]

        def step(r, grad):
            n = chunks - 1 - r
            begin = start + n * chunk
            taken = chunk if aligned else jnp.minimum(chunk, count - n * chunk)

            def read(ref, *rest):
                return read_rows(_rows(ref, begin, chunk, *rest), taken)

            slot = jnp.where(r > 0, base + n + 1, left)
            for i in range(slices):
                dh_ref[slot, pl.ds(i * 64, 64), :] = grad[i].astype(dh_ref.dtype)
            # The chunk's tokens last first (an exchange of rows, its products exact: one nonzero term
            # a row, bf16x6's three parts of an fp32 value its 24 bits, which one product of IEEE
            # fp32 would compile to FMA loops on sm80): the readout's terms summed from the chunk's
            # last token, the smallest first, as the portable math sums them
            # (`recurrence_algebra.remembered`), where summed from the first the initial memory's
            # gradient sat 2.3 to 2.4 times its distance from float64 (tests/test_fla_kda.py).
            index = jnp.arange(chunk)
            exchange = (index[:, None] + index[None, :] == chunk - 1).astype(jnp.float32)

            def reversed_rows(x):
                x = x.astype(jnp.float32)
                if exact:
                    return _dot(exchange, x, exact=True)
                return jnp.dot(exchange, x, precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6,
                               preferred_element_type=jnp.float32)

            out_grad = read(do_ref)
            dv = sum(_dot(read(k_ref, pl.ds(i * 64, 64)), grad[i].astype(k_ref.dtype), exact=exact)
                     for i in range(slices))
            final = begin + taken - 1
            queries = [read(q_ref, pl.ds(i * 64, 64)) for i in range(slices)]
            last_first = reversed_rows(out_grad)
            if per_key:
                dv = dv + read(dv_ref).astype(jnp.float32)
                kept = [jnp.exp2(g_ref[final, pl.ds(i * 64, 64)].astype(jnp.float32))[:, None]
                        for i in range(slices)]
                read_out = [_dot(reversed_rows(query).astype(query.dtype).T, last_first.astype(query.dtype),
                                 exact=exact) for query in queries]
            else:
                gate = _gates(g_ref, begin, taken, chunk)
                last = _row(g_ref, final)
                dv = (dv * jnp.exp2(last - gate)[:, None]
                      + read(dv_ref).astype(jnp.float32))
                decay = jnp.exp2(gate)[:, None]
                kept = [jnp.exp2(last)] * slices
                read_out = [_split_dot(reversed_rows(query.astype(jnp.float32) * decay).T,
                                       last_first.astype(out_grad.dtype), exact=exact)
                            for query in queries]
            write_rows(_rows(dv2_ref, begin, chunk), taken, dv)

            def moved(i):
                written = read(w_ref, pl.ds(i * 64, 64))
                return read_out[i] * scale - _dot(written.T, dv.astype(written.dtype), exact=exact)

            return [grad[i] * kept[i] + moved(i) for i in range(slices)]

        grad = jax.lax.fori_loop(0, chunks, step, grad)

        pl.when(first)(lambda: _stored(dh0_ref, row, grad))


@dataclasses.dataclass(frozen=True)
class ChunkBwdDhu(_Launch):
    """`chunk_bwd_dhu`'s launch (tuning.KernelOp)."""

    revision: int = 2

    NAME = "chunk_bwd_dhu"
    KEYS, VALUES = 'k', 'do'
    # Measured at Qwen3.5-0.8B's shape (8 x 3072, 16 heads of 128, bf16), CUPTI, 2026-10-13: 0/64/4/3
    # 1.09 ms on an A100 and 0.315 on an H100, the fastest on both. One stage for fp32: three took
    # 254,984 bytes of an A100's 166,912 (c58); fla's blocks for 99 KB devices (chunk_delta_h.py:352-353).
    PICKS = _picks(Tiles(0, 64, 4, 3), small=Tiles(0, 64, 4, 1), wide_fp32=Tiles(0, 64, 4, 1),
                   small_fp32=Tiles(0, 32, 4, 1))
    WIDE_KEYS_PICK = Tiles(0, 64, 4, 1)
    """The heuristic past `WIDEST` keys: the memory gradient is held a 64-key slice at a time, so
    256-wide keys stage twice 128-wide ones', and the pick's three stages took 271,368 shared bytes of
    an A100's 166,912 and an H100's 232,448 in 16 bits (2026-10-21); one stage ran on both."""

    def _get_heuristics_config(self, ba) -> Tiles:
        if ba.arguments['k'].shape[-1] > WIDEST:
            return self._tiles(ba, self.WIDE_KEYS_PICK)
        return super()._get_heuristics_config(ba)

    def _fwd(self, q: jax.Array, k: jax.Array, w: jax.Array, g: jax.Array, dht: jax.Array,
             do: jax.Array, dv: jax.Array, chunk_spans: jax.Array, sequence_spans: jax.Array, *, scale: float,
             chunk: int, interpret: bool, config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        bv, warps, stages = config.bv, config.warps, config.stages
        spans, batch, length, _, width, value_heads, columns, group, tokens, aligned = laid(
            q, do, chunk_spans, sequence_spans, chunk)
        held = slots(spans, batch, length, chunk)

        def head(i_v, sh):
            return sh % value_heads

        keys = pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh) // group, 0))
        values = pl.BlockSpec((tokens, None, bv), lambda i_v, sh: (0, head(i_v, sh), i_v))
        rows = pl.BlockSpec((batch, None, width, bv), lambda i_v, sh: (0, head(i_v, sh), 0, i_v))
        per_key = g.ndim == 4
        gated = (pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh), 0)) if per_key
                 else pl.BlockSpec((tokens, None), lambda i_v, sh: (0, head(i_v, sh))))
        dh, dh0, dv2 = pl.pallas_call(
            functools.partial(_dhu_kernel, chunk=chunk, aligned=aligned, length=length, scale=scale,
                              exact=interpret,
                              **({"per_key": True} if per_key else {})),
            grid=(columns // bv, spans.spans.shape[0] * value_heads),
            in_specs=[pl.BlockSpec((None, 5), lambda i_v, sh: (sh // value_heads, 0)), keys, keys,
                      pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh), 0)),
                      gated, rows, values, values],
            out_specs=[pl.BlockSpec((held, None, width, bv), lambda i_v, sh: (0, head(i_v, sh), 0, i_v)),
                       rows,
                       values],
            out_shape=[jax.ShapeDtypeStruct((held, value_heads, width, columns), q.dtype),
                       jax.ShapeDtypeStruct((batch, value_heads, width, columns), jnp.float32),
                       jax.ShapeDtypeStruct((tokens, value_heads, columns), dv.dtype)],
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="fla_chunk_bwd_dhu",
        )(spans.spans, *_flat(q, k, w, g), dht, *_flat(do, dv))
        return (dh, dh0, dv2.reshape(dv.shape)), None


_CHUNK_BWD_DHU = ChunkBwdDhu()


def chunk_bwd_dhu(q, k, w, g, dht, do, dv, *, scale: float, chunk: int = 64,
                  spans: Sequences | None = None, interpret: bool = False):
    """`(dh, dh0, dv2)`: the gradient of the memory each chunk leaves into at its slot `[slots, HV, K,
    V]` (`Sequences`) in `q`'s dtype, of each row's initial memory `[B, HV, K, V]` fp32, and the
    corrected values' full gradient `[B, T, HV, V]` in `dv`'s: fla's
    `chunk_gated_delta_rule_bwd_dhu` (chunk_delta_h.py:740-794). `dht` `[B, HV, K, V]` fp32 is each
    row's final memory's gradient and `dv` the corrected values' from the output alone
    (`chunk_bwd_dv_local`); `g` the cumulated decay `[B, T, HV]`, or `[B, T, HV,
    K]` per key dimension (fla's `gk`). fla's
    A100 choice: BV 32, 4 warps, 3 stages (c53)."""
    return spanned(_CHUNK_BWD_DHU, q, k, w, g, dht, do, dv, chunk=chunk, spans=spans,
                   scale=scale, interpret=interpret)


def _dv_local_kernel(chunk_ref, q_ref, k_ref, g_ref, do_ref, dv_ref, *, chunk: int, aligned: bool,
                     scale: float, bk: int,
                     bv: int, exact: bool):
    """fla's `chunk_bwd_kernel_dv_local` (fla/ops/common/chunk_o.py:452-526) for one chunk of one
    head: the corrected values' gradient from the chunk's own output, `dv = A do` with
    `A[s, t] = scale (k_s . q_t) exp2(g_t - g_s)` for `s <= t`, cast to `do`'s dtype for the product."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        scores = jnp.zeros((chunk, chunk), jnp.float32)
        for block in range(k_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            scores += _dot(read(k_ref, columns), read(q_ref, columns).T, exact=exact) * scale
        gate = _gates(g_ref, start, taken, chunk)
        scores = scores * jnp.exp2(-_pairs(gate, gate))
        index = jnp.arange(chunk)
        scores = jnp.where(index[:, None] <= index[None, :], scores, 0.0).astype(do_ref.dtype)
        for block in range(do_ref.shape[-1] // bv):
            columns = pl.ds(block * bv, bv)
            product = _dot(scores, read(do_ref, columns), exact=exact)
            write_rows(_rows(dv_ref, start, chunk, columns), taken, product)


@dataclasses.dataclass(frozen=True)
class ChunkBwdDvLocal(_Launch):
    """`chunk_bwd_dv_local`'s launch (tuning.KernelOp)."""

    NAME = "chunk_bwd_dv_local"
    KEYS, VALUES = 'k', 'do'
    PICKS = _picks(Tiles(64, 64, 2, 3))

    def _fwd(self, q: jax.Array, k: jax.Array, g: jax.Array, do: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, scale: float, chunk: int, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        bk, bv, warps, stages = config.bk, config.bv, config.warps, config.stages
        spans, _, _, _, width, value_heads, columns, group, tokens, aligned = laid(
            q, do, chunk_spans, sequence_spans, chunk)
        keys = pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0))
        values = pl.BlockSpec((tokens, None, columns), lambda i_t, h: (0, h, 0))
        dv = pl.pallas_call(
            functools.partial(_dv_local_kernel, chunk=chunk, aligned=aligned, scale=scale, bk=min(bk, width),
                              bv=min(bv, columns),
                              exact=interpret),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, keys, keys, pl.BlockSpec((tokens, None), lambda i_t, h: (0, h)), values],
            out_specs=values,
            out_shape=jax.ShapeDtypeStruct((tokens, value_heads, columns), do.dtype),
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="fla_chunk_bwd_dv_local",
        )(spans.chunks, *_flat(q, k, g, do))
        return dv.reshape(do.shape), None


_CHUNK_BWD_DV_LOCAL = ChunkBwdDvLocal()


def chunk_bwd_dv_local(q, k, g, do, *, scale: float, chunk: int = 64, spans: Sequences | None = None,
                       interpret: bool = False):
    """The corrected values' gradient from each chunk's own output, `[B, T, HV, V]` in `do`'s dtype:
    fla's `chunk_bwd_dv_local` (chunk_o.py:640-676). fla tiles `BK` and `BV` at 64 on an A100
    (its CONST_TILING for ampere and ada shared memory) and chose 2 warps, 3 stages there (c53)."""
    return spanned(_CHUNK_BWD_DV_LOCAL, q, k, g, do, chunk=chunk, spans=spans,
                   scale=scale, interpret=interpret)


def _split_dot(a, b, *, exact: bool):
    """`a @ b` with an fp32 operand beside a 16-bit one: the fp32 one as two parts of the other's
    dtype, its rounding and the rounding's remainder, each product exact and summed in fp32, where
    fla takes one TF32 product of both in fp32: two narrow products run as fast as one TF32 product
    and keep 16 bits of the fp32 operand to TF32's 10. Operands of one dtype keep the one product."""
    narrow = min(a.dtype, b.dtype, key=lambda dtype: jnp.dtype(dtype).itemsize)

    def parts(x):
        if x.dtype == narrow:
            return [x]
        high = x.astype(narrow)
        return [high, (x - high.astype(jnp.float32)).astype(narrow)]

    return sum(_dot(left, right, exact=exact) for left in parts(a) for right in parts(b))


def _state_inner_product(memory, memory_grad):
    """fla chunk_o.py:251: multiply h/dh in their storage dtype, then tl.sum widens to fp32.
    In particular, bf16 products round before reduction; widening both inputs omits that round."""
    return jnp.sum((memory * memory_grad).astype(jnp.float32))


def _dqkwg_kernel(chunk_ref, q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, dh_ref, dv_ref, dq_ref,
                  dk_ref, dw_ref, dg_ref, *, chunk: int, aligned: bool, scale: float, bv: int, exact: bool):
    """fla's `chunk_bwd_kernel_dqkwg` (fla/ops/common/chunk_o.py:159-331) for one chunk of one key
    head and `BK` key columns, over the value heads it serves in turn: over the value columns `bv` at
    a time, `ds = do v^T`, `dq = do h^T`, `dk = v dh^T`, `dw = -dv h^T` and `sum(h dh)`; then the
    gates, `ds`'s causal mask, `dq += ds k` and `dk += ds^T q`. The decay's gradient is each token's
    own (`_summed_back`): the scores' terms `ds (q k^T)` through the ranges between their tokens, the
    keys' terms into the memory the chunk leaves through the ranges to its end, and the queries'
    reads of the memory it enters and that memory's decay through the ranges from its start, where
    fla gives each cumulated decay's and sums them back. The value heads' `dq` and `dk` sum in fp32
    and round to their dtype once, where fla rounds each value head's and sums them after. The heads
    are a loop, not unrolled, so a program holds one head's accumulators at a time. The memory the
    chunk enters and its gradient where it leaves are read at their slots (`Sequences`)."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    entered, left = chunk_ref[2], chunk_ref[3]

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        width = q_ref.shape[-1]
        queries, keys = read(q_ref), read(k_ref)
        index = jnp.arange(chunk)
        causal = index[:, None] >= index[None, :]
        strict = index[:, None] > index[None, :]
        last_row = index == taken - 1
        scores = _dot(queries, keys.T, exact=exact)

        def head_grads(head, sums):
            ds = jnp.zeros((chunk, chunk), jnp.float32)
            dq = jnp.zeros((chunk, width), jnp.float32)
            dk = jnp.zeros((chunk, width), jnp.float32)
            dw = jnp.zeros((chunk, width), jnp.float32)
            dg_last = jnp.zeros((), jnp.float32)
            for block in range(v_ref.shape[-1] // bv):
                columns = pl.ds(block * bv, bv)
                value, out_grad = read(v_ref, head, columns), read(do_ref, head, columns)
                local = read(dv_ref, head, columns)
                memory, memory_grad = h_ref[entered, head, :, columns], dh_ref[left, head, :, columns]
                dg_last += _state_inner_product(memory, memory_grad)
                ds += _dot(out_grad, value.T, exact=exact)
                dq += _dot(out_grad, memory.T.astype(out_grad.dtype), exact=exact)
                dk += _dot(value, memory_grad.T.astype(value.dtype), exact=exact)
                dw += _dot(local.astype(value.dtype), memory.T.astype(value.dtype), exact=exact)
            write_rows(_rows(dw_ref, start, chunk, head), taken, -dw)
            gate = _gates(g_ref, start, taken, chunk, 0, head)
            last = _row(g_ref, (start + taken - 1, head))
            dq = dq * jnp.exp2(gate)[:, None] * scale
            dk = dk * jnp.exp2(last - gate)[:, None]
            ds = jnp.where(causal, ds * jnp.exp2(_pairs(gate, gate)), 0.0) * scale
            # The keys' writes into the memory the chunk leaves run through (s, end]: the last row's pairs.
            leaving = jnp.sum(dk * keys.astype(jnp.float32), axis=1)
            pairs = (jnp.where(strict, ds * scores, 0.0)
                     + jnp.where(last_row[:, None] & strict, leaving[None, :], 0.0))
            # The queries' reads of the memory the chunk enters, and that memory's own decay to its end.
            after = (jnp.sum(dq * queries.astype(jnp.float32), axis=1)
                     + jnp.where(last_row, dg_last * jnp.exp2(last), 0.0))
            write_rows(_rows(dg_ref, start, chunk, head), taken, _summed_back(index, pairs, after))
            ds = ds.astype(keys.dtype)
            dq += _dot(ds, keys, exact=exact)
            dk += _dot(ds.T, queries, exact=exact)
            return sums[0] + dq, sums[1] + dk

        zeros = jnp.zeros((chunk, width), jnp.float32)
        dq, dk = jax.lax.fori_loop(0, v_ref.shape[1], head_grads, (zeros, zeros))
        write_rows(_rows(dq_ref, start, chunk), taken, dq)
        write_rows(_rows(dk_ref, start, chunk), taken, dk)


@dataclasses.dataclass(frozen=True)
class ChunkBwdDqkwg(_Launch):
    """`chunk_bwd_dqkwg`'s launch (tuning.KernelOp)."""

    NAME = "chunk_bwd_dqkwg"
    KEYS, VALUES = 'k', 'v'
    # One stage elsewhere: its loop over a key head's value heads holds every block of a head, and a
    # second stage stages the next head's whole, past an RTX's 99 KB at the 9B's two heads a key.
    # Measured at Qwen3.5-0.8B's shape, CUPTI, 2026-10-13: 128/128/8/1 1.33 ms on an A100 (64/64/4/1
    # 2.65, best 128/64/8/1 1.19) and 0.65 on an H100 (64/64/4/1 0.52, 128/64/8/1 1.02).
    PICKS = _picks(Tiles(64, 64, 4, 1), wide_bf16=Tiles(128, 128, 8, 1))

    def _fwd(self, q: jax.Array, k: jax.Array, v: jax.Array, w: jax.Array, g: jax.Array, h: jax.Array,
             dv: jax.Array, do: jax.Array, dh: jax.Array, chunk_spans: jax.Array,
             sequence_spans: jax.Array, *, scale: float, chunk: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        bk, bv, warps, stages = config.bk, config.bv, config.warps, config.stages
        spans, _, _, key_heads, width, value_heads, columns, group, tokens, aligned = laid(
            q, v, chunk_spans, sequence_spans, chunk)
        keys = pl.BlockSpec((tokens, None, bk), lambda i_k, i_h, i_t: (0, i_h, i_k))
        values = pl.BlockSpec((tokens, group, columns), lambda i_k, i_h, i_t: (0, i_h, 0))
        memory = pl.BlockSpec((h.shape[0], group, bk, columns), lambda i_k, i_h, i_t: (0, i_h, i_k, 0))
        dq, dk, dw, dg = pl.pallas_call(
            functools.partial(_dqkwg_kernel, chunk=chunk, aligned=aligned, scale=scale, bv=bv,
                              exact=interpret),
            grid=(width // bk, key_heads, spans.chunks.shape[0]),
            in_specs=[pl.BlockSpec((None, 4), lambda i_k, i_h, i_t: (i_t, 0)), keys, keys, values,
                      pl.BlockSpec((tokens, group), lambda i_k, i_h, i_t: (0, i_h)), memory, values, memory,
                      values],
            out_specs=[keys, keys, pl.BlockSpec((tokens, group, bk), lambda i_k, i_h, i_t: (0, i_h, i_k)),
                       pl.BlockSpec((None, tokens, group), lambda i_k, i_h, i_t: (i_k, 0, i_h))],
            out_shape=[jax.ShapeDtypeStruct((tokens, key_heads, width), q.dtype),
                       jax.ShapeDtypeStruct((tokens, key_heads, width), k.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads, width), w.dtype),
                       jax.ShapeDtypeStruct((width // bk, tokens, value_heads), jnp.float32)],
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="fla_chunk_bwd_dqkwg",
        )(spans.chunks, *_flat(q, k, v, g), h, *_flat(do), dh, *_flat(dv))
        grads = dq.reshape(q.shape), dk.reshape(k.shape), dw.reshape(w.shape), dg.sum(0).reshape(g.shape)
        return grads, None


_CHUNK_BWD_DQKWG = ChunkBwdDqkwg()


def chunk_bwd_dqkwg(q, k, v, w, g, h, dv, do, dh, *, scale: float, chunk: int = 64,
                    spans: Sequences | None = None, interpret: bool = False):
    """`(dq, dk, dw, dg)`: fla's `chunk_bwd_dqkwg` (chunk_o.py:696-743). `q`, `k` `[B, T, H, K]`, `v`
    the corrected values and `do` `[B, T, HV, V]`, `w` `[B, T, HV, K]` (its shape and dtype for `dw`),
    `g` the base-2 cumulated decay, `h` and `dh` the memories and their gradients at their slots
    `[slots, HV, K, V]` (`chunk_fwd_h`, `chunk_bwd_dhu`), `dv` the corrected values' full gradient. One
    program per key head sums its value heads' `dq` and `dk` in fp32 and stores them once, `[B, T,
    H, K]` in `q`'s and `k`'s dtypes; `dw` per value
    head, and `dg` per key block, summed, as fla does. fla tiles `BK`, `BV` at 64 on an A100 and chose
    4 warps, 2 stages (c53)."""
    return spanned(_CHUNK_BWD_DQKWG, q, k, v, w, g, h, dv, do, dh, chunk=chunk, spans=spans,
                   scale=scale, interpret=interpret)


def _summed_back(index, pairs, after):
    """The gradient of each of a chunk's per-token decays, `d/dg_t`, from the gradients of terms through
    ranges of its cumulated decay: `pairs` `[C, C]`, each `(i, j)` a term through the range `(j, i]`
    (strictly lower), and `after` `[C]`, each `s` a term through the range from the chunk's start to
    `s`. Each token sums only the terms whose range holds it, `sum_{j < t <= i} pairs + sum_{s >= t}
    after`, where fla takes the gradient of each cumulated decay (row sums less column sums) and sums
    those back, adding and taking back every term whose range misses the token: fp32's residue where
    the gradient is exactly zero (keys of zero before a token, 1.7e-10 in
    test_a_range_late_in_a_long_decayed_chunk_is_as_exact_as_fla), and the cancellation of terms
    many times the gradient where it is not. Each row's pairs before `t` are its running sum less
    its `t`th, exactly zero where no pair before `t` is nonzero."""
    through = jnp.cumsum(pairs, axis=1) - pairs
    return jnp.sum(jnp.where(index[:, None] >= index[None, :], through + after[:, None], 0.0), axis=0)


def _wy_bwd_kernel(chunk_ref, k_ref, v_ref, beta_ref, g_ref, a_ref, dw_ref, du_ref, *refs, chunk: int,
                   aligned: bool,
                   bk: int, bv: int, group: int | None, exact: bool):
    """fla's `prepare_wy_repr_bwd_kernel` (fla/ops/gated_delta_rule/wy_fast.py:119-243) for one
    chunk of one head. With `X = (I + L)^-1` (fla's `A`), `w = X (k beta exp2(g))` and
    `u = X (v beta)`: their gradients through `X^T`, `dX = dw (k beta exp2 g)^T + du (v beta)^T`, then
    `dL = -X^T tril(dX, -1) X^T` gated by `exp2(g_s - g_t)`, and from it the keys', beta's and the
    decay's gradients. `dk`'s part through `w` is recomputed beside its part through `L` and the
    two stored once, where fla stores the first, rounded, and reads it back behind a barrier
    (wy_fast.py:223): one rounding, and no store that other threads must see before a later read.
    `group`, the value heads per key head, is given when the keys' gradient so far (`refs[0]`, the
    query-side kernel's) is an operand: it joins the sum before that store in the first value head
    of each key head's group, where fla adds the two stored gradients in a pass of their own
    (chunk.py:244). A group of more than one stores each head's part in fp32 (`dk_ref`'s dtype),
    which the caller sums and rounds once."""
    earlier_ref, (dk_ref, dv_ref, db_ref, dg_ref) = (refs[0], refs[1:]) if group else (None, refs)
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    first_of_group = pl.program_id(1) % (group or 1) == 0

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        inverse_t = read(a_ref).T
        beta = read(beta_ref)
        gate = _gates(g_ref, start, taken, chunk)
        decay = jnp.exp2(gate)
        width = k_ref.shape[-1]

        # Each block loop is a loop, not unrolled, so Triton pipelines its loads and holds one block's
        # values at a time: unrolled, the A100 kernel spilled 406 words and ran 1.93 ms against 1.03.
        def through_keys(block, carry):
            d_x, d_beta, d_gate = carry
            columns = pl.ds(block * bk, bk)
            key = read(k_ref, columns)
            scaled = key * (beta * decay)[:, None]
            w_grad = read(dw_ref, columns)
            d_scaled = _dot(inverse_t, w_grad, exact=exact)
            return (d_x + _dot(w_grad, scaled.T.astype(w_grad.dtype), exact=exact),
                    d_beta + jnp.sum(d_scaled * key * decay[:, None], axis=1),
                    d_gate + jnp.sum(d_scaled * scaled, axis=1))

        def through_values(block, carry):
            d_x, d_beta = carry
            columns = pl.ds(block * bv, bv)
            value = read(v_ref, columns)
            scaled = (value * beta[:, None]).astype(value.dtype)
            u_grad = read(du_ref, columns)
            d_scaled = _dot(inverse_t, u_grad, exact=exact)
            write_rows(_rows(dv_ref, start, chunk, columns), taken, d_scaled * beta[:, None])
            return (d_x + _dot(u_grad, scaled.T, exact=exact), d_beta + jnp.sum(d_scaled * value, axis=1))

        empty = jnp.zeros((chunk,), jnp.float32)
        d_x, d_beta, d_gate = jax.lax.fori_loop(0, width // bk, through_keys,
                                                (jnp.zeros((chunk, chunk), jnp.float32), empty, empty))
        d_x, d_beta = jax.lax.fori_loop(0, v_ref.shape[-1] // bv, through_values, (d_x, d_beta))
        index = jnp.arange(chunk)
        strict = index[:, None] > index[None, :]
        d_x = jnp.where(strict, d_x, 0.0)
        d_x = _dot(d_x.astype(inverse_t.dtype), inverse_t, exact=exact)
        d_x = _dot(inverse_t, d_x.astype(inverse_t.dtype), exact=exact)
        d_x = d_x * jnp.exp2(_pairs(gate, gate))
        d_l = jnp.where(strict, -d_x, 0.0).astype(k_ref.dtype)
        def keys_grad(block, carry):
            products, d_beta = carry
            columns = pl.ds(block * bk, bk)
            key = read(k_ref, columns)
            scaled = key * beta[:, None]
            # k k^T first, as fla takes it: taken after the dk store, sm90's pipelined loop returned
            # a different dg on each call of one executable (FitBuzzard, dew-h100).
            products = products + _dot(key, key.T, exact=exact)
            d_key = _dot(d_l, key, exact=exact)
            through_w = _dot(inverse_t, read(dw_ref, columns), exact=exact) * (decay * beta)[:, None]
            # d_l^T (k beta), fla's trans(trans(kb) dA), as one product into the accumulator's layout.
            full = through_w + (d_key * beta[:, None] + _dot(d_l.T, scaled.astype(d_l.dtype), exact=exact))
            if earlier_ref is not None:
                earlier = read(earlier_ref, columns).astype(jnp.float32)
                full += jnp.where(first_of_group, earlier, 0.0)
            write_rows(_rows(dk_ref, start, chunk, columns), taken, full)
            return products, d_beta + jnp.sum(d_key * key, axis=1)

        products, d_beta = jax.lax.fori_loop(0, width // bk, keys_grad,
                                             (jnp.zeros((chunk, chunk), jnp.float32), d_beta))
        write_rows(_rows(db_ref, start, chunk), taken, d_beta)
        weighted = d_l.astype(jnp.float32) * (products * beta[:, None])
        write_rows(_rows(dg_ref, start, chunk), taken, _summed_back(index, weighted, d_gate))


@dataclasses.dataclass(frozen=True)
class PrepareWyReprBwd(_Launch):
    """`prepare_wy_repr_bwd`'s launch (tuning.KernelOp)."""

    NAME = "prepare_wy_repr_bwd"
    KEYS, VALUES = 'k', 'v'
    # Eight warps for fp32 on every GPU: at TF32x3 four took ptxas 40 s on sm80 and 66 s on sm89, eight
    # 17 s; one stage on a 99 KB block, whose looped blocks stage the next one, 102,400 bytes at two
    # (sm89).
    PICKS = _picks(Tiles(64, 64, 4, 2), wide_fp32=Tiles(64, 64, 8, 2), small_fp32=Tiles(64, 64, 8, 1))

    def _fwd(self, k: jax.Array, v: jax.Array, beta: jax.Array, g: jax.Array, a: jax.Array,
             dw: jax.Array, du: jax.Array, dk: jax.Array | None, chunk_spans: jax.Array,
             sequence_spans: jax.Array, *, chunk: int, interpret: bool, config: Tiles,
             return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        bk, bv, warps, stages = config.bk, config.bv, config.warps, config.stages
        spans, batch, length, key_heads, width, value_heads, columns, group, tokens, aligned = laid(
            k, v, chunk_spans, sequence_spans, chunk)

        headed = functools.partial(_headed, tokens)

        gate = pl.BlockSpec((tokens, None), lambda i_t, h: (0, h))
        keys = pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0))
        earlier = () if dk is None else (dk,)
        parts, dv, db, dg = pl.pallas_call(
            functools.partial(_wy_bwd_kernel, chunk=chunk, aligned=aligned, bk=min(bk, width),
                              bv=min(bv, columns), group=group if earlier else None, exact=interpret),
            grid=(spans.chunks.shape[0], value_heads),
            in_specs=[_CHUNK, keys, headed(columns), gate, gate, headed(chunk), headed(width),
                      headed(columns),
                      *(keys for _ in earlier)],
            out_specs=[headed(width), headed(columns), gate, gate],
            # A group's parts stay fp32 until their sum rounds once.
            out_shape=[jax.ShapeDtypeStruct((tokens, value_heads, width),
                                            jnp.float32 if group > 1 else k.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads, columns), v.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads), beta.dtype),
                       jax.ShapeDtypeStruct((tokens, value_heads), g.dtype)],
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="fla_prepare_wy_repr_bwd",
        )(spans.chunks, *_flat(k, v, beta, g, a, dw, du, *earlier))
        dk = parts.reshape(batch, length, key_heads, group, width).sum(3).astype(k.dtype) if group > 1 else \
            parts.reshape(k.shape)
        return (dk, dv.reshape(v.shape), db.reshape(beta.shape), dg.reshape(g.shape)), None


_PREPARE_WY_REPR_BWD = PrepareWyReprBwd()


def prepare_wy_repr_bwd(k, v, beta, g, a, dw, du, dk=None, *, chunk: int = 64,
                        spans: Sequences | None = None, interpret: bool = False):
    """`(dk, dv, db, dg)` through the WY representation: fla's `prepare_wy_repr_bwd` (wy_fast.py:
    296-343). `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `beta`, `g` (the cumulated
    decay) `[B, T, HV]`, `a` the inverse
    per chunk `[B, T, HV, BT]`, `dw` and `du` the gradients of `w` and `u`, and `dk` `[B, T, H, K]`
    the keys' gradient so far, which the returned one includes, added in fp32 before its one
    rounding; without it, fla's call. `dk` is summed over each key head's group. fla tiles `BK`,
    `BV` at 64 on an A100 and chose 4 warps, 2 stages (c53)."""
    return spanned(_PREPARE_WY_REPR_BWD, k, v, beta, g, a, dw, du, dk, chunk=chunk, spans=spans,
                   interpret=interpret)


def _o_kernel(chunk_ref, q_ref, k_ref, v_ref, h_ref, g_ref, o_ref, *, chunk: int, aligned: bool,
              scale: float, bk: int, exact: bool):
    """fla's `chunk_fwd_kernel_o` (fla/ops/common/chunk_o.py:43-147) for one chunk of one head and
    `BV` value columns: `o = scale * ((q h) exp2(g) + tril((q k^T) exp2(g_s - g_t)) v)`, the key
    columns walked `bk` at a time into fp32 accumulators, `A` cast to the values' dtype for its
    product, as fla does. fp32 products are IEEE here (`_dot`'s `ieee`), where the other kernels take
    BF16x6: the output reads them directly, and at TF32x3, the others' products before, a Qwen3.5
    hybrid's fp32 scores on an H100 sat 2.03 times their fp32 reference's distance from float64, 1.17
    at IEEE. The memory the chunk enters is read at its slot (`Sequences`)."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    entered = chunk_ref[2]

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return read_rows(_rows(ref, start, chunk, *rest), taken)

        out = jnp.zeros((chunk, v_ref.shape[-1]), jnp.float32)
        scores = jnp.zeros((chunk, chunk), jnp.float32)
        for block in range(q_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            query = read(q_ref, columns)
            out += _dot(query, h_ref[entered, columns, :], exact=exact, ieee=True)
            scores += _dot(query, read(k_ref, columns).T, exact=exact, ieee=True)
        gate = _gates(g_ref, start, taken, chunk)
        out = out * jnp.exp2(gate)[:, None]
        scores = scores * jnp.exp2(_pairs(gate, gate))
        index = jnp.arange(chunk)
        scores = jnp.where(index[:, None] >= index[None, :], scores, 0.0)
        value = read(v_ref)
        out = out * scale + _dot(scores.astype(value.dtype), value, exact=exact, ieee=True) * scale
        write_rows(_rows(o_ref, start, chunk), taken, out)


@dataclasses.dataclass(frozen=True)
class ChunkFwdO(_Launch):
    """`chunk_fwd_o`'s launch (tuning.KernelOp)."""

    NAME = "chunk_fwd_o"
    KEYS, VALUES = 'k', 'v'
    PICKS = _picks(Tiles(128, 128, 8, 3))

    def _fwd(self, q: jax.Array, k: jax.Array, v: jax.Array, h: jax.Array, g: jax.Array,
             chunk_spans: jax.Array, sequence_spans: jax.Array, *, scale: float, chunk: int,
             interpret: bool, config: Tiles, wide: bool = False, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = Sequences(chunk_spans, sequence_spans)
        bk, bv, warps, stages = config.bk, config.bv, config.warps, config.stages
        batch, length, key_heads, width = q.shape
        value_heads, columns = v.shape[2], v.shape[3]
        group = value_heads // key_heads
        tokens = batch * length
        aligned = _aligned(spans, batch, length, chunk)
        keyed = pl.BlockSpec((tokens, None, width), lambda i_v, i_t, i_h: (0, i_h // group, 0))
        valued = pl.BlockSpec((tokens, None, bv), lambda i_v, i_t, i_h: (0, i_h, i_v))
        out = pl.pallas_call(
            functools.partial(_o_kernel, chunk=chunk, aligned=aligned, scale=scale, bk=min(bk, width),
                              exact=interpret),
            grid=(columns // bv, spans.chunks.shape[0], value_heads),
            in_specs=[pl.BlockSpec((None, 4), lambda i_v, i_t, i_h: (i_t, 0)), keyed, keyed, valued,
                      pl.BlockSpec((h.shape[0], None, width, bv), lambda i_v, i_t, i_h: (0, i_h, 0, i_v)),
                      pl.BlockSpec((tokens, None), lambda i_v, i_t, i_h: (0, i_h))],
            out_specs=valued,
            out_shape=jax.ShapeDtypeStruct((tokens, value_heads, columns), jnp.float32 if wide else v.dtype),
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="fla_chunk_fwd_o",
        )(spans.chunks, *_flat(q, k, v), h, *_flat(g))
        return out.reshape(v.shape), None


_CHUNK_FWD_O = ChunkFwdO()


def chunk_fwd_o(q, k, v, h, g, *, scale: float, chunk: int = 64, spans: Sequences | None = None,
                interpret: bool = False, wide: bool = False):
    """Each chunk's output, `[B, T, HV, V]` in `v`'s dtype, fp32 where `wide`, the readout unrounded for
    an epilogue after it: fla's `chunk_fwd_o` (chunk_o.py:529-575).

    `q`, `k` `[B, T, H, K]`, `v` the corrected values `[B, T, HV, V]`, `h` the memories the chunks
    enter at their slots `[slots, HV, K, V]` (`chunk_fwd_h`), `g` the base-2 log decay cumulated within each
    chunk (`recurrence.cumulated`) `[B, T, HV]` fp32. Grouped queries: value head `i` reads
    key head `i // (HV // H)`. `T` is a whole number of chunks; `bk`, `bv`, `warps` and `stages` are
    one of fla's configs (`BK, BV` 128/128 at 8 warps, 64/64 at 4, 32/32 at 2, all 3 stages)."""
    return spanned(_CHUNK_FWD_O, q, k, v, h, g, chunk=chunk, spans=spans,
                   scale=scale, interpret=interpret, wide=wide)


def _rule(q, k, v, g, beta, h0, cu_seqlens, scale: float, chunk: int, interpret: bool, keep: bool = True,
          wide: bool = False):
    return _rule_fwd(q, k, v, g, beta, h0, cu_seqlens, scale, chunk, interpret, keep, wide)[0]


def _rule_fwd(q, k, v, g, beta, h0, cu_seqlens, scale: float, chunk: int, interpret: bool, keep: bool = True,
              wide: bool = False):
    """fla's `chunk_gated_delta_rule_fwd` (fla/ops/gated_delta_rule/chunk.py:33-123): the decay
    cumulated in base 2, the inverse, `w` and `u`, the memory across chunks, then the output. Saves
    the cumulated decay and the inverse, as fla does, and where
    `keep` its memory: `w`, the memory each chunk enters and the corrected values, which fla
    recomputes in its backward from the initial memory, as the rule does without `keep` (a remat rung
    recomputing 'delta_memory', `dew.nn.backbones.decoder_block.RematPolicy.recompute`). Each is
    named for a block's remat policy, which keeps or recomputes it."""
    return prefixed_fwd(cumulated_fwd, q, k, v, g, beta, h0, cu_seqlens, scale, chunk, interpret, keep, wide)


def prefixed_fwd(cumulated, q, k, v, g, beta, h0, cu_seqlens, scale: float, chunk: int, interpret: bool,
                 keep: bool, wide: bool, *, compensated: bool = False):
    """A port's forward from the natural log decay: the sequences `cu_seqlens` describes, the decay
    cumulated in base 2 within each chunk (`recurrence.cumulated`; its `(hi, lo)` halves where
    `compensated`), named 'delta_decay' for a block's remat policy, then the port's `cumulated` forward
    on it (this module's `cumulated_fwd`, or `fla_kda`'s)."""
    spans = sequences(q.shape[0], q.shape[1], cu_seqlens, chunk=chunk)
    g = checkpoint_name(recurrence.cumulated(g, scale=RCP_LN2, chunk=chunk, spans=spans,
                                             compensated=compensated, interpret=interpret),
                        'delta_decay')
    return cumulated(q, k, v, g, beta, h0, spans, scale, chunk, interpret, keep, wide)


def cumulated_fwd(q, k, v, g, beta, h0, spans: Sequences, scale: float, chunk: int, interpret: bool,
                  keep: bool = True, wide: bool = False):
    """`_rule_fwd` from the decay cumulated in base 2 within each chunk (`recurrence.cumulated`), which is
    all the rest of the rule reads of it: the inverse, `w` and `u`, the memory across chunks, then the
    output, and what the backward saves. The prefix is a value the port rounds and holds, a node of
    its graph (the accuracy gate's source-aware check, `assert_graph_as_exact`), and this the port's
    arithmetic on it."""
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    a = checkpoint_name(kkt_solve(k, g, beta, **kernels), 'delta_inverse')
    w, u = recompute_w_u(k, v, beta, a, g, **kernels)
    h, corrected, final = chunk_fwd_h(k, w, u, g, h0, **kernels)
    o = chunk_fwd_o(q, k, corrected, h, g, scale=scale, wide=wide, **kernels)
    memory = checkpoint_name((w, h, corrected), 'delta_memory') if keep else None
    return (o, final), (q, k, v, g, beta, a, h0, memory, spans)


def _rule_bwd(scale: float, chunk: int, interpret: bool, keep: bool, wide: bool, residual, cotangent):
    """fla's `chunk_gated_delta_rule_bwd` (chunk.py:126-250): `w`, `u` and the memory the forward kept,
    or recomputed from the initial memory as fla does, then the output's local value gradient, the
    reverse recurrence, the query, key, `w` and decay gradients, the WY backward, and the decay's
    gradient, each token's summed over the terms through it (`_summed_back`), which a sequence's first
    token with no memory before it takes as exactly zero. A `wide` output's cotangent is taken in the
    values' dtype, as the kernels take an output's."""
    q, k, v, g, beta, a, h0, memory, spans = residual
    do, dht = cotangent
    do = do.astype(v.dtype) if wide else do
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    if keep:
        w, h, corrected = memory
    else:
        w, u = recompute_w_u(k, v, beta, a, g, **kernels)
        h, corrected, _ = chunk_fwd_h(k, w, u, g, h0, **kernels)
    dv = chunk_bwd_dv_local(q, k, g, do, scale=scale, **kernels)
    dh, dh0, dv = chunk_bwd_dhu(q, k, w, g, dht.astype(jnp.float32), do, dv, scale=scale, **kernels)
    dq, dk, dw, dg = chunk_bwd_dqkwg(q, k, corrected, w, g, h, dv, do, dh, scale=scale, **kernels)
    dk, dv, db, dg2 = prepare_wy_repr_bwd(k, v, beta, g, a, dw, dv, dk, **kernels)
    return dq, dk, dv, (dg + dg2).astype(jnp.float32), db, dh0, None


chunk_gated_delta_rule = jax.custom_vjp(_rule, nondiff_argnums=(7, 8, 9, 10, 11))
"""fla's chunked gated delta rule on the ported kernels: `(o, final)` for `q`, `k` `[B, T, H, K]`
(normalized as the caller wants them), `v` `[B, T, HV, V]`, the per-token natural log decay `g`
`[B, T, HV]` fp32, `beta` `[B, T, HV]` and each row's initial memory `h0` `[B, HV, K, V]` fp32, the
final one each row's. `cu_seqlens` (`[N + 1]` int32 over the `B * T` tokens, fla's varlen) splits the
rows into sequences, the memory starting from zero at each one a row does not begin with; None runs
each row as one. `keep` keeps the rule's memory for the backward, which otherwise recomputes it, as
fla's does (`_rule_fwd`). Its numbers are fla's: bf16 operands store `A`, `w`, `u`, `h` and the
corrected values in bf16, with fp32 accumulation and fp32 decay, the memory fp32 across chunks; the
output in the values' dtype, or fp32 where `wide`, for an epilogue to read unrounded."""
chunk_gated_delta_rule.defvjp(_rule_fwd, _rule_bwd)
