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
V]`, per-token gates `[B, T, HV]`, the memory each chunk enters `[B, T // BT, HV, K, V]`; a program
reads its chunk of its head where it sits, as fla's pointer arithmetic does.

CONFIGS keys the GPU generation and activation dtype: fla's own A100 choices on the generations
whose blocks hold an A100's shared memory, and on the 99 KB ones fla's narrower picks for those
devices. Each config compiles within its GPU's block at the training shapes and at the widest keys
(tests/test_kernel_configs.py); `dew.nn.linear.fla_rule` shapes any other call to them. Every public
kernel accepts explicit block/warp/stage overrides for the lab.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl

from dew.nn.kernels import generation

RCP_LN2 = 1.4426950216
"""fla's 1 / ln 2: its kernels take the cumulated decay in base 2 and exponentiate with exp2."""

_SM80_BF16 = {
    "chunk_local_cumsum": {"warps": 2, "stages": 1},
    "kkt_solve": {"bk": 64, "warps": 1, "stages": 3},
    "recompute_w_u": {"bk": 64, "bv": 64, "warps": 2, "stages": 4},
    "chunk_fwd_h": {"bv": 32, "warps": 4, "stages": 3},
    "chunk_fwd_o": {"bk": 128, "bv": 128, "warps": 8, "stages": 3},
    "chunk_bwd_dv_local": {"bk": 64, "bv": 64, "warps": 2, "stages": 3},
    "chunk_bwd_dhu": {"bv": 32, "warps": 4, "stages": 3},
    # One stage: its loop over a key head's value heads holds every block of a head, and a second
    # stage stages the next head's whole, past an RTX's 99 KB at the 9B's two heads a key.
    "chunk_bwd_dqkwg": {"bk": 64, "bv": 64, "warps": 4, "stages": 1},
    "prepare_wy_repr_bwd": {"bk": 64, "bv": 64, "warps": 4, "stages": 2},
}
# Eight warps for fp32's WY backward on every GPU: at TF32x3 four took ptxas 40 s on sm80 and 66 s on
# sm89, eight 17 s.
_FP32_WY = {"prepare_wy_repr_bwd": {"bk": 64, "bv": 64, "warps": 8, "stages": 2}}
_SM80_FP32 = {**_SM80_BF16, **_FP32_WY,
              "chunk_fwd_h": {"bv": 64, "warps": 4, "stages": 2},
              # One stage: three took 254,984 bytes of an A100's 166,912 (c58).
              "chunk_bwd_dhu": {"bv": 64, "warps": 4, "stages": 1}}
# fla's stage and block lists for devices under an A100's shared memory (chunk_delta_h.py:43-44,
# 352-353), with two warps for the forward h.
_SMALL_BF16 = {**_SM80_BF16,
               "chunk_fwd_h": {"bv": 32, "warps": 2, "stages": 2},
               "chunk_bwd_dhu": {"bv": 64, "warps": 4, "stages": 1}}
_SMALL_FP32 = {**_SMALL_BF16, **_FP32_WY,
               "chunk_fwd_h": {"bv": 64, "warps": 2, "stages": 1},
               "chunk_bwd_dhu": {"bv": 32, "warps": 4, "stages": 1}}
# The RTX PRO 6000's (sm120) bf16 picks, each the fastest of its sweep on a G4 at the 0.8B and 9B
# shapes (kernel lab, 2026-10-12, 4653d4451), ms at 0.8B / 9B against the 99 KB table's: forward h
# 0.447 / 0.423 against 0.463 / 0.445, fla 0.471 / 0.449; dhu 1.06 / 1.02 against 2.05 / 3.24;
# dqkwg 0.85 / 0.77 against 1.36 / 1.36, fla 0.91 / 1.01; WY backward 0.88 / 0.97 against 1.77 /
# 1.67. The WY backward stores dk once, so it has no store to race and takes four warps.
_BLACKWELL_BF16 = {**_SMALL_BF16,
                   "chunk_fwd_h": {"bv": 32, "warps": 4, "stages": 2},
                   "chunk_bwd_dhu": {"bv": 32, "warps": 4, "stages": 1},
                   "chunk_bwd_dqkwg": {"bk": 128, "bv": 64, "warps": 8, "stages": 1},
                   "prepare_wy_repr_bwd": {"bk": 64, "bv": 64, "warps": 4, "stages": 2}}


def _declared(name: str, target: generation.GpuTarget, dtype: str) -> dict:
    wide = (target.shared_limit or 0) >= (generation.TARGETS["sm80"].shared_limit or 0)
    if name == "sm120" and dtype == "bfloat16":
        return _BLACKWELL_BF16
    return {("bfloat16", True): _SM80_BF16, ("float32", True): _SM80_FP32,
            ("bfloat16", False): _SMALL_BF16, ("float32", False): _SMALL_FP32}[dtype, wide]


# Keys wider than 128 hold a wider memory, so their kernels take at most these fields, each a choice
# fla's autotuner offers: on an A100 three-stage dhu took 177,160 bytes of 166,912 at 256-wide keys
# (CPU compile).
_WIDE_KEYS = {"chunk_bwd_dhu": {"stages": 2}}

CONFIGS = {(name, dtype): _declared(name, target, dtype)
           for name, target in generation.TARGETS.items()
           if name.startswith("sm") and int(name[2:]) >= generation.BF16_GPU
           for dtype in ("bfloat16", "float32")}
"""Each kernel's blocks, warps and stages by (generation, activation dtype), for every GPU that runs
Triton: an A100's wherever a block holds as much shared memory, the 99 KB tables elsewhere."""

WIDEST = 128
"""The widest values each of the port's calls takes: its configs compile for the 0.8B's and 9B's
128-wide heads (tests/fla_gdn_support.py's COMPILED_SHAPES); wider values split into more heads."""

WIDEST_KEYS = 256
"""The widest 16-bit keys the port takes, fla's own bound (chunk_delta_h.py's four 64-wide blocks of
the memory), for which its configs compile too; keys cannot split. fp32 keys take at most WIDEST: at
256 an A100's fp32 forward h needs 172,292 shared bytes at its narrowest tile and an RTX 4000's fp32
WY backward compiles past the suite's minute (CPU compile)."""


def _rule_shape(key, value, chunk):
    return {"width": key.shape[-1], "columns": value.shape[-1], "chunk": chunk}


def _config_key(dtype, target: str | None = None) -> tuple[str, str]:
    """The CONFIGS key a call reads: its GPU's, an A100's where the kernels interpret off a GPU, and
    the 99 KB tables' (sm89's) on a GPU the targets do not list, whose block holds at least that; and
    its activation dtype, fp16 taking bf16's, whose itemsize it shares."""
    target = generation.device_generation() if target is None else target
    name = jnp.dtype(dtype).name
    name = "bfloat16" if name == "float16" else name
    if not target.startswith("sm"):
        return "sm80", name
    return (target if (target, name) in CONFIGS else "sm89"), name


def kernel_config(kernel: str, shape: Mapping[str, int], dtype, *, target: str | None = None,
                  table: Mapping[tuple[str, str], Mapping[str, Mapping[str, int]]] | None = None,
                  **overrides: int | None) -> dict[str, int]:
    """`kernel`'s config for `dtype` on `target` (`_config_key`) from `table`, keyed as `CONFIGS` (where
    None, `CONFIGS`; fla_kda's for Kimi's delta attention), capped for keys wider than 128
    (`_WIDE_KEYS`); overrides (the lab's) replace its fields. Blocks are clipped to the
    keys' `width` and values' `columns`."""
    config = dict((CONFIGS if table is None else table)[_config_key(dtype, target)][kernel])
    if shape["width"] > WIDEST:
        config.update({key: min(config[key], cap) for key, cap in _WIDE_KEYS.get(kernel, {}).items()})
    config.update({key: value for key, value in overrides.items() if value is not None})
    return {key: min(value, shape["width"] if key == "bk" else shape["columns"]) if key in ("bk", "bv")
            else value for key, value in config.items()}


class Sequences(NamedTuple):
    """The token spans fla's varlen kernels read (its `cu_seqlens`, `chunk_indices` and `chunk_offsets`,
    fla/ops/utils/index.py) over the `[B * T]` tokens of a call's rows. `chunks` `[NT, 2]` is each
    chunk's first token and count, in order, a chunk never crossing a sequence and those past the
    last empty; `spans` `[N, 3]` is each sequence's first token, count and first chunk. Sequences never
    cross rows, so a row's first sequence takes its initial memory and its last gives its final one."""
    chunks: jax.Array | np.ndarray
    spans: jax.Array | np.ndarray


def sequences(batch: int, length: int, cu_seqlens=None, *, chunk: int = 64) -> Sequences:
    """The spans of `cu_seqlens` (`[N + 1]` int32 over the `batch * length` tokens, each row's
    boundaries among them), or of the rows themselves where None. A traced `cu_seqlens` takes the most
    chunks its N sequences can split `batch * length` tokens into."""
    # The rows' own spans are constants; a traced `cu_seqlens` takes the same arithmetic in XLA.
    xp = np if cu_seqlens is None else jnp
    if cu_seqlens is None:
        cu_seqlens = np.arange(batch + 1, dtype=np.int32) * length
        count = batch * -(-length // chunk)
    else:
        count = -(-batch * length // chunk) + cu_seqlens.shape[0] - 1
    first, lengths = cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1]
    offsets = xp.concatenate([xp.zeros(1, np.int32), xp.cumsum(-(-lengths // chunk))]).astype(np.int32)
    index = xp.arange(count, dtype=np.int32)
    owner = xp.clip(xp.searchsorted(offsets, index, side="right") - 1, 0, lengths.shape[0] - 1)
    start = first[owner] + (index - offsets[owner]) * chunk
    taken = xp.where(index < offsets[-1], xp.minimum(chunk, first[owner] + lengths[owner] - start), 0)
    return Sequences(xp.stack([start, taken], axis=1).astype(np.int32),
                     xp.stack([first, lengths, offsets[:-1]], axis=1).astype(np.int32))


def _aligned(spans: Sequences, batch: int, length: int, chunk: int) -> bool:
    """Whether every chunk starts on a multiple of `chunk`: the rows' own spans over whole chunks."""
    return (length % chunk == 0 and spans.spans.shape[0] == batch
            and spans.chunks.shape[0] == batch * length // chunk)


def _chunk_of(chunk_ref, chunk: int, aligned: bool):
    """A chunk's first token and count (`Sequences.chunks`). Where `aligned` every chunk is whole and
    starts on a multiple of `chunk`, as fla's fixed-length chunks do: the count is the constant
    `chunk`, so its loads and stores take no mask (`_read`), and Triton is told the alignment. A
    packed document's chunks start anywhere and end where it does, as fla's varlen ones."""
    start, taken = chunk_ref[0], chunk_ref[1]
    return (pl.multiple_of(start, chunk), chunk) if aligned else (start, taken)


def _rows(ref, start, size, *rest):
    """The `size` rows from `start` of `ref`, with whatever trailing index `rest` takes."""
    return ref.at[(pl.ds(start, size), *rest)]


def _row_mask(view, count):
    mask = (jnp.arange(view.shape[0]) < count).reshape((-1,) + (1,) * (len(view.shape) - 1))
    return jnp.broadcast_to(mask, view.shape)


def _read(view, count):
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
    values = _read(view, taken - offset).astype(jnp.float32)
    if isinstance(taken, int) and taken - offset >= size:
        return values
    last = g_ref[(start + taken - 1, *rest)].astype(jnp.float32)
    return jnp.where(_row_mask(view, taken - offset), values, last)


def _write(view, count, value):
    """`value` into `view`'s rows below `count`, none past them."""
    from jax.experimental.pallas import triton as plgpu

    if isinstance(count, int) and count >= view.shape[0]:
        view[...] = value.astype(view.dtype)
        return
    plgpu.store(view, value.astype(view.dtype), mask=_row_mask(view, count))


_CHUNK = pl.BlockSpec((None, 2), lambda i_t, *_: (i_t, 0))
"""A per-chunk kernel's own chunk: its first token and count (`Sequences.chunks`)."""


def _flat(*arrays):
    """`[B, T, ...]` arrays as `[B * T, ...]`, the token axis fla's varlen kernels address."""
    return tuple(x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]) for x in arrays)


def _cumsum_kernel(chunk_ref, s_ref, o_ref, *, chunk: int, aligned: bool, scale: float | None, reverse: bool):
    """fla's `chunk_local_cumsum_scalar_kernel` (fla/ops/utils/cumsum.py:33-73) for one chunk of one
    head: the inclusive cumulative sum over its tokens in fp32, of each column of a gate per key
    dimension (fla's vector kernel, :90-133) too, from the end where `reverse`, then `scale`."""
    start, count = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(count > 0)
    def _():
        step = _read(_rows(s_ref, start, chunk), count).astype(jnp.float32)
        total = jnp.cumsum(step, axis=0)
        if reverse:
            total = -total + jnp.sum(step, axis=0) + step
        if scale is not None:
            total = total * scale
        _write(_rows(o_ref, start, chunk), count, total)


def chunk_local_cumsum(g, *, chunk: int = 64, scale: float | None = None, reverse: bool = False,
                       spans: Sequences | None = None, warps: int | None = None, stages: int | None = None,
                       interpret: bool = False):
    """`g` `[B, T, H]`, or `[B, T, H, K]` per key dimension, cumulated within each chunk of `chunk`
    tokens of each sequence (`spans`, the rows where None), fp32: fla's `chunk_local_cumsum`
    (cumsum.py:262-341), `scale` RCP_LN2 for the forward's base-2 decay, `reverse` for its gradient.
    fla autotunes only `num_warps` over 1, 2, 4 and 8 here."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("chunk_local_cumsum", {"chunk": chunk, "width": 1, "columns": 1}, g.dtype,
                           warps=warps, stages=stages)
    warps, stages = config["warps"], config["stages"]

    batch, length, heads, *columns = (int(size) for size in g.shape)
    spans = sequences(batch, length, chunk=chunk) if spans is None else spans
    aligned = _aligned(spans, batch, length, chunk)
    # A decay per key dimension in 64-column blocks, fla's vector kernel's BS.
    blocks = [min(64, width) for width in columns]
    tiles = [width // block for width, block in zip(columns, blocks, strict=True)]
    spec = pl.BlockSpec((batch * length, None, *blocks), lambda i_t, h, *i_k: (0, h, *i_k))
    out = pl.pallas_call(
        functools.partial(_cumsum_kernel, chunk=chunk, aligned=aligned, scale=scale, reverse=reverse),
        grid=(spans.chunks.shape[0], heads, *tiles),
        in_specs=[_CHUNK, spec], out_specs=spec,
        out_shape=jax.ShapeDtypeStruct((batch * length, heads, *columns), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_local_cumsum",
    )(spans.chunks, *_flat(g))
    return out.reshape(g.shape)


def _dot(a, b, *, exact: bool, ieee: bool = False):
    """`a @ b`, same-dtype operands, with fp32 accumulation; reject mismatches before any interpret-mode
    widening, so CPU tests catch dots Triton rejects. Exact interpretation widens bf16/fp16 inputs
    for CPUs without narrow-input/fp32-accumulating dots.

    Each operand dtype names its products, so no global `jax_default_matmul_precision` changes the
    kernel (the suite's 'highest' made every fp32 product IEEE):
    - bf16 or fp16: exact products summed in fp32 (fla's `tl.dot` of narrow operands); no precision.
    - fp32 with `ieee`: IEEE products, fla's `allow_tf32=False` dots (`recompute_w_u`'s `A @ v`).
    - fp32 otherwise: TF32x3 (`DotAlgorithmPreset.TF32_TF32_F32_X3`), three TF32 products near IEEE
      accuracy where fla's own dots take one TF32 (Triton's default `input_precision` for fp32, and
      `SOLVE_TRIL_DOT_PRECISION`, chunk_fwd.py:21). jax 0.11's Pallas lowers `Precision.HIGH` as
      plain TF32 (pallas/triton/lowering.py `_TF32_PRECISIONS`), so the preset is named. IEEE there
      lowers on sm80 to FMA loops ptxas took 3.5 to over 15 minutes to compile for the WY backward;
      TF32x3 compiles in seconds.
    - interpreted: no precision, the CPU's fp32.
    The kkt merge's TF32 products name `Precision.DEFAULT` themselves (`_kkt_solve_kernel`)."""
    if jnp.dtype(a.dtype) != jnp.dtype(b.dtype):
        raise TypeError(f"Pallas dot operands must have the same dtype, got {a.dtype} and {b.dtype}")
    wide = jnp.dtype(a.dtype) == jnp.float32
    if exact:
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        precision = jax.lax.Precision.HIGHEST if ieee and wide else None
    elif wide:
        precision = jax.lax.Precision.HIGHEST if ieee else jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3
    else:
        precision = None
    return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=precision)


def _tril_inverse(diagonal, index):
    """`(I + D)^-1` of a strictly lower square block `D`, row by row: fla's forward substitution
    (chunk_fwd.py's solve, chunk_intra.py's for Kimi's delta attention). `index` is the block's
    `arange` of rows, which the caller's other masks share."""
    sub = diagonal.shape[0]
    inverse = -diagonal
    for i in range(2, sub):
        row = jnp.sum(jnp.where((index == i)[:, None], -diagonal, 0.0), axis=0)
        row = jnp.where(index < i, row, 0.0)
        row = row + jnp.sum(row[:, None] * inverse, axis=0)
        inverse = jnp.where((index == i)[:, None], row[None, :], inverse)
    return inverse + (index[:, None] == index[None, :])


def _tf32_dot(a, b):
    """`a @ b` of fp32 blocks at TF32, fla's SOLVE_TRIL_DOT_PRECISION on an sm80 or later."""
    return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=jax.lax.Precision.DEFAULT)


def _kkt_solve_kernel(chunk_ref, k_ref, g_ref, beta_ref, a_ref, *, chunk: int, aligned: bool, sub: int,
                      bk: int,
                      exact: bool):
    """fla's `chunk_gated_delta_rule_fwd_kkt_solve_kernel` (fla/ops/gated_delta_rule/chunk_fwd.py:40-
    321) for one chunk of one head: `(I + A)^-1` for `A = tril(beta_s (k_s . k_t) exp2(g_s - g_t), -1)`,
    in `sub`-row blocks (fla's BC = 16 of BT = 64). Every lower block of `k k^T` accumulates over the
    key columns `bk` at a time; the gate and beta scale them; forward substitution inverts each
    diagonal block; and neighbouring inverses merge as `-X_i A_ij X_j`, the merge's products at TF32
    as fla's SOLVE_TRIL_DOT_PRECISION takes them on an sm80 or later."""
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
        return _read(_rows(ref, start + i * sub, sub, *rest), taken - i * sub)

    gates = [_gates(g_ref, start, taken, sub, i * sub) for i in range(count)]
    betas = [part(beta_ref, i).astype(jnp.float32) for i in range(count)]
    index = jnp.arange(sub)
    strict = index[:, None] > index[None, :]

    def keys(i, c):
        return part(k_ref, i, pl.ds(c * bk, bk))

    def block(i, j):
        """`A`'s block (i, j), i >= j: the keys' products, gated and scaled by beta."""
        product = sum(_dot(keys(i, c), keys(j, c).T, exact=exact) for c in range(width // bk))
        decay = jnp.exp2(gates[i][:, None] - gates[j][None, :])
        product = product * (jnp.where(strict, decay, 0.0) if i == j else decay)
        return product * betas[i][:, None]

    blocks = {(i, j): block(i, j) for i in range(count) for j in range(i + 1)}

    def tf32(a, b):
        return _tf32_dot(a, b)

    inverse = {(i, i): _tril_inverse(blocks[i, i], index) for i in range(count)}
    for distance in range(1, count):
        for i in range(distance, count):
            j = i - distance
            # X_ij = -X_ii (sum over j <= m < i of A_im X_mj), fla's merge order.
            inner = sum(tf32(blocks[i, m], inverse[m, j]) for m in range(j, i))
            inverse[i, j] = -tf32(inverse[i, i], inner)
    # fla zero-fills A before its kernel stores the lower blocks; here the blocks above are stored as zeros.
    for i in range(count):
        for j in range(count):
            found = inverse[i, j] if j <= i else jnp.zeros((sub, sub), jnp.float32)
            _write(_rows(a_ref, start + i * sub, sub, pl.ds(j * sub, sub)), taken - i * sub, found)


def kkt_solve(k, g, beta, *, chunk: int = 64, sub: int = 16, bk: int | None = None,
              spans: Sequences | None = None, warps: int | None = None, stages: int | None = None,
              interpret: bool = False):
    """`(I + A)^-1` per chunk, `[B, T, HV, BT]` in `k`'s dtype and zero above the diagonal: fla's
    fused kkt and solve for 64-token chunks (chunk_fwd.py:367-396). `k` `[B, T, H, K]`, `g` the
    base-2 cumulated decay and `beta` `[B, T, HV]`. fla autotunes `BK` over 32 and 64 and warps over
    1, 2 and 4."""
    from jax.experimental.pallas import triton as plgpu

    shape = {"chunk": chunk, "width": k.shape[-1], "columns": 1, "sub": sub}
    config = kernel_config("kkt_solve", shape, k.dtype, bk=bk, warps=warps, stages=stages)
    bk, warps, stages = config["bk"], config["warps"], config["stages"]

    batch, length, key_heads, width = k.shape
    value_heads = beta.shape[2]
    group = value_heads // key_heads
    tokens = batch * length
    spans = sequences(batch, length, chunk=chunk) if spans is None else spans
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
    return solved.reshape(batch, length, value_heads, chunk)


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
            return _read(_rows(ref, start, chunk, *rest), taken)

        beta = read(beta_ref)
        inverse = read(a_ref)
        for block in range(v_ref.shape[-1] // bv):
            columns = pl.ds(block * bv, bv)
            value = read(v_ref, columns)
            scaled = (value * beta[:, None]).astype(value.dtype)
            _write(_rows(u_ref, start, chunk, columns), taken, _dot(inverse, scaled, exact=exact, ieee=True))
        decay = jnp.exp2(_gates(g_ref, start, taken, chunk))
        for block in range(k_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            key = read(k_ref, columns)
            scaled = key * beta[:, None] * decay[:, None]
            product = _dot(inverse, scaled.astype(key.dtype), exact=exact)
            _write(_rows(w_ref, start, chunk, columns), taken, product)


def recompute_w_u(k, v, beta, a, g, *, chunk: int = 64, bk: int | None = None, bv: int | None = None,
                  spans: Sequences | None = None, warps: int | None = None, stages: int | None = None,
                  interpret: bool = False):
    """`(w, u)`, in `k`'s and `v`'s dtypes: fla's `recompute_w_u_fwd` (wy_fast.py:247-296).

    `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `beta` `[B, T, HV]` in the inputs' dtype, `a` the
    inverse `(I + beta K K^T)^-1` per chunk `[B, T, HV, BT]` (fla's `A`, rows by token), `g` the
    base-2 cumulated decay `[B, T, HV]` fp32. fla fixes `BK = BV = 64` and autotunes warps over
    2, 4, 8 and stages over 2, 3, 4."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("recompute_w_u", _rule_shape(k, v, chunk), k.dtype,
                           bk=bk, bv=bv, warps=warps, stages=stages)
    bk, bv, warps, stages = config["bk"], config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = k.shape
    value_heads, columns = v.shape[2:]
    group = value_heads // key_heads
    tokens = batch * length
    spans = sequences(batch, length, chunk=chunk) if spans is None else spans
    aligned = _aligned(spans, batch, length, chunk)

    def headed(size):
        return pl.BlockSpec((tokens, None, size), lambda i_t, h: (0, h, 0))

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
    return w.reshape(batch, length, value_heads, width), u.reshape(v.shape)


def _span(span_ref, length: int, chunk: int, aligned: bool):
    """A sequence's first token, count and first chunk (`Sequences.spans`), its row in a call of rows
    `length` long, and whether it starts its row (taking the row's initial memory) and ends it
    (giving the row's final memory). `aligned` as `_chunk_of`."""
    start, count, offset = span_ref[0], span_ref[1], span_ref[2]
    if aligned:
        # The rows' own spans: each `length` long, whole chunks from a multiple of `chunk`.
        start, count = pl.multiple_of(start, chunk), length
    return start, count, offset, start // length, start % length == 0, (start + count) % length == 0


def _h_kernel(span_ref, k_ref, u_ref, w_ref, g_ref, h0_ref, h_ref, corrected_ref, ht_ref, *, chunk: int,
              aligned: bool,
              length: int, exact: bool, per_key: bool = False):
    """fla's `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (fla/ops/common/chunk_delta_h.py:50-
    343) for one sequence's head and `BV` value columns, its chunks in order. The memory is fp32 in
    64-key slices; each chunk stores the memory it enters in `h`'s dtype, then the corrected values
    `u - w h` (fla's `v_new`; `h` cast to `w`'s dtype for the product), gates them by
    `exp2(g_last - g)` and the memory by `exp2(g_last)`, casts the gated values to `k`'s dtype and
    adds `k^T v` to the memory, the fp32 gated values as two parts of `k`'s dtype (`_split_dot`)
    where fla rounds them once: h remains fp32, and h and the corrected values keep fla's storage.
    `per_key` is fla's USE_GK branch: `g_ref` holds a decay per key dimension `[T, K]`, which gates
    the memory's rows by `exp2(g_last)` and not the values, whose keys carry their own decay. A row's
    first sequence starts from its initial memory and its last stores the final one."""
    start, count, offset, row, first, last_in_row = _span(span_ref, length, chunk, aligned)
    slices = k_ref.shape[-1] // 64

    @pl.when(count > 0)
    def _():
        state = [jnp.where(first, h0_ref[row, pl.ds(i * 64, 64), :].astype(jnp.float32), 0.0)
                 for i in range(slices)]

        def step(n, state):
            begin = start + n * chunk
            taken = chunk if aligned else jnp.minimum(chunk, count - n * chunk)

            def read(ref, *rest):
                return _read(_rows(ref, begin, chunk, *rest), taken)

            for i in range(slices):
                h_ref[offset + n, pl.ds(i * 64, 64), :] = state[i].astype(h_ref.dtype)
            predicted = sum(_dot(read(w_ref, pl.ds(i * 64, 64)), state[i].astype(w_ref.dtype), exact=exact)
                            for i in range(slices))
            corrected = read(u_ref).astype(jnp.float32) - predicted
            _write(_rows(corrected_ref, begin, chunk), taken, corrected)
            final = begin + taken - 1
            if per_key:
                return [state[i] * jnp.exp2(g_ref[final, pl.ds(i * 64, 64)].astype(jnp.float32))[:, None]
                        + _split_dot(read(k_ref, pl.ds(i * 64, 64)).T, corrected, exact=exact)
                        for i in range(slices)]
            last = g_ref[final].astype(jnp.float32)
            gated = corrected * jnp.exp2(last - _gates(g_ref, begin, taken, chunk))[:, None]
            return [state[i] * jnp.exp2(last)
                    + _split_dot(read(k_ref, pl.ds(i * 64, 64)).T, gated, exact=exact) for i in range(slices)]

        state = jax.lax.fori_loop(0, -(-count // chunk), step, state)

        @pl.when(last_in_row)
        def _():
            for i in range(slices):
                ht_ref[row, pl.ds(i * 64, 64), :] = state[i].astype(ht_ref.dtype)


def chunk_fwd_h(k, w, u, g, h0, *, chunk: int = 64, bv: int | None = None, spans: Sequences | None = None,
                warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`(h, corrected, final)`: the memory each chunk enters `[B, T // chunk, HV, K, V]` (`[1, NT, ...]`
    over `spans`' chunks) in `k`'s dtype, the corrected values `[B, T, HV, V]` in `u`'s and each row's
    final memory `[B, HV, K, V]` fp32: fla's `chunk_gated_delta_rule_fwd_h` (chunk_delta_h.py:672-737)
    with an initial state. `k` `[B, T, H, K]`, `w` `[B, T, HV, K]`, `u` `[B, T, HV, V]`, `g` the base-2
    cumulated decay `[B, T, HV]`, or per key dimension `[B, T, HV, K]` (fla's `gk`, Kimi's delta
    attention), `h0` `[B, HV, K, V]` fp32, each row's. `K` a multiple of 64 (fla's blockdim64 kernel
    takes up to 256). On an A100 fla's autotuner chose BV 32, 4 warps, 3 stages (c53)."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("chunk_fwd_h", _rule_shape(k, u, chunk), k.dtype,
                           bv=bv, warps=warps, stages=stages)
    bv, warps, stages = config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = k.shape
    value_heads, columns = u.shape[2:]
    group = value_heads // key_heads
    tokens = batch * length
    spans = sequences(batch, length, chunk=chunk) if spans is None else spans
    aligned = _aligned(spans, batch, length, chunk)
    chunks = spans.chunks.shape[0]

    def head(i_v, sh):
        return sh % value_heads

    per_key = g.ndim == 4
    rows = pl.BlockSpec((batch, None, width, bv), lambda i_v, sh: (0, head(i_v, sh), 0, i_v))
    h, corrected, final = pl.pallas_call(
        functools.partial(_h_kernel, chunk=chunk, aligned=aligned, length=length, exact=interpret,
                          **({"per_key": True} if per_key else {})),
        grid=(columns // bv, spans.spans.shape[0] * value_heads),
        in_specs=[pl.BlockSpec((None, 3), lambda i_v, sh: (sh // value_heads, 0)),
                  pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh) // group, 0)),
                  pl.BlockSpec((tokens, None, bv), lambda i_v, sh: (0, head(i_v, sh), i_v)),
                  pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh), 0)),
                  pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh), 0)) if per_key
                  else pl.BlockSpec((tokens, None), lambda i_v, sh: (0, head(i_v, sh))),
                  rows],
        out_specs=[pl.BlockSpec((chunks, None, width, bv), lambda i_v, sh: (0, head(i_v, sh), 0, i_v)),
                   pl.BlockSpec((tokens, None, bv), lambda i_v, sh: (0, head(i_v, sh), i_v)),
                   rows],
        out_shape=[jax.ShapeDtypeStruct((chunks, value_heads, width, columns), k.dtype),
                   jax.ShapeDtypeStruct((tokens, value_heads, columns), u.dtype),
                   jax.ShapeDtypeStruct((batch, value_heads, width, columns), jnp.float32)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_fwd_h",
    )(spans.spans, *_flat(k, u, w, g), h0)
    return _per_row(h, batch, length, chunk, spans), corrected.reshape(u.shape), final


def _per_row(h, batch: int, length: int, chunk: int, spans: Sequences):
    """Per-chunk memories `[NT, HV, K, V]` as `[B, T // chunk, ...]` where the chunks are the rows'
    own, else `[1, NT, ...]`."""
    rows = h.shape[0] == batch * -(-length // chunk) and spans.spans.shape[0] == batch
    return h.reshape(batch, -1, *h.shape[1:]) if rows else h[None]


def _dhu_kernel(span_ref, q_ref, k_ref, w_ref, g_ref, dht_ref, do_ref, dv_ref, dh_ref, dh0_ref, dv2_ref, *,
                chunk: int, aligned: bool, length: int, scale: float, exact: bool, per_key: bool = False):
    """fla's `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` (chunk_delta_h.py:359-669) for one
    sequence's head and `BV` value columns, chunks in reverse. Each stores the gradient of the memory
    it leaves into, `dh`, then `dv2 = dv + (k dh) exp2(g_last - g)` (dh cast to k's dtype), then
    `dh = dh exp2(g_last) + scale (q exp2(g))^T do - w^T dv2`: the gated queries fp32 as two parts
    of `do`'s dtype (`_split_dot`), where fla takes one TF32 product, and dv2 cast to w's dtype for
    its product, as fla casts it. `per_key` is fla's USE_GK branch: `g_ref` `[T, K]` decays the
    gradient's rows by `exp2(g_last)`, and neither `dv2` nor the queries, which carry their own decay,
    are gated; the query product stays in q's dtype. A row's last sequence starts from the final
    memory's gradient and its first stores the initial memory's."""
    start, count, offset, row, first, last_in_row = _span(span_ref, length, chunk, aligned)
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
                return _read(_rows(ref, begin, chunk, *rest), taken)

            for i in range(slices):
                dh_ref[offset + n, pl.ds(i * 64, 64), :] = grad[i].astype(dh_ref.dtype)
            out_grad = read(do_ref)
            dv = sum(_dot(read(k_ref, pl.ds(i * 64, 64)), grad[i].astype(k_ref.dtype), exact=exact)
                     for i in range(slices))
            final = begin + taken - 1
            if per_key:
                dv = dv + read(dv_ref).astype(jnp.float32)
                kept = [jnp.exp2(g_ref[final, pl.ds(i * 64, 64)].astype(jnp.float32))[:, None]
                        for i in range(slices)]

                def queried(query):
                    return _dot(query.T, out_grad.astype(query.dtype), exact=exact)
            else:
                gate = _gates(g_ref, begin, taken, chunk)
                last = g_ref[final].astype(jnp.float32)
                dv = dv * jnp.exp2(last - gate)[:, None] + read(dv_ref).astype(jnp.float32)
                decay = jnp.exp2(gate)[:, None]
                kept = [jnp.exp2(last)] * slices

                def queried(query):
                    return _split_dot((query.astype(jnp.float32) * decay).T, out_grad, exact=exact)
            _write(_rows(dv2_ref, begin, chunk), taken, dv)

            def moved(i):
                written = read(w_ref, pl.ds(i * 64, 64))
                return (queried(read(q_ref, pl.ds(i * 64, 64))) * scale
                        - _dot(written.T, dv.astype(written.dtype), exact=exact))

            return [grad[i] * kept[i] + moved(i) for i in range(slices)]

        grad = jax.lax.fori_loop(0, chunks, step, grad)

        @pl.when(first)
        def _():
            for i in range(slices):
                dh0_ref[row, pl.ds(i * 64, 64), :] = grad[i].astype(dh0_ref.dtype)


def chunk_bwd_dhu(q, k, w, g, dht, do, dv, *, scale: float, chunk: int = 64, bv: int | None = None,
                  spans: Sequences | None = None, warps: int | None = None, stages: int | None = None,
                  interpret: bool = False):
    """`(dh, dh0, dv2)`: the gradient of the memory each chunk leaves into `[B, T // chunk, HV, K, V]`
    (`[1, NT, ...]` over `spans`' chunks) in `q`'s dtype, of each row's initial memory `[B, HV, K, V]`
    fp32, and the corrected values' full gradient `[B, T, HV, V]` in `dv`'s: fla's
    `chunk_gated_delta_rule_bwd_dhu` (chunk_delta_h.py:740-794). `dht` `[B, HV, K, V]` fp32 is each
    row's final memory's gradient and `dv` the corrected values' from the output alone
    (`chunk_bwd_dv_local`); `g` `[B, T, HV]`, or `[B, T, HV, K]` per key dimension (fla's `gk`). fla's
    A100 choice: BV 32, 4 warps, 3 stages (c53)."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("chunk_bwd_dhu", _rule_shape(k, do, chunk), k.dtype,
                           bv=bv, warps=warps, stages=stages)
    bv, warps, stages = config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = q.shape
    value_heads, columns = do.shape[2:]
    group = value_heads // key_heads
    tokens = batch * length
    spans = sequences(batch, length, chunk=chunk) if spans is None else spans
    aligned = _aligned(spans, batch, length, chunk)
    chunks = spans.chunks.shape[0]

    def head(i_v, sh):
        return sh % value_heads

    keys = pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh) // group, 0))
    values = pl.BlockSpec((tokens, None, bv), lambda i_v, sh: (0, head(i_v, sh), i_v))
    rows = pl.BlockSpec((batch, None, width, bv), lambda i_v, sh: (0, head(i_v, sh), 0, i_v))
    per_key = g.ndim == 4
    dh, dh0, dv2 = pl.pallas_call(
        functools.partial(_dhu_kernel, chunk=chunk, aligned=aligned, length=length, scale=scale,
                          exact=interpret,
                          **({"per_key": True} if per_key else {})),
        grid=(columns // bv, spans.spans.shape[0] * value_heads),
        in_specs=[pl.BlockSpec((None, 3), lambda i_v, sh: (sh // value_heads, 0)), keys, keys,
                  pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh), 0)),
                  pl.BlockSpec((tokens, None, width), lambda i_v, sh: (0, head(i_v, sh), 0)) if per_key
                  else pl.BlockSpec((tokens, None), lambda i_v, sh: (0, head(i_v, sh))),
                  rows, values, values],
        out_specs=[pl.BlockSpec((chunks, None, width, bv), lambda i_v, sh: (0, head(i_v, sh), 0, i_v)), rows,
                   values],
        out_shape=[jax.ShapeDtypeStruct((chunks, value_heads, width, columns), q.dtype),
                   jax.ShapeDtypeStruct((batch, value_heads, width, columns), jnp.float32),
                   jax.ShapeDtypeStruct((tokens, value_heads, columns), dv.dtype)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_bwd_dhu",
    )(spans.spans, *_flat(q, k, w, g), dht, *_flat(do, dv))
    return _per_row(dh, batch, length, chunk, spans), dh0, dv2.reshape(dv.shape)


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
            return _read(_rows(ref, start, chunk, *rest), taken)

        scores = jnp.zeros((chunk, chunk), jnp.float32)
        for block in range(k_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            scores += _dot(read(k_ref, columns), read(q_ref, columns).T, exact=exact) * scale
        gate = _gates(g_ref, start, taken, chunk)
        scores = scores * jnp.exp2(gate[None, :] - gate[:, None])
        index = jnp.arange(chunk)
        scores = jnp.where(index[:, None] <= index[None, :], scores, 0.0).astype(do_ref.dtype)
        for block in range(do_ref.shape[-1] // bv):
            columns = pl.ds(block * bv, bv)
            product = _dot(scores, read(do_ref, columns), exact=exact)
            _write(_rows(dv_ref, start, chunk, columns), taken, product)


def chunk_bwd_dv_local(q, k, g, do, *, scale: float, chunk: int = 64, bk: int | None = None,
                       bv: int | None = None, spans: Sequences | None = None, warps: int | None = None,
                       stages: int | None = None, interpret: bool = False):
    """The corrected values' gradient from each chunk's own output, `[B, T, HV, V]` in `do`'s dtype:
    fla's `chunk_bwd_dv_local` (chunk_o.py:640-676). fla tiles `BK` and `BV` at 64 on an A100
    (its CONST_TILING for ampere and ada shared memory) and chose 2 warps, 3 stages there (c53)."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("chunk_bwd_dv_local", _rule_shape(k, do, chunk), k.dtype,
                           bk=bk, bv=bv, warps=warps, stages=stages)
    bk, bv, warps, stages = config["bk"], config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = q.shape
    value_heads, columns = do.shape[2:]
    group = value_heads // key_heads
    tokens = batch * length
    spans = sequences(batch, length, chunk=chunk) if spans is None else spans
    aligned = _aligned(spans, batch, length, chunk)
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
    return dv.reshape(do.shape)


def _split_dot(a, b, *, exact: bool):
    """`a @ b` with an fp32 operand beside a 16-bit one: the fp32 one as two parts of the other's
    dtype, its rounding and the rounding's remainder, each product exact and summed in fp32. fla
    rounds that operand to the narrow dtype for one product, or takes one TF32 product of both in
    fp32; the parts keep 16 bits of it at the cost of a second narrow product, which runs as fast as
    one TF32 product. Operands of one dtype keep the one product."""
    narrow = min(a.dtype, b.dtype, key=lambda dtype: jnp.dtype(dtype).itemsize)

    def parts(x):
        if x.dtype == narrow:
            return [x]
        high = x.astype(narrow)
        return [high, (x - high.astype(jnp.float32)).astype(narrow)]

    products = [_dot(left, right, exact=exact) for left in parts(a) for right in parts(b)]
    return functools.reduce(operator.add, products)


def _state_inner_product(memory, memory_grad):
    """fla chunk_o.py:251: multiply h/dh in their storage dtype, then tl.sum widens to fp32.
    In particular, bf16 products round before reduction; widening both inputs omits that round."""
    return jnp.sum((memory * memory_grad).astype(jnp.float32))


def _dqkwg_kernel(chunk_ref, q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, dh_ref, dv_ref, dq_ref, dk_ref,
                  dw_ref, dg_ref, *, chunk: int, aligned: bool, scale: float, bv: int, exact: bool):
    """fla's `chunk_bwd_kernel_dqkwg` (fla/ops/common/chunk_o.py:159-331) for one chunk of one key
    head and `BK` key columns, over the value heads it serves in turn: over the value columns `bv` at
    a time, `ds = do v^T`, `dq = do h^T`, `dk = v dh^T`, `dw = -dv h^T` and `sum(h dh)`; then the
    gates, `ds`'s causal mask, `dq += ds k`, `dk += ds^T q` and the decay's gradient, its last row
    taking the memory's. The value heads' `dq` and `dk` sum in fp32 and round to their dtype once,
    where fla rounds each value head's and sums them after. The heads are a loop, not unrolled, so a
    program holds one head's accumulators at a time."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return _read(_rows(ref, start, chunk, *rest), taken)

        width = q_ref.shape[-1]
        queries, keys = read(q_ref), read(k_ref)
        index = jnp.arange(chunk)
        causal = index[:, None] >= index[None, :]

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
                memory, memory_grad = h_ref[head, :, columns], dh_ref[head, :, columns]
                dg_last += _state_inner_product(memory, memory_grad)
                ds += _dot(out_grad, value.T, exact=exact)
                dq += _dot(out_grad, memory.T.astype(out_grad.dtype), exact=exact)
                dk += _dot(value, memory_grad.T.astype(value.dtype), exact=exact)
                dw += _dot(local.astype(value.dtype), memory.T.astype(value.dtype), exact=exact)
            _write(_rows(dw_ref, start, chunk, head), taken, -dw)
            gate = _gates(g_ref, start, taken, chunk, 0, head)
            last = g_ref[start + taken - 1, head].astype(jnp.float32)
            dg_last = dg_last * jnp.exp2(last)
            dq = dq * jnp.exp2(gate)[:, None] * scale
            dk = dk * jnp.exp2(last - gate)[:, None]
            dg_last += jnp.sum(dk * keys.astype(jnp.float32))
            ds = jnp.where(causal, ds * jnp.exp2(gate[:, None] - gate[None, :]), 0.0) * scale
            dq += _split_dot(ds, keys, exact=exact)
            dk += _split_dot(ds.T, queries, exact=exact)
            dg = (jnp.sum(dq * queries.astype(jnp.float32), axis=1)
                  - jnp.sum(dk * keys.astype(jnp.float32), axis=1))
            _write(_rows(dg_ref, start, chunk, head), taken, jnp.where(index < taken - 1, dg, dg + dg_last))
            return sums[0] + dq, sums[1] + dk

        zeros = jnp.zeros((chunk, width), jnp.float32)
        dq, dk = jax.lax.fori_loop(0, v_ref.shape[1], head_grads, (zeros, zeros))
        _write(_rows(dq_ref, start, chunk), taken, dq)
        _write(_rows(dk_ref, start, chunk), taken, dk)


def chunk_bwd_dqkwg(q, k, v, w, g, h, dv, do, dh, *, scale: float, chunk: int = 64,
                    bk: int | None = None, bv: int | None = None, spans: Sequences | None = None,
                    warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`(dq, dk, dw, dg)`: fla's `chunk_bwd_dqkwg` (chunk_o.py:696-743). `q`, `k` `[B, T, H, K]`, `v`
    the corrected values and `do` `[B, T, HV, V]`, `w` `[B, T, HV, K]` (its shape and dtype for `dw`),
    `g` the base-2 cumulated decay, `h` and `dh` per chunk `[B, T // chunk, HV, K, V]`, `dv` the
    corrected values' full gradient. One program per key head sums its value heads' `dq` and `dk` in
    fp32 and stores them once, `[B, T, H, K]` in `q`'s and `k`'s dtypes; `dw` per value head, and `dg`
    per key block, summed, as fla does. fla tiles `BK`, `BV` at 64 on an A100 and chose 4 warps, 2
    stages (c53)."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("chunk_bwd_dqkwg", _rule_shape(k, v, chunk), k.dtype,
                           bk=bk, bv=bv, warps=warps, stages=stages)
    bk, bv, warps, stages = config["bk"], config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = q.shape
    value_heads, columns = v.shape[2:]
    group = value_heads // key_heads
    tokens = batch * length
    spans = sequences(batch, length, chunk=chunk) if spans is None else spans
    aligned = _aligned(spans, batch, length, chunk)
    keys = pl.BlockSpec((tokens, None, bk), lambda i_k, i_t, i_h: (0, i_h, i_k))
    values = pl.BlockSpec((tokens, group, columns), lambda i_k, i_t, i_h: (0, i_h, 0))
    memory = pl.BlockSpec((None, group, bk, columns), lambda i_k, i_t, i_h: (i_t, i_h, i_k, 0))
    dq, dk, dw, dg = pl.pallas_call(
        functools.partial(_dqkwg_kernel, chunk=chunk, aligned=aligned, scale=scale, bv=bv, exact=interpret),
        grid=(width // bk, spans.chunks.shape[0], key_heads),
        in_specs=[pl.BlockSpec((None, 2), lambda i_k, i_t, i_h: (i_t, 0)), keys, keys, values,
                  pl.BlockSpec((tokens, group), lambda i_k, i_t, i_h: (0, i_h)), memory, values, memory,
                  values],
        out_specs=[keys, keys, pl.BlockSpec((tokens, group, bk), lambda i_k, i_t, i_h: (0, i_h, i_k)),
                   pl.BlockSpec((None, tokens, group), lambda i_k, i_t, i_h: (i_k, 0, i_h))],
        out_shape=[jax.ShapeDtypeStruct((tokens, key_heads, width), q.dtype),
                   jax.ShapeDtypeStruct((tokens, key_heads, width), k.dtype),
                   jax.ShapeDtypeStruct((tokens, value_heads, width), w.dtype),
                   jax.ShapeDtypeStruct((width // bk, tokens, value_heads), jnp.float32)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_bwd_dqkwg",
    )(spans.chunks, *_flat(q, k, v, g), _chunk_states(h), *_flat(do), _chunk_states(dh), *_flat(dv))
    return dq.reshape(q.shape), dk.reshape(k.shape), dw.reshape(w.shape), dg.sum(0).reshape(g.shape)


def _chunk_states(h):
    """Per-chunk memories, `[B, T // chunk, ...]` or `[1, NT, ...]`, as `[NT, ...]` in chunk order."""
    return h.reshape(-1, *h.shape[2:])


def _wy_gate_sum(d_gate, weighted):
    """fla wy_fast.py:246: subtract the two fp32 reductions before adding the gate accumulator."""
    return d_gate + (jnp.sum(weighted, axis=1) - jnp.sum(weighted, axis=0))


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
    (chunk.py:244)."""
    earlier_ref, (dk_ref, dv_ref, db_ref, dg_ref) = (refs[0], refs[1:]) if group else (None, refs)
    start, taken = _chunk_of(chunk_ref, chunk, aligned)
    first_of_group = pl.program_id(1) % (group or 1) == 0

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return _read(_rows(ref, start, chunk, *rest), taken)

        inverse_t = read(a_ref).T
        beta = read(beta_ref)
        gate = _gates(g_ref, start, taken, chunk)
        decay = jnp.exp2(gate)
        width = k_ref.shape[-1]
        d_x = jnp.zeros((chunk, chunk), jnp.float32)
        d_beta = jnp.zeros((chunk,), jnp.float32)
        d_gate = jnp.zeros((chunk,), jnp.float32)
        for block in range(width // bk):
            columns = pl.ds(block * bk, bk)
            key = read(k_ref, columns)
            scaled = key * (beta * decay)[:, None]
            w_grad = read(dw_ref, columns)
            d_x += _dot(w_grad, scaled.T.astype(w_grad.dtype), exact=exact)
            d_scaled = _dot(inverse_t, w_grad, exact=exact)
            d_beta += jnp.sum(d_scaled * key * decay[:, None], axis=1)
            d_gate += jnp.sum(d_scaled * scaled, axis=1)
        for block in range(v_ref.shape[-1] // bv):
            columns = pl.ds(block * bv, bv)
            value = read(v_ref, columns)
            scaled = (value * beta[:, None]).astype(value.dtype)
            u_grad = read(du_ref, columns)
            d_x += _dot(u_grad, scaled.T, exact=exact)
            d_scaled = _dot(inverse_t, u_grad, exact=exact)
            _write(_rows(dv_ref, start, chunk, columns), taken, d_scaled * beta[:, None])
            d_beta += jnp.sum(d_scaled * value, axis=1)
        index = jnp.arange(chunk)
        strict = index[:, None] > index[None, :]
        d_x = jnp.where(strict, d_x, 0.0)
        d_x = _dot(d_x.astype(inverse_t.dtype), inverse_t, exact=exact)
        d_x = _dot(inverse_t, d_x.astype(inverse_t.dtype), exact=exact)
        d_x = d_x * jnp.exp2(gate[:, None] - gate[None, :])
        d_l = jnp.where(strict, -d_x, 0.0).astype(k_ref.dtype)
        products = jnp.zeros((chunk, chunk), jnp.float32)
        for block in range(width // bk):
            columns = pl.ds(block * bk, bk)
            key = read(k_ref, columns)
            scaled = key * beta[:, None]
            products += _dot(key, key.T, exact=exact)
            d_key = _dot(d_l, key, exact=exact)
            d_beta += jnp.sum(d_key * key, axis=1)
            through_w = _dot(inverse_t, read(dw_ref, columns), exact=exact) * (decay * beta)[:, None]
            full = through_w + (d_key * beta[:, None] + _dot(scaled.T.astype(d_l.dtype), d_l, exact=exact).T)
            if earlier_ref is not None:
                earlier = read(earlier_ref, columns).astype(jnp.float32)
                full += jnp.where(first_of_group, earlier, 0.0)
            _write(_rows(dk_ref, start, chunk, columns), taken, full)
        _write(_rows(db_ref, start, chunk), taken, d_beta)
        weighted = d_l.astype(jnp.float32) * (products * beta[:, None])
        _write(_rows(dg_ref, start, chunk), taken, _wy_gate_sum(d_gate, weighted))


def prepare_wy_repr_bwd(k, v, beta, g, a, dw, du, dk=None, *, chunk: int = 64, bk: int | None = None,
                        bv: int | None = None, spans: Sequences | None = None, warps: int | None = None,
                        stages: int | None = None, interpret: bool = False):
    """`(dk, dv, db, dg)` through the WY representation: fla's `prepare_wy_repr_bwd` (wy_fast.py:
    296-343). `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `beta` and `g` `[B, T, HV]`, `a` the inverse
    per chunk `[B, T, HV, BT]`, `dw` and `du` the gradients of `w` and `u`, and `dk` `[B, T, H, K]`
    the keys' gradient so far, which the returned one includes, added in fp32 before its one
    rounding; without it, fla's call. `dk` is summed over each key head's group. fla tiles `BK`,
    `BV` at 64 on an A100 and chose 4 warps, 2 stages (c53)."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("prepare_wy_repr_bwd", _rule_shape(k, v, chunk), k.dtype,
                           bk=bk, bv=bv, warps=warps, stages=stages)
    bk, bv, warps, stages = config["bk"], config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = k.shape
    value_heads, columns = v.shape[2:]
    group = value_heads // key_heads
    tokens = batch * length
    spans = sequences(batch, length, chunk=chunk) if spans is None else spans
    aligned = _aligned(spans, batch, length, chunk)

    def headed(size):
        return pl.BlockSpec((tokens, None, size), lambda i_t, h: (0, h, 0))

    gate = pl.BlockSpec((tokens, None), lambda i_t, h: (0, h))
    keys = pl.BlockSpec((tokens, None, width), lambda i_t, h: (0, h // group, 0))
    earlier = () if dk is None else (dk,)
    dk, dv, db, dg = pl.pallas_call(
        functools.partial(_wy_bwd_kernel, chunk=chunk, aligned=aligned, bk=min(bk, width),
                          bv=min(bv, columns), group=group if earlier else None, exact=interpret),
        grid=(spans.chunks.shape[0], value_heads),
        in_specs=[_CHUNK, keys, headed(columns), gate, gate, headed(chunk), headed(width), headed(columns),
                  *(keys for _ in earlier)],
        out_specs=[headed(width), headed(columns), gate, gate],
        out_shape=[jax.ShapeDtypeStruct((tokens, value_heads, width), k.dtype),
                   jax.ShapeDtypeStruct((tokens, value_heads, columns), v.dtype),
                   jax.ShapeDtypeStruct((tokens, value_heads), beta.dtype),
                   jax.ShapeDtypeStruct((tokens, value_heads), g.dtype)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_prepare_wy_repr_bwd",
    )(spans.chunks, *_flat(k, v, beta, g, a, dw, du, *earlier))
    dk = dk.reshape(batch, length, key_heads, group, width).sum(3).astype(k.dtype) if group > 1 else \
        dk.reshape(k.shape)
    return dk, dv.reshape(v.shape), db.reshape(beta.shape), dg.reshape(g.shape)


def _o_kernel(chunk_ref, q_ref, k_ref, v_ref, h_ref, g_ref, o_ref, *, chunk: int, aligned: bool,
              scale: float, bk: int,
              exact: bool):
    """fla's `chunk_fwd_kernel_o` (fla/ops/common/chunk_o.py:43-147) for one chunk of one head and
    `BV` value columns: `o = scale * ((q h) exp2(g) + tril((q k^T) exp2(g_s - g_t)) v)`, the key
    columns walked `bk` at a time into fp32 accumulators, `A` cast to the values' dtype for its
    product, as fla does."""
    start, taken = _chunk_of(chunk_ref, chunk, aligned)

    @pl.when(taken > 0)
    def _():
        def read(ref, *rest):
            return _read(_rows(ref, start, chunk, *rest), taken)

        out = jnp.zeros((chunk, v_ref.shape[-1]), jnp.float32)
        scores = jnp.zeros((chunk, chunk), jnp.float32)
        for block in range(q_ref.shape[-1] // bk):
            columns = pl.ds(block * bk, bk)
            query = read(q_ref, columns)
            out += _dot(query, h_ref[columns, :], exact=exact)
            scores += _dot(query, read(k_ref, columns).T, exact=exact)
        gate = _gates(g_ref, start, taken, chunk)
        out = out * jnp.exp2(gate)[:, None]
        scores = scores * jnp.exp2(gate[:, None] - gate[None, :])
        index = jnp.arange(chunk)
        scores = jnp.where(index[:, None] >= index[None, :], scores, 0.0)
        value = read(v_ref)
        out = out * scale + _dot(scores.astype(value.dtype), value, exact=exact) * scale
        _write(_rows(o_ref, start, chunk), taken, out)


def chunk_fwd_o(q, k, v, h, g, *, scale: float, chunk: int = 64, bk: int | None = None,
                bv: int | None = None, spans: Sequences | None = None, warps: int | None = None,
                stages: int | None = None, interpret: bool = False):
    """Each chunk's output, `[B, T, HV, V]` in `v`'s dtype: fla's `chunk_fwd_o` (chunk_o.py:529-575).

    `q`, `k` `[B, T, H, K]`, `v` the corrected values `[B, T, HV, V]`, `h` the memory each chunk
    entered `[B, T // chunk, HV, K, V]`, `g` the base-2 log decay cumulated within each chunk
    `[B, T, HV]` (fp32). Grouped queries: value head `i` reads key head `i // (HV // H)`. `T` is a
    whole number of chunks; `bk`, `bv`, `warps` and `stages` are one of fla's configs
    (`BK, BV` 128/128 at 8 warps, 64/64 at 4, 32/32 at 2, all 3 stages)."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("chunk_fwd_o", _rule_shape(k, v, chunk), k.dtype,
                           bk=bk, bv=bv, warps=warps, stages=stages)
    bk, bv, warps, stages = config["bk"], config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = q.shape
    value_heads, columns = v.shape[2], v.shape[3]
    group = value_heads // key_heads
    tokens = batch * length
    spans = sequences(batch, length, chunk=chunk) if spans is None else spans
    aligned = _aligned(spans, batch, length, chunk)
    keyed = pl.BlockSpec((tokens, None, width), lambda i_v, i_t, i_h: (0, i_h // group, 0))
    valued = pl.BlockSpec((tokens, None, bv), lambda i_v, i_t, i_h: (0, i_h, i_v))
    out = pl.pallas_call(
        functools.partial(_o_kernel, chunk=chunk, aligned=aligned, scale=scale, bk=min(bk, width),
                          exact=interpret),
        grid=(columns // bv, spans.chunks.shape[0], value_heads),
        in_specs=[pl.BlockSpec((None, 2), lambda i_v, i_t, i_h: (i_t, 0)), keyed, keyed, valued,
                  pl.BlockSpec((None, None, width, bv), lambda i_v, i_t, i_h: (i_t, i_h, 0, i_v)),
                  pl.BlockSpec((tokens, None), lambda i_v, i_t, i_h: (0, i_h))],
        out_specs=valued,
        out_shape=jax.ShapeDtypeStruct((tokens, value_heads, columns), v.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_fwd_o",
    )(spans.chunks, *_flat(q, k, v), _chunk_states(h), *_flat(g))
    return out.reshape(v.shape)


def _rule(q, k, v, g, beta, h0, cu_seqlens, scale: float, chunk: int, interpret: bool):
    return _rule_fwd(q, k, v, g, beta, h0, cu_seqlens, scale, chunk, interpret)[0]


def _rule_fwd(q, k, v, g, beta, h0, cu_seqlens, scale: float, chunk: int, interpret: bool):
    """fla's `chunk_gated_delta_rule_fwd` (fla/ops/gated_delta_rule/chunk.py:33-123): the decay
    cumulated in base 2, the inverse, `w` and `u`, the memory across chunks, then the output. Saves
    what fla saves: the cumulated decay and the inverse, named 'delta_decay' and 'delta_inverse' for a
    remat policy, and the initial memory."""
    spans = sequences(q.shape[0], q.shape[1], cu_seqlens, chunk=chunk)
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    gc = checkpoint_name(chunk_local_cumsum(g, scale=RCP_LN2, **kernels), 'delta_decay')
    a = checkpoint_name(kkt_solve(k, gc, beta, **kernels), 'delta_inverse')
    w, u = recompute_w_u(k, v, beta, a, gc, **kernels)
    h, corrected, final = chunk_fwd_h(k, w, u, gc, h0, **kernels)
    o = chunk_fwd_o(q, k, corrected, h, gc, scale=scale, **kernels)
    return (o, final), (q, k, v, gc, beta, a, h0, spans)


def _rule_bwd(scale: float, chunk: int, interpret: bool, residual, cotangent):
    """fla's `chunk_gated_delta_rule_bwd` (chunk.py:126-250): `w`, `u` and the memory recomputed, then
    the output's local value gradient, the reverse recurrence, the query, key, `w` and decay
    gradients, the WY backward, and the decay's gradient summed back over each chunk."""
    q, k, v, gc, beta, a, h0, spans = residual
    do, dht = cotangent
    kernels = {"chunk": chunk, "spans": spans, "interpret": interpret}
    w, u = recompute_w_u(k, v, beta, a, gc, **kernels)
    h, corrected, _ = chunk_fwd_h(k, w, u, gc, h0, **kernels)
    dv = chunk_bwd_dv_local(q, k, gc, do, scale=scale, **kernels)
    dh, dh0, dv = chunk_bwd_dhu(q, k, w, gc, dht.astype(jnp.float32), do, dv, scale=scale, **kernels)
    dq, dk, dw, dg = chunk_bwd_dqkwg(q, k, corrected, w, gc, h, dv, do, dh, scale=scale, **kernels)
    dk, dv, db, dg2 = prepare_wy_repr_bwd(k, v, beta, gc, a, dw, dv, dk, **kernels)
    dg = chunk_local_cumsum(dg + dg2, reverse=True, **kernels)
    return dq, dk, dv, dg.astype(jnp.float32), db, dh0, None


chunk_gated_delta_rule = jax.custom_vjp(_rule, nondiff_argnums=(7, 8, 9))
"""fla's chunked gated delta rule on the ported kernels: `(o, final)` for `q`, `k` `[B, T, H, K]`
(normalized as the caller wants them), `v` `[B, T, HV, V]`, the per-token natural log decay `g`
`[B, T, HV]` fp32, `beta` `[B, T, HV]` and each row's initial memory `h0` `[B, HV, K, V]` fp32, the
final one each row's. `cu_seqlens` (`[N + 1]` int32 over the `B * T` tokens, fla's varlen) splits the
rows into sequences, the memory starting from zero at each one a row does not begin with; None runs
each row as one. Its numbers are fla's: bf16 operands store `A`, `w`, `u`, `h` and the corrected
values in bf16, with fp32 accumulation and fp32 decay, the memory fp32 across chunks."""
chunk_gated_delta_rule.defvjp(_rule_fwd, _rule_bwd)
