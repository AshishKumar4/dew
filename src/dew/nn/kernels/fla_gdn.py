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
devices. Each config compiles within its GPU's block at the training shapes (CONFIG_SHAPES,
tests/test_kernel_configs.py), and the port takes keys and values no wider than those. Every public
kernel accepts explicit block/warp/stage overrides for the lab.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
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
    "chunk_bwd_dqkwg": {"bk": 64, "bv": 64, "warps": 4, "stages": 2},
    "prepare_wy_repr_bwd": {"bk": 64, "bv": 64, "warps": 4, "stages": 2},
}
_SM80_FP32 = {**_SM80_BF16,
              "chunk_fwd_h": {"bv": 64, "warps": 4, "stages": 2},
              # One stage: three took 254,984 bytes of an A100's 166,912 (c58).
              "chunk_bwd_dhu": {"bv": 64, "warps": 4, "stages": 1},
              # Eight warps: four took ptxas 40 s to compile at TF32x3 on sm80, eight 17 s.
              "prepare_wy_repr_bwd": {"bk": 64, "bv": 64, "warps": 8, "stages": 2}}
# fla's stage and block lists for devices under an A100's shared memory (chunk_delta_h.py:43-44,
# 352-353), with two warps for the forward h.
_SMALL_BF16 = {**_SM80_BF16,
               "chunk_fwd_h": {"bv": 32, "warps": 2, "stages": 2},
               "chunk_bwd_dhu": {"bv": 64, "warps": 4, "stages": 1}}
_SMALL_FP32 = {**_SMALL_BF16,
               "chunk_fwd_h": {"bv": 64, "warps": 2, "stages": 1},
               "chunk_bwd_dhu": {"bv": 32, "warps": 4, "stages": 1}}
# Blackwell's two-warp, four-stage WY backward, the one config fla validated there (wy_fast.py:18-22,
# #913) until Triton 3.8 fixes the race the others hit.
_BLACKWELL = {"prepare_wy_repr_bwd": {"bk": 64, "bv": 64, "warps": 2, "stages": 4}}


def _table(name: str, target: generation.GpuTarget, dtype: str) -> dict:
    wide = (target.shared_limit or 0) >= (generation.TARGETS["sm80"].shared_limit or 0)
    table = {("bfloat16", True): _SM80_BF16, ("float32", True): _SM80_FP32,
             ("bfloat16", False): _SMALL_BF16, ("float32", False): _SMALL_FP32}[dtype, wide]
    # fla's IS_NVIDIA_BLACKWELL: compute capability 10.x or 12.x.
    return {**table, **_BLACKWELL} if int(name[2:]) >= 100 else table


CONFIGS = {(name, dtype): _table(name, target, dtype)
           for name, target in generation.TARGETS.items()
           if name.startswith("sm") and int(name[2:]) >= generation.BF16_GPU
           for dtype in ("bfloat16", "float32")}
"""Each kernel's blocks, warps and stages by (generation, activation dtype), for every GPU that runs
Triton: an A100's wherever a block holds as much shared memory, the 99 KB tables elsewhere."""

CONFIG_SHAPES = {
    "0.8b": {"batch": 8, "length": 3072, "key_heads": 16, "value_heads": 16,
             "width": 128, "columns": 128, "chunk": 64},
    "9b": {"batch": 4, "length": 3072, "key_heads": 16, "value_heads": 32,
           "width": 128, "columns": 128, "chunk": 64},
}
"""The training shapes every config compiles at, the widest keys and values the port takes."""


class KernelCase(NamedTuple):
    """One declared kernel call for the compile test (`generation.compile_kernels`): the kernel with
    its config, and abstract operands at a training shape."""

    name: str
    call: Callable[[], tuple[Callable, tuple[jax.ShapeDtypeStruct, ...]]]


def _case_call(kernel, config, shape, dtype, *, reverse=False):
    batch, length, key_heads, value_heads = (shape[name] for name in
                                            ("batch", "length", "key_heads", "value_heads"))
    width, columns, chunk = (shape[name] for name in ("width", "columns", "chunk"))
    narrow = jnp.dtype(dtype)
    keyed = jax.ShapeDtypeStruct((batch, length, key_heads, width), narrow)
    weighted = jax.ShapeDtypeStruct((batch, length, value_heads, width), narrow)
    valued = jax.ShapeDtypeStruct((batch, length, value_heads, columns), narrow)
    gated = jax.ShapeDtypeStruct((batch, length, value_heads), jnp.float32)
    beta = jax.ShapeDtypeStruct(gated.shape, narrow)
    inverse = jax.ShapeDtypeStruct((batch, length, value_heads, chunk), narrow)
    entered = jax.ShapeDtypeStruct((batch, length // chunk, value_heads, width, columns), narrow)
    state = jax.ShapeDtypeStruct((batch, value_heads, width, columns), jnp.float32)
    operands = {
        "chunk_local_cumsum": (gated,),
        "kkt_solve": (keyed, gated, beta),
        "recompute_w_u": (keyed, valued, beta, inverse, gated),
        "chunk_fwd_h": (keyed, weighted, valued, gated, state),
        "chunk_fwd_o": (keyed, keyed, valued, entered, gated),
        "chunk_bwd_dv_local": (keyed, keyed, gated, valued),
        "chunk_bwd_dhu": (keyed, keyed, weighted, gated, state, valued, valued),
        "chunk_bwd_dqkwg": (keyed, keyed, valued, weighted, gated, entered, valued, valued, entered),
        "prepare_wy_repr_bwd": (keyed, valued, beta, gated, inverse, weighted, valued),
    }
    kwargs = {**config, "chunk": chunk, "interpret": False}
    if kernel == "chunk_local_cumsum":
        kwargs.update(scale=RCP_LN2, reverse=reverse)
    elif kernel in ("chunk_fwd_o", "chunk_bwd_dv_local", "chunk_bwd_dhu", "chunk_bwd_dqkwg"):
        kwargs["scale"] = width ** -0.5
    return functools.partial(globals()[kernel], **kwargs), operands[kernel]


def kernel_cases(target: str) -> Iterator[KernelCase]:
    """Each kernel at `target`'s configs for both dtypes and both training shapes, and the reverse
    cumsum, for the compile test."""
    for (name, dtype), table in CONFIGS.items():
        if name != target:
            continue
        for kernel, config in table.items():
            flags = "-".join(f"{key}{value}" for key, value in sorted(config.items()))
            for shape_name, shape in CONFIG_SHAPES.items():
                for reverse in ((False, True) if kernel == "chunk_local_cumsum" else (False,)):
                    name = f"{target}/{kernel}/{dtype}/{flags}/{shape_name}" + "/reverse" * reverse
                    yield KernelCase(name, functools.partial(_case_call, kernel, config, shape, dtype,
                                                             reverse=reverse))


def _rule_shape(key, value, chunk):
    return {"width": key.shape[-1], "columns": value.shape[-1], "chunk": chunk}


def _dtype_name(dtype) -> str:
    """The table a dtype reads: fp16 takes bf16's, whose itemsize it shares."""
    name = jnp.dtype(dtype).name
    return "bfloat16" if name == "float16" else name


def kernel_config(kernel: str, shape: Mapping[str, int], dtype, *, target: str | None = None,
                  **overrides: int | None) -> dict[str, int]:
    """`kernel`'s config on `target` (the default device's generation) for `dtype`, an A100's off a
    GPU, where the kernels interpret; overrides (the lab's) replace its fields. Blocks are clipped to
    the keys' `width` and values' `columns`."""
    target = generation.device_generation() if target is None else target
    name = _dtype_name(dtype)
    config = {**CONFIGS.get((target, name), CONFIGS[("sm80", name)])[kernel],
              **{key: value for key, value in overrides.items() if value is not None}}
    return {key: min(value, shape["width"] if key == "bk" else shape["columns"]) if key in ("bk", "bv")
            else value for key, value in config.items()}


def rule_refusal(key, value, chunk):
    """Why the port does not run on this device for these keys and values: a GPU without its configs,
    keys or values wider than the configs compile for, or a key head serving a number of value heads
    no Triton block holds."""
    if key.dtype != value.dtype:
        return f"the port needs matching key/value dtypes, got {key.dtype} and {value.dtype}"
    target = generation.device_generation()
    if not target.startswith("sm"):
        return None
    if (target, _dtype_name(key.dtype)) not in CONFIGS:
        return f"the port has no configs for {target}"
    widest = max(shape["width"] for shape in CONFIG_SHAPES.values())
    if key.shape[-1] > widest or value.shape[-1] > widest:
        return (f"keys {key.shape[-1]} and values {value.shape[-1]} wide; the port's configs compile for "
                f"at most {widest}")
    group = value.shape[2] // key.shape[2]
    if group & (group - 1):
        return f"{group} value heads a key head; dqkwg's block holds a power of two"
    return None


def _cumsum_kernel(s_ref, o_ref, *, scale: float | None, reverse: bool):
    """fla's `chunk_local_cumsum_scalar_kernel` (fla/ops/utils/cumsum.py:33-73) for one chunk of one
    head: the inclusive cumulative sum in fp32, from the end where `reverse`, then `scale`."""
    step = s_ref[...].astype(jnp.float32)
    total = jnp.cumsum(step)
    if reverse:
        total = -total + jnp.sum(step) + step
    if scale is not None:
        total = total * scale
    o_ref[...] = total.astype(o_ref.dtype)


def chunk_local_cumsum(g, *, chunk: int = 64, scale: float | None = None, reverse: bool = False,
                       warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`g` `[B, T, H]` cumulated within each chunk of `chunk` tokens, fp32: fla's `chunk_local_cumsum`
    of a per-head gate (cumsum.py:262-300), `scale` RCP_LN2 for the forward's base-2 decay, `reverse`
    for its gradient. fla autotunes only `num_warps` over 1, 2, 4 and 8 here."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("chunk_local_cumsum", {"chunk": chunk, "width": 1, "columns": 1}, g.dtype,
                           warps=warps, stages=stages)
    warps, stages = config["warps"], config["stages"]

    batch, length, heads = g.shape
    spec = pl.BlockSpec((None, chunk, None), lambda i_t, bh: (bh // heads, i_t, bh % heads))
    return pl.pallas_call(
        functools.partial(_cumsum_kernel, scale=scale, reverse=reverse),
        grid=(length // chunk, batch * heads), in_specs=[spec], out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(g.shape, jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_local_cumsum",
    )(g)


def _dot(a, b, *, exact: bool, ieee: bool = False):
    """`a @ b` accumulated in fp32, as fla's `tl.dot` of bf16 operands; `ieee` for its
    `allow_tf32=False` products of fp32 operands. Interpreted on a CPU, whose XLA has no
    bf16 x bf16 -> fp32 dot, the operands widen first: a bf16 product is exact in fp32, so the value
    is the same. The shared guard rejects mismatched dtypes before that widening."""
    return generation.strict_dot(a, b, exact=exact, ieee=ieee)


def _kkt_solve_kernel(k_ref, g_ref, beta_ref, a_ref, *, sub: int, bk: int, exact: bool):
    """fla's `chunk_gated_delta_rule_fwd_kkt_solve_kernel` (fla/ops/gated_delta_rule/chunk_fwd.py:40-
    321) for one chunk of one head: `(I + A)^-1` for `A = tril(beta_s (k_s . k_t) exp2(g_s - g_t), -1)`,
    in `sub`-row blocks (fla's BC = 16 of BT = 64). Every lower block of `k k^T` accumulates over the
    key columns `bk` at a time; the gate and beta scale them; forward substitution inverts each
    diagonal block; and neighbouring inverses merge as `-X_i A_ij X_j`, the merge's products at TF32
    as fla's SOLVE_TRIL_DOT_PRECISION takes them on an sm80 or later."""
    # Triton lowers no slice of a loaded value, so every block is loaded from its ref.
    size, width = k_ref.shape
    count = size // sub
    gates = [g_ref[pl.ds(i * sub, sub)].astype(jnp.float32) for i in range(count)]
    betas = [beta_ref[pl.ds(i * sub, sub)].astype(jnp.float32) for i in range(count)]
    index = jnp.arange(sub)
    strict = index[:, None] > index[None, :]

    def keys(i, c):
        return k_ref[pl.ds(i * sub, sub), pl.ds(c * bk, bk)]

    def block(i, j):
        """`A`'s block (i, j), i >= j: the keys' products, gated and scaled by beta."""
        product = sum(_dot(keys(i, c), keys(j, c).T, exact=exact) for c in range(width // bk))
        decay = jnp.exp2(gates[i][:, None] - gates[j][None, :])
        product = product * (jnp.where(strict, decay, 0.0) if i == j else decay)
        return product * betas[i][:, None]

    blocks = {(i, j): block(i, j) for i in range(count) for j in range(i + 1)}

    def inverted(diagonal):
        """`(I + D)^-1` of a strictly lower block, row by row: fla's forward substitution."""
        inverse = -diagonal
        for i in range(2, sub):
            row = jnp.sum(jnp.where((index == i)[:, None], -diagonal, 0.0), axis=0)
            row = jnp.where(index < i, row, 0.0)
            row = row + jnp.sum(row[:, None] * inverse, axis=0)
            inverse = jnp.where((index == i)[:, None], row[None, :], inverse)
        return inverse + (index[:, None] == index[None, :])

    precision = jax.lax.Precision.DEFAULT

    def tf32(a, b):
        return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=precision)

    inverse = {(i, i): inverted(blocks[i, i]) for i in range(count)}
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
            a_ref[pl.ds(i * sub, sub), pl.ds(j * sub, sub)] = found.astype(a_ref.dtype)


def kkt_solve(k, g, beta, *, chunk: int = 64, sub: int = 16, bk: int | None = None,
              warps: int | None = None, stages: int | None = None, interpret: bool = False):
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

    def keyed(i_t, bh):
        return bh // value_heads, i_t, (bh % value_heads) // group, 0

    def gated(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads

    def solved(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads, 0

    return pl.pallas_call(
        functools.partial(_kkt_solve_kernel, sub=sub, bk=min(bk, width), exact=interpret),
        grid=(length // chunk, batch * value_heads),
        in_specs=[pl.BlockSpec((None, chunk, None, width), keyed), pl.BlockSpec((None, chunk, None), gated),
                  pl.BlockSpec((None, chunk, None), gated)],
        out_specs=pl.BlockSpec((None, chunk, None, chunk), solved),
        out_shape=jax.ShapeDtypeStruct((batch, length, value_heads, chunk), k.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_kkt_solve",
    )(k, g, beta)


def _w_u_kernel(k_ref, v_ref, beta_ref, a_ref, g_ref, w_ref, u_ref, *, bk: int, bv: int, exact: bool):
    """fla's `recompute_w_u_fwd_kernel` (fla/ops/gated_delta_rule/wy_fast.py:39-103) for one chunk
    of one head: `u = A (v * beta)` and `w = A (k * beta * exp2(g))`, each operand rounded to its
    input dtype before the product and `A`'s product with `v` at IEEE fp32 (`allow_tf32=False`)."""
    beta = beta_ref[...]
    inverse = a_ref[...]
    for block in range(v_ref.shape[-1] // bv):
        columns = pl.ds(block * bv, bv)
        value = v_ref[:, columns]
        scaled = (value * beta[:, None]).astype(value.dtype)
        u_ref[:, columns] = _dot(inverse, scaled, exact=exact, ieee=True).astype(u_ref.dtype)
    decay = jnp.exp2(g_ref[...].astype(jnp.float32))
    for block in range(k_ref.shape[-1] // bk):
        columns = pl.ds(block * bk, bk)
        key = k_ref[:, columns]
        scaled = key * beta[:, None] * decay[:, None]
        w_ref[:, columns] = _dot(inverse, scaled.astype(key.dtype), exact=exact).astype(w_ref.dtype)


def recompute_w_u(k, v, beta, a, g, *, chunk: int = 64, bk: int | None = None, bv: int | None = None,
                  warps: int | None = None, stages: int | None = None, interpret: bool = False):
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

    def keyed(i_t, bh):
        return bh // value_heads, i_t, (bh % value_heads) // group, 0

    def headed(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads, 0

    def gated(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads

    gate = pl.BlockSpec((None, chunk, None), gated)
    return pl.pallas_call(
        functools.partial(_w_u_kernel, bk=min(bk, width), bv=min(bv, columns), exact=interpret),
        grid=(length // chunk, batch * value_heads),
        in_specs=[pl.BlockSpec((None, chunk, None, width), keyed),
                  pl.BlockSpec((None, chunk, None, columns), headed),
                  gate, pl.BlockSpec((None, chunk, None, chunk), headed), gate],
        out_specs=[pl.BlockSpec((None, chunk, None, width), headed),
                   pl.BlockSpec((None, chunk, None, columns), headed)],
        out_shape=[jax.ShapeDtypeStruct((batch, length, value_heads, width), k.dtype),
                   jax.ShapeDtypeStruct(v.shape, v.dtype)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_recompute_w_u",
    )(k, v, beta, a, g)


def _gated_state_write(key, gated, *, exact: bool):
    """The fp32 gated value as two input-dtype parts for k^T v, summed in fp32. fla rounds gated
    v_new once to bf16/fp16; retaining its residual on chip removes that quantization error with
    one additional narrow dot and zero extra HBM bytes. FP32 inputs keep their existing product."""
    high = gated.astype(key.dtype)
    update = _dot(key.T, high, exact=exact)
    if jnp.dtype(key.dtype).itemsize == 2:
        low = (gated - high.astype(jnp.float32)).astype(key.dtype)
        update += _dot(key.T, low, exact=exact)
    return update


def _h_kernel(k_ref, u_ref, w_ref, g_ref, h0_ref, h_ref, corrected_ref, ht_ref, *, chunk: int, exact: bool):
    """fla's `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (fla/ops/common/chunk_delta_h.py:50-
    343) for one row's head and `BV` value columns, its chunks in order. The memory is fp32 in
    64-key slices; each chunk stores the memory it enters in `h`'s dtype, then the corrected values
    `u - w h` (fla's `v_new`; `h` cast to `w`'s dtype for the product), gates them by
    `exp2(g_last - g)` and the memory by `exp2(g_last)`, casts the gated values to `k`'s dtype and
    adds `k^T v` to the memory. Narrow inputs keep the gated-value residual as a second bf16/fp16
    part on chip: h remains fp32, while h/corrected HBM outputs keep fla's original storage dtypes."""
    length, width = k_ref.shape
    slices = width // 64
    state = [h0_ref[pl.ds(i * 64, 64), :].astype(jnp.float32) for i in range(slices)]

    def step(n, state):
        rows = pl.ds(n * chunk, chunk)
        for i in range(slices):
            h_ref[n, pl.ds(i * 64, 64), :] = state[i].astype(h_ref.dtype)
        predicted = sum(_dot(w_ref[rows, pl.ds(i * 64, 64)], state[i].astype(w_ref.dtype), exact=exact)
                        for i in range(slices))
        corrected = u_ref[rows, :].astype(jnp.float32) - predicted
        corrected_ref[rows, :] = corrected.astype(corrected_ref.dtype)
        gate = g_ref[rows].astype(jnp.float32)
        last = g_ref[n * chunk + chunk - 1].astype(jnp.float32)
        gated = corrected * jnp.exp2(last - gate)[:, None]
        return [state[i] * jnp.exp2(last) + _gated_state_write(k_ref[rows, pl.ds(i * 64, 64)], gated,
                                                            exact=exact)
                for i in range(slices)]

    state = jax.lax.fori_loop(0, length // chunk, step, state)
    for i in range(slices):
        ht_ref[pl.ds(i * 64, 64), :] = state[i].astype(ht_ref.dtype)


def chunk_fwd_h(k, w, u, g, h0, *, chunk: int = 64, bv: int | None = None,
                warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`(h, corrected, final)`: the memory each chunk enters `[B, T // chunk, HV, K, V]` in `k`'s dtype,
    the corrected values `[B, T, HV, V]` in `u`'s and the final memory `[B, HV, K, V]` fp32: fla's
    `chunk_gated_delta_rule_fwd_h` (chunk_delta_h.py:672-737) with an initial state. `k` `[B, T, H,
    K]`, `w` `[B, T, HV, K]`, `u` `[B, T, HV, V]`, `g` the base-2 cumulated decay `[B, T, HV]`, `h0`
    `[B, HV, K, V]` fp32. `K` a multiple of 64 (fla's blockdim64 kernel takes up to 256). On an
    A100 fla's autotuner chose BV 32, 4 warps, 3 stages (c53)."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("chunk_fwd_h", _rule_shape(k, u, chunk), k.dtype,
                           bv=bv, warps=warps, stages=stages)
    bv, warps, stages = config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = k.shape
    value_heads, columns = u.shape[2:]
    group = value_heads // key_heads

    def keyed(i_v, bh):
        return bh // value_heads, 0, (bh % value_heads) // group, 0

    def whole(i_v, bh):
        return bh // value_heads, 0, bh % value_heads, 0

    def valued(i_v, bh):
        return bh // value_heads, 0, bh % value_heads, i_v

    def gated(i_v, bh):
        return bh // value_heads, 0, bh % value_heads

    def state(i_v, bh):
        return bh // value_heads, bh % value_heads, 0, i_v

    def states(i_v, bh):
        return bh // value_heads, 0, bh % value_heads, 0, i_v

    chunks = length // chunk
    return pl.pallas_call(
        functools.partial(_h_kernel, chunk=chunk, exact=interpret),
        grid=(columns // bv, batch * value_heads),
        in_specs=[pl.BlockSpec((None, length, None, width), keyed),
                  pl.BlockSpec((None, length, None, bv), valued),
                  pl.BlockSpec((None, length, None, width), whole),
                  pl.BlockSpec((None, length, None), gated),
                  pl.BlockSpec((None, None, width, bv), state)],
        out_specs=[pl.BlockSpec((None, chunks, None, width, bv), states),
                   pl.BlockSpec((None, length, None, bv), valued),
                   pl.BlockSpec((None, None, width, bv), state)],
        out_shape=[jax.ShapeDtypeStruct((batch, chunks, value_heads, width, columns), k.dtype),
                   jax.ShapeDtypeStruct(u.shape, u.dtype),
                   jax.ShapeDtypeStruct((batch, value_heads, width, columns), jnp.float32)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_fwd_h",
    )(k, u, w, g, h0)


def _dhu_kernel(q_ref, k_ref, w_ref, g_ref, dht_ref, do_ref, dv_ref, dh_ref, dh0_ref, dv2_ref, *,
                chunk: int, scale: float, exact: bool):
    """fla's `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` (chunk_delta_h.py:359-669) for one
    row's head and `BV` value columns, chunks in reverse. Each stores the gradient of the memory it
    leaves into, `dh`, then `dv2 = dv + (k dh) exp2(g_last - g)` (dh cast to k's dtype), then
    `dh = dh exp2(g_last) + scale (q exp2(g))^T do - w^T dv2`, the query product in fp32 and the w
    product with dv2 cast to w's dtype, as fla's dtypes fall."""
    length, width = k_ref.shape
    slices = width // 64
    grad = [dht_ref[pl.ds(i * 64, 64), :].astype(jnp.float32) for i in range(slices)]
    chunks = length // chunk

    def step(r, grad):
        n = chunks - 1 - r
        rows = pl.ds(n * chunk, chunk)
        for i in range(slices):
            dh_ref[n, pl.ds(i * 64, 64), :] = grad[i].astype(dh_ref.dtype)
        gate = g_ref[rows].astype(jnp.float32)
        last = g_ref[n * chunk + chunk - 1].astype(jnp.float32)
        out_grad = do_ref[rows, :]
        dv = sum(_dot(k_ref[rows, pl.ds(i * 64, 64)], grad[i].astype(k_ref.dtype), exact=exact)
                 for i in range(slices))
        dv = dv * jnp.exp2(last - gate)[:, None] + dv_ref[rows, :].astype(jnp.float32)
        dv2_ref[rows, :] = dv.astype(dv2_ref.dtype)
        decay = jnp.exp2(gate)[:, None]

        def moved(i):
            read = q_ref[rows, pl.ds(i * 64, 64)].astype(jnp.float32) * decay
            written = w_ref[rows, pl.ds(i * 64, 64)]
            return (_dot(read.T, out_grad.astype(jnp.float32), exact=exact) * scale
                    - _dot(written.T, dv.astype(written.dtype), exact=exact))

        return [grad[i] * jnp.exp2(last) + moved(i) for i in range(slices)]

    grad = jax.lax.fori_loop(0, chunks, step, grad)
    for i in range(slices):
        dh0_ref[pl.ds(i * 64, 64), :] = grad[i].astype(dh0_ref.dtype)


def chunk_bwd_dhu(q, k, w, g, dht, do, dv, *, scale: float, chunk: int = 64, bv: int | None = None,
                  warps: int | None = None, stages: int | None = None, interpret: bool = False):
    """`(dh, dh0, dv2)`: the gradient of the memory each chunk leaves into `[B, T // chunk, HV, K, V]`
    in `q`'s dtype, of the initial memory `[B, HV, K, V]` fp32, and the corrected values' full
    gradient `[B, T, HV, V]` in `dv`'s: fla's `chunk_gated_delta_rule_bwd_dhu` (chunk_delta_h.py:740-
    794). `dht` `[B, HV, K, V]` fp32 is the final memory's gradient and `dv` the corrected values'
    from the output alone (`chunk_bwd_dv_local`). fla's A100 choice: BV 32, 4 warps, 3 stages (c53)."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("chunk_bwd_dhu", _rule_shape(k, do, chunk), k.dtype,
                           bv=bv, warps=warps, stages=stages)
    bv, warps, stages = config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = q.shape
    value_heads, columns = do.shape[2:]
    group = value_heads // key_heads

    def keyed(i_v, bh):
        return bh // value_heads, 0, (bh % value_heads) // group, 0

    def whole(i_v, bh):
        return bh // value_heads, 0, bh % value_heads, 0

    def valued(i_v, bh):
        return bh // value_heads, 0, bh % value_heads, i_v

    def gated(i_v, bh):
        return bh // value_heads, 0, bh % value_heads

    def state(i_v, bh):
        return bh // value_heads, bh % value_heads, 0, i_v

    def states(i_v, bh):
        return bh // value_heads, 0, bh % value_heads, 0, i_v

    chunks = length // chunk
    keys = pl.BlockSpec((None, length, None, width), keyed)
    values = pl.BlockSpec((None, length, None, bv), valued)
    memory = pl.BlockSpec((None, None, width, bv), state)
    return pl.pallas_call(
        functools.partial(_dhu_kernel, chunk=chunk, scale=scale, exact=interpret),
        grid=(columns // bv, batch * value_heads),
        in_specs=[keys, keys, pl.BlockSpec((None, length, None, width), whole),
                  pl.BlockSpec((None, length, None), gated), memory, values, values],
        out_specs=[pl.BlockSpec((None, chunks, None, width, bv), states), memory, values],
        out_shape=[jax.ShapeDtypeStruct((batch, chunks, value_heads, width, columns), q.dtype),
                   jax.ShapeDtypeStruct((batch, value_heads, width, columns), jnp.float32),
                   jax.ShapeDtypeStruct(dv.shape, dv.dtype)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_bwd_dhu",
    )(q, k, w, g, dht, do, dv)


def _dv_local_kernel(q_ref, k_ref, g_ref, do_ref, dv_ref, *, scale: float, bk: int, bv: int, exact: bool):
    """fla's `chunk_bwd_kernel_dv_local` (fla/ops/common/chunk_o.py:452-526) for one chunk of one
    head: the corrected values' gradient from the chunk's own output, `dv = A do` with
    `A[s, t] = scale (k_s . q_t) exp2(g_t - g_s)` for `s <= t`, cast to `do`'s dtype for the product."""
    size, width = k_ref.shape
    scores = jnp.zeros((size, size), jnp.float32)
    for block in range(width // bk):
        columns = pl.ds(block * bk, bk)
        scores += _dot(k_ref[:, columns], q_ref[:, columns].T, exact=exact) * scale
    gate = g_ref[...].astype(jnp.float32)
    scores = scores * jnp.exp2(gate[None, :] - gate[:, None])
    index = jnp.arange(size)
    scores = jnp.where(index[:, None] <= index[None, :], scores, 0.0).astype(do_ref.dtype)
    for block in range(do_ref.shape[-1] // bv):
        columns = pl.ds(block * bv, bv)
        dv_ref[:, columns] = _dot(scores, do_ref[:, columns], exact=exact).astype(dv_ref.dtype)


def chunk_bwd_dv_local(q, k, g, do, *, scale: float, chunk: int = 64, bk: int | None = None,
                       bv: int | None = None, warps: int | None = None, stages: int | None = None,
                       interpret: bool = False):
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

    def keyed(i_t, bh):
        return bh // value_heads, i_t, (bh % value_heads) // group, 0

    def headed(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads, 0

    def gated(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads

    keys = pl.BlockSpec((None, chunk, None, width), keyed)
    values = pl.BlockSpec((None, chunk, None, columns), headed)
    return pl.pallas_call(
        functools.partial(_dv_local_kernel, scale=scale, bk=min(bk, width), bv=min(bv, columns),
                          exact=interpret),
        grid=(length // chunk, batch * value_heads),
        in_specs=[keys, keys, pl.BlockSpec((None, chunk, None), gated), values],
        out_specs=values,
        out_shape=jax.ShapeDtypeStruct(do.shape, do.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_bwd_dv_local",
    )(q, k, g, do)


def _state_inner_product(memory, memory_grad):
    """fla chunk_o.py:251: multiply h/dh in their storage dtype, then tl.sum widens to fp32.
    In particular, bf16 products round before reduction; widening both inputs omits that round."""
    return jnp.sum((memory * memory_grad).astype(jnp.float32))


def _dqkwg_kernel(q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, dh_ref, dv_ref, dq_ref, dk_ref, dw_ref,
                  dg_ref, *, scale: float, bv: int, exact: bool):
    """fla's `chunk_bwd_kernel_dqkwg` (fla/ops/common/chunk_o.py:159-331) for one chunk of one key
    head and `BK` key columns, over the value heads it serves in turn and their value columns `bv` at
    a time: `ds = do v^T`, `dq = do h^T`, `dk = v dh^T`, `dw = -dv h^T` and `sum(h dh)`; the gates,
    `ds`'s causal mask, `dq += ds k`, `dk += ds^T q` and the decay's gradient, its last row taking the
    memory's. Each product is gated as it lands, so `dq` and `dk` accumulate over the value heads in
    fp32 and round to their dtype once, where fla rounds each value head's and sums them after."""
    size = q_ref.shape[0]
    queries, keys = q_ref[...], k_ref[...]
    wide_queries, wide_keys = queries.astype(jnp.float32), keys.astype(jnp.float32)
    index = jnp.arange(size)
    causal = index[:, None] >= index[None, :]
    dq = jnp.zeros(queries.shape, jnp.float32)
    dk = jnp.zeros(keys.shape, jnp.float32)
    for head in range(v_ref.shape[1]):
        gate = g_ref[:, head].astype(jnp.float32)
        last = g_ref[size - 1, head].astype(jnp.float32)
        read = jnp.exp2(gate)[:, None] * scale
        kept = jnp.exp2(last - gate)[:, None]
        ds = jnp.zeros((size, size), jnp.float32)
        dw = jnp.zeros(queries.shape, jnp.float32)
        dg = jnp.zeros((size,), jnp.float32)
        dg_last = jnp.zeros((), jnp.float32)
        for block in range(v_ref.shape[-1] // bv):
            columns = pl.ds(block * bv, bv)
            value, out_grad = v_ref[:, head, columns], do_ref[:, head, columns]
            local = dv_ref[:, head, columns]
            memory, memory_grad = h_ref[head, :, columns], dh_ref[head, :, columns]
            dg_last += _state_inner_product(memory, memory_grad) * jnp.exp2(last)
            ds += _dot(out_grad, value.T, exact=exact)
            from_memory = _dot(out_grad, memory.T.astype(out_grad.dtype), exact=exact) * read
            to_memory = _dot(value, memory_grad.T.astype(value.dtype), exact=exact) * kept
            dg_last += jnp.sum(to_memory * wide_keys)
            dg += jnp.sum(from_memory * wide_queries, axis=1) - jnp.sum(to_memory * wide_keys, axis=1)
            dq += from_memory
            dk += to_memory
            dw += _dot(local.astype(value.dtype), memory.T.astype(value.dtype), exact=exact)
        dw_ref[:, head, :] = (-dw).astype(dw_ref.dtype)
        ds = jnp.where(causal, ds * jnp.exp2(gate[:, None] - gate[None, :]), 0.0) * scale
        ds = ds.astype(keys.dtype)
        from_scores = _dot(ds, keys, exact=exact)
        to_scores = _dot(ds.T, queries, exact=exact)
        dg += jnp.sum(from_scores * wide_queries, axis=1) - jnp.sum(to_scores * wide_keys, axis=1)
        dq += from_scores
        dk += to_scores
        dg_ref[:, head] = jnp.where(index < size - 1, dg, dg + dg_last)
    dq_ref[...] = dq.astype(dq_ref.dtype)
    dk_ref[...] = dk.astype(dk_ref.dtype)


def chunk_bwd_dqkwg(q, k, v, w, g, h, dv, do, dh, *, scale: float, chunk: int = 64,
                    bk: int | None = None, bv: int | None = None, warps: int | None = None,
                    stages: int | None = None, interpret: bool = False):
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

    def keyed(i_k, i_t, bh):
        return bh // key_heads, i_t, bh % key_heads, i_k

    def served(i_k, i_t, bh):
        return bh // key_heads, i_t, bh % key_heads, 0

    def gated(i_k, i_t, bh):
        return bh // key_heads, i_t, bh % key_heads

    def state(i_k, i_t, bh):
        return bh // key_heads, i_t, bh % key_heads, i_k, 0

    def partial(i_k, i_t, bh):
        return i_k, bh // key_heads, i_t, bh % key_heads

    keys = pl.BlockSpec((None, chunk, None, bk), keyed)
    values = pl.BlockSpec((None, chunk, group, columns), served)
    memory = pl.BlockSpec((None, None, group, bk, columns), state)
    dq, dk, dw, dg = pl.pallas_call(
        functools.partial(_dqkwg_kernel, scale=scale, bv=bv, exact=interpret),
        grid=(width // bk, length // chunk, batch * key_heads),
        in_specs=[keys, keys, values, pl.BlockSpec((None, chunk, group), gated), memory, values, memory,
                  values],
        out_specs=[keys, keys, pl.BlockSpec((None, chunk, group, bk), keyed),
                   pl.BlockSpec((None, None, chunk, group), partial)],
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype),
                   jax.ShapeDtypeStruct(k.shape, k.dtype),
                   jax.ShapeDtypeStruct(w.shape, w.dtype),
                   jax.ShapeDtypeStruct((width // bk, batch, length, value_heads), jnp.float32)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_bwd_dqkwg",
    )(q, k, v, g, h, do, dh, dv)
    return dq, dk, dw, dg.sum(0)


def _wy_gate_sum(d_gate, weighted):
    """fla wy_fast.py:246: subtract the two fp32 reductions before adding the gate accumulator."""
    return d_gate + (jnp.sum(weighted, axis=1) - jnp.sum(weighted, axis=0))


def _wy_bwd_kernel(k_ref, v_ref, beta_ref, g_ref, a_ref, dw_ref, du_ref, dk_ref, dv_ref, db_ref, dg_ref, *,
                   bk: int, bv: int, exact: bool):
    """fla's `prepare_wy_repr_bwd_kernel` (fla/ops/gated_delta_rule/wy_fast.py:119-243) for one
    chunk of one head. With `X = (I + L)^-1` (fla's `A`), `w = X (k beta exp2(g))` and
    `u = X (v beta)`: their gradients through `X^T`, `dX = dw (k beta exp2 g)^T + du (v beta)^T`, then
    `dL = -X^T tril(dX, -1) X^T` gated by `exp2(g_s - g_t)`, and from it the keys', beta's and the
    decay's gradients. `dk` is stored once from the first pass and read back, rounded to its dtype,
    as fla's kernel reads its own store."""
    inverse_t = a_ref[...].T
    beta = beta_ref[...]
    gate = g_ref[...].astype(jnp.float32)
    decay = jnp.exp2(gate)
    size, width = k_ref.shape
    d_x = jnp.zeros((size, size), jnp.float32)
    d_beta = jnp.zeros((size,), jnp.float32)
    d_gate = jnp.zeros((size,), jnp.float32)
    for block in range(width // bk):
        columns = pl.ds(block * bk, bk)
        key = k_ref[:, columns]
        scaled = key * (beta * decay)[:, None]
        w_grad = dw_ref[:, columns]
        d_x += _dot(w_grad, scaled.T.astype(w_grad.dtype), exact=exact)
        d_scaled = _dot(inverse_t, w_grad, exact=exact)
        dk_ref[:, columns] = (d_scaled * (decay * beta)[:, None]).astype(dk_ref.dtype)
        d_beta += jnp.sum(d_scaled * key * decay[:, None], axis=1)
        d_gate += jnp.sum(d_scaled * scaled, axis=1)
    for block in range(v_ref.shape[-1] // bv):
        columns = pl.ds(block * bv, bv)
        value = v_ref[:, columns]
        scaled = (value * beta[:, None]).astype(value.dtype)
        u_grad = du_ref[:, columns]
        d_x += _dot(u_grad, scaled.T, exact=exact)
        d_scaled = _dot(inverse_t, u_grad, exact=exact)
        dv_ref[:, columns] = (d_scaled * beta[:, None]).astype(dv_ref.dtype)
        d_beta += jnp.sum(d_scaled * value, axis=1)
    index = jnp.arange(size)
    strict = index[:, None] > index[None, :]
    d_x = jnp.where(strict, d_x, 0.0)
    d_x = _dot(d_x.astype(inverse_t.dtype), inverse_t, exact=exact)
    d_x = _dot(inverse_t, d_x.astype(inverse_t.dtype), exact=exact)
    d_x = d_x * jnp.exp2(gate[:, None] - gate[None, :])
    d_l = jnp.where(strict, -d_x, 0.0).astype(k_ref.dtype)
    products = jnp.zeros((size, size), jnp.float32)
    for block in range(width // bk):
        columns = pl.ds(block * bk, bk)
        key = k_ref[:, columns]
        scaled = key * beta[:, None]
        products += _dot(key, key.T, exact=exact)
        d_key = _dot(d_l, key, exact=exact)
        d_beta += jnp.sum(d_key * key, axis=1)
        full = d_key * beta[:, None] + _dot(scaled.T.astype(d_l.dtype), d_l, exact=exact).T
        dk_ref[:, columns] = (full + dk_ref[:, columns].astype(jnp.float32)).astype(dk_ref.dtype)
    db_ref[...] = d_beta.astype(db_ref.dtype)
    weighted = d_l.astype(jnp.float32) * (products * beta[:, None])
    dg_ref[...] = _wy_gate_sum(d_gate, weighted).astype(dg_ref.dtype)


def prepare_wy_repr_bwd(k, v, beta, g, a, dw, du, *, chunk: int = 64, bk: int | None = None,
                        bv: int | None = None, warps: int | None = None, stages: int | None = None,
                        interpret: bool = False):
    """`(dk, dv, db, dg)` through the WY representation: fla's `prepare_wy_repr_bwd` (wy_fast.py:
    296-343). `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `beta` and `g` `[B, T, HV]`, `a` the inverse
    per chunk `[B, T, HV, BT]`, `dw` and `du` the gradients of `w` and `u`. `dk` is summed over each
    key head's group. fla tiles `BK`, `BV` at 64 on an A100 and chose 4 warps, 2 stages (c53)."""
    from jax.experimental.pallas import triton as plgpu

    config = kernel_config("prepare_wy_repr_bwd", _rule_shape(k, v, chunk), k.dtype,
                           bk=bk, bv=bv, warps=warps, stages=stages)
    bk, bv, warps, stages = config["bk"], config["bv"], config["warps"], config["stages"]

    batch, length, key_heads, width = k.shape
    value_heads, columns = v.shape[2:]
    group = value_heads // key_heads

    def keyed(i_t, bh):
        return bh // value_heads, i_t, (bh % value_heads) // group, 0

    def headed(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads, 0

    def gated(i_t, bh):
        return bh // value_heads, i_t, bh % value_heads

    gate = pl.BlockSpec((None, chunk, None), gated)
    per_key = pl.BlockSpec((None, chunk, None, width), headed)
    per_value = pl.BlockSpec((None, chunk, None, columns), headed)
    dk, dv, db, dg = pl.pallas_call(
        functools.partial(_wy_bwd_kernel, bk=min(bk, width), bv=min(bv, columns), exact=interpret),
        grid=(length // chunk, batch * value_heads),
        in_specs=[pl.BlockSpec((None, chunk, None, width), keyed), per_value, gate, gate,
                  pl.BlockSpec((None, chunk, None, chunk), headed), per_key, per_value],
        out_specs=[per_key, per_value, gate, gate],
        out_shape=[jax.ShapeDtypeStruct((batch, length, value_heads, width), k.dtype),
                   jax.ShapeDtypeStruct(v.shape, v.dtype),
                   jax.ShapeDtypeStruct(beta.shape, beta.dtype),
                   jax.ShapeDtypeStruct(g.shape, g.dtype)],
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_prepare_wy_repr_bwd",
    )(k, v, beta, g, a, dw, du)
    if group > 1:
        dk = dk.reshape(batch, length, key_heads, group, width).sum(3).astype(k.dtype)
    return dk, dv, db, dg


def _o_kernel(q_ref, k_ref, v_ref, h_ref, g_ref, o_ref, *, scale: float, bk: int, exact: bool):
    """fla's `chunk_fwd_kernel_o` (fla/ops/common/chunk_o.py:43-147) for one chunk of one head and
    `BV` value columns: `o = scale * ((q h) exp2(g) + tril((q k^T) exp2(g_s - g_t)) v)`, the key
    columns walked `bk` at a time into fp32 accumulators, `A` cast to the values' dtype for its
    product, as fla does."""
    size, width = q_ref.shape
    out = jnp.zeros((size, v_ref.shape[-1]), jnp.float32)
    scores = jnp.zeros((size, size), jnp.float32)
    for block in range(width // bk):
        columns = pl.ds(block * bk, bk)
        query = q_ref[:, columns]
        out += _dot(query, h_ref[columns, :], exact=exact)
        scores += _dot(query, k_ref[:, columns].T, exact=exact)
    gate = g_ref[...].astype(jnp.float32)
    out = out * jnp.exp2(gate)[:, None]
    scores = scores * jnp.exp2(gate[:, None] - gate[None, :])
    index = jnp.arange(size)
    scores = jnp.where(index[:, None] >= index[None, :], scores, 0.0)
    value = v_ref[...]
    out = out * scale + _dot(scores.astype(value.dtype), value, exact=exact) * scale
    o_ref[...] = out.astype(o_ref.dtype)


def chunk_fwd_o(q, k, v, h, g, *, scale: float, chunk: int = 64, bk: int | None = None,
                bv: int | None = None, warps: int | None = None, stages: int | None = None,
                interpret: bool = False):
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

    def head(bh):
        return bh // value_heads, bh % value_heads

    def keyed(i_v, i_t, bh):
        b, i_h = head(bh)
        return b, i_t, i_h // group, 0

    def valued(i_v, i_t, bh):
        b, i_h = head(bh)
        return b, i_t, i_h, i_v

    def state(i_v, i_t, bh):
        b, i_h = head(bh)
        return b, i_t, i_h, 0, i_v

    def gated(i_v, i_t, bh):
        b, i_h = head(bh)
        return b, i_t, i_h

    return pl.pallas_call(
        functools.partial(_o_kernel, scale=scale, bk=min(bk, width), exact=interpret),
        grid=(columns // bv, length // chunk, batch * value_heads),
        in_specs=[pl.BlockSpec((None, chunk, None, width), keyed),
                  pl.BlockSpec((None, chunk, None, width), keyed),
                  pl.BlockSpec((None, chunk, None, bv), valued),
                  pl.BlockSpec((None, None, None, width, bv), state),
                  pl.BlockSpec((None, chunk, None), gated)],
        out_specs=pl.BlockSpec((None, chunk, None, bv), valued),
        out_shape=jax.ShapeDtypeStruct(v.shape, v.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=interpret, name="fla_chunk_fwd_o",
    )(q, k, v, h, g)


def reference_o(q, k, v, h, g, *, scale: float, chunk: int = 64):
    """`chunk_fwd_o`'s math in XLA, per chunk, for the tests: the same dtypes and casts."""
    batch, length, key_heads, _ = q.shape
    value_heads = v.shape[2]
    group = value_heads // key_heads
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:])

    # Widened before the products: exact for bf16, and XLA:CPU has no bf16 x bf16 -> fp32 dot.
    qc, kc, vc = (chunks(x).astype(jnp.float32) for x in (q, k, v))
    gc = chunks(g).astype(jnp.float32)
    h = h.astype(jnp.float32)
    out = jnp.einsum('bnchk,bnhkv->bnchv', qc, h, preferred_element_type=jnp.float32)
    out = out * jnp.exp2(gc)[..., None]
    scores = jnp.einsum('bnshk,bnthk->bnhst', qc, kc, preferred_element_type=jnp.float32)
    scores = scores * jnp.exp2(jnp.moveaxis(gc, 3, 2)[..., :, None] - jnp.moveaxis(gc, 3, 2)[..., None, :])
    index = jnp.arange(chunk)
    scores = jnp.where(index[:, None] >= index[None, :], scores, 0.0)
    out = out * scale + jnp.einsum('bnhst,bnthv->bnshv', scores.astype(v.dtype).astype(jnp.float32), vc,
                                   preferred_element_type=jnp.float32) * scale
    return out.reshape(v.shape).astype(v.dtype)


def reference_w_u(k, v, beta, a, g, *, chunk: int = 64):
    """`recompute_w_u`'s math in XLA, for the tests: the same roundings, products widened (exact)."""
    group = v.shape[2] // k.shape[2]
    k = jnp.repeat(k, group, axis=2)
    batch, length = v.shape[:2]

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:]).astype(jnp.float32)

    inverse = chunks(a)
    scaled_v = (v * beta[..., None]).astype(v.dtype)
    scaled_k = (k * beta[..., None] * jnp.exp2(g.astype(jnp.float32))[..., None]).astype(k.dtype)
    u = jnp.einsum('bnshc,bnchv->bnshv', inverse, chunks(scaled_v), precision=jax.lax.Precision.HIGHEST)
    w = jnp.einsum('bnshc,bnchk->bnshk', inverse, chunks(scaled_k), precision=jax.lax.Precision.HIGHEST)
    return w.reshape(*k.shape[:3], k.shape[-1]).astype(k.dtype), u.reshape(v.shape).astype(v.dtype)


def reference_kkt_solve(k, g, beta, *, chunk: int = 64):
    """`kkt_solve`'s math in XLA, for the tests: the inverse in float64 of the same matrix."""
    group = beta.shape[2] // k.shape[2]
    dtype = k.dtype
    k = jnp.repeat(k, group, axis=2).astype(jnp.float32)
    batch, length, heads = beta.shape

    def chunks(x):
        return jnp.moveaxis(x.reshape(batch, length // chunk, chunk, heads, *x.shape[3:]), 3, 2)

    kc, gc, bc = chunks(k), chunks(g.astype(jnp.float32)), chunks(beta.astype(jnp.float32))
    a = jnp.einsum('bnhsk,bnhtk->bnhst', kc, kc, precision=jax.lax.Precision.HIGHEST)
    a = a * jnp.exp2(gc[..., :, None] - gc[..., None, :]) * bc[..., :, None]
    a = jnp.tril(a, -1)
    inverse = jnp.linalg.inv(jnp.eye(chunk) + a)
    return jnp.moveaxis(inverse, 2, 3).reshape(batch, length, heads, chunk).astype(dtype)


def reference_h(k, w, u, g, h0, *, chunk: int = 64):
    """`chunk_fwd_h`'s math in XLA, a scan over chunks with the same roundings, products widened."""
    group = u.shape[2] // k.shape[2]
    dtype = k.dtype
    k = jnp.repeat(k, group, axis=2)
    batch, length = u.shape[:2]
    exact = jax.lax.Precision.HIGHEST

    def chunks(x):
        return jnp.moveaxis(x.reshape(batch, length // chunk, chunk, *x.shape[2:]), 1, 0)

    def step(state, inputs):
        keys, weights, values, gate = inputs
        entered = state.astype(dtype)
        predicted = jnp.einsum('bchk,bhkv->bchv', weights.astype(jnp.float32),
                               state.astype(weights.dtype).astype(jnp.float32), precision=exact)
        corrected = values.astype(jnp.float32) - predicted
        last = gate[:, -1]
        gated = corrected * jnp.exp2(last[:, None] - gate)[..., None]
        high = gated.astype(dtype)
        update = jnp.einsum('bchk,bchv->bhkv', keys.astype(jnp.float32), high.astype(jnp.float32),
                            precision=exact)
        if jnp.dtype(dtype).itemsize == 2:
            low = (gated - high.astype(jnp.float32)).astype(dtype)
            update += jnp.einsum('bchk,bchv->bhkv', keys.astype(jnp.float32), low.astype(jnp.float32),
                                 precision=exact)
        state = state * jnp.exp2(last)[..., None, None] + update
        return state, (entered, corrected.astype(u.dtype))

    final, (h, corrected) = jax.lax.scan(step, h0.astype(jnp.float32),
                                     (chunks(k), chunks(w), chunks(u), chunks(g.astype(jnp.float32))))
    return jnp.moveaxis(h, 0, 1), jnp.moveaxis(corrected, 0, 1).reshape(u.shape), final


def reference_dhu(q, k, w, g, dht, do, dv, *, scale: float, chunk: int = 64):
    """`chunk_bwd_dhu`'s math in XLA, a reverse scan over chunks with the same roundings."""
    group = do.shape[2] // q.shape[2]
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))
    batch, length = do.shape[:2]
    exact = jax.lax.Precision.HIGHEST

    def chunks(x):
        return jnp.moveaxis(x.reshape(batch, length // chunk, chunk, *x.shape[2:]), 1, 0)

    def step(grad, inputs):
        queries, keys, weights, gate, out_grad, local = inputs
        stored = grad.astype(q_dtype)
        last = gate[:, -1]
        full = jnp.einsum('bchk,bhkv->bchv', keys.astype(jnp.float32),
                          grad.astype(k_dtype).astype(jnp.float32), precision=exact)
        full = full * jnp.exp2(last[:, None] - gate)[..., None] + local.astype(jnp.float32)
        read = queries.astype(jnp.float32) * jnp.exp2(gate)[..., None]
        grad = (grad * jnp.exp2(last)[..., None, None]
                + jnp.einsum('bchk,bchv->bhkv', read, out_grad.astype(jnp.float32), precision=exact) * scale
                - jnp.einsum('bchk,bchv->bhkv', weights.astype(jnp.float32),
                             full.astype(weights.dtype).astype(jnp.float32), precision=exact))
        return grad, (stored, full.astype(dv.dtype))

    initial, (dh, dv2) = jax.lax.scan(
        step, dht.astype(jnp.float32),
        (chunks(q), chunks(k), chunks(w), chunks(g.astype(jnp.float32)), chunks(do), chunks(dv)),
        reverse=True)
    return jnp.moveaxis(dh, 0, 1), initial, jnp.moveaxis(dv2, 0, 1).reshape(dv.shape)


def reference_dv_local(q, k, g, do, *, scale: float, chunk: int = 64):
    """`chunk_bwd_dv_local`'s math in XLA, per chunk, with the same casts."""
    group = do.shape[2] // q.shape[2]
    q, k = (jnp.repeat(x, group, axis=2).astype(jnp.float32) for x in (q, k))
    batch, length = do.shape[:2]

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:])

    gc = jnp.moveaxis(chunks(g.astype(jnp.float32)), 3, 2)
    exact = jax.lax.Precision.HIGHEST
    scores = jnp.einsum('bnshk,bnthk->bnhst', chunks(k), chunks(q), precision=exact) * scale
    scores = scores * jnp.exp2(gc[..., None, :] - gc[..., :, None])
    index = jnp.arange(chunk)
    scores = jnp.where(index[:, None] <= index[None, :], scores, 0.0).astype(do.dtype).astype(jnp.float32)
    dv = jnp.einsum('bnhst,bnthv->bnshv', scores, chunks(do).astype(jnp.float32), precision=exact)
    return dv.reshape(do.shape).astype(do.dtype)


def reference_dqkwg(q, k, v, w, g, h, dv, do, dh, *, scale: float, chunk: int = 64):
    """`chunk_bwd_dqkwg`'s math in XLA, per chunk, with the same casts, products widened, and a key
    head's value heads summed in fp32 before `dq` and `dk` round."""
    key_heads = q.shape[2]
    group = v.shape[2] // key_heads
    q_dtype, k_dtype = q.dtype, k.dtype
    batch, length, heads = v.shape[:3]
    exact = jax.lax.Precision.HIGHEST
    f32 = jnp.float32
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:])

    qc, kc, vc, doc, dvc = (chunks(x) for x in (q, k, v, do, dv))
    gc = chunks(g.astype(f32))
    last = gc[:, :, -1]
    dg_last = jnp.sum((h * dh).astype(f32), axis=(-2, -1)) * jnp.exp2(last)
    ds = jnp.einsum('bnshv,bnthv->bnhst', doc.astype(f32), vc.astype(f32), precision=exact)
    dq = jnp.einsum('bnshv,bnhkv->bnshk', doc.astype(f32), h.astype(do.dtype).astype(f32), precision=exact)
    dk = jnp.einsum('bnshv,bnhkv->bnshk', vc.astype(f32), dh.astype(v.dtype).astype(f32), precision=exact)
    dw = -jnp.einsum('bnshv,bnhkv->bnshk', dvc.astype(v.dtype).astype(f32), h.astype(v.dtype).astype(f32),
                     precision=exact)
    dq = dq * jnp.exp2(gc)[..., None] * scale
    dk = dk * jnp.exp2(last[:, :, None] - gc)[..., None]
    dg_last = dg_last + jnp.einsum('bnshk,bnshk->bnh', dk, kc.astype(f32))
    gh = jnp.moveaxis(gc, 3, 2)
    index = jnp.arange(chunk)
    causal = index[:, None] >= index[None, :]
    ds = jnp.where(causal, ds * jnp.exp2(gh[..., :, None] - gh[..., None, :]), 0.0) * scale
    ds = ds.astype(k_dtype).astype(f32)
    dq = dq + jnp.einsum('bnhst,bnthk->bnshk', ds, kc.astype(f32), precision=exact)
    dk = dk + jnp.einsum('bnhst,bnshk->bnthk', ds, qc.astype(f32), precision=exact)
    dg = jnp.sum(dq * qc.astype(f32), -1) - jnp.sum(dk * kc.astype(f32), -1)
    dg = dg.at[:, :, -1].add(dg_last)
    shape = (batch, length, key_heads, group, q.shape[-1])
    dq, dk = dq.reshape(shape).sum(3).astype(q_dtype), dk.reshape(shape).sum(3).astype(k_dtype)
    return dq, dk, dw.reshape(w.shape).astype(w.dtype), dg.reshape(batch, length, heads)


def _rule(q, k, v, g, beta, h0, scale: float, chunk: int, interpret: bool):
    return _rule_fwd(q, k, v, g, beta, h0, scale, chunk, interpret)[0]


def _rule_fwd(q, k, v, g, beta, h0, scale: float, chunk: int, interpret: bool):
    """fla's `chunk_gated_delta_rule_fwd` (fla/ops/gated_delta_rule/chunk.py:33-123): the decay
    cumulated in base 2, the inverse, `w` and `u`, the memory across chunks, then the output. Saves
    what fla saves: the cumulated decay, the inverse and the initial memory."""
    gc = chunk_local_cumsum(g, chunk=chunk, scale=RCP_LN2, interpret=interpret)
    a = kkt_solve(k, gc, beta, chunk=chunk, interpret=interpret)
    w, u = recompute_w_u(k, v, beta, a, gc, chunk=chunk, interpret=interpret)
    h, corrected, final = chunk_fwd_h(k, w, u, gc, h0, chunk=chunk, interpret=interpret)
    o = chunk_fwd_o(q, k, corrected, h, gc, scale=scale, chunk=chunk, interpret=interpret)
    return (o, final), (q, k, v, gc, beta, a, h0)


def _rule_bwd(scale: float, chunk: int, interpret: bool, residual, cotangent):
    """fla's `chunk_gated_delta_rule_bwd` (chunk.py:126-250): `w`, `u` and the memory recomputed, then
    the output's local value gradient, the reverse recurrence, the query, key, `w` and decay
    gradients, the WY backward, and the decay's gradient summed back over each chunk."""
    q, k, v, gc, beta, a, h0 = residual
    do, dht = cotangent
    w, u = recompute_w_u(k, v, beta, a, gc, chunk=chunk, interpret=interpret)
    h, corrected, _ = chunk_fwd_h(k, w, u, gc, h0, chunk=chunk, interpret=interpret)
    dv = chunk_bwd_dv_local(q, k, gc, do, scale=scale, chunk=chunk, interpret=interpret)
    dh, dh0, dv = chunk_bwd_dhu(q, k, w, gc, dht.astype(jnp.float32), do, dv, scale=scale, chunk=chunk,
                                interpret=interpret)
    dq, dk, dw, dg = chunk_bwd_dqkwg(q, k, corrected, w, gc, h, dv, do, dh, scale=scale, chunk=chunk,
                                     interpret=interpret)
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(k, v, beta, gc, a, dw, dv, chunk=chunk, interpret=interpret)
    dk = (dk.astype(jnp.float32) + dk2.astype(jnp.float32)).astype(k.dtype)
    dg = chunk_local_cumsum(dg + dg2, chunk=chunk, reverse=True, interpret=interpret)
    return dq, dk, dv, dg.astype(jnp.float32), db, dh0


chunk_gated_delta_rule = jax.custom_vjp(_rule, nondiff_argnums=(6, 7, 8))
"""fla's chunked gated delta rule on the ported kernels: `(o, final)` for `q`, `k` `[B, T, H, K]`
(normalized as the caller wants them), `v` `[B, T, HV, V]`, the per-token natural log decay `g`
`[B, T, HV]` fp32, `beta` `[B, T, HV]` and the initial memory `h0` `[B, HV, K, V]` fp32; `T` a whole
number of chunks. Its numbers are fla's: bf16 operands store `A`, `w`, `u`, `h` and the corrected
values in bf16, with fp32 accumulation and fp32 decay, the memory fp32 across chunks."""
chunk_gated_delta_rule.defvjp(_rule_fwd, _rule_bwd)
