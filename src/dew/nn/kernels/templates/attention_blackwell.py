"""The attention template's Blackwell mainloop (Mosaic GPU, sm100), the design's first sm100 rung
(docs/design/kernel-templates.md §6): tcgen05's products with their accumulators in tensor memory, as
FlashAttention-4 and Tokamax's sm100 kernel hold them, and the template's algebra (`attention_algebra`)
between the products.

One program a head and a query block of `BLOCK_Q` rows, two warpgroups. The first issues: one warp's TMA
loads fill a ring of key and value stages, and one warp issues S = Q·Kᵀ into tensor memory and, a step
behind, P·V into the output accumulator there, so a step's S is computed while the last step's weights
multiply their values. The second runs the softmax: it reads S from tensor memory, hands it to the algebra
(`modify`, `peak`, `weights`, `total`), rescales the accumulator in tensor memory once the last P·V is
done (`rescale`), writes the narrowed weights back for the next P·V, and finishes. A causal call walks the
key blocks below its block's diagonal unmasked and the diagonal's masked.

The forward over full and causal calls, 16-bit operands, no hooks, one CTA a block: the rung's first piece.
FlashAttention-4's correction warpgroup, two-CTA products and lazy rescale follow as rungs."""

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates.attention import NATURAL, Hooks, KeySpans, padded_call

BLOCK_Q = 128
"""A program's query rows: tcgen05's row count for one CTA."""
TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE
ROW = TMEM.reduce(1)
"""A tile's register layout as tensor memory loads it, and a per-row vector's in it."""
STORED = plgpu.Layout.TCGEN05
"""The accumulator's register layout for its epilogue, the one a swizzled shared-memory tile takes."""
ISSUE, SOFTMAX = 0, 1
"""The warpgroups: the one issuing loads and products, and the softmax's."""
LOAD_WARP, MMA_WARP = 0, 1
"""The issuing warpgroup's warps: the TMA loads' and the products'."""
ISSUE_REGISTERS, SOFTMAX_REGISTERS = 80, 248
"""Registers a thread of each warpgroup holds (`set_max_registers`): the issuing one's warps hold
addresses and counters, the softmax's a tile of S and its rows, within the multiprocessor's 64K."""
CHUNK = 32
"""The accumulator's columns the softmax rescales at a time."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """A launch's key tile (`block_k`, a multiple of 64) and the key and value tiles in flight
    (`stages`)."""

    block_k: int = 128
    stages: int = 2

    def __post_init__(self):
        if self.block_k % 64 or self.stages < 2:
            raise ValueError(f"{self}: a key tile of whole 64 rows, two stages or more")


TILES = Tiles()
"""The tiles a call takes where it names none."""


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, tiles: Tiles = TILES,
            interpret: bool = False) -> jax.Array:
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and the keys `spans` leaves each query (full or causal), as
    `dew.nn.kernels.templates.attention.reference` computes it. `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic GPU's interpret mode)."""
    if hooks is not None and hooks != Hooks():
        raise NotImplementedError("the sm100 mainloop takes no hooks yet")
    if spans.form not in ("full", "causal"):
        raise NotImplementedError(f"the sm100 mainloop takes full and causal calls, not {spans.form!r} ones")
    if jnp.dtype(query.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        raise NotImplementedError(f"the sm100 mainloop takes 16-bit operands, not {jnp.dtype(query.dtype)}")
    value_width = value.shape[-1]
    call = padded_call(query, key, value, spans, Hooks(), max(BLOCK_Q, tiles.block_k))
    spec = algebra.Forward(spans.form, spans.keys, ids=False,
                           scale=query.shape[3] ** -0.5 if scale is None else scale, natural=natural)
    # Swizzled tiles hold whole 64-element rows of 16-bit operands.
    query, key, value = (jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % 64),))
                         for array in (call.query, call.key, call.value))
    out = _forward(query, key, value, spec=spec, tiles=tiles, interpret=interpret)
    return out[:, :call.queries, :, :value_width]


class _Program(NamedTuple):
    """A program's facts and its scratch: its row, head and first query, the key blocks it walks
    (`steps`), and the shared memory, tensor memory and barriers its roles share."""

    row: jax.Array
    head: jax.Array
    first: jax.Array
    steps: jax.Array
    smem: tuple
    tmem: tuple
    barriers: tuple


def _load(program: _Program, q_ref, k_ref, v_ref, *, group: int, tiles: Tiles):
    """The load warp: the query block once, then each key block's keys and values into the next stage
    once its products have read the stage's last ones."""
    q_smem, k_smem, v_smem, _, _ = program.smem
    q_full, k_full, v_full, k_free, v_free, *_ = program.barriers
    kv_head = lax.div(program.head, group)
    queries = q_ref.at[program.row, pl.ds(program.first, BLOCK_Q), program.head]
    plgpu.copy_gmem_to_smem(queries, q_smem, q_full)

    @pl.loop(0, program.steps)
    def _(index):
        slot = lax.rem(index, tiles.stages)
        tile = (program.row, pl.ds(index * tiles.block_k, tiles.block_k), kv_head)
        for ref, smem, full, free in ((k_ref, k_smem, k_full, k_free), (v_ref, v_smem, v_full, v_free)):
            pl.when(index >= tiles.stages)(functools.partial(plgpu.barrier_wait, free.at[slot]))
            plgpu.copy_gmem_to_smem(ref.at[tile], smem.at[slot], full.at[slot])

    # Each stage's last release, observed so its barriers end where they began.
    for slot in range(tiles.stages):
        for free in (k_free, v_free):
            pl.when(slot < program.steps)(functools.partial(plgpu.barrier_wait, free.at[slot]))


def _products(program: _Program, *, tiles: Tiles):
    """The product warp: each step's S = Q·Kᵀ into tensor memory once the softmax has read the last one,
    and a step behind, P·V into the accumulator once its weights are there."""
    q_smem, k_smem, v_smem, _, _ = program.smem
    s_tmem, p_tmem, acc_tmem = program.tmem
    q_full, k_full, v_full, k_free, v_free, s_full, s_free, p_full, o_done = program.barriers
    plgpu.barrier_wait(q_full)

    def values(index):
        slot = lax.rem(index, tiles.stages)
        plgpu.barrier_wait(v_full.at[slot])
        plgpu.barrier_wait(p_full)
        plgpu.tcgen05_mma(acc_tmem, p_tmem, v_smem.at[slot], v_free.at[slot], accumulate=index > 0)
        plgpu.tcgen05_commit_arrive(o_done)

    @pl.loop(0, program.steps)
    def _(index):
        slot = lax.rem(index, tiles.stages)
        plgpu.barrier_wait(k_full.at[slot])
        pl.when(index > 0)(functools.partial(plgpu.barrier_wait, s_free))
        plgpu.tcgen05_mma(s_tmem, q_smem, k_smem.at[slot].T, k_free.at[slot], accumulate=False)
        plgpu.tcgen05_commit_arrive(s_full)
        pl.when(index > 0)(functools.partial(values, index - 1))

    values(program.steps - 1)


def _softmax(program: _Program, out_ref, *, spec: algebra.Forward, tiles: Tiles, blocks: int, dtype):
    """The softmax warpgroup: each step's S through the algebra, the accumulator taken to the step's
    maximum once the last P·V is in it, the narrowed weights into tensor memory; then the epilogue."""
    _, _, _, o_smem, l_smem = program.smem
    s_tmem, p_tmem, acc_tmem = program.tmem
    *_, s_full, s_free, p_full, o_done = program.barriers
    block_k, value_width = tiles.block_k, acc_tmem.shape[1]
    q_pos = program.first + plgpu.broadcasted_iota(jnp.int32, (BLOCK_Q, block_k), 0, layout=TMEM)
    columns = plgpu.broadcasted_iota(jnp.int32, (BLOCK_Q, block_k), 1, layout=TMEM)

    def step(index, carry, *, masked: bool):
        m, total = carry
        plgpu.barrier_wait(s_full)
        s = plgpu.async_load_tmem(s_tmem, layout=TMEM)
        plgpu.wait_load_tmem()
        # The next S may overwrite this one: only a step that has one releases it.
        pl.when(index + 1 < program.steps)(functools.partial(plgpu.barrier_arrive, s_free))
        logits = algebra.modify(spec, s, algebra.Coords(q_pos, index * block_k + columns), masked=masked)
        m, alpha = algebra.peak(spec, m, logits)
        weights = algebra.weights(spec, logits, m)
        total = algebra.total(total, alpha, weights)

        @pl.when(index > 0)
        def _():
            # The last step's P·V is in the accumulator: take it to this step's maximum, a chunk of
            # columns at a time, so S and the accumulator are never both in registers.
            plgpu.barrier_wait(o_done)
            for start in range(0, value_width, CHUNK):
                chunk = acc_tmem.at[:, pl.ds(start, CHUNK)]
                acc = plgpu.async_load_tmem(chunk, layout=TMEM)
                plgpu.async_store_tmem(chunk, algebra.rescale(acc, alpha))

        plgpu.async_store_tmem(p_tmem, algebra.narrow(weights, dtype))
        plgpu.commit_tmem()
        plgpu.barrier_arrive(p_full)
        return m, total

    whole = program.steps
    padded = bool(spec.keys % block_k)
    if spec.form == "causal" or padded:
        # The blocks every query of the block sees whole, below its diagonal and before padding.
        whole = jnp.maximum(jnp.minimum(lax.div(program.first, block_k), blocks - padded), 0)
    start = (plgpu.layout_cast(jnp.full((BLOCK_Q,), -jnp.inf, jnp.float32), ROW),
             plgpu.layout_cast(jnp.zeros((BLOCK_Q,), jnp.float32), ROW))
    carry = lax.fori_loop(0, whole, functools.partial(step, masked=False), start)
    m, total = lax.fori_loop(whole, program.steps, functools.partial(step, masked=True), carry)
    plgpu.barrier_wait(o_done)
    acc = plgpu.async_load_tmem(acc_tmem, layout=STORED)
    plgpu.wait_load_tmem()
    # The denominator to the epilogue's row layout through shared memory, which relayouts it.
    l_smem[...] = total
    total = plgpu.load(l_smem, (), layout=STORED.reduce(1))
    out, _, _ = algebra.finish(spec, algebra.Partial(acc, m, total), initial=False)
    o_smem[...] = out.astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(o_smem, out_ref.at[program.row, pl.ds(program.first, BLOCK_Q), program.head])
    plgpu.wait_smem_to_gmem(0)


def _forward(query, key, value, *, spec: algebra.Forward, tiles: Tiles, interpret: bool):
    batch, queries, heads, width = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    block_k, stages = tiles.block_k, tiles.stages
    blocks = keys // block_k
    dtype = query.dtype

    def kernel(q_ref, k_ref, v_ref, out_ref, *scratch):
        first = lax.axis_index("q_blocks") * BLOCK_Q
        steps = jnp.minimum(pl.cdiv(first + BLOCK_Q, block_k), blocks) if spec.form == "causal" else blocks
        program = _Program(lax.axis_index("batch"), lax.axis_index("heads"), first, steps, scratch[:5],
                           scratch[5:8], scratch[8:])
        wg = lax.axis_index("wg")

        @pl.when(wg == ISSUE)
        def _():
            plgpu.set_max_registers(ISSUE_REGISTERS, action="decrease")

            @plgpu.warp_map
            def _(warp):
                pl.when(warp == LOAD_WARP)(functools.partial(_load, program, q_ref, k_ref, v_ref, group=group,
                                                             tiles=tiles))
                pl.when(warp == MMA_WARP)(functools.partial(_products, program, tiles=tiles))

        @pl.when(wg == SOFTMAX)
        def _():
            plgpu.set_max_registers(SOFTMAX_REGISTERS, action="increase")
            _softmax(program, out_ref, spec=spec, tiles=tiles, blocks=blocks, dtype=dtype)

    transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
    tensor_core = functools.partial(plgpu.Barrier, orders_tensor_core=True)
    smem = (plgpu.SMEM((BLOCK_Q, width), dtype, transforms=transforms),
            plgpu.SMEM((stages, block_k, width), dtype, transforms=transforms),
            plgpu.SMEM((stages, block_k, value_width), dtype, transforms=transforms),
            plgpu.SMEM((BLOCK_Q, value_width), dtype, transforms=transforms),
            plgpu.SMEM((BLOCK_Q,), jnp.float32))
    tmem = (plgpu.TMEM((BLOCK_Q, block_k), jnp.float32), plgpu.TMEM((BLOCK_Q, block_k), dtype, packed=True),
            plgpu.TMEM((BLOCK_Q, value_width), jnp.float32))
    barriers = (plgpu.Barrier(), plgpu.Barrier(num_barriers=stages), plgpu.Barrier(num_barriers=stages),
                tensor_core(num_barriers=stages), tensor_core(num_barriers=stages),
                *(tensor_core() for _ in range(4)))
    return plgpu.kernel(
        kernel, out_type=jax.ShapeDtypeStruct((batch, queries, heads, value_width), dtype),
        grid=(heads, queries // BLOCK_Q, batch), grid_names=("heads", "q_blocks", "batch"),
        num_threads=2, thread_name="wg", scratch_types=(*smem, *tmem, *barriers),
        compiler_params=plgpu.CompilerParams(approx_math=True),
        interpret=plgpu.InterpretGPUParams() if interpret else None,
    )(query, key, value)
