"""The attention template's Blackwell mainloop (Mosaic GPU, sm100), the design's first sm100 rung
(docs/design/kernel-templates.md §6), on the tcgen05 bundle (`mosaic_sm100`, the GEMM template's
`gemm.mosaic_blackwell` beside it): tcgen05's products with their accumulators in tensor memory, as
FlashAttention-4 and Tokamax's sm100 kernel hold them, and the template's algebra (`attention_algebra`)
between the products.

A program is a head's query block of `TCGEN05.rows` rows, in the bundle's two warpgroups
(`Tcgen05.roles`). The issuing one's load warp fills a ring of key stages and one of value stages
(`mosaic_sm100.Stages`), and its product warp issues each step's `S = Q Kᵀ` into tensor memory and, a step
behind, `P V` into the output's accumulator there, so a step's `S` is computed while the last step's
weights multiply their values. The reading warpgroup runs the softmax: it reads `S` out of tensor memory,
hands it to the algebra (`logits_of`, `peak`, `weights`, `denominator_after`), rescales the output's
accumulator in place once the last `P V` is in it (`rescale`, `CHUNK` columns at a time, so `S` and the
accumulator are never both in registers), writes the narrowed weights back for the next `P V`, and
finishes. A causal call walks the key blocks below its block's diagonal unmasked and the diagonal's masked.

The forward over full and causal calls of 16-bit operands, no hooks, one CTA a block: the rung's first
piece. FlashAttention-4's correction warpgroup, two-CTA products and lazy rescale follow as rungs."""

import dataclasses
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_algebra as algebra, attention_portable, mosaic_sm100
from dew.nn.kernels.templates.attention import NATURAL, Hooks, KeySpans, padded_call, prescaled
from dew.nn.kernels.templates.mosaic_sm100 import ROW, STORED, TCGEN05, TMEM

CHUNK = 32
"""The output accumulator's columns the softmax rescales at a time."""


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


def refused(query: jax.Array, spans: KeySpans, hooks: Hooks | None) -> str | None:
    """Why this rung takes no call of `query`'s operands over `spans` under `hooks`, or None: it takes full
    and causal calls of 16-bit operands with no hook."""
    if hooks is not None and hooks != Hooks():
        return "the sm100 mainloop takes no hooks yet"
    if spans.form not in ("full", "causal"):
        return f"the sm100 mainloop takes full and causal calls, not {spans.form!r} ones"
    if jnp.dtype(query.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        return f"the sm100 mainloop takes 16-bit operands, not {jnp.dtype(query.dtype)}"
    return None


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, tiles: Tiles = TILES,
            interpret: bool = False) -> jax.Array:
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and the keys `spans` leaves each query (full or causal), as
    `dew.nn.kernels.templates.attention.reference` computes it. `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic GPU's interpreter)."""
    if reason := refused(query, spans, hooks):
        raise NotImplementedError(reason)
    value_width = value.shape[-1]
    # The query taken by its scale's power of two before its products (`prescaled`), so a logit finite at
    # the call's scale is finite in the product.
    query, scale, power = prescaled(query, query.shape[3] ** -0.5 if scale is None else scale)
    # Queries and keys padded to whole tiles of both: a program's query block and a walk's key tile.
    call = padded_call(query, key, value, spans, Hooks(), math.lcm(TCGEN05.rows, tiles.block_k))
    spec = algebra.Forward(spans.form, spans.keys, ids=False, scale=scale, natural=natural,
                           headroom=algebra.headroom(value.dtype), prescale=power)
    # Swizzled tiles hold whole 64-element rows of 16-bit operands.
    query, key, value = (jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % 64),))
                         for array in (call.query, call.key, call.value))

    def launch(form: algebra.Forward, live: jax.Array | None):
        if live is None:
            return _forward(query, key, value, spec=form, tiles=tiles, interpret=interpret)
        # The rows whose weighted values overflowed, rewalked shifting on the plain-JAX mainloop (the
        # rare launch, `algebra.walked_launches`); its maxima and overflows are not read.
        out, _ = attention_portable.padded_walk(form, call, Hooks(), block_q=TCGEN05.rows,
                                                block_k=tiles.block_k, residuals=False)
        return out[..., :query.shape[3]], jnp.zeros(live.shape, jnp.float32), jnp.zeros_like(live)

    def kept(live: jax.Array, guarded: jax.Array, plain: jax.Array) -> jax.Array:
        return jnp.where(jnp.swapaxes(live, 1, 2)[..., None], guarded, plain)

    out = algebra.walked_launches(spec, launch, kept)
    return out[:, :call.queries, :, :value_width]


class _Scratch(NamedTuple):
    """A program's opened scratch: its query block and its load, the key and value rings
    (`mosaic_sm100.Ring`), `S`'s accumulator (`mosaic_sm100.Tensor`: the products commit it, the softmax
    releases it once read), the weights in tensor memory and their barrier, the output's accumulator and
    the barrier each `P V` commits, and the epilogue's stages."""

    q_smem: object
    q_loaded: object
    keys: mosaic_sm100.Ring
    values: mosaic_sm100.Ring
    scores: mosaic_sm100.Tensor
    p_tmem: object
    p_full: object
    acc_tmem: object
    o_done: object
    o_smem: object
    l_smem: object


class _Block(NamedTuple):
    """A program's query block from the grid: its row, head and first query, and the key blocks it walks
    (`steps`) of the call's (`blocks`), each read where a role reads it, inside the role (a value one role's
    branch closes over from the program's body fails Mosaic GPU's interpreter)."""

    causal: bool
    blocks: int
    block_k: int

    @property
    def row(self):
        return lax.axis_index("batch")

    @property
    def head(self):
        return lax.axis_index("heads")

    @property
    def first(self):
        return lax.axis_index("q_blocks") * TCGEN05.rows

    @property
    def steps(self):
        if not self.causal:
            return self.blocks
        return jnp.minimum(pl.cdiv(self.first + TCGEN05.rows, self.block_k), self.blocks)


def _load(block: _Block, scratch: _Scratch, q_ref, k_ref, v_ref, *, group: int, tiles: Tiles) -> None:
    """The load warp: the query block once, then each key block's keys and values into their rings."""
    queries = q_ref.at[block.row, pl.ds(block.first, TCGEN05.rows), block.head]
    plgpu.copy_gmem_to_smem(queries, scratch.q_smem, scratch.q_loaded)
    kv_head = lax.div(block.head, group)

    @pl.loop(0, block.blocks)
    def _(index):
        @pl.when(index < block.steps)
        def _():
            tile = (block.row, pl.ds(index * tiles.block_k, tiles.block_k), kv_head)
            scratch.keys.load(index, (k_ref.at[tile],))
            scratch.values.load(index, (v_ref.at[tile],))

    # Each stage's last releases, observed so its barriers end where they began (`Ring.drain`).
    scratch.keys.drain(1, block.steps)
    scratch.values.drain(1, block.steps)


def _products(block: _Block, scratch: _Scratch) -> None:
    """The product warp: each step's `S = Q Kᵀ` into tensor memory once the softmax has read the last one,
    and a step behind, `P V` into the output's accumulator once its weights are there."""
    plgpu.barrier_wait(scratch.q_loaded)

    def values(index):
        (v_smem,), consumed = scratch.values.acquire(index)
        plgpu.barrier_wait(scratch.p_full)
        plgpu.tcgen05_mma(scratch.acc_tmem, scratch.p_tmem, v_smem, consumed, accumulate=index > 0)
        plgpu.tcgen05_commit_arrive(scratch.o_done)

    @pl.loop(0, block.blocks)
    def _(index):
        @pl.when(index < block.steps)
        def _():
            (k_smem,), consumed = scratch.keys.acquire(index)
            s_tmem = scratch.scores.acquire(index)
            plgpu.tcgen05_mma(s_tmem, scratch.q_smem, k_smem.T, consumed, accumulate=False)
            scratch.scores.issued(index)
            pl.when(index > 0)(functools.partial(values, index - 1))

    values(block.steps - 1)
    scratch.scores.drain(block.steps)


def _softmax(block: _Block, scratch: _Scratch, outs, *, spec: algebra.Forward, tiles: Tiles, blocks: int,
             dtype) -> None:
    """The reading warpgroup: each step's `S` through the algebra, the output's accumulator taken to the
    step's maximum once the last `P V` is in it, the narrowed weights into tensor memory; then the
    epilogue: the output, and each row's maximum and whether its weighted values overflowed
    (`algebra.walked_launches`' guards)."""
    out_ref, maxima_ref, over_ref = outs
    rows, block_k = TCGEN05.rows, tiles.block_k
    value_width = scratch.acc_tmem.shape[1]
    # A chunk of `CHUNK` columns of a step's tile at a time: the softmax's chains over one row's whole tile
    # spill (a whole 128-column tile's natural exponential, 7.9 KB a thread at sm100).
    q_pos = block.first + plgpu.broadcasted_iota(jnp.int32, (rows, CHUNK), 0, layout=TMEM)
    columns = plgpu.broadcasted_iota(jnp.int32, (rows, CHUNK), 1, layout=TMEM)
    chunks = block_k // CHUNK

    def logits(read, index, chunk, masked):
        coords = algebra.Coords(q_pos, index * block_k + chunk * CHUNK + columns)
        return algebra.logits_of(spec, read(chunk), coords, masked=masked)

    def step(index, carry, *, masked):
        m, total = carry
        read = scratch.scores.read(index, CHUNK, layout=TMEM)
        # The row's new maximum over the tile's chunks, and the factor the state before takes to it.
        new = m
        for chunk in range(chunks):
            new, _ = algebra.peak(spec, new, *logits(read, index, chunk, masked))
        m, alpha = algebra.peaked(spec, m, new)

        @pl.when(index > 0)
        def _():
            # The last step's P V is in the accumulator, its weights read: take the accumulator to this
            # step's maximum, a chunk of columns at a time, before this step's weights overwrite them.
            plgpu.barrier_wait(scratch.o_done)
            for start in range(0, value_width, CHUNK):
                chunk = mosaic_sm100.columns(scratch.acc_tmem, start, CHUNK)
                acc = plgpu.async_load_tmem(chunk, layout=TMEM)
                plgpu.async_store_tmem(chunk, algebra.rescale(acc, alpha))

        for chunk in range(chunks):
            weights = algebra.weights(spec, logits(read, index, chunk, masked)[0], m)
            total = (algebra.denominator_after(total, alpha, weights) if chunk == 0
                     else total + weights.sum(axis=-1))
            plgpu.async_store_tmem(mosaic_sm100.columns(scratch.p_tmem, chunk * CHUNK, CHUNK),
                                   algebra.narrow(weights, dtype))
        # The next S may overwrite this one once it is read out (`Tensor.release`).
        scratch.scores.release(index)
        plgpu.commit_tmem()
        plgpu.barrier_arrive(scratch.p_full)
        return m, total

    whole = block.steps
    padded = bool(spec.keys % block_k)
    if spec.form == "causal" or padded:
        # The blocks every query of the block sees whole, below its diagonal and before padding.
        whole = jnp.maximum(jnp.minimum(lax.div(block.first, block_k), blocks - padded), 0)
    start = (plgpu.layout_cast(jnp.full((rows,), -jnp.inf, jnp.float32), ROW),
             plgpu.layout_cast(jnp.zeros((rows,), jnp.float32), ROW))
    def walked(index, carry):
        # A block the query block walks, masked from its first that some query sees in part.
        return lax.cond(index < block.steps, lambda carry: step(index, carry, masked=index >= whole),
                        lambda carry: carry, carry)

    m, total = lax.fori_loop(0, blocks, walked, start)
    plgpu.barrier_wait(scratch.o_done)
    acc = plgpu.async_load_tmem(scratch.acc_tmem, layout=STORED)
    plgpu.wait_load_tmem()
    # The denominator to the epilogue's row layout through shared memory, which relayouts it.
    scratch.l_smem[...] = total
    total = plgpu.load(scratch.l_smem, layout=STORED.reduce(1))
    out, _, _ = algebra.finish(spec, algebra.Partial(acc, m, total), initial=False)
    scratch.o_smem[...] = out.astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(scratch.o_smem, out_ref.at[block.row, pl.ds(block.first, rows), block.head])
    places = (block.row, block.head, pl.ds(block.first, rows))
    # The rows' maxima, and the overflows of their sums, through the denominator's stage.
    for ref, row in ((maxima_ref, m), (over_ref, algebra.overflowed(acc).astype(jnp.float32))):
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        scratch.l_smem[...] = row
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(scratch.l_smem, ref.at[places])
    plgpu.wait_smem_to_gmem(0)


def _forward(query, key, value, *, spec: algebra.Forward, tiles: Tiles, interpret: bool):
    """The launch at `tiles`: the output `[B, Qp, H, Dv]` and each row's running maximum in the walk's units
    and whether its weighted values overflowed, `[B, H, Qp]` each (`algebra.walked_launches`' plain
    launch)."""
    batch, queries, heads, width = query.shape
    kv_heads, value_width = key.shape[2], value.shape[-1]
    group = heads // kv_heads
    rows, block_k, dtype = TCGEN05.rows, tiles.block_k, query.dtype
    # The key tiles holding the call's keys, none of padding alone.
    blocks = -(-spec.keys // block_k)
    target = generation.gpu_target("sm100")
    transforms = mosaic_sm100.swizzled(dtype, width)
    key_stages = mosaic_sm100.Stages(tiles.stages, (((block_k, width), dtype, transforms),))
    value_stages = mosaic_sm100.Stages(tiles.stages, (((block_k, value_width), dtype,
                                                       mosaic_sm100.swizzled(dtype, value_width)),))
    scores = mosaic_sm100.Accumulators(1, rows, block_k)

    def kernel(q_ref, k_ref, v_ref, out_ref, maxima_ref, over_ref, *refs):
        (q_smem, q_loaded, key_refs, value_refs, score_refs, p_tmem, p_full, acc_tmem, o_done, o_smem,
         l_smem) = refs
        scratch = _Scratch(q_smem, q_loaded, key_stages.opened(*key_refs), value_stages.opened(*value_refs),
                           scores.opened(*score_refs), p_tmem, p_full, acc_tmem, o_done, o_smem, l_smem)
        block = _Block(spec.form == "causal", blocks, block_k)
        TCGEN05.roles(
            functools.partial(_load, block, scratch, q_ref, k_ref, v_ref, group=group, tiles=tiles),
            functools.partial(_products, block, scratch),
            functools.partial(_softmax, block, scratch, (out_ref, maxima_ref, over_ref), spec=spec,
                              tiles=tiles, blocks=blocks, dtype=dtype))

    tensor_core = functools.partial(plgpu.Barrier, orders_tensor_core=True)
    scratch = (plgpu.SMEM((rows, width), dtype, transforms=transforms), plgpu.Barrier(),
               key_stages.scratch(), value_stages.scratch(), scores.scratch(target),
               plgpu.TMEM((rows, block_k), dtype, packed=True), tensor_core(),
               plgpu.TMEM((rows, value_width), jnp.float32), tensor_core(),
               plgpu.SMEM((rows, value_width), dtype, transforms=mosaic_sm100.swizzled(dtype, value_width)),
               plgpu.SMEM((rows,), jnp.float32))
    rows_type = jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32)
    out, maxima, over = TCGEN05.kernel(
        kernel, out_type=(jax.ShapeDtypeStruct((batch, queries, heads, value_width), dtype), rows_type,
                          rows_type),
        grid=(heads, queries // rows, batch), grid_names=("heads", "q_blocks", "batch"), scratch=scratch,
        interpret=plgpu.InterpretGPUParams() if interpret else None,
        # A name a form (the default's fold and its natural rewalk compile apart).
        name="attention_forward_tcgen05" + ("_natural" if spec.natural else ""),
    )(query, key, value)
    return out, maxima, over > 0
