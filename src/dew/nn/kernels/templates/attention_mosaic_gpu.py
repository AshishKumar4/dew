"""The attention template's Hopper mainloop (Mosaic GPU, sm90), the design's first sm90 rung
(docs/design/kernel-templates.md §6): FlashAttention-3's structure as jax's and Tokamax's Mosaic GPU
kernels hold it, with the template's algebra (`attention_algebra`) between the two products.

One program a head and a pair of query blocks: two consumer warpgroups, each a query block of
`Tiles.block_q` rows, and a producer warpgroup whose TMA loads fill a ring of `Tiles.stages` key and
value tiles, a key stage freed once both consumers' S = Q·Kᵀ is issued and a value stage once their
P·V is. The consumers take turns on the tensor cores (a schedule barrier between them), so one's
softmax runs under the other's products. Each step issues S, hands it to the algebra (`modify`, `peak`,
`weights`, `total`, `rescale`, `narrow`), and issues P·V on the narrowed weights; `finish` normalizes.
A causal call walks the tiles below its block's diagonal unmasked and the diagonal's masked, as the
Triton mainloop does; a consumer whose diagonal ends before its partner's keeps releasing the stages its
partner still reads.

The forward over full and causal calls, 16-bit operands, no hooks: the rung's first piece, held bitwise
to the other mainloops' arithmetic by the algebra and to the reference by the template's tests."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates.attention import NATURAL, Hooks, KeySpans, padded_call

WGMMA = plgpu.Layout.WGMMA
ROW = plgpu.Layout.WGMMA.reduce(1)
"""A tile's register layout under wgmma, and a per-row vector's in it."""
CONSUMERS = 2
"""The consumer warpgroups of a program, each a query block (FlashAttention-3's two)."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """A launch's tiles: each consumer's query block (`block_q`, a multiple of 64, wgmma's row unit), the
    key tile (`block_k`) and the key and value tiles in flight (`stages`)."""

    block_q: int = 64
    block_k: int = 128
    stages: int = 2

    def __post_init__(self):
        if self.block_q % 64 or self.block_k % 64 or self.stages < 2:
            raise ValueError(f"{self}: query and key tiles of whole 64 rows, two stages or more")


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, tiles: Tiles = Tiles(),
            interpret: bool = False) -> jax.Array:
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and the keys `spans` leaves each query (full or causal), as
    `dew.nn.kernels.templates.attention.reference` computes it. `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic GPU's interpret mode)."""
    if hooks is not None and hooks != Hooks():
        raise NotImplementedError("the sm90 mainloop takes no hooks yet")
    if spans.form not in ("full", "causal"):
        raise NotImplementedError(f"the sm90 mainloop takes full and causal calls, not {spans.form!r} ones")
    if jnp.dtype(query.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        raise NotImplementedError(f"the sm90 mainloop takes 16-bit operands, not {jnp.dtype(query.dtype)}")
    block, value_width = max(CONSUMERS * tiles.block_q, tiles.block_k), value.shape[-1]
    call = padded_call(query, key, value, spans, Hooks(), block)
    spec = algebra.Forward(spans.form, spans.keys, ids=False,
                           scale=query.shape[3] ** -0.5 if scale is None else scale, natural=natural)
    # wgmma's swizzled tiles hold whole 64-element rows of 16-bit operands.
    query, key, value = (jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % 64),))
                         for array in (call.query, call.key, call.value))
    out = _forward(query, key, value, spec=spec, tiles=tiles, interpret=interpret)
    return out[:, :call.queries, :, :value_width]


def _accumulated(acc: jax.Array, lhs: jax.Array, rhs, then, interpret: bool) -> jax.Array:
    """`acc` plus `lhs` (registers) times `rhs` (shared memory), wgmma accumulating in place, and `then`
    issued while it runs. Mosaic GPU's interpret mode (jax 0.11.2) takes neither a register operand nor
    `pl.run_state`, so interpreted it is the dot the interpreter's wgmma computes, added."""
    if interpret:
        then()
        return acc + jnp.dot(lhs, rhs[...], preferred_element_type=jnp.float32)

    def product(acc_ref):
        plgpu.wgmma(acc_ref, lhs, rhs)
        then()

    return pl.run_state(product)(plgpu.ACC.init(acc))


def _forward(query, key, value, *, spec: algebra.Forward, tiles: Tiles, interpret: bool):
    batch, queries, heads, width = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    block_q, block_k, stages = tiles.block_q, tiles.block_k, tiles.stages
    pair = CONSUMERS * block_q
    steps = keys // block_k
    causal = spec.form == "causal"
    dtype = query.dtype

    def program_steps(first):
        # The key tiles a query block from `first` reads: every one, or those up to its diagonal.
        if not causal:
            return steps
        return jnp.minimum(pl.cdiv(first + block_q, block_k), steps)

    def kernel(q_ref, k_ref, v_ref, out_ref, scoped):
        (q_smem, k_smem, v_smem), (k_full, v_full, q_full), (k_free, v_free), schedule = scoped
        row, head, wg = lax.axis_index("batch"), lax.axis_index("heads"), lax.axis_index("wg")
        pair_first = lax.axis_index("q_tiles") * pair
        most = program_steps(pair_first + pair - block_q)

        def turn():
            plgpu.barrier_arrive(schedule)
            plgpu.barrier_wait(schedule)

        @pl.when(wg < CONSUMERS)
        def consumer():
            plgpu.set_max_registers(232, action="increase")
            first = pair_first + wg * block_q
            mine = program_steps(first)
            plgpu.copy_gmem_to_smem(q_ref.at[row, pl.ds(first, block_q), head], q_smem.at[wg], q_full.at[wg])
            plgpu.barrier_wait(q_full.at[wg])
            q_pos = first + plgpu.broadcasted_iota(jnp.int32, (block_q, block_k), 0, layout=WGMMA)
            columns = plgpu.broadcasted_iota(jnp.int32, (block_q, block_k), 1, layout=WGMMA)
            state = algebra.Partial(
                plgpu.layout_cast(jnp.zeros((block_q, value_width), jnp.float32), WGMMA),
                plgpu.layout_cast(jnp.full((block_q,), -jnp.inf, jnp.float32), ROW),
                plgpu.layout_cast(jnp.zeros((block_q,), jnp.float32), ROW))

            @pl.when(mine > 0)
            def _():
                plgpu.barrier_wait(k_full.at[0])

            pl.when(wg == 1)(turn)

            def step(index, state: algebra.Partial, *, masked: bool) -> algebra.Partial:
                slot = lax.rem(index, stages)

                def scores(acc_ref):
                    plgpu.wgmma(acc_ref, q_smem.at[wg], k_smem.at[slot].transpose((1, 0)))
                    turn()
                    return acc_ref[...]

                s = pl.run_scoped(scores, plgpu.ACC((block_q, block_k), jnp.float32))
                plgpu.barrier_arrive(k_free.at[slot])
                coords = algebra.Coords(q_pos, index * block_k + columns)
                logits = algebra.modify(spec, s, coords, masked=masked)
                m, alpha = algebra.peak(spec, state.m, logits)
                weights = algebra.weights(spec, logits, m)
                denominator = algebra.total(state.total, alpha, weights)
                acc = algebra.rescale(state.acc, alpha)
                narrowed = algebra.narrow(weights, dtype)
                plgpu.barrier_arrive(schedule)
                plgpu.barrier_wait(v_full.at[slot])
                plgpu.barrier_wait(schedule)

                def prefetched():
                    following = index + 1

                    @pl.when(following < mine)
                    def _():
                        plgpu.barrier_wait(k_full.at[lax.rem(following, stages)])

                acc = _accumulated(acc, narrowed, v_smem.at[slot], prefetched, interpret)
                plgpu.barrier_arrive(v_free.at[slot])
                return algebra.Partial(acc, m, denominator)

            whole = mine
            if causal or spec.keys % block_k:
                # The tiles every query of the block sees whole, below its diagonal and before padding.
                whole = jnp.minimum(lax.div(first, block_k), keys // block_k - bool(spec.keys % block_k))
                whole = jnp.maximum(whole, 0)
            state = lax.fori_loop(0, whole, functools.partial(step, masked=False), state)
            state = lax.fori_loop(whole, mine, functools.partial(step, masked=True), state)

            def released(index, carry):
                # The stages a consumer whose diagonal ends first leaves its partner, freed as read.
                slot = lax.rem(index, stages)
                plgpu.barrier_arrive(k_free.at[slot])
                plgpu.barrier_arrive(v_free.at[slot])
                turn()
                turn()
                return carry

            lax.fori_loop(mine, most, released, None)
            pl.when(wg == 0)(turn)
            out, _, _ = algebra.finish(spec, state, initial=False)
            q_smem.at[wg][...] = out.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem.at[wg], out_ref.at[row, pl.ds(first, block_q), head])
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == CONSUMERS)
        def producer():
            plgpu.set_max_registers(40, action="decrease")
            kv_head = lax.div(head, group)

            def load(index, slot):
                tile = (row, pl.ds(index * block_k, block_k), kv_head)
                plgpu.copy_gmem_to_smem(k_ref.at[tile], k_smem.at[slot], k_full.at[slot])
                plgpu.copy_gmem_to_smem(v_ref.at[tile], v_smem.at[slot], v_full.at[slot])

            for slot in range(stages):
                pl.when(jnp.int32(slot) < most)(functools.partial(load, slot, slot))

            @pl.loop(stages, most)
            def _(index):
                slot = lax.rem(index, stages)
                plgpu.barrier_wait(k_free.at[slot])
                tile = (row, pl.ds(index * block_k, block_k), kv_head)
                plgpu.copy_gmem_to_smem(k_ref.at[tile], k_smem.at[slot], k_full.at[slot])
                plgpu.barrier_wait(v_free.at[slot])
                plgpu.copy_gmem_to_smem(v_ref.at[tile], v_smem.at[slot], v_full.at[slot])

    def entry(q_ref, k_ref, v_ref, out_ref):
        transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
        buffers = (plgpu.SMEM((CONSUMERS, block_q, width), dtype, transforms=transforms),
                   plgpu.SMEM((stages, block_k, width), dtype, transforms=transforms),
                   plgpu.SMEM((stages, block_k, value_width), dtype, transforms=transforms))
        loaded = (plgpu.Barrier(num_barriers=stages), plgpu.Barrier(num_barriers=stages),
                  plgpu.Barrier(num_barriers=CONSUMERS))
        freed = (plgpu.Barrier(num_arrivals=CONSUMERS, num_barriers=stages),) * 2
        pl.run_scoped(lambda *scoped: kernel(q_ref, k_ref, v_ref, out_ref, scoped), buffers, loaded, freed,
                      plgpu.Barrier(num_arrivals=CONSUMERS), collective_axes="wg")

    return plgpu.kernel(
        entry, out_type=jax.ShapeDtypeStruct((batch, queries, heads, value_width), dtype),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"),
        num_threads=CONSUMERS + 1, thread_name="wg", compiler_params=plgpu.CompilerParams(approx_math=True),
        interpret=plgpu.InterpretGPUParams() if interpret else None,
    )(query, key, value)
