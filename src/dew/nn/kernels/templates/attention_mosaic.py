"""The attention template's Mosaic GPU mainloop on the wgmma family (Hopper, sm90; design §6):
FlashAttention-3's structure, built from the shared toolkit (`mosaic`: the roles, the operand rings) and
the family's issue (`mosaic_sm90`), with the template's algebra (`attention_algebra`) between the products.

One program a query head and a pair of query blocks: two consumer warpgroups, each a block of `Tiles.block_q`
rows, and a producer warpgroup streaming the key and value tiles through a ring each (`mosaic.Ring`), every
slot released by both consumers. The consumers take turns on the tensor cores (a schedule barrier between
them), so one's softmax runs under the other's products. A step issues S = Q·Kᵀ, releases the key slot, and
hands S to the algebra as a tile (`algebra.Keys`) whose values are the value slot and whose product
accumulates the narrowed weights from registers into the running sums in place; while it runs the consumer
waits for its next key tile. A causal block masks only the tiles past its diagonal's start, and a consumer
whose diagonal ends before its partner's keeps releasing the slots its partner still reads.

The forward over full and causal calls of 16-bit operands, the base-2 fold and the natural form without
headroom (fp16 values, which cannot overflow it), no hooks."""

import dataclasses

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import attention_algebra as algebra, mosaic, mosaic_sm90
from dew.nn.kernels.templates.attention import NATURAL, Hooks, KeySpans, padded_call

CONSUMERS = 2
"""The consumer warpgroups of a program, each a query block (FlashAttention-3's two)."""
FAMILY = mosaic_sm90.FAMILY


@dataclasses.dataclass(frozen=True, order=True)
class Tiles:
    """A launch's tiles: each consumer's query block (`block_q`, whole warpgroup rows of the family), the key
    tile (`block_k`) and the key and value tiles in flight (`stages`)."""

    block_q: int
    block_k: int
    stages: int

    def __post_init__(self):
        if self.block_q % FAMILY.rows or self.block_k % FAMILY.rows or self.stages < 2:
            raise ValueError(f"{self}: query and key tiles of whole {FAMILY.rows} rows, two stages or more")


DEFAULT = Tiles(64, 128, 2)
"""The forward's tiles: FlashAttention-3's at width 128, a warpgroup's 64 queries over 128 keys in two stages,
the fastest of 64 or 128 queries over 64 or 128 keys in two to four stages on an H100 (llama-d128 causal bf16,
2 x 4096: 0.65 ms; 128 queries a consumer spill, 2.9 to 4.6 ms)."""


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, tiles: Tiles = DEFAULT,
            interpret: bool = False) -> jax.Array:
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]`
    and the keys `spans` leaves each query (full or causal), as `dew.nn.kernels.templates.attention.reference`
    computes it. `natural` keeps the logits natural (`algebra.Forward.base2`); `interpret` runs it on the CPU
    (Mosaic GPU's interpret mode)."""
    if hooks is not None and hooks != Hooks():
        raise NotImplementedError("the sm90 mainloop takes no hooks yet")
    if spans.form not in ("full", "causal"):
        raise NotImplementedError(f"the sm90 mainloop takes full and causal calls, not {spans.form!r} ones")
    if jnp.dtype(query.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        raise NotImplementedError(f"the sm90 mainloop takes 16-bit operands, not {jnp.dtype(query.dtype)}")
    width, value_width = query.shape[3], value.shape[3]
    if width != value_width:
        raise NotImplementedError("the sm90 mainloop's output reuses the query's stage: equal head widths")
    call = padded_call(query, key, value, spans, Hooks(), max(CONSUMERS * tiles.block_q, tiles.block_k))
    spec = algebra.Forward(spans.form, spans.keys, ids=False, scale=width ** -0.5 if scale is None else scale,
                           natural=natural)
    if spec.natural and algebra.headroom(value.dtype):
        raise NotImplementedError("the sm90 mainloop's natural form takes no headroom guard yet")
    # The family's swizzled tiles hold whole rows of `FAMILY.rows` 16-bit entries.
    query, key, value = (jnp.pad(array, ((0, 0),) * 3 + ((0, -array.shape[3] % FAMILY.rows),))
                         for array in (call.query, call.key, call.value))
    out = _launch(query, key, value, spec=spec, tiles=tiles, interpret=interpret)
    return out[:, :call.queries, :, :value_width]


@dataclasses.dataclass(frozen=True)
class _Program:
    """A program's static geometry: its tiles, the call's key tiles (`steps`), the slots a ring holds, its key
    count and whether it is causal."""

    tiles: Tiles
    steps: int
    stages: int
    keys: int
    causal: bool

    def walked(self, first):
        """The key tiles a query block from `first` reads: every one, or those up to its diagonal."""
        if not self.causal:
            return jnp.int32(self.steps)
        return jnp.minimum(pl.cdiv(first + self.tiles.block_q, self.tiles.block_k), self.steps)

    def whole(self, first, walked):
        """The tiles every query of the block from `first` sees whole: below its diagonal, before padding."""
        if not self.causal and not self.keys % self.tiles.block_k:
            return walked
        whole = jnp.minimum(lax.div(first, self.tiles.block_k),
                            self.steps - bool(self.keys % self.tiles.block_k))
        return jnp.maximum(whole if self.causal else jnp.int32(self.steps - 1), 0)


def _consumer(program: _Program, spec: algebra.Forward, wg, q_view, out_view, q_smem, q_loaded, keys, values,
              turn, schedule, most, first, *, interpret: bool):
    """A consumer warpgroup's query block: its walk of the key tiles through the rings (`algebra.plain_walk`),
    then its output stored through its query stage."""
    tiles = program.tiles
    shape = (tiles.block_q, tiles.block_k)
    walked = program.walked(first)
    whole = program.whole(first, walked)
    plgpu.copy_gmem_to_smem(q_view, q_smem, q_loaded)
    plgpu.barrier_wait(q_loaded)
    q_pos, columns = first + mosaic_sm90.iota(shape, 0), mosaic_sm90.iota(shape, 1)
    width = q_smem.shape[-1]
    state = algebra.Partial(mosaic_sm90.zeros((tiles.block_q, width)),
                            plgpu.layout_cast(jnp.full((tiles.block_q,), -jnp.inf, jnp.float32), FAMILY.row),
                            mosaic_sm90.zeros((tiles.block_q,)))
    def awaited(index):
        @pl.when(index < walked)
        def _():
            keys.acquire(index)

    awaited(jnp.int32(0))
    pl.when(wg == 1)(turn)

    def keys_at(index) -> algebra.Keys:
        s = mosaic_sm90.product(q_smem, keys.at(index).transpose((1, 0)), shape, turn)
        keys.release(index)
        coords = algebra.Coords(q_pos, index * tiles.block_k + columns)
        # A tile every query sees whole takes no mask: one walk, the choice a tile's (a walk of each, these
        # rings' waits and arrivals in two loops, faulted on an H100 under jax 0.11.2's lowering).
        logits = lax.cond(index < whole, lambda: algebra.modify(spec, s, coords, masked=False),
                          lambda: algebra.modify(spec, s, coords, masked=True))

        def load():
            # The value slot, once the partner's products are issued: its turn on the tensor cores is over
            # when this one's come.
            plgpu.barrier_arrive(schedule)
            slot = values.acquire(index)
            plgpu.barrier_wait(schedule)
            return slot

        def accumulate(acc, lhs, rhs):
            acc = mosaic_sm90.accumulate(acc, lhs, rhs, interpret)
            values.release(index)
            # The next key tile's wait, before its scores need it.
            awaited(index + 1)
            return acc

        def dot(lhs, rhs):
            return accumulate(mosaic_sm90.zeros((lhs.shape[0], rhs.shape[1])), lhs, rhs)

        return algebra.Keys(coords, logits, load, dot, accumulate)

    state = algebra.plain_walk(spec, 0, walked, keys_at, state)

    def released(index, carry):
        # The slots a consumer whose diagonal ends first leaves its partner, released as it would have.
        keys.release(index)
        values.release(index)
        turn()
        turn()
        return carry

    lax.fori_loop(walked, most, released, None)
    pl.when(wg == 0)(turn)
    out, _, _ = algebra.finish(spec, state, initial=False)
    q_smem[...] = out.astype(q_smem.dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(q_smem, out_view)
    plgpu.wait_smem_to_gmem(0)


def _launch(query, key, value, *, spec: algebra.Forward, tiles: Tiles, interpret: bool):
    batch, queries, heads, width = query.shape
    keys_count, kv_heads = key.shape[1], key.shape[2]
    group = heads // kv_heads
    block_q, block_k = tiles.block_q, tiles.block_k
    pair = CONSUMERS * block_q
    steps = keys_count // block_k
    # No more stages than key tiles: a slot past them would load past the keys.
    program = _Program(tiles, steps, min(tiles.stages, steps), spec.keys, spec.form == "causal")

    def body(q_ref, k_ref, v_ref, out_ref, q_smem, q_loaded, schedule, key_ring, value_ring):
        keys, values = mosaic.Ring(**key_ring), mosaic.Ring(**value_ring)
        row, head, wg = lax.axis_index("batch"), lax.axis_index("heads"), lax.axis_index("wg")
        pair_first = lax.axis_index("q_tiles") * pair
        most = program.walked(pair_first + pair - block_q)

        def turn():
            plgpu.barrier_arrive(schedule)
            plgpu.barrier_wait(schedule)

        def consumer(wg):
            first = pair_first + wg * block_q
            rows = (row, pl.ds(first, block_q), head)
            _consumer(program, spec, wg, q_ref.at[rows], out_ref.at[rows], q_smem.at[wg], q_loaded.at[wg],
                      keys, values, turn, schedule, most, first, interpret=interpret)

        def producer():
            kv_head = lax.div(head, group)

            @pl.loop(0, most)
            def _(index):
                tile = (row, pl.ds(index * block_k, block_k), kv_head)
                keys.load(index, k_ref.at[tile])
                values.load(index, v_ref.at[tile])

        mosaic.roles(wg, CONSUMERS, consumer, producer)

    def entry(q_ref, k_ref, v_ref, out_ref):
        pl.run_scoped(
            lambda *scoped, **rings: body(q_ref, k_ref, v_ref, out_ref, *scoped, **rings),
            plgpu.SMEM((CONSUMERS, block_q, width), query.dtype, transforms=FAMILY.transforms),
            plgpu.Barrier(num_barriers=CONSUMERS), plgpu.Barrier(num_arrivals=CONSUMERS),
            key_ring=mosaic.ring((block_k, width), key.dtype, program.stages, consumers=CONSUMERS,
                                 transforms=FAMILY.transforms),
            value_ring=mosaic.ring((block_k, value.shape[-1]), value.dtype, program.stages,
                                   consumers=CONSUMERS, transforms=FAMILY.transforms),
            collective_axes="wg")

    return plgpu.kernel(
        entry, out_type=jax.ShapeDtypeStruct((batch, queries, heads, value.shape[-1]), query.dtype),
        grid=(heads, queries // pair, batch), grid_names=("heads", "q_tiles", "batch"),
        num_threads=CONSUMERS + 1, thread_name="wg", compiler_params=plgpu.CompilerParams(approx_math=True),
        interpret=plgpu.InterpretGPUParams() if interpret else None,
    )(query, key, value)
