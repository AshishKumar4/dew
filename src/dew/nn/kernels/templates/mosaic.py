"""The templates' Pallas Mosaic GPU toolkit (design §6), every instruction family's: the launch,
warpgroup roles, barriers, shared-memory staging, operand streams and stores every Mosaic mainloop builds on.
The math is each template's algebra, written once; what an instruction family adds is its issue and its
layout facts alone (`Family`: mosaic_sm90's wgmma; mma.sync's and tcgen05's to come). Each template keeps
its own mainloop, product dependencies and launch DAG; only these primitives are shared.

A launch takes each warpgroup's role (`roles`: a producer issuing the loads beside consumers computing where
the family's roles split the register file, `Family.consumers`), persistent over the multiprocessors, a pair
of them a cluster where they share one operand's blocks (multicast: `ring`'s, `Family.streams`'). The
consumers meet (`meet`) where one consumer's writes to shared memory are the other's operand, and a tile's
rows reach global memory through `store_rows`, whose stores have static extents however many rows a tile
holds. What a program asks of a multiprocessor, and the derivation's shared steps, are `resources`'.

A mainloop that drives its own loads (the GEMM's kernel gradient, its stages streaming across tiles)
streams each operand through a `Ring` whose counter runs across tiles, and takes each warpgroup's role
(`roles`).
"""

import abc
import dataclasses
import functools
import math
from collections.abc import Callable

import jax
from jax import lax, numpy as jnp
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import product, resources
from dew.nn.sharding import per_shard

Ref = state_types.AbstractRef | state_types.TransformedRef
Operand = Ref | jax.Array
"""A product's operand: a tile in shared memory (a ref) or in registers (an array)."""


def compiler_params(*, approx_math: bool = False, reduction_scratch: int = 0) -> plgpu.CompilerParams:
    """A mainloop's compiler parameters: Mosaic GPU's warpgroup lowering, whose arrays are a warpgroup's,
    and with `approx_math` the hardware's approximate exponential and reciprocal (ex2.approx, as Triton's
    exp2 and FlashAttention's are). A block's warpgroups reduce at once, so each takes its own scratch: a
    consumer's planned reductions its share of the kernel's allocations (`Family.column_sums`), the
    implicit ones its arithmetic forms (a unit row a hook slices) an equal share of `reduction_scratch`
    bytes, which Mosaic GPU splits between the block's warpgroups; with none, such a reduction fails to
    lower rather than share one."""
    return plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup, approx_math=approx_math,
                                reduction_scratch_bytes=reduction_scratch)


CLUSTER = "cluster"
"""The axis a launch's cluster of programs runs along (`Family.kernel`'s `cluster`)."""


def cluster_index(cluster: int) -> jax.Array | int:
    """This program's index among its cluster's `cluster` programs (`CLUSTER`): 0 alone."""
    return lax.axis_index(CLUSTER) if cluster > 1 else 0


def meet(halves) -> None:
    """The two consumer warpgroups wait for each other at `halves`, two barriers of two arrivals, in turn:
    neither arrives at a barrier's next phase before the other has seen its last, however the meets fall
    (the interpreter's invariant, and the hardware's parity waits)."""
    for barrier in (halves.at[0], halves.at[1]):
        plgpu.barrier_arrive(barrier)
        plgpu.barrier_wait(barrier)


def meeting(consumers: int):
    """The barriers two consumer warpgroups `meet` at: two, of `consumers` arrivals each."""
    return plgpu.Barrier(num_arrivals=consumers, num_barriers=2)


def staging(name: str, shaped) -> dict:
    """Each consumer warpgroup's staging buffer for each operand a tile reads (`shaped`: its tile's shape and
    its operand) and a barrier for each, as `{name}_smems` and `{name}_barriers`."""
    shaped = list(shaped)
    return {f'{name}_smems': [plgpu.SMEM((2, *shape), gmem.dtype) for shape, gmem in shaped],
            f'{name}_barriers': [plgpu.Barrier(num_barriers=2) for _ in shaped]}


STAGED_BYTES = 128
"""The widest row, in bytes, a consumer stages an output in at a time: a swizzle's row. A consumer's 256
bf16 columns written to shared memory as one 512-byte row conflicted on its banks 195 times as often as in
two 256-byte halves (ncu, Qwen3's down projection forward: 18.3M conflicts against 94K, the kernel 16%
slower); 128-byte chunks, two of them buffered (`output_staging`), stage a 256-column bf16 tile in half its
width, which leaves the ring a stage more (unmeasured for conflicts at 128 bytes)."""

OUTPUT_BUFFERS = 2
"""The chunk buffers a consumer stages an output tile through: one written while the last one's store
reads it."""


def staged_columns(columns: int, dtypes) -> int:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in: the most that divide the
    tile within `STAGED_BYTES` of the widest output (a 192-column tile's bf16 output in chunks of 64)."""
    return math.gcd(columns, STAGED_BYTES // max(jnp.dtype(dtype).itemsize for dtype in dtypes))


def output_staging(columns: int, dtypes) -> tuple[int, int]:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in (`staged_columns`) and
    the chunk buffers it holds (`OUTPUT_BUFFERS`, at most its chunks): the tile's output staged in two
    chunks' shared memory rather than its whole width, CUTLASS's sub-tiled epilogue, which leaves the
    mainloop's ring the rest of the block (its StageCountAutoCarveout)."""
    chunk = staged_columns(columns, dtypes)
    return chunk, min(columns // chunk, OUTPUT_BUFFERS)


def interpreted(interpret) -> plgpu.InterpretGPUParams | None:
    """A launch's interpreter: `interpret`'s own parameters where it gives them (a read past an operand
    poisoned, `grouped_matmul.INTERPRETED`), the default where it is True, none where False."""
    if isinstance(interpret, plgpu.InterpretGPUParams):
        return interpret
    return plgpu.InterpretGPUParams() if interpret else None


def unwalked(family: "Family") -> Callable:
    """A walk of no steps (`Family.persist`'s, where a tile streams nothing): its carry and accumulators as
    they start."""
    def walk(step, start, shapes, *, initial=None):
        del step
        initial = (None,) * len(shapes) if initial is None else initial
        return start, tuple(family.zeros(shape) if value is None else value
                            for shape, value in zip(shapes, initial, strict=True))

    return walk


def viewed(view: tuple, block: tuple, step):
    """Step `step`'s tile of a stream's view `(ref, index_map)` (`Family.opened`): the `block`-shaped tile of
    `ref` at the block indices `index_map(step)` gives, as a `BlockSpec`'s index map does."""
    ref, index_map = view
    return ref.at[tuple(pl.ds(index * extent, extent)
                        for index, extent in zip(index_map(step), block, strict=True))]


def streamed(rings: tuple, views, lo, hi) -> None:
    """A producer's loads of steps `lo` to `hi` into `rings`, each ring's step counted from `lo` and loaded
    from its view of `views(step)`, then each ring drained."""
    @pl.loop(lo, hi)
    def _(index):
        for ring, view in zip(rings, views(index), strict=True):
            ring.load(index - lo, view)

    for ring in rings:
        ring.drain(hi - lo)


def passed(rings: tuple, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
    """A consumer's steps `lo` to `hi` of `rings` it reads none of (another consumer's), each ring's load
    awaited and its slot released as a reading consumer's are (counted from `first`): a wait takes its
    barrier's next phase, so every consumer waits on every load; `between()` after each step."""
    @pl.loop(lo, hi)
    def _(index):
        for ring in rings:
            ring.acquire(index - first)
            ring.release(index - first)
        if between is not None:
            between()


def loaded(copies) -> None:
    """Each `(view, ref, barrier)` copied whole into shared memory, every copy issued before any is
    awaited."""
    for view, ref, barrier in copies:
        plgpu.copy_gmem_to_smem(view, ref, barrier)
    for _, _, barrier in copies:
        plgpu.barrier_wait(barrier)


def stored(parts, views, *, smem, wait: bool = True) -> None:
    """Each of `parts` written whole to its view of `views` through the shared buffer `smem`, in turn, the
    buffer reused once its copy has read it; the last copy waited for unless `wait` is false, when it
    stays in flight, one store group (`Family.store_wait`)."""
    last = len(parts) - 1
    for index, (part, view) in enumerate(zip(parts, views, strict=True)):
        smem[...] = part.astype(smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(smem, view)
        if index < last:
            # The next part writes `smem`: this copy's reads of it done first.
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        elif wait:
            plgpu.wait_smem_to_gmem(0)


def store_rows(o_smem, o_gmem, row_start, first, stored, column, most: int, least: int = 1,
               copy: Callable = plgpu.copy_smem_to_gmem, widest: int | None = None) -> None:
    """Rows `first` to `first + stored` of `o_smem` to `o_gmem` at row `row_start + first` and `column`, in a
    power-of-two ladder of static-size `copy`s (a TMA store's descriptor has static extents; `copied`
    moves the rows through registers where a GPU has no TMA), the largest `most` rows', the smallest
    `least` (`stored` a multiple of it: a rung below it is never taken, and none is emitted); with
    `widest`, in copies of at most `widest` rows, whole ones first (a copy through registers holds its
    rows: `resources.copied_rows`)."""
    start, size = first, most
    if widest is not None and widest < most:
        for whole in range(most // widest):
            @pl.when(stored // widest > whole)
            def _(whole=whole):
                copy(o_smem.at[pl.ds(first + whole * widest, widest)],
                     o_gmem.at[pl.ds(row_start + first + whole * widest, widest),
                               pl.ds(column, o_smem.shape[1])])

        start, size = first + stored // widest * widest, widest - 1
    while size >= least:
        rung = 1 << int(math.log2(size))
        size //= 2

        @pl.when(stored & rung != 0)
        def _(start=start, rung=rung):
            copy(o_smem.at[pl.ds(start, rung)],
                 o_gmem.at[pl.ds(row_start + start, rung), pl.ds(column, o_smem.shape[1])])

        start = start + (stored & rung)


def copied(source, destination) -> None:
    """`source` (in shared memory) to `destination` (in global memory) through registers: a GPU without
    TMA's `store_rows` copy, explicitly unoptimized. JAX 0.12 takes a transfer with no `optimized` flag as
    optimized-only (0.11 downgraded it), which a ladder rung of a row or two cannot be, so its layout
    inference searched every rung's layouts to exhaustion (ns-gate1: the sm80 forward failed after about
    1000 s; without the ladder, or with these transfers unoptimized, it lowers in seconds:
    ns-noladder2, ns-abort1)."""
    plgpu.store(destination, plgpu.load(source, optimized=False), optimized=False)


@dataclasses.dataclass(frozen=True)
class Ring:
    """A ring of `stages` shared-memory slots one operand streams through: a producer loads step `counter`
    into slot `counter % stages` (`load`), a consumer waits for it (`acquire`) and gives it back
    (`release`). The counter runs across a persistent kernel's tiles, so a tile's first loads reuse the
    slots the last tile left, and each tile may take its own number of steps (a causal tile's trip range):
    a slot is free for its first fill and, after, once every consumer released it; the producer drains the
    ring after its last load (`drain`). Allocated with `ring`, its slot count the allocation's own.

    A ring a cluster's programs share (`ring`'s `multicast`) takes each step's block once between them:
    every program's producer loads the same view, its share of the block multicast into each program's slot,
    and each program's load barrier awaits the whole block. A release arrives at each program's barrier, so
    a slot is free once every consumer of every program has released it (CUTLASS's PipelineTmaAsync), and
    the producer's drain holds its program until the others' last releases have reached it."""
    smem: Ref
    full: state_types.AbstractRef
    consumed: state_types.AbstractRef

    @property
    def stages(self) -> int:
        """The ring's slots: its allocation's leading dimension."""
        return int(self.smem.shape[0])

    @property
    def multicast(self) -> tuple | None:
        """The cluster axes the ring's loads are multicast along: its release barrier's, a cluster's (`ring`);
        None where the ring is its program's own."""
        dtype = self.consumed.dtype
        return dtype.collective_axes if isinstance(dtype, plgpu_core.ClusterBarrierType) else None

    def slot(self, counter):
        """The slot step `counter` (traced, or a Python int) uses."""
        return counter % self.stages

    def load(self, counter, view) -> None:
        """The producer's load of step `counter` from `view`, once its slot is free."""
        slot = self.slot(counter)
        pl.when(counter >= self.stages)(lambda: plgpu.barrier_wait(self.consumed.at[slot]))
        plgpu.copy_gmem_to_smem(view, self.smem.at[slot], self.full.at[slot], collective_axes=self.multicast)

    def acquire(self, counter):
        """A consumer's wait for step `counter`'s load: its slot. Each load is waited for once (a wait takes
        the barrier's next phase)."""
        slot = self.slot(counter)
        plgpu.barrier_wait(self.full.at[slot])
        return self.smem.at[slot]

    def at(self, counter):
        """Step `counter`'s slot, its load waited for already (`acquire`, issued under an earlier step's
        products)."""
        return self.smem.at[self.slot(counter)]

    def release(self, counter) -> None:
        """A consumer's release of step `counter`'s slot, whose reads (and MMAs) are done."""
        plgpu.barrier_arrive(self.consumed.at[self.slot(counter)])

    def drain(self, end) -> None:
        """The producer's wait, after its last load (step `end - 1`), for the consumers' release of each slot
        it filled last, so no slot is freed (or reallocated) while a consumer still reads it, and every
        release is seen."""
        for back in range(self.stages):
            step = end - 1 - back
            pl.when(step >= 0)(lambda step=step: plgpu.barrier_wait(self.consumed.at[self.slot(step)]))


def ring(block: tuple, dtype, stages: int, *, consumers: int = 1, multicast: bool = False,
         swizzle: int | None = None) -> dict:
    """A `Ring`'s allocations for `pl.run_scoped`: `stages` slots of `block` in `dtype` (their rows staged
    at `swizzle`, `staged`), a load barrier each, and a release barrier each that `consumers` arrive at,
    with `multicast` the `consumers` of each program of the cluster (`CLUSTER`), whose programs share the
    ring's blocks. `Ring(**allocated)` once scoped."""
    consumed = (plgpu.ClusterBarrier(collective_axes=(CLUSTER,), num_arrivals=consumers, num_barriers=stages)
                if multicast else plgpu.Barrier(num_arrivals=consumers, num_barriers=stages))
    return {'smem': staged((stages, *block), dtype, swizzle), 'full': plgpu.Barrier(num_barriers=stages),
            'consumed': consumed}


SWIZZLE_BYTES = 128
"""The widest swizzle a shared-memory block's rows are staged at, which an MMA reads conflict-free."""

NARROWEST_SWIZZLE_BYTES = 32
"""The narrowest swizzle Mosaic GPU stages a block's rows at (`plgpu.find_swizzle`'s least)."""


def swizzle_for(dtype, *contractions: int) -> int:
    """The widest swizzle, in bytes, a shared-memory block of `dtype` takes whose group of elements (the
    swizzle over the itemsize: Mosaic's wgmma issues a product's contraction in groups of it, one swizzle
    for both its shared operands, wgmma.py's `k_group_elems`) divides each of `contractions`, the elements
    the products it takes part in contract over a step. A product contracting over 32 rows of 16-bit
    operands so stages them at 64 bytes, at which its partner block must stage too (K32SWZ3: S = K Qᵀ
    over a 128-wide head lowers with both at 64 and both at 128, not mixed; dV += P dO over 32 rows lowers
    at 64, not 128)."""
    size = jnp.dtype(dtype).itemsize
    return next((swizzle for swizzle in (SWIZZLE_BYTES, SWIZZLE_BYTES // 2, NARROWEST_SWIZZLE_BYTES)
                 if all(contraction % (swizzle // size) == 0 for contraction in contractions)),
                NARROWEST_SWIZZLE_BYTES)


TILING: Callable[[tuple[int, int]], object] = plgpu.TilingTransform
SWIZZLE: Callable[[int], object] = plgpu.SwizzleTransform
"""Mosaic GPU's tile transforms, which it declares as protocol classes (a type checker takes them for
uninstantiable)."""


def swizzle_transforms(dtype, swizzle: int) -> tuple:
    """A shared-memory tile's transforms at `swizzle` bytes: 8-row tiles a swizzle's row wide, swizzled."""
    return TILING((8, swizzle // jnp.dtype(dtype).itemsize)), SWIZZLE(swizzle)


def staged(shape: tuple, dtype, swizzle: int | None = None):
    """A shared-memory block of `shape` whose rows are staged at `swizzle` bytes: as layout inference stages
    them where unstated (the widest that fits); otherwise in 8-row tiles of the swizzle's row, explicitly,
    which inference does not take for a block whose products contract over fewer elements than the widest
    swizzle's row (K32SWZ5: a register operand at a 32-row contraction lowered only so)."""
    if swizzle is None:
        return plgpu.SMEM(shape, dtype)
    return plgpu.SMEM(shape, dtype, transforms=swizzle_transforms(dtype, swizzle))


def roles(wg, consumers: int, consumer, producer) -> None:
    """Each warpgroup's role in a cooperative block: the `consumers` first run `consumer(wg)` with their
    register budget raised to their share (`resources.raised_registers`), the last runs `producer()`
    with its lowered to `resources.PRODUCER_REGISTERS`, where the GPU's compile honors them
    (`resources.RAISES_HONORED`)."""
    @pl.when(wg < consumers)
    def _():
        plgpu.set_max_registers(resources.raised_registers(consumers), action="increase")
        consumer(wg)

    @pl.when(wg == consumers)
    def _():
        plgpu.set_max_registers(resources.PRODUCER_REGISTERS, action="decrease")
        producer()


class Slots:
    """A walk step's operand slots (`Family.walk`), by stream name: each acquired when the step first reads
    it (`slots[name]`) and released at the step's end, once its MMAs are done, unless the step `release`s it
    sooner (a slot its last product has read)."""

    def __init__(self, rings: dict[str, Ring], counter):
        self._rings, self._counter = rings, counter
        self._held: dict[str, Ref] = {}
        self._released: set[str] = set()

    def __getitem__(self, name: str):
        if name not in self._held:
            self._held[name] = self._rings[name].acquire(self._counter)
        return self._held[name]

    def release(self, name: str) -> None:
        """Stream `name`'s slot given back now, its reads done (acquired first if the step read none)."""
        self[name]
        self._rings[name].release(self._counter)
        self._released.add(name)

    def done(self) -> None:
        """Every slot the step has not given back given back: each consumer waits on every load (a wait
        takes its barrier's next phase), so a slot it never read is still acquired before it is released."""
        for name in self._rings:
            if name not in self._released:
                self.release(name)

    def held(self) -> None:
        """Every slot the step has not read acquired, none given back: a step whose products stay running
        past it (`Family.walk`'s `in_flight`) gives its slots back a step later (`released`)."""
        if self._released:
            raise ValueError("a step whose products run past it gives back no slot early")
        for name in self._rings:
            self[name]


def released(rings: dict[str, Ring], counter) -> None:
    """Every ring's slot of step `counter` given back: a step's slots its walk held past it (`Slots.held`)."""
    for ring in rings.values():
        ring.release(counter)


class Family[Allocated, Opened, Accumulator](abc.ABC):
    """An instruction family's bundle (design §6, one mainloop per overlap structure): everything a Mosaic
    mainloop issues that the hardware decides, so a mainloop holds no instruction, layout or barrier of its
    own. Its facts (`name`, the MMA's row unit `rows`), its launch and roles (`kernel`, `roles`), its operand
    streams (`streams`, `opened`, `stream` the producer's, `walk` and `passed` the consumer's), its products
    (`product` into a fresh accumulator, `products`, `register_products` and `staged_products` in place,
    `settled`), and its accumulator's layouts (`layout`, `row`, `column`, which `zeros`, `full`, `iota`,
    `rows_of`, `columns_of`, `column_zeros`, `column_sums` and `as_row` read). It is generic in its streams'
    allocations (`Allocated`, `streams`'), its opened streams (`Opened`, `opened`'s) and the accumulator its
    products issue into in place (`Accumulator`: a warpgroup accumulator ref, or a register array)."""

    name: str
    rows: int
    consumers: int
    """The compute roles a program may take (sm90 two warpgroups, sm80 one)."""
    bounded_loads: bool
    """Whether a load past an operand's extent reads zeros (TMA's) rather than past it (cp.async's), so an
    operand need not be padded to whole tiles."""
    role_rows: int
    """The rows a consumer role's products cover at least: one warpgroup's (wgmma's one MMA of 64 rows,
    mma.sync's four warps of 16 each)."""

    @property
    def transposed_rows(self) -> int:
        """The rows a step of an operand its products read from registers transposed (a kernel gradient's
        `[rows, depth]` stage) takes at least: a role's (`role_rows`)."""
        return self.role_rows
    held_registers: int = 40
    """The registers a consumer thread holds beside its accumulator and operand fragments (addresses,
    indices, the walk's carry): CUTLASS's cooperative sm90 kernel's 168 less its accumulator's 128."""
    register_step: int = 8
    """The allocator's registers per thread in one step (setmaxnreg's eight-register granularity).
    A compile gate tests admits at their actual cap, refusals one such step below it."""
    operands_in_registers: bool
    """Whether a product reads its operands from registers a consumer loads them into (mma.sync's), where
    warpgroup MMA reads both from shared memory."""
    products_overlap: bool
    """Whether a warpgroup's products run asynchronously, an issued product's accumulator held while the
    program goes on (warpgroup MMA's), where each mma.sync product completes before the next: a launch's
    products issued in turn hold their accumulators at once on the one, one at a time on the other."""
    fixes_up: bool = False
    """Whether a tile's steps may be walked by several programs in turn (`tail.Tail`), each but the first
    leaving its fp32 partial in global memory behind a GPU-scope semaphore the first waits on (`fixed_up`,
    `partials_ready`)."""
    stores_through_registers: bool
    """Whether a program copies its stored output tiles out through its registers (no TMA store before
    Hopper: `mosaic_sm80.Mma.persist`), so its epilogue holds its share of the stored tile beside the
    accumulator (`resources.tile_registers`)."""
    @property
    def lhs_copies(self) -> int:
        """The warps of a role each holding the left operand's fragment for the same rows (mma.sync's warps
        along a tile's columns, JAX's `MMALayouts`' `lhs` replicated over them): one, unless replicated."""
        return 1

    @property
    def rhs_copies(self) -> int:
        """The warps of a role each holding the right operand's fragment for the same columns (mma.sync's
        warps along a tile's rows, JAX's `MMALayouts`' `rhs` replicated over them): one, unless
        replicated."""
        return 1

    def fragment_depth(self, depth: int) -> int:
        """The depth of a step's right operand fragments a consumer holds at once, where its products read
        registers (`operands_in_registers`): the step's, unless the family loads them a sub-step at a
        time (the left operand's it holds a step deep)."""
        return depth
    widest_product: int | None = None
    """The most columns one product spans (warpgroup MMA's 256, wgmma's widest N), which a tile's columns
    are at most; None where a tile's columns are many products' (mma.sync's 8 each), held by its
    registers alone (`resources.tile_registers`)."""
    copy_extent: int | None = None
    """The most elements one asynchronous copy moves along a dimension of its box after the copy's tiling
    (TMA's, 256): an operand block tiled in 8 rows by a swizzled row stays far below it at any tile its
    registers hold, and a one-dimensional side operand's block, untiled, is at most it
    (`grouped_matmul._Mosaic.fitting`); None where cp.async moves a block (mma.sync's)."""
    takes_tf32: bool = False
    """Whether its products take fp32 operands, as TF32 (warpgroup MMA's kind::tf32: Mosaic's wgmma issues
    fp32 shared-memory operands so); where not (mma.sync, whose Mosaic lowering takes 16-bit, fp8, int
    and f64 operands), a template splits fp32 operands into bf16 parts instead."""
    copies_vectors: bool = True
    """Whether its copies move a one-dimensional block untiled (TMA's), where cp.async spreads a copy over a
    warpgroup's threads, 128 elements at least ('CP_ASYNC requires the number of elements to be a multiple
    of 128', R9-gpu-a100): a step reads a side vector it does not copy from global memory instead."""
    counts_hooks: bool = True
    """Whether its register model counts what a template's hooks and exact sums hold (side operands' tiles,
    a running sum), so a hooked or exactly summed call derives a tile that spills nothing; where not
    (mma.sync's), such a call is not its launch's."""
    most_shared: int
    """The most shared memory a block of any GPU of the family holds (CUDA's opt-in limit), which bounds
    the stages a generated tile takes; a GPU's own (`generation.shared_budget`) bounds the tiles its
    launches keep."""
    cluster: int = 1
    """The programs a cluster of the family's launch takes where they read a block in common (`kernel`'s
    `cluster`), each loading its share of the block multicast into every one's shared memory (`streams`'
    `multicast`): TMA multicast's pair; 1 where a load reaches its own program's shared memory alone
    (cp.async's)."""
    walks_scoped: bool = False
    """Whether its walks scope their streams apart, each in its own scope (`emit_pipeline`'s slots): a tile
    scoped after a walk then takes the larger of the two (`resources.shared_bytes`)."""

    def programs(self, consumers: int) -> int:
        """The most programs of one launch a multiprocessor's registers hold at once, each `consumers`
        consumer roles: one where the family's roles split the register file between them (warpgroup
        MMA's consumers beside its producer, `resources.consumer_registers`). Each stages its share of the
        multiprocessor's shared memory (`resources.program_shared`), which may hold no tile at that many."""
        del consumers
        return 1

    @abc.abstractmethod
    def registers(self, consumers: int) -> int:
        """The registers a thread of each of `consumers` consumer roles may hold."""

    @abc.abstractmethod
    def column_unit(self, dtype) -> int:
        """The columns a tile's width steps by for operands of `dtype`: the swizzle a block's rows are
        staged in, or the row its stores lay over a role's threads."""

    def stores_columns(self, columns: int, rows: int) -> bool:
        """Whether its stores take an output tile `columns` wide (a multiple of `column_unit`) whose stored
        rows come in multiples of `rows` (a grouped call's, ragged to a row: 1; a dense call's, whole
        `TMA_ROWS`): any, unless its stores reload rows through registers laid over a role's threads
        (`Mma`)."""
        del columns, rows
        return True

    @abc.abstractmethod
    def depths(self, dtype) -> tuple[int, ...]:
        """The depths a step may take for operands of `dtype`, widest first: the swizzles its products
        read a block's rows at."""

    def column_registers(self, columns: int, dtype) -> int:
        """The registers a consumer thread holds for a vector of `columns` entries of `dtype` in the
        accumulator's column layout (`column`), broadcast over its rows: a family whose register model
        prices such a vector (a hook's, a template's per-row statistics) states it."""
        raise NotImplementedError(f"{self.name}'s register model takes no column vector")

    def narrowest_columns(self, dtype) -> int:
        """The fewest columns a staged operand block of `dtype` takes: a row of the narrowest swizzle
        (`NARROWEST_SWIZZLE_BYTES`), 16 bf16 columns, which both families' products read (wgmma's N in
        eights, mma.sync's n of 8 twice through ldmatrix). A template splitting its columns finer than its
        stores' `column_unit` to fill the multiprocessors (the recurrence's value blocks) steps by it."""
        return NARROWEST_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize

    @property
    @abc.abstractmethod
    def layout(self) -> plgpu_core.SomeLayout:
        """The accumulator's register layout, which the accumulator's layout members (`zeros` to `as_row`)
        read."""

    @property
    @abc.abstractmethod
    def row(self) -> plgpu_core.SomeLayout:
        """A row vector's layout: the accumulator's reduced over its columns."""

    @property
    @abc.abstractmethod
    def column(self) -> plgpu_core.SomeLayout:
        """A column vector's layout: the accumulator's reduced over its rows."""

    def warpgroups(self, consumers: int) -> int:
        """The warpgroups a program of `consumers` compute roles runs: theirs, and a producer's where the
        family streams from a role of its own (`roles`)."""
        return consumers

    def staged(self, value: jax.Array, buffer) -> Ref:
        """`value` in the shared-memory tile `buffer`, in its dtype, for a product to read: a register right
        operand's way to an MMA (`staged_products` stages left ones). Mosaic GPU brackets a store with the
        warpgroup's barriers (its automatic barriers), so every warp's reads of `buffer` are done before it
        and its writes seen after; the family publishes them to its asynchronous readers (`published`)."""
        buffer[...] = value.astype(buffer.dtype)
        self.published()
        return buffer

    def summed(self, shape: tuple[int, int], issue, interpret: bool) -> jax.Array:
        """`issue(acc) -> acc` issuing products in place (`products`) into a fresh fp32 accumulator of
        `shape` (zero), its value once they are done: a register accumulator's, which a family whose
        accumulator is a ref scopes (wgmma's)."""
        del interpret
        return issue(self.zeros(shape))

    @abc.abstractmethod
    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | None = None,
               reducing_warpgroups: int = 0) -> Callable:
        """`entry` launched over `grid`, `consumers` compute roles a program beside the family's own, with
        `approx_math` the hardware's approximate exponential and reciprocal (`compiler_params`), named
        `name` where given (Mosaic names its kernel and its compile reports by it); each of `grid`'s
        programs a cluster of `cluster` along `CLUSTER` (at most the family's `cluster`); its `cost`, the
        work it states to XLA and to Dew's FLOP count (the custom call's `cost_estimate_json`); its
        `reducing_warpgroups` (the block's, where its arithmetic may reduce across warps implicitly) each
        reserving its share of reduction scratch (`warpgroup_reduction_scratch`)."""

    @abc.abstractmethod
    def scoped(self, body, *allocations, streams):
        """`body(*refs, streams=...)` with `allocations` scoped across the program's roles and its streams'
        (`streams`), and what it returns."""

    @abc.abstractmethod
    def roles(self, consumers: int, consumer, producer) -> None:
        """Each role of a program: `consumer(index)` the compute roles', `producer()` the loads' where the
        family streams from a role of its own."""

    @abc.abstractmethod
    def streams(self, blocks: dict, stages: int, consumers: int,
                multicast: tuple[str, ...] = (), swizzles: dict[str, int] | None = None) -> Allocated:
        """The scoped allocations of a stream of each `blocks[name]` `(shape, dtype)` in `stages` slots read
        by `consumers` (`opened` once scoped), each stream `multicast` names one the programs of a cluster
        share, loaded once between them (`kernel`'s `cluster`), each `swizzles` names staged at its swizzle
        (`swizzle_for`; inference's own otherwise)."""

    @abc.abstractmethod
    def opened(self, allocated: Allocated, views: dict | None = None) -> Opened:
        """The streams of `streams`' allocations, scoped, by name; their `views` where the family's walk
        loads them (`viewed`: each a ref and the block indices of a step's tile in it)."""

    @abc.abstractmethod
    def stream(self, streams, views: dict, lo, hi) -> None:
        """The producer's loads of steps `lo` to `hi` from `views` (`opened`'s), where the family has one."""

    @abc.abstractmethod
    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, consumer: Callable, *,
                consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0) -> None:
        """A persistent kernel's tiles, the grid `tiles` spread over the launch's programs along `axis`, its
        streams (`streams`, as `scoped` hands them) running on across them: `schedule(index)` gives a tile's
        `(active, steps, views, tile)`, whether it streams anything, its step count, each stream's view
        (`viewed`: a ref and the block indices of the tile's step `j`, counted from 0) and what its
        consumers are given. Each consumer role runs `consumer(index, tile, walk, store, active=...)` a tile,
        once for each value of `active`: `walk(step, start, shapes, *, initial=None)` walks the tile's steps
        as `walk` does, a step's index the tile's own (where not active it takes none, its accumulators as
        they start), and `store(output, value, ref, place)` stores a consumer's `value` of output `output`
        (`outputs`' `(shape, dtype)`) to `ref` at `place`, `store_rows`' `(row_start, first, stored,
        column, most)` and, where the stored rows come in whole multiples, `least`. The family owns the
        loads, the step counter across tiles, the streams' drain, the outputs' staging and the stores'
        completion. A cluster's programs run the same tiles: where a stream is multicast (`streams`),
        `schedule` gives each of them the same `active`, `steps` and view of it."""

    @abc.abstractmethod
    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None
                ) -> tuple[C, tuple[jax.Array, ...]]:
        """A consumer's walk of `segments`, each `(step, lo, hi)`, `step(index, slots, carry, accumulators)
        -> (carry, accumulators)` given each stream's slots (`Slots`, the streams' steps counted from
        `first`), from `start`, with fp32 accumulators of `shapes` its products issue into in place, zero
        or their `initial` values (in the accumulator's layout; None, zero). `in_flight` steps' products stay
        running past their step (0 or 1): their slots go back once they are done, a step later, so the next
        step's operands are waited for while they run. Returns the carry and the accumulators' values."""

    @abc.abstractmethod
    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        """A consumer's steps `lo` to `hi` it reads none of, each load awaited and released as a reading
        consumer's (counted from `first`); `between()` after each step."""

    @abc.abstractmethod
    def loaded(self, copies) -> None:
        """Each `(view, ref, barrier)` copied whole into its shared-memory `ref`."""

    def output_staging(self, outputs: tuple, consumers: int) -> list:
        """The shared memory `persist` stages each of `outputs` (`(shape, dtype)`, a tile's) in, which a
        template's allocations count as its own (`resources.Scope`): the whole tile, where a family stages
        it whole."""
        del consumers
        return [plgpu.SMEM(shape, dtype) for shape, dtype in outputs]

    def published(self) -> None:
        """A consumer's writes to shared memory made visible to the readers that read it asynchronously
        (warpgroup MMA, TMA), before any of them reads it: a family whose products read shared memory
        through its consumers' own loads (mma.sync's) has nothing to publish."""
        plgpu.commit_smem()

    @abc.abstractmethod
    def stored(self, parts, views, *, smem, wait: bool = True) -> None:
        """Each of `parts` written whole to its view of `views`, through `smem` where the family stores so.
        Unless `wait`, the last copy out of `smem` stays in flight as one store group, so the caller's
        next work overlaps it, and the caller owns `smem`'s reuse: it waits (`store_wait`) before writing
        `smem` again, an earlier deferred store's or an add's (`added`) reads of it included, so stores
        into other tiles stay in flight past the call (`store_wait(stores - 1)` keeps one a tile). A call
        of several parts still waits for each part's reads before writing the next, every earlier store
        group with it."""

    def store_wait(self, pending: int) -> None:
        """Waits until at most `pending` of this thread's deferred stores (`stored(wait=False)`) still read
        their shared tiles: a family that stores through registers has none."""
        del pending

    @abc.abstractmethod
    def added(self, value: jax.Array, view, interpret: bool, *, smem) -> None:
        """`value` added into its view (fp32, in global memory) through `smem`, atomically against every
        other role's adds into it; interpreted, a read and a write of the view, atomic against other
        programs' alone (the interpreter runs a program's roles at once and a grid's programs in turn)."""

    @abc.abstractmethod
    def held(self, ref: Ref) -> Operand:
        """An operand every step reads (`ref`, in shared memory) where the family keeps it: shared memory,
        or registers."""

    @abc.abstractmethod
    def product(self, lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
        """`lhs @ rhs` into a fresh fp32 accumulator of `shape`, its value, with `then` issued while the
        product runs."""

    @abc.abstractmethod
    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        """`lhs @ rhs` into a fresh fp32 accumulator, `lhs` in registers (16-bit), its value."""

    @abc.abstractmethod
    def products(self, acc: Accumulator, lhs, rhs, interpret: bool) -> Accumulator:
        """`acc` with `lhs @ rhs` issued into it in place, both operands in shared memory (a GEMM's step)."""

    def issued(self, acc: Accumulator, left: tuple, right: tuple, interpret: bool) -> Accumulator:
        """`acc` with the products of `left`'s and `right`'s parts (`product.formed`, each a part's
        shared-memory tile) issued into it in place, the pairs `product.terms` gives, smallest first
        (`products`)."""
        for i, j in product.terms(len(left), len(right)):
            acc = self.products(acc, left[i], right[j], interpret)
        return acc

    def apart(self, shape: tuple[int, int], left: tuple, right: tuple, interpret: bool) -> jax.Array:
        """The products of `left`'s and `right`'s parts (`issued`'s), each into its own fresh fp32
        accumulator of `shape` (`summed`), added left to right in `product.terms`' order, so the smallest
        meet first: a tensor core's accumulation truncates (mma.sync's), so a part accumulated onto a larger
        one loses its last bit where a sum rounds it."""
        def issue(i: int, j: int, acc: Accumulator) -> Accumulator:
            return self.products(acc, left[i], right[j], interpret)

        return functools.reduce(jnp.add, (self.summed(shape, functools.partial(issue, i, j), interpret)
                                          for i, j in product.terms(len(left), len(right))))

    @abc.abstractmethod
    def register_products(self, acc: Accumulator, parts, rhs, interpret: bool) -> Accumulator:
        """`acc` with `parts`' products with `rhs` issued into it in place, each part's left operand in
        registers, `parts` an iterable that may form each as it is taken
        (`attention_algebra.formed_parts`)."""

    @abc.abstractmethod
    def staged_products(self, acc: Accumulator, parts, rhs, buffers, interpret: bool, *,
                        transposed: bool = False) -> Accumulator:
        """`register_products` of `parts` each staged in `buffers` first (its transpose the left operand
        where `transposed`)."""

    @abc.abstractmethod
    def rescaled_products(self, acc: Accumulator, scale, parts, rhs, interpret: bool) -> Accumulator:
        """`acc` taken by `scale(value)` in place, then `register_products` of `parts` with `rhs` into it."""

    def partials_ready(self, shape: tuple[int, ...], interpret: bool):
        """The semaphores of `shape` a tile's partials are signalled on across programs (`fixed_up`); none
        interpreted, where the programs' order stands in for them (`tail.ordered`)."""
        raise NotImplementedError(f"{self.name} fixes no tile up across programs (`fixes_up`)")

    def fixed_up(self, summed, partials, ready, program, start, gathered, store) -> None:
        """A balanced tail's consumer's share of a tile (`tail.Tail`), `summed` from its program's steps:
        left in `partials` (each program's, this consumer's) behind `ready` (`partials_ready`) where its
        range starts inside the tile; otherwise added to every partial the programs after it up to
        `gathered` left, each awaited, and stored (`store`)."""
        raise NotImplementedError(f"{self.name} fixes no tile up across programs (`fixes_up`)")

    @abc.abstractmethod
    def settled(self, pending: int = 0) -> None:
        """Every product issued so far done, but the last `pending` issued."""

    def zeros(self, shape: tuple[int, ...]) -> jax.Array:
        """fp32 zeros in the accumulator's layout (two-dimensional) or a row's (one-dimensional)."""
        return self.full(shape, 0.0)

    def full(self, shape: tuple[int, ...], value: float, dtype=jnp.float32) -> jax.Array:
        """`value` in `dtype` (fp32 unless given) in the accumulator's layout or a row's (`zeros`)."""
        layout = self.layout if len(shape) == 2 else self.row
        return plgpu.layout_cast(jnp.full(shape, value, dtype), layout)

    def iota(self, shape: tuple[int, int], dimension: int) -> jax.Array:
        """Indices along `dimension` of a tile in the accumulator's layout."""
        return plgpu.broadcasted_iota(jnp.int32, shape, dimension, layout=self.layout)

    def rows_of(self, ref) -> jax.Array:
        """A row vector read from `ref` (`[rows]`) in a row's layout."""
        return plgpu.load(ref, layout=self.row, optimized=False)

    def columns_of(self, ref) -> jax.Array:
        """A column vector read from `ref` (`[columns]`) in a column's layout (a tile's keys')."""
        return plgpu.load(ref, layout=self.column, optimized=False)

    def tile_of(self, ref) -> jax.Array:
        """A tile read from `ref` (`[rows, columns]`) in the accumulator's layout."""
        return plgpu.load(ref, layout=self.layout, optimized=False)

    def column_zeros(self, count: int) -> jax.Array:
        """fp32 zeros `[count]` in a column's layout (`columns_of`'s), the carry of a tile's key sums."""
        return plgpu.layout_cast(jnp.zeros((count,), jnp.float32), self.column)

    column_scratch: int
    """The fp32 entries of shared memory a consumer's `column_sums` takes: a reduction across the warps
    holding a tile's rows takes one register's partial sums at a time."""

    reduction_scratch: int = 0
    """The bytes of shared memory the family's kernels reserve for Mosaic GPU's implicit reductions across
    warps (its compiler parameters' `reduction_scratch_bytes`) where they reduce (`resources.Scope.reduces`):
    none where a reduction takes the consumer's own (`column_sums`)."""

    warpgroup_reduction_scratch: int = 0
    """The bytes of Mosaic GPU's reduction scratch a warpgroup of a kernel whose arithmetic reduces across
    warps implicitly takes as its share (`kernel`'s `reducing_warpgroups`, `resources.Scope`'s): a family
    whose kernels reserve no scratch by default (`reduction_scratch`)."""

    def column_sums(self, value: jax.Array, scratch, interpret: bool) -> jax.Array:
        """The sums of `value`'s columns `[columns]` in a column's layout, `value` `[rows, columns]` in the
        accumulator's layout: across the warps holding its rows, through the consumer's own `scratch`
        (`column_scratch` fp32 entries), as the block's other warpgroups reduce at once."""
        if interpret:
            return value.sum(axis=0)

        def sums(ctx, tile, scratch):
            del ctx
            return tile.reduce("add", axis=0, scratch=scratch)

        # Mosaic GPU's own reduction across the warps, through the given scratch; inline_mgpu passes the
        # launch context first and takes the tile and its scratch.
        summed: Callable[..., jax.Array] = plgpu.inline_mgpu(
            arg_types=(self.layout, plgpu.RefType()),
            return_type=plgpu.ShapeDtypeStruct((value.shape[1],), jnp.dtype(jnp.float32),
                                               layout=self.column))(sums)
        return summed(value, scratch)

    def as_row(self, vector: jax.Array) -> jax.Array:
        """`vector` `[rows]` in a row's layout, whatever its own."""
        return plgpu.layout_cast(vector, self.row)


class RingFamily[Accumulator](Family[dict, dict[str, Ring], Accumulator]):
    """A family whose loads a producer warpgroup issues by TMA into `Ring`s, beside one or two consumer
    warpgroups at `setmaxnreg` budgets (`roles`): sm90's and sm100's. Its walk acquires each step's slots as
    the step reads them and releases them once its products are done (`Slots`). Its products read their
    operands from the rings, a block's rows staged at the widest swizzle, 128 bytes."""
    bounded_loads = True
    operands_in_registers = False
    products_overlap = True
    fixes_up = True
    # Mosaic GPU's default for a warpgroup (its CompilerParams' 128 x 4 x 4 bytes).
    warpgroup_reduction_scratch = 2048
    stores_through_registers = False
    copy_extent = 256

    def registers(self, consumers: int) -> int:
        return resources.consumer_registers(consumers)

    def column_unit(self, dtype) -> int:
        return SWIZZLE_BYTES // jnp.dtype(dtype).itemsize

    def depths(self, dtype) -> tuple[int, ...]:
        return (SWIZZLE_BYTES // jnp.dtype(dtype).itemsize,)

    def warpgroups(self, consumers: int) -> int:
        return consumers + 1

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | None = None,
               reducing_warpgroups: int = 0) -> Callable:
        if cluster > self.cluster:
            raise ValueError(f"{self.name} launches a cluster of at most {self.cluster}, not {cluster}")
        per_shard(name or self.name)
        clustered = {'cluster': (cluster,), 'cluster_names': (CLUSTER,)} if cluster > 1 else {}
        return plgpu.kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names,
                            num_threads=self.warpgroups(consumers), thread_name="wg",
                            compiler_params=compiler_params(
                                approx_math=approx_math,
                                reduction_scratch=self.warpgroup_reduction_scratch * reducing_warpgroups),
                            interpret=interpreted(interpret), kernel_name=name, cost_estimate=cost,
                            **clustered)

    def scoped(self, body, *allocations, streams):
        return pl.run_scoped(lambda *refs, **rings: body(*refs, streams=rings), *allocations, **streams,
                             collective_axes="wg")

    def roles(self, consumers: int, consumer, producer) -> None:
        roles(lax.axis_index("wg"), consumers, consumer, producer)

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None) -> dict:
        return {name: ring(shape, dtype, stages, consumers=consumers, multicast=name in multicast,
                           swizzle=(swizzles or {}).get(name))
                for name, (shape, dtype) in blocks.items()}

    def opened(self, allocated: dict, views: dict | None = None) -> dict[str, Ring]:
        del views
        return {name: Ring(**scoped) for name, scoped in allocated.items()}

    def fed(self, streams: dict[str, Ring], views: dict, lo, hi, wg, *, first: jax.Array | int = 0,
            refills: bool = True) -> "dict[str, Ring] | Feed":
        """What a consumer's walk of steps `lo` to `hi` reads (`walk`'s `streams`), the walk taking the same
        `first`: on a layout with a producer, the rings, which its producer loads from `views` (`stream`);
        on one whose consumers feed them (`ConsumerFed.fed`), a `Feed`."""
        del views, lo, hi, wg, first, refills
        return streams

    def stream(self, streams, views: dict, lo, hi) -> None:
        rings = tuple(streams.values())

        def tiles(index):
            return tuple(viewed(views[name], ring.smem.shape[1:], index)
                         for name, ring in zip(streams, rings, strict=True))

        streamed(rings, tiles, lo, hi)

    def output_staging(self, outputs: tuple, consumers: int) -> list:
        # Each consumer's share of each output in two chunk buffers, not the whole tile (`output_staging`).
        return [plgpu.SMEM((consumers, output_staging(shape[1], (dtype,))[1], shape[0],
                            output_staging(shape[1], (dtype,))[0]), dtype)
                for shape, dtype in outputs]

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, consumer: Callable, *,
                consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0) -> None:
        # The producer loads every tile's steps into the rings at a counter running across the tiles, so
        # a tile's first loads go out while the consumers store the last one (`_tiled`).
        rings = self.opened(streams)
        staging = self.output_staging(outputs, consumers)

        def producer():
            @plgpu.nd_loop(tiles, collective_axes=axis, init_carry=jnp.int32(0))
            def loaded(loop: plgpu.NDLoopInfo, carry):
                counter = carry  # the step every ring is at
                active, steps, views, _ = schedule(loop.index)
                count = jnp.where(active, steps, 0)

                @pl.loop(0, count)
                def _(step):
                    for name, ring in rings.items():
                        ring.load(counter + step, viewed(views[name], ring.smem.shape[1:], step))

                return counter + count

            for ring in rings.values():
                ring.drain(loaded)

        def consumed(smems):
            return self._tiled(rings, tiles, schedule, consumer, smems, axis=axis, interpret=interpret,
                               in_flight=in_flight)

        pl.run_scoped(lambda *smems: self.roles(consumers, consumed(smems), producer), *staging,
                      collective_axes="wg")

    def _tiled(self, rings: dict[str, Ring], tiles: tuple[int, ...], schedule: Callable, consumer: Callable,
               smems, *, axis: str, interpret: bool, in_flight: int) -> Callable:
        """`persist`'s consumers: each walks its tiles' steps at a counter running across the tiles
        (`_tile_walk`), stages its share of each output in chunks (`output_staging`: two buffers of
        `staged_columns`) and stores each by TMA, waiting only for a buffer's last store to read it before
        it writes it again; the rings are left as `_ended` leaves them."""
        def run(index):
            def store(output, value, ref, place):
                smem = smems[output].at[index]
                buffers, width = smem.shape[0], smem.shape[-1]
                row_start, first, stored, column, *ladder = place
                for chunk in range(value.shape[1] // width):
                    if chunk == 0 or chunk >= buffers:
                        # The buffer's last store (the last tile's, at its first chunk) has read it.
                        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    buffer = smem.at[chunk % buffers]
                    buffer[...] = value[:, chunk * width:(chunk + 1) * width].astype(smem.dtype)
                    plgpu.commit_smem()
                    store_rows(buffer, ref, row_start, first, stored, column + chunk * width, *ladder)

            @plgpu.nd_loop(tiles, collective_axes=axis, init_carry=jnp.int32(0))
            def ended(loop: plgpu.NDLoopInfo, carry):
                counter = carry  # the step every ring is at
                active, steps, views, tile = schedule(loop.index)
                walk = self._tile_walk(rings, counter, jnp.where(active, steps, 0), views, index,
                                       interpret=interpret, in_flight=in_flight)
                pl.when(active)(lambda: consumer(index, tile, walk, store, active=True))
                pl.when(~active)(lambda: consumer(index, tile, unwalked(self), store, active=False))
                return counter + jnp.where(active, steps, 0)

            self._ended(rings, ended, index)
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

        return run

    def _tile_walk(self, rings: dict[str, Ring], counter, steps, views: dict, wg, *, interpret: bool,
                   in_flight: int) -> Callable:
        """`persist`'s walk of one tile's `steps` for consumer `wg` (its consumer's `walk(step, start,
        shapes, initial=)`), the rings at `counter`, each step's index counted from the tile's first, over
        the streams the layout gives the tile (`_tile_streams`)."""
        streams = self._tile_streams(rings, counter, steps, views, wg)

        def walk(step, start, shapes, *, initial=None):
            def stepped(at, slots, carry, accumulators):
                return step(at - counter, slots, carry, accumulators)

            return self.walk(streams, ((stepped, counter, counter + steps),), start, shapes, interpret,
                             in_flight=in_flight, initial=initial)

        return walk

    def _tile_streams(self, rings: dict[str, Ring], counter, steps, views: dict,
                      wg) -> "dict[str, Ring] | Feed":
        """The streams a tile's walk reads (`_tile_walk`): the rings, which the producer loads from
        `views`."""
        del counter, steps, views, wg
        return rings

    def _ended(self, rings: dict[str, Ring], end, wg) -> None:
        """Consumer `wg`'s rings after `persist`'s last tile, at counter `end`: the producer drains them."""
        del rings, end, wg

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None
                ) -> tuple[C, tuple[jax.Array, ...]]:
        return self._walked(streams, segments, start, shapes, interpret, first=first, in_flight=in_flight,
                            initial=initial)

    def _walked[C](self, streams: dict[str, Ring], segments, start: C, shapes: tuple, interpret: bool, *,
                   first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                   freed: Callable | None = None) -> tuple[C, tuple[jax.Array, ...]]:
        """`walk`, `freed(counter)` run after each step's slots are given back (step `counter`'s)."""
        if in_flight not in (0, 1):
            raise ValueError(f"a walk keeps no step's products or one step's in flight, not {in_flight}")

        def given(counter, *, release: bool = False) -> None:
            # Step `counter`'s slots given back (`release`: here, a step its products ran past), then `freed`.
            if release:
                released(streams, counter)
            if freed is not None:
                freed(counter)

        def slotted(step, lo):
            def run(index, carry, accumulators):
                slots = Slots(streams, index - first)
                carry, accumulators = step(index, slots, carry, accumulators)
                if not in_flight:
                    # The step's products done before its slots are given back.
                    self.settled()
                    slots.done()
                    given(index - first)
                    return carry, accumulators
                # The step's products run on while the next step's operands are waited for: the previous
                # step's, done now, give back its slots.
                slots.held()
                self.settled(in_flight)
                pl.when(index > lo)(lambda: given(index - 1 - first, release=True))
                return carry, accumulators

            return run

        walked = self.accumulated(tuple((slotted(step, lo), lo, hi) for step, lo, hi in segments), start,
                                  shapes, interpret, initial)
        if in_flight:
            # Each segment's last step, its products done by the walk's end.
            for _, lo, hi in segments:
                pl.when(hi > lo)(lambda hi=hi: given(hi - 1 - first, release=True))
        return walked

    @abc.abstractmethod
    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool, initial: tuple | None = None
                       ) -> tuple[C, tuple[jax.Array, ...]]:
        """A walk of `segments`, `step(index, carry, accumulators)`, with the family's accumulators of
        `shapes` held across it from their `initial` values (`walk`)."""

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        passed(tuple(streams.values()), lo, hi, first=first, between=between)

    def loaded(self, copies) -> None:
        loaded(copies)

    def stored(self, parts, views, *, smem, wait: bool = True) -> None:
        # An add still reading `smem` (`added`) done first, unless the caller defers and so waits itself
        # (`Family.stored`): waiting here would drain its other tiles' deferred stores too.
        if wait:
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        stored(parts, views, smem=smem, wait=wait)

    def store_wait(self, pending: int) -> None:
        plgpu.wait_smem_to_gmem(pending, wait_read_only=True)

    def added(self, value: jax.Array, view, interpret: bool, *, smem) -> None:
        if interpret:
            # Mosaic GPU's interpreter (jax 0.11.2) takes no TMA reduction.
            view[...] = view[...] + value
            return
        # A TMA reduce from `smem`, the previous add's reads of it done first; its writes go on.
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        smem[...] = value
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(smem, view, reduction_op="add")

    def partials_ready(self, shape: tuple[int, ...], interpret: bool):
        return None if interpret else pl.get_global(plgpu.SemaphoreType.REGULAR(shape))

    def fixed_up(self, summed, partials, ready, program, start, gathered, store) -> None:
        # GPU-scope signals and waits, proven to lower and run on an H100 at the pinned JAX: TGHYBRID1-h100's
        # SASS releases each partial with MEMBAR.GPU and a strong global add, and acquires it with a strong
        # CAS. Interpreted, the partials are left before they are read (`tail.ordered`).
        @pl.when(start > 0)
        def _():
            partials[program] = summed
            if ready is not None:
                plgpu.semaphore_signal(ready.at[program], memory_scope="gpu")

        @pl.when(start == 0)
        def _():
            def added(other, running):
                if ready is not None:
                    plgpu.semaphore_wait(ready.at[other], memory_scope="gpu")
                return running + partials[other]

            store(lax.fori_loop(program + 1, gathered, added, summed))

    def held(self, ref: Ref) -> Operand:
        return ref


@dataclasses.dataclass(frozen=True)
class Feed:
    """A walk's steps `lo` to `hi` (its indices, `walk`'s segment bounds) as a consumer-fed walk reads them
    (`ConsumerFed`), step `index` in the rings' slot of counter `index - first` (`walk`'s `first`, as
    `Slots` takes it): the loading consumer (`loads`, one warpgroup's traced flag) loads each step into
    each of `rings` from its view of `views` at `index - origin` (a plain walk's views take its own
    indices, as a producer's `stream` gives them: `origin` 0; `persist`'s a tile's steps counted from its
    first), the first steps before the walk (`primed`) and each later one once the slot it reuses is given
    back
    (`refilled`, at the given step's counter): by the walk after each step's release, or where `refills`
    is False by the template, at a point of its own (a walk whose turns order its consumers refills past
    the turn that lets the other finish); the loading consumer waits at the walk's end for every slot's
    last release where it `drains` (a plain walk's; `persist` drains after a program's last tile)."""
    rings: dict[str, Ring]
    views: dict
    lo: jax.Array
    hi: jax.Array
    loads: jax.Array
    first: jax.Array | int = 0
    origin: jax.Array | int = 0
    refills: bool = True
    drains: bool = False

    @property
    def stages(self) -> int:
        return min(ring.stages for ring in self.rings.values())

    def load(self, index) -> None:
        for name, ring in self.rings.items():
            ring.load(index - self.first, viewed(self.views[name], ring.smem.shape[1:], index - self.origin))

    def primed(self) -> None:
        """The walk's first steps, a slot each, loaded."""
        for ahead in range(self.stages):
            pl.when(self.loads & (self.lo + ahead < self.hi))(lambda ahead=ahead: self.load(self.lo + ahead))

    def refilled(self, counter) -> None:
        """The step at ring counter `counter`, its slots given back, has them take the walk's step a ring
        ahead where it has one: its load waits until every consumer has given them back (`Ring.load`)."""
        ahead = counter + self.first + self.stages
        pl.when(self.loads & (ahead < self.hi))(lambda: self.load(ahead))

    def drained(self) -> None:
        """The loading consumer's wait for the release of each slot the walk filled last (`Ring.drain`), so
        no slot is reused or freed while another consumer reads it."""
        @pl.when(self.loads)
        def _():
            for ring in self.rings.values():
                ring.drain(self.hi - self.first)


class ConsumerFed[Accumulator](RingFamily[Accumulator]):
    """A ring family's layout without a producer (`RingFamily`'s otherwise), an instruction family's bundle
    taking it first among its bases (`mosaic_sm90.WgmmaFed`): its
    consumers alone run, the first issuing every ring's loads between its own steps (`Feed`; Mosaic GPU
    elects one of its threads to issue each copy), so each holds its program's entry registers, which no
    raise moves (`resources.entry_registers`: two consumers' 255, where a producer's third warpgroup leaves
    them 168 while the GPU's compile ignores setmaxnreg, `resources.RAISES_HONORED`). SETMAXNREG-FFI3-h100:
    attention's dK/dV at two consumers so, 0.732 ms, against 1.588 ms with a producer at 168 registers,
    spilling. Its streams run through `persist`, or a plain walk over a `Feed` (`fed`) every consumer walks
    whole (no `passed`: a consumer's steps are every consumer's); it takes no cluster (a multicast ring's
    loads are each program's producer's)."""
    cluster = 1
    loader = 0
    """The consumer that issues the loads."""

    def registers(self, consumers: int) -> int:
        return resources.entry_registers(consumers)

    def warpgroups(self, consumers: int) -> int:
        return consumers

    def roles(self, consumers: int, consumer, producer) -> None:
        del consumers, producer
        consumer(lax.axis_index("wg"))

    def stream(self, streams, views: dict, lo, hi) -> None:
        raise NotImplementedError(f"{self.name}'s consumers feed their own rings (`fed`)")

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        raise NotImplementedError(f"{self.name}'s consumers each walk every step of their rings (`fed`)")

    def fed(self, streams: dict[str, Ring], views: dict, lo, hi, wg, *, first: jax.Array | int = 0,
            refills: bool = True) -> "dict[str, Ring] | Feed":
        """`RingFamily.fed`: a `Feed` of steps `lo` to `hi` from `views`, at the walk's `first`, its first
        steps primed by the loading consumer, drained at its walk's end."""
        feed = Feed(streams, views, lo, hi, wg == self.loader, first=first, refills=refills, drains=True)
        feed.primed()
        return feed

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None
                ) -> tuple[C, tuple[jax.Array, ...]]:
        if not isinstance(streams, Feed):
            raise NotImplementedError(f"{self.name}'s consumers feed their own rings (`fed`)")
        # The feed loads each step at the ring counter the walk reads it at, both `index - first`.
        first = streams.first if isinstance(first, int) and first == 0 else first
        if first is not streams.first:
            raise ValueError(f"{self.name}'s fed walk takes the `first` its feed was made with (`fed`)")
        if in_flight and len(segments) > 1:
            # A segment's last step's release waits for the walk's end, which the feed's next segment, its
            # loads behind that release, would never reach.
            raise NotImplementedError(f"{self.name}'s fed walk keeps no products in flight across segments")
        walked = self._walked(streams.rings, segments, start, shapes, interpret, first=first,
                              in_flight=in_flight, initial=initial,
                              freed=streams.refilled if streams.refills else None)
        if streams.drains:
            streams.drained()
        return walked

    def _tile_streams(self, rings: dict[str, Ring], counter, steps, views: dict,
                      wg) -> "dict[str, Ring] | Feed":
        # The tile's steps as its loading consumer feeds them, the first ones primed.
        feed = Feed(rings, views, counter, counter + steps, wg == self.loader, origin=counter)
        feed.primed()
        return feed

    def _ended(self, rings: dict[str, Ring], end, wg) -> None:
        # The loading consumer has seen every release but the last ring's worth: it waits for those too, so
        # no slot is reused or freed while another consumer reads it, and every release is seen.
        @pl.when(wg == self.loader)
        def _():
            for ring in rings.values():
                ring.drain(end)


def summing(shape: tuple[int, ...], consumers: int) -> dict:
    """`summed_across`' allocations for `pl.run_scoped` (collective across the consumers, in a scope after
    the walk's, whose shared memory it reuses): every consumer's fp32 part but the first's, the barrier they
    arrive at once theirs is written, and the one the first arrives at once it has read them; none for one
    consumer, whose part is the sum."""
    if consumers == 1:
        return {}
    return {'parts': plgpu.SMEM((consumers - 1, *shape), jnp.float32),
            'ready': plgpu.Barrier(num_arrivals=consumers - 1, num_barriers=1),
            'read': plgpu.Barrier(num_arrivals=1, num_barriers=1)}


def summed_across(value, scratch: dict, wg, consumers: int, store: Callable) -> None:
    """The consumers' fp32 parts of one tile (each consumer's `value`) added and stored once: every consumer
    but the first leaves its part in `scratch` (`summing`'s, scoped) and arrives at its `ready`, then waits
    until the first has read every part (`read`), so the scratch serves the next sum; the first waits, adds
    them in the consumers' order (one sum, the same each run), arrives at `read` and stores the total
    (`store(total)`). One consumer stores its own."""
    if consumers == 1:
        store(value)
        return
    parts, ready, read = scratch['parts'], scratch['ready'], scratch['read']

    @pl.when(wg > 0)
    def _():
        parts[wg - 1] = value
        plgpu.barrier_arrive(ready.at[0])
        plgpu.barrier_wait(read.at[0])

    @pl.when(wg == 0)
    def _():
        plgpu.barrier_wait(ready.at[0])
        total = value
        for other in range(consumers - 1):
            total = total + parts[other]
        plgpu.barrier_arrive(read.at[0])
        store(total)


def unscaled(parts) -> None:
    """A product's parts issued in place take no power of two back (bf16's, `cotangent_parts`)."""
    if any(exponent is not None for _, exponent in parts):
        raise NotImplementedError("a scaled part's product is no in-place accumulation (fp16's)")


def unscaled_part(part: tuple) -> jax.Array:
    """`unscaled` of one part taken as it is formed: its operand."""
    unscaled((part,))
    return part[0]
