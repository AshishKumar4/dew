"""The templates' Pallas Mosaic GPU toolkit (design §6), every instruction family's: the launch,
warpgroup roles, barriers, shared-memory staging, operand streams and stores every Mosaic mainloop builds on.
The math is each template's algebra, written once; what an instruction family adds is its issue and its
layout facts alone (`Family`: mosaic_sm90's wgmma; mma.sync's and tcgen05's to come). Each template keeps
its own mainloop, product dependencies and launch DAG; only these primitives are shared.

A cooperative launch (`cooperative`) runs a producer warpgroup issuing the loads beside consumer warpgroups
computing, persistent over the multiprocessors, a pair of them a cluster where it shares one operand's blocks
(multicast, `multicast_axes`). The consumers report a tile drained (`drained`, `drain`) so a persistent
producer never loads a next tile's stage over one an MMA still reads, and meet (`meet`) where one consumer's
writes to shared memory are the other's operand. Each consumer's accumulator lives for one output tile
(`finished`), and a tile's rows reach global memory through `store_rows`, whose stores have static extents
however many rows a tile holds.

A mainloop that drives its own loads (the GEMM's kernel gradient, its stages streaming across tiles)
streams each operand through a `Ring` whose counter runs across tiles, and takes each warpgroup's role
(`roles`).
"""

import abc
import dataclasses
import math
from collections.abc import Callable

import jax
from jax import lax, numpy as jnp
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

Ref = state_types.AbstractRef | state_types.TransformedRef
Operand = Ref | jax.Array
"""A product's operand: a tile in shared memory (a ref) or in registers (an array)."""
"""A scoped allocation a primitive takes: shared memory, transformed (tiled and swizzled) or not, or
barriers."""


def compiler_params(*, approx_math: bool = False) -> plgpu.CompilerParams:
    """A mainloop's compiler parameters: Mosaic GPU's warpgroup lowering, whose arrays are a warpgroup's,
    and with `approx_math` the hardware's approximate exponential and reciprocal (ex2.approx, as Triton's
    exp2 and FlashAttention's are). No shared memory is reserved for reductions across warps: a block's
    warpgroups reduce at once, so each takes its own scratch, the consumers' from the kernel's allocations
    (`Family.column_sums`), and a reduction that would take Mosaic GPU's implicit scratch fails to lower
    rather than share it."""
    return plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup, approx_math=approx_math,
                                reduction_scratch_bytes=0)


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


def stored(parts, views, *, smem) -> None:
    """Each of `parts` written whole to its view of `views` through the shared buffer `smem`, in turn, the
    buffer reused once its copy is done."""
    for part, view in zip(parts, views, strict=True):
        smem[...] = part.astype(smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(smem, view)
        plgpu.wait_smem_to_gmem(0)


def store_rows(o_smem, o_gmem, row_start, first, stored, column, most: int,
               copy: Callable = plgpu.copy_smem_to_gmem) -> None:
    """Rows `first` to `first + stored` of `o_smem` to `o_gmem` at row `row_start + first` and `column`, in a
    power-of-two ladder of static-size `copy`s (a TMA store's descriptor has static extents; `copied`
    moves the rows through registers where a GPU has no TMA), the largest `most` rows'."""
    start, size = first, most
    while size:
        rung = 1 << int(math.log2(size))
        size //= 2

        @pl.when(stored & rung != 0)
        def _(start=start, rung=rung):
            copy(o_smem.at[pl.ds(start, rung)],
                 o_gmem.at[pl.ds(row_start + start, rung), pl.ds(column, o_smem.shape[1])])

        start = start + (stored & rung)


def copied(source, destination) -> None:
    """`source` (in shared memory) to `destination` (in global memory) through registers: a GPU without
    TMA's `store_rows` copy."""
    destination[...] = source[...]


@dataclasses.dataclass(frozen=True)
class Ring:
    """A ring of `stages` shared-memory slots one operand streams through: a producer loads step `counter`
    into slot `counter % stages` (`load`), a consumer waits for it (`acquire`) and gives it back
    (`release`). The counter runs across a persistent kernel's tiles, so a tile's first loads reuse the
    slots the last tile left, and each tile may take its own number of steps (a causal tile's trip range):
    a slot is free for its first fill and, after, once every consumer released it; the producer drains the
    ring after its last load (`drain`). Allocated with `ring`, its slot count the allocation's own."""
    smem: Ref
    full: state_types.AbstractRef
    consumed: state_types.AbstractRef

    @property
    def stages(self) -> int:
        """The ring's slots: its allocation's leading dimension."""
        return int(self.smem.shape[0])

    def slot(self, counter):
        """The slot step `counter` (traced, or a Python int) uses."""
        return counter % self.stages

    def load(self, counter, view) -> None:
        """The producer's load of step `counter` from `view`, once its slot is free."""
        slot = self.slot(counter)
        pl.when(counter >= self.stages)(lambda: plgpu.barrier_wait(self.consumed.at[slot]))
        plgpu.copy_gmem_to_smem(view, self.smem.at[slot], self.full.at[slot])

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


def ring(block: tuple, dtype, stages: int, *, consumers: int = 1) -> dict:
    """A `Ring`'s allocations for `pl.run_scoped`: `stages` slots of `block` in `dtype`, a load barrier
    each, and a release barrier each that `consumers` arrive at. `Ring(**allocated)` once scoped."""
    return {'smem': plgpu.SMEM((stages, *block), dtype),
            'full': plgpu.Barrier(num_barriers=stages),
            'consumed': plgpu.Barrier(num_arrivals=consumers, num_barriers=stages)}


PRODUCER_REGISTERS = 40
"""The registers a thread of a cooperative block's producer warpgroup holds: it only issues loads."""

THREADS = 128
"""The threads of a role: a warpgroup's, each family's compute and producer roles alike."""

SMEM_ALIGNMENT = 1024
"""The bytes each shared-memory allocation of a Mosaic GPU kernel starts at a multiple of (its lowering's
`SMEM_ALIGNMENT`)."""

BARRIER_BYTES = 8
"""A barrier's shared memory."""


def allocated(*sizes: int) -> int:
    """The shared memory allocations of `sizes` bytes take, each rounded up to `SMEM_ALIGNMENT`."""
    return sum(-(-size // SMEM_ALIGNMENT) * SMEM_ALIGNMENT for size in sizes)

SWIZZLE_BYTES = 128
"""The widest swizzle a shared-memory block's rows are staged at, which an MMA reads conflict-free."""

NARROWEST_SWIZZLE_BYTES = 32
"""The narrowest swizzle Mosaic GPU stages a block's rows at (`plgpu.find_swizzle`'s least)."""


def live_registers(family: "Family", carried, *phases) -> int:
    """The registers a consumer thread of `family` holds over a live set: the tiles it carries across its
    walk (`carried`, each `(rows, columns, dtype)`: its accumulators, a running statistic) and the largest
    of its `phases`' tiles held at once (a step's operand fragments, a rewrite's operands), each tile's
    32-bit words spread over a role's threads, and what it holds beside them (`Family.held_registers`).
    The one register estimate every Mosaic template's launches state (`base.Template.registers`), which
    a legal config holds within a thread's budget (`base.Template.legal`)."""
    def words(tiles) -> float:
        return sum(rows * columns * jnp.dtype(dtype).itemsize / 4 for rows, columns, dtype in tiles)

    phase = max((words(tiles) for tiles in phases), default=0.0)
    return int(-(-(words(carried) + phase) // THREADS)) + family.held_registers


def tile_registers(family: "Family", consumers: int, block: tuple[int, int, int], dtype) -> int:
    """The registers a consumer thread holds at a `(rows, columns, depth)` GEMM tile of `consumers` roles
    of `family` (`live_registers`): its share of the fp32 accumulator carried, and a step's operand
    fragments where its products read registers (`Family.operands_in_registers`; the right operand's in
    each of its replicating warps: `lhs_copies`, `rhs_copies`), a sub-step's where it loads them so
    (`Family.fragment_depth`)."""
    rows, columns, depth = block
    share, depth = rows // consumers, family.fragment_depth(depth)
    fragments = ([(share * family.lhs_copies, depth, dtype), (depth, columns * family.rhs_copies, dtype)]
                 if family.operands_in_registers else [])
    return live_registers(family, [(share, columns, jnp.float32)], fragments)


def held_rows(family: "Family", consumers: int, columns: int) -> int:
    """The most rows, in whole roles of `consumers` consumers of `family` (`Family.role_rows` each), whose
    fp32 accumulator of `columns` columns their threads hold beside what they hold besides
    (`Family.held_registers`): the rows a derived tile's registers bound (docs/design/kernel-templates.md
    §7). A family whose products read registers holds a step's fragments too (`tile_registers`), which
    its derivation checks at each step."""
    share = (family.registers(consumers) - family.held_registers) * THREADS // columns
    return share // family.role_rows * family.role_rows * consumers


def deepest[C](make: Callable[[int], C], holds: Callable[[C], bool], most: int) -> C | None:
    """`make(stages)` at the most stages, from `most` down to two (one loading beside one consumed), that
    `holds` (a launch's legality: its ring within a block's shared memory); None where none does. A
    derived ring is the deepest a block holds (CUTLASS's StageCountAutoCarveout's rule: the block's shared
    memory less what else it stages, over a stage's bytes)."""
    return next((made for made in map(make, range(most, 1, -1)) if holds(made)), None)


def band(multiprocessors: int, block_m: int, block_n: int, column_tiles: int,
         group_row_tiles: int | None = None) -> int:
    """The planar snake's column tiles a band for a wave of one program a multiprocessor sharing its
    blocks in the L2 (`grid_block_n`): the power of two, at most `column_tiles`, nearest below the band
    that reads each operand fewest times from the device. The rows' and the columns' blocks balanced
    (a wave as near a square of output tiles as their sides allow: √(programs · rows / columns)) or,
    where the rows split into groups of `group_row_tiles` row tiles each, wider while a wave still spans
    each group's row tiles whole (programs over a group's row tiles), whose columns' blocks it then
    reads once a group."""
    balanced = math.sqrt(multiprocessors * block_m / block_n)
    spanning = multiprocessors / group_row_tiles if group_row_tiles else 0.0
    return 1 << (max(1, min(int(max(balanced, spanning)), column_tiles)).bit_length() - 1)


def consumer_registers(consumers: int) -> int:
    """The registers a thread of each of `consumers` consumer warpgroups holds: the multiprocessor's 64K less
    the producer's, shared among them in setmaxnreg's steps of 8 and at most its 256 (two consumers' 232,
    JAX's warp-specialized pipeline's `_compute_registers(40, 2)`)."""
    share = (64 * 1024 // 128 - PRODUCER_REGISTERS) // consumers
    return min(256, share // 8 * 8)


CONSUMER_REGISTERS = consumer_registers(2)
"""Two consumer warpgroups' registers a thread (`consumer_registers`)."""


def roles(wg, consumers: int, consumer, producer) -> None:
    """Each warpgroup's role in a cooperative block: the `consumers` first run `consumer(wg)` with their
    register budget raised to their share (`consumer_registers`), the last runs `producer()` with its
    lowered to `PRODUCER_REGISTERS`."""
    @pl.when(wg < consumers)
    def _():
        plgpu.set_max_registers(consumer_registers(consumers), action="increase")
        consumer(wg)

    @pl.when(wg == consumers)
    def _():
        plgpu.set_max_registers(PRODUCER_REGISTERS, action="decrease")
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
    held_registers: int = 40
    """The registers a consumer thread holds beside its accumulator and operand fragments (addresses,
    indices, the walk's carry): CUTLASS's cooperative sm90 kernel's 168 less its accumulator's 128."""
    operands_in_registers: bool
    """Whether a product reads its operands from registers a consumer loads them into (mma.sync's), where
    warpgroup MMA reads both from shared memory."""
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
        """The depth of a step's operand fragments a consumer holds at once, where its products read
        registers (`operands_in_registers`): the step's, unless the family loads them a sub-step at a
        time."""
        return depth
    widest_product: int | None = None
    """The most columns one product spans (warpgroup MMA's 256, wgmma's widest N), which a tile's columns
    are at most; None where a tile's columns are many products' (mma.sync's 8 each), held by its
    registers alone (`tile_registers`)."""
    copy_extent: int | None = None
    """The most elements one asynchronous copy moves along a dimension of its box after the copy's tiling
    (TMA's, 256): an operand block tiled in 8 rows by a swizzled row stays far below it at any tile its
    registers hold, and a one-dimensional side operand's block, untiled, is at most it
    (`grouped_matmul._Mosaic.fitting`); None where cp.async moves a block (mma.sync's)."""
    most_shared: int
    """The most shared memory a block of any GPU of the family holds (CUDA's opt-in limit), which bounds
    the stages a generated tile takes; a GPU's own (`generation.shared_budget`) bounds the tiles its
    launches keep."""

    @abc.abstractmethod
    def registers(self, consumers: int) -> int:
        """The registers a thread of each of `consumers` consumer roles may hold."""

    @abc.abstractmethod
    def column_unit(self, dtype) -> int:
        """The columns a tile's width steps by for operands of `dtype`: the swizzle a block's rows are
        staged in, or the row its stores lay over a role's threads."""

    @abc.abstractmethod
    def depths(self, dtype) -> tuple[int, ...]:
        """The depths a step may take for operands of `dtype`, widest first: the swizzles its products
        read a block's rows at."""

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
               interpret: bool, approx_math: bool = True, name: str | None = None) -> Callable:
        """`entry` launched over `grid`, `consumers` compute roles a program beside the family's own, with
        `approx_math` the hardware's approximate exponential and reciprocal (`compiler_params`), named
        `name` where given (Mosaic names its kernel and its compile reports by it)."""

    @abc.abstractmethod
    def scoped(self, body, *allocations, streams) -> None:
        """`body(*refs, streams=...)` with `allocations` scoped across the program's roles and its streams'
        (`streams`)."""

    @abc.abstractmethod
    def roles(self, consumers: int, consumer, producer) -> None:
        """Each role of a program: `consumer(index)` the compute roles', `producer()` the loads' where the
        family streams from a role of its own."""

    @abc.abstractmethod
    def streams(self, blocks: dict, stages: int, consumers: int) -> Allocated:
        """The scoped allocations of a stream of each `blocks[name]` `(shape, dtype)` in `stages` slots read
        by `consumers` (`opened` once scoped)."""

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
        column, most)`. The family owns the loads, the step counter across tiles, the streams' drain, the
        outputs' staging and the stores' completion."""

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

    def output_columns(self, columns: int, dtypes) -> int:
        """The columns of a `columns`-wide output tile a consumer stages in shared memory at once
        (`persist`'s stores): the whole tile's, where a family stages it whole."""
        del dtypes
        return columns

    def stores(self, dtype) -> bool:
        """Whether a tile of `dtype` in the accumulator's layout reaches global memory (`persist`'s stores,
        `stored`): any dtype, unless the family's path to global memory refuses it."""
        del dtype
        return True

    def published(self) -> None:
        """A consumer's writes to shared memory made visible to the readers that read it asynchronously
        (warpgroup MMA, TMA), before any of them reads it: a family whose products read shared memory
        through its consumers' own loads (mma.sync's) has nothing to publish."""
        plgpu.commit_smem()

    @abc.abstractmethod
    def stored(self, parts, views, *, smem) -> None:
        """Each of `parts` written whole to its view of `views`, through `smem` where the family stores so."""

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

    @abc.abstractmethod
    def register_products(self, acc: Accumulator, parts, rhs, interpret: bool) -> Accumulator:
        """`acc` with `parts`' products with `rhs` issued into it in place, each part's left operand in
        registers."""

    @abc.abstractmethod
    def staged_products(self, acc: Accumulator, parts, rhs, buffers, interpret: bool, *,
                        transposed: bool = False) -> Accumulator:
        """`register_products` of `parts` each staged in `buffers` first (its transpose the left operand
        where `transposed`)."""

    @abc.abstractmethod
    def rescaled_products(self, acc: Accumulator, scale, parts, rhs, interpret: bool) -> Accumulator:
        """`acc` taken by `scale(value)` in place, then `register_products` of `parts` with `rhs` into it."""

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
    warps (its compiler parameters' `reduction_scratch_bytes`): none where a reduction takes the consumer's
    own (`column_sums`)."""

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
    copy_extent = 256

    def registers(self, consumers: int) -> int:
        return consumer_registers(consumers)

    def column_unit(self, dtype) -> int:
        return SWIZZLE_BYTES // jnp.dtype(dtype).itemsize

    def depths(self, dtype) -> tuple[int, ...]:
        return (SWIZZLE_BYTES // jnp.dtype(dtype).itemsize,)

    def warpgroups(self, consumers: int) -> int:
        return consumers + 1

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None) -> Callable:
        return plgpu.kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names,
                            num_threads=consumers + 1, thread_name="wg",
                            compiler_params=compiler_params(approx_math=approx_math),
                            interpret=interpreted(interpret), kernel_name=name)

    def scoped(self, body, *allocations, streams) -> None:
        pl.run_scoped(lambda *refs, **rings: body(*refs, streams=rings), *allocations, **streams,
                      collective_axes="wg")

    def roles(self, consumers: int, consumer, producer) -> None:
        roles(lax.axis_index("wg"), consumers, consumer, producer)

    def streams(self, blocks: dict, stages: int, consumers: int) -> dict:
        return {name: ring(shape, dtype, stages, consumers=consumers)
                for name, (shape, dtype) in blocks.items()}

    def opened(self, allocated: dict, views: dict | None = None) -> dict[str, Ring]:
        del views
        return {name: Ring(**scoped) for name, scoped in allocated.items()}

    def stream(self, streams, views: dict, lo, hi) -> None:
        rings = tuple(streams.values())

        def tiles(index):
            return tuple(viewed(views[name], ring.smem.shape[1:], index)
                         for name, ring in zip(streams, rings, strict=True))

        streamed(rings, tiles, lo, hi)

    def output_columns(self, columns: int, dtypes) -> int:
        # Two chunk buffers, not the whole tile (`output_staging`).
        return math.prod(output_staging(columns, dtypes))

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, consumer: Callable, *,
                consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0) -> None:
        # The producer loads every tile's steps into the rings at a counter running across the tiles, so
        # a tile's first loads go out while the consumers store the last one; each consumer stages its
        # share of each output in chunks (`output_staging`: two buffers of `staged_columns`) and stores each
        # by TMA, waiting only for a buffer's last store to read it before it writes it again.
        rings = self.opened(streams)
        staging = [plgpu.SMEM((consumers, output_staging(shape[1], (dtype,))[1], shape[0],
                               output_staging(shape[1], (dtype,))[0]), dtype)
                   for shape, dtype in outputs]

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
            def run(index):
                def store(output, value, ref, place):
                    smem = smems[output].at[index]
                    buffers, width = smem.shape[0], smem.shape[-1]
                    row_start, first, stored, column, most = place
                    for chunk in range(value.shape[1] // width):
                        if chunk == 0 or chunk >= buffers:
                            # The buffer's last store (the last tile's, at its first chunk) has read it.
                            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                        buffer = smem.at[chunk % buffers]
                        buffer[...] = value[:, chunk * width:(chunk + 1) * width].astype(smem.dtype)
                        plgpu.commit_smem()
                        store_rows(buffer, ref, row_start, first, stored, column + chunk * width, most)

                @plgpu.nd_loop(tiles, collective_axes=axis, init_carry=jnp.int32(0))
                def _(loop: plgpu.NDLoopInfo, carry):
                    counter = carry  # the step every ring is at
                    active, steps, _, tile = schedule(loop.index)

                    def walk(step, start, shapes, *, initial=None):
                        def stepped(at, slots, carry, accumulators):
                            return step(at - counter, slots, carry, accumulators)

                        return self.walk(rings, ((stepped, counter, counter + steps),), start, shapes,
                                         interpret, in_flight=in_flight, initial=initial)

                    pl.when(active)(lambda: consumer(index, tile, walk, store, active=True))
                    pl.when(~active)(lambda: consumer(index, tile, unwalked(self), store, active=False))
                    return counter + jnp.where(active, steps, 0)

                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

            return run

        pl.run_scoped(lambda *smems: self.roles(consumers, consumed(smems), producer), *staging,
                      collective_axes="wg")

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None
                ) -> tuple[C, tuple[jax.Array, ...]]:
        if in_flight not in (0, 1):
            raise ValueError(f"a walk keeps no step's products or one step's in flight, not {in_flight}")

        def slotted(step, lo):
            def run(index, carry, accumulators):
                slots = Slots(streams, index - first)
                carry, accumulators = step(index, slots, carry, accumulators)
                if not in_flight:
                    # The step's products done before its slots are given back.
                    self.settled()
                    slots.done()
                    return carry, accumulators
                # The step's products run on while the next step's operands are waited for: the previous
                # step's, done now, give back its slots.
                slots.held()
                self.settled(in_flight)
                pl.when(index > lo)(lambda: released(streams, index - 1 - first))
                return carry, accumulators

            return run

        walked = self.accumulated(tuple((slotted(step, lo), lo, hi) for step, lo, hi in segments), start,
                                  shapes, interpret, initial)
        if in_flight:
            # Each segment's last step, its products done by the walk's end.
            for _, lo, hi in segments:
                pl.when(hi > lo)(lambda hi=hi: released(streams, hi - 1 - first))
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

    def stored(self, parts, views, *, smem) -> None:
        # An add still reading `smem` (`added`) done first.
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        stored(parts, views, smem=smem)

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

    def held(self, ref: Ref) -> Operand:
        return ref


def unscaled(parts) -> None:
    """A product's parts issued in place take no power of two back (bf16's, `cotangent_parts`)."""
    if any(exponent is not None for _, exponent in parts):
        raise NotImplementedError("a scaled part's product is no in-place accumulation (fp16's)")
