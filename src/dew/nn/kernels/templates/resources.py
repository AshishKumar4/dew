"""The templates' Mosaic GPU resources, one owner for every template (attention, GEMM, recurrence): what a
program asks of a multiprocessor and what a family's registers and shared memory hold. A program's shared
memory, from its allocations by lifetime (`Scope`, `shared_bytes`, `allocated`); a consumer's registers over
its live set (`live_registers`, `tile_registers`) and the budgets its roles run at (`consumer_registers`,
`producer_floor`); the capacity those give a derivation (`held_area`, `held_rows`, `program_shared`); and
the derivation's shared steps (`deepest`, `band`). Compiled kernels' own reports are the authority over these
estimates (`report`)."""

import contextlib
import contextvars
import dataclasses
import math
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation

if TYPE_CHECKING:
    from dew.nn.kernels.generation import GpuTarget
    from dew.nn.kernels.templates.mosaic import Family


CONTROL_REGISTERS = 16
"""The registers a consumer thread keeps for its walk's control (indices, barrier phases, addresses) beside
its accumulators where no step's operand descriptors are live: in an epilogue, or between a step's scoped
products. The relocatable compile's needs at sm90 (FS-K01 model2 and CAUSE-VECTOR, ptxas's least spill-free
cap): a plain mainloop the accumulator and 32 (its step's descriptors too), exact sums two accumulators and
16, an epilogue with a column vector the accumulator, the vector and 16."""

def facts() -> generation.Multiprocessor:
    """The resources of the multiprocessor the launches derive for (`generation.Multiprocessor`): the
    context's target's, an assumed one's or the device's, which every function here reads."""
    return generation.gpu_target().multiprocessor


def producer_floor() -> int:
    """The registers a thread of a cooperative block's producer warpgroup lowers to (`set_max_registers`):
    setmaxnreg's least (`Multiprocessor.register_raises`, 24), beside two consumers' raise 240 (FlashAttention
    3's). A producer's rule is the least of setmaxnreg's steps that holds its issue loop's live set, and every
    family's holds at the least: ptxas spills nothing at 24 in the attention forward's 154 kernels and
    backward's 17 and the recurrence's 75 under 12.9, and in the D128 forward's 144, the training and natural
    forms among them, under 13.4 (row 3639's scans, each kernel's decrease rewritten to 24 through 96). A
    producer that outgrew it would spill, which the CPU compile gate refuses (tests/kernel_compile.py's
    `mosaic_failures`)."""
    return facts().raises().lowest


THREADS = 128
"""The threads of a role: a warpgroup's, each family's compute and producer roles alike."""

COPIED_REGISTERS = 32
"""The registers a thread holds for one copy of an output's rows out through registers (no TMA store before
Hopper: `templates.mosaic.copied`), which bounds its rows (`copied_rows`): a whole 128 x 128 fp32 tile's 128
beside the next tile's accumulator spilled on sm86 and sm89 at some shapes and held at others (DECIDE3)."""


def copied_rows(columns: int, dtype) -> int:
    """The most rows of `columns` columns of `dtype` one copy through registers moves (`COPIED_REGISTERS` a
    thread of a role's): a power of two, at least one."""
    rows = COPIED_REGISTERS * 4 * THREADS // (columns * jnp.dtype(dtype).itemsize)
    return 1 << max(0, rows.bit_length() - 1)


SMEM_ALIGNMENT = 1024
"""The bytes each shared-memory allocation of a Mosaic GPU kernel starts at a multiple of (its lowering's
`SMEM_ALIGNMENT`)."""


BARRIER_BYTES = 8
"""A barrier's shared memory."""


def allocated(*sizes: int) -> int:
    """The shared memory allocations of `sizes` bytes take, each rounded up to `SMEM_ALIGNMENT`."""
    return sum(-(-size // SMEM_ALIGNMENT) * SMEM_ALIGNMENT for size in sizes)


def live_registers(family: "Family", carried, *phases, addressed=()) -> int:
    """The registers a consumer thread of `family` holds over a live set: the tiles it carries across its
    walk (`carried`, each `(rows, columns, dtype)`: its accumulators, a running statistic) and the largest
    of its `phases`' tiles held at once (a step's operand fragments, a rewrite's operands), each tile's
    32-bit words spread over a role's threads, and what it holds beside them: its scalars and its
    addresses into the shared tiles it reads and writes itself (`addressed`, each `(shape, dtype)`:
    `Family.scalar_registers`).
    The one register estimate every Mosaic template's launches state (`base.Template.registers`), which
    a legal config holds within a thread's budget (`base.Template.legal`)."""
    def words(tiles) -> float:
        return sum(rows * columns * jnp.dtype(dtype).itemsize / 4 for rows, columns, dtype in tiles)

    phase = max((words(tiles) for tiles in phases), default=0.0)
    return family.scalar_registers(addressed) + int(-(-(words(carried) + phase) // THREADS))


def operand_fragments(family: "Family", rows: int, depth: int, columns: int, lhs, rhs, *, lhs_parts: int = 1,
                      rhs_parts: int = 1, rhs_transposed: bool = False,
                      rhs_swizzle: int | None = None) -> list:
    """The operand fragments a product `[rows, depth] @ [depth, columns]` holds at once where its family's
    products read registers (`Family.operands_in_registers`), none where they read shared memory: its left
    operand's loaded depth (`Family.lhs_fragment_depth`), its right one's at its stage's swizzle
    (`Family.rhs_fragment_depth`; `rhs_swizzle` bytes, the family's default where None), each in the warps
    replicating it (`lhs_copies`, `rhs_copies`) and its dtype (`lhs`, `rhs`), repeated for its parts (an
    fp32 operand split into 16-bit parts): GEMM's, recurrence's and attention's products'."""
    if not family.operands_in_registers:
        return []
    return ([(rows * family.lhs_copies, family.lhs_fragment_depth(depth), lhs)] * lhs_parts
            + [(family.rhs_fragment_depth(depth, transposed=rhs_transposed, swizzle=rhs_swizzle),
                columns * family.rhs_copies, rhs)] * rhs_parts)


def tile_registers(family: "Family", consumers: int, block: tuple[int, int, int], dtype, *,
                   stored=(), rhs_transposed: bool = False) -> int:
    """The registers a consumer thread holds at a `(rows, columns, depth)` GEMM tile of `consumers` roles
    of `family` (`live_registers`): its share of the fp32 accumulator carried, and a step's operand
    fragments where its products read registers (`Family.operands_in_registers`; the right operand's in
    each of its replicating warps: `lhs_copies`, `rhs_copies`), the right one's a sub-step deep where
    it loads it so (`Family.rhs_fragment_depth`), or held whole when its transposed storage requires it.
    Where the family copies its outputs out through its registers (`Family.stores_through_registers`),
    its epilogue holds one copy's rows (`copied_rows`) in the widest `stored` dtype beside the accumulator."""
    rows, columns, depth = block
    share = rows // consumers
    fragments = operand_fragments(family, share, depth, columns, dtype, dtype,
                                 rhs_transposed=rhs_transposed)
    accumulator = (share, columns, jnp.float32)
    walk = live_registers(family, [accumulator], fragments)
    if not (family.stores_through_registers and stored):
        return walk
    widest = max(stored, key=lambda stored_dtype: jnp.dtype(stored_dtype).itemsize)
    copied = (min(share, copied_rows(columns, widest)), columns, widest)
    return max(walk, live_registers(family, [accumulator, copied]))


def program_shared(block: int, programs: int) -> int:
    """The shared memory each of `programs` programs a multiprocessor runs at once may stage, on a GPU whose
    block holds `block` bytes: the multiprocessor's (the block's and one reservation,
    `Multiprocessor.reserved_shared`) split between them, less each one's own reservation."""
    reserved = facts().reserved_shared
    return (block + reserved) // programs - reserved


def held_area(family: "Family", consumers: int) -> int:
    """The fp32 accumulator elements `consumers` consumers of `family` hold beside what they hold besides
    (`Family.held_registers`): a derived tile's rows times columns at most (`held_rows` at any columns)."""
    return (family.registers(consumers) - family.held_registers) * THREADS * consumers


def held_rows(family: "Family", consumers: int, columns: int) -> int:
    """The most rows, in whole roles of `consumers` consumers of `family` (`Family.role_rows` each), whose
    fp32 accumulator of `columns` columns their threads hold beside what they hold besides
    (`Family.held_registers`): the rows a derived tile's registers bound (docs/design/kernel-templates.md
    §7). A family whose products read registers holds a step's fragments too (`tile_registers`), which
    its derivation checks at each step."""
    share = held_area(family, consumers) // consumers // columns
    return share // family.role_rows * family.role_rows * consumers


def band(multiprocessors: int, block_m: int, block_n: int, column_tiles: int,
         group_row_tiles: int | None = None, *, cluster: int = 1) -> int:
    """The planar snake's column tiles a band for a wave of one program a multiprocessor sharing its
    blocks in the L2 (`grid_block_n`): the power of two, at most `column_tiles`, nearest below the band
    that reads each operand fewest times from the device. The rows' and the columns' blocks balanced
    (a wave as near a square of output tiles as their sides allow: √(programs · rows / columns)) or,
    where the rows split into groups of `group_row_tiles` row tiles each, wider while a wave still spans
    each group's row tiles whole (programs over a group's row tiles), whose columns' blocks it then
    reads once a group. A cluster of `cluster` multiprocessors is one program, its tile `cluster` row
    tiles (`group_row_tiles` counts a group's clusters' tiles)."""
    programs, rows = multiprocessors // cluster, block_m * cluster
    balanced = math.sqrt(programs * rows / block_n)
    spanning = programs / group_row_tiles if group_row_tiles else 0.0
    return 1 << (max(1, min(int(max(balanced, spanning)), column_tiles)).bit_length() - 1)


RAISES_HONORED = True
"""Whether the GPU's compile of a Mosaic GPU kernel honors its warpgroups' register raises (setmaxnreg): Dew's
JAX build from 0.12.0.1.post43 compiles each kernel whole in its PJRT plugin (Mosaic GPU's nvJitLink off,
jax-ml/jax f365a65e3a, issue 42094), so a raise holds; post42's stock PJRT compiled relocatably, where ptxas
ignores it (ET-HONOR3-h100: the raised D256 forward's C7504, 168 registers and 476 bytes spilled;
SAMECOMPILE1-h100). Under post44 a program launches with its raises' pool: a one-consumer launch raised to
the whole multiprocessor's share held one program where its 127 registers had held two (Nemotron-H 30B's
SSD query/key launch, 161 us at two, 216 to 226 at one, FS-WHY3-h100), so a raise is its residency's share
(`raised_registers`)."""


def entry_registers(warpgroups: int, programs: int = 1) -> int:
    """The registers a thread of a program of `warpgroups` warpgroups launches with, `programs` of it held by
    a multiprocessor at once: their threads' share of its registers (`Multiprocessor.registers`, 64K) in the
    allocator's steps (8), at most the thread's (255): 168 at three warpgroups and one program, 128 at two
    warpgroups and two."""
    held = facts()
    share = held.registers // (programs * warpgroups * THREADS)
    return min(held.thread_registers, share // held.register_step * held.register_step)


def raise_ceiling() -> int:
    """The most a consumer's registers are raised to: the largest of setmaxnreg's steps within the registers a
    thread addresses (`Multiprocessor.register_raises`, `thread_registers`: 248 of 255), not the instruction's
    256, which ptxas plans past those it can use: the 27B gated delta net's packed output launch, one consumer
    at one program, spilled a 64-bit address at a raise of 256 and none at any of 192 to 248
    (FS-LT-OUTPUT-CAPS1, its PTX otherwise the same)."""
    held = facts()
    raises = held.raises()
    return min(raises.highest, held.thread_registers // raises.step * raises.step)


def raised_registers(consumers: int, programs: int = 1, producer: int | None = None) -> int:
    """The registers each of `consumers` consumer warpgroups raises a thread's to (`roles`' setmaxnreg),
    `programs` programs held by a multiprocessor at once: the program's launch pool at that residency less
    the producer's, shared among them in setmaxnreg's steps and at most `raise_ceiling` (one consumer's
    248 at one program and 232 at two; two consumers' 240 at one). The pool is what the program launches with
    (`entry_registers`), which setmaxnreg moves between a program's warps and never past ("The total number
    of registers allocated to all warps in the CTA cannot exceed the initial CTA launch", NVIDIA forums
    302068): four consumers' 640 threads launch at 96 a thread, so they hold 104, where the multiprocessor's
    share would ask 112 and wait for registers no warp gives back. The producer lowers to `producer` a thread
    (`producer_floor`, a model's own where it states another, `RegisterModel.cap`)."""
    producer = producer_floor() if producer is None else producer
    step = facts().raises().step
    pool = entry_registers(consumers + 1, programs) * (consumers + 1) * THREADS
    share = (pool - producer * THREADS) // (consumers * THREADS)
    return min(raise_ceiling(), share // step * step)


def consumer_registers(consumers: int, programs: int = 1) -> int:
    """The registers a thread of each of `consumers` consumer warpgroups holds, `programs` programs a
    multiprocessor: its raise (`raised_registers`) where the GPU's compile honors it (`RAISES_HONORED`), its
    program's entry's otherwise."""
    return (raised_registers(consumers, programs) if RAISES_HONORED
            else entry_registers(consumers + 1, programs))


def producer_registers(consumers: int, programs: int = 1) -> int:
    """The registers a thread of a program's producer beside `consumers` consumer warpgroups holds, `programs`
    programs a multiprocessor: its lowered `producer_floor` where the GPU's compile honors the program's
    raises, its entry's otherwise."""
    return producer_floor() if RAISES_HONORED else entry_registers(consumers + 1, programs)


@dataclasses.dataclass(frozen=True)
class Scope:
    """A program's shared-memory allocations by lifetime, as its kernel scopes them (`shared_bytes`): those
    live for the whole `program` (arrays, barriers, a ring's: `plgpu.SMEM`, `plgpu.Barrier`,
    `plgpu.ClusterBarrier`, nested in sequences and mappings); each walk's own, one after another (`walks`:
    an `emit_pipeline`'s slots); tiles scoped `after` the walks, each `(shape, dtype)`; whether the
    program `reduces` across warps, reserving its family's scratch (`Family.reduction_scratch`); and its
    `reducing_warpgroups`, each reserving its share where its arithmetic reduces implicitly
    (`Family.warpgroup_reduction_scratch`, the kernel's)."""
    program: tuple = ()
    walks: tuple = ()
    after: tuple = ()
    reduces: bool = False
    reducing_warpgroups: int = 0


def _count(barriers) -> int:
    """The barriers an allocation's `num_barriers` (an int, or a shape of them) states."""
    return math.prod(barriers) if isinstance(barriers, tuple) else int(barriers)


def _barrier(leaf) -> bool:
    return isinstance(leaf, (plgpu.Barrier, plgpu.ClusterBarrier))


def scratch_bytes(family: "Family", scope: Scope) -> int:
    """The shared memory a program of `scope` asks for besides its barriers, as Mosaic GPU's resource
    estimate counts it (its `smem_scratch_bytes`): every array rounded up to `SMEM_ALIGNMENT`
    (`allocated`), the walks the largest of them, the tiles scoped after the walks the larger of them and
    the walks on a family that scopes its walks apart (`Family.walks_scoped`) and beside the program's
    otherwise, and the family's reduction scratch where the program reduces (`Family.reduction_scratch`)
    and its warpgroups' shares where they reduce implicitly (`Family.warpgroup_reduction_scratch`)."""
    def arrays(tree) -> list[int]:
        return [math.prod(leaf.shape) * jnp.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(tree)
                if not _barrier(leaf)]

    sizes = arrays(scope.program)
    walks = [allocated(*arrays(walk)) for walk in scope.walks]
    after = [math.prod(shape) * jnp.dtype(dtype).itemsize for shape, dtype in scope.after]
    if family.walks_scoped:
        walks += [allocated(size) for size in after]
    else:
        sizes += after
    scratch = [family.reduction_scratch] if scope.reduces and family.reduction_scratch else []
    if scope.reducing_warpgroups:
        scratch.append(family.warpgroup_reduction_scratch * scope.reducing_warpgroups)
    return allocated(*sizes, *scratch) + max([0, *walks])


def shared_bytes(family: "Family", scope: Scope) -> int:
    """The shared memory a program of `scope` asks for: its scratch (`scratch_bytes`), then every
    barrier (`BARRIER_BYTES`)."""
    barriers = sum(_count(leaf.num_barriers) for tree in (scope.program, scope.walks)
                   for leaf in jax.tree.leaves(tree) if _barrier(leaf))
    return scratch_bytes(family, scope) + BARRIER_BYTES * barriers


MOST_STAGES = 8
"""The deepest ring a launch streams at: deeper rings cover no latency a GPU's loads have (`Launch.most`, the
recurrence's derived ring)."""


@dataclasses.dataclass(frozen=True)
class Launch:
    """One launch a derivation plans (`plan`): its program's allocations at a ring depth (`scope`), its
    consumer roles, the registers a consumer thread of it holds (`live_registers`, `tile_registers`), and
    the fewest and most stages it streams at (`floor`: two keep one stage loading beside the one consumed;
    `most`: deeper rings cover no latency a GPU's loads have)."""
    scope: Callable[[int], Scope]
    consumers: int
    registers: int
    floor: int = 2
    most: int = MOST_STAGES
    tile: object = None
    """The template's tile the launch is at, which a plan's evidence names (`recording`)."""


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    """A launch's plan: its ring depth, the programs a multiprocessor runs of it, and its shared bytes."""
    stages: int
    programs: int
    shared: int


def _register_allocation(registers: int, consumers: int, producer: int, step: int) -> int:
    consumed = -(-registers // step) * step
    produced = -(-producer // step) * step
    return (consumers * consumed + produced) * THREADS


def resident_warpgroups() -> int:
    """The most programs of one warpgroup a multiprocessor holds at once: its resident threads' warpgroups
    within its resident blocks (`Multiprocessor.resident_threads`, `resident_blocks`: sixteen on sm80 to
    sm100, twelve on sm86, sm89 and sm120), the bound `register_programs` searches from."""
    held = facts()
    return min(held.resident_threads // THREADS, held.resident_blocks)


def register_programs(family: "Family", consumers: int, registers: int | None = None) -> int:
    """The most programs of a launch of `consumers` consumer roles a multiprocessor's registers hold at once,
    each consumer thread holding `registers` (None: the family's budget at one program, `Family.registers`):
    the most at whose share (`Family.registers` at that residency, a raise's where the family's roles raise)
    its consumers hold them, beside its producer's where its layout has one. A launch's own need sets its
    residency: two programs of a one-consumer launch at 127 registers ran its walk 1.35 times as fast as one
    (FS-WHY3-h100)."""
    warpgroups = family.warpgroups(consumers)
    need = family.registers(consumers) if registers is None else registers
    for programs in range(resident_warpgroups() // warpgroups, 0, -1):
        producer = producer_registers(consumers, programs) if warpgroups > consumers else 0
        if need <= family.registers(consumers, programs) and programs * _register_allocation(
                need, consumers, producer, family.register_step) <= facts().registers:
            return programs
    return 0


def resident_programs(family: "Family", consumers: int, registers: int | None, target) -> int:
    """The most programs of a launch of `consumers` consumer roles a multiprocessor of `target` holds at
    once, each consumer thread holding `registers`: its registers' (`Family.programs`) within its resident
    threads (an sm86's or sm120's 1536, twelve warpgroups)."""
    threads = target.multiprocessor.resident_threads // (family.warpgroups(consumers) * THREADS)
    return min(family.programs(consumers, registers), threads)


@dataclasses.dataclass(frozen=True)
class RegisterModel:
    """A launch's register decision as its template's model states it, which the compile gate holds the
    GPU's compile to (tests/kernel_compile.py `decisions`): a consumer thread's registers (None where a case
    states only its budget), the family's budget for them at its consumers (`Family.registers`), its
    consumer warpgroups, the most programs its family runs a multiprocessor (`Family.programs`), the
    allocator's step (`Family.register_step`), a producer warpgroup's registers a thread (0 without one) and
    the scratch its planner sized (`scratch_bytes`, None where unstated)."""
    registers: int | None
    budget: int
    consumers: int
    most: int
    step: int
    producer: int
    shared: int | None = None
    family: "Family | None" = dataclasses.field(default=None, compare=False, repr=False)

    def resident(self, registers: int, *, honored: bool = RAISES_HONORED) -> int:
        """Residency at this consumer allocation, bounded by the launch's other resources."""
        producer = self.producer if honored or not self.producer else registers
        allocated = _register_allocation(registers, self.consumers, producer, self.step)
        return min(self.most, facts().registers // allocated) if allocated else self.most

    def cap(self, programs: int, *, honored: bool = RAISES_HONORED) -> int:
        """The consumer cap that leaves room for `programs` copies of this launch: the registers the launch
        gives a consumer thread there, its raise (`raised_registers`) where the compile honors raises, its
        program's entry share otherwise (`entry_registers`, every warpgroup's alike), and without a producer
        its consumers' share. A cap of the multiprocessor's share less the producer's passed a kernel past
        its launch's raise (144 against 136, one consumer at three programs)."""
        if not self.producer:
            return entry_registers(self.consumers, programs)
        if honored:
            return raised_registers(self.consumers, programs, self.producer)
        return entry_registers(self.consumers + 1, programs)

    def record(self) -> dict:
        """The model's numeric facts; its live family supplies rules, not serialized state."""
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)
                if field.name != "family"}


def register_model(family: "Family", consumers: int, registers: int | None, *,
                   shared: int | None = None, target: "GpuTarget | None" = None,
                   tmem: int = 0) -> RegisterModel:
    """`family`'s register decision for a launch of `consumers` whose consumer thread holds `registers`
    (`RegisterModel`): its producer's registers where its layout has one (`Family.warpgroups`), and the most
    programs a multiprocessor holds of it at those registers (`register_programs`), as its plan's."""
    warpgroups = family.warpgroups(consumers)
    if target is None:
        most = register_programs(family, consumers, registers)
    else:
        from dew.nn.kernels import generation

        most = resident_programs(family, consumers, registers, target)
        if shared is not None:
            block = generation.shared_budget(target)
            most = next((count for count in range(most, 0, -1) if shared <= program_shared(block, count)), 0)
        if tmem:
            most = min(most, (target.tmem_columns or 0) // tmem)
    return RegisterModel(registers, family.registers(consumers), consumers, most, family.register_step,
                         producer_registers(consumers) if warpgroups > consumers else 0, shared, family)


@dataclasses.dataclass(frozen=True)
class Asked:
    """A plan a derivation asked for (`plan`, `recording`), its static facts alone (no launch's scope, which
    may close over a trace's values): each launch's tile (`Launch.tile`) and registers, the family it asked
    on, the residency it fixed (None: the most that hold) and the plans, None where a launch held at none."""
    tiles: tuple
    registers: tuple[int, ...]
    family: "Family"
    programs: int | None
    plans: tuple[LaunchPlan, ...] | None


_ASKED: contextvars.ContextVar[list[Asked] | None] = contextvars.ContextVar("asked", default=None)


@contextlib.contextmanager
def recording() -> Iterator[list[Asked]]:
    """Within the block, every plan asked (`plan`) appended to the list it gives, in order: the evidence a
    derivation's choices rest on, which the compile gate holds to the GPU's compile (the tiles it refused
    for their registers alone among them)."""
    asked: list[Asked] = []
    token = _ASKED.set(asked)
    try:
        yield asked
    finally:
        _ASKED.reset(token)


def placed(launches_of: Callable, layouts: tuple["Family", ...], target, *,
           programs: int | None = None) -> tuple["Family", tuple[LaunchPlan, ...]] | None:
    """The layout a program runs on, and its launches' plans: the first of `layouts` (an instruction family's
    bundles in `bundles.layouts`' order, a producer's first) whose plan holds every launch it takes there
    (`launches_of(layout)`, None where the layout takes none of them; `plan`). A program whose consumers'
    registers a producer's layout holds keeps its producer, whose loads overlap no consumer's issue; one
    whose consumers need more than a producer leaves them runs on a layout without it, its first consumer
    issuing the loads (`mosaic.ConsumerFed`). None where no layout holds them."""
    for layout in layouts:
        launches = launches_of(layout)
        planned = None if launches is None else plan(launches, layout, target, programs=programs)
        if planned is not None:
            return layout, planned
    return None


def hidden(family: "Family", consumers: int) -> Callable[[int, int], tuple[int, int]]:
    """The order of a launch's plans, the most latency hidden first, each by its programs a multiprocessor
    runs at once and its stages a program (its ring's depth): residency and depth are one choice.

    A family whose producer issues a program's loads and whose products run asynchronously to its consumers'
    warps (warpgroup MMA beside a TMA producer) hides its loads' latency by the bytes it holds in flight, its
    programs times its stages (a stage's bytes are the launch's own at every plan), ties to fewer programs
    (a larger raise each, its state held once): an H100 ran the 27B gated delta net's states launch one
    program five deep (184 KiB in flight), where its most programs, two one deep (74 KiB), slowed the
    forward 1.4 to 1.5 times (FS-RESIDENT-WINDOW1-h100, FS-INFLIGHT1); Nemotron-H 30B's SSD query/key launch,
    one stage, two programs at 1.35 times one's rate (FS-SSD-RESIDENT1-h100). A family whose consumers issue
    their own loads and synchronous products (mma.sync, no producer) hides their issue latency by its
    resident warps, its programs first, then its ring: an A100 held its states launch at three programs
    where its deepest ring alone held one (SE-family-a100-8c5b5b16a)."""
    if family.warpgroups(consumers) > consumers:
        return lambda programs, stages: (programs * stages, -programs)
    return lambda programs, stages: (programs, stages)


def plan(launches, family: "Family", target, *, programs: int | None = None) -> tuple[LaunchPlan, ...] | None:
    """Each of `launches`' plan on `target` (a GPU's, or the one a CPU process states), or None where any
    launch holds at no plan: within its family's registers (`Family.registers`), of the residencies a
    multiprocessor's registers hold of it at its own registers (`Family.programs`; `programs` where the
    caller fixes the residency, which its registers must hold) whose share of its shared memory
    (`program_shared`) holds the launch at its `floor`, each at the deepest ring that share holds, the one
    that hides the most latency (`hidden`). Each launch plans at its own residency and depth; a template's
    legality is that every launch holds."""
    planned = _planned(launches, family, target, programs)
    if (asked := _ASKED.get()) is not None:
        asked.append(Asked(tuple(launch.tile for launch in launches),
                           tuple(launch.registers for launch in launches), family, programs, planned))
    return planned


def _planned(launches, family: "Family", target, programs: int | None) -> tuple[LaunchPlan, ...] | None:
    """`plan`'s plans."""
    from dew.nn.kernels import generation

    block = generation.shared_budget(target)
    plans = []
    for launch in launches:
        if launch.registers > family.registers(launch.consumers):
            return None
        most = resident_programs(family, launch.consumers, launch.registers, target)
        residencies = range(most, 0, -1) if programs is None else (programs,) if programs <= most else ()
        order = hidden(family, launch.consumers)
        found = None
        for held in residencies:
            share = program_shared(block, held)
            if shared_bytes(family, launch.scope(launch.floor)) > share:
                continue
            stages = launch.floor
            while stages < launch.most and shared_bytes(family, launch.scope(stages + 1)) <= share:
                stages += 1
            if found is None or order(held, stages) > order(found.programs, found.stages):
                found = LaunchPlan(stages, held, shared_bytes(family, launch.scope(stages)))
        if found is None:
            return None
        plans.append(found)
    return tuple(plans)


def widest_swizzle(dtype, columns: int) -> int:
    """The widest swizzle, in bytes, a row of `columns` elements of `dtype` fills (`plgpu.find_swizzle`): the
    one a block staged at its rows' width takes, as layout inference and jax's Ampere examples stage it."""
    return plgpu.find_swizzle(columns * jnp.dtype(dtype).itemsize * 8)


def addresses(tiles) -> int:
    """The registers a thread holds for its own addresses into the shared-memory tiles `tiles` (each
    `(shape, dtype)`) it reads or writes through registers: one for each swizzle pattern its accesses to a
    tile take, the swizzle its rows' width fills (`widest_swizzle`) over the bytes one pattern covers, and
    one for a vector. A 16-bit tile moves in ldmatrix or stmatrix rows, two patterns at a 128-byte swizzle
    and one at 64; a 32-bit one in each lane's own columns (an accumulator's), four at 128 bytes. A ring
    slot's offset is an immediate, so a tile costs this once whatever its stages. A tile read by descriptor
    or TMA costs none: its base is a uniform register (3117's census on the sm90 recurrence and GEMM kernel
    gradient: the SSD, GDN and KDA states launches' staging and landing tiles 5 to 10, the exact normed
    kernel gradient's rewrite 8, its statistics vector 1 and its output staging 4, each a SASS address
    register of one pattern)."""
    def patterns(shape: tuple, dtype) -> int:
        itemsize = jnp.dtype(dtype).itemsize
        # A vector, or rows of no whole 16-byte chunk, stage unswizzled: one pattern.
        if len(shape) < 2 or shape[-1] * itemsize % 16:
            return 1
        return max(1, widest_swizzle(dtype, shape[-1]) // (64 if itemsize == 2 else 32))

    return sum(patterns(tuple(shape), dtype) for shape, dtype in tiles)
