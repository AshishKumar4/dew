"""The templates' Mosaic GPU resources, one owner for every template (attention, GEMM, recurrence): what a
program asks of a multiprocessor and what a family's registers and shared memory hold. A program's shared
memory, from its allocations by lifetime (`Scope`, `shared_bytes`, `allocated`); a consumer's registers over
its live set (`live_registers`, `tile_registers`) and the budgets its roles run at (`consumer_registers`,
`PRODUCER_REGISTERS`); the capacity those give a derivation (`held_area`, `held_rows`, `program_shared`); and
the derivation's shared steps (`deepest`, `band`). Compiled kernels' own reports are the authority over these
estimates (`report`)."""

import dataclasses
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import cost

if TYPE_CHECKING:
    from dew.nn.kernels.templates.mosaic import Family


PRODUCER_REGISTERS = 40
"""The registers a thread of a cooperative block's producer warpgroup holds: it only issues loads."""


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


def tile_registers(family: "Family", consumers: int, block: tuple[int, int, int], dtype, *,
                   stored=()) -> int:
    """The registers a consumer thread holds at a `(rows, columns, depth)` GEMM tile of `consumers` roles
    of `family` (`live_registers`): its share of the fp32 accumulator carried, and a step's operand
    fragments where its products read registers (`Family.operands_in_registers`; the right operand's in
    each of its replicating warps: `lhs_copies`, `rhs_copies`), the right one's a sub-step deep where
    it loads it so (`Family.fragment_depth`), a transposed one's too: the input gradient needs no more
    than the forward at any mma.sync tile measured (GEMMMODEL4/5: 104-120 against 112-128 at 64 x 128 x 32,
    184-216 against 184-224 at 128 x 128 x 32). Where the family copies its outputs out through its
    registers (`Family.stores_through_registers`), its epilogue holds one copy's rows (`copied_rows`) in the
    widest of the `stored` dtypes beside the accumulator."""
    rows, columns, depth = block
    share, sub = rows // consumers, family.fragment_depth(depth)
    fragments = ([(share * family.lhs_copies, depth, dtype), (sub, columns * family.rhs_copies, dtype)]
                 if family.operands_in_registers else [])
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
    `cost.RESERVED_SHARED`) split between them, less each one's own reservation."""
    return (block + cost.RESERVED_SHARED) // programs - cost.RESERVED_SHARED


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


def deepest[C](make: Callable[[int], C], holds: Callable[[C], bool], most: int) -> C | None:
    """`make(stages)` at the most stages, from `most` down to two (one loading beside one consumed), that
    `holds` (a launch's legality: its ring within a block's shared memory); None where none does. A
    derived ring is the deepest a block holds (CUTLASS's StageCountAutoCarveout's rule: the block's shared
    memory less what else it stages, over a stage's bytes)."""
    return next((made for made in map(make, range(most, 1, -1)) if holds(made)), None)


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


def consumer_registers(consumers: int) -> int:
    """The registers a thread of each of `consumers` consumer warpgroups holds: the multiprocessor's 64K less
    the producer's, shared among them in setmaxnreg's steps of 8 and at most its 256 (two consumers' 232,
    JAX's warp-specialized pipeline's `_compute_registers(40, 2)`)."""
    share = (64 * 1024 // 128 - PRODUCER_REGISTERS) // consumers
    return min(256, share // 8 * 8)


CONSUMER_REGISTERS = consumer_registers(2)
"""Two consumer warpgroups' registers a thread (`consumer_registers`)."""

@dataclasses.dataclass(frozen=True)
class Scope:
    """A program's shared-memory allocations by lifetime, as its kernel scopes them (`shared_bytes`): those
    live for the whole `program` (arrays, barriers, a ring's: `plgpu.SMEM`, `plgpu.Barrier`,
    `plgpu.ClusterBarrier`, nested in sequences and mappings); each walk's own, one after another (`walks`:
    an `emit_pipeline`'s slots); tiles scoped `after` the walks, each `(shape, dtype)`; and whether the
    program `reduces` across warps, reserving its family's scratch."""
    program: tuple = ()
    walks: tuple = ()
    after: tuple = ()
    reduces: bool = False


def _count(barriers) -> int:
    """The barriers an allocation's `num_barriers` (an int, or a shape of them) states."""
    return math.prod(barriers) if isinstance(barriers, tuple) else int(barriers)


def shared_bytes(family: "Family", scope: Scope) -> int:
    """The shared memory a program of `scope` asks for, as Mosaic GPU's resource estimate counts it: every
    array rounded up to `SMEM_ALIGNMENT` (`allocated`), the walks the largest of them, the tiles scoped after
    the walks the larger of them and the walks on a family that scopes its walks apart
    (`Family.walks_scoped`) and beside the program's otherwise, the family's reduction scratch where the
    program reduces (`Family.reduction_scratch`), then every barrier (`BARRIER_BYTES`)."""
    def barrier(leaf) -> bool:
        return isinstance(leaf, (plgpu.Barrier, plgpu.ClusterBarrier))

    def arrays(tree) -> list[int]:
        return [math.prod(leaf.shape) * jnp.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(tree)
                if not barrier(leaf)]

    barriers = sum(_count(leaf.num_barriers) for tree in (scope.program, scope.walks)
                   for leaf in jax.tree.leaves(tree) if barrier(leaf))
    sizes = arrays(scope.program)
    walks = [allocated(*arrays(walk)) for walk in scope.walks]
    after = [math.prod(shape) * jnp.dtype(dtype).itemsize for shape, dtype in scope.after]
    if family.walks_scoped:
        walks += [allocated(size) for size in after]
    else:
        sizes += after
    scratch = [family.reduction_scratch] if scope.reduces and family.reduction_scratch else []
    return allocated(*sizes, *scratch) + max([0, *walks]) + BARRIER_BYTES * barriers


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
    most: int = 8


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    """A launch's plan: its ring depth, the programs a multiprocessor runs of it, and its shared bytes."""
    stages: int
    programs: int
    shared: int


def plan(launches, family: "Family", target, *, programs: int | None = None) -> tuple[LaunchPlan, ...] | None:
    """Each of `launches`' plan on `target` (a GPU's, or the one a CPU process states), or None where any
    launch holds at no plan: within its family's registers (`Family.registers`), the most programs a
    multiprocessor holds of it (`Family.programs`, or `programs` where the caller fixes the residency)
    whose share of its shared memory (`program_shared`) holds the launch at its `floor`, then the deepest
    ring that share holds at that residency, so stages never trade away a program. Each launch plans at
    its own residency and depth; a template's legality is that every launch holds."""
    from dew.nn.kernels import generation

    block = generation.shared_budget(target)
    plans = []
    for launch in launches:
        if launch.registers > family.registers(launch.consumers):
            return None
        residencies = range(family.programs(launch.consumers), 0, -1) if programs is None else (programs,)
        found = None
        for held in residencies:
            share = program_shared(block, held)
            if shared_bytes(family, launch.scope(launch.floor)) > share:
                continue
            stages = launch.floor
            while stages < launch.most and shared_bytes(family, launch.scope(stages + 1)) <= share:
                stages += 1
            found = LaunchPlan(stages, held, shared_bytes(family, launch.scope(stages)))
            break
        if found is None:
            return None
        plans.append(found)
    return tuple(plans)
