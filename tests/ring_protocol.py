"""A model checker of the Mosaic GPU toolkit's synchronization (docs/design/kernel-templates.md, "The ring
protocol, model-checked"): a launch's kernel traced as the toolkit's own code builds it (`jax.make_jaxpr`, no
GPU), each warpgroup's program (each of its warps', where it runs warp roles: tcgen05's load and product
warps, `plgpu.warp_map`) interpreted concretely into its synchronization events, and every interleaving of
those programs explored under CUDA's mbarrier and asynchronous-copy semantics.

The events are the kernel's own equations: an `mbarrier` wait (`barrier_wait`, at its named phase, or
without one, Mosaic's tracked phase), an arrival (`barrier_arrive`, or a tcgen05 commit's,
`tcgen05_commit_arrive`, asynchronous, once the products its thread issued before it complete), a TMA
load (`copy_gmem_to_smem`: an arrival with its transaction at issue, its bytes landing later, out of order
with other copies), a TMA store (`copy_smem_to_gmem`, reading shared memory until a `wait_smem_to_gmem`
retires its group, which it or a `commit_group` commits), a generic proxy's fence (`commit_smem`), a
warpgroup product (`wgmma`, reading shared memory until `wgmma_wait` retires it) or a tcgen05 one
(`tcgen05_mma`, until its thread's next commit completes), a shared-memory read or write (`get`, `swap`),
and an inlined body's (`inline_mgpu`, lane-level code the jaxpr does not show) as its body's own fields
state them (`INLINED`: a gathered ring's barriers' arrivals, its parity waits, and each load warp's gathers
and the first's expecting arrival, its step's entries read where the step before filled them, and the next
step's filled: `fill`, an asynchronous write landing later, `fill_wait` its thread's wait for its fills,
cp.async's). Index arithmetic, loops and branches are evaluated concretely per
warpgroup and warp (`axis_index`), so each program is its own fixed sequence of events: no control flow
reads shared memory a peer writes.

The semantics assumed (PTX ISA, `mbarrier`, `cp.async.bulk`, `fence.proxy.async`):
- a barrier's phase completes once its arrivals (its `num_arrivals`) and its pending transactions are all in;
  a wait at phase `k` passes once the barrier's phase count's parity is not `k`'s (a phase of `k`'s parity
  completed), one without a phase at the parity its thread tracks, flipped as it passes;
- an arrival releases, a passing wait acquires: everything before the arrivals of a completed phase happens
  before everything after a wait that observes it; a TMA load's bytes are written after its issue and before
  the transaction completes;
- a TMA store reads its shared-memory source asynchronously until a `wait_smem_to_gmem` retires its group
  (the wait's count is the latest committed groups it leaves in flight, an empty one among them; a store no
  commit has closed it does not retire), and
  reads a generic write only after a `commit_smem` (`fence.proxy.async`) that follows it;
- a tcgen05 commit completes asynchronously (`tcgen05.commit`): once the products its thread issued before
  it complete, their operands' reads ending then, it arrives at its barrier, ordered to every thread, its
  own too, only through a wait that observes that arrival;
- the scheduler is weakly fair: a thread that can move eventually does. A state where none can and some
  program has not ended is a deadlock.

The properties checked in every reachable state of every interleaving, within the bounds a test gives:
- no deadlock, and at the end every barrier's phase whole (no arrival left in a phase that never completes),
  every completed phase observed by some thread that waits on the barrier, and by each that waits on it
  naming no phase (a producer's drain: no slot freed while a release is unseen); a thread naming each wait's
  phase may pass over phases its peers observe (ping-pong's consumers, each its own tiles' steps);
- each wait observes the completion it names, or, naming none, the one it is the thread's next wait for on
  that barrier: no barrier runs past the completion a thread's next wait on it awaits (a parity wait's
  aliasing),
  and no thread arrives twice in one phase of a barrier (an early arrival counted toward a phase it does not
  belong to), but a producer's copies of one step, each arriving at its load barrier with its bytes;
- with that, every execution synchronizes alike, so its happens-before order is one: every read of shared
  memory (a load's consumer's, a product's, a store's) happens after the write it reads has landed and
  before the next write to the location lands, and a store's source was fenced after its last generic
  write.

It checks bounded configurations: the claim is coverage of the bounds a test states, not a proof for every
configuration."""

import dataclasses
import itertools
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.state import discharge
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import core as jcore

from dew.nn import derivatives
from dew.nn.kernels.templates import across, gathered
from dew.nn.programs import called

THREAD_AXIS = "wg"
"""The axis a launch's warpgroups are indexed along (`mosaic.kernel`'s `thread_name`)."""


@dataclasses.dataclass(frozen=True)
class Ref:
    """A shared-memory allocation of the kernel: a block of `slots` (its leading axis) or of barriers, each
    `num_arrivals` arrivals a phase, of `shape`; or a global operand, its `value` where the caller states it
    (`programs`' `known`), which its reads give."""

    name: str
    barrier: bool
    num_arrivals: int
    shape: tuple = ()
    value: np.ndarray | None = dataclasses.field(default=None, compare=False)


@dataclasses.dataclass(frozen=True)
class Event:
    """One synchronization event of a warpgroup's program: `kind`, the location it touches (a ref and its
    slot, and a part of the slot where it touches one: a gathered step's tile), and its argument (a wait's
    phase, a product's or a store's sources, a wait's pending count). A `load` arrives at its barrier
    expecting its own copy; an `expect` arrives once expecting `argument` copies, which `copy` events (a load
    that makes no arrival, its landing completing its bytes) issue, from any thread (a gathered ring's step
    spread over the load warps, `gathered.Gathers`)."""

    kind: str
    at: tuple = ()
    argument: object = None


class _Interpreter:
    """A warpgroup's program: the kernel's body evaluated with its index (`thread`), each equation of the
    toolkit's synchronization an `Event` (`events`), every other evaluated concretely (`bind`) or, a value no
    synchronization reads (a tile's data), left unevaluated; where it runs warp roles, each warp's events a
    program of its own (`warps`), its warp's index along the warps' axis (`warp`); and the arrivals a phase
    each barrier its inlined bodies initialize takes (`initialized`, by name)."""

    def __init__(self, thread: int, rare: bool = False):
        self.thread, self.events, self.refs, self.allocated, self.rare = thread, [], 0, [], rare
        self.warp, self.warps, self.initialized = None, [], {}

    def programs(self) -> list[list[Event]]:
        """The warpgroup's program, or its warps' each, where it runs warp roles: those alone, a warpgroup's
        events beside its warps' being no one thread's order."""
        if not self.warps:
            return [self.events]
        if self.events:
            raise ValueError(f"warpgroup {self.thread} has events beside its warps' ({self.events[:3]}...)")
        return self.warps

    def run(self, jaxpr: jcore.Jaxpr, consts, args) -> list:
        env: dict = dict(zip((*jaxpr.constvars, *jaxpr.invars), (*consts, *args), strict=True))

        def read(atom):
            return atom.val if isinstance(atom, jcore.Literal) else env.get(atom)

        for equation in jaxpr.eqns:
            outs = self.equation(equation, [read(atom) for atom in equation.invars])
            env.update(zip(equation.outvars, outs, strict=True))
        return [read(atom) for atom in jaxpr.outvars]

    def equation(self, equation, values) -> list:
        name, params = equation.primitive.name, equation.params
        if name in ("axis_index", "fresh_axis_index"):
            # A grid, thread or warp axis's index, read once or afresh where it is used
            # (`plgpu.fresh_axis_index`).
            axis = params["axis_name"]
            if self.warp is not None and axis is self.warp[0]:
                return [np.int32(self.warp[1])]
            return [np.int32(self.thread if axis == THREAD_AXIS else 0)]
        if name == "layout_cast":
            # A register layout's statement (`plgpu.layout_cast`), its value its operand's.
            return [values[0]]
        if name == "marked_call" and isinstance(params["kind"], derivatives.Rare):
            # A rare span's verdict (a decided walk's, `derivatives.Rare`): the run's own (`programs`'
            # `rare`), each program's alike.
            return [np.bool_(self.rare)]
        if name == "mpmd_map":
            return self.warped(equation, values)
        if name == "inline_mgpu":
            return self.inlined(equation, values)
        # A call's program, read where every reader reads it (`dew.nn.programs.called`).
        inner = called(equation)
        if inner is not None:
            closed = inner if isinstance(inner, jcore.ClosedJaxpr) else jcore.ClosedJaxpr(inner, ())
            return self.run(closed.jaxpr, closed.consts, values)
        if name == "cond":
            branch = params["branches"][int(_governing(values[0], "a branch's index"))]
            return self.run(branch.jaxpr, branch.consts, values[1:])
        if name == "while":
            return self.loop(params, values)
        if name == "scan":
            return self.scanned(params, values)
        if name == "run_scoped":
            return self.scoped(params, values)
        if name in SYNC:
            return SYNC[name](self, equation, values)
        if any(value is None or isinstance(value, Ref) for value in values) or not equation.outvars:
            # A value no synchronization reads (a tile's data, a register budget), left unevaluated.
            return [None] * len(equation.outvars)
        found = equation.primitive.bind(*values, **params)
        return list(found) if equation.primitive.multiple_results else [np.asarray(found)]

    def loop(self, params, values) -> list:
        cond_consts = values[:params["cond_nconsts"]]
        body_consts = values[params["cond_nconsts"]:params["cond_nconsts"] + params["body_nconsts"]]
        carry = values[params["cond_nconsts"] + params["body_nconsts"]:]
        cond, body = params["cond_jaxpr"], params["body_jaxpr"]
        def going(carry) -> bool:
            [more] = self.run(cond.jaxpr, cond.consts, [*cond_consts, *carry])
            return bool(_governing(more, "a loop's trips"))

        while going(carry):
            carry = self.run(body.jaxpr, body.consts, [*body_consts, *carry])
        return carry

    def scanned(self, params, values) -> list:
        # The scan's operands in its three groups (`ft_in`: consts, carry, stacked), each group's count its
        # leaves'.
        num_consts, num_carry, _ = (len(jax.tree.leaves(group, is_leaf=lambda leaf: leaf is None))
                                    for group in params["ft_in"].elts)
        consts = values[:num_consts]
        carry = values[num_consts:num_consts + num_carry]
        stacked = values[num_consts + num_carry:]
        body = params["jaxpr"]
        for index in range(params["length"]):
            sliced = [None if value is None else value[index] for value in stacked]
            carry = self.run(body.jaxpr, body.consts, [*consts, *carry, *sliced])[:num_carry]
        return [*carry, *([None] * (len(body.out_avals) - num_carry))]

    def warped(self, equation, values) -> list:
        """A warpgroup's warps each running its role (`plgpu.warp_map`'s `mpmd_map`: tcgen05's load and
        product warps): each warp's body interpreted with its index, its events a program of its own."""
        (mesh,), (inner,) = equation.params["meshes"], equation.params["jaxprs"]
        ((axis, count),) = mesh.shape.items()
        closed = inner if isinstance(inner, jcore.ClosedJaxpr) else jcore.ClosedJaxpr(inner, ())
        held = self.events
        for warp in range(count):
            self.events, self.warp = [], (axis, warp)
            self.run(closed.jaxpr, closed.consts, values)
            self.warps.append(self.events)
        self.events, self.warp = held, None
        return [None] * len(equation.outvars)

    def inlined(self, equation, values) -> list:
        """An inlined body (`plgpu.inline_mgpu`), whose lane-level code the jaxpr does not show: its events as
        its body's own fields state them (`INLINED`, by the body's type), at its operands' locations; refused
        for a body of a type there is none for, whose waits or arrivals the checker would not see."""
        params = equation.params
        body = params["mgpu_fn"]
        read = INLINED.get(type(body))
        if read is None:
            raise ValueError(f"an inlined body the checker reads no events of ({body!r}): state them, or "
                             "that it has none, in `INLINED`")
        count = params["pytree_args"].num_leaves
        transforms = jax.tree.unflatten(params["pytree_ref_transforms"], values[count:])
        located = [_located(value, transform) if isinstance(value, Ref) else value
                   for value, transform in zip(values[:count], transforms, strict=True)]
        read(self, body, values[:count], located)
        return [None] * len(equation.outvars)

    def scoped(self, params, values) -> list:
        jaxpr = params["jaxpr"]
        allocated = []
        for var in jaxpr.invars:
            aval = var.aval
            inner = aval.inner_aval
            barrier = hasattr(inner.dtype, "num_arrivals")
            self.refs += 1
            allocated.append(Ref(f"{'barrier' if barrier else 'smem'}{self.refs}", barrier=barrier,
                                 num_arrivals=inner.dtype.num_arrivals if barrier else 0,
                                 shape=tuple(inner.shape)))
        self.allocated += allocated
        return self.run(jaxpr, values, allocated)

    def emit(self, kind: str, at: tuple = (), argument=None) -> list:
        self.events.append(Event(kind, at, argument))
        return []


def _slot(ref, treedef, leaves) -> tuple:
    """A transformed ref's location (`_located`), its transforms flattened (`treedef`, `leaves`)."""
    if ref is None:
        return ()
    if treedef is None or (not treedef.num_leaves and not leaves):
        return (ref.name, None)
    return _located(ref, jax.tree_util.tree_unflatten(treedef, list(leaves)))


def _located(ref, transforms) -> tuple:
    """A ref's location through `transforms`: the ref and each leading index that is one value, in order (a
    ring's slot; a consumer's staging buffer and its chunk's buffer within it, `stores.staged_chunks`; a
    one-entry slice of an axis, which keeps it, last: a gathered ring's slot mbarrier, `gathered.Ring`), else
    the ref whole. Two locations share memory where one's indices are a prefix of the other's (`_overlaps`):
    a consumer's two chunk buffers, keyed by its own index alone, read as one buffer, so its double-buffered
    stores raced (base review f86cb5b03's witness at 128 x 256)."""
    found = []

    def whole(index, extent) -> bool:
        return (hasattr(index, "start") and index.start is not None and int(np.asarray(index.start)) == 0
                and index.size == extent)

    for transform in transforms or ():
        if getattr(transform, "indices", None) is None:
            # A transform that moves axes (a transpose, a reshape): later indices are not the base's
            # coordinates, so the location stops at the prefix known (base review 2e4651b2e F3). A tiling's,
            # a swizzle's or a bitcast's changes the trailing axes alone, the leading ones indexed as before.
            if hasattr(transform, "permutation") or "reshape" in type(transform).__name__.lower():
                return (ref.name, *found) if found else (ref.name, None)
            continue
        indices, shape = transform.indices, getattr(transform, "shape", None)
        for axis, index in enumerate(indices):
            if index is None:
                return (ref.name, *found) if found else (ref.name, None)
            if not hasattr(index, "start"):
                # An array index (advanced indexing, ndim > 0) introduces its own output axis, so later
                # transforms do not index the base's next axis: keep the single base coordinate if constant
                # and stop at the known base prefix (row 8978).
                if (one := _one_value(index)) is None:
                    return (ref.name, *found) if found else (ref.name, None)
                found.append(one)
                if np.asarray(index).ndim > 0:
                    return (ref.name, *found)
                continue
            # A slice: the axes from it on whole, the next transform indexes them; else its one entry, if it
            # takes one, the location's last.
            if shape is not None and all(whole(rest, shape[at])
                                         for at, rest in enumerate(indices) if at >= axis):
                break
            if index.size == 1 and index.start is not None:
                found.append(int(np.asarray(index.start)))
            return (ref.name, *found) if found else (ref.name, None)
    return (ref.name, *found) if found else (ref.name, None)


def _one_value(index) -> int | None:
    """An indexer's index where it names one entry of its axis (a value, an array of one value broadcast
    beside a gathered index, or a one-entry slice), else None (a slice of more, or an array of several
    indices: a gathered tile's rows)."""
    if hasattr(index, "start"):
        return int(np.asarray(index.start)) if index.size == 1 and index.start is not None else None
    values = None if index is None else np.unique(np.asarray(index))
    return None if values is None or values.size != 1 else int(values[0])


def _governing(value, what: str):
    """`value`, which governs which events a program has (`what`: a branch's index, a loop's trips, a wait's
    parity, a predicate), refused where the checker leaves it unevaluated (a table's entry in global memory):
    explored at a guess, the program would miss or invent events (row 3588)."""
    if value is None or isinstance(value, Ref):
        raise ValueError(f"{what} reads a value the checker does not evaluate (global memory, a tile's data)")
    return value


def _wait(self: _Interpreter, equation, values) -> list:
    tree = equation.params["transforms_treedef"]
    count = tree.num_leaves
    at = _slot(values[0], tree, values[1:1 + count])
    phase = (int(np.asarray(_governing(values[1 + count], "a wait's phase")))
             if equation.params.get("has_phase") else None)
    return self.emit("wait", at, phase)


def _arrival(tree_name: str, kind: str = "arrive") -> Callable:
    """An arrival at its barrier's slot, an event of `kind`, its transforms' tree the equation's `tree_name`:
    a `barrier_arrive`, or a tcgen05 commit's (`committed`: `tcgen05_commit_arrive`, asynchronous)."""
    def arrive(self: _Interpreter, equation, values) -> list:
        tree = equation.params[tree_name]
        count = tree.num_leaves
        if equation.params.get("has_user_predicate") and not bool(
                np.asarray(_governing(values[1 + count], "an arrival's predicate"))):
            return []
        return self.emit(kind, _slot(values[0], tree, values[1:1 + count]))

    return arrive


def _load(self: _Interpreter, equation, values) -> list:
    params = equation.params
    trees = [params["src_transforms_treedef"], params["dst_transforms_treedef"],
             params["barrier_transforms_treedef"] if params.get("has_barrier") else None]
    counts = [tree.num_leaves if tree is not None else 0 for tree in trees]
    refs = values[:3] if params.get("has_barrier") else (*values[:2], None)
    leaves = values[len([ref for ref in refs if ref is not None]):]
    split = np.cumsum([0, *counts])
    if params.get("has_user_predicate") and not bool(np.asarray(_governing(leaves[split[-1]],
                                                                          "a load's predicate"))):
        return []
    destination = _slot(refs[1], trees[1], leaves[split[1]:split[2]])
    barrier = _slot(refs[2], trees[2], leaves[split[2]:split[3]]) if refs[2] is not None else ()
    return self.emit("load", destination, barrier)


def _store(self: _Interpreter, equation, values) -> list:
    """A TMA store, its argument whether it commits its thread's open store group (`commit_group`: a group of
    it and every store before it no commit closed)."""
    params = equation.params
    tree = params["src_transforms_treedef"]
    count = tree.num_leaves
    return self.emit("store", _slot(values[0], tree, values[2:2 + count]), bool(params["commit_group"]))


def _store_group(self: _Interpreter, equation, values) -> list:
    """A commit of the thread's open store group (`commit_smem_to_gmem_group`): a group, empty where no store
    is open, which a wait's count counts as any other."""
    return self.emit("store_group")


def _store_wait(self: _Interpreter, equation, values) -> list:
    return self.emit("store_wait", (), equation.params["n"])


def _commit(self: _Interpreter, equation, values) -> list:
    return self.emit("commit")


def _access(kind: str, indexed: int) -> Callable:
    """A ref's access, an event of `kind` at the slot its indexers name, the equation's values from `indexed`
    on: a get's after its ref, a swap's after its ref and the value it writes."""
    def access(self: _Interpreter, equation, values) -> list:
        ref = values[0]
        if isinstance(ref, Ref):
            tree = equation.params.get("tree")
            if tree is None:
                tree = equation.params["transforms_treedef"]
            self.emit(kind, _slot(ref, tree, values[indexed:]))
            if kind == "read" and ref.value is not None:
                # A known operand's read, through its transforms as JAX's state discharge reads one.
                transforms = jax.tree_util.tree_unflatten(tree, values[indexed:])
                return [np.asarray(discharge.transform_array(jnp.asarray(ref.value), transforms))]
        return [None] * len(equation.outvars)
    return access


def _product(self: _Interpreter, equation, values) -> list:
    sources = tuple((ref.name, None) for ref in values if isinstance(ref, Ref))
    self.emit("product", (), sources)
    return [None] * len(equation.outvars)


def _product_wait(self: _Interpreter, equation, values) -> list:
    return self.emit("product_wait", (), equation.params["n"])


def _operand_slots(params, lhs, rhs, leaves) -> tuple:
    """A product's shared-memory operands, each at its slot (`_slot`, its transforms' leaves of `leaves`,
    the accumulator's, then `lhs`'s and `rhs`'s, `*_transforms_tree`), an operand in registers none."""
    trees = [params[name] for name in ("acc_transforms_tree", "a_transforms_tree", "b_transforms_tree")]
    split = np.cumsum([0, *(tree.num_leaves if tree is not None else 0 for tree in trees)])
    return tuple(_slot(ref, tree, leaves[split[index]:split[index + 1]])
                 for index, (ref, tree) in enumerate(zip((lhs, rhs), trees[1:], strict=True), start=1)
                 if isinstance(ref, Ref))


def _wgmma_product(self: _Interpreter, equation, values) -> list:
    """A warpgroup MMA into its accumulator's ref (`wgmma_ref`, the toolkit's every Hopper product): its
    shared-memory operands, each at its slot, read asynchronously until a `wgmma_wait` retires its group
    (`product_wait`). Read as `wgmma` alone, the walks' products were no events at all: no slot of theirs
    was checked against the loads that refill it (row 8680's bounded walk showed none)."""
    acc, lhs, rhs = values[:3]
    del acc
    return self.emit("product", (), _operand_slots(equation.params, lhs, rhs, values[3:]))


def _tcgen05_product(self: _Interpreter, equation, values) -> list:
    """A tcgen05 product (`tcgen05_mma`): its operands, each at its slot, read asynchronously until the
    completion of a commit of its thread after it (a `committed` event); its accumulator written in tensor
    memory on that completion. A tensor-memory left operand (`WEIGHTS` over `LOGITS` or `QUERY`) is read as
    filled, its latest overlapping writer before it not a product. Refused where it arrives at a barrier of
    its own once it completes (`arrive`), an asynchronous arrival the checker does not order."""
    params = equation.params
    if params["arrive"]:
        raise ValueError("a tcgen05 product arriving at a barrier as it completes: the checker orders no "
                         "asynchronous arrival")
    acc, lhs, rhs, _ = values[:4]
    leaves = values[4 + 2 * int(params["scaled"]) + int(params["sparse"]):]
    acc_tree = params["acc_transforms_tree"]
    acc_count = acc_tree.num_leaves if acc_tree is not None else 0
    destination = _slot(acc, acc_tree, leaves[:acc_count]) if isinstance(acc, Ref) else ()
    slots = _operand_slots(params, lhs, rhs, leaves)
    if isinstance(lhs, Ref) and equation.invars[1].aval.memory_space == plgpu.TMEM:
        # Its left operand in tensor memory, its thread's or another's stores' (`_tmem_store`): a read of
        # what a store or copy must have written last, landed before it (no store's, or overwritten since by
        # an MMA through an aliased view, a read of nothing).
        self.emit("read", slots[0], "filled")
    return self.emit("product", destination, slots)


_tcgen05_commit = _arrival("barrier_transforms_tree", "committed")
"""A tcgen05 commit (`tcgen05_commit_arrive`): asynchronous, completing once every product its thread issued
before it completes, their reads ending then, and arriving at its barrier then: ordered to a thread, its own
too, only through a wait that observes that arrival."""


def _tmem_copy(self: _Interpreter, equation, values) -> list:
    """`async_copy_smem_to_tmem`: reads its SMEM source asynchronously (retired by `tcgen05_commit_arrive`)
    and writes its TMEM destination (ordered before any following `tcgen05_mma` on the issuing warp)."""
    params = equation.params
    smem_ref, tmem_ref = values[0], values[1]
    smem_tree, tmem_tree = params["smem_tree"], params["tmem_tree"]
    smem_count = smem_tree.num_leaves
    leaves = values[2:]
    if isinstance(tmem_ref, Ref):
        self.emit("write", _slot(tmem_ref, tmem_tree, leaves[smem_count:smem_count + tmem_tree.num_leaves]))
    if isinstance(smem_ref, Ref):
        self.emit("product", (), (_slot(smem_ref, smem_tree, leaves[:smem_count]),))
    return []


def _tmem_store(self: _Interpreter, equation, values) -> list:
    """`async_store_tmem`: registers written into tensor memory at its slot, visible to the tensor core's
    products once its thread's next `commit_tmem` follows (`_tmem_commit`), where the write lands: a turn's
    weights over their logits (`placed.Tcgen05Placed`), published by a commit and then an arrival
    (`Tcgen05Issuer.published`)."""
    ref = values[0]
    if isinstance(ref, Ref):
        self.emit("tmem_store", _slot(ref, equation.params["tree"], values[2:]))
    return []


def _tmem_load(self: _Interpreter, equation, values) -> list:
    """`async_load_tmem`: tensor memory read into registers asynchronously at its slot until its thread's
    next `wait_load_tmem` retires it (`_tmem_load_wait`)."""
    ref = values[0]
    if isinstance(ref, Ref):
        self.emit("tmem_load", _slot(ref, equation.params["tree"], values[1:]))
    return [None] * len(equation.outvars)


def _tmem_load_wait(self: _Interpreter, equation, values) -> list:
    """`wait_load_tmem`: retires every pending `async_load_tmem` read of its thread (`_tmem_load`)."""
    del equation, values
    return self.emit("tmem_load_wait")


def _tmem_commit(self: _Interpreter, equation, values) -> list:
    """`commit_tmem`: its thread's tensor-memory stores before it landed (`_tmem_store`)."""
    del equation, values
    return self.emit("tmem_commit")


SYNC: dict[str, Callable] = {
    "barrier_wait": _wait, "barrier_arrive": _arrival("transforms_treedef"),
    "tcgen05_mma": _tcgen05_product, "tcgen05_commit_arrive": _tcgen05_commit,
    "async_copy_smem_to_tmem": _tmem_copy, "async_store_tmem": _tmem_store, "async_load": _tmem_load,
    "wait_load_tmem": _tmem_load_wait, "commit_tmem": _tmem_commit, "copy_gmem_to_smem": _load,
    "copy_smem_to_gmem": _store, "commit_group": _store_group, "wait_smem_to_gmem": _store_wait,
    "commit_smem": _commit,
    "get": _access("read", 1), "load": _access("read", 1), "swap": _access("write", 2),
    "wgmma": _product, "wgmma_ref": _wgmma_product, "wgmma_wait": _product_wait,
}
"""Each primitive of the toolkit's synchronization and shared memory, as an `Event`."""


def _initialized(self: _Interpreter, body: gathered.Initialized, values, located) -> None:
    (barriers,) = located
    self.initialized[barriers[0]] = body.arrivals


def _waited(self: _Interpreter, body: gathered.Waited, values, located) -> None:
    barrier, rotation = located
    self.emit("wait", barrier, int(np.asarray(_governing(rotation, "a wait's phase"))))


def _gathers(self: _Interpreter, body: gathered.Gathers, values, located) -> None:
    """A load warp's gathers of a step (`gathered.Gathers`): the first warp's arrival expecting every tile's
    copy, and a copy of each tile the warp issues into its part of the slot, each completing its bytes at
    the slot's barrier; read a step ahead (`body.ahead`), its tiles' entries read from the entries' parity
    the step before filled, once its fills are awaited (`gathered.Held.awaited`), a warp's fills an
    asynchronous write of its own; a tile's first step fills its own entries first (`gathered.Held.awaited`);
    the next step's fill is the load's next body (`_filled`).
    The warp's lanes are its one program: the body's warp syncs, which make a lane's fill seen by the
    others, are its own statement, not an event."""
    _, _, slot, barrier, _, *ahead = located
    # The slot's tiles: its leading axis as its allocation stages it (`swizzling.swizzle_transforms`'
    # `TILE_ROWS`-row tiles), the memref's the body reads.
    tiles = values[2].shape[1]
    if body.expects:
        self.emit("expect", barrier, tiles)
    if body.ahead:
        entries = ahead[0]
        parity, held = (int(np.asarray(_governing(value, "an entry's parity"))) for value in ahead[1:])
        # A warp's fills of a step are one cp.async group, awaited together: one write of its parity's
        # entries, its tiles' all (each warp's its own rows: `gathered.Gathers.tiles`).
        warp = body.share[0]
        if not held:
            self.emit("fill", (entries[0], parity, warp))
        self.emit("fill_wait")
        # A read of entries a fill must have written: no fill ordered before it is a read of nothing.
        self.emit("read", (entries[0], parity, warp), "filled")
    for tile in body.tiles(tiles):
        self.emit("copy", (*slot, tile), barrier)


def _unsynchronized(self: _Interpreter, body, values, located) -> None:
    """A body of no synchronization: a register's move between lanes (`across.LaneZero`, a shuffle)."""
    del self, body, values, located


def _scale_smem(self: _Interpreter, body: across.ScaleSmem, values, located) -> None:
    """In-place SMEM query scaling (`across.ScaleSmem`): reads and overwrites the query stage."""
    del body, values
    (slot,) = located
    self.emit("read", slot)
    self.emit("write", slot)


def _filled(self: _Interpreter, body: gathered.Filled, values, located) -> None:
    """A load warp's fill of its next step's entries (`gathered.Filled`, where the walk has a next step):
    one asynchronous write of the parity it fills, its tiles' all, a cp.async group the next step awaits."""
    entries, _, parity = located
    self.emit("fill", (entries[0], int(np.asarray(_governing(parity, "an entry's parity"))), body.share[0]))


INLINED: dict[type, Callable] = {gathered.Initialized: _initialized, gathered.Waited: _waited,
                                 gathered.Gathers: _gathers, gathered.Filled: _filled,
                                 across.LaneZero: _unsynchronized, across.ScaleSmem: _scale_smem}
"""Each inlined body a checked kernel takes (`plgpu.inline_mgpu`'s function, by its type), as the `Event`s
its fields and its operands state: the arrivals a phase of the barriers a gathered ring initializes, its
parity waits at a turn's rotation, and its load warps' gathers. A body of no synchronization (a fence, a
register form) takes a reader that emits nothing, its reason beside it; one of no type here is refused."""


def programs(kernel: Callable, *operands, threads: int, launch: int = 0, rare: bool = False,
             known: Callable[[int, Any], np.ndarray | None] = lambda index, aval: None
             ) -> tuple[list[list[Event]], dict[str, int]]:
    """Each program of `kernel`'s `launch`-th Mosaic GPU launch (one the toolkit built, `mosaic.kernel`, in
    the order its trace makes them: a chained input gradient's flagged walk, then its bounded one) over
    `operands`, of its `threads` warpgroups, its rare spans taken where `rare` (`derivatives.Rare`: a decided
    walk's second): a warpgroup's events, or where it runs warp roles, each of its warps' in turn
    (`_Interpreter.programs`), by warpgroup index; and each barrier's arrivals a phase, by name: a barrier
    allocation's, or a shared `uint64`'s its inlined initialization states (`gathered.Initialized`).
    `known(index, aval)`, a global operand's value where the caller states it (a bounded walk's flags, which
    decide the tiles it walks), else None: its reads give it."""
    closed = jax.make_jaxpr(kernel)(*operands)
    body = list(_bodies(closed.jaxpr))[launch]
    found, arrivals = [], {}
    for thread in range(threads):
        interpreter = _Interpreter(thread, rare)
        interpreter.run(body.jaxpr, body.consts, [
            Ref(f"gmem{index}", barrier=False, num_arrivals=0, value=known(index, variable.aval.inner_aval))
            for index, variable in enumerate(body.jaxpr.invars)])
        found += interpreter.programs()
        arrivals.update({ref.name: ref.num_arrivals for ref in interpreter.allocated if ref.barrier})
        arrivals.update(interpreter.initialized)
    return found, arrivals


def _bodies(jaxpr: jcore.Jaxpr) -> Iterator[jcore.ClosedJaxpr]:
    """The kernel bodies' jaxprs inside a traced program, in its order: each `mpmd_map`'s per-program
    jaxpr."""
    for equation in jaxpr.eqns:
        if equation.primitive.name == "mpmd_map":
            (inner,) = equation.params["jaxprs"]
            yield inner if isinstance(inner, jcore.ClosedJaxpr) else jcore.ClosedJaxpr(inner, ())
            continue
        for value in equation.params.values():
            for inner in value if isinstance(value, (tuple, list)) else (value,):
                nested = inner.jaxpr if isinstance(inner, jcore.ClosedJaxpr) else inner
                if isinstance(nested, jcore.Jaxpr):
                    yield from _bodies(nested)


@dataclasses.dataclass
class Violation(AssertionError):
    """A property the exploration found broken: its class (`kind`: "deadlock", "unfinished" phase,
    "unobserved" phase, "aliased" parity, "early" arrival, "unordered" access, "unfenced" store, "bounds",
    an "unwritten" entries' read),
    what broke (`what`), and the
    interleaving reaching it (`trace`, each move a thread's index and its event's place, or a landing)."""

    kind: str
    what: str
    trace: list

    def __str__(self) -> str:
        shown = "\n".join(f"  {mover}: {event}" for mover, event in self.trace[-24:])
        return f"{self.what}\n{shown}"


@dataclasses.dataclass(frozen=True)
class _Barrier:
    """A barrier slot's state: its completed phases, the arrivals and transactions its current phase still
    awaits, and the threads that arrived in it."""

    phases: int
    pending: int
    transactions: int
    arrived: frozenset


def explore(threads: list[list[Event]], arrivals: dict[str, int], *, limit: int = 2_000_000) -> int:
    """Every interleaving of `threads`' programs (each a warpgroup's or a warp's `Event`s) and their loads'
    landings, each barrier's arrivals a phase by name (`arrivals`): the properties of the module's account
    checked in every reachable state, and the happens-before order of one execution checked for races.
    Returns the states explored; raises `Violation` with its interleaving where a property fails."""
    counts = [_waits(program) for program in threads]
    awaited = [_awaited(program) for program in threads]
    seen: set = set()
    stack: list = [(_State(len(threads)), [])]
    completed = None
    while stack:
        state, trace = stack.pop()
        key = state.key()
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > limit:
            raise Violation("bounds", f"more than {limit} states: the bounds are past the checker's", trace)
        moves = list(state.moves(threads, arrivals, counts, awaited, trace))
        if not moves:
            if not state.ended(threads):
                raise Violation("deadlock", "no warpgroup can move and some has not ended", trace)
            state.whole(arrivals, [_observed(program) for program in threads], trace)
            completed = trace if completed is None else completed
            continue
        stack.extend(moves)
    _races(threads, completed or [], arrivals)
    return len(seen)


def _waits(program: list[Event]) -> list[dict]:
    """For each position of a program, how many waits it has passed on each barrier slot."""
    found, passed = [], {}
    for event in program:
        found.append(dict(passed))
        if event.kind == "wait":
            passed[event.at] = passed.get(event.at, 0) + 1
    found.append(dict(passed))
    return found


def _awaited(program: list[Event]) -> list[dict]:
    """For each position of a program, the completion its next wait on each barrier slot awaits: the one it
    names (its phase's, counted from one), or, naming none, the one past those its waits before observed."""
    found, upcoming = [], {}
    for event in reversed(program):
        found.append(dict(upcoming))
        if event.kind == "wait":
            upcoming[event.at] = event.argument + 1 if event.argument is not None else None
    found.append(dict(upcoming))
    found.reverse()
    # A wait naming no phase awaits the completion past those observed before it.
    observed: dict = {}
    for position, event in enumerate(program):
        found[position] = {at: observed.get(at, 0) + 1 if completion is None else completion
                           for at, completion in found[position].items()}
        if event.kind == "wait":
            observed[event.at] = (observed.get(event.at, 0) + 1 if event.argument is None
                                  else event.argument + 1)
    return found


def _observed(program: list[Event]) -> dict:
    """Each barrier slot a program waits on: the completions its waits observe (counted from one) and whether
    each names its phase."""
    found: dict = {}
    for event in program:
        if event.kind == "wait":
            seen, named = found.get(event.at, (frozenset(), True))
            completion = len(seen) + 1 if event.argument is None else event.argument + 1
            found[event.at] = (seen | {completion}, named and event.argument is not None)
    return found


class _State:
    """An exploration state: each program's position, each barrier slot's state, the loads in flight (each
    its issuing thread, position and barrier), and each thread's tracked parity on the barriers it waits on
    without one."""

    def __init__(self, threads: int):
        self.positions = tuple(0 for _ in range(threads))
        self.barriers: dict = {}
        self.flight: frozenset = frozenset()
        self.tracked: tuple = tuple(frozenset() for _ in range(threads))

    def key(self):
        return self.positions, tuple(sorted(self.barriers.items())), self.flight, self.tracked

    def copy(self) -> "_State":
        made = _State.__new__(_State)
        made.positions, made.barriers, made.flight, made.tracked = (self.positions, dict(self.barriers),
                                                                     self.flight, self.tracked)
        return made

    def barrier(self, at, arrivals) -> _Barrier:
        return self.barriers.get(at) or _Barrier(0, arrivals[at[0]], 0, frozenset())

    def ended(self, threads) -> bool:
        return not self.flight and all(position == len(program)
                                       for position, program in zip(self.positions, threads, strict=True))

    def whole(self, arrivals, observed: list[dict], trace) -> None:
        """The ended state's barriers: each phase complete, each completed phase seen by some thread that
        waits on it, and every one by each that waits on it naming no phase (`observed`, each thread's
        completions observed on each barrier slot, `_observed`)."""
        for at, state in self.barriers.items():
            if state.pending != arrivals[at[0]] or state.transactions:
                left = arrivals[at[0]] - state.pending
                raise Violation("unfinished", f"{at}: {left} arrivals and {state.transactions} transactions "
                                "left in a phase that never completes", trace)
            seen = set()
            for thread, waits in enumerate(observed):
                if at not in waits:
                    continue
                completions, named = waits[at]
                if not named and len(completions) != state.phases:
                    raise Violation("unobserved", f"thread {thread} waited for {len(completions)} of {at}'s "
                                    f"{state.phases} completed phases", trace)
                seen |= completions
            unseen = sorted(set(range(1, state.phases + 1)) - seen)
            if any(at in waits for waits in observed) and unseen:
                raise Violation("unobserved", f"{at}'s completions {unseen} observed by no thread that waits "
                                "on it", trace)

    def arrive(self, at, arrivals, thread, trace, *, transaction: int = 0, count: bool = True) -> None:
        state = self.barrier(at, arrivals)
        # A load's arrival (its expected bytes, `transaction`) may come from one thread as many times as the
        # barrier counts copies (`rings.ring`'s copies, a joined pair's members' every copy, `rings.joined`);
        # any other arrival twice in a phase is an early one.
        if count and not transaction and thread in state.arrived:
            raise Violation("early", f"thread {thread} arrives twice in phase {state.phases} of {at}", trace)
        pending = state.pending - int(count)
        transactions = state.transactions + transaction
        if pending < 0:
            raise Violation("early", f"{at}: more arrivals than its {arrivals[at[0]]} a phase", trace)
        if pending == 0 and transactions == 0:
            self.barriers[at] = _Barrier(state.phases + 1, arrivals[at[0]], 0, frozenset())
        else:
            self.barriers[at] = _Barrier(state.phases, pending, transactions,
                                         state.arrived | ({thread} if count else frozenset()))

    def moves(self, threads, arrivals, counts, awaited, trace) -> Iterator:
        for flight in sorted(self.flight):
            issuer, position, barrier = flight
            made = self.copy()
            made.flight = self.flight - {flight}
            moved = [*trace, ("landed", (issuer, position))]
            if threads[issuer][position].kind == "committed":
                # A commit's completion: its arrival, the issuing thread's.
                made.arrive(barrier, arrivals, issuer, moved)
            elif barrier:
                made.arrive(barrier, arrivals, issuer, moved, transaction=-1, count=False)
            yield made, moved
        for thread, program in enumerate(threads):
            position = self.positions[thread]
            if position == len(program):
                continue
            moved = [*trace, (thread, (thread, position))]
            made = self.step(thread, program[position], arrivals, counts, moved)
            if made is not None:
                made.positions = (*self.positions[:thread], position + 1, *self.positions[thread + 1:])
                made.check(awaited, moved)
                yield made, moved

    def step(self, thread, event, arrivals, counts, trace) -> "_State | None":
        made = self.copy()
        if event.kind == "wait":
            state = self.barrier(event.at, arrivals)
            tracked = event.at in self.tracked[thread]
            parity = tracked if event.argument is None else event.argument % 2 == 1
            if state.phases % 2 == int(parity):
                return None
            # The completion it waits for: the one it names, or its next.
            awaited = (counts[thread][self.positions[thread]].get(event.at, 0) if event.argument is None
                       else event.argument) + 1
            if state.phases != awaited:
                raise Violation(
                    "aliased", f"thread {thread}'s wait for completion {awaited} on {event.at} passes at "
                                f"phase {state.phases}: a parity aliased", trace)
            if event.argument is None:
                made.tracked = (*self.tracked[:thread], self.tracked[thread] ^ {event.at},
                                *self.tracked[thread + 1:])
        elif event.kind == "arrive":
            made.arrive(event.at, arrivals, thread, trace)
        elif event.kind == "expect":
            made.arrive(event.at, arrivals, thread, trace, transaction=event.argument)
        elif event.kind in ("load", "copy") and event.argument:
            if event.kind == "load":
                made.arrive(event.argument, arrivals, thread, trace, transaction=1)
            made.flight = self.flight | {(thread, self.positions[thread], event.argument)}
        elif event.kind == "committed":
            made.flight = self.flight | {(thread, self.positions[thread], event.at)}
        elif event.kind == "fill":
            # A thread's asynchronous write (cp.async's), no barrier's: it lands later, `fill_wait` its wait.
            made.flight = self.flight | {(thread, self.positions[thread], None)}
        elif event.kind == "fill_wait" and any(issuer == thread and barrier is None
                                                for issuer, _, barrier in self.flight):
            return None
        return made

    def check(self, awaited, trace) -> None:
        """No barrier runs past the completion a thread's next wait on it awaits (`_awaited`): a wait naming
        no phase, two phases past the thread's last observed."""
        for thread, position in enumerate(self.positions):
            for at, completion in awaited[thread][position].items():
                state = self.barriers.get(at)
                if state is not None and state.phases > completion:
                    raise Violation(
                        "aliased", f"{at} runs past completion {completion}, which thread {thread}'s next "
                                   "wait on it awaits", trace)


def _joined(one: dict, other: dict) -> dict:
    """Two vector clocks' join."""
    return {key: max(one.get(key, 0), other.get(key, 0)) for key in one.keys() | other.keys()}


def _before(one: dict, other: dict) -> bool:
    """Whether vector clock `one` happens before or at `other`."""
    return all(other.get(key, 0) >= count for key, count in one.items())


def _races(threads, trace, arrivals) -> None:
    """The happens-before order of one completed execution (`trace`), every execution's where each wait
    observes its own completion (`explore`'s checks), checked (`_Order`)."""
    order = _Order(threads, arrivals)
    for mover, (thread, position) in trace:
        event = threads[thread][position]
        if mover == "landed":
            order.landed(thread, position, event)
        else:
            order.moved(thread, position, event)
    for thread, entries in enumerate(order.pending):
        for entry_kind, where, prior_event, *_ in entries:
            if entry_kind == "tmem_load":
                raise Violation("unordered", f"tmem_load of {where} never waited on",
                                [(thread, prior_event)])


class _Order:
    """An execution's happens-before order, built a move at a time: each thread's vector clock, each load's
    landing its own clock joined into its barrier's phase; each access of a location ordered after every
    write of it before (a read after the write lands) and each write after every read of it before (an
    asynchronous read, a product's, a store's or a tensor-memory load's, until the wait that retires it);
    each store's source fenced after its last generic write."""

    def __init__(self, threads, arrivals):
        self.threads, self.arrivals = threads, arrivals
        self.clocks = [{("thread", thread): 0} for thread in range(len(threads))]
        self.accumulating: dict = {}
        self.completions: dict = {}
        self.writes: list = []
        self.reads: list = []
        self.pending: list = [[] for _ in threads]
        self.groups: list = [0 for _ in threads]
        self.unfenced: list = [set() for _ in threads]
        self.issued: dict = {}
        self.committed: list = [[] for _ in threads]
        self.committing: dict = {}
        self.fills: dict = {}
        self.stored: list = [[] for _ in threads]

    def arrived(self, at, clock, *, counted: bool, transaction: int) -> None:
        joined, count, flying = self.accumulating.get(at, ({}, 0, 0))
        joined, count, flying = _joined(joined, clock), count + int(counted), flying + transaction
        if count == self.arrivals[at[0]] and flying == 0:
            self.completions.setdefault(at, []).append(joined)
            self.accumulating.pop(at, None)
        else:
            self.accumulating[at] = (joined, count, flying)

    def ordered(self, at, clock, event, *, write: bool, thread: int | None = None) -> None:
        # Two tcgen05 products issued by the same warp are serialized by the tensor core pipeline.
        def same_pipeline(prior_event: Event, prior_clock: dict | None, prior_thread: int | None) -> bool:
            if event.kind != "product" or prior_event.kind != "product" or thread is None:
                return False
            if prior_thread is not None:
                return prior_thread == thread
            return any(key[0] == "completed" and key[1] == thread for key in (prior_clock or ()))

        for where, prior, prior_event in self.writes:
            if _overlaps(where, at) and not _before(prior, clock):
                if same_pipeline(prior_event, prior, None):
                    continue
                raise Violation(
                    "unordered", f"{event.kind} of {at} unordered with the {prior_event.kind} of {where}",
                                [(None, prior_event), (None, event)])
        if write:
            for where, prior, prior_event in self.reads:
                if _overlaps(where, at) and not _before(prior, clock):
                    if same_pipeline(prior_event, prior, None):
                        continue
                    raise Violation(
                        "unordered", f"write of {at} unordered with the read of {where} before it",
                                    [(None, prior_event), (None, event)])
            # A read still in flight (a product's, a store's or a tensor-memory load's not retired, a
            # tcgen05 product no commit's completion has ended) ends after this write, whatever orders the
            # write.
            flying = [entry for entries in self.pending for entry in entries]
            flying += [entry for products in self.committed for entry, done in products if done is None]
            for _, where, prior_event, _, (prior_thread, _), _ in flying:
                if _overlaps(where, at):
                    if same_pipeline(prior_event, None, prior_thread):
                        continue
                    said = f"write of {at} while the {prior_event.kind} reading {where} is in flight"
                    raise Violation("unordered", said, [(None, prior_event), (None, event)])

    def landed(self, thread, position, event) -> None:
        landing = _joined(self.issued[(thread, position)], {("load", thread, position): 1})
        if event.kind == "committed":
            # A commit's completion, after every product its thread issued before it completes. A product's
            # completion is its own: after its issue and its token (`completed`), no later event of its
            # thread's, its reads ending and its tensor-memory accumulator write landing there, whichever
            # covering commit lands first; each commit's arrival joins the completions it covers.
            written = set()
            for product in self.committing.pop((thread, position)):
                if product[1] is None:
                    _, source, read, issued, key, _ = product[0]
                    product[1] = _joined(issued, {("completed", *key): 1})
                    self.reads.append((source, product[1], read))
                    if read.at and key not in written:
                        written.add(key)
                        self.ordered(read.at, product[1], read, write=True, thread=key[0])
                        self.writes.append((read.at, product[1], read))
                landing = _joined(landing, product[1])
            self.arrived(event.at, landing, counted=True, transaction=0)
            return
        self.ordered(event.at, landing, event, write=True)
        self.writes.append((event.at, landing, event))
        if event.kind == "fill":
            self.fills.setdefault(thread, []).append(landing)
        elif event.argument:
            self.arrived(event.argument, landing, counted=False, transaction=-1)

    def moved(self, thread, position, event) -> None:
        clock = self.clocks[thread]
        clock[("thread", thread)] += 1
        kind = event.kind
        if kind == "wait" and self.completions.get(event.at):
            self.clocks[thread] = clock = _joined(clock, self.completions[event.at][-1])
        elif kind == "arrive":
            self.arrived(event.at, clock, counted=True, transaction=0)
        elif kind == "expect":
            self.arrived(event.at, clock, counted=True, transaction=event.argument)
        elif kind == "fill_wait":
            # Its fills landed (`_State.step` holds it till then): each landing before it.
            for landing in self.fills.pop(thread, ()):
                clock = _joined(clock, landing)
            self.clocks[thread] = clock
        elif kind in ("load", "copy", "fill"):
            self.issued[(thread, position)] = dict(clock)
            if event.argument and kind == "load":
                self.arrived(event.argument, clock, counted=True, transaction=1)
        elif kind in ("read", "write"):
            if kind == "read" and event.argument == "filled":
                prior_writes = [(prior, writer) for where, prior, writer in self.writes
                                if _overlaps(where, event.at) and _before(prior, clock)]
                latest = [writer for prior, writer in prior_writes
                          if not any(_before(prior, other) and not _before(other, prior)
                                     for other, _ in prior_writes)]
                if not latest or any(writer.kind == "product" for writer in latest):
                    raise Violation("unwritten", f"read of {event.at}, which no fill ordered before it wrote",
                                    [(thread, event)])
            self.ordered(event.at, clock, event, write=kind == "write")
            (self.writes if kind == "write" else self.reads).append((event.at, dict(clock), event))
            if kind == "write":
                self.unfenced[thread].add(event.at)
        elif kind == "commit":
            self.unfenced[thread].clear()
        elif kind == "tmem_store":
            # Lands at its thread's next `tmem_commit` (`_tmem_store`).
            self.stored[thread].append(event)
        elif kind == "tmem_commit":
            for store in self.stored[thread]:
                self.ordered(store.at, clock, store, write=True)
                self.writes.append((store.at, dict(clock), store))
            self.stored[thread] = []
        elif kind == "tmem_load":
            self.asynchronous(thread, position, event, clock)
        elif kind == "tmem_load_wait":
            self.retired(thread, "tmem_load", 0, clock)
        elif kind == "committed":
            # Every product its thread issued before it, each a read and its completion (None until a commit
            # covering it lands, `landed`).
            self.issued[(thread, position)] = dict(clock)
            self.committed[thread] += [[entry, None] for entry in self.pending[thread]
                                       if entry[0] == "product"]
            self.pending[thread] = [entry for entry in self.pending[thread] if entry[0] != "product"]
            self.committing[(thread, position)] = list(self.committed[thread])
        elif kind in ("store", "product"):
            self.asynchronous(thread, position, event, clock)
            self.groups[thread] += commits_group(event)
        elif kind == "store_group":
            self.groups[thread] += commits_group(event)
        elif kind == "product_wait":
            self.retired(thread, "product", int(event.argument), clock)
        elif kind == "store_wait":
            self.retired_groups(thread, int(event.argument), clock)

    def asynchronous(self, thread, position, event, clock) -> None:
        """An asynchronous read issued: a store's source, fenced, a tensor-memory load's source, or a
        product's operands, each read until the wait that retires it (`retired`) or, a tcgen05 product's, its
        completion (`landed`); each held with its issue's clock and place."""
        sources = (event.at,) if event.kind in ("store", "tmem_load") else event.argument
        if event.kind == "store" and any(_overlaps(event.at, at) for at in self.unfenced[thread]):
            raise Violation("unfenced", f"a store reads {event.at} past a generic write no fence follows",
                            [(thread, event)])
        for source in sources:
            self.ordered(source, clock, event, write=False)
            # A store's read, the group it joins (its thread's open one) after it.
            self.pending[thread].append((event.kind, source, event, dict(clock), (thread, position),
                                         self.groups[thread]))

    def retired_groups(self, thread, kept: int, clock) -> None:
        """A store wait retiring every committed store group but its `kept` latest: each of their reads ends
        here; a store no commit has closed stays in flight."""
        retiring = self.groups[thread] - kept
        for entry in [entry for entry in self.pending[thread] if entry[0] == "store" and entry[5] < retiring]:
            self.reads.append((entry[1], dict(clock), entry[2]))
            self.pending[thread].remove(entry)

    def retired(self, thread, kind, kept: int, clock) -> None:
        """A wait retiring all but its `kept` latest asynchronous reads of `kind`: each read ends here."""
        mine = [entry for entry in self.pending[thread] if entry[0] == kind]
        for entry in mine[:max(len(mine) - kept, 0)]:
            self.reads.append((entry[1], dict(clock), entry[2]))
            self.pending[thread].remove(entry)


def commits_group(event: Event) -> bool:
    """Whether `event` commits its thread's open store group: a TMA store committing it (`commit_group`) or
    an explicit commit (`commit_smem_to_gmem_group`), the one statement a store wait's retirement counts and
    a test of a program's groups reads."""
    return (event.kind == "store" and bool(event.argument)) or event.kind == "store_group"


def _overlaps(one: tuple, other: tuple) -> bool:
    """Whether two locations share memory: one ref, one slot or the ref whole, and one part of the slot or
    the slot whole (a gathered step's tile, `Event`)."""
    return bool(one) and bool(other) and one[0] == other[0] and all(
        mine is None or theirs is None or mine == theirs
        for mine, theirs in itertools.zip_longest(one[1:], other[1:]))


def configurations(**bounds) -> Iterator[dict]:
    """Every configuration of `bounds`' values (a name's each value in turn)."""
    names = list(bounds)
    for values in itertools.product(*(bounds[name] for name in names)):
        yield dict(zip(names, values, strict=True))
