"""A model checker of the Mosaic GPU toolkit's synchronization (docs/design/kernel-templates.md, "The ring
protocol, model-checked"): a launch's kernel traced as the toolkit's own code builds it (`jax.make_jaxpr`, no
GPU), each warpgroup's program interpreted concretely into its synchronization events, and every
interleaving of those programs explored under CUDA's mbarrier and asynchronous-copy semantics.

The events are the kernel's own equations: an `mbarrier` wait (`barrier_wait`, with its parity, or without,
Mosaic's tracked phase), an arrival (`barrier_arrive`), a TMA load (`copy_gmem_to_smem`: an arrival with its
transaction at issue, its bytes landing later, out of order with other copies), a TMA store
(`copy_smem_to_gmem`, reading shared memory until `wait_smem_to_gmem` retires it), a generic proxy's fence
(`commit_smem`), a warpgroup product (`wgmma`, reading shared memory until `wgmma_wait` retires it), and a
shared-memory read or write (`get`, `swap`). Index arithmetic, loops and branches are evaluated concretely
per warpgroup (`axis_index`), so each program is its own fixed sequence of events: no control flow reads
shared memory a peer writes.

The semantics assumed (PTX ISA, `mbarrier`, `cp.async.bulk`, `fence.proxy.async`):
- a barrier's phase completes once its arrivals (its `num_arrivals`) and its pending transactions are all in;
  a wait of parity `p` passes once the barrier's phase count's parity is not `p` (the phase of parity `p`
  completed), one without a parity at the parity its thread tracks, flipped as it passes;
- an arrival releases, a passing wait acquires: everything before the arrivals of a completed phase happens
  before everything after a wait that observes it; a TMA load's bytes are written after its issue and before
  the transaction completes;
- a TMA store reads its shared-memory source asynchronously until a `wait_smem_to_gmem` retires it, and
  reads a generic write only after a `commit_smem` (`fence.proxy.async`) that follows it;
- the scheduler is weakly fair: a thread that can move eventually does. A state where none can and some
  program has not ended is a deadlock.

The properties checked in every reachable state of every interleaving, within the bounds a test gives:
- no deadlock, and at the end every barrier's phase whole (no arrival left in a phase that never completes);
- each wait observes the completion it is the thread's next wait for on that barrier: no barrier runs two
  phases past a thread that still waits on it (a parity wait's aliasing), and no thread arrives twice in one
  phase of a barrier (an early arrival counted toward a phase it does not belong to);
- with that, every execution synchronizes alike, so its happens-before order is one: every read of shared
  memory (a load's consumer's, a product's, a store's) happens after the write it reads has landed and
  before the next write to the location lands, and a store's source was fenced after its last generic
  write.

It checks bounded configurations: the claim is coverage of the bounds a test states, not a proof for every
configuration."""

import dataclasses
import itertools
from collections.abc import Callable, Iterator

import jax
import numpy as np
from jax.extend import core as jcore

from dew.nn.programs import called

THREAD_AXIS = "wg"
"""The axis a launch's warpgroups are indexed along (`mosaic.kernel`'s `thread_name`)."""


@dataclasses.dataclass(frozen=True)
class Ref:
    """A shared-memory allocation of the kernel: a block of `slots` (its leading axis) or of barriers, each
    `num_arrivals` arrivals a phase."""

    name: str
    barrier: bool
    num_arrivals: int


@dataclasses.dataclass(frozen=True)
class Event:
    """One synchronization event of a warpgroup's program: `kind`, the location it touches (a ref and its
    slot), and its argument (a wait's parity, a product's or a store's sources, a wait's pending count)."""

    kind: str
    at: tuple = ()
    argument: object = None


class _Interpreter:
    """A warpgroup's program: the kernel's body evaluated with its index (`thread`), each equation of the
    toolkit's synchronization an `Event` (`events`), every other evaluated concretely (`bind`) or, a value no
    synchronization reads (a tile's data), left unevaluated."""

    def __init__(self, thread: int):
        self.thread, self.events, self.refs, self.allocated = thread, [], 0, []

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
        if name == "axis_index":
            return [np.int32(self.thread if params["axis_name"] == THREAD_AXIS else 0)]
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

    def scoped(self, params, values) -> list:
        jaxpr = params["jaxpr"]
        allocated = []
        for var in jaxpr.invars:
            aval = var.aval
            inner = aval.inner_aval
            barrier = hasattr(inner.dtype, "num_arrivals")
            self.refs += 1
            allocated.append(Ref(f"{'barrier' if barrier else 'smem'}{self.refs}", barrier=barrier,
                                 num_arrivals=inner.dtype.num_arrivals if barrier else 0))
        self.allocated += allocated
        return self.run(jaxpr, values, allocated)

    def emit(self, kind: str, at: tuple = (), argument=None) -> list:
        self.events.append(Event(kind, at, argument))
        return []


def _slot(ref, treedef, leaves) -> tuple:
    """A transformed ref's location: the ref and its leading index where it is one value (a ring's slot),
    else the ref whole."""
    if ref is None:
        return ()
    if treedef is None or (not treedef.num_leaves and not leaves):
        return (ref.name, None)
    transforms = jax.tree_util.tree_unflatten(treedef, list(leaves))
    for transform in transforms:
        indices = getattr(transform, "indices", None)
        if indices:
            first = indices[0]
            if not hasattr(first, "start") and first is not None:
                return (ref.name, int(np.asarray(first)))
    return (ref.name, None)


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
    parity = (bool(np.asarray(_governing(values[1 + count], "a wait's parity")))
              if equation.params.get("has_parity") else None)
    return self.emit("wait", at, parity)


def _arrive(self: _Interpreter, equation, values) -> list:
    tree = equation.params["transforms_treedef"]
    count = tree.num_leaves
    if equation.params.get("has_user_predicate") and not bool(
            np.asarray(_governing(values[1 + count], "an arrival's predicate"))):
        return []
    return self.emit("arrive", _slot(values[0], tree, values[1:1 + count]))


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
    params = equation.params
    tree = params["src_transforms_treedef"]
    count = tree.num_leaves
    return self.emit("store", _slot(values[0], tree, values[2:2 + count]))


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
            self.emit(kind, _slot(ref, equation.params["tree"], values[indexed:]))
        return [None] * len(equation.outvars)

    return access


def _product(self: _Interpreter, equation, values) -> list:
    sources = tuple((ref.name, None) for ref in values if isinstance(ref, Ref))
    self.emit("product", (), sources)
    return [None] * len(equation.outvars)


def _product_wait(self: _Interpreter, equation, values) -> list:
    return self.emit("product_wait", (), equation.params["n"])


SYNC: dict[str, Callable] = {
    "barrier_wait": _wait, "barrier_arrive": _arrive, "copy_gmem_to_smem": _load,
    "copy_smem_to_gmem": _store, "wait_smem_to_gmem": _store_wait, "commit_smem": _commit,
    "get": _access("read", 1), "swap": _access("write", 2), "wgmma": _product, "wgmma_wait": _product_wait,
}
"""Each primitive of the toolkit's synchronization and shared memory, as an `Event`."""


def programs(kernel: Callable, *operands, threads: int) -> tuple[list[list[Event]], dict[str, int]]:
    """Each warpgroup's program of `kernel` (a Mosaic GPU launch the toolkit built, `mosaic.kernel`) over
    `operands`: its events, by warpgroup index, and each barrier allocation's arrivals a phase, by name."""
    closed = jax.make_jaxpr(kernel)(*operands)
    body = _body(closed.jaxpr)
    found, arrivals = [], {}
    for thread in range(threads):
        interpreter = _Interpreter(thread)
        interpreter.run(body.jaxpr, body.consts, [Ref(f"gmem{index}", barrier=False, num_arrivals=0)
                                                  for index in range(len(body.jaxpr.invars))])
        found.append(interpreter.events)
        arrivals.update({ref.name: ref.num_arrivals for ref in interpreter.allocated if ref.barrier})
    return found, arrivals


def _body(jaxpr: jcore.Jaxpr) -> jcore.ClosedJaxpr:
    """The kernel body's jaxpr inside a traced launch: the per-program jaxpr of its `mpmd_map`."""
    for equation in jaxpr.eqns:
        if equation.primitive.name == "mpmd_map":
            (inner,) = equation.params["jaxprs"]
            return inner if isinstance(inner, jcore.ClosedJaxpr) else jcore.ClosedJaxpr(inner, ())
        for value in equation.params.values():
            for inner in value if isinstance(value, (tuple, list)) else (value,):
                nested = inner.jaxpr if isinstance(inner, jcore.ClosedJaxpr) else inner
                if isinstance(nested, jcore.Jaxpr):
                    try:
                        return _body(nested)
                    except LookupError:
                        continue
    raise LookupError("no Mosaic GPU kernel body in the traced launch")


@dataclasses.dataclass
class Violation(AssertionError):
    """A property the exploration found broken: its class (`kind`: "deadlock", "unfinished" phase, "aliased"
    parity, "early" arrival, "unordered" access, "unfenced" store, "bounds"), what broke (`what`), and the
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
    """Every interleaving of `threads`' programs (each a warpgroup's `Event`s) and their loads' landings,
    each barrier's arrivals a phase by name (`arrivals`): the properties of the module's account checked in
    every reachable state, and the happens-before order of one execution checked for races. Returns the
    states explored; raises `Violation` with its interleaving where a property fails."""
    counts = [_waits(program) for program in threads]
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
        moves = list(state.moves(threads, arrivals, counts, trace))
        if not moves:
            if not state.ended(threads):
                raise Violation("deadlock", "no warpgroup can move and some has not ended", trace)
            state.whole(arrivals, trace)
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

    def whole(self, arrivals, trace) -> None:
        for at, state in self.barriers.items():
            if state.pending != arrivals[at[0]] or state.transactions:
                left = arrivals[at[0]] - state.pending
                raise Violation("unfinished", f"{at}: {left} arrivals and {state.transactions} transactions "
                                "left in a phase that never completes", trace)

    def arrive(self, at, arrivals, thread, trace, *, transaction: int = 0, count: bool = True) -> None:
        state = self.barrier(at, arrivals)
        if count and thread in state.arrived:
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

    def moves(self, threads, arrivals, counts, trace) -> Iterator:
        for flight in sorted(self.flight):
            issuer, position, barrier = flight
            made = self.copy()
            made.flight = self.flight - {flight}
            moved = [*trace, ("landed", (issuer, position))]
            if barrier:
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
                made.check(threads, counts, moved)
                yield made, moved

    def step(self, thread, event, arrivals, counts, trace) -> "_State | None":
        made = self.copy()
        if event.kind == "wait":
            state = self.barrier(event.at, arrivals)
            tracked = event.at in self.tracked[thread]
            parity = event.argument if event.argument is not None else tracked
            if state.phases % 2 == int(parity):
                return None
            passed = counts[thread][self.positions[thread]].get(event.at, 0)
            if state.phases != passed + 1:
                raise Violation(
                    "aliased", f"thread {thread}'s wait {passed + 1} on {event.at} passes at phase "
                                f"{state.phases}: a parity aliased", trace)
            if event.argument is None:
                made.tracked = (*self.tracked[:thread], self.tracked[thread] ^ {event.at},
                                *self.tracked[thread + 1:])
        elif event.kind == "arrive":
            made.arrive(event.at, arrivals, thread, trace)
        elif event.kind == "load" and event.argument:
            made.arrive(event.argument, arrivals, thread, trace, transaction=1)
            made.flight = self.flight | {(thread, self.positions[thread], event.argument)}
        return made

    def check(self, threads, counts, trace) -> None:
        """No barrier runs two phases past a thread that still waits on it."""
        for thread, position in enumerate(self.positions):
            done = counts[thread][position]
            for at, state in self.barriers.items():
                still = counts[thread][-1].get(at, 0) - done.get(at, 0)
                if still and state.phases - done.get(at, 0) > 1:
                    raise Violation(
                        "aliased", f"{at} runs two phases past thread {thread}, which still waits on it",
                                    trace)


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


class _Order:
    """An execution's happens-before order, built a move at a time: each thread's vector clock, each load's
    landing its own clock joined into its barrier's phase; each access of a location ordered after every
    write of it before (a read after the write lands) and each write after every read of it before (an
    asynchronous read, a product's or a store's, until the wait that retires it); each store's source
    fenced after its last generic write."""

    def __init__(self, threads, arrivals):
        self.threads, self.arrivals = threads, arrivals
        self.clocks = [{("thread", thread): 0} for thread in range(len(threads))]
        self.accumulating: dict = {}
        self.completions: dict = {}
        self.writes: list = []
        self.reads: list = []
        self.pending: list = [[] for _ in threads]
        self.unfenced: list = [set() for _ in threads]
        self.issued: dict = {}

    def arrived(self, at, clock, *, counted: bool, transaction: int) -> None:
        joined, count, flying = self.accumulating.get(at, ({}, 0, 0))
        joined, count, flying = _joined(joined, clock), count + int(counted), flying + transaction
        if count == self.arrivals[at[0]] and flying == 0:
            self.completions.setdefault(at, []).append(joined)
            self.accumulating.pop(at, None)
        else:
            self.accumulating[at] = (joined, count, flying)

    def ordered(self, at, clock, event, *, write: bool) -> None:
        for where, prior, prior_event in self.writes:
            if _overlaps(where, at) and not _before(prior, clock):
                raise Violation(
                    "unordered", f"{event.kind} of {at} unordered with the {prior_event.kind} of {where}",
                                [(None, prior_event), (None, event)])
        if write:
            for where, prior, prior_event in self.reads:
                if _overlaps(where, at) and not _before(prior, clock):
                    raise Violation(
                        "unordered", f"write of {at} unordered with the read of {where} before it",
                                    [(None, prior_event), (None, event)])

    def landed(self, thread, position, event) -> None:
        landing = _joined(self.issued[(thread, position)], {("load", thread, position): 1})
        self.ordered(event.at, landing, event, write=True)
        self.writes.append((event.at, landing, event))
        if event.argument:
            self.arrived(event.argument, landing, counted=False, transaction=-1)

    def moved(self, thread, position, event) -> None:
        clock = self.clocks[thread]
        clock[("thread", thread)] += 1
        kind = event.kind
        if kind == "wait" and self.completions.get(event.at):
            self.clocks[thread] = clock = _joined(clock, self.completions[event.at][-1])
        elif kind == "arrive":
            self.arrived(event.at, clock, counted=True, transaction=0)
        elif kind == "load":
            self.issued[(thread, position)] = dict(clock)
            if event.argument:
                self.arrived(event.argument, clock, counted=True, transaction=1)
        elif kind in ("read", "write"):
            self.ordered(event.at, clock, event, write=kind == "write")
            (self.writes if kind == "write" else self.reads).append((event.at, dict(clock), event))
            if kind == "write":
                self.unfenced[thread].add(event.at)
        elif kind == "commit":
            self.unfenced[thread].clear()
        elif kind in ("store", "product"):
            self.asynchronous(thread, event, clock)
        elif kind in ("product_wait", "store_wait"):
            self.retired(thread, "product" if kind == "product_wait" else "store", int(event.argument), clock)

    def asynchronous(self, thread, event, clock) -> None:
        """An asynchronous read issued: a store's source, fenced, or a product's operands, each read until
        the wait that retires it (`retired`)."""
        sources = (event.at,) if event.kind == "store" else event.argument
        if event.kind == "store" and any(_overlaps(event.at, at) for at in self.unfenced[thread]):
            raise Violation("unfenced", f"a store reads {event.at} past a generic write no fence follows",
                            [(thread, event)])
        for source in sources:
            self.ordered(source, clock, event, write=False)
            self.pending[thread].append((event.kind, source, event))

    def retired(self, thread, kind, kept: int, clock) -> None:
        """A wait retiring all but its `kept` latest asynchronous reads of `kind`: each read ends here."""
        mine = [entry for entry in self.pending[thread] if entry[0] == kind]
        for entry in mine[:max(len(mine) - kept, 0)]:
            self.reads.append((entry[1], dict(clock), entry[2]))
            self.pending[thread].remove(entry)


def _overlaps(one: tuple, other: tuple) -> bool:
    """Whether two locations share memory: one ref, and one slot or the ref whole."""
    return bool(one) and bool(other) and one[0] == other[0] and (
        one[1] is None or other[1] is None or one[1] == other[1])


def configurations(**bounds) -> Iterator[dict]:
    """Every configuration of `bounds`' values (a name's each value in turn)."""
    names = list(bounds)
    for values in itertools.product(*(bounds[name] for name in names)):
        yield dict(zip(names, values, strict=True))
