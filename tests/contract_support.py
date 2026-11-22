"""The derivative contracts' harness (ledger 2641): each custom-derivative seam's orders (`orders`), the
contract a seam is held to (`assert_contract`), the draws at float32's exponent edges (`EDGES`,
`normal_draw`) and the orders a seam may not take (`FORWARD_ORDERS`, `REVERSE_ONLY` and the rest). The
contracts themselves are tests/test_contracts_*.py, one an owning area, and each seam names its own at
the seam (`dew.nn.derivatives.contracted`, held by tools/lint_slop.py's SLOP028).

Float-range and nested-derivative contracts of Dew's custom-derivative seams (ledger row 2283, theme
G12): each seam's primal, its forward and reverse derivatives, a forward derivative of its gradient and a
second forward derivative, the last two again under a rematerialized call (`jax.checkpoint`), run at a
normal draw and at draws at float32's exponent edges. Each order runs (a custom derivative's tangent tree
and dtypes as JAX requires: rows 1861 and 1862, a boolean residual's tangent and a cotangent tree of the
wrong length), and each of its leaves is finite wherever the same order's float64 run is a normal float32
(row 1768, a contraction that overflowed before its division) and no order of its sums can overflow (row
5426, `assert_contract`). Behaviour alone: no route, structure or bits are read, and each seam is its
public call."""


import contextlib
import dataclasses
import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import drawn_on_the_host, on_float64_device, widened
from float64_terms import float64_signed_sums

LEAST = float(np.finfo(np.float32).tiny)
"""Float32's least normal value: a held position's truth is at least it (testing.md's derivative
contracts)."""
MOST = float(np.finfo(np.float32).max)
"""Float32's largest value: a held position's truth, and each signed sum of its terms, at most it."""

CANCELLING_LEVELS = ((2.0 ** 26, 2.0 ** 127), (2.0, 2.0 ** 127), (1.0, -(2.0 ** 103)), (2.0, -(2.0 ** 127)),
                     (1.0, 2.0 ** 103), (2.0 ** 26, -(2.0 ** 127)), (1.0, MOST))
"""Seven products of a float32 key and value, 2^153, 2^128, -2^103, -2^128, 2^103, -2^153 and the largest,
summing to the largest: three levels cancelling beside it, which floats summed in some order lose (one
dropping -2^103 at a tie as 2^103 survives summed to inf, row 8757). Each a multiple of 2^103 below 2^154, so
float64 sums them, and any power of two times them, exactly in any order."""


GIVEN = 1
"""The memories a forked call's row table holds before its documents store theirs (`Chunk.loads`: a row's
start, a cache)."""

FORKED = (((1, 2, 2), (1, 0, 0)), ((1, 2, 3), (1, 2, 0)))
"""The forked tests' table, each row's documents' table slots and stored slots
(`recurrence_chunk.Chunk.loads`, `Chunk.stores`), one given memory a row then two stored: the first row's
start continued by its first document, whose final both later documents enter (a slot's gradient summing
two documents'); the second row a chain from its start, each document entering its predecessor's final.
Every memory is entered, so no gradient's row is float64 zero, whose terms the gate would trace each draw.
Its token boundaries are each route's (test_recurrence_chunk's, test_ssd_kernel's `DOCUMENTS`)."""

HELD = 2
"""The stored slots `FORKED`'s rows hold (`Chunk.held`)."""

ADDS = ((0, 0, 1), (0, 0, 2))
"""Each `FORKED` row's documents' added slots (`Chunk.adds`): the first row's last document adding its row's
given memory to its parent's final, the second row's the first document's final to its predecessor's (row
7662)."""


class Entries(NamedTuple):
    """A forked call's entry writes of `CANCELLING_LEVELS` (`cancelling_entries`): keys `[R, D, 7, Hk, K]`,
    values `[R, D, 7, H, V]` and weights `[R, D, H]`, and `signed(shape)`, a cotangent of that shape of one
    power of two a draw, each element's sign drawn."""

    key: np.ndarray
    value: np.ndarray
    weight: np.ndarray
    signed: Callable[[tuple[int, ...]], np.ndarray]


def cancelling_orbit(key_heads: int, heads: int, values: int) -> int:
    """The draws `cancelling_entries` takes to write every value coordinate it moves through once: a value
    head of a key head's group and a column a draw, `heads // key_heads` of the one and `values` of the other,
    so an exact fixture declaring this many seeds holds every coordinate in its first stage (check 1 redraws
    only past a failure, which a coordinate never written cannot raise)."""
    return heads // key_heads * values


def cancelling_entries(seed: int, key_heads: int, keys: int, heads: int, values: int) -> Entries:
    """Entry writes of `CANCELLING_LEVELS`' seven products on their keys' first dimension for a forked call
    over `FORKED`'s table, as its route and its oracle read them alike: one value coordinate a key head, a
    value head of its group and a column, both moving with the seed so the draws visit every one, scaled by
    a drawn sign and a power of two of 2^-28 to 2^-33 a draw, so even a row's three writes' terms sum inside
    float32's range, as the gates' sums must (testing.md's derivative contracts), their ties kept, and every
    other coordinate zero, so an entry key's gradient is one product of the levels' (rounded once, as
    float64's: summed over a key head's value heads and columns, float32 rounded each term, 1.9 to 2.6 times
    the rounded truth's distance; one fixed coordinate left a dropped write elsewhere unseen, row 9531);
    weights of another, each document's signed by the memory it enters (`_entered_signs`), so every memory
    is zero, one or two writes, a stored one cancelled exactly among them; and cotangents (`Entries.signed`)
    of one power of two a draw, so a sum of the call's outputs and gradients float32 holds exactly or rounds
    once, as float64's. Every other column's draws reach float32's largest itself (row 9074, the endpoint the
    scaled draws stay below): unscaled, each write's levels summing to it, and weights of a half, so a memory
    of two writes is it or its negative, their randomness spent as the scaled draws'."""
    rows, documents = len(FORKED), len(FORKED[0][0])
    rng = np.random.default_rng([8840, seed])
    factors, levels = (np.asarray(part) for part in zip(*CANCELLING_LEVELS, strict=True))
    key = np.zeros((rows, documents, len(factors), key_heads, keys), np.float32)
    key[..., 0] = factors[:, None]
    group, columns = heads // key_heads, np.zeros((heads, values))
    column = seed // group % values
    largest = column % 2 == 1
    columns[np.arange(key_heads) * group + seed % group, column] = (
        rng.choice([-1.0, 1.0], key_heads) * 2.0 ** -(0 if largest else int(rng.integers(28, 34))))
    value = np.broadcast_to(levels[:, None, None] * columns, (rows, documents, len(levels), heads, values))
    signs = _entered_signs(rng.choice([-1.0, 1.0], (rows, heads)))
    scale = 2.0 ** -int(rng.integers(10, 15))

    def signed(shape):
        return (rng.choice([-1.0, 1.0], shape) * scale).astype(np.float32)

    weight = signs * 2.0 ** -(1 if largest else int(rng.integers(0, 4)))
    return Entries(key, value.astype(np.float32), weight.astype(np.float32), signed)


def _entered_signs(drawn: np.ndarray) -> np.ndarray:
    """Each `FORKED` document's write's sign `[R, D, H]`, a row's first `drawn` `[R, H]`, by the signed count
    of writes the memory it enters holds: one entering a memory of none (a given one, zero in the levels'
    draws, or one cancelled) takes the row's; one entering a memory of one write cancels it where its own
    final is stored and a later document enters it, so an exact zero crosses a stored slot, and adds a second
    of its sign where not; one entering two writes takes the other sign. The memory each leaves is zero, one
    or two writes."""
    signs = np.zeros((*drawn.shape[:1], len(FORKED[0][0]), *drawn.shape[1:]))
    for row, (loads, stores) in enumerate(FORKED):
        counts: dict[int, np.ndarray] = {}
        for document, load in enumerate(loads):
            entered = counts.get(load, np.zeros_like(drawn[row]))
            carried = stores[document] and GIVEN + stores[document] in loads[document + 1:]
            continued = np.where(np.abs(entered) == 1, -np.sign(entered) if carried else np.sign(entered),
                                 -np.sign(entered))
            signs[row, document] = np.where(entered == 0, drawn[row], continued)
            if stores[document]:
                counts[GIVEN + stores[document]] = entered + signs[row, document]
    return signs


class LessFinite(AssertionError):
    """A seam's order not finite where the contract holds it (`assert_contract`): what the contract refuses,
    apart from an order that does not run."""


def normal_float32(values) -> np.ndarray:
    """Whether each float64 value is a normal float32's magnitude: from the least normal to the largest,
    or zero."""
    size = np.abs(np.asarray(values, np.float64))
    return (size == 0) | ((size >= LEAST) & (size <= MOST))


def pulled(fn, primals, cotangent):
    """`fn`'s reverse derivative at `primals` pulling back `cotangent` in the output's own dtype."""
    out, pull = jax.vjp(fn, *primals)
    return pull(jax.tree.map(lambda given, made: jnp.asarray(given, made.dtype), cotangent, out))


FORWARD_ORDERS = ("jvp", "second jvp", "jvp of grad", "jvp of grad, remat", "second jvp, remat",
                  "jvp of grad, nested jit", "jvp of grad, scanned")
"""The orders a forward derivative takes part in: those a custom VJP, which has no forward mode, does not
take (a row names each such seam's gap)."""


GRADIENT_ORDERS = ("jvp of grad", "jvp of grad, remat", "jvp of grad, nested jit", "jvp of grad, scanned",
                   "grad of grad")
"""The orders that differentiate a gradient."""


REVERSE_ORDERS = ("vjp", *GRADIENT_ORDERS)
"""The orders a reverse derivative takes part in: those a forward mode alone, whose reverse mode is another
seam's, does not take."""


OWN_FORWARD_ORDERS = ("jvp", "second jvp", "second jvp, remat")
"""The orders that take a forward derivative of the seam itself, not of its gradient: those a custom VJP whose
backward JAX differentiates does not run."""


REVERSE_ONLY = (*FORWARD_ORDERS, "grad of grad")
"""The orders a reverse-only derivative over kernels with none of their own does not run (`jax.custom_vjp`):
every order through a forward derivative, and reverse over reverse, its backward kernels' own derivative
(a fused attention kernel's, ledger 2654, which `forward_mode_attention` serves forward)."""


ORDER_SETS = {"first orders": ("primal", "jvp", "vjp"), "jvp of grad": ("jvp of grad",),
              "second jvp": ("second jvp",), "grad of grad": ("grad of grad",),
              "jvp of grad, remat": ("jvp of grad, remat",), "second jvp, remat": ("second jvp, remat",),
              "jvp of grad, nested jit": ("jvp of grad, nested jit",),
              "jvp of grad, scanned": ("jvp of grad, scanned",)}
"""The contract's orders (`orders`) a case takes at once, the chunked delta rule's, a padded merge's and the
convex mean's: all ten in one case took 127 to 156 s near float32's largest, each order's program compiled
for itself (row 4433); the three second orders in one case 172 s over packed documents there (row 5348), so
each is a case of its own, the scanned one apart, and each staged second order too: the three in one took
193 s on a padded merge (row 9799) and 136 s on the convex mean, each compiled cold (row 11720). The derived
delta rule's and a turned template call's every order in one case, their rule and references dispatched op by
op, took 178 to 193 s and 132 s at unit operands (row 9512): each a case an order set, its rule and
references staged (`jax.jit`)."""


def order_set(taken: str, without: tuple[str, ...] = (), sets: dict = ORDER_SETS) -> tuple[str, ...]:
    """The orders of set `taken` (in `sets`, `ORDER_SETS` or a contract's own split of them) a case takes,
    skipped where `without` leaves it none (row 11778: a remat set under an interpreted kernel's `without`
    held none, and its cells passed taking no order); its id stays on every lane, so a plan's literal sites
    (`CHECK1_SHARDS`) name the same cells."""
    if not set(sets[taken]) - set(without):
        pytest.skip(f"the order set {taken!r} holds only orders this lane leaves out: {sorted(without)}")
    return sets[taken]


def orders(fn, primals, direction, cotangent):
    """`fn` at each order the contract holds, by name: the primal, a forward and a reverse derivative, a
    forward derivative of the gradient of the output's sum against `cotangent` and a second forward
    derivative, the last two again with `fn` rematerialized; and the gradient's forward derivative traced
    as programs nest it (ledger 2641): under a jit within a jit, within a scan's body, and a reverse
    derivative of the gradient along `direction` (reverse over reverse)."""
    def scalar(f):
        return lambda *x: sum(jnp.sum(out * along) for out, along in zip(
            jax.tree.leaves(f(*x)), jax.tree.leaves(cotangent), strict=True))

    def forward(f):
        return lambda *x: jax.jvp(f, x, direction)[1]

    def gradient(f):
        return jax.grad(scalar(f), argnums=tuple(range(len(primals))))

    remat = jax.checkpoint(fn)
    return {
        "primal": lambda: fn(*primals),
        "jvp": lambda: forward(fn)(*primals),
        "vjp": lambda: pulled(fn, primals, cotangent),
        "jvp of grad": lambda: jax.jvp(gradient(fn), primals, direction)[1],
        "second jvp": lambda: jax.jvp(forward(fn), primals, direction)[1],
        "jvp of grad, remat": lambda: jax.jvp(gradient(remat), primals, direction)[1],
        "second jvp, remat": lambda: jax.jvp(forward(remat), primals, direction)[1],
        "jvp of grad, nested jit": lambda: jax.jit(
            lambda *x: jax.jvp(jax.jit(gradient(fn)), x, direction)[1])(*primals),
        "jvp of grad, scanned": lambda: jax.lax.scan(
            lambda carry, _: (carry, jax.jvp(gradient(fn), primals, direction)[1]), 0, None, length=1)[1],
        "grad of grad": lambda: jax.grad(
            lambda *x: sum(jnp.sum(jnp.asarray(along, part.dtype) * part)
                           for along, part in zip(direction, gradient(fn)(*x), strict=True)),
            argnums=tuple(range(len(primals))))(*primals),
    }


def ran(order, label: str, name: str):
    """`order()`'s leaves as arrays, a failure to run named by the seam and the order that raised it."""
    with running(label, name):
        return jax.tree.map(np.asarray, order())


@contextlib.contextmanager
def running(label: str, name: str):
    """A failure within, an order's lowering, compile or run, named by the seam and the order that raised
    it (`ran`, `_compiled_ahead`)."""
    try:
        yield
    except Exception as error:
        raise AssertionError(f"{label}: its {name} does not run: {type(error).__name__}: {error}") from error


def assert_contract(fn, primals, direction, cotangent, label: str, *, truth_fn=None, plain_fn=None,
                    taking: tuple[str, ...] | None = None, without: tuple[str, ...] = (), draw=None,
                    seeds: int | None = None) -> None:
    """Every order of `fn` runs at the draw, and each leaf is finite at every position the contract holds:
    where the same order of its float64 run is a normal float32; where, given `plain_fn` (the plain float32
    formula, JAX's own derivatives of it), the plain formula's is finite; and where no order of summation
    can overflow: of the terms the float64 run's sums add (`float64_signed_sums`, a sum behind a sum at its
    own), the positive ones' sum and the size of the negative ones' sum each inside float32's range, the
    largest any order of them passes through (the owner's choice of 2026-11-10, row 5426: finiteness held
    only where no summation order can overflow). So a form that overflows where the plain formula's own
    operations stay in range fails (row 1768, a contraction Dew summed before the division that brought it
    back), as does one that overflows over terms that cancel inside the range in every order (2^126 twice
    and its negative twice); and a sum whose terms float32 can overflow in one order and not another is
    held to neither order (row 5426: two orders each overflow partials the other does not). The
    float64 run is `fn`'s own, or `truth_fn`'s, the plain float64 formula, where `fn` computes in a fixed
    precision whatever its inputs' (a declared dot algorithm). `taking` names the orders a case holds, where
    it holds some alone (a seam's orders split across cases, a witness of one order), every order where
    None; `without` names orders the contract does not take yet, each joining with its row's fix (a row's
    head brings its own test). A name of no order (`orders`) in either is refused, and so is a case they leave
    no order (`order_set`).

    And every order runs on training's mesh of two devices (`_on_a_mesh`), as a training step runs it (row
    7973: a launch whose specs left out two of its operands raised under any mesh, and the suite, run off
    one, passed it), a check-1 site there over the cell's seeded `draw` where it gives one, of `seeds` where
    given (a cell sharded across CI's map tasks, its mark's count: `map_judge.marked_seeds`)."""

    every = set(orders(fn, primals, direction, cotangent))
    named = {*(taking or ()), *without} - every
    if named:
        raise ValueError(f"{label}: no order of the contract's is named {sorted(named)}")
    taken = frozenset(name for name in every if (taking is None or name in taking) and name not in without)
    if not taken:
        # A case of no order holds nothing: one its lane leaves no order skips (`order_set`, row 11778).
        raise ValueError(f"{label}: takes no order, {sorted(taking or ())} all left out by {sorted(without)}")

    def orders_of(f, *drawn):
        return {name: order for name, order in orders(f, *drawn).items() if name in taken}

    # Each arm's run of each order, built once for its compile ahead and its run (`_Run`): each order one
    # program, made once a function and an order (`_mesh_program`), so a cell's edges, one shape at three
    # scales, compile it once (dispatched op by op, every edge's run compiled its order's primitives again,
    # most of a cell's 250 s, row 11720); the mesh's inputs placed on it once.
    given = (primals, direction, cotangent)
    with _on_float64():
        wide_given = widened(given)
    in_order = [name for name in orders(fn, *given) if name in taken]

    def runs(f, arguments, context, site):
        return {name: _Run(_mesh_program(f, name), arguments, context, site(name), name) for name in in_order}

    narrow = runs(fn, given, contextlib.nullcontext, lambda _: label)
    plainly = {} if plain_fn is None else runs(plain_fn, given, contextlib.nullcontext,
                                               lambda _: f"{label}, plainly")
    wide_runs = runs(truth_fn or fn, wide_given, _on_float64, lambda _: f"{label} in float64")
    mesh = _training_mesh()
    meshed = {} if mesh is None else runs(fn, _placed(mesh, given), functools.partial(jax.set_mesh, mesh),
                                         lambda name: f"{label}, {name}, on training's mesh of two")
    _compiled_ahead([*narrow.values(), *plainly.values(), *wide_runs.values(), *meshed.values()])
    plain = {name: run() for name, run in plainly.items()}
    truths = {name: run() for name, run in wide_runs.items()}
    with _on_float64():
        # Its eager orders, which `float64_signed_sums` walks where an order has a value to lose.
        wide = orders_of(truth_fn or fn, *wide_given)
    def held(values, name: str, site: str) -> None:
        """An order's `values` held to the contract: each finite where the same order of the float64 run is a
        normal float32, the plain formula's is finite and no order of its sums overflows."""
        plains = jax.tree.leaves(plain[name]) if plain else [None] * len(values)
        lost = {}
        for index, (value, truth, plainly) in enumerate(zip(values, jax.tree.leaves(truths[name]), plains,
                                                            strict=True)):
            # JAX's dtype lattice, which holds bfloat16 (ml_dtypes' is no NumPy inexact type).
            if not jnp.issubdtype(value.dtype, jnp.inexact):
                continue
            kept = normal_float32(truth) & np.isfinite(truth)
            if plainly is not None:
                kept &= np.isfinite(plainly)
            if (kept & ~np.isfinite(value)).any():
                lost[index] = kept & ~np.isfinite(value)
        if lost:
            # The float64 run's terms, read only for an order with a value to lose: its program walked once.
            with jax.enable_x64(new_val=True):
                _, positive, negative = map(jax.tree.leaves, float64_signed_sums(wide[name]))
            lost = {index: where & (positive[index] <= MOST) & (negative[index] <= MOST)
                    for index, where in lost.items()}
        broken = [f"{site}, {name}, leaf {index}: {int(where.sum())} values not finite where float64's are "
                  f"normal float32s and no order of their sums overflows, e.g. {values[index][where][:3]}"
                  for index, where in lost.items() if where.any()]
        if broken:
            raise LessFinite(broken[0])

    for name, run in narrow.items():
        held(jax.tree.leaves(run()), name, label)
    if mesh is not None:
        _on_a_mesh(fn, mesh, meshed, truth_fn=truth_fn, draw=draw, seeds=seeds, held=held)


@dataclasses.dataclass(frozen=True)
class _Run:
    """One order of one arm of a case (`assert_contract`): its program (`_mesh_program`), the fixed arguments
    and context it runs in, and the site and order that name its failure (`running`), built once for its
    compile ahead (`_compiled_ahead`) and its run, which lowers alike and finds it compiled."""

    program: Callable
    arguments: tuple
    context: Callable
    site: str
    name: str

    def lowered(self) -> jax.stages.Lowered:
        with self.context(), running(self.site, self.name):
            return self.program.lower(*self.arguments)

    def __call__(self):
        return self.at(*self.arguments)

    def at(self, *arguments):
        """Its program's leaves at `arguments` (a seeded draw's, `_on_a_mesh`), run in its context, a failure
        named by its site and order."""
        with self.context():
            return ran(lambda: self.program(*arguments), self.site, self.name)


def _on_a_mesh(fn, mesh, meshed: dict, *, truth_fn, draw, seeds, held) -> None:
    """Each order of `fn` the contract takes (`meshed`, each its run on `mesh`, `_Run`) on training's mesh of
    two of this process's devices (`MeshSpec`'s, its data axis two), its inputs placed on it whole: at the
    cell's own inputs held to the
    contract itself (`held`, as off the mesh; a split its specs do not fit raises, row 7973's class); and
    where the cell gives its seeded `draw` (a seed's primals, direction and cotangent in its domain, at a
    scale whose values float32 holds: an edge's, which the contract excuses where float64 is no normal
    float32, check 1 cannot measure), a check-1 site (`reference_error.assert_as_exact`) against the
    float64 truth (`truth_fn`'s orders where given), its reference the same order off a mesh, over `seeds`
    where declared, else the seeds `seeds_for` sizes on every run (a cell's sizing most of its time, row
    8429's 300 s): the mesh's program is another, partitioned, whose floats no common bits hold
    (docs/design/testing.md), and a sum it counts twice fails by its own size. Where the cell set its own
    mesh, its orders already run under it."""
    from arm_forms import as_given, dew_path
    from reference_error import assert_as_exact, seeds_for

    for name, run in meshed.items():
        site = run.site

        # A draw's run a placement and the order's run there (`_Run`), which the gate runs as given
        # (`as_given`).
        @as_given
        def on(primals, direction, cotangent, run=run):
            return run.at(*_placed(mesh, (primals, direction, cotangent)))

        # At the cell's own inputs, the contract itself (`held`), on the mesh as off it.
        held(jax.tree.leaves(run()), name, site)
        if draw is None:
            continue

        # The same program off the mesh: one of Dew's own paths, an invariance claim (`dew_path`).
        @dew_path
        def off(primals, direction, cotangent, name=name):
            return orders(fn, primals, direction, cotangent)[name]()

        def truth(primals, direction, cotangent, name=name):
            return orders(truth_fn, primals, direction, cotangent)[name]()

        truth_of = None if truth_fn is None else truth
        count = seeds_for(off, draw, site, truth_fn=truth_of, dew_fn=on) if seeds is None else seeds
        assert_as_exact(on, off, draw, site, seeds=count, truth_fn=truth_of)


def _training_mesh():
    """Training's mesh of two of this process's devices (`MeshSpec`'s, its data axis two) a case's orders run
    on (`_on_a_mesh`); None where the cell set its own mesh, under which its orders already run."""
    from dew.training import MeshSpec

    if not jax.sharding.get_abstract_mesh().empty:
        return None
    return MeshSpec().build(jax.devices()[:2])


def _placed(mesh, drawn):
    """A case's inputs `drawn` placed on `mesh` whole, as its orders read them there (`_on_a_mesh`)."""
    from jax.sharding import NamedSharding, PartitionSpec

    with jax.set_mesh(mesh):
        return jax.device_put(drawn, NamedSharding(mesh, PartitionSpec()))


@contextlib.contextmanager
def _on_float64():
    """The float64 run's context (`assert_contract`'s truths): x64 on its float64 device."""
    with jax.enable_x64(new_val=True), on_float64_device():
        yield


def _compiled_ahead(runs: list[_Run]) -> None:
    """Each program a case runs (`runs`, `_Run`) lowered in turn in its context and compiled side by side
    by Dew's compile of several (`report.attempts`), which its run's lowering finds: a convex mean's
    narrow, float64 and mesh programs of one order, 141,776 operations each, compiled one after another took
    its cell 140 s cold (row 11720). A compile's error is raised named by its site and order (`running`)."""
    from dew.nn.kernels.templates import report

    for run, attempt in zip(runs, report.attempts([(run.lowered(), None) for run in runs]), strict=True):
        if attempt.error is not None:
            with running(run.site, run.name):
                raise attempt.error


@functools.lru_cache(maxsize=256)
def _mesh_program(fn, name: str):
    """Order `name` of `fn` (`orders`) as one jitted program of its primals, direction and cotangent, made
    once a function and an order: compiled once a shape, each seed's draw and each of a cell's edges (`EDGES`,
    one shape at three scales) running it (rows 9806, 9807, 10132: the convex mean's mesh runs compiled each
    order again at every edge, a third of its cell)."""
    return jax.jit(lambda primals, direction, cotangent: orders(fn, primals, direction, cotangent)[name]())


EDGES = {"normal": 1.0, "near the largest": 2.0**63, "near the least normal": 2.0**-63}
"""The draws' scales: unit values, and values whose products (a squared distance, a second derivative's
pairs) reach float32's largest or its least normal."""


def assert_drawn_contract(fn, inputs, scale: float, label: str, **options) -> None:
    """`assert_contract` over a cell's one builder of its inputs, `inputs(seed, scale)`: a seed's primals,
    direction and cotangent at `scale` (an `EDGES` value), in the cell's domain. Its fixed inputs are the
    builder's at seed 0, and at the normal scale alone the builder is the mesh's check-1 draw (`draw`),
    check 1 measuring what float32 holds while an edge's values, which the contract excuses where float64
    is no normal float32, stay judged by the contract alone (Main's ruling, 2026-11-18)."""
    draw = (lambda seed: inputs(seed, scale)) if scale == EDGES["normal"] else None
    assert_contract(fn, *inputs(0, scale), label, draw=draw, **options)


def on_a_tpu(fn):
    """`fn` traced as for a TPU (an assumed v5e, `generation.assumed`): the forms Dew takes there, its own
    where a TPU's backend is not accurate enough (`precision.accurate_exp`), run on this host."""
    from dew.nn.kernels import generation

    @functools.wraps(fn)
    def run(*args, **kwargs):
        with generation.assumed("v5e"):
            return fn(*args, **kwargs)

    return run


def normal_draw(seed, *shapes, scale: float = 1.0, dtype=jnp.float32) -> tuple:
    """Standard normal arrays of `shapes` at `scale` in `dtype`, from `seed` (a seed, or a site's salt and
    a seed: NumPy's `SeedSequence` entropy), made on the host's CPU (`drawn_on_the_host`)."""
    def drawn(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.standard_normal(shape) * scale, dtype) for shape in shapes)

    return drawn_on_the_host(drawn, seed)


def normal_drawn(salt: int, *shapes, dtype=jnp.float32):
    """A check-1 draw (`draw(seed)`) of standard normal arrays of `shapes` in `dtype`, a site's `salt`
    keeping its stream apart (`normal_draw`)."""
    return lambda seed: normal_draw([salt, seed], *shapes, dtype=dtype)
