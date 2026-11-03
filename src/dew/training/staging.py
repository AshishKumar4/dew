"""A commit's host-resident fields brought to the device a chunk at a time (row 3526).

A layout that keeps the optimizer state or the EMA in pinned host memory (`Layout.host`) has its step read
them only as it commits, past its gradient (`Transaction.past`). Fetched whole there, the device held all
of it at once beside the parameters and the gradient: a whole Qwen3.5-4B's AdamW step on a GH200, its
moments in host memory, compiled to 37.9 to 55.8 GB of temporaries and never fit the device, where the same
step with its moments on the device peaked at 63.6 GiB (row 3526's receipts). `staged` runs the rest of the
step over the fields a chunk at a time instead: a chunk is the leaves that operations of the step read
together (`chunks`), each fetched once the chunk before it is computed and the one before that is back in
host memory, so the device holds about two chunks of the fields beside what the step holds anyway. Its
program is the step's own, each operation bound once in an order its data allows, with optimization
barriers between the chunks: the same arithmetic in another schedule.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from collections.abc import Callable, Sequence

import jax
import jax.extend.core as jcore
from jax.core import ShapedArray

from dew.distributed.placement import placed_bytes
from dew.training.compiled_program import bodies, computed, inlined

PURE_CALLS = frozenset({"jit", "pjit", "closed_call", "core_call"})
"""The operations that call a program on their operands and do nothing else, which `flattened` runs in
their caller: an optax helper's `jit` over a whole tree reads each leaf apart from the rest within it."""

CHUNKS = 16
"""The most chunks the fields' bytes are cut into where no group of leaves read together is larger
(`chunks`): each chunk's fetch and store wait on the chunks before them, so a field of many small leaves
cut one leaf a chunk would pay that wait thousands of times for no device memory that matters."""


def flattened(program: jcore.Jaxpr, operands: Sequence, consts: Sequence = ()) -> tuple[list, list]:
    """`program`'s equations over `operands`, each call of `PURE_CALLS` that places nothing replaced by its
    own program's (`inlined`, fresh values for each), and its results."""
    equations, returned = inlined(program, operands, consts)
    expanded: list[jcore.JaxprEqn] = []
    renamed: dict = {}

    def read(value):
        return renamed.get(value, value) if computed(value) else value

    for equation in equations:
        reading = [read(value) for value in equation.invars]
        called = bodies(equation)
        placing = [sharding for key in ("in_shardings", "out_shardings")
                   for sharding in equation.params.get(key, ())
                   if isinstance(sharding, jax.sharding.Sharding)]
        if equation.primitive.name in PURE_CALLS and len(called) == 1 and not placing:
            inner, outputs = flattened(called[0], reading, tuple(called[0].consts))
            expanded.extend(inner)
            renamed.update(zip(equation.outvars, outputs, strict=True))
        else:
            expanded.append(equation.replace(invars=reading))
    return expanded, [read(value) for value in returned]


def elements(value: jcore.Var) -> int:
    """The elements `value` holds: an array's, a word for any other."""
    return math.prod(value.aval.shape) if isinstance(value.aval, ShapedArray) else 1


def chunks(equations: Sequence[jcore.JaxprEqn], fields: Sequence[jcore.Var], sizes: Sequence[int]
           ) -> tuple[list[list[int]], list[frozenset[int]], dict[jcore.Var, frozenset[int]]]:
    """The chunks `fields` (a program's variables, the leaves of its host-resident fields, `sizes` their
    bytes) come to the device in for `equations`, each a list of their indices; which of them each
    equation reads; and which each value the equations make reads. Leaves an operation reads together,
    through every operation between, share a chunk: an elementwise update reads one parameter's moments
    alone, a norm over the whole state reads every leaf into each update, and an operation with an effect
    keeps its place among all of them. A summary of the leaves joins no chunks: a scalar made of them, or
    what an operation makes of summaries and other scalars in no more elements than the distinct scalars
    they are made of (a sampled step's norms of its update, summed across leaves and stacked into one
    vector: `dew.training.inspection`), so a summary is a few words however it is rearranged, never a
    scalar spread back over a leaf's shape. Each is made in its chunk and held until the last chunk it
    reads is made (`staged`'s phases).
    Groups of leaves are packed in the fields' order up to the larger of the largest group and a
    `CHUNKS`th of the fields' bytes. A scalar field is no chunk's: it comes to the device first."""
    grouped = {index for index, field in enumerate(fields)
               if isinstance(field.aval, ShapedArray) and field.aval.shape}
    # Each leaf's group: the leaves some operation read together with it, through every one between.
    together = {index: frozenset({index}) for index in grouped}
    reads: dict[jcore.Var, frozenset[int]] = {fields[index]: frozenset({index}) for index in grouped}
    # Each summary's words: the distinct scalars it is made of.
    summaries: dict[jcore.Var, frozenset] = {}
    taking = []
    for equation in equations:
        read = frozenset(grouped) if equation.effects else frozenset().union(
            *(reads.get(value, frozenset()) for value in equation.invars if computed(value)))
        # Summaries of the leaves combine without holding any leaf: their chunks stay apart.
        derived = [value for value in equation.invars if computed(value) and reads.get(value)]
        words = frozenset().union(
            *(summaries.get(value, frozenset()) for value in derived),
            (value for value in equation.invars if computed(value) and not reads.get(value)
             and elements(value) == 1))
        summarized = (bool(derived) and not equation.effects and all(value in summaries for value in derived)
                      and sum(map(elements, equation.outvars)) <= len(words))
        if len(read) > 1 and not summarized:
            joined = frozenset().union(*(together[index] for index in read))
            together.update(dict.fromkeys(joined, joined))
        if read:
            reads.update(dict.fromkeys(equation.outvars, read))
            summaries.update({value: words if summarized else frozenset({value}) for value in equation.outvars
                              if summarized or elements(value) == 1})
        taking.append(read)
    groups = sorted(set(together.values()), key=min)
    budget = max([*(sum(sizes[index] for index in group) for group in groups),
                  math.ceil(sum(sizes) / CHUNKS)])
    cut: list[list[int]] = []
    filled = 0
    for group in groups:
        weight = sum(sizes[index] for index in group)
        if not cut or filled + weight > budget:
            cut.append([])
            filled = 0
        cut[-1].extend(sorted(group))
        filled += weight
    return cut, taking, reads


_MEASURED: contextvars.ContextVar[tuple[list[int], ...]] = contextvars.ContextVar("staged_bytes", default=())
"""Where `staged` reports, as a step is traced, the bytes of its fields the device holds at once: each
`measured` the trace runs within (`measured`)."""


@contextlib.contextmanager
def measured():
    """The bytes of its host-resident fields a commit traced within holds on the device at once, each
    `staged` region's (a device's share of each leaf: the chunk it computes and the one it fetches as it
    stores the chunk before, the two largest): the planner's measure of a staged commit is the step's own
    cut (`dew.training.footprint.Footprint.staged`). A measure within another reports to both."""
    found: list[int] = []
    token = _MEASURED.set((*_MEASURED.get(), found))
    try:
        yield found
    finally:
        _MEASURED.reset(token)


def in_device(tree) -> bool:
    """Whether every leaf of `tree` is in device memory."""
    return all(jax.typeof(leaf).memory_space == jax.memory.Space.Device for leaf in jax.tree.leaves(tree))


def on_device(placement):
    """`placement`, a tree of shardings, each in device memory: where a host-resident leaf a step or an
    evaluation reads is brought (`fetched`, `staged`)."""
    return jax.tree.map(lambda sharding: sharding.with_memory_kind("device"), placement)


def fetched(tree, placement):
    """`tree`, host-resident leaves placed by `placement`, brought to the device whole (`on_device`): every
    fetch of a layout's host fields (`Trainer._fetched`, `Transaction.fetched`)."""
    return jax.device_put(tree, on_device(placement))


def staged(fn: Callable, resting, shardings, arguments: tuple, after) -> tuple:
    """`fn(resting, *arguments)`, which returns `(kept, others)`, `kept` the new values of the fields
    `resting` (leaves in pinned host memory, placed by `shardings` there) and `others` the rest, with
    `resting` brought to the device a chunk at a time (`chunks`), the first once `after` (the gradient) is
    ready, and `kept` returned in host memory, each chunk of it stored as soon as it is made."""
    fields, structure = jax.tree.flatten(resting)
    resting_shardings = structure.flatten_up_to(shardings)
    onto_device = on_device(resting_shardings)
    operands, given = jax.tree.flatten(arguments)

    def whole(fetched, operands):
        kept, others = fn(structure.unflatten(fetched), *given.unflatten(operands))
        return structure.flatten_up_to(kept), others

    shapes = [jax.ShapeDtypeStruct(field.shape, field.dtype, sharding=sharding,
                                   weak_type=jax.typeof(field).weak_type)
              for field, sharding in zip(fields, onto_device, strict=True)]
    traced, (_, others_shape) = jax.make_jaxpr(whole, return_shape=True)(shapes, operands)
    program = traced.jaxpr
    equations, returned = flattened(program, program.invars, traced.consts)
    variables = program.invars[:len(fields)]
    cut, taking, reads = chunks(equations, variables, [math.prod(field.shape) * field.dtype.itemsize
                                                       for field in fields])
    chunk_of = {index: number for number, members in enumerate(cut) for index in members}
    if measuring := _MEASURED.get():
        # The scalar fields, no chunk's, are fetched first and held throughout.
        shares = sorted((placed_bytes([shapes[index] for index in members]) for members in cut), reverse=True)
        first = placed_bytes([shape for index, shape in enumerate(shapes) if index not in chunk_of])
        for found in measuring:
            found.append(min(placed_bytes(shapes), first + sum(shares[:2])))
    position = {variable: index for index, variable in enumerate(variables)}

    def phase(read: frozenset[int]) -> int:
        # An equation runs once the last chunk it reads is on the device (a scalar of each, `chunks`).
        return max(chunk_of[index] for index in read) if read else -1

    def made_in(value) -> int:
        return phase(reads.get(value, frozenset())) if computed(value) else -1

    phases: dict[int, list[jcore.JaxprEqn]] = {}
    for equation, taken in zip(equations, taking, strict=True):
        phases.setdefault(phase(taken), []).append(equation)
    values = dict(zip(program.invars[len(fields):], operands, strict=True))
    values.update({variable: jax.device_put(fields[index], onto_device[index])
                   for index, variable in enumerate(variables) if index not in chunk_of})

    def read(value):
        return value.val if isinstance(value, jcore.Literal) else values[value]

    def run(number: int) -> None:
        for equation in phases.get(number, []):
            with equation.ctx.manager:
                made = equation.primitive.bind(*map(read, equation.invars),
                                               **equation.primitive.get_bind_params(equation.params))
            values.update(zip(equation.outvars, made if equation.primitive.multiple_results else [made],
                              strict=True))

    kept, others = returned[:len(fields)], returned[len(fields):]
    # A kept leaf that is a field's own leaf as it rests stays where it rests, never fetched.
    stored = [fields[position[value]] if computed(value) and value in position else None for value in kept]

    def store(number: int) -> list:
        for index, value in enumerate(kept):
            if stored[index] is None and made_in(value) == number:
                stored[index] = jax.device_put(read(value), resting_shardings[index])
        return [stored[index] for index, value in enumerate(kept)
                if made_in(value) == number and not (computed(value) and value in position)]

    run(-1)
    store(-1)
    previous, before = after, []
    for number, members in enumerate(cut):
        _, _, arriving = jax.lax.optimization_barrier((previous, before,
                                                       [fields[index] for index in members]))
        values.update({variables[index]: jax.device_put(field, onto_device[index])
                       for index, field in zip(members, arriving, strict=True)})
        run(number)
        made = [read(value) for value in returned
                if computed(value) and value not in position and made_in(value) == number]
        before, previous = store(number), made or arriving
    return structure.unflatten(stored), jax.tree.unflatten(jax.tree.structure(others_shape),
                                                          [read(value) for value in others])
