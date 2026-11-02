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

import math
from collections.abc import Callable, Sequence

import jax
import jax.extend.core as jcore
from jax.core import ShapedArray

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


def chunks(equations: Sequence[jcore.JaxprEqn], fields: Sequence[jcore.Var], sizes: Sequence[int]
           ) -> tuple[list[list[int]], list[frozenset[int]], dict[jcore.Var, frozenset[int]]]:
    """The chunks `fields` (a program's variables, the leaves of its host-resident fields, `sizes` their
    bytes) come to the device in for `equations`, each a list of their indices; which of them each
    equation reads; and which each value the equations make reads. Leaves an operation reads together,
    through every operation between, share a chunk: an elementwise update reads one parameter's moments
    alone, a norm over the whole state reads every leaf, and an operation with an effect keeps its place
    among all of them. Groups of leaves are packed in the fields' order up to the larger of the largest
    group and a `CHUNKS`th of the fields' bytes. A scalar is no chunk's: it comes to the device first."""
    grouped = {index for index, field in enumerate(fields)
               if isinstance(field.aval, ShapedArray) and field.aval.shape}
    # Each leaf's group: the leaves some operation read together with it, through every one between.
    together = {index: frozenset({index}) for index in grouped}
    reads: dict[jcore.Var, frozenset[int]] = {fields[index]: frozenset({index}) for index in grouped}
    taking = []
    for equation in equations:
        read = frozenset(grouped) if equation.effects else frozenset().union(
            *(reads.get(value, frozenset()) for value in equation.invars if computed(value)))
        if len(read) > 1:
            joined = frozenset().union(*(together[index] for index in read))
            together.update(dict.fromkeys(joined, joined))
        if read:
            reads.update(dict.fromkeys(equation.outvars, read))
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


def staged(fn: Callable, resting, shardings, arguments: tuple, after) -> tuple:
    """`fn(resting, *arguments)`, which returns `(kept, others)`, `kept` the new values of the fields
    `resting` (leaves in pinned host memory, placed by `shardings` there) and `others` the rest, with
    `resting` brought to the device a chunk at a time (`chunks`), the first once `after` (the gradient) is
    ready, and `kept` returned in host memory, each chunk of it stored as soon as it is made."""
    fields, structure = jax.tree.flatten(resting)
    resting_shardings = structure.flatten_up_to(shardings)
    onto_device = [sharding.with_memory_kind("device") for sharding in resting_shardings]
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
    position = {variable: index for index, variable in enumerate(variables)}

    def phase(read: frozenset[int]) -> int:
        return chunk_of[min(read)] if read else -1

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
