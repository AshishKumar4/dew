"""A forked call's table as the recurrence template's walks keep it (`mainloop.sequence_walk`'s `table`;
kernel-templates.md section 3): each document of a row takes its memory up from a slot of the call's table
at a chunk's edge, a given memory or the final an earlier document of the row stored, and a storing document
leaves its final in its slot. A walk reads a slot into its registers (`Memories`), a stored one written back
through global memory by the walk itself (`mosaic.Family.kept`); its reverse walk writes each taken memory's
gradient and sums a stored slot's (`Returning`)."""

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

if TYPE_CHECKING:
    from dew.nn.kernels.templates.recurrence.mainloop import Call


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Table:
    """A forked call's table as the walks take it (`chunk_forward.ForkedTable`, laid on
    `chunked_launch.AlignedDocuments`): each row's given memories `given` `[rows * count * HV * Dk, Dv]` fp32,
    laid as the memories are; the slot each chunk takes its memory up from, `loads` `[chunks]` (one-based,
    zero a zero memory, -1 none: the chunk continues the one before), and the stored slot it leaves its
    memory in, `stores` (zero none); each chunk's entry write `entries` `[chunks * HV * Dk, Dv]` fp32, zero
    but where it takes a document up, where the call gives one; and a row's slots, `count` given and `held`
    stored."""

    given: jax.Array
    loads: jax.Array
    stores: jax.Array
    entries: jax.Array | None
    count: int = dataclasses.field(metadata={'static': True})
    held: int = dataclasses.field(metadata={'static': True})

    @property
    def operands(self) -> tuple:
        """What a walk's launch reads of it, in order (`Taking.of`, `Returning.of`)."""
        return (self.given, self.loads, self.stores, *(() if self.entries is None else (self.entries,)))


@dataclasses.dataclass(frozen=True)
class Memories:
    """A role's view of a table's memories (`Table`): a memory `[Dk, Dv]` of `base` (a chunk, or a row's slot)
    the key slices `[block, bv]` of its head and the role's columns."""

    call: "Call"
    head: jax.Array
    column: jax.Array
    bv: int

    def view(self, ref, base, i):
        """Key slice `i` of the memory at `base` of `ref`."""
        call = self.call
        return ref.at[pl.ds((base * call.heads + self.head) * call.width + i * call.block, call.block),
                      pl.ds(self.column, self.bv)]

    def views(self, ref, base) -> list:
        return [self.view(ref, base, i) for i in range(self.call.width // self.call.block)]

    def read(self, ref, base, i):
        """Key slice `i` of the memory at `base` of `ref`, in the walk's registers (`Family.tile_of`): a
        walk reads a memory a slice at a time, each beside the slices it carries alone."""
        return self.call.family.tile_of(self.view(ref, base, i))

    def zero(self):
        return self.call.family.zeros((self.call.block, self.bv))


@dataclasses.dataclass(frozen=True)
class Taking:
    """A forward walk's table for one role (`sequence_walk`'s `table`): before a chunk that takes a document
    up (`Table.loads`), its memory the table's slot's, given or stored by an earlier document of the row, and
    its entry write added; after a chunk that ends a storing document (`Table.stores`), its memory written to
    the row's stored slot, which a later chunk of the same walk reads back (`Family.kept`)."""

    slots: Memories
    refs: tuple
    stored: object
    row: jax.Array
    count: int
    held: int
    smem: object

    @classmethod
    def of(cls, table: Table, refs: Sequence, stored, slots: Memories, row, smem) -> "Taking":
        """`table`'s walk from its refs as a launch reads them (`Table.operands`) and its stored slots'
        output `stored`."""
        given, loads, stores, *entries = refs
        return cls(slots, (given, loads, stores, entries[0] if entries else None), stored, row, table.count,
                   table.held, smem)

    def opened(self) -> None:
        """Nothing: a stored slot is written by a document before any later one reads it."""

    def before(self, n, state):
        given, loads, _, entries = self.refs
        load = loads[n]

        def taken(i):
            def from_given():
                return lax.cond(load > 0, lambda: self.slots.read(given, self.row * self.count + load - 1, i),
                                self.slots.zero)

            def from_stored():
                return self.slots.read(self.stored, self.row * self.held + load - self.count - 1, i)

            memory = lax.cond(load > self.count, from_stored, from_given) if self.held else from_given()
            return memory if entries is None else memory + self.slots.read(entries, n, i)

        # A slice at a time: each taken up beside the slices carried, the one it replaces dead after it.
        return [lax.cond(load >= 0, functools.partial(taken, i), lambda part=part: part)
                for i, part in enumerate(state)]

    def after(self, n, state):
        if not self.held:
            return state
        slot = self.refs[2][n]

        @pl.when(slot > 0)
        def _():
            call = self.slots.call
            call.family.kept(list(state), self.slots.views(self.stored, self.row * self.held + slot - 1),
                             call.interpret, smem=self.smem)

        return state


@dataclasses.dataclass(frozen=True)
class Returning:
    """A reverse walk's table for one role (`sequence_walk`'s `table`), from a row's last chunk to its first:
    before a chunk that ended a storing document (`Table.stores`), the gradient its stored slot gathered from
    the documents that took it up added to the gradient the chunk leaves; after a chunk that takes a document
    up (`Table.loads`), the gradient of the memory it entered written as its own (`taken`: the caller sums a
    given slot's and forms the entry write's from it) and, for a stored slot, added into the slot's
    (`returned`, read back by the walk, `Family.kept`), the walk before it carrying none."""

    slots: Memories
    loads: Any
    stores: Any
    taken: Any
    returned: Any
    row: jax.Array
    count: int
    held: int
    chunks: int
    smem: object

    def opened(self) -> None:
        """The row's stored slots' gradients zeroed for the walk's sums."""
        call = self.slots.call
        zeros = [self.slots.zero()] * (call.width // call.block)
        for slot in range(self.held):
            call.family.kept(zeros, self.slots.views(self.returned, self.row * self.held + slot),
                             call.interpret, smem=self.smem)

    def before(self, f, grad):
        if not self.held:
            return grad
        slot = self.stores[self.chunks - 1 - f]

        def stored(i, part):
            return part + self.slots.read(self.returned, self.row * self.held + slot - 1, i)

        return [lax.cond(slot > 0, functools.partial(stored, i, part), lambda part=part: part)
                for i, part in enumerate(grad)]

    def after(self, f, grad):
        n = self.chunks - 1 - f
        load = self.loads[n]
        call = self.slots.call

        def taken():
            call.family.stored(list(grad), self.slots.views(self.taken, n), smem=self.smem)
            if self.held:
                @pl.when(load > self.count)
                def _():
                    base = self.row * self.held + load - self.count - 1
                    # A slice at a time: each summed beside the gradient's slices, then kept.
                    summed = [part + self.slots.read(self.returned, base, i) for i, part in enumerate(grad)]
                    call.family.kept(summed, self.slots.views(self.returned, base), call.interpret,
                                     smem=self.smem)
            return [self.slots.zero()] * len(grad)

        return lax.cond(load >= 0, taken, lambda: list(grad))


def program_refs(table: Table | None, refs: Sequence) -> tuple:
    """A forward walk's refs past its operands split: its table's (`Table.operands`, none without one), its
    own outputs, and its stored slots' output last where the table holds any (`table_stored`), else None."""
    taken = 0 if table is None else len(table.operands)
    table_refs, outputs = tuple(refs[:taken]), tuple(refs[taken:])
    if table is None or not table.held:
        return table_refs, outputs, None
    return table_refs, outputs[:-1], outputs[-1]


def taken_by(table: Table | None, refs: Sequence, stored, call: "Call", head, column: Callable, bv: int, row,
             landing: Callable) -> Callable | None:
    """A forward walk's table (`sequence_walk`'s `table`): each role's `Taking` of `table`, its operands'
    refs `refs` and its stored slots' output `stored`, at the role's columns `column(role)` of `head`, `bv`
    wide, landing through `landing(role)`; None for a call without one."""
    if table is None:
        return None
    return lambda role: Taking.of(table, refs, stored, Memories(call, head, column(role), bv), row,
                                  landing(role))


def returned_by(table: Table | None, refs: Sequence, returned: Sequence, call: "Call", head,
                column: Callable, bv: int, row, landing: Callable) -> Callable | None:
    """A reverse walk's table (`sequence_walk`'s `table`): each role's `Returning` of `table`, its loads' and
    stores' refs `refs` and its outputs `returned` (`table_returned`'s), at the role's columns `column(role)`
    of `head`, `bv` wide, landing through `landing(role)`; None for a call without one."""
    if table is None:
        return None
    taken, *gathered = returned
    return lambda role: Returning(Memories(call, head, column(role), bv), refs[0], refs[1], taken,
                                  gathered[0] if gathered else None, row, table.count, table.held,
                                  call.chunks, landing(role))


def stored_split(table: Table | None, launched: Sequence) -> tuple[tuple, jax.Array | None]:
    """A walk's launch's outputs, and its table's stored slots, the finals its storing documents left
    (`table_stored`, its last output) where the table holds any, else None."""
    if table is None or not table.held:
        return tuple(launched), None
    return tuple(launched[:-1]), launched[-1]


def table_returned(call: "Call", table: Table, batch: int) -> tuple:
    """A reverse walk's table outputs (`Returning`): the gradient of the memory each chunk takes up
    `[chunks * HV * Dk, Dv]` fp32 (a chunk that takes nothing up written by none), and its stored slots'
    gathered gradients where the table holds any, as `table_stored` lays them."""
    return (jax.ShapeDtypeStruct((call.chunks * call.heads * call.width, call.value_width), jnp.float32),
            *table_stored(call, table, batch))


def table_stored(call: "Call", table: Table | None, batch: int) -> tuple:
    """A forward walk's stored slots' output where its `table` holds any (`Taking`): `held` memories a row,
    fp32, as the memories are laid."""
    if table is None or not table.held:
        return ()
    return (jax.ShapeDtypeStruct((batch * table.held * call.heads * call.width, call.value_width),
                                 jnp.float32),)


def table_landing(call: "Call", keeps: bool, bv: int) -> dict:
    """A walk's tile its table's memories land through (`Family.kept`), a key slice `[block, bv]` fp32, where
    it `keeps` any (a forward's stored slots, a reverse walk's taken gradients)."""
    return {"table": ((call.block, bv), jnp.float32)} if keeps else {}
