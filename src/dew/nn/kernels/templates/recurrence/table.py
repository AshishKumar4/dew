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
    """A forked call's table as the walks take it (`chunk_forward.ForkedTable`, on a packed call's chunks,
    `chunked_launch.ForkedChunks`): each row's given memories `given` `[rows * count * HV * Dk, Dv]` fp32,
    laid as the memories are; the slot each chunk takes its memory up from, `loads` `[chunks]` (one-based,
    zero a zero memory, -1 none: the chunk continues the one before), and the stored slot it leaves its
    memory in, `stores` (zero none); each chunk's entry write `entries` `[chunks * HV * Dk, Dv]` fp32, zero
    but where it takes a document up, where the call gives one; the row each chunk walks (`rows` `[chunks]`,
    which the gradients' sums read and no launch); a row's slots, `count` given and `held` stored; the second
    slot each chunk taking a document up adds to its memory, `adds` `[chunks]` (one-based, zero none;
    `Chunk.adds`), where the call gives them; and where a turned call gives it, each chunk's load rotation
    `turning` (`turning_rows`: each key row's cosine and its pair partner's sines, `[chunks, HV * Dk]` fp32
    each), its memories then laid one row past each end (`pad`), so a row's partner a row away reads in
    bounds."""

    given: jax.Array
    loads: jax.Array
    stores: jax.Array
    entries: jax.Array | None
    rows: jax.Array
    count: int = dataclasses.field(metadata={'static': True})
    held: int = dataclasses.field(metadata={'static': True})
    adds: jax.Array | None = None
    turning: tuple | None = None

    @property
    def pad(self) -> int:
        """The rows its memories are laid past each end: one where it turns them (`turning`), else none."""
        return 0 if self.turning is None else 1

    @property
    def operands(self) -> tuple:
        """What a walk's launch reads of it, in order (`Taking.of`, `Returning.of`)."""
        return (self.given, self.loads, self.stores, *(() if self.adds is None else (self.adds,)),
                *(() if self.entries is None else (self.entries,)),
                *(() if self.turning is None else self.turning))

    @property
    def returning(self) -> tuple:
        """What a reverse walk's launch reads of it (`returned_by`): its loads, stores, added slots and load
        rotation."""
        return (self.loads, self.stores, *(() if self.adds is None else (self.adds,)),
                *(() if self.turning is None else self.turning))


def turning_rows(cos, sin, width: int) -> tuple:
    """A load rotation `(cos, sin)` `[..., A]` (`chunked_launch.Forked.turning`) as a memory's key rows take
    it `[..., width]` (`Taking`): each row's cosine, one past the `A` pairs, and its partner's coefficient, a
    pair's first row's from the row after it (minus the sine), its second's from the row before (the sine),
    each zero elsewhere, so a row turned is `cos x_r + after x_{r+1} + before x_{r-1}`."""
    pairs = cos.shape[-1]
    rest = width - 2 * pairs
    zero, one = jnp.zeros_like(cos), jnp.ones((*cos.shape[:-1], rest), cos.dtype)
    rows = jnp.concatenate([jnp.repeat(cos, 2, axis=-1), one], axis=-1)
    after = jnp.concatenate([jnp.stack([-sin, zero], axis=-1).reshape(*cos.shape[:-1], -1), 0 * one], axis=-1)
    before = jnp.concatenate([jnp.stack([zero, sin], axis=-1).reshape(*cos.shape[:-1], -1), 0 * one], axis=-1)
    return rows, after, before


@dataclasses.dataclass(frozen=True)
class Memories:
    """A role's view of a table's memories (`Table`): a memory `[Dk, Dv]` of `base` (a chunk, or a row's slot)
    the key slices `[block, bv]` of its head and the role's columns."""

    call: "Call"
    head: jax.Array
    column: jax.Array
    bv: int
    pad: int = 0

    def view(self, ref, base, i, by: int = 0):
        """Key slice `i` of the memory at `base` of `ref`, its rows `by` rows on (a turned row's partner)."""
        call = self.call
        first = (base * call.heads + self.head) * call.width + i * call.block + self.pad + by
        return ref.at[pl.ds(first, call.block), pl.ds(self.column, self.bv)]

    def views(self, ref, base) -> list:
        return [self.view(ref, base, i) for i in range(self.call.width // self.call.block)]

    def read(self, ref, base, i):
        """Key slice `i` of the memory at `base` of `ref`, in the walk's registers (`Family.tile_of`): a
        walk reads a memory a slice at a time, each beside the slices it carries alone."""
        return self.call.family.tile_of(self.view(ref, base, i))

    def turned(self, ref, base, i, turning, n, sign: float = 1.0):
        """Key slice `i` of the memory at `base` of `ref` turned by chunk `n`'s load rotation (`turning`, its
        refs `[chunks, HV * Dk]`, `turning_rows`), back by it where `sign` is -1: each row's cosine times it
        and its partner's coefficient times the partner, read a row on or back."""
        call, family = self.call, self.call.family
        span = pl.ds(self.head * call.width + i * call.block, call.block)
        rows, after, before = (family.rows_of(part.at[n, span])[:, None] for part in turning)

        def partner(coefficient, by):
            # Selected, not multiplied, where a row has no partner: the row a step on or back may be one no
            # walk wrote yet (a slot's neighbour, a pad), whose bits are anything.
            near = family.tile_of(self.view(ref, base, i, by))
            return jnp.where(coefficient != 0, sign * coefficient * near, 0.0)

        return rows * self.read(ref, base, i) + partner(after, 1) + partner(before, -1)

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

    turning: tuple | None = None

    @classmethod
    def of(cls, table: Table, refs: Sequence, stored, slots: Memories, row, smem) -> "Taking":
        """`table`'s walk from its refs as a launch reads them (`Table.operands`) and its stored slots'
        output `stored`."""
        given, loads, stores, *rest = refs
        adds, rest = (rest[0], rest[1:]) if table.adds is not None else (None, rest)
        turning = tuple(rest[-3:]) if table.turning is not None else None
        entries = rest[:-3] if table.turning is not None else rest
        return cls(slots, (given, loads, stores, adds, entries[0] if entries else None), stored, row,
                   table.count, table.held, smem, turning)

    def opened(self) -> None:
        """Nothing: a stored slot is written by a document before any later one reads it."""

    def before(self, n, state):
        given, loads, _, adds, entries = self.refs
        load = loads[n]

        def read(ref, base, i):
            # A turned call's memory turned by its document's load rotation as it is taken up.
            if self.turning is None:
                return self.slots.read(ref, base, i)
            return self.slots.turned(ref, base, i, self.turning, n)

        def slot(at, i):
            def from_given():
                return lax.cond(at > 0, lambda: read(given, self.row * self.count + at - 1, i),
                                self.slots.zero)

            def from_stored():
                return read(self.stored, self.row * self.held + at - self.count - 1, i)

            return lax.cond(at > self.count, from_stored, from_given) if self.held else from_given()

        def taken(i):
            memory = slot(load, i)
            if adds is not None:
                # The second slot a document adds to the one it enters (`Table.adds`), zero for none.
                memory = memory + slot(adds[n], i)
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
    turning: tuple | None = None
    adds: Any = None

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

        def returned(at, parts):
            # A stored slot's gradient gathers this chunk's `parts`: a slice at a time, each summed beside the
            # gradient's slices, then kept.
            @pl.when(at > self.count)
            def _():
                base = self.row * self.held + at - self.count - 1
                summed = [part + self.slots.read(self.returned, base, i) for i, part in enumerate(parts)]
                call.family.kept(summed, self.slots.views(self.returned, base), call.interpret,
                                 smem=self.smem)

        def taken():
            if self.turning is None:
                call.family.stored(list(grad), self.slots.views(self.taken, n), smem=self.smem)
            else:
                # Kept, so its rows turn back from global memory, each beside its partner.
                call.family.kept(list(grad), self.slots.views(self.taken, n), call.interpret, smem=self.smem)
            if self.held:
                # Turned back into its slot's frame where the call turns it.
                parts = (grad if self.turning is None else
                         [self.slots.turned(self.taken, n, i, self.turning, n, sign=-1.0)
                          for i in range(len(grad))])
                returned(load, parts)
                if self.adds is not None:
                    # The slot it added (`Table.adds`) takes the same gradient.
                    returned(self.adds[n], parts)
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
    return lambda role: Taking.of(table, refs, stored, Memories(call, head, column(role), bv, table.pad), row,
                                  landing(role))


def returned_by(table: Table | None, refs: Sequence, returned: Sequence, call: "Call", head,
                column: Callable, bv: int, row, landing: Callable) -> Callable | None:
    """A reverse walk's table (`sequence_walk`'s `table`): each role's `Returning` of `table`, its loads' and
    stores' refs `refs` and its outputs `returned` (`table_returned`'s), at the role's columns `column(role)`
    of `head`, `bv` wide, landing through `landing(role)`; None for a call without one."""
    if table is None:
        return None
    taken, *gathered = returned
    rest = refs[2:]
    adds = rest[0] if table.adds is not None else None
    turning = tuple(rest[-3:]) if table.turning is not None else None
    return lambda role: Returning(Memories(call, head, column(role), bv, table.pad), refs[0], refs[1], taken,
                                  gathered[0] if gathered else None, row, table.count, table.held,
                                  call.chunks, landing(role), turning, adds)


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
    return (jax.ShapeDtypeStruct((call.chunks * call.heads * call.width + 2 * table.pad, call.value_width),
                                 jnp.float32), *table_stored(call, table, batch))


def table_stored(call: "Call", table: Table | None, batch: int) -> tuple:
    """A forward walk's stored slots' output where its `table` holds any (`Taking`): `held` memories a row,
    fp32, as the memories are laid."""
    if table is None or not table.held:
        return ()
    return (jax.ShapeDtypeStruct((batch * table.held * call.heads * call.width + 2 * table.pad,
                                  call.value_width), jnp.float32),)


def table_landing(call: "Call", keeps: bool, bv: int) -> dict:
    """A walk's tile its table's memories land through (`Family.kept`), a key slice `[block, bv]` fp32, where
    it `keeps` any (a forward's stored slots, a reverse walk's taken gradients)."""
    return {"table": ((call.block, bv), jnp.float32)} if keeps else {}
