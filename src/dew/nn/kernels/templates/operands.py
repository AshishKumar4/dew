"""Prepared operands (design §6, one protocol for every template): a tile in shared memory as the parts its
products take (`Parts`), and products issued on a family from such tiles or from register values, a register
operand staged first into its product's buffers, an fp32 one as its parts (`product.formed`), or taken in
registers where the family's products read them there (`Issuer`). The recurrence's products issue so; a
template's own parts move onto it."""

import functools
import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels.templates import mosaic, product

# A product `Issuer.summed` takes: a block of tiles in memory, or its name and its two operands.
Entry = product.Blocks | tuple


class Issuing(Protocol):
    """What an `Issuer` reads of its call: the bundle its products issue by and whether they interpret."""

    @property
    def family(self) -> mosaic.Family: ...

    @property
    def interpret(self) -> bool: ...


def staged(family: mosaic.Family, value: jax.Array, buffers: Sequence[mosaic.Ref], *,
           before: Callable[[mosaic.Ref], None] | None = None) -> "Parts":
    """`value` staged once in shared memory as the parts its products take, one a buffer of `buffers`: its
    16-bit parts where they are more than one (`product.formed`), itself in one (`Family.staged`), each
    after `before(buffer)` (an issuer's wait for the products in flight that may read it): an issuer's
    operands' staging (`Issuer._operand`)."""
    values = product.formed(value, len(buffers)) if len(buffers) > 1 else (value,)
    refs = []
    for part, buffer in zip(values, buffers, strict=True):
        if before is not None:
            before(buffer)
        refs.append(family.staged(part, buffer))
    return Parts(refs, value.dtype)


def _in_memory(x) -> bool:
    """Whether operand `x` is a tile in shared memory already (`Parts`, or one read transposed), which its
    product reads staging nothing."""
    return isinstance(x, Parts) or (isinstance(x, product.Transposed) and _in_memory(x.value))


class Parts:
    """A tile in shared memory as the 16-bit parts its products take (one, or an fp32 tile's three,
    `product.formed`), each a view of a stream's slot, and `dtype` the tile's own."""

    def __init__(self, refs, dtype):
        self.refs, self.dtype = tuple(refs), jnp.dtype(dtype)

    @property
    def shape(self) -> tuple:
        return self.refs[0].shape

    @property
    def T(self) -> "Parts":
        return Parts((ref.T for ref in self.refs), self.dtype)

    def rows(self, start, count: int) -> "Parts":
        """Its rows `start` to `start + count`, each part's: a product of a block of them, the rest of the
        tile staged once for the others (KDA's pairs, a key sub-chunk's)."""
        return Parts((ref.at[pl.ds(start, count)] for ref in self.refs), self.dtype)

    def astype(self, dtype) -> "Parts":
        """The tile as `dtype`'s, which a product takes it as: its own dtype alone."""
        if jnp.dtype(dtype) != self.dtype:
            raise TypeError(f"a {self.dtype} tile in shared memory taken as {jnp.dtype(dtype)}")
        return self

    def value(self) -> jax.Array:
        """The tile in registers, in its dtype: its parts summed in fp32, exactly."""
        if len(self.refs) == 1:
            return self.refs[0][...]
        return functools.reduce(jnp.add, (ref[...].astype(jnp.float32) for ref in self.refs))


class Issuer:
    """The algebra's products (`algebra.Issue`) on the call's family: an operand a `Parts` tile in shared
    memory, or a register value the family stages first (`staged`) into its product's `buffers` (by product
    name, a tile a part), an fp32 one as its parts, as many as its buffers (one, a cast to theirs). A product
    whose buffers are a list of such sets keeps each value it stages there for the products after it that
    take the same value, staging a new one into the set its last product read longest ago. A product whose
    buffers are a count takes a register left operand in that many parts as they are, staged nowhere, where
    the family's product takes its left operand in registers (`Family.register_products`: mma.sync's
    fragments, wgmma's register form). Operands of three parts issue the products whose parts' orders sum
    under three, smallest first; of two, under two. A scaled operand (`product.Scaled`) takes its product
    into an accumulator of its own, restored by the algebra's `restored(value, exponent)` (attention's
    exact power of two)."""

    def __init__(self, call: Issuing, buffers: dict | None = None, restored: Callable | None = None):
        self.call, self.buffers, self.restored = call, buffers or {}, restored
        # By product name: each set's value (held, so its identity stays its own) and its last use.
        self.held: dict = {}

    def _operand(self, x, name: str, other=None, read: list | None = None) -> tuple:
        """`x`'s parts' shared-memory tiles, staged into its product's buffers where it is a register value;
        `read` the tiles products issued before it may still read (`summed`'s), waited for where it would
        stage into one of them."""
        if callable(x):
            # An operand formed when its product stages it (`summed`'s).
            return self._operand(x(), name, other, read)
        if isinstance(x, Parts):
            return x.refs
        if isinstance(x, product.Exact):
            # One part, in the product's own tile for exact operands (`{name}.exact`).
            return staged(self.call.family, x.value, self.buffers[f"{name}.exact"],
                          before=functools.partial(self._settle, read=read)).refs
        if isinstance(x, product.Transposed):
            # Staged as it is and read transposed (`product.Transposed`).
            return tuple(ref.T for ref in self._operand(x.value, name, other, read))
        buffers = self.buffers[name]
        if isinstance(buffers, list):
            held = self.held.setdefault(name, [[None, -1] for _ in buffers])
            uses = max(use for _, use in held) + 1
            for place, (value, _) in enumerate(held):
                if value is x:
                    held[place][1] = uses
                    return buffers[place]
            free = [place for place in range(len(buffers)) if held[place][0] is not other]
            place = min(free, key=lambda place: held[place][1])
            held[place] = [x, uses]
            buffers = buffers[place]
        return staged(self.call.family, x, buffers, before=functools.partial(self._settle, read=read)).refs

    def _settle(self, buffer, read: list | None) -> None:
        """Ahead of a staging into `buffer`: the products in flight done first where one may read it (`read`,
        then emptied), and `buffer` added to `read` for the products after it."""
        if read is not None:
            if any(buffer is tile for tile in read):
                self.call.family.settled(0)
                read.clear()
            read.append(buffer)

    def summed(self, products: Sequence[Entry] | Iterator[Entry], shape: tuple[int, int] | None = None, *,
               retired: bool = False, into=None):
        """The sum of `products`, each `(name, lhs, rhs)` as `__call__` takes it, issued into one fp32
        accumulator in place (`Family.issued`): one accumulator held where each its own held two (Kimi delta
        attention's output, `algebra.key_decayed_output`). Each product's operands are staged as it is issued,
        so `products` may form them one at a time (`algebra.key_block_pairs`, attention's parts), a staging
        into a tile an issued product may still read waiting for the products in flight first
        (`Family.settled`); `retired`, each product done before the next is formed, so a register operand's
        registers are free again (attention's parts, formed a part at a time). A product of a scaled operand
        (`product.Scaled`) takes an accumulator of its own and is restored (`restored`), the unscaled products
        before and after it each summed in one. An operand may be a callable forming it then, where `shape`,
        the sum's, is given. A register left operand whose product takes it in registers (its buffers a count)
        is issued so (`Family.register_products`). Products whose operands take three parts (an fp32 call's
        accurate forms) are summed whole, each its terms apart (`_apart`, as `__call__`), and added in its
        place among the rest, a group of its own as a scaled product's is (`_whole`): the sum's one order,
        whatever its products are given as (a list, a stream, a callable's, blocks). `into`, an accumulator
        the walk carries (its home, a warpgroup accumulator ref or a register array), takes every unscaled
        product in place and is returned, no value read; a scaled operand's product its own accumulator,
        restored and added into it (`Family.added_into`). A `product.Blocks` entry, a product a block of tiles
        in memory, is issued as a loop over its blocks where the family's products read their operands from
        registers (`Family.operands_in_registers`, whose accumulator is a value the loop carries): unrolled,
        ptxas held every block's fragments at once, an A100's scores backward at keys and values 256 wide
        spilling 280 bytes a thread (wide-kv at e1f928585); elsewhere a product a block, as listed."""
        family, interpret = self.call.family, self.call.interpret
        rest = iter(self._entries(products))
        first = next(rest, None)
        if first is None:
            raise ValueError("a sum of no products")
        products = rest
        if not isinstance(first, product.Blocks) and isinstance(first[1], product.Scaled):
            # Scaled products (fp16's) each issue in a scope of their own (`Family.summed`): every part formed
            # here, before any, as a part formed in one product's scope (the next taken as it is issued) is
            # no value of the next's (Mosaic GPU: "captures non-scalar array constants", ET-FP16-h100).
            products = iter(list(products))
        if shape is None:
            lhs, rhs = (first.left(0), first.right(0)) if isinstance(first, product.Blocks) else first[1:]
            shape = (lhs.shape[0], rhs.shape[1])
        held: list = [first]  # the next product not yet issued, where a group stopped before it

        def group(acc, alone: bool):
            # Products into `acc` until a scaled one or one of three parts (left held for its own group),
            # or the one scaled product alone.
            read: list = []
            while held:
                if isinstance(held[-1], product.Blocks):
                    # The accumulator laid out as products leave it, the loop's carry one layout.
                    blocks = held.pop()
                    acc = jax.lax.fori_loop(0, blocks.count, functools.partial(self._block, blocks),
                                            family.as_tile(acc))
                    held.extend(itertools.islice(products, 1))
                    continue
                held[-1] = name, lhs, rhs = self._formed(held[-1])
                if self._whole(name, lhs, rhs):
                    return acc
                if isinstance(lhs, product.Scaled):
                    if not alone:
                        return acc
                    lhs = lhs.value
                held.pop()
                registered = self._registered(name, lhs)
                if registered is not None:
                    (right,) = self._operand(rhs, name, lhs, read)
                    acc = family.register_products(acc, registered, right)
                else:
                    left, right = self._operand(lhs, name, rhs, read), self._operand(rhs, name, lhs, read)
                    acc = family.issued(acc, left, right, interpret)
                if retired:
                    family.settled(0)
                held.extend(itertools.islice(products, 1))
                if alone:
                    return acc
            return acc

        def summed_group() -> jax.Array:
            # The group at the held product: a product of three parts whole (`_apart`), a scaled product's
            # own accumulator, restored, or the unscaled products from it to the next such one in one.
            # A sum of blocks is unscaled (`product.Blocks`).
            if not isinstance(held[-1], product.Blocks):
                held[-1] = name, lhs, rhs = self._formed(held[-1])
                if self._whole(name, lhs, rhs):
                    held.pop()
                    whole = self._apart(name, lhs, rhs, shape)
                    if retired:
                        family.settled(0)
                    held.extend(itertools.islice(products, 1))
                    return whole
            lhs = None if isinstance(held[-1], product.Blocks) else held[-1][1]
            scaled = lhs if isinstance(lhs, product.Scaled) else None
            part = family.summed(shape, lambda acc: group(acc, alone=scaled is not None))
            if scaled is None:
                return part
            if self.restored is None:
                raise ValueError("a scaled operand's product needs the algebra's restore (`restored`)")
            return self.restored(part, scaled.exponent)

        if into is not None:
            into = group(into, alone=False)
            while held:
                # A scaled product's own accumulator, restored, then the carried one's (fp16's parts: the
                # carried accumulator rescaled in place would leave fp32's range at either end, its small
                # entries subnormal under a large part's power or its large ones past it under a small's),
                # the products in flight on the carried one done first (`Family.added_into`).
                part = summed_group()
                into = family.added_into(into, part, interpret)
            return into
        total = summed_group()
        while held:
            total = total + summed_group()
        return total

    def _formed(self, entry: tuple) -> tuple:
        """`entry` (`name, lhs, rhs`) with an operand a callable forms formed, once, as its product issues
        (`summed` holds the formed entry for the group that issues it)."""
        name, lhs, rhs = entry
        return name, lhs() if callable(lhs) else lhs, rhs() if callable(rhs) else rhs

    def _whole(self, name: str, lhs, rhs) -> bool:
        """Whether the product of `lhs` and `rhs` (formed, `_formed`) takes three parts of an operand (an fp32
        call's accurate forms): summed whole, its terms apart (`_apart`), and added in its place among the
        sum's products, so a sum's partial sums are its whole products' in their order (base reviews
        0dac2d104, 60efbe75f, 670386203, 6da3c48c8 F1/F2: a sum's terms or its products set aside left
        float32's range where the products in order stayed in it)."""
        if isinstance(lhs, product.Scaled):
            return False
        registered = self._registered(name, lhs)
        if registered is not None:
            return len(registered) == 3
        return max(self._count(name, lhs), self._count(name, rhs)) == 3

    def _apart(self, name: str, lhs, rhs, shape: tuple[int, int]) -> jax.Array:
        """The product of three parts `lhs @ rhs` (`_whole`) as `__call__` takes it: each term its own fresh
        accumulator, added smallest first (`Family.apart`; a register left operand's parts, a product
        each)."""
        family, interpret = self.call.family, self.call.interpret
        registered = self._registered(name, lhs)
        if registered is not None:
            (right,) = self._operand(rhs, name, lhs)
            return functools.reduce(jnp.add, (family.registers_product(registered[i], right)
                                              for i, _ in product.terms(len(registered), 1)))
        return family.apart(shape, self._operand(lhs, name, rhs), self._operand(rhs, name, lhs), interpret)

    def _count(self, name: str, x) -> int:
        """How many parts `_operand` issues operand `x` of product `name` in: a tile in memory's own, an exact
        value's one, a register value's (or one formed later's) its product's buffers' (`staged`)."""
        if isinstance(x, Parts):
            return len(x.refs)
        if isinstance(x, product.Exact):
            return 1
        if isinstance(x, product.Transposed):
            return self._count(name, x.value)
        buffers = self.buffers[name]
        return len(buffers[0] if isinstance(buffers, list) else buffers)

    def summed_together(self, groups: list) -> tuple[Callable[[], jax.Array], ...]:
        """Readers of `groups`' sums (each a list `summed` takes), none of which reads another's: issued
        together where every product is an unscaled pair of operands of at most two parts each, formed (no
        callable), with no register left operand and no `product.Blocks` (`Family.summed_together`: one wait
        for them all where products are awaited, tcgen05's), each its own accumulator; else each summed where
        it is read (`summed`, the sum's one order: a product of three parts whole in its place), in the
        reader's order, as `Family.summed_together`'s default is, its products formed as that sum issues
        them. Groups an operand of which is staged as its product issues (a register value, an exact one) are
        issued apart, each awaited before the next (`apart`): a staging waits for the products in flight that
        may read its tile (`_settle`) within its group alone, so another group's product still reading the
        tile would read the new operand."""
        plain = all(self._together(entry) for group in groups for entry in group)
        if not plain:
            return tuple(functools.partial(self.summed, group) for group in groups)
        triples = [list(group) for group in groups]
        family, interpret = self.call.family, self.call.interpret

        def issued(found: list, acc):
            read: list = []
            for name, lhs, rhs in found:
                left, right = self._operand(lhs, name, rhs, read), self._operand(rhs, name, lhs, read)
                acc = family.issued(acc, left, right, interpret)
            return acc

        staged = not all(_in_memory(lhs) and _in_memory(rhs) for found in triples for _, lhs, rhs in found)
        shapes = tuple((found[0][1].shape[0], found[0][2].shape[1]) for found in triples)
        return family.summed_together(shapes, tuple(functools.partial(issued, found) for found in triples),
                                      apart=staged)

    def _together(self, entry: Entry) -> bool:
        """Whether `entry` issues together with its group's others (`summed_together`): an unscaled pair of
        formed operands of at most two parts, no register left operand, no blocks."""
        if isinstance(entry, product.Blocks):
            return False
        name, lhs, rhs = entry
        return (not callable(lhs) and not callable(rhs) and not isinstance(lhs, product.Scaled)
                and self._registered(name, lhs) is None
                and max(self._count(name, lhs), self._count(name, rhs)) < 3)

    def _entries(self, products: Iterable) -> Iterator[Entry]:
        """`products` as `summed` issues them: a `product.Blocks` it loops over as it is, more than one block
        of tiles in memory (`Parts`) on a family whose products read registers; any other as its blocks'
        products, one a block."""
        for entry in products:
            as_is = (not isinstance(entry, product.Blocks)
                     or (entry.count > 1 and self.call.family.operands_in_registers
                         and isinstance(entry.left(0), Parts) and isinstance(entry.right(0), Parts)))
            if as_is:
                yield entry
            else:
                yield from ((entry.name, entry.left(j), entry.right(j)) for j in range(entry.count))

    def _block(self, blocks: product.Blocks, j, acc):
        """`acc` with block `j`'s product of `blocks` issued onto it (`summed`'s loop): a product of three
        parts whole first, its terms apart (`Family.apart`), then added (`Family.added_into`), as `summed`
        adds one in its place."""
        family, interpret = self.call.family, self.call.interpret
        left, right = blocks.left(j).refs, blocks.right(j).refs
        if max(len(left), len(right)) < 3:
            return family.issued(acc, left, right, interpret)
        shape = (left[0].shape[0], right[0].shape[1])
        return family.added_into(acc, family.apart(shape, left, right, interpret), interpret)

    def _registered(self, name: str, lhs) -> tuple | None:
        """`lhs`'s parts in registers where its product takes it there (its buffers a count, `Issuer`'s),
        else None."""
        count = self.buffers.get(name)
        if not isinstance(count, int):
            return None
        value = lhs() if callable(lhs) else lhs
        if not isinstance(value, jax.Array):
            # A shared tile (`Parts`), or an operand staged its own way (`product.Exact`, `Transposed`).
            return None
        return product.formed(value, count) if count > 1 else (value,)

    def __call__(self, name: str, lhs, rhs) -> jax.Array:
        """The product `lhs @ rhs`: its operands' parts' products (`product.terms`), each into its own
        accumulator where an operand takes three parts (an fp32 call's accurate forms, `Family.apart`: a part
        accumulated onto a larger one loses its last bit), else into one in place. Two parts are the fast
        forms' (`mainloop.FAST_PARTS`), TF32's precision and past it, which that last bit is far below; apart,
        an fp32 call's solve held its products' accumulators at once, 576 bytes a thread spilled on an H100
        (fast32d-sm90 at 83b47d66f6)."""
        family, interpret = self.call.family, self.call.interpret
        shape = (lhs.shape[0], rhs.shape[1])
        registered = self._registered(name, lhs)
        if registered is not None:
            (right,) = self._operand(rhs, name, lhs)
            if len(registered) <= 2:
                return family.summed(
                    shape, lambda acc: family.register_products(acc, registered, right))
            # Each part's product its own accumulator, smallest first (`product.terms`), as `Family.apart`.
            return functools.reduce(jnp.add, (family.registers_product(registered[i], right)
                                              for i, _ in product.terms(len(registered), 1)))
        left = self._operand(lhs, name, rhs)
        right = self._operand(rhs, name, lhs)
        if max(len(left), len(right)) == 2:
            return family.summed(shape, lambda acc: family.issued(acc, left, right, interpret))
        return family.apart(shape, left, right, interpret)

