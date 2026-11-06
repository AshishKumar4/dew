"""Prepared operands (design §6, one protocol for every template): a tile in shared memory as the parts its
products take (`Parts`), and products issued on a family from such tiles or from register values, a register
operand staged first into its product's buffers, an fp32 one as its parts (`product.formed`), or taken in
registers where the family's products read them there (`Issuer`). The recurrence's products issue so; a
template's own parts move onto it."""

import functools
import itertools
from collections.abc import Callable, Iterable
from typing import Protocol

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import mosaic, product


class Issuing(Protocol):
    """What an `Issuer` reads of its call: the bundle its products issue by and whether they interpret."""

    @property
    def family(self) -> mosaic.Family: ...

    @property
    def interpret(self) -> bool: ...


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
            (buffer,) = self.buffers[f"{name}.exact"]
            return (self._staged(x.value, buffer, read),)
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
        values = product.formed(x, len(buffers)) if len(buffers) > 1 else (x,)
        return tuple(self._staged(value, buffer, read) for value, buffer in zip(values, buffers, strict=True))

    def _staged(self, value, buffer, read: list | None):
        """`value` staged into `buffer` (`Family.staged`), the products in flight done first where one may
        read it (`read`, then emptied), and `buffer` added to `read` for the products after it."""
        if read is not None:
            if any(buffer is tile for tile in read):
                self.call.family.settled(0)
                read.clear()
            read.append(buffer)
        return self.call.family.staged(value, buffer)

    def summed(self, products: Iterable, shape: tuple[int, int] | None = None, *, retired: bool = False,
               into=None):
        """The sum of `products`, each `(name, lhs, rhs)` as `__call__` takes it, issued into one fp32
        accumulator in place (`Family.issued`): one accumulator held where each its own held two (Kimi delta
        attention's output, `algebra.key_decayed_output`). Each product's operands are staged as it is
        issued, so `products` may form them one at a time (`algebra.key_block_pairs`, attention's parts), a
        staging into a tile an issued product may still read waiting for the products in flight first
        (`Family.settled`); `retired`, each product done before the next is formed, so a register operand's
        registers are free again (attention's parts, formed a part at a time). A product of a scaled operand
        (`product.Scaled`) takes an accumulator of its own and is restored (`restored`), the unscaled
        products before and after it each summed in one. An operand may be a callable forming it then, where
        `shape`, the sum's, is given. A register left operand whose product takes it in registers (its buffers
        a count) is issued so (`Family.register_products`). `into`, an accumulator the walk carries (its home,
        a warpgroup accumulator ref or a register array), takes every unscaled product in place and is
        returned, no value read; a scaled operand's product its own accumulator, restored and added into it
        (`Family.added_into`). A `product.Blocks` entry, a product a block of tiles in
        memory, is issued as a loop over its blocks where the family's products read their operands from
        registers (`Family.operands_in_registers`, whose accumulator is a value the loop carries): unrolled,
        ptxas held every block's fragments at once, an A100's scores backward at keys and values 256 wide
        spilling 280 bytes a thread (wide-kv at e1f928585); elsewhere a product a block, as listed."""
        products = self._entries(products)
        first = next(products)
        if not isinstance(first, product.Blocks) and isinstance(first[1], product.Scaled):
            # Scaled products (fp16's) each issue in a scope of their own (`Family.summed`): every part formed
            # here, before any, as a part formed in one product's scope (the next taken as it is issued) is
            # no value of the next's (Mosaic GPU: "captures non-scalar array constants", ET-FP16-h100).
            products = iter(list(products))
        family, interpret = self.call.family, self.call.interpret
        if shape is None:
            lhs, rhs = (first.left(0), first.right(0)) if isinstance(first, product.Blocks) else first[1:]
            shape = (lhs.shape[0], rhs.shape[1])
        held: list = [first]  # the next product not yet issued, where a group stopped before it

        def group(acc, alone: bool):
            # Products into `acc` until a scaled one (left held for its own group), or the one scaled
            # product alone.
            read: list = []
            while held:
                if isinstance(held[-1], product.Blocks):
                    # The accumulator laid out as products leave it, the loop's carry one layout.
                    blocks = held.pop()
                    acc = jax.lax.fori_loop(0, blocks.count, functools.partial(self._block, blocks),
                                            family.as_tile(acc))
                    held.extend(itertools.islice(products, 1))
                    continue
                name, lhs, rhs = held[-1]
                if isinstance(lhs, product.Scaled):
                    if not alone:
                        return acc
                    lhs = lhs.value
                held.pop()
                registered = self._registered(name, lhs)
                if registered is not None:
                    (right,) = self._operand(rhs, name, lhs, read)
                    acc = family.register_products(acc, registered, right, interpret)
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
            # The group at the held product: a scaled product's own accumulator, restored, or the unscaled
            # products from it to the next scaled one in one.
            # A sum of blocks is unscaled (`product.Blocks`).
            lhs = None if isinstance(held[-1], product.Blocks) else held[-1][1]
            scaled = lhs if isinstance(lhs, product.Scaled) else None
            part = family.summed(shape, lambda acc: group(acc, alone=scaled is not None), interpret)
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

    def _entries(self, products: Iterable):
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
        """`acc` with block `j`'s product of `blocks` issued onto it (`summed`'s loop)."""
        left, right = blocks.left(j), blocks.right(j)
        return self.call.family.issued(acc, left.refs, right.refs, self.call.interpret)

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
                    shape, lambda acc: family.register_products(acc, registered, right, interpret), interpret)
            # Each part's product its own accumulator, smallest first (`product.terms`), as `Family.apart`.
            return functools.reduce(jnp.add, (family.registers_product(registered[i], right, interpret)
                                              for i, _ in product.terms(len(registered), 1)))
        left = self._operand(lhs, name, rhs)
        right = self._operand(rhs, name, lhs)
        if max(len(left), len(right)) == 2:
            return family.summed(shape, lambda acc: family.issued(acc, left, right, interpret), interpret)
        return family.apart(shape, left, right, interpret)

