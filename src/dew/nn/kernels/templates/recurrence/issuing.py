"""The recurrence mainloop's products on its family (`templates.recurrence.mainloop`): a tile in shared memory
as the parts its products take (`Parts`), and the algebra's products (`recurrence_algebra.Issue`) issued on
the family, a register operand staged first (`Issuer`)."""

import functools
import itertools
from collections.abc import Iterable
from typing import Protocol

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import mosaic, product


class Issuing(Protocol):
    """What an `Issuer` reads of its call: the bundle its products issue by and whether they interpret."""

    family: mosaic.Family
    interpret: bool


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
    take the same value, staging a new one into the set its last product read longest ago. Operands of three
    parts issue the products whose parts' orders sum under three, smallest first; of two, under two."""

    def __init__(self, call: Issuing, buffers: dict | None = None):
        self.call, self.buffers = call, buffers or {}
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

    def summed(self, products: Iterable, shape: tuple[int, int] | None = None) -> jax.Array:
        """The sum of `products`, each `(name, lhs, rhs)` as `__call__` takes it, issued into one fp32
        accumulator in place (`Family.issued`): one accumulator held where each its own held two (Kimi delta
        attention's output, `algebra.key_decayed_output`). Each product's operands are staged as it is
        issued, so `products` may form them one at a time (`algebra.key_decayed_pairs`), a staging into a
        tile an issued product may still read waiting for the products in flight first
        (`Family.settled`). An operand may be a callable forming it then, where `shape`, the sum's, is
        given."""
        products = iter(products)
        first = next(products)
        family, interpret = self.call.family, self.call.interpret

        def issued(acc):
            read: list = []
            for name, lhs, rhs in itertools.chain([first], products):
                left, right = self._operand(lhs, name, rhs, read), self._operand(rhs, name, lhs, read)
                acc = family.issued(acc, left, right, interpret)
            return acc

        shape = (first[1].shape[0], first[2].shape[1]) if shape is None else shape
        return family.summed(shape, issued, interpret)

    def __call__(self, name: str, lhs, rhs) -> jax.Array:
        left = self._operand(lhs, name, rhs)
        right = self._operand(rhs, name, lhs)
        return self.call.family.apart((lhs.shape[0], rhs.shape[1]), left, right, self.call.interpret)
