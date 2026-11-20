"""A kernel's outputs stated as its operands' axes: the one expression a launch's output shapes and the
dimensions each descends from are both derived from, so a template cannot state a map that disagrees with the
shapes it emits, and a buffer its kernel defines reads which operand axes it scales with
(`jax.stages.Compiled.peak_live`'s `dims`, through its custom call's `OUTPUT_DIMS`). A dimension is a
product of operand axes and constant sizes (`axis`, `Dim.__mul__`), divided or padded by a constant
(`Dim.__floordiv__`, `Dim.ceil_div`, `Dim.padded`): a packed row of a batch's sequences is the product of
their axes, a chunk count its sequence's axis over the chunk. An output stated as a plain
`jax.ShapeDtypeStruct` states none (unknown)."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence

import jax

OUTPUT_DIMS = "output_dims"
"""The frontend attribute a kernel's custom call states its outputs' dimensions by (`stated`): JSON, a list an
output (in the call's order of its stated outputs), an entry a dimension, each the `[operand, axis]` pairs it
is the product of, the operands in the call's order: empty for a constant size, null for an output stated with
none (its dimensions unknown)."""


@dataclasses.dataclass(frozen=True, eq=False)
class Dim:
    """An output dimension: the product of its `axes` (each an operand and one of its axes) and `constant`,
    divided by `divisor` (rounded up where `ceil`) and rounded up to a multiple of `multiple`."""

    axes: tuple[tuple[jax.Array, int], ...]
    constant: int = 1
    divisor: int = 1
    ceil: bool = False
    multiple: int = 1
    stated: int | None = None

    @property
    def size(self) -> int:
        """Its size, from its operands' own shapes, or the one its launch states (`sized`)."""
        if self.stated is not None:
            return self.stated
        product = self.constant * math.prod(operand.shape[index] for operand, index in self.axes)
        quotient = -(-product // self.divisor) if self.ceil else product // self.divisor
        return -(-quotient // self.multiple) * self.multiple

    def __mul__(self, other: Dim | int) -> Dim:
        if self.divisor != 1 or self.multiple != 1:
            raise ValueError("a dimension divided or padded is a product's last step, not a factor")
        if isinstance(other, int):
            return dataclasses.replace(self, constant=self.constant * other)
        if other.divisor != 1 or other.multiple != 1:
            raise ValueError("a dimension divided or padded is a product's last step, not a factor")
        return Dim((*self.axes, *other.axes), self.constant * other.constant)

    __rmul__ = __mul__

    def __floordiv__(self, divisor: int) -> Dim:
        return self._divided(divisor, ceil=False)

    def ceil_div(self, divisor: int) -> Dim:
        """It over `divisor`, rounded up (a tile count over a ragged edge)."""
        return self._divided(divisor, ceil=True)

    def sized(self, size: int) -> Dim:
        """It at `size`, which its launch computes of its axes otherwise than a product, a quotient or a
        padding states (a GEMM's rows: its walked rows less the extension past its last tile): the axes it
        descends from its own, its size the launch's."""
        return dataclasses.replace(self, stated=size)

    def padded(self, multiple: int) -> Dim:
        """It rounded up to a multiple of `multiple` (a tile's rows past a ragged edge)."""
        if self.multiple != 1:
            raise ValueError("a dimension is padded once")
        return dataclasses.replace(self, multiple=multiple)

    def _divided(self, divisor: int, *, ceil: bool) -> Dim:
        if self.divisor != 1 or self.multiple != 1:
            raise ValueError("a dimension is divided once, before it is padded")
        return dataclasses.replace(self, divisor=divisor, ceil=ceil)


def axis(operand: jax.Array, index: int) -> Dim:
    """Axis `index` of `operand`, one of the kernel's operands, as an output dimension."""
    if not -len(operand.shape) <= index < len(operand.shape):
        raise ValueError(f"an operand of shape {operand.shape} has no axis {index}")
    return Dim(((operand, index % len(operand.shape)),))


@dataclasses.dataclass(frozen=True, eq=False)
class Output:
    """A kernel output of `dtype` whose dimensions its operands' axes state (`Dim`; an int, a constant
    size)."""

    dims: tuple[Dim | int, ...]
    dtype: jax.typing.DTypeLike

    @property
    def shape(self) -> tuple[int, ...]:
        """Its sizes, its dimensions' (`Dim.size`), as a `jax.ShapeDtypeStruct`'s are read."""
        return tuple(dim if isinstance(dim, int) else dim.size for dim in self.dims)

    @property
    def struct(self) -> jax.ShapeDtypeStruct:
        """Its shape and dtype."""
        return jax.ShapeDtypeStruct(self.shape, self.dtype)


def output(dtype: jax.typing.DTypeLike, *dims: Dim | int) -> Output:
    """An output of `dtype` whose dimensions are `dims`, each its operands' axes (`axis`) or a constant
    size."""
    return Output(dims, dtype)


def _stated_leaf(leaf) -> bool:
    return isinstance(leaf, Output)


def dimensioned(out_type) -> bool:
    """Whether `out_type`, a kernel's outputs, states any as its operands' axes (`Output`)."""
    return any(isinstance(leaf, Output) for leaf in jax.tree.leaves(out_type, is_leaf=_stated_leaf))


def structs[Stated](out_type: Stated) -> Stated:
    """`out_type`, a kernel's outputs, each stated `Output` as its shape and dtype (`Output.struct`)."""
    return jax.tree.map(lambda leaf: leaf.struct if isinstance(leaf, Output) else leaf, out_type,
                        is_leaf=_stated_leaf)


def declared(out_type, operands: Sequence[jax.Array]) -> dict[str, str]:
    """The frontend attributes a kernel called with `operands` whose outputs are `out_type` declares its
    outputs' dimensions by (`OUTPUT_DIMS`, `stated`): none where it states no output as its operands' axes."""
    return {OUTPUT_DIMS: stated(out_type, operands)} if dimensioned(out_type) else {}


def stated(out_type, operands: Sequence[jax.Array]) -> str:
    """`OUTPUT_DIMS` of a kernel called with `operands` whose outputs are `out_type`: each output's dimensions
    as the `[operand, axis]` pairs they are the products of, an operand by its place among `operands`. An
    operand a dimension names that the call is not given is refused: its place would be another's."""
    def place(operand: jax.Array) -> int:
        found = [index for index, given in enumerate(operands) if given is operand]
        if not found:
            raise ValueError("an output dimension names an operand the kernel is not called with")
        return found[0]

    def dims(leaf) -> list | None:
        if not isinstance(leaf, Output):
            return None
        return [[] if isinstance(dim, int) else [[place(operand), index] for operand, index in dim.axes]
                for dim in leaf.dims]

    leaves = jax.tree.leaves(out_type, is_leaf=_stated_leaf)
    return json.dumps([dims(leaf) for leaf in leaves], separators=(",", ":"))
