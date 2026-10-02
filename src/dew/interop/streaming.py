"""Translated checkpoint leaves that stay in the mapped file until a device reads them.

A loader that builds its whole tree on the host before placing it holds the
model once in anonymous memory, in FP32 by default, and a checkpoint larger
than the host does not load at all. A `SourceLeaf` is instead the recipe for
one leaf: the stored tensors it comes from (memory-mapped views), whether
their trailing pair of axes swaps (torch Linear `[out, in]` to Dense
`[in, out]`), whether they stack onto a leading expert axis, and the dtype
the leaf is kept in. `read(index)` builds only the part a device asks for,
and `dew.training.host.place_leaf` puts each part on its device before it
reads the next, so a process holds one device shard of one leaf at a time.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dew.objectives.base import Variables
from dew.training.host import evict


@dataclass(frozen=True, eq=False)
class SourceLeaf:
    """One translated leaf, read from its stored tensors on demand.

    `members` holds one tensor, or with `stacked` one per expert in expert
    order. `transposed` swaps each member's last two axes. `dtype` is the
    storage the leaf is read into, which casts only the read part.
    """

    members: tuple[np.ndarray, ...]
    dtype: np.dtype
    transposed: bool = False
    stacked: bool = False
    concatenated: bool = False

    @staticmethod
    def concatenate(leaves: Sequence[SourceLeaf]) -> SourceLeaf:
        """Keep a projection group's columns mapped until its device shard is read."""
        first = leaves[0]
        if any(leaf.stacked or leaf.concatenated or leaf.shape[:-1] != first.shape[:-1]
               or leaf.dtype != first.dtype or leaf.transposed != first.transposed for leaf in leaves):
            raise ValueError("concatenated source leaves must agree on dtype, input axes and layout")
        return SourceLeaf(tuple(leaf.members[0] for leaf in leaves), first.dtype,
                          first.transposed, concatenated=True)

    @staticmethod
    def stack(leaves: Sequence[SourceLeaf], name: str) -> SourceLeaf:
        """Stack single-tensor leaves onto a new leading axis, refusing leaves
        that disagree on shape, dtype or layout, as `np.stack` would."""
        first = leaves[0]
        disagree = sorted({(leaf.shape, str(leaf.dtype), leaf.transposed) for leaf in leaves})
        if len(disagree) != 1 or any(leaf.stacked or leaf.concatenated for leaf in leaves):
            raise ValueError(f"{name} members disagree: {disagree}")
        return SourceLeaf(tuple(leaf.members[0] for leaf in leaves), first.dtype,
                          first.transposed, stacked=True)

    @property
    def _member_shape(self) -> tuple[int, ...]:
        shape = self.members[0].shape
        return (*shape[:-2], shape[-1], shape[-2]) if self.transposed else shape

    @property
    def shape(self) -> tuple[int, ...]:
        if self.concatenated:
            return (*self._member_shape[:-1], sum(self._view(member).shape[-1] for member in self.members))
        return (len(self.members), *self._member_shape) if self.stacked else self._member_shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def _view(self, member: np.ndarray) -> np.ndarray:
        return np.swapaxes(member, -1, -2) if self.transposed else member

    def read(self, index: tuple[slice, ...] | None = None) -> np.ndarray:
        """The leaf's values at `index` (None: all of it), C-ordered in `dtype`.

        A read of the whole of a leaf that needs no cast and no transpose
        returns the stored view itself, so a checkpoint kept in its own
        dtype borrows the mapped bytes rather than copying them.
        """
        index = () if index is None else index
        if self.concatenated:
            index = (*index, *(slice(None) for _ in range(self.ndim - len(index))))
            shape = tuple(len(range(*part.indices(size)))
                          for part, size in zip(index, self.shape, strict=True))
            output = np.empty(shape, dtype=self.dtype)
            columns = np.arange(*index[-1].indices(self.shape[-1]))
            offset = 0
            step = index[-1].step or 1
            for member in self.members:
                view = self._view(member)
                positions = np.flatnonzero((columns >= offset) & (columns < offset + view.shape[-1]))
                if len(positions):
                    first, last = int(positions[0]), int(positions[-1])
                    stop = int(columns[last]) - offset + step
                    source = (*index[:-1], slice(int(columns[first]) - offset,
                                                None if step < 0 and stop < 0 else stop, step))
                    # Storage dtype is part of the recipe, as in the ordinary
                    # read: copy directly into the shard, without a cast buffer.
                    np.copyto(output[..., first:last + 1], view[source], casting="unsafe")
                offset += view.shape[-1]
            return output
        if not self.stacked:
            return np.asarray(self._view(self.members[0])[index], dtype=self.dtype, order="C")
        experts, rest = (index[0], index[1:]) if index else (slice(None), ())
        return np.stack([np.asarray(self._view(member)[rest], dtype=self.dtype)
                         for member in self.members[experts]])

    def release(self) -> None:
        """Give back the mapped pages the reads faulted in."""
        for member in self.members:
            evict(member)


type LazyTree = dict[str, np.ndarray | SourceLeaf | LazyTree]
"""A variables collection whose leaves are stored arrays or `SourceLeaf` recipes."""


def materialize(tree: LazyTree) -> Variables:
    """Read every `SourceLeaf` of `tree` whole, leaving arrays as they are."""
    return {name: (materialize(value) if isinstance(value, dict)
                   else value.read() if isinstance(value, SourceLeaf) else value)
            for name, value in tree.items()}
