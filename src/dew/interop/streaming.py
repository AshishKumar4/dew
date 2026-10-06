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

`WeightLayout` is the way back: which leaves one source tensor is built
from and how its storage is rebuilt, one tensor at a time.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dew.interop.weights import swapped
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

    def _member(self, member: np.ndarray, index: tuple[slice, ...]) -> np.ndarray:
        """One member's values at `index` of its view, C-ordered in `dtype`:
        read and cast in the stored layout, which is contiguous, then
        swapped in tiles (`swapped`) where the leaf is transposed."""
        if not self.transposed:
            return np.asarray(member[index], dtype=self.dtype, order="C")
        index = (*index, *(slice(None) for _ in range(member.ndim - len(index))))
        stored = (*index[:-2], index[-1], index[-2])
        return swapped(np.asarray(member[stored], dtype=self.dtype, order="C"))

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
                    if self.transposed:
                        output[..., first:last + 1] = self._member(member, source)
                    else:
                        # Storage dtype is part of the recipe: copy straight
                        # into the shard, without a cast buffer.
                        np.copyto(output[..., first:last + 1], member[source], casting="unsafe")
                offset += view.shape[-1]
            return output
        if not self.stacked:
            return self._member(self.members[0], index)
        experts, rest = (index[0], index[1:]) if index else (slice(None), ())
        return np.stack([self._member(member, rest) for member in self.members[experts]])

    def release(self) -> None:
        """Give back the mapped pages the reads faulted in."""
        for member in self.members:
            evict(member)


type LazyTree = dict[str, np.ndarray | SourceLeaf | LazyTree]
"""A variables collection whose leaves are stored arrays or `SourceLeaf` recipes."""


def materialize(tree: LazyTree) -> LazyTree:
    """Read every `SourceLeaf` of `tree` whole, leaving arrays as they are."""
    return {name: (materialize(value) if isinstance(value, dict)
                   else value.read() if isinstance(value, SourceLeaf) else value)
            for name, value in tree.items()}


@dataclass(frozen=True)
class WeightLayout:
    """Holds an existing source tensor's location and reversible storage layout.

    `expert_index` is the expert a per-expert source tensor holds. The
    loader stacks those tensors onto an expert dimension
    (`decoder_paths._stack_experts`), so one stacked leaf answers for every
    expert of a layer and the index says which slice this tensor is.

    `dtype` is the width the source stores this tensor in where that is
    not its leaf's: DeepSeek V4's token-to-expert table is int64 on disk
    and int32 in the collection, and the export writes back what the
    checkpoint held.

    `padded` is the length a 1-D source tensor stores past its leaf's, as
    zeros, for the names its family declares (`DecoderFamily.zero_padded`):
    Kimi K3 ships each KDA layer's `A_log` for 96 heads padded to 128
    entries. The family's prepare step checks and trims the tail, and export
    writes the zeros back.
    """

    name: str
    paths: tuple[tuple[str, ...], ...]
    shape: tuple[int, ...]
    transpose: tuple[int, ...] | None = None
    concatenate: int | None = None
    expert_index: int | None = None
    dtype: np.dtype | None = None
    padded: int | None = None

    def _leaf(self, variables: Mapping[str, object], path: tuple[str, ...],
              scalar_mode: str | None) -> np.ndarray | jax.Array:
        if path[-1] == "layer_scalar":
            if scalar_mode not in ("frozen", "trainable"):
                raise ValueError("layer_scalar export requires an explicit model mode")
            path = (("constants" if scalar_mode == "frozen" else "params"), *path[1:])
        node: object = variables
        for part in path:
            if not isinstance(node, Mapping):
                raise ValueError(f"parameter path {path} does not traverse a mapping")
            node = node[part]
        if not isinstance(node, (np.ndarray, jax.Array)):
            raise ValueError(
                f"{self.name} reads {path}, which holds {type(node).__name__} rather than an array"
            )
        return node

    def stored_dtype(self, variables: Mapping[str, object], scalar_mode: str | None = None) -> np.dtype:
        """The dtype `export` writes, read from the leaf without copying it."""
        if self.dtype is not None:
            return np.dtype(self.dtype)
        return np.dtype(self._leaf(variables, self.paths[0], scalar_mode).dtype)

    def export(self, variables: Mapping[str, object], scalar_mode: str | None = None) -> np.ndarray:
        """The source tensor, built from its leaves and brought to the host."""
        leaves = []
        for path in self.paths:
            node = self._leaf(variables, path, scalar_mode)
            if self.expert_index is not None:
                # Slice the expert where the leaf lives. One stacked leaf
                # answers for E source tensors, so copying it to the host
                # per tensor would move the whole stack E times.
                if node.ndim == 0 or not 0 <= self.expert_index < node.shape[0]:
                    raise ValueError(
                        f"{self.name} is expert {self.expert_index} of {path}, which "
                        f"holds {node.shape}")
                node = node[self.expert_index]
            leaves.append(node)
        if (len(leaves) == 1 and self.transpose is not None
                and jax.dtypes.canonicalize_dtype(leaves[0].dtype) == leaves[0].dtype):
            # One leaf is transposed on its device and copied to the host
            # contiguous: numpy copies a transposed bfloat16 kernel at 0.15
            # GiB/s on the RTX 3090 box's CPU, where the round trip over PCIe
            # moves several GiB/s.
            value = np.asarray(jnp.transpose(jnp.asarray(leaves[0]), self.transpose))
        else:
            arrays = [np.asarray(leaf) for leaf in leaves]
            value = arrays[0] if self.concatenate is None else np.concatenate(arrays, axis=self.concatenate)
            if self.transpose is not None:
                value = value.transpose(self.transpose)
        if self.padded is not None:
            value = np.pad(np.asarray(value), (0, self.padded - value.shape[0]))
        if value.size != math.prod(self.shape):
            raise ValueError(
                f"{self.name} assembles {value.shape} from {self.paths}, which does not "
                f"fill the source's {self.shape}")
        value = np.ascontiguousarray(value).reshape(self.shape)
        return value if self.dtype is None else value.astype(self.dtype)

    def restore(self, tensor: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Return the leaf of `shape` whose export is `tensor`.

        The inverse of `export` for a layout that binds one whole leaf; a
        tensor assembled from several leaves has no single leaf to restore.
        """
        if len(self.paths) != 1 or self.concatenate is not None or self.expert_index is not None:
            raise ValueError(f"{self.name} is assembled from several leaves, so no one leaf restores it")
        if tensor.shape != self.shape:
            raise ValueError(f"{self.name} stores {self.shape}, not {tensor.shape}")
        transpose = self.transpose or tuple(range(len(shape)))
        stored = tensor.reshape(tuple(shape[axis] for axis in transpose))
        return np.ascontiguousarray(stored.transpose(sorted(range(len(shape)), key=transpose.__getitem__)))
