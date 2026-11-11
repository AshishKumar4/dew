"""Tree speculative decoding's tree (ledger 4527): its shape, and its verify block as the target reads it.

A drafter proposes a tree of candidate tokens after the row's next token, the root: several children a node,
each a guess at the token after its parent. The target verifies every node in one pass over its cache
(`TreeBlock`): the root and the nodes laid at the row's next cache slots in tree order, each a document
continuing its parent's (`dew.batch.Forks`), so each node reads the row's prefix and its own ancestors alone,
at the place its depth gives it, and its logits are the target's distribution after the path to it.
"""

import dataclasses
import functools
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from dew.batch import Coordinates, Forks, coordinates


@dataclasses.dataclass(frozen=True)
class TreeShape:
    """A static tree of draft nodes after the root: `parents[j]` is node `j`'s parent, -1 the root (the row's
    next token, the target's own draw), each parent before its children (tree order)."""

    parents: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.parents or any(not -1 <= parent < node for node, parent in enumerate(self.parents)):
            raise ValueError("a tree's nodes follow their parents: parents[j] in [-1, j)")

    @classmethod
    def widths(cls, *widths: int) -> "TreeShape":
        """The full tree whose every node at depth `d` (the root's 0) has `widths[d]` children, in breadth
        order."""
        parents, level = [], [-1]
        for width in widths:
            following = []
            for parent in level:
                for _ in range(width):
                    following.append(len(parents))
                    parents.append(parent)
            level = following
        return cls(tuple(parents))

    @property
    def nodes(self) -> int:
        """The draft nodes, the root not among them."""
        return len(self.parents)

    @functools.cached_property
    def depths(self) -> np.ndarray:
        """Each draft node's depth `[N]`, a root's child's 1."""
        depths = np.zeros(self.nodes, np.int32)
        for node, parent in enumerate(self.parents):
            depths[node] = 1 if parent < 0 else depths[parent] + 1
        return depths

    @functools.cached_property
    def children(self) -> tuple[tuple[int, ...], ...]:
        """Each fed token's children, the root's first: `children[0]` the root's, `children[1 + j]` node
        `j`'s, in tree order."""
        found: list[list[int]] = [[] for _ in range(self.nodes + 1)]
        for node, parent in enumerate(self.parents):
            found[parent + 1].append(node)
        return tuple(tuple(each) for each in found)

    def ancestry(self, node: int) -> tuple[int, ...]:
        """Node `node`'s path from the root's child down to itself."""
        path = []
        while node >= 0:
            path.append(node)
            node = self.parents[node]
        return tuple(reversed(path))


class TreeBlock(NamedTuple):
    """A verify block of a tree (`TreeShape`) as the target reads it: the root and the draft nodes fed in tree
    order from each row's next cache slot `cursor` `[rows]` (its real tokens), the root at coordinate `base`
    `[rows]`. Each fed token is a document of its own continuing its parent's (`dew.batch.Forks`), the root's
    continuing the row's cache, so each reads the row's prefix and its own ancestors alone, at the place its
    depth gives it (`coordinates`)."""

    shape: TreeShape
    cursor: jax.Array
    base: jax.Array

    def coordinates(self, valid: jax.Array) -> Coordinates:
        """The fed tokens' layout `[rows, 1 + N]` for the cached call (`valid` its real tokens): a document a
        token, numbered in tree order from 1, each continuing its parent's, the root's the row's cache,
        each at the root's `base` plus its depth (`Forks.positions`)."""
        rows, fed = valid.shape
        segments = jnp.broadcast_to(jnp.arange(1, fed + 1, dtype=jnp.int32), (rows, fed))
        parents = np.asarray([0, *(parent + 2 for parent in self.shape.parents)], np.int32)
        forks = Forks(jnp.broadcast_to(jnp.asarray(parents), (rows, fed)),
                      depth=Forks.depth_of(parents[None]), held=Forks.held_of(parents[None]))
        laid = coordinates(segments=segments, valid=valid,
                           positions=forks.positions(segments, self.base, valid), max_segments=fed)
        return dataclasses.replace(laid, forks=forks)


class TreeDraft(NamedTuple):
    """A drafter's tree (`DraftTree`): each row's node tokens `[rows, N]` in its shape's order, and the
    log distribution each was drawn from `[rows, N, V]`, None where every node is a pick (a top-k child's,
    its distribution one-hot at its token)."""

    tokens: jax.Array
    log_q: jax.Array | None


class DraftTree(Protocol):
    """`DecodeOps.draft_tree`: a tree of `shape`'s nodes after each row's `root` token `[rows]`, drawn with
    the rows' `keys` where it samples (`TreeDraft`), the root at coordinate `base` `[rows]` and each node
    its depth past it."""

    def __call__(self, state, root: jax.Array, shape: TreeShape, keys: jax.Array, base: jax.Array,
                 /) -> TreeDraft: ...
