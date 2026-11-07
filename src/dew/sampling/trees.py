"""Tree speculative decoding's tree (ledger 4527): its shape, and its verify block as the target reads it.

A drafter proposes a tree of candidate tokens after the row's next token, the root: several children a node,
each a guess at the token after its parent. The target verifies every node in one pass over its cache
(`TreeBlock`): the root and the nodes laid at the row's next cache slots in tree order, each node reading the
row's prefix and its own ancestors alone (`TreeBlock.runs`), at the place its depth gives it
(`TreeBlock.positions`), so each node's logits are the target's distribution after the path to it.
"""

import dataclasses
import functools
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np


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

    @property
    def fed_parents(self) -> tuple[int, ...]:
        """Each fed token's parent among the fed tokens, the root's -1 (the row's last cached token), as a
        cached call states its tree (`dew.nn.inputs.AttentionMetadata.tree`)."""
        return (-1, *(parent + 1 for parent in self.parents))

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
    `[rows]`."""

    shape: TreeShape
    cursor: jax.Array
    base: jax.Array

    def positions(self) -> jax.Array:
        """Each fed token's place `[rows, 1 + N]`: the root's `base`, a node's its depth past it."""
        depths = np.concatenate([[0], self.shape.depths])
        return self.base[:, None] + jnp.asarray(depths)[None, :]

    def key_positions(self, capacity: int | None) -> jax.Array:
        """Each of a row's `capacity` cache slots' place `[rows, capacity]`
        (`AttentionMetadata.key_positions`): the prefix's slots counted back from the root's place, a
        node's slot its depth past the root's. A tree is verified over a bounded cache."""
        assert capacity is not None, "a tree verifies over a cache of bounded capacity"
        rows = self.cursor.shape[0]
        slots = jnp.arange(capacity)[None, :]
        places = self.base[:, None] - self.cursor[:, None] + slots
        nodes = self.cursor[:, None] + 1 + jnp.arange(self.shape.nodes)[None, :]
        depths = self.base[:, None] + jnp.asarray(self.shape.depths)[None, :]
        return places.at[jnp.arange(rows)[:, None], nodes].set(depths, mode="drop").astype(jnp.int32)

    def runs(self) -> jax.Array:
        """Each fed token's key runs `[rows, 1 + N, R, 2]` (`AttentionMetadata.key_runs`), over the cache's
        slots: the row's prefix and the root, `[0, cursor + 1)`, then a node's ancestors and itself a slot
        each, `R` the deepest node's depth plus one, empty runs `[0, 0)` past a shallower one's."""
        shape = self.shape
        count = 1 + int(shape.depths.max())
        # Each fed token's ancestors' offsets past the root's slot, -1 past its path.
        offsets = np.full((shape.nodes + 1, count - 1), -1, np.int32)
        for node in range(shape.nodes):
            path = shape.ancestry(node)
            offsets[1 + node, :len(path)] = 1 + np.asarray(path)
        cursor = self.cursor[:, None, None]
        held = jnp.asarray(offsets)[None]
        starts = jnp.where(held >= 0, cursor + held, 0)
        ends = jnp.where(held >= 0, cursor + held + 1, 0)
        rows = self.cursor.shape[0]
        prefix = jnp.stack([jnp.zeros_like(self.cursor), self.cursor + 1], -1)[:, None, None]
        prefix = jnp.broadcast_to(prefix, (rows, shape.nodes + 1, 1, 2))
        return jnp.concatenate([prefix, jnp.stack([starts, ends], -1)], axis=2).astype(jnp.int32)


class TreeDraft(NamedTuple):
    """A drafter's tree (`DraftTree`): each row's node tokens `[rows, N]` in its shape's order, and the
    log distribution each was drawn from `[rows, N, V]`, None where every node is a pick (a top-k child's,
    its distribution one-hot at its token)."""

    tokens: jax.Array
    log_q: jax.Array | None


class DraftTree(Protocol):
    """`DecodeOps.draft_tree`: a tree of `shape`'s nodes after each row's `root` token `[rows]`, drawn with
    the rows' `keys` where it samples (`TreeDraft`)."""

    def __call__(self, state, root: jax.Array, shape: TreeShape, keys: jax.Array, /) -> TreeDraft: ...
