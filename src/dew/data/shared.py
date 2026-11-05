"""An immutable mapping of fixed keys that shares its storage with the mappings it was made from.

A read's position is captured before each record it gives (a packer captures it before each document,
to resume at the one that did not fit), and a position must be the read's at that moment however the
read moves on. A mixture's position holds each member's, and between two captures one member moved:
copying every member's position at each capture costs a record what the mixture's size is (row 4079: a
thousand members' counts copied before each packed document). `Shared` holds its values in a trie of
`WIDTH` slots a node, so a mapping that differs from another in one value copies the nodes on that
value's path and shares the rest: a capture costs a record the trie's depth, two nodes at a thousand
members.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

WIDTH = 32
"""The slots of a trie node: a thousand values two nodes deep."""


class Shared[V](Mapping[str, V]):
    """An immutable mapping of `keys`, in their order, to values held in a trie shared with the mappings
    it was `replaced` from. It reads as any mapping (`dict(shared)` is its copy, as a position's JSON is
    written, `dew._records.json_value`)."""

    __slots__ = ("_depth", "_index", "_keys", "_root")

    def __init__(self, keys: Sequence[str], values: Sequence[V]):
        if len(keys) != len(values):
            raise ValueError(f"{len(keys)} keys and {len(values)} values")
        self._keys = tuple(keys)
        self._index = {key: place for place, key in enumerate(self._keys)}
        if len(self._index) != len(self._keys):
            raise ValueError("a shared mapping's keys are distinct")
        depth, span = 1, WIDTH
        while span < len(values):
            depth, span = depth + 1, span * WIDTH
        self._depth = depth
        self._root = _built(tuple(values), depth)

    @classmethod
    def of(cls, given: Mapping[str, V]) -> Shared[V]:
        return cls(tuple(given), tuple(given.values()))

    def replaced(self, changes: Mapping[str, V]) -> Shared[V]:
        """This mapping with each of `changes`' keys, every one among its own, holding its new value: the
        nodes on their paths copied, the rest shared."""
        if not changes:
            return self
        root = self._root
        for key, value in changes.items():
            root = _set(root, self._index[key], value, self._depth)
        made = object.__new__(Shared)
        made._keys, made._index, made._depth, made._root = self._keys, self._index, self._depth, root
        return made

    def __getitem__(self, key: str) -> V:
        place = self._index[key]
        node: tuple = self._root
        for level in range(self._depth - 1, 0, -1):
            node = node[place // WIDTH ** level % WIDTH]
        return node[place % WIDTH]

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f"Shared({dict(self)!r})"

    def __reduce__(self):
        return Shared, (self._keys, tuple(self[key] for key in self._keys))

    def __copy__(self) -> Shared[V]:
        return self

    def __deepcopy__(self, memo: dict) -> Shared[V]:
        """Itself: neither it nor the positions it holds are written once held, each a snapshot of its own,
        so a copy of it holds them as it does. Grain's prefetch copies a read's position deep for each
        element it reads ahead (`ThreadPrefetchDatasetIterator`), which, member by member, cost an element
        a mixture's size (row 4079)."""
        return self


def _built(values: tuple, depth: int) -> tuple:
    """The trie of `depth` levels holding `values` in order, each node `WIDTH` slots or fewer."""
    if depth == 1:
        return values
    span = WIDTH ** (depth - 1)
    return tuple(_built(values[start:start + span], depth - 1) for start in range(0, len(values), span))


def _set[V](node: tuple, place: int, value: V, depth: int) -> tuple:
    """`node`, a trie of `depth` levels, with the value at `place` replaced: the nodes on its path copied."""
    slot = place // WIDTH ** (depth - 1) % WIDTH
    held = value if depth == 1 else _set(node[slot], place, value, depth - 1)
    return (*node[:slot], held, *node[slot + 1:])

