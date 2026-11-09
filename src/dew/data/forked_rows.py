"""Rows that share a prefix, computed once (`ForkedRows`).

A scoring or training call often holds rows that begin alike: a request's questions over one state, a
prompt's labels or completions, a preference pair's two halves, a group's sampled chains. Laid out each
whole, every row computes the prefix again. `ForkedRows.pack` groups the rows by their longest common
prefix, lays each group's prefix once as a document and each row's rest as a document continuing it
(`dew.batch.Forks`), and packs the groups first-fit into windows. A model reads the windows as it reads
any packed rows; each row's tokens map back to their slots (`ForkedRows.gathered`), so whatever reads a
row's outputs reads them in the row's own layout, and a gradient taken through that map sums each shared
token's cotangent over the rows that hold it, as the rows laid out whole would.

The shared form is the rows' own exactly where a model computes each document from its ancestors and
itself alone (`dew.nn.protocols.ForkedCalls`) and draws nothing at random; a caller lays the rows whole
(`share=False`) where it does not, through the same packer.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from dew.batch import Coordinates, Forks, coordinates
from dew.data.tokens import common_prefix, first_fit


@struct.dataclass
class ForkedRows:
    """Rows packed once a shared prefix (module docstring).

    `tokens` `[P, W]` are the windows' ids and `coordinates` their layout, each group's prefix a document
    and each row's rest a document continuing it (`Coordinates.forks`), each token at its own row's place.
    `index` `[N, L]` is each row's token's slot within its block's windows (`window * W + slot`), -1 past
    the row's end, and `fields` the per-token arrays the rows carried, laid out as `tokens` are. The rows
    are cut into `blocks` contiguous blocks, each packed into its own `P / blocks` windows, so a mesh that
    splits the rows and the windows alike by block reads each row's outputs on the device that computed
    them."""

    tokens: jax.Array
    coordinates: Coordinates
    index: jax.Array
    fields: Mapping[str, jax.Array] = struct.field(default_factory=dict)
    blocks: int = struct.field(pytree_node=False, default=1)

    @property
    def rows(self) -> int:
        """The rows it lays out, `N`."""
        return self.index.shape[0]

    def gathered(self, values: jax.Array) -> jax.Array:
        """`values` `[P, W, ...]` read at each row's tokens' slots, `[N, L, ...]`, zero past a row's end:
        a row's outputs in its own layout. Its gradient adds each slot's cotangents over every row that
        reads it."""
        blocks = self.blocks
        windows, width = values.shape[:2]
        rest = values.shape[2:]
        flat = values.reshape(blocks, (windows // blocks) * width, *rest)
        index = self.index.reshape(blocks, -1, self.index.shape[-1])

        def taken(block, at):
            picked = jnp.take(block, jnp.maximum(at, 0), axis=0)
            return jnp.where((at >= 0).reshape(*at.shape, *(1,) * len(rest)), picked, 0)

        out = jax.vmap(taken)(flat, index)
        return out.reshape(self.rows, self.index.shape[-1], *rest)

    def targets(self, values: jax.Array) -> jax.Array:
        """A next-token loss's `values` over the windows `[P, W - 1, ...]` (the token at `t + 1`'s at `t`, as
        `TokenScorer.token_scores` gives them over rows whose coordinates carry forks) read for each row,
        `[N, L - 1, ...]`, its token at `t + 1`'s at `t`: each token's value scored from its predecessor
        (`dew.batch.Forks.predecessors`), zero past a row's end."""
        widened = jnp.pad(values, ((0, 0), (1, 0), *((0, 0),) * (values.ndim - 2)))
        return self.gathered(widened)[:, 1:]

    @classmethod
    def needed(cls, rows: Sequence[Sequence[int]], width: int, *, share: bool = True,
               positions: Sequence[Sequence[int]] | None = None,
               fields: Mapping[str, Sequence[np.ndarray]] | None = None) -> int:
        """How many windows of `width` slots `pack` lays `rows` out in, one block (`windows_needed`)."""
        laid, _, _, keys = _keyed(rows, positions, fields, share)
        return windows_needed([row.size for row in laid], width, keys=keys)

    @classmethod
    def pack(cls, rows: Sequence[Sequence[int]], width: int, *, windows: int | None = None,
             length: int | None = None, documents: int | None = None, blocks: int = 1, share: bool = True,
             positions: Sequence[Sequence[int]] | None = None,
             fields: Mapping[str, Sequence[np.ndarray]] | None = None, xp=np) -> ForkedRows:
        """`rows` (token ids, each at least one) packed into windows of `width` slots, each group of rows
        that begin alike over one copy of their common prefix.

        Within each of `blocks` contiguous blocks of rows, the rows are sorted and cut into groups, each
        the run of sorted rows that lays out fewest tokens over one copy of its common prefix and fits a
        window; a group's prefix is the longest its rows share, its tokens and every one of `fields` alike
        (a per-token array a row's tokens carry, `[len(row), ...]`, such as a rollout's routing records,
        laid out beside the tokens). The groups are packed first-fit, longest first, into
        `windows / blocks` windows a block (as many as they need where None; refused where they need
        more); `share=False` lays each row whole, packed as the groups are, at most `documents` a window,
        so `documents=1` lays each row in a window of its own, as the rows laid out alone are.
        `positions` are each row's tokens' places, its indices where None. `length` is the rows' laid
        length `L` (the longest row's where None) and `documents` the documents a window may hold (the
        most any holds where None): stated, they keep a compiled program's shapes across calls. `xp` is
        the array module the result is in."""
        if width < 1 or blocks < 1:
            raise ValueError(f"windows hold at least one slot in at least one block, got width {width} and "
                             f"{blocks} blocks")
        laid, places, extra, keys = _keyed(rows, positions, fields, share)
        if len(laid) % blocks:
            raise ValueError(f"{len(laid)} rows split into {blocks} blocks, and they do not divide")
        longest = max(row.size for row in laid)
        length = longest if length is None else length
        if longest > min(length, width):
            raise ValueError(f"a row of {longest} tokens fits neither the rows' length {length} nor a window "
                             f"of {width} slots")
        per_block = len(laid) // blocks
        plans = []
        for block in range(blocks):
            members = range(block * per_block, (block + 1) * per_block)
            planned = _planned([laid[number].size for number in members], width,
                               None if keys is None else [keys[number] for number in members], documents)
            plans.append([[(prefix, [members[at] for at in group]) for prefix, group in window]
                          for window in planned])
        needed = max(len(plan) for plan in plans)
        held = needed if windows is None else windows // blocks
        if windows is not None and (windows % blocks or needed > held):
            raise ValueError(f"the rows need {needed} windows a block of {width} slots, past the "
                             f"{windows} asked for over {blocks} blocks")
        count = max(_documents(window, laid) for plan in plans for window in plan)
        count = count if documents is None else documents
        layout = _Layout(blocks * held, width, count, length, len(laid), extra, laid)
        for block, plan in enumerate(plans):
            for at, window in enumerate(plan):
                layout.write(block * held + at, at, window, laid, places, extra)
        return layout.forked(blocks, xp)


def window_forks(segments: np.ndarray, local: np.ndarray, parent: np.ndarray,
                 documents: int) -> tuple[np.ndarray, np.ndarray]:
    """A packed window's documents `[W]` and the document each continues `[documents]` (`dew.batch.Forks`),
    from each token's row in the window (`segments`, from 1, 0 for padding), its document in its row
    (`local`, from 1) and that document's parent in its row (`parent`, by its local number, 0 for none):
    the window's documents numbered from 1 in the order they stand, each continuing its row's parent's.
    A window holding more than `documents` is refused: its rows' bounds state the most it holds."""
    segments, local, parent = (np.asarray(x, np.int64) for x in (segments, local, parent))
    width = int(local.max(initial=0)) + 1
    real = segments > 0
    keys = segments * width + local
    standing, numbered = np.unique(keys[real], return_inverse=True)
    if len(standing) > documents:
        raise ValueError(f"a window holds {len(standing)} documents, past the {documents} its rows' bounds "
                         "allow")
    laid = np.zeros(segments.shape, np.int32)
    laid[real] = numbered + 1
    # Each document's parent's key in its row, its first token's word for it.
    parents = np.zeros(documents, np.int32)
    above = np.zeros(len(standing), np.int64)
    np.put(above, numbered, (segments * width + parent)[real])
    held = (above % width) > 0
    parents[:len(standing)][held] = np.searchsorted(standing, above[held]) + 1
    return laid, parents


def windows_needed(lengths: Sequence[int], width: int, *, keys: Sequence[Sequence[bytes]] | None = None,
                   documents: int | None = None) -> int:
    """How many windows of `width` slots `ForkedRows.pack` lays rows of `lengths` tokens out in, one block:
    each group of rows that begin alike over one copy of their common prefix where `keys` gives each row's
    tokens' keys (`token_keys`), else each row whole, at most `documents` a window. A planner that holds
    its rows' keys counts windows with no layout written."""
    return len(_planned(lengths, width, keys, documents))


def _keyed(rows: Sequence[Sequence[int]], positions: Sequence[Sequence[int]] | None,
           fields: Mapping[str, Sequence[np.ndarray]] | None, share: bool):
    """`rows` as `pack` reads them: each row's ids, its tokens' places (its indices where `positions` is
    None) and the per-token `fields`, and with `share` each row's tokens' keys (`token_keys`), else None:
    a row laid whole is planned by its length alone."""
    laid = [np.asarray(row, np.int64).reshape(-1) for row in rows]
    if not laid or any(row.size == 0 for row in laid):
        raise ValueError("forked rows pack at least one row, each of at least one token")
    places = ([np.arange(row.size, dtype=np.int64) for row in laid] if positions is None
              else [np.asarray(place, np.int64).reshape(-1) for place in positions])
    extra = {name: [np.asarray(value) for value in values] for name, values in (fields or {}).items()}
    for name, given in (("positions", places), *extra.items()):
        if len(given) != len(laid) or any(len(value) != row.size for value, row in zip(given, laid,
                                                                                        strict=True)):
            raise ValueError(f"{name} holds one entry a token of each row")
    keys = ([token_keys(row, place, [values[number] for values in extra.values()])
             for number, (row, place) in enumerate(zip(laid, places, strict=True))] if share else None)
    return laid, places, extra, keys


def _planned(lengths: Sequence[int], width: int, keys: Sequence[Sequence[bytes]] | None,
             documents: int | None) -> list[list[tuple[int, list[int]]]]:
    """Rows of `lengths` planned into windows (`_windows`): grouped by their keys' common prefixes where
    `keys` is given (`_groups`), else each a group of its own, at most `documents` a window."""
    groups = _groups(keys, width) if keys is not None else [[at] for at in range(len(lengths))]
    return _windows(groups, lengths, keys, width, None if keys is not None else documents)


def token_keys(row: np.ndarray, places: np.ndarray, extra: Sequence[np.ndarray]) -> list[bytes]:
    """Each token's identity as a shared prefix compares it: its id, its place and every field it carries."""
    return [b"".join([np.int64(token).tobytes(), np.int64(place).tobytes(),
                      *(np.ascontiguousarray(values[at]).tobytes() for values in extra)])
            for at, (token, place) in enumerate(zip(row.tolist(), places.tolist(), strict=True))]


def _groups(keys: Sequence[Sequence[bytes]], width: int) -> list[list[int]]:
    """The rows cut into groups, each laid over one copy of its common prefix: sorted, then each run of
    sorted rows a group where that lays out fewest tokens in all (a group's tokens its prefix and each row's
    rest, within one window of `width`), a run grown only while its rows share a token."""
    order = sorted(range(len(keys)), key=lambda number: tuple(keys[number]))
    shared = [common_prefix(keys[order[at]], keys[order[at + 1]]) for at in range(len(order) - 1)]
    sizes = [len(keys[number]) for number in order]
    # best[j]: the fewest tokens the first j sorted rows lay out in, and where its last group begins.
    best = [(0, 0)] + [(np.iinfo(np.int64).max, 0)] * len(order)
    for end in range(1, len(order) + 1):
        prefix, total = sizes[end - 1], 0
        for begin in range(end - 1, -1, -1):
            if begin < end - 1:
                prefix = min(prefix, shared[begin])
            total += sizes[begin]
            laid = prefix + total - (end - begin) * prefix
            if laid > width or (begin < end - 1 and prefix == 0):
                break
            if best[begin][0] + laid < best[end][0]:
                best[end] = (best[begin][0] + laid, begin)
    groups, end = [], len(order)
    while end:
        begin = best[end][1]
        groups.append([order[at] for at in range(begin, end)])
        end = begin
    return groups[::-1]


def _prefix(group: Sequence[int], lengths: Sequence[int], keys: Sequence[Sequence[bytes]] | None) -> int:
    """How many leading tokens every row of `group` shares; a row alone shares all of its own."""
    prefix = lengths[group[0]]
    for number in group[1:]:
        assert keys is not None, "rows grouped by their keys"
        prefix = min(prefix, common_prefix(keys[group[0]], keys[number]))
    return prefix


def _documents(window: Sequence[tuple[int, Sequence[int]]], laid: Sequence[np.ndarray]) -> int:
    """The documents a window of groups lays out: each prefix, and each row's rest past it."""
    return sum((prefix > 0) + sum(len(laid[row]) > prefix for row in group) for prefix, group in window)


def _windows(groups: Sequence[list[int]], lengths: Sequence[int], keys: Sequence[Sequence[bytes]] | None,
             width: int, documents: int | None = None) -> list[list[tuple[int, list[int]]]]:
    """`groups` of rows of `lengths` packed first-fit, longest first, into windows of `width`, at most
    `documents` groups a window where given: each window its groups in order, each as its prefix's length
    (`_prefix`, by `keys`) and its rows."""
    planned = [(_prefix(group, lengths, keys), group) for group in groups]
    sizes = np.asarray([prefix + sum(lengths[number] - prefix for number in group)
                        for prefix, group in planned], np.int64)
    order = np.argsort(-sizes, kind="stable")
    grouped, starts = first_fit(sizes[order], width, max(len(order), 1), documents)
    return [[planned[order[at]] for at in grouped[start:stop]]
            for start, stop in itertools.pairwise(starts)]


@dataclasses.dataclass
class _Layout:
    """The windows `ForkedRows.pack` writes, on the host."""

    windows: int
    width: int
    documents: int
    length: int
    rows: int
    extra: Mapping[str, Sequence[np.ndarray]]
    laid: Sequence[np.ndarray]

    def __post_init__(self) -> None:
        shape = (self.windows, self.width)
        self.tokens = np.zeros(shape, np.int32)
        self.segments = np.zeros(shape, np.int32)
        self.positions = np.zeros(shape, np.int32)
        self.parents = np.zeros((self.windows, self.documents), np.int32)
        self.index = np.full((self.rows, self.length), -1, np.int32)
        self.fields = {name: np.zeros((*shape, *values[0].shape[1:]), values[0].dtype)
                       for name, values in self.extra.items()}

    def write(self, window: int, local: int, groups, laid, places, extra) -> None:
        """Lay `groups` (each its prefix's length and its rows) into `window`, the `local`-th of its block,
        each prefix a document and each row's rest one continuing it."""
        slot, number = 0, 0

        def document(row: int, begin: int, end: int, parent: int) -> int:
            nonlocal slot, number
            if number >= self.documents:
                raise ValueError(f"a window holds more than the {self.documents} documents asked for")
            number += 1
            span = slice(slot, slot + end - begin)
            self.tokens[window, span] = laid[row][begin:end]
            self.segments[window, span] = number
            self.positions[window, span] = places[row][begin:end]
            for name, values in extra.items():
                self.fields[name][window, span] = values[row][begin:end]
            self.parents[window, number - 1] = parent
            first = slot
            slot += end - begin
            return first

        for prefix, group in groups:
            root = 0
            if prefix:
                first = document(group[0], 0, prefix, 0)
                root = number
                for row in group:
                    self.index[row, :prefix] = local * self.width + first + np.arange(prefix)
            for row in group:
                size = len(laid[row])
                if size > prefix:
                    first = document(row, prefix, size, root)
                    self.index[row, prefix:size] = local * self.width + first + np.arange(size - prefix)

    def forked(self, blocks: int, xp) -> ForkedRows:
        """The windows as `ForkedRows`, in the array module `xp`."""
        laid = coordinates(segments=self.segments, valid=self.segments > 0, positions=self.positions,
                           max_segments=self.documents, xp=np)
        depth = Forks.depth_of(self.parents)
        # Windows of whole rows alone are plain packed rows, which every model reads.
        if depth:
            laid = dataclasses.replace(laid, forks=Forks(self.parents, depth=depth))
        if xp is not np:
            laid = jax.tree.map(xp.asarray, laid)
        return ForkedRows(xp.asarray(self.tokens), laid, xp.asarray(self.index),
                          {name: xp.asarray(value) for name, value in self.fields.items()}, blocks=blocks)
