"""The host's ledger of a paged KV cache: which rows hold which pages.

A paged cache (`dew.nn.kv_cache`) is one pool of fixed-size pages on the
device; the server decides which pages each row's page table names. `Pages`
keeps that ledger. A page is free, held by one or more rows, or cached: held
by none but still holding the keys of a prompt prefix another request may
share.

Prefix caching follows vLLM's automatic prefix caching: a full page of
prompt tokens is identified by a chained hash of every token up to its end,
so two prompts share a page exactly when they agree on everything before
and inside it. A prompt whose keys depend on more than its tokens (the
context a cross-attention decoder reads) names that as a salt, which seeds
the chain, as vLLM's `cache_salt` seeds its first block's hash. A request
that starts with cached pages takes them into its table and prefills only
the rest. Only full pages of prompt tokens are
shared; the page a prompt ends inside, and every page its draws fill, stay
private to the row, so a shared page is never written again. A cached page
is reclaimed, least recently released first, when the free list runs dry.

A decoder that keeps a row's own state beside its pages (a recurrent mixer's
memory, latent attention's index keys) reads a prefix with the state its
prompt left at the prefix's end as well, which `States` names as vLLM's
hybrid prefix caching does: a row reuses pages only as deep as a state is
held, and starts from that state.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Sequence

import numpy as np


def prefix_hashes(tokens: np.ndarray, page_size: int, pages: int, salt: bytes = b"") -> list[bytes]:
    """The chained hash of each of the first `pages` full pages of `tokens`, the chain seeded by `salt`."""
    hashes: list[bytes] = []
    previous = salt
    for index in range(pages):
        block = np.ascontiguousarray(tokens[index * page_size:(index + 1) * page_size], np.int64)
        previous = hashlib.blake2b(previous + block.tobytes(), digest_size=16).digest()
        hashes.append(previous)
    return hashes


class States:
    """The row states a prefix's pages are read with: `count` entries, each one slot's state after a
    prompt's first full pages, named by the chain digest of the last of them (`prefix_hashes`). The least
    recently used entry gives way to a new state; an entry whose page is reclaimed goes with it."""

    def __init__(self, count: int) -> None:
        self.count = count
        self._free: deque[int] = deque(range(count))
        self._by_hash: OrderedDict[bytes, int] = OrderedDict()

    def __contains__(self, digest: bytes) -> bool:
        return digest in self._by_hash

    def entry(self, digest: bytes) -> int:
        """The entry holding the state after `digest`'s page, now the most recently used."""
        self._by_hash.move_to_end(digest)
        return self._by_hash[digest]

    def take(self, digest: bytes) -> int | None:
        """An entry to write the state after `digest`'s page into: a free one, or the least recently used
        one's; None where a state after that page is held already or the store holds none."""
        if digest in self._by_hash or not self.count:
            return None
        entry = self._free.popleft() if self._free else self._by_hash.popitem(last=False)[1]
        self._by_hash[digest] = entry
        return entry

    def drop(self, digest: bytes) -> None:
        """Forget the state after `digest`'s page, whose page is reclaimed."""
        entry = self._by_hash.pop(digest, None)
        if entry is not None:
            self._free.append(entry)

    def forget(self) -> None:
        """Hold no state: the weights that wrote them are replaced."""
        self._free.extend(self._by_hash.values())
        self._by_hash.clear()


class Pages:
    """Free, held and cached pages of one pool of `count` pages of `size` slots, and with `states` the row
    states a shared prefix is read with (`States`)."""

    def __init__(self, count: int, size: int, *, prefix_cache: bool, states: States | None = None) -> None:
        self.size = size
        self.count = count
        self.prefix_cache = prefix_cache
        self.states = states
        self._free: deque[int] = deque(range(count))
        self._holders = np.zeros(count, np.int32)
        self._cached: OrderedDict[int, None] = OrderedDict()
        """Pages held by no row whose keys a prefix hash still names, oldest release first; each
        one's digest is `_hash_of`'s."""
        self._by_hash: dict[bytes, int] = {}
        self._hash_of: dict[int, bytes] = {}

    @property
    def available(self) -> int:
        """Pages a new request can take: free ones and cached ones no row holds."""
        return len(self._free) + len(self._cached)

    def shared(self, prompt: np.ndarray, salt: bytes = b"") -> list[int]:
        """The cached pages that hold `prompt`'s leading full pages under `salt`, in order, as deep as a
        state after the last of them is held where the pool's rows keep states (`States`).

        The last prompt token is never covered, since its logits score the
        first draw and have to be computed.
        """
        return self._matched(prompt, salt)[0]

    def _matched(self, prompt: np.ndarray, salt: bytes) -> tuple[list[int], bytes | None]:
        """`shared`'s pages and the digest of the last of them, None for none."""
        pages: list[int] = []
        depth, last = 0, None
        if self.prefix_cache:
            for digest in prefix_hashes(prompt, self.size, (len(prompt) - 1) // self.size, salt):
                page = self._by_hash.get(digest)
                if page is None:
                    break
                pages.append(page)
                if self.states is None or digest in self.states:
                    depth, last = len(pages), digest
        return pages[:depth], last

    def reserve(self, prompt: np.ndarray, total: int,
                salt: bytes = b"") -> tuple[list[int], int, int | None] | None:
        """Pages for a row that will hold `total` tokens of which `prompt`, under `salt`, comes first.

        Returns the row's pages, how many leading prompt tokens they
        already hold, and the `States` entry holding the state the row starts
        from after them (None where it starts afresh), or None when the pool
        cannot cover the row; then nothing changes.
        """
        needed = -(-total // self.size)
        shared, last = self._matched(prompt, salt)
        reclaimable = len(self._free) + sum(page not in shared for page in self._cached)
        if needed - len(shared) > reclaimable:
            return None
        for page in shared:
            self._hold(page)
        entry = None if self.states is None or last is None else self.states.entry(last)
        return (shared + [self._take() for _ in range(needed - len(shared))], len(shared) * self.size,
                entry)

    def publish(self, prompt: np.ndarray, pages: Sequence[int], salt: bytes = b"") -> None:
        """Name the row's full prompt pages by their prefix hash under `salt`, once their keys are written."""
        if not self.prefix_cache:
            return
        for digest, page in zip(prefix_hashes(prompt, self.size, len(prompt) // self.size, salt), pages,
                                strict=False):
            if digest not in self._by_hash and page not in self._hash_of:
                self._by_hash[digest] = page
                self._hash_of[page] = digest

    def release(self, pages: Sequence[int]) -> None:
        """A row gives its pages back; a page nobody holds is cached or freed.

        The last page goes first, so the reclaim order, oldest release
        first, takes a prefix chain from its tail and the head that later
        prompts share stays reachable longest.
        """
        for page in reversed(pages):
            self._holders[page] -= 1
            if self._holders[page]:
                continue
            if page in self._hash_of:
                self._cached[page] = None
            else:
                self._free.append(page)

    def forget(self) -> None:
        """Share no page written so far: the keys it holds came from other weights.

        Cached pages return to the free list; held pages stay with their
        rows, unnamed, and are freed when those rows release them.
        """
        self._free.extend(self._cached)
        self._cached.clear()
        self._by_hash.clear()
        self._hash_of.clear()
        if self.states is not None:
            self.states.forget()

    def _hold(self, page: int) -> None:
        self._cached.pop(page, None)
        self._holders[page] += 1

    def _take(self) -> int:
        if self._free:
            page = self._free.popleft()
        else:
            page, _ = self._cached.popitem(last=False)
            digest = self._hash_of.pop(page)
            del self._by_hash[digest]
            if self.states is not None:
                self.states.drop(digest)
        self._holders[page] += 1
        return page
