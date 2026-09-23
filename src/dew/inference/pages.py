"""The host's ledger of a paged KV cache: which rows hold which pages.

A paged cache (`dew.nn.kv_cache`) is one pool of fixed-size pages on the
device; the server decides which pages each row's page table names. `Pages`
keeps that ledger. A page is free, held by one or more rows, or cached: held
by none but still holding the keys of a prompt prefix another request may
share.

Prefix caching follows vLLM's automatic prefix caching: a full page of
prompt tokens is identified by a chained hash of every token up to its end,
so two prompts share a page exactly when they agree on everything before
and inside it. A request that starts with cached pages takes them into its
table and prefills only the rest. Only full pages of prompt tokens are
shared; the page a prompt ends inside, and every page its draws fill, stay
private to the row, so a shared page is never written again. A cached page
is reclaimed, least recently released first, when the free list runs dry.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Sequence

import numpy as np


def prefix_hashes(tokens: np.ndarray, page_size: int, pages: int) -> list[bytes]:
    """The chained hash of each of the first `pages` full pages of `tokens`."""
    hashes: list[bytes] = []
    previous = b""
    for index in range(pages):
        block = np.ascontiguousarray(tokens[index * page_size:(index + 1) * page_size], np.int64)
        previous = hashlib.blake2b(previous + block.tobytes(), digest_size=16).digest()
        hashes.append(previous)
    return hashes


class Pages:
    """Free, held and cached pages of one pool of `count` pages of `size` slots."""

    def __init__(self, count: int, size: int, *, prefix_cache: bool) -> None:
        self.size = size
        self.count = count
        self.prefix_cache = prefix_cache
        self._free: deque[int] = deque(range(count))
        self._holders = np.zeros(count, np.int32)
        self._cached: OrderedDict[int, bytes] = OrderedDict()
        """Pages held by no row whose keys a prefix hash still names, oldest release first."""
        self._by_hash: dict[bytes, int] = {}
        self._hash_of: dict[int, bytes] = {}

    @property
    def available(self) -> int:
        """Pages a new request can take: free ones and cached ones no row holds."""
        return len(self._free) + len(self._cached)

    def reserve(self, prompt: np.ndarray, total: int) -> tuple[list[int], int] | None:
        """Pages for a row that will hold `total` tokens of which `prompt` comes first.

        Returns the row's pages and how many leading prompt tokens they
        already hold, or None when the pool cannot cover the row; then
        nothing changes. The last prompt token is always prefilled, since its
        logits score the first draw.
        """
        needed = -(-total // self.size)
        shared: list[int] = []
        if self.prefix_cache:
            for digest in prefix_hashes(prompt, self.size, (len(prompt) - 1) // self.size):
                page = self._by_hash.get(digest)
                if page is None:
                    break
                shared.append(page)
        reclaimable = len(self._free) + sum(page not in shared for page in self._cached)
        if needed - len(shared) > reclaimable:
            return None
        for page in shared:
            self._hold(page)
        return shared + [self._take() for _ in range(needed - len(shared))], len(shared) * self.size

    def publish(self, prompt: np.ndarray, pages: Sequence[int]) -> None:
        """Name the row's full prompt pages by their prefix hash, once their keys are written."""
        if not self.prefix_cache:
            return
        for digest, page in zip(prefix_hashes(prompt, self.size, len(prompt) // self.size), pages,
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
                self._cached[page] = self._hash_of[page]
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

    def _hold(self, page: int) -> None:
        self._cached.pop(page, None)
        self._holders[page] += 1

    def _take(self) -> int:
        if self._free:
            page = self._free.popleft()
        else:
            page, digest = self._cached.popitem(last=False)
            del self._by_hash[digest], self._hash_of[page]
        self._holders[page] += 1
        return page
