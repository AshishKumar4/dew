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

A server may keep the pages it reclaims in host memory (`HostTier`): the
reclaimed pages' keys are copied out before their slots are written again,
and a prompt whose chain continues past the device's cached pages into the
host's loads those back instead of prefilling them.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter, OrderedDict, deque
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


@dataclasses.dataclass(frozen=True)
class HostBatch:
    """The pages one admission reclaimed, on the host: `payload` holds every pooled leaf's rows, one
    a page, in the order of `digests`, each `(group, digest)`; `nbytes` is what it holds, every row
    counted; `epoch` the weights it was written under."""

    payload: list
    digests: tuple[tuple[int, bytes], ...]
    nbytes: int
    epoch: int


@dataclasses.dataclass(frozen=True)
class HostPage:
    """A host page a row loads: row `row` of host batch `batch`, into the row's page `page`."""

    batch: int
    row: int
    page: int


class HostTier:
    """The pages a server's prefix cache reclaimed, kept in host memory within `budget` bytes.

    A batch is one admission's reclaimed pages, immutable once added, so a row reading it never races a
    writer. Batches leave least recently used first, a batch a row has leased (a load in flight) never;
    a batch that does not fit beside the leased ones is dropped, and so is every batch of older weights
    (`forget`). `dropped` counts the pages dropped, by reason."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.epoch = 0
        self.held = 0
        """Bytes the batches hold, leased ones included."""
        self.dropped: Counter[str] = Counter()
        self._batches: OrderedDict[int, HostBatch] = OrderedDict()
        self._index: dict[tuple[int, bytes], tuple[int, int]] = {}
        self._leases: Counter[int] = Counter()
        self._next = 0

    def find(self, group: int, digest: bytes) -> tuple[int, int] | None:
        """The batch and row holding `group`'s page of prefix hash `digest`, if the host holds it."""
        return self._index.get((group, digest))

    def add(self, payload: list, digests: Sequence[tuple[int, bytes]], nbytes: int) -> None:
        """Hold one admission's reclaimed pages, making room by dropping the least recently used
        unleased batches; pages that still do not fit are dropped."""
        while self.held + nbytes > self.budget and self._evictable():
            self._drop(self._evictable()[0], "budget")
        if self.held + nbytes > self.budget:
            self.dropped["budget"] += len(digests)
            return
        number, self._next = self._next, self._next + 1
        self._batches[number] = HostBatch(payload, tuple(digests), nbytes, self.epoch)
        self.held += nbytes
        for row, key in enumerate(digests):
            self._index[key] = (number, row)

    def batch(self, number: int) -> HostBatch:
        """Host batch `number`, now the most recently used."""
        self._batches.move_to_end(number)
        return self._batches[number]

    def lease(self, number: int) -> None:
        self._leases[number] += 1

    def release(self, number: int) -> None:
        """A load of batch `number` has been dispatched; a batch of older weights goes with its last
        lease."""
        self._leases[number] -= 1
        if not self._leases[number]:
            del self._leases[number]
            if self._batches[number].epoch != self.epoch:
                self._drop(number, "reload")

    def forget(self) -> None:
        """Share nothing the old weights wrote: every unleased batch goes now, and a leased one with its
        last lease."""
        self.epoch += 1
        for number in self._evictable():
            self._drop(number, "reload")
        self._index = {key: place for key, place in self._index.items()
                       if self._batches[place[0]].epoch == self.epoch}

    def _evictable(self) -> list[int]:
        return [number for number in self._batches if number not in self._leases]

    def _drop(self, number: int, reason: str) -> None:
        batch = self._batches.pop(number)
        self.held -= batch.nbytes
        for row, key in enumerate(batch.digests):
            if self._index.get(key) == (number, row):
                del self._index[key]
        self.dropped[reason] += len(batch.digests)


class Pages:
    """Free, held and cached pages of one pool of `count` pages of `size` slots: group `group` of a
    server's pool, whose reclaimed cached pages `host` keeps where it is given."""

    def __init__(self, count: int, size: int, *, prefix_cache: bool, group: int = 0,
                 host: HostTier | None = None) -> None:
        self.size = size
        self.count = count
        self.prefix_cache = prefix_cache
        self.group = group
        self.host = host if prefix_cache else None
        self._reclaimed: list[tuple[int, bytes]] = []
        """Cached pages taken for new keys since `reclaimed` was last read, with their digests: the
        pages the host tier copies out before the next step writes them."""
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

    def shared(self, prompt: np.ndarray) -> list[int]:
        """The cached pages that hold `prompt`'s leading full pages, in order.

        The last prompt token is never covered, since its logits score the
        first draw and have to be computed.
        """
        pages: list[int] = []
        if self.prefix_cache:
            for digest in prefix_hashes(prompt, self.size, (len(prompt) - 1) // self.size):
                page = self._by_hash.get(digest)
                if page is None:
                    break
                pages.append(page)
        return pages

    def hosted(self, prompt: np.ndarray, after: int) -> list[tuple[int, int]]:
        """The host's batch and row of each of `prompt`'s leading full pages past its first `after`,
        as long as the host holds them in order; the last prompt token is never covered."""
        if self.host is None:
            return []
        found: list[tuple[int, int]] = []
        for digest in prefix_hashes(prompt, self.size, (len(prompt) - 1) // self.size)[after:]:
            place = self.host.find(self.group, digest)
            if place is None:
                break
            found.append(place)
        return found

    def chain(self, prompt: np.ndarray) -> int:
        """How many of `prompt`'s leading full pages the device's cache and then the host's hold."""
        shared = self.shared(prompt)
        return len(shared) + len(self.hosted(prompt, len(shared)))

    def reserve(self, prompt: np.ndarray, total: int) -> tuple[list[int], int, list[HostPage]] | None:
        """Pages for a row that will hold `total` tokens of which `prompt` comes first.

        Returns the row's pages, how many leading prompt tokens they hold
        once its host pages are loaded, and those loads, each a host batch
        leased until its load is dispatched; or None when the pool cannot
        cover the row, and then nothing changes. A host page takes a fresh
        device page like the rest.
        """
        needed = -(-total // self.size)
        shared = self.shared(prompt)
        reclaimable = len(self._free) + sum(page not in shared for page in self._cached)
        if needed - len(shared) > reclaimable:
            return None
        hosted = self.hosted(prompt, len(shared))
        for page in shared:
            self._hold(page)
        for number, _ in hosted:
            # Leased before any page is taken, so no batch this reservation reads leaves while it runs.
            assert self.host is not None
            self.host.lease(number)
        pages = shared + [self._take() for _ in range(needed - len(shared))]
        loads = [HostPage(number, row, pages[len(shared) + index])
                 for index, (number, row) in enumerate(hosted)]
        return pages, (len(shared) + len(hosted)) * self.size, loads

    def reclaimed(self) -> list[tuple[int, bytes]]:
        """The cached pages taken since the last read, with their digests, for the host to copy out
        before the next step writes them."""
        taken, self._reclaimed = self._reclaimed, []
        return taken

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
        self._reclaimed.clear()

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
            if self.host is not None and self.host.find(self.group, digest) is None:
                self._reclaimed.append((page, digest))
        self._holders[page] += 1
        return page
