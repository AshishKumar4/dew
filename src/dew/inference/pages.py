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

A host tier keeps prefix pages past the device's: a cached page the pool reclaims
is copied to one of `host` page slots in host memory first, still named by its
hash, and a later prompt that shares it takes a fresh device page that the host
copy is read back into before its prefill, as vLLM's and SGLang's hierarchical
prefix caches offload and reload blocks. The ledger names the copies each step
owes (`Transfers`); the server makes them before the step's program writes any
page.

A decoder that keeps a row's own state beside its pages (a recurrent mixer's
memory, latent attention's index keys) reads a prefix with the state its
prompt left at the prefix's end as well, which `States` names as vLLM's
hybrid prefix caching does: a row reuses pages only as deep as a state is
held, and starts from that state.
"""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

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


@dataclass
class Transfers:
    """The page copies a step owes before its program runs: `spills`, (device page, host slot) pairs whose
    device page is reclaimed and goes to the host first, then `uploads`, (host slot, device page) pairs a
    seated prompt reads back from the host into a fresh device page."""

    spills: list[tuple[int, int]] = field(default_factory=list)
    uploads: list[tuple[int, int]] = field(default_factory=list)


class Pages:
    """Free, held and cached pages of one pool of `count` pages of `size` slots, with `host` page slots in
    host memory that a reclaimed prefix page goes to (`Transfers`), and with `states` the row states a
    shared prefix is read with (`States`), a prefix on the host read as deep as one is kept."""

    def __init__(self, count: int, size: int, *, prefix_cache: bool, states: States | None = None,
                 host: int = 0) -> None:
        self.size = size
        self.count = count
        self.prefix_cache = prefix_cache
        self.states = states
        self.host = host
        self._host_free: deque[int] = deque(range(host))
        self._on_host: OrderedDict[bytes, int] = OrderedDict()
        """Prefix pages held in host slots, by their hash, least recently used first."""
        self.transfers = Transfers()
        """The copies owed before the next step's program (`drained`)."""
        self._reading: list[int] = []
        """Host slots an owed upload still reads, free again once the copies are made."""
        self._landing: dict[int, tuple[int, bytes]] = {}
        """The device pages an owed upload writes, each with the host slot it reads and its page's hash:
        reclaimed before the copy, such a page's keys are still the slot's, which keeps them."""
        self._free: deque[int] = deque(range(count))
        self._holders = np.zeros(count, np.int32)
        self._cached: OrderedDict[int, None] = OrderedDict()
        """Pages held by no row whose keys a prefix hash still names, oldest release first; each
        one's digest is `_hash_of`'s."""
        self._by_hash: dict[bytes, int] = {}
        self._hash_of: dict[int, bytes] = {}
        self._writing: Counter[bytes] = Counter()
        """The prefix hashes of full prompt pages rows hold and have not yet published: a prompt that
        shares one waits for it (`waits`), so it reads those keys where it would write them again."""

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
        """`shared`'s pages and the digest of the last of them, None for none: a page on the device as
        its index, one in a host slot as the slot's index, negated past one (`-1 - slot`)."""
        pages: list[int] = []
        depth, last = 0, None
        if self.prefix_cache:
            for digest in prefix_hashes(prompt, self.size, (len(prompt) - 1) // self.size, salt):
                page = self._by_hash.get(digest)
                if page is None and digest in self._on_host:
                    page = -1 - self._on_host[digest]
                if page is None:
                    break
                pages.append(page)
                if self.states is None or digest in self.states:
                    depth, last = len(pages), digest
        return pages[:depth], last

    def waits(self, prompt: np.ndarray, salt: bytes = b"") -> bool:
        """Whether `prompt`, under `salt`, shares its next full page past those cached with a row that is
        writing it now, so it reads that page once the row publishes it (`publish`) instead of writing
        its keys a second time: as vLLM's prefix caching shares a running prompt's blocks, the samples of
        one prompt prefill once."""
        if not self.prefix_cache:
            return False
        pages = (len(prompt) - 1) // self.size
        shared = len(self._matched(prompt, salt)[0])
        return shared < pages and prefix_hashes(prompt, self.size, shared + 1, salt)[-1] in self._writing

    def reserve(self, prompt: np.ndarray, total: int,
                salt: bytes = b"") -> tuple[list[int], int, int | None, tuple[bytes, ...]] | None:
        """Pages for a row that will hold `total` tokens of which `prompt`, under `salt`, comes first.

        Returns the row's pages, how many leading prompt tokens they
        already hold, the `States` entry holding the state the row starts
        from after them (None where it starts afresh), and the hashes of the
        full prompt pages it writes that no page names yet, which other
        prompts wait for (`waits`) until it publishes them or gives its pages
        back (`written`); or None when the pool cannot cover the row; then
        nothing changes.
        """
        needed = -(-total // self.size)
        shared, last = self._matched(prompt, salt)
        resident = [page for page in shared if page >= 0]
        reclaimable = len(self._free) + sum(page not in resident for page in self._cached)
        if needed - len(resident) > reclaimable:
            return None
        for page in resident:
            self._hold(page)
        entry = None if self.states is None or last is None else self.states.entry(last)
        hashes = prefix_hashes(prompt, self.size, len(prompt) // self.size, salt) if self.prefix_cache else []
        # A page on the host comes back into a fresh device page, named by its hash there again; every
        # host copy the row reads leaves the host's ledger first, so no spill the fresh pages cause evicts it.
        landing = {index: (-1 - page, hashes[index]) for index, page in enumerate(shared) if page < 0}
        for slot, digest in landing.values():
            del self._on_host[digest]
            self._reading.append(slot)
        pages = [page if index not in landing else self._uploaded(*landing[index])
                 for index, page in enumerate(shared)]
        writing = tuple(digest for digest in hashes[len(shared):] if digest not in self._by_hash)
        self._writing.update(writing)
        return (pages + [self._take() for _ in range(needed - len(shared))], len(shared) * self.size,
                entry, writing)

    def _uploaded(self, slot: int, digest: bytes) -> int:
        """A fresh device page that host `slot`'s copy of `digest`'s page is read back into, held."""
        page = self._take()
        self.transfers.uploads.append((slot, page))
        self._landing[page] = (slot, digest)
        self._by_hash[digest] = page
        self._hash_of[page] = digest
        return page

    def drained(self) -> Transfers:
        """The copies owed before the next step's program, which its caller makes, spills first; the host
        slots the uploads read are free again after them."""
        owed, self.transfers = self.transfers, Transfers()
        self._host_free.extend(self._reading)
        self._reading = []
        self._landing.clear()
        return owed

    def written(self, writing: Sequence[bytes]) -> None:
        """A row's pages of `writing` (`reserve`'s) are written no more: it published them (`publish`) or
        gave its pages back first."""
        self._writing.subtract(writing)
        self._writing = +self._writing

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
        self._host_free.extend(self._on_host.values())
        self._on_host.clear()
        self._landing.clear()
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
            landing = self._landing.pop(page, None)
            if landing is not None:
                # Its upload has not run: the host slot keeps the page as it was.
                slot, _ = landing
                self.transfers.uploads.remove((slot, page))
                self._reading.remove(slot)
                self._on_host[digest] = slot
            elif not self._spilled(page, digest) and self.states is not None:
                self.states.drop(digest)
        self._holders[page] += 1
        return page

    def _spilled(self, page: int, digest: bytes) -> bool:
        """Keep reclaimed `page`'s keys in a host slot under `digest`, a free slot or the least recently used
        one's, whose page is forgotten; whether a slot took it (none where every slot is read by an owed
        upload, or the pool keeps no host tier). A state kept after the page (`States`) stays with it."""
        if self._host_free:
            slot = self._host_free.popleft()
        elif self._on_host:
            slot = self._on_host.popitem(last=False)[1]
        else:
            return False
        self._on_host[digest] = slot
        self.transfers.spills.append((page, slot))
        return True
