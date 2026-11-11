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

Prompts seated in one step share a page a prompt seated before them writes in
that step, where it prefills its whole prompt there and no row state goes with
the page (`Pages.reserve`'s `whole`): the step writes every row's keys before
any attention reads them, so a later prompt names the page as it would a cached
one rather than waiting a step for it.
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


class HostTier:
    """`slots` slots in host memory an evicted device entry goes to by its digest (a prefix page's hash, the
    digest of the page a row state is kept after): a free slot, else the least recently kept one's, whose
    entry is forgotten (`kept`). A slot an owed upload reads leaves the tier until the step's moves are
    made (`read`, `drained`). Pages' and row states' host tiers each hold one (`Pages`, `States`)."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self._free: deque[int] = deque(range(slots))
        self._held: OrderedDict[bytes, int] = OrderedDict()
        """Each held entry's slot by its digest, least recently kept first."""
        self._reading: list[int] = []
        """Slots an owed upload still reads, free again once the moves are made."""

    def __contains__(self, digest: bytes) -> bool:
        return digest in self._held

    def slot(self, digest: bytes) -> int:
        """The slot holding `digest`'s entry."""
        return self._held[digest]

    def kept(self, digest: bytes) -> int | None:
        """A slot to hold `digest`'s entry in: a free one, or the least recently kept one's, its entry
        forgotten; None where the tier holds no slot an owed upload does not read."""
        if self._free:
            slot = self._free.popleft()
        elif self._held:
            slot = self._held.popitem(last=False)[1]
        else:
            return None
        self._held[digest] = slot
        return slot

    def read(self, digest: bytes) -> int:
        """Take `digest`'s slot out of the tier for an owed upload, which reads it before it is free."""
        slot = self._held.pop(digest)
        self._reading.append(slot)
        return slot

    def unread(self, digest: bytes, slot: int) -> None:
        """Hand back `slot`, whose owed upload of `digest`'s entry will not run, the entry still in it."""
        self._reading.remove(slot)
        self._held[digest] = slot

    def drop(self, digest: bytes) -> None:
        """Forget `digest`'s entry, its slot free."""
        slot = self._held.pop(digest, None)
        if slot is not None:
            self._free.append(slot)

    def drained(self) -> None:
        """The step's moves are made: the slots its uploads read are free."""
        self._free.extend(self._reading)
        self._reading = []

    def forget(self) -> None:
        """Hold no entry."""
        self._free.extend(self._held.values())
        self._held.clear()


class States:
    """The row states a prefix's pages are read with: `count` entries, each one slot's state after a
    prompt's first full pages, named by the chain digest of the last of them (`prefix_hashes`), and `host`
    slots in host memory. The least recently used entry gives way to a new state, going to a host slot
    first where the store keeps any (its own free one, or the least recently used one's), from which a
    prompt sharing its prefix reads it back into an entry (`entry`); an entry whose page is reclaimed
    goes with it. The moves a step owes go out with its page copies (`Transfers`)."""

    def __init__(self, count: int, host: int = 0) -> None:
        self.count = count
        self._free: deque[int] = deque(range(count))
        self._by_hash: OrderedDict[bytes, int] = OrderedDict()
        self.host = HostTier(host)
        """States held in host slots, by their page's digest."""
        self._pinned: set[int] = set()
        """Entries a seated row restores from, an owed upload writes or an owed capture writes, kept from
        giving way until the step's moves are made (`drained`)."""
        self.spills: list[tuple[int, int]] = []
        """The (entry, host slot) moves owed: an entry giving way, out to its slot."""
        self.uploads: list[tuple[int, int]] = []
        """The (host slot, entry) moves owed after the spills: a host state back into an entry."""

    def __contains__(self, digest: bytes) -> bool:
        """Whether a state after `digest`'s page is held: in an entry, or in a host slot an entry can
        take it back into."""
        return digest in self._by_hash or (digest in self.host and self._vacant())

    def _vacant(self) -> bool:
        return bool(self._free) or any(entry not in self._pinned for entry in self._by_hash.values())

    def entry(self, digest: bytes) -> int:
        """The entry holding the state after `digest`'s page, now the most recently used, kept until the
        step's moves are made: a host state comes back into one (`uploads`)."""
        if digest in self._by_hash:
            self._by_hash.move_to_end(digest)
            entry = self._by_hash[digest]
        else:
            slot = self.host.read(digest)
            entry = self._vacated()
            self.uploads.append((slot, entry))
            self._by_hash[digest] = entry
        self._pinned.add(entry)
        return entry

    def take(self, digest: bytes) -> int | None:
        """An entry to write the state after `digest`'s page into: a free one, or the least recently used
        one's, which goes to the host first; None where a state after that page is held already, the store
        holds none, or every entry is kept for the step."""
        if digest in self._by_hash or digest in self.host or not self.count or not self._vacant():
            return None
        entry = self._vacated()
        self._by_hash[digest] = entry
        # Its capture is owed: no other state takes the entry, nor sends it to the host, before it.
        self._pinned.add(entry)
        return entry

    def _vacated(self) -> int:
        """A free entry, or the least recently used one not kept, its state out to a host slot where one
        can take it (`spills`)."""
        if self._free:
            return self._free.popleft()
        digest = next(held for held, entry in self._by_hash.items() if entry not in self._pinned)
        entry = self._by_hash.pop(digest)
        slot = self.host.kept(digest)
        if slot is not None:
            self.spills.append((entry, slot))
        return entry

    def drop(self, digest: bytes) -> None:
        """Forget the state after `digest`'s page, whose page is reclaimed."""
        entry = self._by_hash.pop(digest, None)
        if entry is not None:
            self._free.append(entry)
        self.host.drop(digest)

    def drained(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """The moves owed before the next step's restores, spills first; the kept entries and the host
        slots the uploads read are free again after them."""
        owed = self.spills, self.uploads
        self.spills, self.uploads = [], []
        self.host.drained()
        self._pinned.clear()
        return owed

    def forget(self) -> None:
        """Hold no state: the weights that wrote them are replaced."""
        self._free.extend(self._by_hash.values())
        self._by_hash.clear()
        self.host.forget()
        self._pinned.clear()


@dataclass
class Transfers:
    """The page copies a step owes before its program runs: `spills`, (device page, host slot) pairs whose
    device page is reclaimed and goes to the host first; then `uploads`, (host slot, device page) pairs a
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
        self._on_host = HostTier(host)
        """Prefix pages held in host slots, by their hash."""
        self.transfers = Transfers()
        """The copies owed before the next step's program (`drained`)."""
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
        self._this_step: dict[bytes, int] = {}
        """Full prompt pages a row seated this step writes in it, by their prefix hash (`reserve`'s
        `whole`): a prompt seated after it in the step names them as it would a cached page's."""
        self._writing: Counter[bytes] = Counter()
        """The prefix hashes of full prompt pages rows hold and have not yet published: a prompt that
        shares one waits for it (`waits`), so it reads those keys where it would write them again."""

    @property
    def available(self) -> int:
        """Pages a new request can take: free ones and cached ones no row holds."""
        return len(self._free) + len(self._cached)

    def shared(self, prompt: np.ndarray, salt: bytes = b"") -> list[int]:
        """The cached pages that hold `prompt`'s leading full pages under `salt`, in order, as deep as a
        state after the last of them is held where the pool's rows keep states (`States`), and the pages a
        row seated this step writes in it (`reserve`'s `whole`).

        The last prompt token is never covered, since its logits score the
        first draw and have to be computed.
        """
        return self._matched(prompt, salt)[0]

    def _located(self, digest: bytes) -> int | None:
        """The page `digest` names: on the device its index, in a host slot the slot's index negated past
        one (`-1 - slot`); None for none."""
        page = self._by_hash.get(digest, self._this_step.get(digest))
        if page is None and digest in self._on_host:
            page = -1 - self._on_host.slot(digest)
        return page

    def _matched(self, prompt: np.ndarray, salt: bytes) -> tuple[list[int], bytes | None, int]:
        """`shared`'s pages, the digest of the last of them (None for none) and the prompt tokens they
        hold."""
        pages: list[int] = []
        depth, last = 0, None
        if not self.prefix_cache:
            return [], None, 0
        for digest in prefix_hashes(prompt, self.size, (len(prompt) - 1) // self.size, salt):
            page = self._located(digest)
            if page is None:
                break
            pages.append(page)
            if self.states is None or digest in self.states:
                depth, last = len(pages), digest
        return pages[:depth], last, depth * self.size

    def waits(self, prompt: np.ndarray, salt: bytes = b"") -> bool:
        """Whether `prompt`, under `salt`, shares its next full page past those it can name with a row that
        is writing it in a later step, so it reads that page once the row publishes it (`publish`) instead
        of writing its keys a second time: as vLLM's prefix caching shares a running prompt's blocks, the
        samples of one prompt prefill it once. A page a row seated this step writes in it is named now
        (`reserve`'s `whole`)."""
        if not self.prefix_cache:
            return False
        pages = (len(prompt) - 1) // self.size
        shared = self._matched(prompt, salt)[2] // self.size
        return shared < pages and prefix_hashes(prompt, self.size, shared + 1, salt)[-1] in self._writing

    def reserve(self, prompt: np.ndarray, total: int, salt: bytes = b"", *,
                whole: bool = False) -> tuple[list[int], int, int | None, tuple[bytes, ...]] | None:
        """Pages for a row that will hold `total` tokens of which `prompt`, under `salt`, comes first.

        Returns the row's pages, how many leading prompt tokens they
        already hold, the `States` entry holding the state the row starts
        from after them (None where it starts afresh), and the hashes of the
        full prompt pages it writes that no page names yet, which other
        prompts wait for (`waits`) until it publishes them or gives its pages
        back (`written`); or None when the pool cannot cover the row; then
        nothing changes. With `whole` (the row prefills the rest of its
        prompt in the step it is seated in, and the pool keeps no row
        states), a prompt seated after it in this step names those pages now.
        """
        needed = -(-total // self.size)
        shared, last, held = self._matched(prompt, salt)
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
        for _, digest in landing.values():
            self._on_host.read(digest)
        pages = [page if index not in landing else self._uploaded(*landing[index])
                 for index, page in enumerate(shared)]
        fresh = [self._take() for _ in range(needed - len(pages))]
        writing = tuple(digest for digest in hashes[len(shared):]
                        if digest not in self._by_hash and digest not in self._this_step)
        self._writing.update(writing)
        if whole and self.states is None:
            # The step writes these pages' keys before any attention reads them.
            written = zip(hashes[len(shared):], fresh, strict=False)
            self._this_step.update((digest, page) for digest, page in written if digest in writing)
        return pages + fresh, held, entry, writing

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
        self._on_host.drained()
        self._landing.clear()
        # The step's seating is over: a page a row writes now is named once it publishes it.
        self._this_step.clear()
        return owed

    def written(self, writing: Sequence[bytes]) -> None:
        """A row's pages of `writing` (`reserve`'s) are written no more: it published them (`publish`) or
        gave its pages back first."""
        self._writing.subtract(writing)
        self._writing = +self._writing
        for digest in writing:
            # Published, or never to be written: a later prompt names it once published alone.
            self._this_step.pop(digest, None)

    def publish(self, prompt: np.ndarray, pages: Sequence[int], salt: bytes = b"") -> None:
        """Name the row's full prompt pages by their prefix hash under `salt`, once their keys are
        written."""
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
        self._this_step.clear()
        self._on_host.forget()
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
                self._on_host.unread(digest, slot)
            elif not self._spilled(page, digest) and self.states is not None:
                self.states.drop(digest)
        self._holders[page] += 1
        return page

    def _spilled(self, page: int, digest: bytes) -> bool:
        """Keep reclaimed `page`'s keys in a host slot under `digest`, a free slot or the least recently used
        one's, whose page is forgotten; whether a slot took it (none where every slot is read by an owed
        upload, or the pool keeps no host tier). A state kept after the page (`States`) stays with it."""
        slot = self._on_host.kept(digest)
        if slot is None:
            return False
        self.transfers.spills.append((page, slot))
        return True
