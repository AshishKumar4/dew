"""A shared layer under JAX's persistent compilation cache, for CI on armada and the GPU queue.

Each armada task and GPU pass starts on a cold machine, so the on-disk cache conftest turns on
(`dew.cache.enable_compilation_cache`) is always empty there, and the suite spends most of its time
in XLA: on the Colab A100 tier, compiling took 55% of every file's wall time. With
DEW_XLA_CACHE_URL and DEW_XLA_CACHE_TOKEN set (.armada.json), `install` keeps each test file's
programs in that file's bundle on the dew-xla-cache Worker (deployed from the private dew-ops
repository), through JAX's own cache directory: one GET as the file is collected writes its last
run's programs into the directory, and one PUT after it ran, where it compiled any program the
bundle lacked, sends the programs compiled under it and those fetched for it. Bundles are kept
under jaxlib's version, the backend, the Python minor and the device's kind, beside
JAX's own key of each program and its flags.

JAX keeps a program as one file, `<key>-cache`, in `jax_compilation_cache_dir`, and with no
`jax_compilation_cache_max_size` (Dew sets none) it neither evicts nor stamps them
(jax/_src/lru_cache.py in jax 0.11.2), so the files a directory gains while a file's tests run are
the programs they compiled. The programs they only read are not seen, so a program another
file's bundle brought in joins this file's bundle only where it is compiled under it. Each pytest
worker that keeps a bundle has a directory of its own (conftest), so the files a worker's bundle
gains are its own whole writes, never another worker's half-written one.

A bundle is a file's in one execution partition (conftest's `bundle`): a split file's group, or an
xdist worker, has its own, so partitions that compile apart never replace each other's programs. A
group with no bundle yet, its file's cut changed, seeds the programs of up to `SIBLINGS` groups of its
file's other cuts whose share of the file overlaps its own (`_siblings`), and writes its own bundle
under its own cut (2026-10-28: 565 of 1,182 recut groups restored nothing after CI's target moved). A
bundle and not a program at a time, as a test that runs a model eagerly compiles thousands of
programs of a few milliseconds each (14,068 of the A100 tier's 16,537, under 50 ms), quicker to
compile than to ask a Worker for: asking about each one took a DeepSeek drafting test from 103 s to
962 s. A file's bundle is its programs' few megabytes (the CPU lane's same-bits test, 6546
programs, 46 MB).

The remote can only make a task slower: a fetch or an upload that fails, or a local read or write
of the directory that fails, counts against it, and after `FAILURES` in a row it is left alone for
the rest of the process, so an outage costs a few timeouts. Being left alone is final and is
checked as each request is made: an answer that arrives after it, a request already under way,
does not take the remote back, and the uploads still queued are dropped unsent. Uploads run on a
background thread, and the process waits up to `DRAIN_SECONDS` for them at exit.
"""

import importlib.util
import itertools
import logging
import os
import queue
import re
import struct
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path

FAILURES = 3
TIMEOUT_SECONDS = 30.0
DRAIN_SECONDS = 120.0
_R2 = importlib.util.spec_from_file_location(
    "armada_r2", Path(__file__).resolve().parents[1] / "tools" / "armada" / "r2.py")
r2 = importlib.util.module_from_spec(_R2)
_R2.loader.exec_module(r2)  # the Worker's one client and its contract (tools/armada/r2.py)
_HOST = importlib.util.spec_from_file_location(
    "armada_host", Path(__file__).resolve().parents[1] / "tools" / "armada" / "host.py")
HOST = importlib.util.module_from_spec(_HOST)
_HOST.loader.exec_module(HOST)  # the host's CPU class, which keys a bundle (tools/armada/host.py)
MAX_BYTES = r2.MAX_BYTES
"""The largest bundle sent or read: the Worker's bound (tools/armada/r2.json)."""
SIBLINGS = 3
"""The most other-cut groups a recut group seeds from, nearest overlap first, within one `MAX_BYTES`."""
SIBLING_KEYS = 64
"""The most bundle keys a recut group lists for its file's other cuts."""
GROUP = re.compile(r"^(?P<path>.+?)/(?P<index>\d+)-of-(?P<count>\d+)(?P<rest>(?:/.*)?)$")
"""A split group's bundle: its file, its group G of N (`bundle`) and any worker after it."""
SUFFIX = "-cache"
"""What JAX's cache names a program's file after its key (jax/_src/lru_cache.py)."""
UNREACHABLE = (OSError, r2.R2Error)
"""A request the Worker did not answer as asked (`r2.R2Error`), or a local read or write of the cache
directory that fails (an OSError)."""

logger = logging.getLogger(__name__)


class RemoteCache:
    """JAX's cache `directory`, each test file's programs also read from and written to `url`."""

    def __init__(self, directory: Path, url: str, token: str, prefix: str):
        self.directory, self.prefix = Path(directory), prefix.strip("/")
        self.store = r2.Store(url, token, "dew-ci-xla-cache", TIMEOUT_SECONDS)
        self.failures = 0
        self.given_up = False
        self.lock = threading.Lock()
        """Guards the failure count and the counters."""
        self.state = threading.Lock()
        """Guards the bundles' bookkeeping and the directory's view: a bundle's switch, its seeding, its
        sealing; held across no request."""
        self.file: str | None = None
        self.seen: set[str] | None = None
        """The programs in the directory when the current bundle's run began or last moved on."""
        self.written: dict[str, list[str]] = {}
        """The programs compiled while each bundle's tests ran, in the order they appeared."""
        self.fetched: dict[str, list[str]] = {}
        """The programs each bundle's GET brought, in the bundle's order."""
        self.held: dict[str, set[str]] = {}
        """The programs each bundle holds on the Worker, as read or last written."""
        self.queued: dict[str, set[str]] = {}
        """The compiled programs each bundle's last upload was made for: a bundle the cap cut short is not
        sent again until its tests compile another."""
        self.unrestored: set[str] = set()
        """The bundles whose fetched programs did not all reach the directory: never written back, so a
        partial restore does not replace a whole bundle."""
        # Daemon threads, which the interpreter does not wait for at exit; `drain` bounds the wait.
        self.uploads: queue.Queue[tuple[str, bytes]] = queue.Queue()
        self.uploader = threading.Thread(target=self._uploading, daemon=True, name="xla-cache")
        self.uploader.start()
        self.programs_fetched = self.uploaded = 0
        """The programs the remote's bundles put in the directory, and the bundles it took."""
        self.bytes_fetched = self.bytes_uploaded = 0
        """The bundles' bytes read and sent, which the summary reports."""

    @property
    def available(self) -> bool:
        return not self.given_up

    def _failed(self, error: Exception) -> None:
        with self.lock:
            self.failures += 1
            if self.failures >= FAILURES and not self.given_up:
                self.given_up = True
                logger.warning("the shared XLA cache is unreachable (%s); compiling without it", error)

    def _answered(self) -> None:
        """A request answered: the failures in a row end, unless the remote was given up meanwhile."""
        with self.lock:
            if not self.given_up:
                self.failures = 0

    def _key(self, file: str) -> str:
        return f"{self.prefix}/bundles/{file}"

    def _programs(self) -> set[str]:
        return {path.name.removesuffix(SUFFIX) for path in self.directory.glob(f"*{SUFFIX}")}

    def _moved_on(self) -> None:
        """The programs the directory gained since the current bundle's run began or last moved on,
        written to its account (under `state`)."""
        if self.file is None or self.seen is None:
            return
        programs = self._programs()
        written = self.written.setdefault(self.file, [])
        written += sorted(programs - self.seen - set(written))
        self.seen = programs

    def opened(self, file: str) -> None:
        """Bundle `file` in the local cache, and the programs compiled from here on its."""
        if not self.available:
            return
        try:
            with self.state:
                if file != self.file:
                    self._moved_on()
                    self.file = file
                    self.seen = self._programs()
                if file in self.held:
                    return
                self.held[file] = set()
            entries = self._fetched(file)
            borrowed = [] if entries else self._siblings(file)
        except UNREACHABLE as error:
            self._failed(error)
            return
        except ValueError as error:  # an answer that is no whole bundle is a miss
            self._failed(error)
            return
        keys, new = [], 0
        with self.state:
            try:
                for key, value in entries:
                    keys.append(key)
                    # One the directory had already (another file's tests compiled it meanwhile) stays
                    # among the programs the current file's run gained.
                    if self._seeded(key, value):
                        new += 1
                        self.seen.add(key)
                # Another cut's programs: fetched for this group, which holds none of them yet, so
                # `closed` writes them into its own bundle under its own cut, once.
                for key, value in borrowed:
                    keys.append(key)
                    if self._seeded(key, value):
                        new += 1
                        self.seen.add(key)
            except UNREACHABLE as error:
                self.unrestored.add(file)
                self._failed(error)
            self.fetched[file] = keys
            self.held[file] = {key for key, _ in entries}
        if keys and file not in self.unrestored:
            self._answered()
        with self.lock:
            self.programs_fetched += new

    def _fetched(self, file: str) -> list[tuple[str, bytes]]:
        """The entries of `file`'s bundle on the Worker; none where it has none or the remote was given
        up. An answer counts as one only once its programs are in the directory (`opened`)."""
        if self.given_up:  # checked as each request is made
            return []
        bundle = self.store.get(self._key(file), limit=MAX_BYTES)
        if bundle is None:
            return []
        with self.lock:
            self.bytes_fetched += len(bundle)
        return unpacked(bundle)

    def _siblings(self, file: str) -> list[tuple[str, bytes]]:
        """For split group `file` with no bundle of its own, the entries of up to SIBLINGS groups of its
        file's other cuts whose share of the file overlaps its own, the nearest overlap first, together
        within MAX_BYTES; none for a file that is no split group."""
        group = GROUP.match(file)
        if group is None or self.given_up:
            return []
        index, count = int(group["index"]), int(group["count"])
        start, end = (index - 1) / count, index / count
        overlaps = []
        for name in itertools.islice(self.store.keys(f"{self._key(group['path'])}/"), SIBLING_KEYS):
            other = GROUP.match(f"{group['path']}/{name}")
            if other is None or other["rest"] != group["rest"] or int(other["count"]) == count:
                continue
            at, of = int(other["index"]), int(other["count"])
            low, high = (at - 1) / of, at / of
            if low < end and high > start:
                overlaps.append((min(end, high) - max(start, low), name))
        entries, left = [], MAX_BYTES
        for _, name in sorted(overlaps, reverse=True)[:SIBLINGS]:
            if self.given_up or left <= 0:
                break
            try:
                bundle = self.store.get(self._key(f"{group['path']}/{name}"), limit=left)
            except ValueError:  # past what is left, or no whole bundle: the next may fit
                continue
            if bundle is not None:
                left -= len(bundle)
                with self.lock:
                    self.bytes_fetched += len(bundle)
                entries += unpacked(bundle)
        return entries

    def _seeded(self, key: str, value: bytes) -> bool:
        """`value` as program `key`'s file, written to a name of its own and linked in whole, so a
        reader sees all of it or none; False, writing nothing, where the directory has it already."""
        handle, staged = tempfile.mkstemp(prefix=".", suffix=".partial", dir=self.directory)
        try:
            with os.fdopen(handle, "wb") as written:
                written.write(value)
            os.link(staged, self.directory / f"{key}{SUFFIX}")
        except FileExistsError:
            return False
        finally:
            os.unlink(staged)
        return True

    def closed(self, file: str) -> None:
        """Bundle `file` written back, where its tests compiled a program it lacked: the programs
        compiled under it, then those fetched for it, each that fits within `MAX_BYTES`; nothing sent
        where none of those it lacked fits."""
        if not self.available or file in self.unrestored:
            return
        try:
            with self.state:
                if file == self.file:
                    self._moved_on()
                written = list(self.written.get(file, []))
                fetched = list(self.fetched.get(file, []))
                # Its own bundle's fetched programs are held already; a recut group's, borrowed from
                # its file's other cuts (`_siblings`), are not, and are written under its own cut once.
                if not {*written, *fetched} - self.held.get(file, set()) - self.queued.get(file, set()):
                    return
            entries, size = [], 0
            for key in dict.fromkeys([*written, *fetched]):
                path = self.directory / f"{key}{SUFFIX}"
                if not path.exists():
                    continue
                cost = HEADER.size + len(key.encode()) + path.stat().st_size
                if size + cost > MAX_BYTES:  # a program past what is left goes without; the rest may fit
                    continue
                value = path.read_bytes()
                size += HEADER.size + len(key.encode()) + len(value)
                entries.append((key, value))
        except UNREACHABLE as error:
            self._failed(error)
            return
        with self.state:
            self.queued[file] = {*written, *fetched}
            if not {key for key, _ in entries} - self.held.get(file, set()):
                return  # nothing new fits: the Worker's bundle stays as it is
            self.held[file] = {key for key, _ in entries}
        self.uploads.put((file, packed(entries)))

    def _uploading(self) -> None:
        while True:
            file, bundle = self.uploads.get()
            try:
                if not self.given_up:  # checked as each request is made: a queued bundle goes unsent
                    self.store.put(self._key(file), bundle)
                    self._answered()
                    with self.lock:
                        self.uploaded += 1
                        self.bytes_uploaded += len(bundle)
            except (*UNREACHABLE, ValueError) as error:  # a bundle that does not reach it is compiled again
                self._failed(error)
            finally:
                self.uploads.task_done()

    def summary(self) -> str:
        """What the remote did for this process, which a task's log ends with: a slow task whose
        compilations missed it says so."""
        return (f"shared XLA cache: {self.programs_fetched} programs fetched "
                f"({self.bytes_fetched / 1e6:.1f} MB read), {self.uploaded} bundles uploaded "
                f"({self.bytes_uploaded / 1e6:.1f} MB)"
                + ("" if self.available else ", given up as unreachable"))

    def drain(self, seconds: float = DRAIN_SECONDS) -> None:
        """Every bundle written back, waiting up to `seconds` for the uploads, at the end of pytest's
        session; what is left is abandoned with the process, as it is where the upload thread
        stopped on an error it does not expect, which its own traceback reports."""
        if self.file is not None:
            self.closed(self.file)
        for file in list(self.written):
            self.closed(file)
        began = time.monotonic()
        while (self.uploads.unfinished_tasks and self.uploader.is_alive()
               and time.monotonic() - began < seconds):
            time.sleep(0.1)


def bundle(node: str, group: str | None, worker: str | None) -> str:
    """The bundle of test `node`'s file in its execution partition: a split file's group
    (tools/armada/split.py's `G/N`) and an xdist worker compile apart, so each keeps its own."""
    parts = (group and group.replace("/", "-of-"), worker)
    return "/".join([node.split("::")[0], *(part for part in parts if part)])


HEADER = struct.Struct(">IQ")
"""A bundle entry's key's and value's lengths."""


def packed(entries: list[tuple[str, bytes]]) -> bytes:
    """`entries` as one bundle: each key's and value's length, then the key and the value."""
    return b"".join(HEADER.pack(len(key.encode()), len(value)) + key.encode() + value
                    for key, value in entries)


def unpacked(bundle: bytes) -> list[tuple[str, bytes]]:
    """The `(key, value)` entries of a `packed` bundle, or ValueError where it is not one whole: more
    than `MAX_BYTES`, an entry its lengths run past, or a key that is no file name in one directory
    (empty, `.` or `..`, a separator or NUL in it, or too long for its `-cache` file's name)."""
    if len(bundle) > MAX_BYTES:
        raise ValueError(f"a bundle of more than {MAX_BYTES} bytes")
    entries, at = [], 0
    while at < len(bundle):
        if at + HEADER.size > len(bundle):
            raise ValueError(f"a bundle's entry header at {at} runs past its {len(bundle)} bytes")
        keyed, sized = HEADER.unpack_from(bundle, at)
        at += HEADER.size
        if at + keyed + sized > len(bundle):
            raise ValueError(f"a bundle's entry at {at} runs past its {len(bundle)} bytes")
        key = bundle[at:at + keyed].decode()
        if (key in ("", ".", "..") or any(character in key for character in "/\\\0")
                or len(f"{key}{SUFFIX}".encode()) > 255):
            raise ValueError(f"a bundle's key {key[:80]!r} is no file name")
        entries.append((key, bundle[at + keyed:at + keyed + sized]))
        at += keyed + sized
    return entries


def prefix() -> str:
    """The bundles' leading parts: what a cached executable is only valid for beside JAX's own key,
    and the device's kind, whose programs are another file's on another GPU. jax's own version is
    not among them: JAX's key holds the program and jaxlib's version
    (jax/_src/cache_key.py), so a release of jax alone that lowers a program differently keys it
    anew, and one that does not (a post-release of Dew's patch stack) keeps every file's programs.
    The Python minor is among them because jax 0.11.2 names 3.14's stdlib zstd codec "zlib" in its
    key (`dew.cache.default_compilation_cache_dir`); and the host's CPU class (`HOST.cpu_class`),
    on every backend, as XLA:CPU compiles for the features of the host that compiles and a GPU lane's
    bundles hold CPU executables too (its host-side computations): armada's hosts span classes, and a
    program from another class need not link here, after which the process compiles nothing more."""
    import jax
    import jaxlib

    kind = "".join(character if character.isalnum() else "-" for character in jax.devices()[0].device_kind)
    backend = jax.default_backend()
    minor = f"py{sys.version_info.major}.{sys.version_info.minor}"
    return f"jaxlib{jaxlib.__version__}/{backend}/{minor}/{kind}/{HOST.cpu_class()}"


def install(url: str, token: str) -> RemoteCache | None:
    """A `RemoteCache` at `url` over the cache directory JAX is configured with, or None, logged,
    where JAX keeps no local directory (the cache off, none set, or one on a remote store) or `url`
    is no http(s) address."""
    import jax

    directory = jax.config.jax_compilation_cache_dir
    if not jax.config.jax_enable_compilation_cache or not directory or "://" in directory:
        logger.warning("the shared XLA cache needs JAX's cache on in a local directory, not %r", directory)
        return None
    try:
        address = urllib.parse.urlparse(url)
        # Reading the port raises here, not at the first request, where it is no number.
        valid = address.scheme in ("http", "https") and bool(address.hostname) and address.port != 0
    except ValueError:
        valid = False
    if not valid:
        logger.warning("the shared XLA cache's URL %r is no http(s) address", url)
        return None
    return RemoteCache(Path(directory), url, token, prefix())


def install_from_environment() -> RemoteCache | None:
    """`install` where DEW_XLA_CACHE_URL and DEW_XLA_CACHE_TOKEN are set. The local cache keeps no
    LRU bound: a task's container is short-lived, and the bound makes JAX lock and stamp its index on
    every compilation. The remote's bound is the bucket's 14-day expiry."""
    url, token = os.environ.get("DEW_XLA_CACHE_URL"), os.environ.get("DEW_XLA_CACHE_TOKEN")
    return install(url, token) if url and token else None
