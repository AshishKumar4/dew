"""A shared layer under JAX's persistent compilation cache, for CI on armada and the GPU queue.

Each armada task and GPU pass starts on a cold machine, so the on-disk cache conftest turns on
(`dew.cache.enable_compilation_cache`) is always empty there, and the suite spends most of its time
in XLA: on the Colab A100 tier, compiling took 55% of every file's wall time. With
DEW_XLA_CACHE_URL and DEW_XLA_CACHE_TOKEN set (.armada.json), `install` puts `RemoteCache` over
JAX's own cache, and every program a test file compiles or reads joins that file's bundle on the
dew-xla-cache Worker (deployed from the private dew-ops repository): one GET as the file is
collected puts its last run's programs in the local cache, and one PUT after it ran, where it used
any program the bundle lacked, writes them back. Bundles are kept under jax's and jaxlib's
versions, the backend, the Python minor and the device's kind, beside JAX's own key of each
program and its flags.

A bundle is a file's in one execution partition (conftest's `bundle`): a split file's group, or an
xdist worker, has its own, so partitions that compile apart never replace each other's programs. A
bundle and not a program at a time, as a test that runs a model eagerly compiles thousands of
programs of a few milliseconds each (14,068 of the A100 tier's 16,537, under 50 ms), quicker to
compile than to ask a Worker for: asking about each one took a DeepSeek drafting test from 103 s to
962 s. A file's bundle is its programs' few megabytes (the CPU lane's same-bits test, 6546
programs, 46 MB).

The remote can only make a task slower: a fetch or an upload that fails counts against it, and
after `FAILURES` in a row it is left alone for the rest of the process, so an outage costs a few
timeouts. Being left alone is final and is checked as each request is made: an answer that arrives
after it, a request already under way, does not take the remote back, and the uploads still queued
are dropped unsent. Uploads run on a background thread, and the process waits up to `DRAIN_SECONDS`
for them at exit.
"""

import logging
import os
import queue
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

FAILURES = 3
TIMEOUT_SECONDS = 30.0
DRAIN_SECONDS = 120.0
MAX_BYTES = 96 * 1024 * 1024
"""The largest bundle sent: the Worker's request bound, beside the headers."""

logger = logging.getLogger(__name__)


class RemoteCache:
    """`base` (a JAX `CacheInterface`), each test file's entries also read from and written to `url`."""

    def __init__(self, base, url: str, token: str, prefix: str):
        self.base, self.url, self.prefix = base, url.rstrip("/"), prefix.strip("/")
        # Cloudflare refuses urllib's own User-Agent with a 403 (error 1010) before the Worker sees it.
        self.headers = {"Authorization": f"Bearer {token}", "User-Agent": "dew-ci-xla-cache"}
        self.failures = 0
        self.given_up = False
        self.lock = threading.Lock()
        self.file: str | None = None
        self.used: dict[str, set[str]] = {}
        """The programs each bundle's tests compiled or read in this process, by JAX's key."""
        self.held: dict[str, set[str]] = {}
        """The programs each bundle holds on the Worker, as read or last written."""
        # Daemon threads, which the interpreter does not wait for at exit; `drain` bounds the wait.
        self.uploads: queue.Queue[tuple[str, bytes]] = queue.Queue()
        threading.Thread(target=self._uploading, daemon=True, name="xla-cache").start()
        self.fetched = self.uploaded = 0

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

    def _request(self, file: str, method: str, data: bytes | None = None):
        """The answer to one request for `file`'s bundle, or None, unsent, once the remote is given up."""
        if self.given_up:
            return None
        request = urllib.request.Request(f"{self.url}/{self.prefix}/bundles/{urllib.parse.quote(file)}",
                                         data=data, method=method, headers=self.headers)
        return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)

    def get(self, key: str):
        value = self.base.get(key)
        if value is not None:
            self._used(key)
        return value

    def put(self, key: str, value: bytes) -> None:
        self.base.put(key, value)
        self._used(key)

    def _used(self, key: str) -> None:
        with self.lock:
            if self.file is not None:
                self.used.setdefault(self.file, set()).add(key)

    def opened(self, file: str) -> None:
        """Bundle `file` in the local cache, and the programs compiled or read from here on its."""
        self.file = file
        if file in self.held or not self.available:
            return
        self.held[file] = set()
        try:
            answer = self._request(file, "GET")
            if answer is None:
                return
            with answer as response:
                entries = unpacked(response.read(MAX_BYTES + 1))
        except urllib.error.HTTPError as error:
            if error.code != 404:
                self._failed(error)
            return
        except Exception as error:  # a remote that cannot answer, or answers no whole bundle, is a miss
            self._failed(error)
            return
        self._answered()
        for key, value in entries:
            self.base.put(key, value)
            self.held[file].add(key)
        with self.lock:
            self.fetched += len(entries)

    def closed(self, file: str) -> None:
        """Bundle `file` written back, where its tests used a program it lacked: every program they
        used in this process, so a file whose tests ran apart (another file's between) sends them all."""
        with self.lock:
            used = set(self.used.get(file, ()))
        if not used - self.held.get(file, set()) or not self.available:
            return
        entries = [(key, value) for key in sorted(used) if (value := self.base.get(key)) is not None]
        bundle = packed(entries)
        if len(bundle) <= MAX_BYTES:
            self.held[file] = used
            self.uploads.put((file, bundle))

    def _uploading(self) -> None:
        while True:
            file, bundle = self.uploads.get()
            try:
                answer = self._request(file, "PUT", bundle)
                if answer is not None:
                    answer.close()
                    self._answered()
                    with self.lock:
                        self.uploaded += 1
            except Exception as error:  # a bundle that does not reach the remote is compiled again elsewhere
                self._failed(error)
            finally:
                self.uploads.task_done()

    def summary(self) -> str:
        """What the remote did for this process, which a task's log ends with: a slow task whose
        compilations missed it says so."""
        return (f"shared XLA cache: {self.fetched} programs fetched, {self.uploaded} bundles uploaded"
                + ("" if self.available else ", given up as unreachable"))

    def drain(self, seconds: float = DRAIN_SECONDS) -> None:
        """Every file's bundle written back, waiting up to `seconds` for the uploads, at the end of
        pytest's session; what is left is abandoned with the process."""
        for file in list(self.used):
            self.closed(file)
        began = time.monotonic()
        while self.uploads.unfinished_tasks and time.monotonic() - began < seconds:
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
    than `MAX_BYTES`, or an entry its lengths run past."""
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
        entries.append((bundle[at:at + keyed].decode(), bundle[at + keyed:at + keyed + sized]))
        at += keyed + sized
    return entries


def prefix() -> str:
    """The bundles' leading parts: what a cached executable is only valid for beside JAX's own key,
    and the device's kind, whose programs are another file's on another GPU. The Python minor is
    among them because jax 0.11.2 names 3.14's stdlib zstd codec "zlib" in its key
    (`dew.cache.default_compilation_cache_dir`)."""
    import jax
    import jaxlib

    kind = "".join(character if character.isalnum() else "-" for character in jax.devices()[0].device_kind)
    return f"jax{jax.__version__}/jaxlib{jaxlib.__version__}/{jax.default_backend()}/" \
           f"py{sys.version_info.major}.{sys.version_info.minor}/{kind}"


def install(url: str, token: str) -> RemoteCache | None:
    """Put a `RemoteCache` at `url` over JAX's initialized persistent cache, or leave JAX's cache as
    it is, and return None, where its internals are not what this expects."""
    try:
        from jax._src import compilation_cache

        compilation_cache._initialize_cache()
        if compilation_cache._cache is None or isinstance(compilation_cache._cache, RemoteCache):
            return None
        remote = RemoteCache(compilation_cache._cache, url, token, prefix())
        compilation_cache._cache = remote
    except Exception as error:  # a jax whose cache has moved runs on its local cache alone
        logger.warning("the shared XLA cache is not installed: %s", error)
        return None
    return remote


def install_from_environment() -> RemoteCache | None:
    """`install` where DEW_XLA_CACHE_URL and DEW_XLA_CACHE_TOKEN are set. The local cache keeps no
    LRU bound: a task's container is short-lived, and the bound makes JAX lock and stamp its index on
    every compilation. The remote's bound is the bucket's 14-day expiry."""
    url, token = os.environ.get("DEW_XLA_CACHE_URL"), os.environ.get("DEW_XLA_CACHE_TOKEN")
    return install(url, token) if url and token else None
