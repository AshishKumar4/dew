"""A shared layer under JAX's persistent compilation cache, for CI on armada.

Each armada task starts in a fresh container, so the on-disk cache conftest
turns on (`dew.cache.enable_compilation_cache`) is always cold there, and
the suite spends most of its time in XLA: a parallelism-matrix cell ran 367
s cold and 99 s with its compilations cached. With DEW_XLA_CACHE_URL and
DEW_XLA_CACHE_TOKEN set (.armada.json), `install` puts `RemoteCache` over
JAX's own cache: a local miss is fetched from the URL, the dew-xla-cache
Worker (tools/armada/xla_cache), and an entry that took at least `REMOTE_SECONDS`
to compile is written to both. Entries are keyed under jax's and jaxlib's
versions, the backend and the Python minor, beside JAX's own key of the
program and its flags.

A test that runs a model eagerly compiles thousands of small programs, each
quicker than `REMOTE_SECONDS`, so never in the remote: asking the Worker about each
one took a DeepSeek drafting test from 103 s to 962 s. So the remote also
keeps the name of each program it holds (`names/<module>`), a process lists
those names once, and `get` asks only about a program of one of them.

The remote can only make a task slower: a fetch or an upload that fails
counts against it, and after `FAILURES` in a row it is left alone for the
rest of the process, so an outage costs a few timeouts. Uploads run on a
background thread, and the process waits up to `DRAIN_SECONDS` for them at
exit.
"""

import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

FAILURES = 3
REMOTE_SECONDS = 1
TIMEOUT_SECONDS = 10.0
DRAIN_SECONDS = 60.0
MAX_BYTES = 256 * 1024 * 1024

logger = logging.getLogger(__name__)


class RemoteCache:
    """`base` (a JAX `CacheInterface`), with entries also read from and written to `url`."""

    def __init__(self, base, url: str, token: str, prefix: str):
        self.base, self.url, self.prefix = base, url.rstrip("/"), prefix.strip("/")
        # Cloudflare refuses urllib's own User-Agent with a 403 (error 1010) before the Worker sees it.
        self.headers = {"Authorization": f"Bearer {token}", "User-Agent": "dew-ci-xla-cache"}
        self.failures = 0
        self.lock = threading.Lock()
        # Daemon threads, which the interpreter does not wait for at exit; `drain` bounds the wait.
        self.uploads: queue.Queue[tuple[str, bytes]] = queue.Queue()
        for _ in range(2):
            threading.Thread(target=self._uploading, daemon=True, name="xla-cache").start()
        self.fetched = self.uploaded = 0
        self.names: set[str] = set()

    @property
    def available(self) -> bool:
        return self.failures < FAILURES

    def _failed(self, error: Exception) -> None:
        with self.lock:
            self.failures += 1
            if self.failures == FAILURES:
                logger.warning("the shared XLA cache is unreachable (%s); compiling without it", error)

    def _request(self, key: str, method: str, data: bytes | None = None):
        request = urllib.request.Request(f"{self.url}/{self.prefix}/{key}", data=data, method=method,
                                         headers=self.headers)
        return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)

    def get(self, key: str):
        value = self.base.get(key)
        if value is not None or not self.available or module(key) not in self.names:
            return value
        try:
            with self._request(key, "GET") as response:
                value = response.read()
        except urllib.error.HTTPError as error:
            if error.code != 404:
                self._failed(error)
            return None
        except Exception as error:  # a remote that cannot answer is a miss
            self._failed(error)
            return None
        with self.lock:
            self.failures = 0
            self.fetched += 1
        self.base.put(key, value)
        return value

    def load_names(self) -> None:
        """The names of the programs the remote holds, a page of a thousand at a time."""
        cursor = None
        try:
            while True:
                fields = {"list": f"{self.prefix}/names/"} | ({"cursor": cursor} if cursor else {})
                request = urllib.request.Request(f"{self.url}/?{urllib.parse.urlencode(fields)}",
                                                 headers=self.headers)
                with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    page = json.load(response)
                self.names.update(page["keys"])
                cursor = page["cursor"]
                if not cursor:
                    return
        except Exception as error:  # names that cannot be read leave the remote alone
            with self.lock:
                self.failures = FAILURES
            logger.warning("the shared XLA cache's names are unreadable (%s); compiling without it", error)

    def _upload(self, key: str, value: bytes) -> None:
        try:
            self._request(key, "PUT", value).close()
            if module(key) not in self.names:
                self._request(f"names/{module(key)}", "PUT", b"1").close()
                self.names.add(module(key))
        except Exception as error:  # an entry that does not reach the remote is compiled again elsewhere
            self._failed(error)
        else:
            with self.lock:
                self.failures = 0
                self.uploaded += 1

    def put(self, key: str, value: bytes) -> None:
        self.base.put(key, value)
        if self.available and len(value) <= MAX_BYTES and compile_seconds(value) >= REMOTE_SECONDS:
            self.uploads.put((key, bytes(value)))

    def _uploading(self) -> None:
        while True:
            key, value = self.uploads.get()
            try:
                self._upload(key, value)
            finally:
                self.uploads.task_done()

    def drain(self, seconds: float = DRAIN_SECONDS) -> None:
        """Wait up to `seconds` for the uploads still queued or running, at the end of pytest's
        session; what is left is abandoned with the process."""
        began = time.monotonic()
        while self.uploads.unfinished_tasks and time.monotonic() - began < seconds:
            time.sleep(0.1)


def module(key: str) -> str:
    """The program's name in JAX's cache key, `<module>-<hash>`."""
    return key.rsplit("-", 1)[0]


def compile_seconds(value: bytes) -> int:
    """How long the entry `value` took to compile, in whole seconds, as JAX records it in the entry's
    first bytes."""
    from jax._src import compilation_cache

    try:
        entry = compilation_cache.decompress_executable(value)
        return compilation_cache.extract_executable_and_time(entry)[1]
    except Exception as error:  # an entry this cannot read stays local
        logger.debug("a cache entry's compile time is unreadable: %s", error)
        return 0


def prefix() -> str:
    """The key's leading parts: what a cached executable is only valid for beside JAX's own key.
    The Python minor is among them because jax 0.11.2 names 3.14's stdlib zstd codec "zlib" in its
    key (`dew.cache.default_compilation_cache_dir`)."""
    import jax
    import jaxlib

    return f"jax{jax.__version__}/jaxlib{jaxlib.__version__}/{jax.default_backend()}/" \
           f"py{sys.version_info.major}.{sys.version_info.minor}"


def install(url: str, token: str) -> RemoteCache | None:
    """Put a `RemoteCache` at `url` over JAX's initialized persistent cache, or leave JAX's cache as
    it is, and return None, where its internals are not what this expects."""
    try:
        from jax._src import compilation_cache

        compilation_cache._initialize_cache()
        if compilation_cache._cache is None or isinstance(compilation_cache._cache, RemoteCache):
            return None
        remote = RemoteCache(compilation_cache._cache, url, token, prefix())
        remote.load_names()
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
