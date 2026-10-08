"""A shared layer under JAX's persistent compilation cache, for CI on armada.

Each armada task starts in a fresh container, so the on-disk cache conftest
turns on (`dew.cache.enable_compilation_cache`) is always cold there, and
the suite spends most of its time in XLA: a parallelism-matrix cell ran 367
s cold and 99 s with its compilations cached. With DEW_XLA_CACHE_URL and
DEW_XLA_CACHE_TOKEN set (.armada.json), `install` puts `RemoteCache` over
JAX's own cache: a local miss is fetched from the URL, the dew-xla-cache
Worker (tools/armada/xla_cache), and a new entry is written to both. Entries
are keyed under jax's and jaxlib's versions, the backend and the Python
minor, beside JAX's own key of the program and its flags.

The remote can only make a task slower: a fetch or an upload that fails
counts against it, and after `FAILURES` in a row it is left alone for the
rest of the process, so an outage costs a few timeouts. Uploads run on a
background thread, and the process waits up to `DRAIN_SECONDS` for them at
exit.
"""

import atexit
import concurrent.futures
import logging
import os
import sys
import threading
import urllib.error
import urllib.request

FAILURES = 3
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
        self.uploads = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="xla-cache")
        self.pending: set[concurrent.futures.Future] = set()
        self.fetched = self.uploaded = 0

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
        if value is not None or not self.available:
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

    def _upload(self, key: str, value: bytes) -> None:
        try:
            self._request(key, "PUT", value).close()
        except Exception as error:  # an entry that does not reach the remote is compiled again elsewhere
            self._failed(error)
        else:
            with self.lock:
                self.failures = 0
                self.uploaded += 1

    def put(self, key: str, value: bytes) -> None:
        self.base.put(key, value)
        if self.available and len(value) <= MAX_BYTES:
            future = self.uploads.submit(self._upload, key, bytes(value))
            with self.lock:
                self.pending.add(future)
            future.add_done_callback(self._settled)

    def _settled(self, future) -> None:
        with self.lock:
            self.pending.discard(future)

    def drain(self, seconds: float = DRAIN_SECONDS) -> None:
        """Wait up to `seconds` for the uploads still running."""
        with self.lock:
            pending = set(self.pending)
        concurrent.futures.wait(pending, timeout=seconds)
        self.uploads.shutdown(wait=False, cancel_futures=True)


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
        compilation_cache._cache = remote
    except Exception as error:  # a jax whose cache has moved runs on its local cache alone
        logger.warning("the shared XLA cache is not installed: %s", error)
        return None
    atexit.register(remote.drain)
    return remote


LOCAL_MAX_BYTES = 4 * 1024 ** 3
"""The local cache's own LRU bound under the remote; the remote's is the bucket's 14-day expiry."""


def install_from_environment() -> RemoteCache | None:
    url, token = os.environ.get("DEW_XLA_CACHE_URL"), os.environ.get("DEW_XLA_CACHE_TOKEN")
    if not (url and token):
        return None
    import jax

    jax.config.update("jax_compilation_cache_max_size", LOCAL_MAX_BYTES)
    return install(url, token)
