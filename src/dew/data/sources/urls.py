"""URL catalogs and volatile fetches, executed by the ordinary Grain worker path."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import logging
import os
import threading
import time
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import urljoin, urlsplit

import aiohttp
from fsspec.asyn import get_loop, sync
from fsspec.implementations.http import HTTPFileSystem

_log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "dew-data/1", "Accept-Encoding": "identity"}
# As urllib's redirect handler: a redirect's status and the hops followed before giving up.
_REDIRECTS, _HOPS = (301, 302, 303, 307, 308), 10

_FILESYSTEMS: dict[tuple[int, int], tuple[asyncio.AbstractEventLoop, HTTPFileSystem]] = {}
"""Each process's HTTP filesystem for each of fsspec's IO loops it fetches on, by the process and the loop,
the loop held beside it. A process can run two such loops: two threads taking fsspec's lazily made lock at
once each start one, as a process's first fetches on a fetch's own threads can. fsspec's instance cache
tells running loops apart by nothing (its token reads a loop's repr), so it would hand a fetch on one the
filesystem whose aiohttp session lives on the other, whose futures that loop never runs (row 1942)."""


def _filesystem(loop: asyncio.AbstractEventLoop) -> HTTPFileSystem:
    """The HTTP filesystem whose session lives on `loop`, in this process (`_FILESYSTEMS`)."""
    key = (os.getpid(), id(loop))
    held = _FILESYSTEMS.get(key)
    if held is None:
        options = {"auto_decompress": False, "timeout": aiohttp.ClientTimeout()}
        made = HTTPFileSystem(loop=loop, client_kwargs=options, skip_instance_cache=True)
        # Two threads making one at once keep whichever lands first; the other opened no session.
        held = _FILESYSTEMS.setdefault(key, (loop, made))
    return held[1]


def _proxy(url: str) -> str | None:
    """The proxy urllib's environment names for `url`, none where its host
    and port are excluded (NO_PROXY), as urllib decides each hop; one named
    by its authority alone is an HTTP proxy, as urllib reads it."""
    parts = urlsplit(url)
    if urllib.request.proxy_bypass(parts.netloc):
        return None
    proxy = urllib.request.getproxies().get(parts.scheme)
    return proxy if proxy is None or "://" in proxy else f"http://{proxy}"


async def _get(session: aiohttp.ClientSession, url: str, limit: int,
               seconds: float) -> tuple[int, bytes, str | None]:
    """The status of a GET of `url` and at most `limit` bytes of its body as
    its server sent them, read as they arrive, all within `seconds`: every hop
    of the redirects it follows (`Location`, else `URI`), each through its own
    proxy, and the body. As urllib's, a redirect goes on to an HTTP(S) url or
    ends at an FTP one, returned unread for urllib to read; to any other, or
    past the hops urllib follows, it ends at its redirect status."""
    status = 0
    async with asyncio.timeout(seconds):
        for _ in range(_HOPS + 1):
            request = session.get(url, headers=_HEADERS, allow_redirects=False, proxy=_proxy(url))
            async with request as response:
                status = response.status
                target = response.headers.get("Location") or response.headers.get("URI")
                if status in _REDIRECTS and target is not None:
                    url = urljoin(url, target)
                    scheme = urlsplit(url).scheme
                    if scheme in ("http", "https"):
                        continue
                    return status, b"", url if scheme == "ftp" else None
                payload = bytearray()
                if status < 300:
                    async for chunk in response.content.iter_chunked(2**16):
                        payload += chunk
                        if len(payload) >= limit:
                            break
                return status, bytes(payload[:limit]), None
    return status, b"", None


@dataclass(frozen=True)
class Fetch:
    """Fetch the bytes at a record's url into `output`, in one attempt: a url
    that fails now is dropped as transient, for a later epoch or a cache
    refill to fetch again; one its server reports gone (HTTP 404 or 410) is
    dropped for good. An HTTP(S) url is read through fsspec's aiohttp session
    as urllib would read it, its redirects and environment's proxies
    included, so a reader waits at most `timeout` on it from connecting to its
    last byte, however slowly a server sends; any other url (a file's) by
    urllib. Resolving a host is the system resolver's, which takes no timeout.
    A cache never freezes a fetch's outcome; only `snapshot(url)` seals
    outcomes. `threads` fetches that many records' urls at once on a pool of
    the stage's own, whatever the read's threads, and delivers them in the
    order they were read; however the step is read (the shards an interleave
    opens, a cache's reads of its inputs), each loading process runs at most
    that many of its fetches at once. It changes no record, so no plan or key
    records it."""

    field: str = "url"
    output: str = "bytes"
    timeout: float = 10.0
    max_bytes: int = 64 * 2**20
    threads: int | None = dataclasses.field(default=None, compare=False, repr=False)
    _running: threading.BoundedSemaphore | None = dataclasses.field(
        default=None, init=False, compare=False, repr=False)
    volatile: ClassVar[bool] = True
    rejects: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.timeout <= 0 or self.max_bytes < 1:
            raise ValueError("Fetch needs a positive timeout and max_bytes")
        if self.threads is not None and self.threads < 1:
            raise ValueError("Fetch threads must be positive")
        running = None if self.threads is None else threading.BoundedSemaphore(self.threads)
        object.__setattr__(self, "_running", running)

    def __getstate__(self) -> Mapping[str, object]:
        # The fetches running at once are counted in each process: one that reads the step counts its own.
        return {name: value for name, value in vars(self).items() if name != "_running"}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        vars(self).update(state)
        self.__post_init__()

    def __call__(self, record: Mapping[str, object]):
        from dew.data.pipeline import Rejected

        if self.field not in record:
            raise ValueError(f"Fetch reads its url from {self.field!r}; the record has {sorted(record)}")
        url = record[self.field]
        if not isinstance(url, str):
            return Rejected("invalid_url")
        try:
            with contextlib.nullcontext() if self._running is None else self._running:
                status, payload = self._read(url)
        except ValueError:
            return Rejected("invalid_url")
        except (OSError, aiohttp.ClientError) as error:
            _log.debug("fetch %s failed: %s", url, error)
            return Rejected("transient", permanent=False)
        if status in (404, 410):
            return Rejected(f"http_{status}")
        if not 200 <= status < 300:
            _log.debug("fetch %s returned HTTP %d", url, status)
            return Rejected("transient", permanent=False)
        if len(payload) > self.max_bytes:
            return Rejected("too_large")
        return {**record, self.output: payload, "content_sha256": hashlib.sha256(payload).hexdigest()}

    def _read(self, url: str) -> tuple[int, bytes]:
        """The status and at most `max_bytes` + 1 bytes of `url`; a url urllib
        reads (a file's) has no status but 200."""
        limit, deadline = self.max_bytes + 1, time.monotonic() + self.timeout
        if urlsplit(url).scheme in ("http", "https"):
            # fsspec's own IO loop, where the loop's filesystem keeps its session for every fetch on it
            # (`_filesystem`); the fetch's deadline is the only one (aiohttp's defaults bound each hop
            # otherwise).
            loop = get_loop()
            http = _filesystem(loop)
            status, payload, ftp = sync(loop, _get, sync(loop, http.set_session), url, limit, self.timeout)
            if ftp is None:
                return status, payload
            url = ftp
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"no time left to read {url}")
        # urllib's timeout bounds each wait, not a body that keeps arriving, so the attempt's deadline
        # judges the read once it ends.
        with urllib.request.urlopen(url, timeout=left) as response:
            payload = response.read(limit)
        if time.monotonic() > deadline:
            raise TimeoutError(f"{url} outlasted its {self.timeout:.1f} s")
        return 200, payload
