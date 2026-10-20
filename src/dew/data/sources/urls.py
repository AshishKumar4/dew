"""URL catalogs and volatile fetches, executed by the ordinary Grain worker path."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import urlsplit

import aiohttp
import fsspec
from fsspec.asyn import sync
from fsspec.implementations.http import HTTPFileSystem

_log = logging.getLogger(__name__)


async def _get(session: aiohttp.ClientSession, url: str, limit: int) -> tuple[int, bytes]:
    """The status of a GET of `url` and at most `limit` bytes of its body, read as they arrive."""
    async with session.get(url, headers={"User-Agent": "dew-data/1"}) as response:
        payload = bytearray()
        if response.status < 400:
            async for chunk in response.content.iter_chunked(2**16):
                payload += chunk
                if len(payload) >= limit:
                    break
        return response.status, bytes(payload[:limit])


@dataclass(frozen=True)
class Fetch:
    """Fetch the bytes at a record's url into `output`, in one attempt: a url
    that fails now is dropped as transient, for a later epoch or a cache
    refill to fetch again; one its server reports gone (HTTP 404 or 410) is
    dropped for good. An HTTP(S) url is read through fsspec's aiohttp session,
    the proxies its environment names and the redirects its server gives
    included, so a reader waits at most `timeout` on it from connecting to its
    last byte, however slowly a server sends (aiohttp's total deadline); any
    other url, through the store fsspec reads it from. Resolving a host is the
    system resolver's, which takes no timeout. A cache never freezes a fetch's
    outcome; only `snapshot(url)` seals outcomes."""

    field: str = "url"
    output: str = "bytes"
    timeout: float = 10.0
    max_bytes: int = 64 * 2**20
    volatile: ClassVar[bool] = True
    rejects: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.timeout <= 0 or self.max_bytes < 1:
            raise ValueError("Fetch needs a positive timeout and max_bytes")

    def __call__(self, record: Mapping[str, object]):
        from dew.data.pipeline import Rejected

        if self.field not in record:
            raise ValueError(f"Fetch reads its url from {self.field!r}; the record has {sorted(record)}")
        url = record[self.field]
        if not isinstance(url, str):
            return Rejected("invalid_url")
        try:
            status, payload = self._read(url)
        except ValueError:
            return Rejected("invalid_url")
        except (OSError, aiohttp.ClientError) as error:
            _log.debug("fetch %s failed: %s", url, error)
            return Rejected("transient", permanent=False)
        if status in (404, 410):
            return Rejected(f"http_{status}")
        if status >= 400:
            _log.debug("fetch %s returned HTTP %d", url, status)
            return Rejected("transient", permanent=False)
        if len(payload) > self.max_bytes:
            return Rejected("too_large")
        return {**record, self.output: payload, "content_sha256": hashlib.sha256(payload).hexdigest()}

    def _read(self, url: str) -> tuple[int, bytes]:
        """The status and at most `max_bytes` + 1 bytes of `url`; a url read from a store
        other than HTTP(S) has no status but 200."""
        limit = self.max_bytes + 1
        if urlsplit(url).scheme in ("http", "https"):
            http = HTTPFileSystem(client_kwargs={"trust_env": True,
                                                 "timeout": aiohttp.ClientTimeout(total=self.timeout)})
            return sync(http.loop, _get, sync(http.loop, http.set_session), url, limit)
        with fsspec.open(url, "rb") as file:
            return 200, file.read(limit)
