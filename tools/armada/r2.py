"""The one client of the R2 Worker (dew-xla-cache, deployed from dew-ops), and its contract, r2.json beside
this file: the bodies the Worker takes (none empty, and none past Cloudflare's request bound on the
account's plan) and its key rule. The Worker reads the same file, so the client refuses what the Worker
would before sending it.

One request path (`Store.request`) holds the Worker's protocol: the bearer token, the caller's User-Agent
(Cloudflare refuses urllib's own with error 1010 before the Worker sees it), no redirect followed (urllib's
would resend the token wherever the answer points), a body read as it arrives, to a bound and to the length
the Worker announced, and each status as an outcome or a typed error. A 404 and a 304 are answers; a refusal
(401, 403 or any other status) is `Refused`, an unmet precondition (409, 412) `Conflict`, and a Worker that is
busy or down (429, 5xx), or a request that did not complete, `Unavailable`. Each caller keeps its own policy:
what a failure costs it, and whether to retry.

The standard library alone: tools/armada/ci.py imports it as its sibling and tests/remote_cache.py loads it by
its path, both before any environment exists to import Dew from."""

import contextlib
import http.client
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

CONTRACT = json.loads(Path(__file__).with_name("r2.json").read_text())
MAX_BYTES: int = CONTRACT["max_bytes"]
"""The largest body the Worker takes or gives."""
KEY = re.compile(CONTRACT["key"])


class R2Error(Exception):
    """A request the Worker did not answer as asked."""


class Refused(R2Error):
    """A request the Worker refused: a token it does not take (401, 403) or another status that is no
    answer, a redirect among them."""


class Conflict(R2Error):
    """A precondition the object did not meet (409, 412)."""


class Unavailable(R2Error):
    """A Worker that is busy or down (429, 5xx), or a request that did not complete."""


def valid(key: str) -> bool:
    """Whether the Worker takes `key` (r2.json): at most max_key_length characters of letters, digits and
    `._=-` in /-separated parts, none of them `..`."""
    return (len(key) <= CONTRACT["max_key_length"] and KEY.fullmatch(key) is not None
            and ".." not in key.split("/"))


class _Unredirected(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_Unredirected)


class Body:
    """An answer's body as it arrives, read at most `limit` bytes in all: past them a ValueError, and an end
    before the length the Worker announced, or a connection cut while reading, `Unavailable`."""

    def __init__(self, response: http.client.HTTPResponse, limit: int, what: str):
        self.response, self.left, self.what = response, limit, what
        announced = response.headers.get("Content-Length")
        self.owed = None if announced is None else int(announced)

    def read(self, size: int = -1) -> bytes:
        """At most `size` bytes more (all the rest, under -1); b"" at the body's end."""
        wanted = self.left + 1 if size < 0 else min(size, self.left + 1)
        try:
            data = self.response.read(wanted)
        except (OSError, http.client.HTTPException) as error:
            raise Unavailable(f"{self.what} was cut short: {error}") from error
        if len(data) > self.left:
            raise ValueError(f"{self.what} is past its {self.left + len(data) - 1}-byte bound")
        self.left -= len(data)
        if self.owed is not None:
            self.owed -= len(data)
            if not data and wanted and self.owed > 0:
                raise Unavailable(f"{self.what} ended {self.owed} bytes before its announced length")
        return data

    def whole(self) -> bytes:
        """The rest of the body."""
        parts = []
        while part := self.read(2 ** 20):
            parts.append(part)
        return b"".join(parts)


@dataclass(frozen=True)
class Reply:
    """An answer: its status (2xx, 304 or 404), its headers, and its body as it arrives."""

    status: int
    headers: Mapping[str, str]
    body: Body | None


@dataclass(frozen=True)
class Store:
    """The Worker at `url`, asked with `token` under the caller's `user_agent`, each request bounded by
    `timeout` seconds. Immutable, so threads share one."""

    url: str
    token: str
    user_agent: str
    timeout: float

    @contextlib.contextmanager
    def request(self, method: str, key: str = "", *, query: Mapping[str, str] | None = None,
                headers: Mapping[str, str] | None = None, body: bytes | None = None,
                limit: int = MAX_BYTES) -> Iterator[Reply]:
        """The Worker's answer to `method` on `key` (the bucket's root, for a listing, when empty), held
        open while the caller reads its body (at most `limit` bytes; none for a 304 or 404). A key or a body
        the Worker would refuse is a ValueError, raised before anything is sent; any other failure an
        `R2Error`."""
        if key and not valid(key):
            raise ValueError(f"the R2 Worker takes no key {key!r}")
        if body is not None and not CONTRACT["min_bytes"] <= len(body) <= MAX_BYTES:
            raise ValueError(f"a body of {len(body)} bytes, outside the Worker's {CONTRACT['min_bytes']} to "
                             f"{MAX_BYTES}")
        url = f"{self.url.rstrip('/')}/{urllib.parse.quote(key, safe='/=')}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        what = f"{method} {key or url}"
        request = urllib.request.Request(url, data=body, method=method, headers={
            "Authorization": f"Bearer {self.token}", "User-Agent": self.user_agent, **(headers or {})})
        try:
            response = _OPENER.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            if error.code in (304, 404):
                error.close()
                yield Reply(error.code, dict(error.headers), None)
                return
            raised = (Conflict if error.code in (409, 412) else
                      Unavailable if error.code == 429 or error.code >= 500 else Refused)
            raise raised(f"the R2 Worker answered {error.code} to {what}") from error
        except (OSError, http.client.HTTPException) as error:  # URLError, a timeout, a connection cut short
            raise Unavailable(f"{what} did not complete: {error}") from error
        with response:
            answer = Body(response, limit, f"the answer to {what}")
            yield Reply(response.status, dict(response.headers), answer)

    def get(self, key: str, limit: int = MAX_BYTES) -> bytes | None:
        """The object at `key`, or None where there is none."""
        with self.request("GET", key, limit=limit) as reply:
            return None if reply.body is None else reply.body.whole()

    def put(self, key: str, body: bytes) -> None:
        """`body` written at `key`."""
        with self.request("PUT", key, body=body):
            pass

    def keys(self, prefix: str) -> Iterator[str]:
        """The keys under `prefix`, without it, a page at a time."""
        cursor = None
        while True:
            query = {"list": prefix} | ({"cursor": cursor} if cursor else {})
            with self.request("GET", query=query) as reply:
                page = json.loads(reply.body.whole())
            yield from page["keys"]
            cursor = page["cursor"]
            if not cursor:
                return
