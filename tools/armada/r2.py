"""The one client of the R2 Worker (dew-xla-cache, deployed from dew-ops), and its contract, r2.json beside
this file: the largest body the Worker takes (Cloudflare's request bound on the account's plan) and its key
rule. The Worker reads the same file, so the client refuses what the Worker would before sending it.

One request path (`Store.request`) holds the Worker's protocol: the bearer token, the caller's User-Agent
(Cloudflare refuses urllib's own with error 1010 before the Worker sees it), no redirect followed (urllib's
would resend the token wherever the answer points), a body read to a bound, and each status as an outcome
or a typed error. A 404 and a 304 are answers; a refusal (401, 403 or any other status) is `Refused`, an unmet
precondition (409, 412) `Conflict`, and a Worker that is busy or down (429, 5xx), or a request that did not
complete, `Unavailable`. Each caller keeps its own policy: what a failure costs it, and whether to retry.

The standard library alone: tools/armada/ci.py imports it as its sibling and tests/remote_cache.py loads it by
its path, both before any environment exists to import Dew from."""

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


@dataclass(frozen=True)
class Reply:
    """An answer: its status (2xx, 304 or 404), its headers, and its body."""

    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class Store:
    """The Worker at `url`, asked with `token` under the caller's `user_agent`, each request bounded by
    `timeout` seconds. Immutable, so threads share one."""

    url: str
    token: str
    user_agent: str
    timeout: float

    def request(self, method: str, key: str = "", *, query: Mapping[str, str] | None = None,
                headers: Mapping[str, str] | None = None, body: bytes | None = None,
                limit: int = MAX_BYTES) -> Reply:
        """The Worker's answer to `method` on `key` (the bucket's root, for a listing, when empty): a
        `Reply`, with a body of at most `limit` bytes. A key the Worker would refuse, and a body or an
        answer past its bound, are a ValueError; any other failure an `R2Error`."""
        if key and not valid(key):
            raise ValueError(f"the R2 Worker takes no key {key!r}")
        if body is not None and len(body) > MAX_BYTES:
            raise ValueError(f"a body of {len(body)} bytes, past the Worker's {MAX_BYTES}")
        url = f"{self.url.rstrip('/')}/{urllib.parse.quote(key, safe='/=')}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, data=body, method=method, headers={
            "Authorization": f"Bearer {self.token}", "User-Agent": self.user_agent, **(headers or {})})
        try:
            with _OPENER.open(request, timeout=self.timeout) as response:
                read = response.read(limit + 1)
                status, given = response.status, dict(response.headers)
        except urllib.error.HTTPError as error:
            if error.code in (304, 404):
                return Reply(error.code, dict(error.headers), b"")
            raised = (Conflict if error.code in (409, 412) else
                      Unavailable if error.code == 429 or error.code >= 500 else Refused)
            raise raised(f"the R2 Worker answered {error.code} to {method} {key or url}") from error
        except (OSError, http.client.HTTPException) as error:  # URLError, a timeout, a connection cut short
            raise Unavailable(f"{method} {key or url} did not complete: {error}") from error
        if len(read) > limit:
            raise ValueError(f"the R2 Worker's answer to {method} {key} is past {limit} bytes")
        return Reply(status, given, read)

    def get(self, key: str, limit: int = MAX_BYTES) -> bytes | None:
        """The object at `key`, or None where there is none."""
        reply = self.request("GET", key, limit=limit)
        return None if reply.status == 404 else reply.body

    def put(self, key: str, body: bytes) -> None:
        """`body` written at `key`."""
        self.request("PUT", key, body=body)

    def keys(self, prefix: str) -> Iterator[str]:
        """The keys under `prefix`, without it, a page at a time."""
        cursor = None
        while True:
            query = {"list": prefix} | ({"cursor": cursor} if cursor else {})
            page = json.loads(self.request("GET", query=query).body)
            yield from page["keys"]
            cursor = page["cursor"]
            if not cursor:
                return
