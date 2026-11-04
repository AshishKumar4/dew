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
what a failure costs it, and whether to retry. What sends a request is a `Sender`: the standard library's
(`urllib_send`, a connection a request) by default, `KeptSend` (a connection a thread keeps), or the caller's
own, such as dew-ops' pooled one, which carries the same protocol.

The standard library alone: tools/armada/ci.py imports it as its sibling and tests/remote_cache.py loads it by
its path, both before any environment exists to import Dew from."""

import contextlib
import http.client
import json
import re
import socket
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

CONTRACT = json.loads(Path(__file__).with_name("r2.json").read_text())
MAX_BYTES: int = CONTRACT["max_bytes"]
"""The largest body the Worker takes or gives."""
KEY = re.compile(CONTRACT["key"])
MIN_RATE = 2 ** 20
"""The slowest an answer's body may arrive, in bytes a second, past the request's `timeout`: one slower is
given up as `Unavailable`, as a stalled one is by the socket's own timeout (2026-11-03: a test cache's bundle
read 28 minutes before its group's first test, every wait within the socket's timeout)."""


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


class Stream(Protocol):
    """An answer's body as a sender gives it: read in pieces, each what one receive gives, then closed."""

    def read1(self, size: int, seconds: float) -> bytes:
        """At most `size` bytes, what one receive gives, waiting at most `seconds` for it (past them a
        TimeoutError); b"" at the body's end."""
        ...

    def close(self) -> None: ...


Sender = Callable[[str, str, Mapping[str, str], "bytes | None", float], tuple[int, Mapping[str, str], Stream]]
"""What sends a request, `(method, url, headers, body, timeout)`, and gives its answer whatever its status, as
`(status, headers, body)`, following no redirect. A request that does not complete, and a body cut short
while it is read, raise an OSError (or http.client's HTTPException); everything else is the `Store`'s."""


class _Answer:
    """An http.client answer as a `Stream`: each read one receive of its socket, bounded as asked."""

    def __init__(self, connection: http.client.HTTPConnection, sock: socket.socket,
                 response: http.client.HTTPResponse):
        self.connection, self.sock, self.response = connection, sock, response

    def read1(self, size: int, seconds: float) -> bytes:
        self.sock.settimeout(seconds)
        return self.response.read1(size)

    def close(self) -> None:
        self.response.close()
        self.connection.close()


def urllib_send(method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                timeout: float) -> tuple[int, Mapping[str, str], Stream]:
    """The standard library's sender: a connection a request, following no redirect, which a caller that
    runs before any environment exists (ci.py, remote_cache) can use. Its socket is held from the request
    on, so each read of the answer's body waits as long as the `Body` reading it allows."""
    parts = urllib.parse.urlsplit(url)
    opened = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = opened(parts.hostname, parts.port, timeout=timeout)
    connection.request(method, f"{parts.path or '/'}{f'?{parts.query}' if parts.query else ''}", body=body,
                       headers=dict(headers))
    sock = connection.sock
    response = connection.getresponse()
    return response.status, dict(response.headers), _Answer(connection, sock, response)


DRAINED = 2 ** 16
"""The most of an unread answer's body a kept connection reads off to be used again (`KeptSend`): a miss's or
a refusal's few bytes, never an object's."""
DRAIN_SECONDS = 1.0
"""The longest a kept connection waits for an unread answer's body to read it off (`DRAINED`), past which it
is closed: a trickling refusal holds up no caller."""


class KeptSend:
    """A sender that keeps a connection a thread and host open across its requests, where `urllib_send` opens
    one a request and every request pays its TCP and TLS handshakes: a plan's thousands of record lookups
    from a CI runner took 103 s, the same plan's on armada, beside the Worker, 28 s. An answer read to its
    end, or a miss's or a refusal's unread body of at most `DRAINED` bytes that it reads off within
    `DRAIN_SECONDS`, gives its connection back to its thread; any other, or one the Worker said it would
    close, closes it. A request a kept connection could not send, as
    the Worker had closed it meanwhile, goes once on a new one. A connection is never shared between
    threads, so a `Store` with this sender is shared as one with `urllib_send` is."""

    def __init__(self):
        self.local = threading.local()

    def __call__(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                 timeout: float) -> tuple[int, Mapping[str, str], Stream]:
        parts = urllib.parse.urlsplit(url)
        place = (parts.scheme, parts.hostname, parts.port)
        kept = getattr(self.local, "connections", None)
        if kept is None:
            kept = self.local.connections = {}
        target = f"{parts.path or '/'}{f'?{parts.query}' if parts.query else ''}"
        opened = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        held = kept.pop(place, None)
        while True:
            connection = held or opened(parts.hostname, parts.port, timeout=timeout)
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
            try:
                connection.request(method, target, body=body, headers=dict(headers))
                # Held before the answer, which detaches it from a connection the Worker closes after it, as
                # `urllib_send` holds it.
                sock = connection.sock
                response = connection.getresponse()
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                connection.close()
                if held is None:
                    raise
                held = None
                continue
            return response.status, dict(response.headers), _Kept(kept, place, connection, sock, response)


class _Kept(_Answer):
    """A `KeptSend` answer: closed, its connection goes back to its thread's where it can carry another."""

    def __init__(self, kept: dict, place: tuple, connection: http.client.HTTPConnection, sock: socket.socket,
                 response: http.client.HTTPResponse):
        super().__init__(connection, sock, response)
        self.kept, self.place = kept, place

    def close(self) -> None:
        response = self.response
        # An answer with nothing left to read (a PUT's 204, a HEAD's) ends at once. A success's body is the
        # `Body`'s to read, and one it left unread (past its limit or its deadline) closes the connection; a
        # miss's or a refusal's, which no `Body` reads, is read off in its bounds.
        if not response.isclosed() and response.length == 0:
            response.read()
        elif (not response.isclosed() and not 200 <= response.status < 300 and response.length is not None
                and response.length <= DRAINED):
            self.drained()
        if response.isclosed() and not response.will_close and self.place not in self.kept:
            self.kept[self.place] = self.connection
        else:
            super().close()

    def drained(self) -> None:
        """The answer's unread body read off until it ends or `DRAIN_SECONDS` pass, each receive waiting at
        most what is left of them; a body that does not come whole in time leaves the answer open, and its
        connection is closed."""
        ends = time.monotonic() + DRAIN_SECONDS
        # A body cut short or refused leaves the answer open, which `close` reads as a connection to close.
        with contextlib.suppress(OSError, http.client.HTTPException):
            while not self.response.isclosed() and (left := ends - time.monotonic()) > 0:
                if not self.read1(DRAINED, left):
                    break


class Body:
    """An answer's body as it arrives, read at most `limit` bytes in all: past them a ValueError, and an end
    before the length the Worker announced, a connection cut while reading, or a body arriving slower than
    `MIN_RATE` past `timeout` seconds, `Unavailable`."""

    def __init__(self, stream: Stream, headers: Mapping[str, str], limit: int, what: str, timeout: float):
        self.stream, self.left, self.what, self.timeout = stream, limit, what, timeout
        announced = {name.lower(): value for name, value in headers.items()}.get("content-length")
        self.owed = None if announced is None else int(announced)
        self.ends = time.monotonic() + timeout + (limit if self.owed is None else self.owed) / MIN_RATE

    def read(self, size: int = -1) -> bytes:
        """At most `size` bytes more, what one receive gives (all the rest, under -1); b"" at the body's end.
        Each receive waits at most the request's timeout and what is left of the body's deadline."""
        if size < 0:
            return self.whole()
        if size == 0 or self.owed == 0:
            return b""
        slow = f"{self.what} arrived slower than {MIN_RATE} bytes a second"
        left = self.ends - time.monotonic()
        if left <= 0:
            raise Unavailable(slow)
        try:
            data = self.stream.read1(min(size, self.left + 1), min(self.timeout, left))
        except (OSError, http.client.HTTPException) as error:
            raise Unavailable(slow if time.monotonic() >= self.ends
                              else f"{self.what} was cut short: {error}") from error
        if len(data) > self.left:
            raise ValueError(f"{self.what} is past its {self.left + len(data) - 1}-byte bound")
        self.left -= len(data)
        if self.owed is not None:
            self.owed -= len(data)
            # The connection's end before the length it announced: the body stopped short.
            if not data and self.owed > 0:
                raise Unavailable(f"{self.what} ended {self.owed} bytes before its announced length")
        return data

    def whole(self) -> bytes:
        """The rest of the body."""
        parts = []
        while part := self.read(2 ** 16):
            parts.append(part)
        return b"".join(parts)


@dataclass(frozen=True)
class Reply:
    """An answer: its status (2xx, 304 or 404), its headers, and its body as it arrives: None for a 304, a
    404 and a HEAD, whose Content-Length is the object's."""

    status: int
    headers: Mapping[str, str]
    body: Body | None


@dataclass(frozen=True)
class Store:
    """The Worker at `url`, asked with `token` under the caller's `user_agent`, each wait for it bounded by
    `timeout` seconds and each answer's body to arrive at `MIN_RATE` past them, sent by `send` (`urllib_send`,
    else a caller's own, such as a pooled one). Immutable, so threads share one."""

    url: str
    token: str
    user_agent: str
    timeout: float
    send: Sender = urllib_send

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
        try:
            status, given, stream = self.send(method, url, {
                "Authorization": f"Bearer {self.token}", "User-Agent": self.user_agent, **(headers or {})},
                body, self.timeout)
        except (OSError, http.client.HTTPException) as error:  # a request that did not complete
            raise Unavailable(f"{what} did not complete: {error}") from error
        with contextlib.closing(stream):
            if not 200 <= status < 300 and status not in (304, 404):
                raised = (Conflict if status in (409, 412) else
                          Unavailable if status == 429 or status >= 500 else Refused)
                raise raised(f"the R2 Worker answered {status} to {what}")
            answer = (None if method == "HEAD" or status in (304, 404) else
                      Body(stream, given, limit, f"the answer to {what}", self.timeout))
            yield Reply(status, dict(given), answer)

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
