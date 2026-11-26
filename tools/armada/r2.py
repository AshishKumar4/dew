"""The one client of the R2 Worker (dew-xla-cache), and its contract, r2.json beside
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
own, such as a pooled one, which carries the same protocol.

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


class Malformed(R2Error):
    """An answer that is not what the Worker's contract says it gives (a listing that is no page of keys)."""


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


Sender = Callable[[str, str, Mapping[str, str], "bytes | None", float, "float | None"],
                  tuple[int, Mapping[str, str], Stream]]
"""What sends a request, `(method, url, headers, body, timeout, until)`, and gives its answer whatever its
status, as `(status, headers, body)`, following no redirect: each wait at most `timeout` seconds and, where a
caller gives `until` (a `time.monotonic` instant), the whole of it, connecting, sending and every receive of
its answer's headers, ended by then. A request that does not complete, and a body cut short while it is read,
raise an OSError (or http.client's HTTPException); everything else is the `Store`'s."""


class _Guard:
    """A request's sockets shut at `until`, which ends whatever receive or send waits on one then: its every
    wait, however many, within its caller's end. It watches each socket the request comes to use (`watch`):
    the one connected, TLS's wrapping of it, and the one its answer reads, even once a `Connection: close`
    answer detaches it from its connection. Disarmed (`disarm`) under the same lock it fires under, so once
    disarmed no shutdown follows, and whether it fired first says whether its connection may be kept. None
    (`until` None) guards nothing."""

    def __init__(self, until: float | None):
        self.lock = threading.Lock()
        self.fired = self.disarmed = False
        self.sockets: list[socket.socket] = []
        self.timer = None
        if until is not None:
            self.timer = threading.Timer(max(until - time.monotonic(), 0.0), self.fire)
            self.timer.daemon = True
            self.timer.start()

    @staticmethod
    def shut(sock: socket.socket) -> None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)

    def watch(self, sock: socket.socket | None) -> None:
        """`sock` shut at `until` too, at once where it has passed."""
        if sock is None:
            return
        with self.lock:
            if self.fired and not self.disarmed:
                self.shut(sock)
            elif sock not in self.sockets:
                self.sockets.append(sock)

    def fire(self) -> None:
        with self.lock:
            if self.disarmed:
                return
            self.fired = True
            for sock in self.sockets:
                self.shut(sock)

    def disarm(self) -> bool:
        """The guard stopped: True where it had not fired, its sockets untouched."""
        with self.lock:
            self.disarmed = True
            if self.timer is not None:
                self.timer.cancel()
            return not self.fired


def _connect(connection: http.client.HTTPConnection, guard: _Guard, until: float | None) -> None:
    """`connection` connected (its name resolved, its TCP and TLS handshakes) within `until` where one is
    given, its socket then watched by `guard`. A connect still going at `until` is left to finish on its own
    thread, which closes what it made; the caller is given a TimeoutError then."""
    if until is None:
        connection.connect()
        guard.watch(connection.sock)
        return
    outcome: dict[str, BaseException | bool] = {}
    abandoned = threading.Event()

    def connect() -> None:
        try:
            connection.connect()
            outcome["done"] = True
        except BaseException as error:  # handed to the caller below
            outcome["error"] = error
        if abandoned.is_set():
            connection.close()

    thread = threading.Thread(target=connect, daemon=True)
    thread.start()
    thread.join(max(until - time.monotonic(), 0.0))
    if thread.is_alive():
        abandoned.set()
        raise TimeoutError("the connection was not made in time")
    if isinstance(error := outcome.get("error"), BaseException):
        raise error
    guard.watch(connection.sock)


def _wait(timeout: float, until: float | None, what: str) -> float:
    """A wait of at most `timeout` seconds and what is left until `until` (none, None); a TimeoutError where
    nothing is left."""
    if until is None:
        return timeout
    left = until - time.monotonic()
    if left <= 0:
        raise TimeoutError(f"{what}: no time left")
    return min(timeout, left)


class _Answer:
    """An http.client answer as a `Stream`: each read one receive of its socket, bounded as asked."""

    def __init__(self, connection: http.client.HTTPConnection, sock: socket.socket,
                 response: http.client.HTTPResponse, guard: _Guard, until: float | None):
        self.connection, self.sock, self.response = connection, sock, response
        self.guard, self.until = guard, until
        # The socket its answer reads, its connection's or one a `Connection: close` answer detached.
        guard.watch(sock)

    def read1(self, size: int, seconds: float) -> bytes:
        self.sock.settimeout(_wait(seconds, self.until, "a body's read"))
        data = self.response.read1(size)
        # Read on a socket its guard shut: an end of the stream that is no end of the body (a chunk's size
        # line cut short reads as the last chunk), never data.
        if self.guard.fired:
            raise TimeoutError("the answer's body was read past its caller's end")
        return data

    def close(self) -> None:
        self.guard.disarm()
        self.response.close()
        self.connection.close()


def urllib_send(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float,
                until: float | None = None) -> tuple[int, Mapping[str, str], Stream]:
    """The standard library's sender: a connection a request, following no redirect, which a caller that
    runs before any environment exists (ci.py, remote_cache) can use. Its socket is held from the request
    on, so each read of the answer's body waits as long as the `Body` reading it allows; the request whole
    ends by `until` (`_Guard`)."""
    parts = urllib.parse.urlsplit(url)
    if parts.hostname is None:
        raise ValueError(f"{url} names no host")
    opened = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = opened(parts.hostname, parts.port, timeout=_wait(timeout, until, url))
    guard = _Guard(until)
    try:
        _connect(connection, guard, until)
        connection.request(method, f"{parts.path or '/'}{f'?{parts.query}' if parts.query else ''}",
                           body=body, headers=dict(headers))
        sock = connection.sock
        response = connection.getresponse()
        if guard.fired:
            raise TimeoutError("the answer's headers came past its caller's end")
    except BaseException:
        guard.disarm()
        connection.close()
        raise
    return response.status, dict(response.headers), _Answer(connection, sock, response, guard, until)


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

    def __call__(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float,
                 until: float | None = None) -> tuple[int, Mapping[str, str], Stream]:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
        if host is None:
            raise ValueError(f"{url} names no host")
        place = (parts.scheme, host, parts.port)
        kept = getattr(self.local, "connections", None)
        if kept is None:
            kept = self.local.connections = {}
        target = f"{parts.path or '/'}{f'?{parts.query}' if parts.query else ''}"
        opened = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        held = kept.pop(place, None)
        while True:
            # What is left of `until`, read afresh for each attempt: a retry on a new connection waits no
            # longer than the first left it.
            wait = _wait(timeout, until, url)
            connection = held or opened(host, parts.port, timeout=wait)
            connection.timeout = wait
            guard = _Guard(until)
            try:
                if connection.sock is None:
                    _connect(connection, guard, until)
                else:
                    connection.sock.settimeout(wait)
                    guard.watch(connection.sock)
                connection.request(method, target, body=body, headers=dict(headers))
                # Held before the answer, which detaches it from a connection the Worker closes after it, as
                # `urllib_send` holds it.
                sock = connection.sock
                response = connection.getresponse()
                if guard.fired:
                    raise TimeoutError("the answer's headers came past its caller's end")
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                in_time = guard.disarm()
                connection.close()
                if held is None or not in_time:
                    raise
                held = None
                continue
            except BaseException:
                guard.disarm()
                connection.close()
                raise
            return (response.status, dict(response.headers),
                    _Kept(kept, place, connection, sock, response, guard, until))


class _Kept(_Answer):
    """A `KeptSend` answer: closed, its connection goes back to its thread's where it can carry another."""

    def __init__(self, kept: dict, place: tuple, connection: http.client.HTTPConnection, sock: socket.socket,
                 response: http.client.HTTPResponse, guard: _Guard, until: float | None):
        super().__init__(connection, sock, response, guard, until)
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
        # Disarmed before it is kept: a guard that fired first, or fires no more once disarmed, leaves no
        # shutdown pending on a connection the next request takes.
        in_time = self.guard.disarm()
        if (in_time and response.isclosed() and not response.will_close
                and self.place not in self.kept):
            self.kept[self.place] = self.connection
        else:
            super().close()

    def drained(self) -> None:
        """The answer's unread body read off until it ends or `DRAIN_SECONDS` pass, each receive waiting at
        most what is left of them; a body that does not come whole in time leaves the answer open, and its
        connection is closed."""
        ends = time.monotonic() + DRAIN_SECONDS
        ends = ends if self.until is None else min(ends, self.until)
        # A body cut short or refused leaves the answer open, which `close` reads as a connection to close.
        with contextlib.suppress(OSError, http.client.HTTPException):
            while not self.response.isclosed() and (left := ends - time.monotonic()) > 0:
                if not self.read1(DRAINED, left):
                    break


class Body:
    """An answer's body as it arrives, read at most `limit` bytes in all: past them a ValueError, and an end
    before the length the Worker announced, a connection cut while reading, or a body arriving slower than
    `MIN_RATE` past `timeout` seconds, `Unavailable`."""

    def __init__(self, stream: Stream, headers: Mapping[str, str], limit: int, what: str, timeout: float,
                 until: float | None = None):
        self.stream, self.left, self.what, self.timeout = stream, limit, what, timeout
        announced = {name.lower(): value for name, value in headers.items()}.get("content-length")
        self.owed = None if announced is None else int(announced)
        self.ends = time.monotonic() + timeout + (limit if self.owed is None else self.owed) / MIN_RATE
        # A caller's own end (`until`, a `time.monotonic` instant) bounds it whole, no allowance past it.
        if until is not None:
            self.ends = min(self.ends, until)

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
                limit: int = MAX_BYTES, until: float | None = None) -> Iterator[Reply]:
        """The Worker's answer to `method` on `key` (the bucket's root, for a listing, when empty), held
        open while the caller reads its body (at most `limit` bytes; none for a 304 or 404), the whole of it
        within `until` (a `time.monotonic` instant) where one is given: its every wait no longer than what is
        left, no allowance past it. A key or a body the Worker would refuse is a ValueError, raised before
        anything is sent; any other failure, its time out among them, an `R2Error`."""
        if key and not valid(key):
            raise ValueError(f"the R2 Worker takes no key {key!r}")
        if body is not None and not CONTRACT["min_bytes"] <= len(body) <= MAX_BYTES:
            raise ValueError(f"a body of {len(body)} bytes, outside the Worker's {CONTRACT['min_bytes']} to "
                             f"{MAX_BYTES}")
        url = f"{self.url.rstrip('/')}/{urllib.parse.quote(key, safe='/=')}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        what = f"{method} {key or url}"
        timeout = self.timeout if until is None else min(self.timeout, until - time.monotonic())
        if timeout <= 0:
            raise Unavailable(f"{what}: no time left to ask")
        try:
            status, given, stream = self.send(method, url, {
                "Authorization": f"Bearer {self.token}", "User-Agent": self.user_agent, **(headers or {})},
                body, timeout, until)
        except (OSError, http.client.HTTPException) as error:  # a request that did not complete
            raise Unavailable(f"{what} did not complete: {error}") from error
        with contextlib.closing(stream):
            if not 200 <= status < 300 and status not in (304, 404):
                raised = (Conflict if status in (409, 412) else
                          Unavailable if status == 429 or status >= 500 else Refused)
                raise raised(f"the R2 Worker answered {status} to {what}")
            answer = (None if method == "HEAD" or status in (304, 404) else
                      Body(stream, given, limit, f"the answer to {what}", timeout, until))
            yield Reply(status, dict(given), answer)

    def get(self, key: str, limit: int = MAX_BYTES, until: float | None = None) -> bytes | None:
        """The object at `key`, or None where there is none, within `until` where one is given (`request`)."""
        with self.request("GET", key, limit=limit, until=until) as reply:
            return None if reply.body is None else reply.body.whole()

    def put(self, key: str, body: bytes, until: float | None = None) -> None:
        """`body` written at `key`, within `until` where one is given (`request`)."""
        with self.request("PUT", key, body=body, until=until):
            pass

    def delete(self, key: str) -> None:
        """The object at `key` deleted, where there is one."""
        with self.request("DELETE", key):
            pass

    def keys(self, prefix: str, until: float | None = None) -> Iterator[str]:
        """The keys under `prefix`, without it, a page at a time, each page within `until` where one is given
        (`request`). A page that is no object of a list of keys and a cursor (a string, or none at the
        listing's end) is `Malformed`."""
        cursor = None
        while True:
            query = {"list": prefix} | ({"cursor": cursor} if cursor else {})
            with self.request("GET", query=query, until=until) as reply:
                if reply.body is None:
                    raise Malformed(f"a listing of {prefix} has no body")
                try:
                    page = json.loads(reply.body.whole())
                except ValueError as error:
                    raise Malformed(f"a listing of {prefix} is no JSON: {error}") from error
            if not (isinstance(page, dict) and isinstance(page.get("keys"), list)
                    and all(isinstance(key, str) for key in page["keys"]) and "cursor" in page
                    and (page["cursor"] is None or isinstance(page["cursor"], str))):
                raise Malformed(f"a listing of {prefix} is no page of keys and a cursor")
            yield from page["keys"]
            cursor = page["cursor"]
            if not cursor:
                return
