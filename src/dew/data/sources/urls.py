"""URL catalogs and volatile fetches, executed by the ordinary Grain worker path."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import http.client
import logging
import socket
import threading
import types
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

_log = logging.getLogger(__name__)


class _Deadline:
    """One fetch attempt's deadline. A socket timeout bounds each receive, not a response a server
    trickles a byte at a time, so when the deadline passes every connection the attempt opened,
    a redirect's included, is shut down, which ends a receive that is waiting."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.passed = False
        self._sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._pass)
        self._timer.daemon = True

    def __enter__(self) -> _Deadline:
        self._timer.start()
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: types.TracebackType | None) -> None:
        self._timer.cancel()

    def watch(self, connection: socket.socket) -> None:
        with self._lock:
            self._sockets.append(connection)
            passed = self.passed
        if passed:
            _shut(connection)

    def _pass(self) -> None:
        with self._lock:
            self.passed = True
            connections = list(self._sockets)
        for connection in connections:
            _shut(connection)


def _shut(connection: socket.socket) -> None:
    # A connection the server already closed has nothing left to end.
    with contextlib.suppress(OSError):
        connection.shutdown(socket.SHUT_RDWR)


class _WatchedHTTP(http.client.HTTPConnection):
    def __init__(self, *arguments, deadline: _Deadline, **options):
        super().__init__(*arguments, **options)
        self.deadline = deadline

    def connect(self) -> None:
        super().connect()
        self.deadline.watch(self.sock)


class _WatchedHTTPS(http.client.HTTPSConnection):
    def __init__(self, *arguments, deadline: _Deadline, **options):
        super().__init__(*arguments, **options)
        self.deadline = deadline

    def connect(self) -> None:
        super().connect()
        self.deadline.watch(self.sock)


class _HTTP(urllib.request.HTTPHandler):
    def __init__(self, deadline: _Deadline):
        super().__init__()
        self.deadline = deadline

    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(functools.partial(_WatchedHTTP, deadline=self.deadline), req)


class _HTTPS(urllib.request.HTTPSHandler):
    def __init__(self, deadline: _Deadline):
        super().__init__()
        self.deadline = deadline

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        # urllib's default context, which HTTPSConnection makes when given none.
        return self.do_open(functools.partial(_WatchedHTTPS, deadline=self.deadline), req)


def _opener(deadline: _Deadline) -> urllib.request.OpenerDirector:
    """urllib's opener, its HTTP and HTTPS connections each watched by `deadline`."""
    return urllib.request.build_opener(_HTTP(deadline), _HTTPS(deadline))


@dataclass(frozen=True)
class Fetch:
    """Fetch the bytes at a record's url into `output`, in one attempt: a url
    that fails now is dropped as transient, for a later epoch or a cache
    refill to fetch again, so a reader waits at most `timeout` on it. A cache
    never freezes a fetch's outcome; only `snapshot(url)` seals outcomes."""

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
        deadline = _Deadline(self.timeout)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dew-data/1"})
            with deadline, _opener(deadline).open(request, timeout=self.timeout) as response:
                payload = response.read(self.max_bytes + 1)
        except urllib.error.HTTPError as error:
            if error.code in (404, 410):
                return Rejected(f"http_{error.code}")
            _log.debug("fetch %s returned HTTP %d", url, error.code)
            return Rejected("transient", permanent=False)
        except ValueError:
            return Rejected("invalid_url")
        except (OSError, http.client.HTTPException) as error:
            _log.debug("fetch %s failed: %s", url, error)
            return Rejected("transient", permanent=False)
        if deadline.passed:
            _log.debug("fetch %s outlasted its %.1f s", url, self.timeout)
            return Rejected("transient", permanent=False)
        if len(payload) > self.max_bytes:
            return Rejected("too_large")
        return {**record, self.output: payload, "content_sha256": hashlib.sha256(payload).hexdigest()}
