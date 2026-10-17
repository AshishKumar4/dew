"""URL catalogs and volatile fetches, executed by the ordinary Grain worker path."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import http.client
import logging
import socket
import threading
import time
import types
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

_log = logging.getLogger(__name__)


class _Deadline:
    """One fetch attempt's deadline, over every connection it opens (a redirect's included) once
    its host is resolved: connecting, a proxy's tunnel, the TLS handshake and the response. A
    socket timeout bounds each receive, not a response a server trickles a byte at a time, so when
    the deadline passes each connection is shut down, which ends a wait in any of those stages.
    The deadline shuts its own duplicate of each socket, which no close by urllib can hand to
    another connection, and closes them only once its timer can no longer run."""

    def __init__(self, seconds: float):
        self.at = time.monotonic() + seconds
        self._passed = False
        self._finished: float | None = None
        self._handles: list[socket.socket] = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._pass)
        self._timer.daemon = True

    def __enter__(self) -> _Deadline:
        self._timer.start()
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: types.TracebackType | None) -> None:
        self._timer.cancel()
        with self._lock:
            self._finished = time.monotonic()
            handles, self._handles = self._handles, []
        for handle in handles:
            handle.close()

    @property
    def passed(self) -> bool:
        """Whether the attempt outlasted its deadline: the timer shut it, or it ended late."""
        return self._passed or (self._finished is not None and self._finished >= self.at)

    def connection(self, address: tuple[str, int], timeout: float | None = None,
                   source_address: tuple[str, int] | None = None) -> socket.socket:
        """`socket.create_connection` under the deadline: each socket watched from before it
        connects, and each wait at most the time the deadline leaves, in place of `timeout`."""
        host, port = address
        failure: OSError | None = None
        for family, kind, protocol, _, target in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM):
            left = self.at - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no time left to connect to {host}")
            connection = socket.socket(family, kind, protocol)
            self._watch(connection)
            try:
                connection.settimeout(left)
                if source_address is not None:
                    connection.bind(source_address)
                connection.connect(target)
            except OSError as error:
                connection.close()
                failure = error
                continue
            return connection
        raise failure if failure is not None else OSError(f"{host} resolves to no address")

    def _watch(self, connection: socket.socket) -> None:
        with self._lock:
            self._handles.append(connection.dup())
            if self._passed:
                _shut(self._handles[-1])

    def _pass(self) -> None:
        with self._lock:
            if self._finished is not None:
                return
            self._passed = True
            for handle in self._handles:
                _shut(handle)


def _shut(connection: socket.socket) -> None:
    # A connection the server already closed has nothing left to end.
    with contextlib.suppress(OSError):
        connection.shutdown(socket.SHUT_RDWR)


class _WatchedHTTP(http.client.HTTPConnection):
    def __init__(self, *arguments, deadline: _Deadline, **options):
        super().__init__(*arguments, **options)
        # http.client's hook for how a connection opens its socket.
        self._create_connection = deadline.connection


class _WatchedHTTPS(http.client.HTTPSConnection):
    def __init__(self, *arguments, deadline: _Deadline, **options):
        super().__init__(*arguments, **options)
        self._create_connection = deadline.connection


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
    refill to fetch again, so a reader waits at most `timeout` on it from
    connecting to its last byte, however slowly a server sends. Resolving its
    host is the system resolver's, which takes no timeout, as in urllib3 and
    requests. A cache never freezes a fetch's outcome; only `snapshot(url)`
    seals outcomes."""

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
