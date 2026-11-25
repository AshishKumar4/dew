"""tools/armada/r2.py, the R2 Worker's one client, against the stand-in Worker (conftest's cache_worker): an
object round-trips, a miss is None, a listing pages to its end, a key or a body the Worker's contract refuses
never reaches it, and an answer cut short, or arriving too slowly, is no object."""

import contextlib
import http.server
import socket
import threading
import time

import pytest
from remote_cache import r2
from tool_support import load

SENDERS = pytest.mark.parametrize("send", [r2.urllib_send, r2.KeptSend()],
                                  ids=["a connection a request", "kept"])
"""Each sender the Store sends by: the standard library's, a connection a request, and one that keeps a
connection a thread (`KeptSend`)."""


def test_an_object_round_trips_a_miss_is_none_and_a_listing_pages_to_its_end(cache_worker):
    url, _, _ = cache_worker
    store = r2.Store(url, "token", "dew-test", timeout=10)
    for name in ("a", "b", "c"):
        store.put(f"records/{name}", name.encode())
    assert store.get("records/b") == b"b" and store.get("records/d") is None
    assert list(store.keys("records/")) == ["a", "b", "c"]
    with pytest.raises(r2.Refused):
        r2.Store(url, "another", "dew-test", timeout=10).get("records/a")


def test_a_callers_own_sender_carries_every_request_with_the_workers_protocol(cache_worker):
    """A caller's sender (a pooled one) sends what the Store builds: each request with the token and
    the caller's User-Agent, its answers read as the standard library's are."""
    url, _, _ = cache_worker
    sent = []

    def recording(method, address, headers, body, timeout, until=None):
        sent.append((method, headers["Authorization"], headers["User-Agent"]))
        return r2.urllib_send(method, address, headers, body, timeout, until)

    store = r2.Store(url, "token", "dew-pooled", timeout=10, send=recording)
    store.put("records/a", b"a")
    assert store.get("records/a") == b"a" and store.get("records/b") is None
    assert list(store.keys("records/")) == ["a"]
    assert sent == [(method, "Bearer token", "dew-pooled") for method in ("PUT", "GET", "GET", "GET")]


def test_an_answer_with_no_success_status_is_refused():
    """An answer whose final status is neither a success nor a miss (here a 101) is `Refused`, never an
    empty object."""

    class Switching(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(101)
            self.end_headers()
            self.close_connection = True

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Switching)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=10)
        with pytest.raises(r2.Refused):
            store.get("records/switching")
    finally:
        server.shutdown()


def test_what_the_contract_refuses_never_reaches_the_worker(cache_worker):
    url, stored, seen = cache_worker
    store = r2.Store(url, "token", "dew-test", timeout=10)
    for key in ("records/../a", "records/a b", "x" * (r2.CONTRACT["max_key_length"] + 1)):
        with pytest.raises(ValueError):
            store.put(key, b"value")
    for body in (b"", bytes(r2.MAX_BYTES + 1)):
        with pytest.raises(ValueError):
            store.put("records/body", body)
    assert seen == [] and stored == {}


@SENDERS
def test_an_answer_that_ends_before_its_announced_length_is_unavailable_and_a_head_has_no_body(send):
    """A Worker whose answer stops short, the connection closed before the length it announced, gave no
    object: a read of it, whole or at once, is `Unavailable`, which a caller counts as a failure, never a
    shorter object. A HEAD's Content-Length is the object's, and its answer has no body to read."""

    class Short(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()

        def do_GET(self):
            self.do_HEAD()
            self.wfile.write(b"x" * 10)
            self.close_connection = True

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Short)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=10,
                         send=send)
        with pytest.raises(r2.Unavailable):
            store.get("records/short")
        with pytest.raises(r2.Unavailable), store.request("GET", "records/short") as reply:
            reply.body.read()
        with store.request("HEAD", "records/short") as reply:
            assert (reply.status, reply.headers["Content-Length"], reply.body) == (200, "100", None)
    finally:
        server.shutdown()


def served(handler: type[http.server.BaseHTTPRequestHandler]) -> http.server.ThreadingHTTPServer:
    """A server of `handler`'s answers on a port of its own, each connection in a daemon thread."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def trickling(status: int) -> type[http.server.BaseHTTPRequestHandler]:
    """A Worker that answers `status` with a body of 4 KiB it sends 64 bytes every 0.2 s, each within a
    socket's timeout, 12.8 s in all, over HTTP/1.1."""

    class Trickle(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Length", str(64 * 64))
            self.end_headers()
            # Written until the client gives up on it, which closes the connection.
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                for _ in range(64):
                    self.wfile.write(b"x" * 64)
                    self.wfile.flush()
                    time.sleep(0.2)

    return Trickle


@SENDERS
def test_an_answer_arriving_slower_than_its_floor_is_unavailable_within_its_deadline(send):
    """A Worker that sends its answer's body a few bytes at a time, each within the socket's timeout, gave
    no object in time: its read is `Unavailable` at `timeout` plus its length at `MIN_RATE`, each receive
    waiting at most what is left of that, never after the whole trickle's 12.8 s (2026-11-03: a test cache's
    bundle read 28 minutes before its group's first test)."""

    server = served(trickling(200))
    try:
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=1,
                         send=send)
        began = time.monotonic()
        with pytest.raises(r2.Unavailable):
            store.get("records/slow")
        assert time.monotonic() - began < 3
    finally:
        server.shutdown()


def answering(body: bytes, late: str | None = None) -> type[http.server.BaseHTTPRequestHandler]:
    """A Worker that answers every GET 200 with `body`, a path beginning `late` after 0.05 s, else at once."""

    class Answering(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if late is not None and self.path.startswith(late):
                time.sleep(0.05)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Answering


@pytest.mark.parametrize("page", [
    b'{"keys": "abc", "cursor": null}', b'{"keys": ["a"]}', b'{"keys": ["a", 1], "cursor": null}', b'["a"]',
    b'{"keys": [], "cursor": 3}', b"not json"], ids=[
    "keys a string", "no cursor", "a key no string", "no object", "a cursor no string", "no JSON"])
def test_a_listing_page_that_is_no_page_of_keys_is_malformed(page):
    """A listing's page that is no object of a list of string keys and a cursor (a string, or null at its
    end) is `Malformed`, an `R2Error`: never keys read from its characters, nor an empty listing."""
    server = served(answering(page))
    try:
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=5)
        with pytest.raises(r2.Malformed):
            list(store.keys("records/"))
    finally:
        server.shutdown()


class HeadersTrickle(http.server.BaseHTTPRequestHandler):
    """A Worker that sends its answer's headers a byte every 0.2 s, each receive within a socket's timeout."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        head = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n" + b"X-Slow: " + b"y" * 64 + b"\r\n\r\nok"
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            for byte in head:
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(0.2)


class ChunkTrickle(http.server.BaseHTTPRequestHandler):
    """A Worker that answers at once, chunked and closing its connection after, then sends its first chunk's
    size line a byte every 0.2 s."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            for byte in b"00000000000000000000000000000002\r\nok\r\n0\r\n\r\n":
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(0.2)


@pytest.mark.parametrize("worker", [HeadersTrickle, ChunkTrickle], ids=["headers", "chunks"])
@SENDERS
def test_an_answer_that_trickles_ends_at_its_callers_end(send, worker):
    """A Worker that sends its headers a byte at a time, or its headers at once (chunked and detached from its
    connection by `Connection: close`) then a chunk's size line a byte at a time, each within the socket's
    timeout, holds a request given `until` no longer than until it, whichever sender carries it."""
    server = served(worker)
    try:
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=5,
                         send=send)
        began = time.monotonic()
        with pytest.raises(r2.R2Error):
            store.get("records/slow", until=began + 1)
        assert time.monotonic() - began < 2
    finally:
        server.shutdown()


@SENDERS
def test_a_connect_that_stalls_ends_at_its_callers_end_and_closes_what_it_makes_late(send, monkeypatch):
    """A connect whose handshake stalls (held here until its request has ended) ends at the caller's end, not
    the socket's timeout; the connection it makes late is closed by the connect left to finish, never left
    open."""
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(5)
    released = threading.Event()
    connected = socket.create_connection

    def stalled(address, *args, **kwargs):
        released.wait(5)
        return connected(listener.getsockname(), *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", stalled)
    try:
        store = r2.Store("http://127.0.0.1:9", "token", "dew-test", timeout=10, send=send)
        began = time.monotonic()
        with pytest.raises(r2.R2Error):
            store.get("records/a", until=began + 0.5)
        assert time.monotonic() - began < 1.5
        released.set()
        accepted, _ = listener.accept()
        with accepted:
            accepted.settimeout(5)
            assert accepted.recv(1) == b""
    finally:
        released.set()
        listener.close()


def test_a_request_at_its_deadline_leaves_no_shutdown_for_the_next_on_a_kept_connection():
    """A request answered right at its caller's end, its guard firing as it closes, leaves its kept connection
    to the next request only where it was disarmed first: the next request, sent at once, is answered whole
    every time, never shut down by a guard that fired after its connection was kept."""
    server = served(answering(b"ok", late="/records/late"))
    try:
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=5,
                         send=r2.KeptSend())
        for _ in range(40):
            with contextlib.suppress(r2.R2Error):
                store.get("records/late", until=time.monotonic() + 0.05)
            assert store.get("records/next", until=time.monotonic() + 5) == b"ok"
    finally:
        server.shutdown()


def test_a_refusals_unread_body_is_drained_within_its_callers_end():
    """A refusal whose body trickles is read off for the kept connection's next request at most until the
    caller's end, never the whole drain past it."""
    server = served(trickling(403))
    try:
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=5,
                         send=r2.KeptSend())
        began = time.monotonic()
        with pytest.raises(r2.Refused):
            store.get("records/refused", until=began + 0.3)
        assert time.monotonic() - began < 0.3 + 0.5
    finally:
        server.shutdown()


def test_the_records_one_thread_reads_in_turn_share_its_connections_to_the_worker(cache_worker, monkeypatch):
    """A thread's record lookups (tools/armada/ci.py's `recorded`, as the plan's and a task's run) go on the
    connection it keeps to the Worker (`KeptSend`), not one a lookup, each paying a TCP and a TLS handshake:
    ten records read in turn open four connections to the Worker's stand-in, which ends each after three
    requests without a word, as a server ends an idle one, and the lookup sent on one it ended goes again on
    a new one, every record read whole."""
    url, stored, seen = cache_worker
    monkeypatch.setenv("DEW_XLA_CACHE_URL", url)
    ci = load("ci", "tools/armada")
    for index in range(10):
        stored[f"/records/{index}"] = f'{{"index": {index}}}'.encode()
    assert [ci.recorded(f"records/{index}", "token") for index in range(10)] == [
        {"index": index} for index in range(10)]
    assert len(seen) == 10 and len(seen.connections) == 4


def test_answers_with_no_body_to_read_leave_their_kept_connection_for_the_next_request(cache_worker):
    """A PUT's 204, which the Store never reads, ends its answer at once, so the connection `KeptSend` keeps
    carries the next request: two writes and a read go on one connection to the Worker's stand-in."""
    url, _, seen = cache_worker
    store = r2.Store(url, "token", "dew-test", timeout=10, send=r2.KeptSend())
    store.put("records/a", b"a")
    store.put("records/b", b"b")
    assert store.get("records/a") == b"a" and len(seen.connections) == 1


def test_a_kept_connection_the_worker_closes_after_an_answer_reads_that_answer_whole():
    """An answer the Worker says it closes its connection after (Connection: close) is read whole on a
    connection `KeptSend` keeps, which then opens a new one for the next request."""

    class Closing(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            body = self.path.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

    server = served(Closing)
    try:
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=10,
                         send=r2.KeptSend())
        assert [store.get(f"records/{name}") for name in ("a", "b")] == [b"/records/a", b"/records/b"]
    finally:
        server.shutdown()


def test_a_miss_whose_body_trickles_is_none_within_the_kept_connections_drain():
    """A miss (a 404) whose few bytes of body the Worker sends slowly is None as soon as it is answered: a
    kept connection reads such a body off for its next request at most `DRAIN_SECONDS`, then closes it,
    never after the whole trickle's 12.8 s."""

    server = served(trickling(404))
    try:
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=10,
                         send=r2.KeptSend())
        began = time.monotonic()
        assert store.get("records/missing") is None
        assert time.monotonic() - began < r2.DRAIN_SECONDS + 2
    finally:
        server.shutdown()
