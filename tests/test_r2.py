"""tools/armada/r2.py, the R2 Worker's one client, against the stand-in Worker (conftest's cache_worker): an
object round-trips, a miss is None, a listing pages to its end, a key or a body the Worker's contract refuses
never reaches it, and an answer cut short is no object."""

import http.server
import threading

import pytest
from remote_cache import r2


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
    """A caller's sender (dew-ops' pooled one) sends what the Store builds: each request with the token and
    the caller's User-Agent, its answers read as the standard library's are."""
    url, _, _ = cache_worker
    sent = []

    def recording(method, address, headers, body, timeout):
        sent.append((method, headers["Authorization"], headers["User-Agent"]))
        return r2.urllib_send(method, address, headers, body, timeout)

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


def test_an_answer_that_ends_before_its_announced_length_is_unavailable_and_a_head_has_no_body():
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
        store = r2.Store(f"http://127.0.0.1:{server.server_address[1]}", "token", "dew-test", timeout=10)
        with pytest.raises(r2.Unavailable):
            store.get("records/short")
        with pytest.raises(r2.Unavailable), store.request("GET", "records/short") as reply:
            reply.body.read()
        with store.request("HEAD", "records/short") as reply:
            assert (reply.status, reply.headers["Content-Length"], reply.body) == (200, "100", None)
    finally:
        server.shutdown()
