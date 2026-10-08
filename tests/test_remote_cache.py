"""The shared layer under JAX's compilation cache (tests/remote_cache.py), against
a stand-in for the dew-xla-cache Worker and against no Worker at all."""

import http.server
import socket
import threading
import time

import pytest
import remote_cache


class Local:
    """A JAX cache of a dict."""

    def __init__(self):
        self.entries = {}

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, value):
        self.entries[key] = value


@pytest.fixture
def worker():
    """The Worker's protocol over a dict: GET, PUT, a 404 on a miss, a 401 without the token,
    and Cloudflare's 403 for urllib's own User-Agent, which never reaches the Worker."""
    stored = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _authorized(self):
            if self.headers.get("User-Agent", "").startswith("Python-urllib"):
                self.send_response(403)
                self.end_headers()
                return False
            if self.headers.get("Authorization") == "Bearer token":
                return True
            self.send_response(401)
            self.end_headers()
            return False

        def do_GET(self):
            if not self._authorized():
                return
            value = stored.get(self.path)
            self.send_response(404 if value is None else 200)
            self.end_headers()
            if value is not None:
                self.wfile.write(value)

        def do_PUT(self):
            if not self._authorized():
                return
            stored[self.path] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(204)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", stored
    server.shutdown()


def test_an_entry_one_task_compiles_is_read_by_the_next(worker):
    url, stored = worker
    first = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
    first.put("program", b"executable")
    first.drain()
    assert stored == {"/jax/cpu/program": b"executable"}

    second = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
    assert second.get("program") == b"executable"
    assert second.base.get("program") == b"executable"  # kept locally, so the next read is local
    assert second.get("elsewhere") is None and second.available


def test_a_wrong_token_is_a_miss_and_never_an_error(worker):
    url, stored = worker
    cache = remote_cache.RemoteCache(Local(), url, "wrong", "jax/cpu")
    cache.put("program", b"executable")
    cache.drain()
    assert stored == {} and cache.get("other") is None
    assert cache.base.get("program") == b"executable"


def test_an_unreachable_remote_only_slows_a_task_and_is_then_left_alone(monkeypatch):
    """No Worker at the URL: every read is a miss, every write stays local, no
    call raises, and after `FAILURES` failed calls in a row the remote is not
    asked again, so an outage costs a few failed connections."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    calls = []
    request = remote_cache.RemoteCache._request
    monkeypatch.setattr(remote_cache.RemoteCache, "_request",
                        lambda self, *args, **kwargs: (calls.append(args), request(self, *args, **kwargs))[1])
    cache = remote_cache.RemoteCache(Local(), f"http://127.0.0.1:{port}", "token", "jax/cpu")
    began = time.monotonic()
    for index in range(10):
        assert cache.get(f"program{index}") is None
        cache.put(f"program{index}", b"executable")
    cache.drain()
    assert time.monotonic() - began < 10
    assert not cache.available
    assert len(calls) <= remote_cache.FAILURES + 2  # uploads already queued when the third failure lands
    assert all(cache.base.get(f"program{index}") == b"executable" for index in range(10))


def test_a_process_compiles_through_the_shared_layer_and_the_next_reads_it(worker, tmp_path):
    """With jax as Dew pins it: `install` wraps JAX's own persistent cache, a
    process's compilation reaches the Worker under the version prefix, and a
    process with an empty local cache reads it back instead of compiling."""
    import os
    import subprocess
    import sys

    url, stored = worker
    script = """
import sys
import jax
import remote_cache
from dew.cache import enable_compilation_cache
enable_compilation_cache(sys.argv[1])
remote = remote_cache.install(sys.argv[2], "token")
assert remote is not None
print(float(jax.jit(lambda x: (x * 3.0).sum())(jax.numpy.arange(4.0))))
remote.drain()
print(remote.fetched, remote.uploaded)
"""
    path = os.pathsep.join([os.path.dirname(__file__), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path}

    def run(local):
        done = subprocess.run([sys.executable, "-c", script, str(tmp_path / local), url], env=env,
                              capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stderr[-2000:]
        return done.stdout.split()

    value, fetched, uploaded = run("first")
    assert (value, fetched) == ("18.0", "0") and int(uploaded) == len(stored) >= 1
    assert all(key.startswith(f"/jax{__import__('jax').__version__}/") for key in stored)
    assert run("second") == ["18.0", uploaded, "0"]
