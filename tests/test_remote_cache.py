"""The shared layer under JAX's compilation cache (tests/remote_cache.py), against
a stand-in for the dew-xla-cache Worker and against no Worker at all."""

import http.server
import json
import socket
import threading
import time
import urllib.parse
import urllib.request

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


def entry(milliseconds: int) -> bytes:
    """A cache entry as JAX writes one, for a program that took `milliseconds` to compile."""
    from jax._src import compilation_cache

    return compilation_cache.compress_executable(
        compilation_cache.combine_executable_and_time(b"executable", milliseconds))


@pytest.fixture
def worker():
    """The Worker's protocol over a dict: GET, PUT, a listing two keys a page, a 404 on a miss, a
    401 without the token, and Cloudflare's 403 for urllib's own User-Agent, which never reaches
    the Worker. Each request's path is kept."""
    stored, seen = {}, []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _authorized(self):
            seen.append(self.path)
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
            url = urllib.parse.urlparse(self.path)
            if url.path == "/":
                query = urllib.parse.parse_qs(url.query)
                prefix, start = "/" + query["list"][0], int(query.get("cursor", ["0"])[0])
                keys = sorted(key[len(prefix):] for key in stored if key.startswith(prefix))
                body = json.dumps({"keys": keys[start:start + 2],
                                   "cursor": str(start + 2) if start + 2 < len(keys) else None}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)
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
    yield f"http://127.0.0.1:{server.server_address[1]}", stored, seen
    server.shutdown()


def test_a_slow_compilation_one_task_makes_is_read_by_the_next(worker):
    """An entry that took a second or more to compile is uploaded with its
    program's name and a quick one stays local; a later process lists the
    names and asks the remote only about programs of those names."""
    url, stored, seen = worker
    first = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
    first.load_names()
    first.put("jit_step-1", entry(1500))
    first.put("jit_step-2", entry(10))
    first.put("jit_init-3", entry(10))
    first.drain()
    assert sorted(stored) == ["/jax/cpu/jit_step-1", "/jax/cpu/names/jit_step"]

    for name in ("jit_a", "jit_b"):
        stored[f"/jax/cpu/names/{name}"] = b"1"
    second = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
    second.load_names()
    assert second.names == {"jit_step", "jit_a", "jit_b"}  # over two pages
    seen.clear()
    assert second.get("jit_step-1") == entry(1500)
    assert second.base.get("jit_step-1") == entry(1500)  # kept locally, so the next read is local
    assert second.get("jit_step-1") == entry(1500)
    assert second.get("jit_step-2") is None and second.get("jit_init-3") is None
    assert seen == ["/jax/cpu/jit_step-1", "/jax/cpu/jit_step-2"] and second.available


def test_a_wrong_token_leaves_the_remote_alone_and_never_raises(worker):
    url, stored, _ = worker
    cache = remote_cache.RemoteCache(Local(), url, "wrong", "jax/cpu")
    cache.load_names()
    cache.put("jit_step-1", entry(1500))
    cache.drain()
    assert not cache.available and stored == {} and cache.get("jit_step-2") is None
    assert cache.base.get("jit_step-1") == entry(1500)


def test_an_unreachable_remote_only_slows_a_task_by_one_failed_listing(monkeypatch):
    """No Worker at the URL: the listing of names fails, the remote is left
    alone, every read is a local miss and every write stays local, and no call
    raises or asks the URL again."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    opened = []
    urlopen = urllib.request.urlopen
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *args, **kwargs: (opened.append(args), urlopen(*args, **kwargs))[1])
    cache = remote_cache.RemoteCache(Local(), f"http://127.0.0.1:{port}", "token", "jax/cpu")
    began = time.monotonic()
    cache.load_names()
    for index in range(10):
        assert cache.get(f"jit_step-{index}") is None
        cache.put(f"jit_step-{index}", entry(1500))
    cache.drain()
    assert time.monotonic() - began < 10
    assert not cache.available and len(opened) == 1
    assert all(cache.base.get(f"jit_step-{index}") == entry(1500) for index in range(10))


def test_a_process_compiles_through_the_shared_layer_and_the_next_reads_it(worker, tmp_path):
    """With jax as Dew pins it: `install` wraps JAX's own persistent cache, a
    process's compilation reaches the Worker under the version prefix, and a
    process with an empty local cache reads it back instead of compiling."""
    import os
    import subprocess
    import sys

    url, stored, _ = worker
    script = """
import sys
import jax
import remote_cache
from dew.cache import enable_compilation_cache
enable_compilation_cache(sys.argv[1])
remote_cache.REMOTE_MS = 0
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
    assert (value, fetched) == ("18.0", "0") and int(uploaded) == len(stored) // 2 >= 1
    assert all(key.startswith(f"/jax{__import__('jax').__version__}/") for key in stored)
    programs = [key for key in stored if "/names/" not in key]
    assert run("second") == ["18.0", str(len(programs)), "0"]
