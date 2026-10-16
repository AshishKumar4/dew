"""The shared layer under JAX's compilation cache (tests/remote_cache.py), against
a stand-in for the dew-xla-cache Worker and against no Worker at all."""

import http.server
import socket
import threading
import time
import urllib.parse
import urllib.request
from typing import ClassVar

import remote_cache


class Local:
    """A JAX cache of a dict."""

    def __init__(self):
        self.entries = {}

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, value):
        self.entries[key] = value


def entry(seconds: int) -> bytes:
    """A cache entry as JAX writes one, for a program that took `seconds` to compile."""
    from jax._src import compilation_cache

    return compilation_cache.compress_executable(
        compilation_cache.combine_executable_and_time(b"executable", seconds))


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
        cache.put(f"jit_step-{index}", entry(2))
    cache.drain()
    assert time.monotonic() - began < 10
    assert not cache.available and len(opened) == 1
    assert all(cache.base.get(f"jit_step-{index}") == entry(2) for index in range(10))


def test_a_process_compiles_through_the_shared_layer_and_the_next_reads_it(cache_worker, tmp_path):
    """With jax as Dew pins it: `install` wraps JAX's own persistent cache, a
    process's compilation reaches the Worker under the version prefix, and a
    process with an empty local cache reads it back instead of compiling."""
    import os
    import subprocess
    import sys

    url, stored, _ = cache_worker
    script = """
import sys
import jax
import remote_cache
from dew.cache import enable_compilation_cache
enable_compilation_cache(sys.argv[1])
remote_cache.REMOTE_SECONDS = 0
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


def test_a_slow_remote_never_holds_a_process_exit():
    """Uploads queued behind a remote that takes a minute a write do not keep
    the process from exiting: they run on daemon threads, and only the drain
    at the end of pytest's session waits for them, up to its bound."""
    import os
    import subprocess
    import sys

    class Slow(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_PUT(self):
            time.sleep(60)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    script = f"""
import remote_cache
from jax._src import compilation_cache
remote_cache.TIMEOUT_SECONDS = 2
class Local(dict):
    def put(self, key, value):
        self[key] = value
remote = remote_cache.RemoteCache(Local(), "http://127.0.0.1:{server.server_address[1]}", "token", "jax/cpu")
for index in range(20):
    remote.put(f"jit_step-{{index}}", compilation_cache.compress_executable(
        compilation_cache.combine_executable_and_time(b"executable", 5)))
"""
    path = os.pathsep.join([os.path.dirname(__file__), os.environ.get("PYTHONPATH", "")])
    began = time.monotonic()
    done = subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": path},
                          capture_output=True, text=True, timeout=120)
    server.shutdown()
    assert done.returncode == 0, done.stderr[-2000:]
    assert time.monotonic() - began < 15


class Counted(http.server.BaseHTTPRequestHandler):
    """A remote that fails every request at once, but answers a GET of a `slow` program after a
    second; it counts the requests that reach it."""

    requests: ClassVar[list[str]] = []

    def log_message(self, *args):
        pass

    def answer(self):
        Counted.requests.append(f"{self.command} {self.path}")
        if self.command == "PUT":
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.command == "GET" and "/slow-" in self.path:
            time.sleep(1)
            body = entry(2)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_PUT = answer


def counted() -> tuple[http.server.ThreadingHTTPServer, str]:
    Counted.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Counted)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_a_remote_given_up_sends_none_of_the_uploads_still_queued():
    """Twenty uploads queued behind a remote that fails each: once `FAILURES`
    have failed, the ones still queued are dropped unsent, beyond those the
    two upload threads had already begun."""
    server, url = counted()
    cache = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
    for index in range(20):
        cache.put(f"jit_step-{index}", entry(2))
    cache.drain()
    server.shutdown()
    assert not cache.available
    assert remote_cache.FAILURES <= len(Counted.requests) <= remote_cache.FAILURES + 1
    assert all(cache.base.get(f"jit_step-{index}") == entry(2) for index in range(20))


def test_an_answer_that_arrives_after_the_remote_was_given_up_does_not_take_it_back():
    """A fetch under way while other requests fail the remote: its answer is
    kept, but the remote stays given up, and no request reaches it after."""
    server, url = counted()
    cache = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
    cache.names = {"slow", "fail"}
    fetched = []
    slow = threading.Thread(target=lambda: fetched.append(cache.get("slow-1")))
    slow.start()
    while not Counted.requests:
        time.sleep(0.01)
    for index in range(remote_cache.FAILURES):
        assert cache.get(f"fail-{index}") is None
    slow.join()
    sent = len(Counted.requests)
    assert cache.get("fail-late") is None
    cache.put("fail-late", entry(2))
    cache.drain()
    server.shutdown()
    assert fetched == [entry(2)] and not cache.available and len(Counted.requests) == sent
