"""The shared layer under JAX's compilation cache (tests/remote_cache.py), each test file's
programs one bundle, against a stand-in for the dew-xla-cache Worker and against no Worker at all."""

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


def entry(name: str) -> bytes:
    """A cache entry as JAX writes one."""
    from jax._src import compilation_cache

    return compilation_cache.compress_executable(
        compilation_cache.combine_executable_and_time(name.encode(), 2))


def test_an_unreachable_remote_only_slows_a_task_by_its_first_failed_requests(monkeypatch):
    """No Worker at the URL: each file's bundle fails to arrive or leave until
    `FAILURES` requests have, the remote is then left alone, every read is a
    local miss and every write stays local, and no call raises or asks the URL
    again."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    opened = []
    urlopen = urllib.request.urlopen
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *args, **kwargs: (opened.append(args), urlopen(*args, **kwargs))[1])
    cache = remote_cache.RemoteCache(Local(), f"http://127.0.0.1:{port}", "token", "jax/cpu")
    began = time.monotonic()
    for index in range(10):
        cache.opened(f"tests/test_{index}.py")
        assert cache.get(f"jit_step-{index}") is None
        cache.put(f"jit_step-{index}", entry("step"))
        cache.closed(f"tests/test_{index}.py")
    cache.drain()
    assert time.monotonic() - began < 10
    # An upload already under way as the remote was given up may be one request more.
    assert not cache.available and remote_cache.FAILURES <= len(opened) <= remote_cache.FAILURES + 1
    assert all(cache.base.get(f"jit_step-{index}") == entry("step") for index in range(10))


def test_a_files_programs_reach_the_next_process_in_one_bundle(cache_worker, tmp_path):
    """With jax as Dew pins it: `install` wraps JAX's own persistent cache, a
    file's compilations reach the Worker as one bundle under the version and
    device prefix, and a process with an empty local cache reads them back
    instead of compiling, its programs all held, so it writes nothing back. A
    program the bundle lacked is written back with the rest."""
    import os
    import subprocess
    import sys

    url, stored, _ = cache_worker
    script = """
import sys
import jax
import remote_cache
from dew.cache import enable_compilation_cache
from dew.telemetry.instrumentation import compilation_cache_counts, count_compilation_cache
enable_compilation_cache(sys.argv[1])
remote = remote_cache.install(sys.argv[2], "token")
assert remote is not None
count_compilation_cache()
remote.opened("tests/test_step.py")
print(float(jax.jit(lambda x: (x * 3.0).sum())(jax.numpy.arange(4.0))))
if sys.argv[3] == "more":
    jax.jit(lambda x: x - 1.0)(1.0)
remote.drain()
counts = compilation_cache_counts()
print(remote.fetched, remote.uploaded, int(counts.get("train/cache_misses", 0)))
"""
    path = os.pathsep.join([os.path.dirname(__file__), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path}

    def run(local, more="-"):
        done = subprocess.run([sys.executable, "-c", script, str(tmp_path / local), url, more],
                              env=env, capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stderr[-2000:]
        return done.stdout.split()

    value, fetched, uploaded, compiled = run("first")
    assert (value, fetched, uploaded) == ("18.0", "0", "1") and int(compiled) >= 1
    [key] = stored
    assert key.startswith(f"/jax{__import__('jax').__version__}/")
    assert key.endswith("/cpu/bundles/tests/test_step.py")
    programs = len(list(remote_cache.unpacked(stored[key])))
    assert programs >= 1
    # Read back, nothing compiles: JAX's persistent cache misses nothing.
    assert run("second") == ["18.0", str(programs), "0", "0"]
    assert run("third", "more") == ["18.0", str(programs), "1", "1"]
    assert len(list(remote_cache.unpacked(stored[key]))) > programs


def test_a_slow_remote_never_holds_a_process_exit():
    """A bundle queued behind a remote that takes a minute a write does not
    keep the process from exiting: it uploads on a daemon thread, and only the
    drain at the end of pytest's session waits for it, up to its bound."""
    import os
    import subprocess
    import sys

    class Slow(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_PUT(self):
            time.sleep(60)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    script = f"""
import remote_cache
remote_cache.TIMEOUT_SECONDS = 2
class Local(dict):
    def put(self, key, value):
        self[key] = value
remote = remote_cache.RemoteCache(Local(), "http://127.0.0.1:{server.server_address[1]}", "token", "jax/cpu")
for index in range(20):
    remote.opened(f"tests/test_{{index}}.py")
    remote.put(f"jit_step-{{index}}", b"executable")
    remote.closed(f"tests/test_{{index}}.py")
"""
    path = os.pathsep.join([os.path.dirname(__file__), os.environ.get("PYTHONPATH", "")])
    began = time.monotonic()
    done = subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": path},
                          capture_output=True, text=True, timeout=120)
    server.shutdown()
    assert done.returncode == 0, done.stderr[-2000:]
    assert time.monotonic() - began < 30


class Counted(http.server.BaseHTTPRequestHandler):
    """A remote that fails every request at once, but answers a GET of a `slow` file's bundle once
    `released` is set; it counts the requests that reach it."""

    requests: ClassVar[list[str]] = []
    released: ClassVar[threading.Event] = threading.Event()

    def log_message(self, *args):
        pass

    def answer(self):
        Counted.requests.append(f"{self.command} {self.path}")
        if self.command == "PUT":
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.command == "GET" and "/slow" in self.path:
            Counted.released.wait(60)
            body = remote_cache.packed([("jit_slow-1", entry("slow"))])
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
    Counted.released = threading.Event()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Counted)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_a_remote_given_up_sends_none_of_the_uploads_still_queued():
    """Twenty files' bundles queued behind a remote that fails each: once
    `FAILURES` requests have failed, the bundles still queued are dropped unsent,
    and each file's programs stay in the local cache."""
    server, url = counted()
    cache = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
    for index in range(20):
        cache.put(f"jit_step-{index}", entry("step"))
    cache.used = {f"tests/test_{index}.py": {f"jit_step-{index}"} for index in range(20)}
    for index in range(20):
        cache.closed(f"tests/test_{index}.py")
    cache.drain()
    server.shutdown()
    assert not cache.available
    assert remote_cache.FAILURES <= len(Counted.requests) <= remote_cache.FAILURES + 1
    assert all(cache.base.get(f"jit_step-{index}") == entry("step") for index in range(20))


def test_an_answer_that_arrives_after_the_remote_was_given_up_does_not_take_it_back():
    """A bundle's fetch under way while other requests fail the remote: its
    programs are kept, but the remote stays given up, and no request reaches it
    after."""
    server, url = counted()
    cache = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
    slow = threading.Thread(target=cache.opened, args=("tests/slow.py",))
    slow.start()
    try:
        while not Counted.requests:
            time.sleep(0.01)
        for index in range(remote_cache.FAILURES):
            cache.opened(f"tests/fail_{index}.py")
        assert not cache.available
    finally:
        # The slow answer arrives only once the remote was given up.
        Counted.released.set()
    slow.join()
    sent = len(Counted.requests)
    cache.opened("tests/late.py")
    cache.put("jit_late-1", entry("late"))
    cache.closed("tests/late.py")
    cache.drain()
    server.shutdown()
    assert cache.base.get("jit_slow-1") == entry("slow") and not cache.available
    assert len(Counted.requests) == sent


def test_a_bundle_holds_every_program_its_files_tests_used_however_they_ran(cache_worker):
    """A file whose tests ran apart, another file's between them, writes back every program its tests
    used, not the last run's alone; a split file's groups each keep a bundle of their own, which no
    other group's replaces; and a bundle the Worker answers cut short or past its bound is a miss
    that puts nothing in the local cache."""
    url, stored, _ = cache_worker
    cache = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
    for file, program in (("tests/a.py", "first"), ("tests/b.py", "other"), ("tests/a.py", "second")):
        cache.opened(file)
        cache.put(f"jit_{program}-1", entry(program))
        cache.closed(file)
    cache.drain()
    [key] = [key for key in stored if key.endswith("/bundles/tests/a.py")]
    assert dict(remote_cache.unpacked(stored[key])) == {"jit_first-1": entry("first"),
                                                         "jit_second-1": entry("second")}
    for group in ("1/2", "2/2"):
        cache.opened(remote_cache.bundle("tests/split.py::test_x", group, None))
        cache.put(f"jit_{group[0]}-1", entry(group))
        cache.closed(remote_cache.bundle("tests/split.py::test_x", group, None))
    cache.drain()
    assert [len(remote_cache.unpacked(stored[f"/jax/cpu/bundles/tests/split.py/{group}-of-2"]))
            for group in "12"] == [1, 1]
    whole = remote_cache.packed([("jit_cut-1", entry("cut")), ("jit_cut-2", entry("cut"))])
    for name, body in (("cut", whole[:-3]), ("header", whole + b"\0\0")):
        stored[f"/jax/cpu/bundles/tests/{name}.py"] = body
        fresh = remote_cache.RemoteCache(Local(), url, "token", "jax/cpu")
        fresh.opened(f"tests/{name}.py")
        assert fresh.base.entries == {} and fresh.fetched == 0 and fresh.failures == 1
