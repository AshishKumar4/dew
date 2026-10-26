"""The shared layer under JAX's compilation cache (tests/remote_cache.py), each test file's
programs one bundle, against a stand-in for the dew-xla-cache Worker and against no Worker at all.
JAX's cache directory is the boundary: a program JAX compiles is a `<key>-cache` file there, which
`compiled` writes as JAX does where a test needs no real compilation."""

import http.server
import json
import socket
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import remote_cache


def compiled(directory: Path, key: str, value: bytes = b"executable") -> None:
    """Program `key` in JAX's cache `directory`, as JAX writes one."""
    (directory / f"{key}{remote_cache.SUFFIX}").write_bytes(value)


def programs(directory: Path) -> dict[str, bytes]:
    return {path.name.removesuffix(remote_cache.SUFFIX): path.read_bytes()
            for path in directory.glob(f"*{remote_cache.SUFFIX}")}


def test_a_bundle_is_its_host_class_s_alone_on_every_backend():
    """A bundle is its host class's (`remote_cache.HOST.cpu_class`) whatever the backend: XLA:CPU
    compiles for the host that compiles, and a GPU lane's programs hold CPU executables too (its
    host-side computations), so a program from another class need not link here ("Failed to
    materialize symbols"), after which the process compiles nothing more. On the CPU, whose device kind
    is "cpu" on every host, and on a GPU lane's backend alike."""
    import os
    import subprocess
    import sys

    script = """
import remote_cache
for named in ("AuthenticAMD-25-17", "AuthenticAMD-25-1"):
    remote_cache.HOST.cpu_class = lambda named=named: named
    print(remote_cache.prefix())
"""
    path = os.pathsep.join([os.path.dirname(__file__), os.environ.get("PYTHONPATH", "")])
    for platforms in ({"JAX_PLATFORMS": "cpu"}, {}):
        env = {**os.environ, "PYTHONPATH": path, **platforms}
        done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                              timeout=300)
        assert done.returncode == 0, done.stderr[-2000:]
        zen4, zen3 = done.stdout.split()
        assert zen4.endswith("/AuthenticAMD-25-17") and zen3.endswith("/AuthenticAMD-25-1"), (platforms, zen4)


def test_an_unreachable_remote_only_slows_a_task_by_its_first_failed_requests(monkeypatch, tmp_path):
    """No Worker at the URL: each file's bundle fails to arrive or leave until
    `FAILURES` requests have, the remote is then left alone, every program stays
    in the local cache, and no call raises or asks the URL again."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    opened = []
    open_ = urllib.request.OpenerDirector.open
    monkeypatch.setattr(urllib.request.OpenerDirector, "open",
                        lambda self, *args, **kwargs: (opened.append(args), open_(self, *args, **kwargs))[1])
    cache = remote_cache.RemoteCache(tmp_path, f"http://127.0.0.1:{port}", "token", "jax/cpu")
    began = time.monotonic()
    for index in range(10):
        cache.opened(f"tests/test_{index}.py")
        compiled(tmp_path, f"jit_step-{index}")
        cache.closed(f"tests/test_{index}.py")
    cache.drain()
    assert time.monotonic() - began < 10
    # An upload already under way as the remote was given up may be one request more.
    assert not cache.available and remote_cache.FAILURES <= len(opened) <= remote_cache.FAILURES + 1
    assert set(programs(tmp_path)) == {f"jit_step-{index}" for index in range(10)}


def test_a_cache_directory_that_cannot_be_written_only_slows_a_task(cache_worker, tmp_path):
    """A bundle that cannot be written into the local cache (here the cache's path is a file) is a
    miss counted against the remote, as a request that fails is: it raises nothing."""
    url, stored, _ = cache_worker
    blocked = tmp_path / "file"
    blocked.write_text("")
    cache = remote_cache.RemoteCache(blocked, url, "token", "jax/cpu")
    for index in range(remote_cache.FAILURES):
        stored[f"/jax/cpu/bundles/tests/test_{index}.py"] = remote_cache.packed([(f"jit_{index}-1", b"x")])
        cache.opened(f"tests/test_{index}.py")
    assert not cache.available and cache.programs_fetched == 0


def test_a_files_programs_reach_the_next_process_in_one_bundle(cache_worker, tmp_path):
    """With jax as Dew pins it: `install` keeps JAX's own cache directory, a
    file's compilations reach the Worker as one bundle under the version and
    device prefix, and a process with an empty local cache reads them back
    instead of compiling, its programs all held, so it writes nothing back,
    under another release of jax with the same jaxlib too (a post-release of
    Dew's patch stack). A program the bundle lacked is written back with the
    rest."""
    import os
    import subprocess
    import sys

    url, stored, _ = cache_worker
    script = """
import sys
import jax
if sys.argv[4] == "bumped":
    jax.__version__ += ".post999"
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
print(remote.programs_fetched, remote.uploaded, int(counts.get("train/cache_misses", 0)))
"""
    path = os.pathsep.join([os.path.dirname(__file__), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path}

    def run(local, more="-", release="pinned"):
        done = subprocess.run([sys.executable, "-c", script, str(tmp_path / local), url, more, release],
                              env=env, capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stderr[-2000:]
        return done.stdout.split()

    value, fetched, uploaded, compiled = run("first")
    assert (value, fetched, uploaded) == ("18.0", "0", "1") and int(compiled) >= 1
    [key] = stored
    assert key.startswith(f"/jaxlib{__import__('jaxlib').__version__}/")
    assert key.endswith(f"/cpu/{remote_cache.HOST.cpu_class()}/bundles/tests/test_step.py")
    programs = len(list(remote_cache.unpacked(stored[key])))
    assert programs >= 1
    # Read back, nothing compiles: JAX's persistent cache misses nothing.
    assert run("second") == ["18.0", str(programs), "0", "0"]
    assert run("bumped", release="bumped") == ["18.0", str(programs), "0", "0"]
    assert run("third", "more") == ["18.0", str(programs), "1", "1"]
    assert len(list(remote_cache.unpacked(stored[key]))) > programs


def test_a_slow_remote_never_holds_a_process_exit(tmp_path):
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
import pathlib
import sys
import remote_cache
remote_cache.TIMEOUT_SECONDS = 2
directory = pathlib.Path(sys.argv[1])
url = "http://127.0.0.1:{server.server_address[1]}"
remote = remote_cache.RemoteCache(directory, url, "token", "jax/cpu")
for index in range(20):
    remote.opened(f"tests/test_{{index}}.py")
    (directory / f"jit_step-{{index}}-cache").write_bytes(b"executable")
    remote.closed(f"tests/test_{{index}}.py")
"""
    path = os.pathsep.join([os.path.dirname(__file__), os.environ.get("PYTHONPATH", "")])
    began = time.monotonic()
    done = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": path}, timeout=120)
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
            body = remote_cache.packed([("jit_slow-1", b"slow")])
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


class Refusing(http.server.BaseHTTPRequestHandler):
    """A remote that has no bundle and refuses every upload; it counts the uploads that reach it."""

    uploads: ClassVar[list[str]] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_PUT(self):
        Refusing.uploads.append(self.path)
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()


def test_a_remote_given_up_sends_none_of_the_uploads_still_queued(tmp_path):
    """Twenty files' bundles queued behind a remote that refuses each: once
    `FAILURES` uploads have failed, the bundles still queued are dropped unsent,
    and each file's programs stay in the local cache."""
    Refusing.uploads = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Refusing)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    cache = remote_cache.RemoteCache(tmp_path, url, "token", "jax/cpu")
    for index in range(20):
        cache.opened(f"tests/test_{index}.py")
        compiled(tmp_path, f"jit_step-{index}")
        cache.closed(f"tests/test_{index}.py")
    cache.drain()
    server.shutdown()
    assert not cache.available
    assert remote_cache.FAILURES <= len(Refusing.uploads) <= remote_cache.FAILURES + 1
    assert set(programs(tmp_path)) == {f"jit_step-{index}" for index in range(20)}


def test_an_answer_that_arrives_after_the_remote_was_given_up_does_not_take_it_back(tmp_path):
    """A bundle's fetch under way while other requests fail the remote: its
    programs are kept, but the remote stays given up, and no request reaches it
    after."""
    server, url = counted()
    cache = remote_cache.RemoteCache(tmp_path, url, "token", "jax/cpu")
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
    compiled(tmp_path, "jit_late-1")
    cache.closed("tests/late.py")
    cache.drain()
    server.shutdown()
    assert programs(tmp_path)["jit_slow-1"] == b"slow" and not cache.available
    assert len(Counted.requests) == sent


def test_a_bundle_holds_every_program_its_files_tests_compiled_however_they_ran(cache_worker, tmp_path):
    """A file whose tests ran apart, another file's between them, writes back every program its tests
    compiled, not the last run's alone, and a program compiled as a file is collected joins its bundle;
    a split file's groups each keep a bundle of their own, which no other group's replaces; and a
    bundle the Worker answers cut short, past its bound or with a key that is no file name is a miss
    that puts nothing in the local cache."""
    url, stored, _ = cache_worker
    cache = remote_cache.RemoteCache(tmp_path, url, "token", "jax/cpu")
    cache.opened("tests/a.py")
    compiled(tmp_path, "jit_collected-1")
    cache.opened("tests/b.py")
    for file, program in (("tests/a.py", "first"), ("tests/b.py", "other"), ("tests/a.py", "second")):
        cache.opened(file)
        compiled(tmp_path, f"jit_{program}-1")
        cache.closed(file)
    cache.drain()
    [key] = [key for key in stored if key.endswith("/bundles/tests/a.py")]
    assert dict(remote_cache.unpacked(stored[key])) == dict.fromkeys(
        ("jit_collected-1", "jit_first-1", "jit_second-1"), b"executable")
    for group in ("1/2", "2/2"):
        cache.opened(remote_cache.bundle("tests/split.py::test_x", group, None))
        compiled(tmp_path, f"jit_{group[0]}-1")
        cache.closed(remote_cache.bundle("tests/split.py::test_x", group, None))
    cache.drain()
    assert [len(remote_cache.unpacked(stored[f"/jax/cpu/bundles/tests/split.py/{group}-of-2"]))
            for group in "12"] == [1, 1]
    whole = remote_cache.packed([("jit_cut-1", b"cut"), ("jit_cut-2", b"cut")])
    for name, body in (("cut", whole[:-3]), ("header", whole + b"\0\0"),
                       ("escape", remote_cache.packed([("../jit_escape-1", b"x")]))):
        stored[f"/jax/cpu/bundles/tests/{name}.py"] = body
        fresh = remote_cache.RemoteCache(tmp_path / name, url, "token", "jax/cpu")
        (tmp_path / name).mkdir()
        fresh.opened(f"tests/{name}.py")
        assert programs(tmp_path / name) == {} and fresh.programs_fetched == 0 and fresh.failures == 1
    assert not (tmp_path / "jit_escape-1-cache").exists()

def run_child(script: str, *arguments: str) -> list[str]:
    """`script` run in a fresh Python with the tests' directory on its path and JAX on the CPU: its
    printed words."""
    import os
    import subprocess
    import sys

    path = os.pathsep.join([os.path.dirname(__file__), os.environ.get("PYTHONPATH", "")])
    done = subprocess.run([sys.executable, "-c", script, *arguments], capture_output=True, text=True,
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path}, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout.split()


def test_jaxs_own_cache_reset_works_under_the_shared_cache(cache_worker, tmp_path):
    """JAX's public `reset_cache` (jax.experimental.compilation_cache), which a run calls to point the
    cache elsewhere, works with the shared cache installed, and the program compiled before it is
    read back from the local cache after it."""
    url, _, _ = cache_worker
    script = """
import sys
import jax
from jax.experimental.compilation_cache import compilation_cache
import remote_cache
from dew.cache import enable_compilation_cache
from dew.telemetry.instrumentation import compilation_cache_counts, count_compilation_cache
enable_compilation_cache(sys.argv[1])
remote = remote_cache.install(sys.argv[2], "token")
remote.opened("tests/test_reset.py")
step = lambda x: (x * 3.0).sum()
jax.jit(step)(jax.numpy.arange(4.0))
compilation_cache.reset_cache()
jax.clear_caches()
count_compilation_cache()
value = float(jax.jit(step)(jax.numpy.arange(4.0)))
print(value, int(compilation_cache_counts().get("train/cache_misses", 0)))
"""
    assert run_child(script, str(tmp_path / "cache"), url) == ["18.0", "0"]


def test_a_program_fetched_and_not_used_again_stays_in_its_bundle(cache_worker, tmp_path):
    """A process that fetches a file's bundle and compiles only a new program writes back the new one
    with the fetched ones: a test that ran in one partition and not in this one keeps its programs."""
    url, stored, _ = cache_worker
    script = """
import sys
import jax
import remote_cache
from dew.cache import enable_compilation_cache
enable_compilation_cache(sys.argv[1])
remote = remote_cache.install(sys.argv[2], "token")
remote.opened("tests/test_kept.py")
if sys.argv[3] == "first":
    jax.jit(lambda x: (x * 3.0).sum())(jax.numpy.arange(4.0))
else:
    jax.jit(lambda x: (x - 7.0).max())(jax.numpy.arange(5.0))
remote.closed("tests/test_kept.py")
remote.drain()
print(remote.uploaded)
"""
    assert run_child(script, str(tmp_path / "first"), url, "first") == ["1"]
    [key] = [key for key in stored if key.endswith("/bundles/tests/test_kept.py")]
    first = dict(remote_cache.unpacked(stored[key]))
    assert run_child(script, str(tmp_path / "second"), url, "second") == ["1"]
    second = dict(remote_cache.unpacked(stored[key]))
    assert set(first) < set(second) and all(second[name] == first[name] for name in first)


def test_a_url_that_is_no_address_leaves_the_local_cache_as_it_is(tmp_path):
    """A URL that does not parse (an unclosed IPv6 bracket) or names no port is refused with a warning
    when the cache is installed, as one that is no http(s) address is: the run compiles with JAX's own
    cache and raises nothing."""
    script = """
import sys
import remote_cache
from dew.cache import enable_compilation_cache
enable_compilation_cache(sys.argv[1])
print(*(remote_cache.install(url, "token") is None for url in ("http://[::1", "http://host:port", "ftp://host")))
"""
    assert run_child(script, str(tmp_path / "cache")) == ["True", "True", "True"]


def bundles_while_one_is_on_its_way(tmp_path: Path, meanwhile: Callable[[], None]) -> dict[str, bytes]:
    """The bundles a Worker keeps from a cache whose fetch of tests/slow.py's bundle is held on its way
    while tests/other.py opens, `meanwhile` runs, and tests/other.py closes."""
    server, url = counted()
    stored = {}

    class Keeping(Counted):
        def do_PUT(self):
            stored[self.path] = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(204)
            self.end_headers()

    server.RequestHandlerClass = Keeping
    cache = remote_cache.RemoteCache(tmp_path, url, "token", "jax/cpu")
    slow = threading.Thread(target=cache.opened, args=("tests/slow.py",))
    slow.start()
    while not Counted.requests:
        time.sleep(0.01)
    cache.opened("tests/other.py")
    meanwhile()
    Counted.released.set()
    slow.join()
    cache.closed("tests/other.py")
    cache.drain()
    server.shutdown()
    return stored


def test_a_bundle_fetched_while_another_file_runs_is_not_that_files(tmp_path):
    """A file's bundle that arrives while another file's tests run is the first file's: the second's
    bundle holds only what its own tests compiled."""
    stored = bundles_while_one_is_on_its_way(tmp_path, lambda: compiled(tmp_path, "jit_other-1"))
    assert programs(tmp_path)["jit_slow-1"] == b"slow"
    bundle = stored["/jax/cpu/bundles/tests/other.py"]
    assert [key for key, _ in remote_cache.unpacked(bundle)] == ["jit_other-1"]


def test_a_bundle_the_cap_cut_short_is_sent_once(cache_worker, tmp_path, monkeypatch):
    """Two programs that together pass `MAX_BYTES`: the bundle carries the first, is sent once, and is
    not sent again by a later close or the drain while its tests compile nothing new."""
    url, stored, seen = cache_worker
    monkeypatch.setattr(remote_cache, "MAX_BYTES", 1000)
    cache = remote_cache.RemoteCache(tmp_path, url, "token", "jax/cpu")
    cache.opened("tests/big.py")
    compiled(tmp_path, "jit_a-1", b"a" * 600)
    compiled(tmp_path, "jit_b-1", b"b" * 600)
    cache.closed("tests/big.py")
    cache.closed("tests/big.py")
    cache.drain()
    assert [key for key, _ in remote_cache.unpacked(stored["/jax/cpu/bundles/tests/big.py"])] == ["jit_a-1"]
    assert seen.count("/jax/cpu/bundles/tests/big.py") == 2  # its GET, and one PUT


def test_a_program_whose_key_fills_a_file_name_round_trips(cache_worker, tmp_path):
    """A key as long as JAX makes one for a long function's name (`jit_` and the name, `-` and a
    SHA-256 in hex), whose `-cache` file's name is 236 bytes, reaches the local cache from its bundle."""
    url, stored, _ = cache_worker
    key = "jit_" + "f" * 161 + "-" + "0" * 64
    stored["/jax/cpu/bundles/tests/long.py"] = remote_cache.packed([(key, b"executable")])
    cache = remote_cache.RemoteCache(tmp_path, url, "token", "jax/cpu")
    cache.opened("tests/long.py")
    assert programs(tmp_path) == {key: b"executable"} and cache.programs_fetched == 1


def test_a_bundle_whose_program_could_not_be_read_goes_up_once_it_can(cache_worker, tmp_path):
    """A close that cannot read a program the file's tests compiled sends nothing and counts against
    the remote; the next close, once the program reads, sends the bundle."""
    url, stored, _ = cache_worker
    cache = remote_cache.RemoteCache(tmp_path, url, "token", "jax/cpu")
    cache.opened("tests/read.py")
    unreadable = tmp_path / f"jit_read-1{remote_cache.SUFFIX}"
    unreadable.mkdir()
    cache.closed("tests/read.py")
    assert cache.failures == 1 and cache.uploads.unfinished_tasks == 0
    unreadable.rmdir()
    compiled(tmp_path, "jit_read-1")
    cache.closed("tests/read.py")
    cache.drain()
    bundle = stored["/jax/cpu/bundles/tests/read.py"]
    assert [key for key, _ in remote_cache.unpacked(bundle)] == ["jit_read-1"]


def test_a_program_another_file_compiled_while_a_bundle_was_on_its_way_stays_that_files(tmp_path):
    """A file's bundle that brings a program another file's tests compiled while it was on its way
    leaves the program the other file's: its bundle carries it."""
    stored = bundles_while_one_is_on_its_way(tmp_path, lambda: compiled(tmp_path, "jit_slow-1", b"mine"))
    bundle = stored["/jax/cpu/bundles/tests/other.py"]
    assert dict(remote_cache.unpacked(bundle)) == {"jit_slow-1": b"mine"}


def test_a_bundle_takes_what_fits_within_the_workers_bound_and_round_trips(cache_worker, tmp_path):
    """The Worker takes no body past its contract's bound (tools/armada/r2.json, 100 MB): a program just
    over it goes without, and the bundle of the programs that fit reaches the next cache whole. The large
    program's file is sparse, so a close that leaves it out never reads it."""
    url, stored, _ = cache_worker
    bound = json.loads((Path(__file__).resolve().parents[1] / "tools" / "armada" / "r2.json").read_text())
    (tmp_path / "first").mkdir()
    cache = remote_cache.RemoteCache(tmp_path / "first", url, "token", "jax/cpu")
    cache.opened("tests/large.py")
    with (tmp_path / "first" / f"jit_large-1{remote_cache.SUFFIX}").open("wb") as large:
        large.truncate(bound["max_bytes"] - 16)
    compiled(tmp_path / "first", "jit_small-1", b"small")
    cache.closed("tests/large.py")
    cache.drain()
    assert all(len(body) <= bound["max_bytes"] for body in stored.values())
    (tmp_path / "next").mkdir()
    after = remote_cache.RemoteCache(tmp_path / "next", url, "token", "jax/cpu")
    after.opened("tests/large.py")
    assert programs(tmp_path / "next") == {"jit_small-1": b"small"}


def test_a_program_past_the_cap_leaves_its_files_bundle_as_it_was(cache_worker, tmp_path, monkeypatch):
    """A file whose tests compile one program larger than `MAX_BYTES` sends no bundle: the one the
    Worker holds, with the programs that fit, stays."""
    url, stored, seen = cache_worker
    monkeypatch.setattr(remote_cache, "MAX_BYTES", 1000)
    stored["/jax/cpu/bundles/tests/big.py"] = remote_cache.packed([("jit_old-1", b"old")])
    cache = remote_cache.RemoteCache(tmp_path, url, "token", "jax/cpu")
    cache.opened("tests/big.py")
    compiled(tmp_path, "jit_big-1", b"b" * 2000)
    cache.closed("tests/big.py")
    cache.drain()
    assert dict(remote_cache.unpacked(stored["/jax/cpu/bundles/tests/big.py"])) == {"jit_old-1": b"old"}
    assert seen.count("/jax/cpu/bundles/tests/big.py") == 1  # its GET alone


def test_the_token_never_follows_a_redirect(tmp_path):
    """A Worker answering with a redirect to another origin is a failed request: nothing is sent to that
    origin, the bearer token least of all."""
    elsewhere: list[str | None] = []

    class Elsewhere(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            elsewhere.append(self.headers.get("Authorization"))
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    other = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Elsewhere)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    target = f"http://127.0.0.1:{other.server_address[1]}"

    class Redirecting(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", target + self.path)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirecting)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    cache = remote_cache.RemoteCache(tmp_path, url, "token", "jax/cpu")
    cache.opened("tests/a.py")
    server.shutdown()
    other.shutdown()
    assert elsewhere == [] and cache.failures == 1
