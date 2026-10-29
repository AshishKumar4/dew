import gc
import http.server
import json
import logging
import os
import sys
import threading
import urllib.parse
from pathlib import Path

import lane_environment

# The lane's environment, before JAX reads it.
os.environ.update(lane_environment.configured(os.environ))
import jax
import pytest
import remote_cache

from dew.cache import default_compilation_cache_dir, enable_compilation_cache

# The suite compiles the same kernels every run, on both lanes, in every xdist
# worker. XLA's persistent cache is keyed by the executable, so a second run
# reuses the first one's compilations. DEW_TEST_NO_CACHE=1 measures the cold
# cost; the numbers in docs/performance.md were taken with it set.
_remote = None
if os.environ.get("DEW_TEST_NO_CACHE"):
    # JAX's own switch, which a run's setup reads too (`dew.cache.persist_compilations`), for this
    # process and the processes its tests start.
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    jax.config.update("jax_enable_compilation_cache", val=False)
else:
    _cache = default_compilation_cache_dir()
    if _cache:
        # A worker that keeps bundles of the shared cache below writes a directory of its own, so the
        # programs a bundle gains are that worker's whole writes, not another's half-written one.
        if os.environ.get("PYTEST_XDIST_WORKER") and os.environ.get("DEW_XLA_CACHE_URL"):
            _cache = os.path.join(_cache, os.environ["PYTEST_XDIST_WORKER"])
        enable_compilation_cache(_cache)
        # The processes its tests start (a checker run as a script, a CLI) compile into the same cache under
        # the same settings, so their programs reach the shared bundles below and their next run is warm
        # (2026-10-29, row 2323: tools/check_native_diffusion.py compiled cold each run, 139.6 s
        # against 43.3 s warm), as JAX's own switch reaches them under DEW_TEST_NO_CACHE.
        for _name in ("jax_compilation_cache_dir", "jax_persistent_cache_min_compile_time_secs",
                      "jax_persistent_cache_min_entry_size_bytes", "jax_persistent_cache_enable_xla_caches"):
            os.environ[_name.upper()] = str(getattr(jax.config, _name))
        # On armada, where each task's container starts cold, a cache the tasks share.
        _remote = remote_cache.install_from_environment()

# XLA parses XLA_FLAGS once, at the first compile, not when the backend
# opens. A test that edits the variable, such as `without_deterministic_ops`
# or the apply_xla_flags tests, would otherwise hand its flags to the whole
# run whenever it happens to run first: the cuda lane then loses
# --xla_gpu_deterministic_ops and its bitwise checks see two compilations of
# one forward disagree. Compiling here fixes the flags set above.
jax.jit(lambda x: x + 1)(0).block_until_ready()


def bundle(node) -> str:
    """The shared cache's bundle of `node`'s test file in this execution partition (`remote_cache.bundle`)."""
    return remote_cache.bundle(node.nodeid, node.config.getoption("ci_group", None),
                               os.environ.get("PYTEST_XDIST_WORKER"))


def pytest_collectstart(collector):
    if _remote is not None and isinstance(collector, pytest.Module):
        _remote.opened(bundle(collector))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Each file's programs join its bundle of the shared cache, written back after each run of its
    tests, every program they used so far."""
    if _remote is not None:
        _remote.opened(bundle(item))
    yield
    if _remote is not None and (nextitem is None or bundle(nextitem) != bundle(item)):
        _remote.closed(bundle(item))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A test the gate fails reports its findings as data (`reference_error.Finding`), the JUnit property
    `gate_findings`, a JSON list of each finding's leaf, rule, measure (an infinite one `Infinity`) and
    the phase it failed: what a promotion matches a candidate's failures to its parent's by."""
    outcome = yield
    # Only a test that loaded the gate can raise its GateFailure: a skip or another failure loads none, so
    # the files whose tests skip are not keyed on the gate's every change in the test cache.
    gate = sys.modules.get("reference_error")
    if call.excinfo is not None and gate is not None and isinstance(call.excinfo.value, gate.GateFailure):
        found = ("gate_findings", json.dumps([{"leaf": finding.leaf, "rule": finding.rule,
                                               "measure": finding.measure, "phase": call.when}
                                              for finding in call.excinfo.value.findings]))
        # This phase's report, and the item's, which its teardown's report copies and the JUnit
        # report reads a test's properties from where only one phase fails.
        outcome.get_result().user_properties.append(found)
        item.user_properties.append(found)


def pytest_sessionfinish(session, exitstatus):
    if _remote is not None:
        _remote.drain()


def pytest_terminal_summary(terminalreporter):
    if _remote is not None:
        terminalreporter.write_line(_remote.summary())


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("mesh")
    needed = marker.kwargs.get("devices", lane_environment.MESH_DEVICES) if marker else 0
    if jax.device_count() < needed:
        pytest.skip(f"needs a {needed}-device mesh; this run has "
                    f"{jax.device_count()} {jax.default_backend()} device(s)")
    for backend, device in (("gpu", "a cuda device"), ("tpu", "a tpu")):
        marker = item.get_closest_marker(backend)
        if marker and jax.default_backend() != backend:
            pytest.skip(marker.kwargs.get("reason", f"runs on {device}"))


@pytest.fixture(autouse=True)
def own_rung_records(monkeypatch, tmp_path_factory):
    """Each test keeps its own rung records (`dew.training.rungs`). The suite
    shares one compilation cache, beside which a run records its step's
    rung, and a rung one test forces would be another's floor."""
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path_factory.mktemp("rungs")))


@pytest.fixture(autouse=True)
def own_data_cache(monkeypatch, tmp_path_factory):
    """Each test keeps its own automatic data cache (`dew.data.cache.automatic_root`): records one
    test's maps made are not another's to read back."""
    monkeypatch.setenv("DEW_DATA_CACHE", str(tmp_path_factory.mktemp("data-cache")))


@pytest.fixture(autouse=True)
def own_hub_answers(monkeypatch, tmp_path_factory):
    """Each test keeps its own recorded Hub answers (`dew.cache.hub_answer`): a commit one test's
    stand-in answered is not another's offline record."""
    monkeypatch.setenv("DEW_HUB_ANSWERS", str(tmp_path_factory.mktemp("hub-answers")))


MAPPINGS = Path("/proc/self/maps")
"""This process's memory mappings, a line each, where Linux lists them."""
MAPPINGS_BUDGET = (int(Path("/proc/sys/vm/max_map_count").read_text()) // 2
                   if Path("/proc/sys/vm/max_map_count").exists() else None)
"""The mappings past which a test's end releases the compiled executables (`released_past_budget`): half
the kernel's limit, leaving the next test the other half to compile in; None where no limit is listed."""


def release() -> None:
    """Release every compiled executable JAX holds, and the code XLA mapped for it."""
    jax.clear_caches()
    gc.collect()


@pytest.fixture(autouse=True, scope="module")
def release_compilations():
    """Each test file's compiled executables are released when it ends.

    XLA:CPU maps every executable's code into the process, and JAX keeps an
    executable for as long as its jit cache does. A run that goes on through
    many files reached 65,391 mappings, past the 65,530 a Linux kernel allows
    by default, and the next compile failed with "Failed to materialize
    symbols". Ubuntu 24.04 allows 1,048,576, so GitHub's runners never saw
    it; Debian and most kernels do.
    """
    yield
    release()


@pytest.fixture(autouse=True)
def released_past_budget():
    """A test that leaves the process past `MAPPINGS_BUDGET` mappings releases them as it ends, so no file
    compiles its way to the limit: test_serving.py alone reached 64,581 of armada's 65,530 and failed its
    next compile ("LLVM compilation error: Cannot allocate memory"); held to the budget it peaked at
    32,683 and passed. Reading the list takes a few milliseconds at the budget."""
    yield
    if MAPPINGS_BUDGET is not None:
        with MAPPINGS.open("rb") as mapped:
            if sum(1 for _ in mapped) > MAPPINGS_BUDGET:
                release()


@pytest.fixture
def without_deterministic_ops(monkeypatch):
    """The cuda lane runs the whole suite under --xla_gpu_deterministic_ops,
    which Dew refuses cudnn attention under (openxla/xla#46500). XLA read the
    variable at the compile above, so taking the flag back out of the
    environment leaves this executable's reductions as they are and lets a
    test reach the cudnn path it is about."""
    flags = os.environ.get("XLA_FLAGS", "")
    monkeypatch.setenv("XLA_FLAGS", lane_environment.without_deterministic_ops(flags))


@pytest.fixture
def caplog(caplog):
    """Capture Dew's logger as well as the application's root logger. Dew's records reach the
    capture once, through a handler of their own that does not pass them on to the root, whether
    or not the run's setup configured the logger (`dew.logging.configure`), which then leaves it."""
    logger = logging.getLogger("dew")
    propagated = logger.propagate
    logger.addHandler(caplog.handler)
    logger.propagate = False
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.propagate = propagated


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def cache_worker():
    """The dew-xla-cache Worker's protocol over a dict: GET, PUT, a listing two keys a page, a 404 on
    a miss, a 401 without the token, Cloudflare's 403 for urllib's own User-Agent, which never
    reaches the Worker, and the Worker's 413 for a body outside its contract's bounds
    (tools/armada/r2.json). Each request's path is kept."""
    stored, seen = {}, []
    contract = Path(__file__).resolve().parents[1] / "tools" / "armada" / "r2.json"
    bounds = json.loads(contract.read_text())

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
            if not bounds["min_bytes"] <= int(self.headers["Content-Length"]) <= bounds["max_bytes"]:
                self.send_response(413)
                self.send_header("Connection", "close")
                self.end_headers()
                return
            stored[self.path] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(204)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", stored, seen
    server.shutdown()
