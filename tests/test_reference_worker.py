"""A recorder's session with a reference's worker (tools/reference_worker.py): real worker processes run
as a reference's are, in sessions the recorder owns, over both carriers the arrays travel by."""

import contextlib
import importlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_process_support import gone_within
from tool_support import load

from dew.cleanup import owned_group

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "tests" / "reference_worker_probe.py"
protocol = load("reference_worker")
CARRIERS = ["paths", "frames"]


@contextlib.contextmanager
def bounded(seconds: float = 120):
    """The block ended with a TimeoutError once `seconds` have passed, however it waits: a worker that never
    answers is waited on without end by design, and a test that finds one fails here, not at its job's end."""
    def expired(number, frame):
        raise TimeoutError(f"still waiting after {seconds} s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def probing(kind: str, **options):
    """A session with the probe worker `kind` (tests/reference_worker_probe.py), bounded."""
    with bounded(), protocol.session([sys.executable, str(PROBE), kind], **options) as served:
        yield served


@pytest.fixture
def scratch(tmp_path, monkeypatch) -> Path:
    """Where the sessions of a test make their directories, empty."""
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture
def probe_log(tmp_path, monkeypatch) -> Path:
    """The file each probe worker writes its kind and pid to."""
    log = tmp_path / "workers.log"
    monkeypatch.setenv("REFERENCE_PROBE_LOG", str(log))
    return log


def ask(served, mode: str, **arrays: np.ndarray) -> dict[str, np.ndarray]:
    return served.ask({"mode": np.asarray(mode), **arrays})


def facts(served) -> dict[str, str | int]:
    """The probe worker's pid, its descendant's, its temporary directory and a file it made there."""
    answer = ask(served, "facts")
    return {name: str(answer[name]) if name in ("temporary", "created") else int(answer[name])
            for name in answer}


def assert_same(answer: dict[str, np.ndarray], sent: dict[str, np.ndarray]) -> None:
    """The reference's and the truth's arrays of `answer` are exactly `sent`, dtype and shape and values."""
    for arm in protocol.arms(answer):
        assert arm.keys() == sent.keys()
        for name, array in sent.items():
            assert arm[name].dtype == array.dtype and arm[name].shape == array.shape, name
            np.testing.assert_array_equal(arm[name], array)


@pytest.mark.parametrize("carrier", CARRIERS)
def test_a_workers_library_output_reaches_stderr_and_never_the_answers(carrier):
    """A library prints while it imports and while it runs, by Python and by a native write, a line `done`
    and the bytes of a frame among it; the recorder's answers are still each its own request's, and a
    worker whose stdout a library closed still answers."""
    rng = np.random.default_rng(0)
    with probing(carrier) as served:
        assert served.started == "probe 1"
        for size in (1000, 3, 50):
            sent = {"x": rng.normal(size=size)}
            assert_same(ask(served, "noisy", **sent), sent)
        ask(served, "stdout")
        sent = {"x": rng.normal(size=7)}
        assert_same(ask(served, "echo", **sent), sent)


@pytest.mark.parametrize("carrier", CARRIERS)
def test_every_array_comes_back_as_it_went_whatever_the_request_before_it(carrier):
    """Float, integer, boolean and text arrays of any shape, an empty one among them, travel both ways
    exactly, a request smaller than the one before it too."""
    rng = np.random.default_rng(1)
    everything = {"f64": rng.normal(size=(5, 3)), "f32": rng.normal(size=(4,)).astype(np.float32),
                  "u8": np.arange(256, dtype=np.uint8), "i32": np.arange(-3, 3, dtype=np.int32),
                  "flag": np.asarray([True, False, True]), "name": np.asarray("a text option"),
                  "empty": np.zeros((0, 4)), "scalar": np.asarray(2.5), "nan": np.asarray([np.nan, np.inf])}
    with probing(carrier) as served:
        for sent in (everything, {"f64": np.asarray([1.0])}, everything):
            assert_same(ask(served, "echo", **sent), sent)


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("mode", ["exit", "raise", "kill", "close"])
def test_a_worker_that_ends_on_a_request_ends_the_recorders_ask_with_an_error(carrier, mode):
    """A worker that exits, fails, is killed, or closes the pipe it answers by and lives on, ends the
    recorder's ask with an error, as a live process the recorder waits on never does; the session then
    ends it with what it started."""
    with probing(carrier) as served:
        known = facts(served)
        with pytest.raises(RuntimeError):
            ask(served, mode)
        process = served.process
    assert process.returncode is not None
    assert gone_within(known["worker"], 30) and gone_within(known["descendant"], 30)


@pytest.mark.parametrize("kind", ["stops-at-start", "no-ready", "unknown-carrier"])
def test_a_worker_that_does_not_say_it_is_ready_ends_the_session_at_its_start(kind, scratch, probe_log):
    """A worker that exits before it speaks, or whose first line is no readiness or names no carrier,
    fails the session as it opens, its process ended and its directory removed."""
    with pytest.raises(RuntimeError), probing(kind):
        pytest.fail("the session opened")
    assert gone_within(int(probe_log.read_text().split()[1]), 30)
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize(("kind", "carrier"), [("wrong-answer", "paths"), ("cut-frame", "frames")])
def test_a_reply_that_is_no_answer_or_a_frame_cut_short_ends_the_ask(kind, carrier, scratch):
    """The worker that replies to a request with something other than `done`, or whose frame ends before its
    size says, fails the ask."""
    with pytest.raises(RuntimeError), probing(kind) as served:
        served.ask({"x": np.arange(3)})
    assert list(scratch.iterdir()) == []


def test_a_recorder_interrupted_while_it_waits_on_a_silent_worker_leaves_nothing_running(scratch, probe_log):
    """A worker that never speaks is waited on for as long as the recorder lives, and no timeout of the
    session's ends it; a recorder interrupted there (here by the harness's own deadline) ends the worker
    and removes its directory."""
    with pytest.raises(TimeoutError), bounded(3), protocol.session([sys.executable, str(PROBE),
                                                                    "silent-at-start"]):
        pytest.fail("the session opened")
    assert gone_within(int(probe_log.read_text().split()[1]), 30)
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("ending", [None, ValueError, KeyboardInterrupt])
def test_a_session_ends_its_group_and_removes_its_files_however_its_block_ends(ending, scratch):
    """The worker, what it started, its temporary directory and the files a library made there are gone
    after the block, whether it ends normally or by an error, an interrupt among them; the worker is
    reaped, and the directory the session made is removed."""
    seen = {}

    def block() -> None:
        with probing("paths") as served:
            seen.update(facts(served), process=served.process)
            assert Path(seen["created"]).is_file()
            if ending is not None:
                raise ending("the block's own")

    if ending is None:
        block()
    else:
        with pytest.raises(ending):
            block()
    assert seen["process"].returncode is not None
    assert gone_within(seen["descendant"], 30)
    assert not Path(seen["created"]).exists() and not Path(seen["temporary"]).exists()
    assert list(scratch.iterdir()) == []


class Own(Exception):
    """A block's own error, which no cleanup failure may replace."""


def test_a_cleanup_that_fails_leaves_the_blocks_own_error_the_one_raised_and_ends_the_rest(scratch):
    """A pipe that fails to close, the request a dead worker never read left unflushed in it, is a note on the
    block's own error, which stays the error raised: the group is still stopped, the other pipe closed and
    the directory removed."""
    own = Own("the block's own")
    seen = {}
    with pytest.raises(Own) as raised, probing("paths") as served:
        seen.update(facts(served), served=served)
        with pytest.raises(RuntimeError):
            ask(served, "exit")
        served.process.stdin.write(b"a request the dead worker never reads")
        raise own
    assert raised.value is own and len(own.__notes__) == 1
    assert seen["served"].process.stdout.closed and seen["served"].process.returncode is not None
    assert gone_within(seen["descendant"], 30)
    assert list(scratch.iterdir()) == []


def test_a_cleanup_that_fails_after_a_block_that_did_not_is_raised_with_the_rest_ended(scratch):
    """With no error of the block's own, the failure to close a pipe is the error raised, and the group is
    still stopped and the directory removed."""
    seen = {}
    with pytest.raises(BrokenPipeError), probing("paths") as served:
        seen.update(facts(served), served=served)
        with pytest.raises(RuntimeError):
            ask(served, "exit")
        served.process.stdin.write(b"a request the dead worker never reads")
    assert seen["served"].process.stdout.closed and seen["served"].process.returncode is not None
    assert gone_within(seen["descendant"], 30)
    assert list(scratch.iterdir()) == []


RECORDER = """
import sys
import numpy as np
import reference_worker

with reference_worker.session([sys.executable, {probe!r}, "paths"]) as served:
    answer = served.ask({{"mode": np.asarray("facts")}})
    print(int(answer["worker"]), int(answer["descendant"]), flush=True)
    sys.stdin.read()
"""
"""A recorder that opens a session on the probe worker, says which processes it started and waits."""


@pytest.mark.parametrize("number", [signal.SIGKILL, signal.SIGTERM])
def test_a_worker_ends_with_its_recorder_however_the_recorder_dies(number):
    """A recorder killed by SIGKILL, or by a SIGTERM that unwinds nothing, leaves no worker and no process the
    worker started: the group's ownership outlives no owner (a worker outlived its killed owner 43 hours
    at 195% CPU)."""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
        (str(REPO_ROOT / "src"), str(REPO_ROOT / "tools")))}
    recorder = subprocess.Popen([sys.executable, "-c", RECORDER.format(probe=str(PROBE))], env=environment,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                start_new_session=True)
    try:
        worker, descendant = (int(pid) for pid in recorder.stdout.readline().split())
        os.kill(recorder.pid, number)
        recorder.wait(timeout=60)
        assert gone_within(worker, 30) and gone_within(descendant, 30)
    finally:
        recorder.kill()
        recorder.wait()


@pytest.mark.parametrize("carrier", CARRIERS)
def test_a_worker_whose_recorder_closes_its_input_ends_cleanly(carrier):
    """A recorder that ends its requests ends the worker with exit status 0, and a frame is cut short only
    if its size or its bytes are: the pipe closed between frames is the end, closed within one is an error."""
    worker = [sys.executable, str(REPO_ROOT / "tools" / "reference_worker.py"), str(PROBE), carrier]
    sent = {"mode": np.asarray("echo"), "x": np.arange(4.0)}
    request = io.BytesIO()
    protocol.send(request, sent)
    ended = subprocess.run(worker, input=b"", capture_output=True, timeout=60, check=False)
    assert ended.returncode == 0
    if carrier == "frames":
        whole = subprocess.run(worker, input=request.getvalue(), capture_output=True, timeout=60, check=False)
        assert whole.returncode == 0
        answers = io.BytesIO(whole.stdout)
        answers.readline()
        assert_same(protocol.received(answers), {"x": sent["x"]})
        assert protocol.received(answers) is None
        for cut in (3, 8, 8 + 10, len(request.getvalue()) - 1):
            assert subprocess.run(worker, input=request.getvalue()[:cut], capture_output=True, timeout=60,
                                  check=False).returncode != 0, cut


def test_a_frame_arrives_whole_though_its_bytes_arrive_a_few_at_a_time():
    """The frames a recorder sends and the answers a worker sends are each read whole when their bytes come
    in pieces, with the worker's end of the pipe in pieces too."""
    request = io.BytesIO()
    sent = {"x": np.arange(30.0), "name": np.asarray("text")}
    protocol.send(request, sent)
    frame = request.getvalue()
    with bounded(), owned_group([sys.executable, str(REPO_ROOT / "tools" / "reference_worker.py"), str(PROBE),
                                 "fragments"], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as worker:
        assert worker.stdout.readline().startswith(b"ready")
        for start in range(0, len(frame), 5):
            worker.stdin.write(frame[start:start + 5])
            worker.stdin.flush()
            time.sleep(0.001)
        assert_same(protocol.received(worker.stdout), sent)


def test_a_recording_through_a_worker_is_the_recording_made_in_process(monkeypatch):
    """A site recorded by tools/venv_reference.py through a worker that runs it under `answered` (JAX's
    float32 run, then its float64 run) is exactly the recording `record_case` makes of the same site run
    here: every array of it, the declared seeds and the evidence read by check 1 alike."""
    venv = load("venv_reference")
    recorder = load("recorded_references")
    site = importlib.import_module("reference_probe_site")
    known = {"test": "reference_probe_site", "worker": str(PROBE), "reference": "a product", "license": "MIT"}

    def evaluate(inputs):
        with jax.enable_x64(new_val=False):
            reference = {"y": np.asarray(jnp.dot(jnp.asarray(inputs["x"], jnp.float32),
                                                 jnp.asarray(inputs["param.w"], jnp.float32),
                                                 precision="highest"))}
        truth = {"y": np.asarray(jnp.dot(jnp.asarray(inputs["x"], jnp.float64),
                                         jnp.asarray(inputs["param.w"], jnp.float64), precision="highest"))}
        return reference, truth

    monkeypatch.setitem(venv.SITES, "applied", known)
    with bounded(600), jax.enable_x64(new_val=True):
        through = venv.recording("applied", sys.executable, None)
        here = recorder.record_case("applied", site.draw, evaluate, store_inputs=False, summary=True,
                                    dew=jax.jit(lambda inputs: site.dew(jnp.float32, inputs)))
    assert through.keys() == here.keys()
    for name in here:
        np.testing.assert_array_equal(through[name], here[name], err_msg=name)
