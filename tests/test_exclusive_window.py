"""`dew.telemetry.benchmark.exclusive_window`: a job's measured region held alone on its node's GPUs by the
node's session, which a thread of the test plays (`tool_support.StandInSession`: the job's `$SESSION/wake` and
`$OUT`); a process with no session (a laptop, CI) waits for nothing."""

import threading

import jax.numpy as jnp
import pytest
from tool_support import StandInSession

from dew.telemetry import benchmark


def heavy():
    """A computation the CPU backend takes seconds over, dispatched and not waited for."""
    x = jnp.ones((2000, 2000))
    return x @ x @ x @ x


def in_session(session: StandInSession, monkeypatch) -> None:
    """The process given the stand-in session's directory and its job's $OUT, as a session gives a job."""
    for name, value in session.environment.items():
        monkeypatch.setenv(name, value)


def test_a_job_asks_for_its_window_waits_for_the_grant_and_waits_for_the_session_to_take_it_back(
        tmp_path, monkeypatch):
    """The request names the job's cores and the job does not enter until the session grants it; ending the
    window removes the request and returns only once the session has acknowledged (removed its grant)."""
    with StandInSession(tmp_path) as session:
        in_session(session, monkeypatch)
        session.grants.clear()
        session.acks.clear()
        entered, leave, left = threading.Event(), threading.Event(), threading.Event()

        def job():
            with benchmark.exclusive_window(cores=(5, 2)):
                entered.set()
                leave.wait(30)
            left.set()

        thread = threading.Thread(target=job)
        thread.start()
        assert not entered.wait(1.0) and session.asked == ["2,5"]
        session.grants.set()
        assert entered.wait(30) and (session.out / "window.granted").exists()
        leave.set()
        assert not left.wait(1.0) and not (session.out / "window").exists()
        session.acks.set()
        assert left.wait(30)
        thread.join()
        assert not (session.out / "window.granted").exists() and len(session.released) == 1


def test_the_window_opens_and_closes_on_an_idle_device(tmp_path, monkeypatch):
    """Work the job dispatched before the window is done when it opens, and work it dispatched inside is done
    when the session sees the window end, so the GPUs are idle at both."""
    ready, before = [], []
    with StandInSession(tmp_path) as session:
        in_session(session, monkeypatch)

        def released() -> None:
            ready.append(inside["work"].is_ready())

        inside = {}
        session.on_release = released
        queued = heavy()
        with benchmark.exclusive_window():
            before.append(queued.is_ready())
            inside["work"] = heavy()
    assert before == [True] and ready == [True]


def test_a_window_the_body_leaves_by_an_error_is_given_back(tmp_path, monkeypatch):
    with StandInSession(tmp_path) as session:
        in_session(session, monkeypatch)
        with pytest.raises(RuntimeError), benchmark.exclusive_window():
            raise RuntimeError("the measured region failed")
    assert len(session.released) == 1 and not (session.out / "window").exists()
    assert not (session.out / "window.granted").exists()


def test_a_window_the_session_refuses_raises_and_its_body_never_runs(tmp_path, monkeypatch):
    ran = []
    with StandInSession(tmp_path) as session:
        in_session(session, monkeypatch)
        session.refusal = "every CPU named"
        with pytest.raises(benchmark.WindowRefused), benchmark.exclusive_window(cores=(0, 1)):
            ran.append(True)
        session.refusal = None
        with benchmark.exclusive_window(cores=(0,)):
            ran.append(False)
    assert ran == [False] and not (session.out / "window.refused").exists()


def test_without_a_session_a_window_waits_for_nothing_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("SESSION", raising=False)
    monkeypatch.setenv("OUT", str(tmp_path))
    with benchmark.exclusive_window(cores=(0,)):
        inside = True
    assert inside and not list(tmp_path.iterdir())
