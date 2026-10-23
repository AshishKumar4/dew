"""The serving engine (`dew.inference.engine.Engine`) and the scoring kind (`dew.inference.scoring`):
a request is scored by the task and weights that admitted it, a step settles its whole batch or fails
it whole, a turn's bound counts what it drains, a request's callback may reload from its own step, and
a request that waits once closing is seen is answered."""

import threading
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from dew.inference import Engine, Scored, ScoringServer


@dataclass(frozen=True)
class Named:
    """A scoring task whose answer names it, the task that laid the request out, the request, and the
    weight it scored with; `short` answers that many requests too few."""

    name: str
    short: int = 0

    def admit(self, request: str) -> Scored[tuple[str, str]]:
        return Scored(tokens=1, payload=(self.name, request))

    def score(self, variables, payloads):
        weight = float(variables["w"])
        answers = [(self.name, laid, request, weight) for laid, request in payloads]
        return answers[:len(answers) - self.short]


def weights(value: float):
    return {"w": jnp.full((), value, jnp.float32)}


def test_a_request_admitted_before_a_reload_is_scored_by_the_task_that_laid_it_out():
    """A reload serves the new pair to what is submitted after it; what was admitted before is scored
    by its own task and weights, in batches of its own."""
    server = ScoringServer.from_task(Named("a"), weights(1.0))
    before = server.submit("x")
    server.reload(Named("b"), weights(2.0))
    after = server.submit("y")
    server.run()
    assert before.result() == ("a", "a", "x", 1.0)
    assert after.result() == ("b", "b", "y", 2.0)


def test_an_answer_too_few_fails_the_whole_batch():
    """A task that answers fewer requests than it was given fails every request of the batch, and the
    step raises, so no request waits for an answer that never comes."""
    server = ScoringServer.from_task(Named("a", short=1), weights(1.0))
    futures = [server.submit(request) for request in "xy"]
    with pytest.raises(ValueError):
        server.step()
    for future in futures:
        with pytest.raises(ValueError):
            future.result()


def test_a_turn_counts_the_cancelled_requests_it_drains():
    """A step takes at most `rows` requests off the queue, cancelled ones among them, so a queue of
    cancelled requests never holds one turn for its whole length."""
    server = ScoringServer.from_task(Named("a"), weights(1.0), rows=2)
    futures = [server.submit(request) for request in "vwxyz"]
    for future in futures:
        future.cancel()
    server.step()
    assert server.queued == 3


def test_a_requests_callback_reloads_from_inside_its_step():
    """A request's done-callback runs on the serving thread, inside its kind's step, and may reload the
    engine's kind there; the next request is scored by the reloaded pair."""
    server = ScoringServer.from_task(Named("a"), weights(1.0))
    engine = Engine(server)
    reloaded = threading.Event()

    def reload(_):
        engine.reload(lambda: server.reload(Named("b"), weights(2.0)))
        reloaded.set()

    first = server.submit("x")
    first.add_done_callback(reload)
    engine.start()
    try:
        assert first.result() == ("a", "a", "x", 1.0)
        reloaded.wait()
        assert server.submit("y").result() == ("b", "b", "y", 2.0)
    finally:
        engine.close()


class Looked:
    """A serving kind that answers every waiting request a step, and whose next look at its own work
    runs `hook` once after the look, so the look reports what waited before the hook."""

    def __init__(self) -> None:
        self.waiting: list[str] = []
        self.answered: list[str] = []
        self.arrived = lambda: None
        self.hook: Callable[[], None] | None = None
        self.failure: BaseException | None = None

    @property
    def busy(self) -> bool:
        busy = bool(self.waiting)
        hook, self.hook = self.hook, None
        if hook is not None:
            hook()
        return busy

    def step(self) -> None:
        self.answered += self.waiting
        self.waiting.clear()

    def fail(self, failure: BaseException) -> None:
        self.failure = failure


def test_a_request_that_waits_before_closing_is_answered():
    """A request submitted, then the engine closed, while the serving thread looks at the kinds'
    work: the look saw no work, but closing came after the request, so the request is answered before
    the thread ends. Closed from the serving thread itself, the engine ends once its turn returns."""
    kind = Looked()
    engine = Engine(kind)
    looked = threading.Event()

    def submitted_then_closed():
        try:
            kind.waiting.append("late")
            kind.arrived()
            engine.close()
        finally:
            looked.set()

    kind.hook = submitted_then_closed
    engine.start()
    looked.wait()
    engine.close()
    assert kind.failure is None
    assert kind.answered == ["late"]


def test_a_stop_while_a_reload_holds_the_step_runs_no_step_after_it():
    """A stop asked while a reload holds the engine's step, its thread waiting behind the reload with
    work in hand, ends the thread at the step it would have taken: no step runs after the stop."""
    kind = Looked()
    engine = Engine(kind)
    held, release, looked = threading.Event(), threading.Event(), threading.Event()

    def holding():
        held.set()
        release.wait()

    reloading = threading.Thread(target=engine.reload, args=(holding,))
    reloading.start()
    held.wait()
    kind.hook = looked.set
    kind.waiting.append("waiting")
    engine.start()
    looked.wait()
    stopping = threading.Thread(target=engine.stop)
    stopping.start()
    release.set()
    reloading.join()
    stopping.join()
    assert kind.answered == []


def test_an_engine_closed_from_its_own_thread_ends_free_to_start_again():
    """A close from a request's callback on the serving thread returns at once, and the thread, once
    it has answered what waits, leaves the engine free: it starts again without an outside close."""
    kind = Looked()
    engine = Engine(kind, name="closed-from-its-own-thread")
    kind.hook = engine.close
    engine.start()
    for thread in [thread for thread in threading.enumerate() if thread.name == engine.name]:
        thread.join()
    engine.start()
    engine.close()
    assert kind.failure is None
