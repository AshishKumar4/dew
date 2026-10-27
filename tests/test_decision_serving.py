"""examples/serve_decisions.py: Jev's endpoint over Starlette, on Laya's toy checkpoint."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import jax
import numpy as np
import pytest
from starlette.testclient import TestClient
from tool_support import load

from dew.decision import Decide
from dew.decision.calibration import Temperatures
from dew.inference import ScoringServer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "laya"
REQUEST = json.loads((FIXTURES / "cases.json").read_text())["quickstart"]


@pytest.fixture(scope="module")
def example():
    return load("serve_decisions", "examples")


@pytest.fixture(scope="module")
def decide() -> Decide:
    return Decide.from_pretrained(FIXTURES / "tiny", attention_impl="xla")


def test_the_endpoint_answers_what_systemone_answers(example, decide):
    """`POST /v1/systemone` returns the task's own Jev response, field for field."""
    with TestClient(example.app(example.served(decide))) as client:
        response = client.post("/v1/systemone", json=REQUEST)
        assert response.status_code == 200
        assert response.json() == json.loads(json.dumps(decide.systemone(REQUEST)))
        assert client.get("/health").json() == {"model": decide.name}


@pytest.mark.parametrize("body", [
    b"{not json", b"[1, 2]",
    json.dumps({"questions": REQUEST["questions"]}).encode(),
    json.dumps({"state": "x", "questions": {"q": {"type": "rank", "instructions": "x"}}}).encode(),
])
def test_a_request_jev_would_refuse_gets_a_422_with_its_reason(example, decide, body):
    with TestClient(example.app(example.served(decide))) as client:
        response = client.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422 and response.json()["error"]


def test_an_api_key_is_required_once_set(example, decide):
    with TestClient(example.app(example.served(decide), api_key="secret")) as client:
        def status(key: str | None) -> int:
            headers = {} if key is None else {"authorization": f"Bearer {key}"}
            return client.post("/v1/systemone", json=REQUEST, headers=headers).status_code

        assert (status(None), status("wrong"), status("secret")) == (401, 401, 200)


def test_a_strict_endpoint_drops_the_images_jev_does_not_know(example, decide):
    body = {**REQUEST, "images": ["aGVsbG8="]}
    with TestClient(example.app(example.served(decide))) as client:
        assert client.post("/v1/systemone", json=body).status_code == 422
    with TestClient(example.app(example.served(decide, strict=True))) as client:
        strict = client.post("/v1/systemone", json=body)
    assert strict.status_code == 200 and strict.json() == json.loads(json.dumps(decide.systemone(REQUEST)))


class Counted:
    """`task` whose passes are counted, each by the rows it scores; its first pass waits until
    `expected` requests were admitted, so the rest wait in the queue meanwhile."""

    def __init__(self, task, expected: int):
        self.task, self.expected, self.calls = task, expected, []
        self.admitted, self.all_in = 0, threading.Event()

    def admit(self, request):
        scored = self.task.admit(request)
        self.admitted += 1
        if self.admitted == self.expected:
            self.all_in.set()
        return scored

    def fits(self, rows):
        return self.task.fits(rows)

    def forward(self, variables, rows):
        if not self.calls:
            self.all_in.wait()
        self.calls.append(len(rows))
        return self.task.forward(variables, rows)

    def answer(self, payload, outputs):
        return self.task.answer(payload, outputs)


def test_concurrent_requests_are_answered_together_each_with_its_own_response(example, decide):
    """Six requests posted at once, one of them a body Jev refuses: the refused one gets its own 422
    at admission, and the five's rows are answered in at most two passes, the first one's wait for
    them all holding the rest in the queue, each as it is alone in every field a batch cannot move
    (tests/test_decision_batches.py)."""
    from test_decision_batches import CASES, discrete

    bodies = [*CASES.values(), {"questions": REQUEST["questions"]}]
    task, variables = decide.scoring()
    counted = Counted(task, expected=len(bodies) - 1)
    together = threading.Barrier(len(bodies))
    with TestClient(example.app(ScoringServer.from_task(counted, variables))) as client:
        def post(body):
            together.wait()
            return client.post("/v1/systemone", json=body)

        with ThreadPoolExecutor(len(bodies)) as pool:
            responses = list(pool.map(post, bodies))
    assert counted.admitted == len(bodies) - 1 and len(counted.calls) <= 2 and max(counted.calls) >= 4
    for body, response in zip(bodies, responses, strict=True):
        if "state" not in body:
            assert response.status_code == 422 and response.json()["error"]
        else:
            assert response.status_code == 200
            assert discrete(response.json()) == discrete(json.loads(json.dumps(decide.systemone(body))))


def test_a_reload_under_concurrent_requests_pairs_each_answer_with_one_tasks_weights_and_calibration(
        example, decide):
    """A reload, while eight threads post requests one after another, swaps the task and its weights at
    once: every answer is exactly the lone answer of the task its response names, the old one's or
    the new one's, its weights with its own calibration, never one's weights with the other's
    calibration. One request a pass keeps each pass the lone answer's program. The reload comes once
    eight answers are back, so answers of both come."""
    from test_decision_batches import CASES

    old = replace(decide, name="v1")
    new = replace(decide, name="v2", calibration=replace(decide.calibration, temperatures=Temperatures(
        types={"choice": 3.0, "score": 3.0, "noul": 3.0})),
        variables=jax.tree.map(lambda leaf: leaf * 1.25 if np.issubdtype(leaf.dtype, np.floating) else leaf,
                               decide.variables))
    alone = {task.name: task for task in (old, new)}
    server = ScoringServer.from_task(*old.scoring(), rows=1)
    answered: list[tuple[dict, httpx.Response]] = []
    some = threading.Event()

    def poster(client, bodies):
        for body in bodies:
            answered.append((body, client.post("/v1/systemone", json=body)))
            if len(answered) >= 8:
                some.set()

    with TestClient(example.app(server)) as client, ThreadPoolExecutor(8) as pool:
        workers = [pool.submit(poster, client, list(CASES.values()) * 2) for _ in range(8)]
        assert some.wait(120), "eight answers did not come back"
        server.reload(*new.scoring())
        for worker in workers:
            worker.result()
    named = {response.json()["model"] for _, response in answered}
    assert named == {"v1", "v2"}
    for body, response in answered:
        assert response.status_code == 200
        assert response.json() == json.loads(json.dumps(alone[response.json()["model"]].systemone(body)))


def test_the_served_weights_are_restored_in_the_dtype_they_serve_in(example):
    """The endpoint restores a task's weights in the dtype it serves in unless told another, so a run
    trained in float32 takes the memory it is served in (ledger 1889): its leaves' dtypes are a
    bfloat16 load's, and a stated storage dtype holds."""
    import jax

    def dtypes(decide):
        return jax.tree.map(lambda leaf: leaf.dtype, decide.variables)

    tiny = str(FIXTURES / "tiny")
    served = example.loaded(example.Options(model=tiny, dtype="bfloat16"))
    narrow = Decide.from_pretrained(tiny, dtype="bfloat16", param_dtype="bfloat16")
    wide = Decide.from_pretrained(tiny, dtype="bfloat16", param_dtype="float32")
    assert dtypes(served) == dtypes(narrow) != dtypes(wide)
    stated = example.loaded(example.Options(model=tiny, dtype="bfloat16", param_dtype="float32"))
    assert dtypes(stated) == dtypes(wide)
