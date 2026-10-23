"""examples/serve_decisions.py: Jev's endpoint over Starlette, on Laya's toy checkpoint."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from tool_support import load

from dew.decision import Decide

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
    with TestClient(example.app(decide)) as client:
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
    with TestClient(example.app(decide)) as client:
        response = client.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422 and response.json()["error"]


def test_an_api_key_is_required_once_set(example, decide):
    with TestClient(example.app(decide, api_key="secret")) as client:
        def status(key: str | None) -> int:
            headers = {} if key is None else {"authorization": f"Bearer {key}"}
            return client.post("/v1/systemone", json=REQUEST, headers=headers).status_code

        assert (status(None), status("wrong"), status("secret")) == (401, 401, 200)


def test_a_strict_endpoint_drops_the_images_jev_does_not_know(example, decide):
    body = {**REQUEST, "images": ["aGVsbG8="]}
    with TestClient(example.app(decide)) as client:
        assert client.post("/v1/systemone", json=body).status_code == 422
    with TestClient(example.app(decide, strict=True)) as client:
        strict = client.post("/v1/systemone", json=body)
    assert strict.status_code == 200 and strict.json() == json.loads(json.dumps(decide.systemone(REQUEST)))


class Counted:
    """`decide` whose `systemone` calls are counted, each by the bodies it answers."""

    def __init__(self, decide: Decide):
        self.decide, self.name, self.calls = decide, decide.name, []

    def systemone(self, bodies, **answering):
        self.calls.append(len(bodies))
        return self.decide.systemone(bodies, **answering)


def test_concurrent_requests_are_answered_together_each_with_its_own_response(example, decide):
    """Six requests posted at once, one of them a body Jev refuses, are answered by one systemone
    call: the five each as they are alone in every field a batch cannot move
    (tests/test_decision_batches.py), the refused one with its own 422."""
    from test_decision_batches import CASES, discrete

    bodies = [*CASES.values(), {"questions": REQUEST["questions"]}]
    counted = Counted(decide)
    # The batch closes at its sixth request, long before its window would.
    served = example.app(counted, batch_window=60.0, max_batch=len(bodies))
    together = threading.Barrier(len(bodies))
    with TestClient(served) as client:
        def post(body):
            together.wait()
            return client.post("/v1/systemone", json=body)

        with ThreadPoolExecutor(len(bodies)) as pool:
            responses = list(pool.map(post, bodies))
    assert counted.calls == [len(bodies)]
    for body, response in zip(bodies, responses, strict=True):
        if "state" not in body:
            assert response.status_code == 422 and response.json()["error"]
        else:
            assert response.status_code == 200
            assert discrete(response.json()) == discrete(json.loads(json.dumps(decide.systemone(body))))


def test_an_app_outside_its_lifespan_refuses_to_answer(example, decide):
    """The batches run within the app's lifespan; a request outside it is refused, not left waiting."""
    with pytest.raises(RuntimeError):
        TestClient(example.app(decide)).post("/v1/systemone", json=REQUEST)
