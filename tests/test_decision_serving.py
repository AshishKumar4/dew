"""examples/serve_decisions.py: Jev's endpoint over Starlette, on Laya's toy checkpoint."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from dew.decision import Decide

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "laya"
REQUEST = json.loads((FIXTURES / "cases.json").read_text())["quickstart"]


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("serve_decisions", ROOT / "examples" / "serve_decisions.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def decide() -> Decide:
    return Decide.from_pretrained(FIXTURES / "tiny", attention_impl="xla")


def test_the_endpoint_answers_what_systemone_answers(example, decide):
    """`POST /v1/systemone` returns the task's own Jev response, field for field."""
    client = TestClient(example.app(decide))
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
    response = TestClient(example.app(decide)).post("/v1/systemone", content=body,
                                                     headers={"content-type": "application/json"})
    assert response.status_code == 422 and response.json()["error"]


def test_an_api_key_is_required_once_set(example, decide):
    client = TestClient(example.app(decide, api_key="secret"))

    def status(key: str | None) -> int:
        headers = {} if key is None else {"authorization": f"Bearer {key}"}
        return client.post("/v1/systemone", json=REQUEST, headers=headers).status_code

    assert (status(None), status("wrong"), status("secret")) == (401, 401, 200)


def test_a_strict_endpoint_drops_the_images_jev_does_not_know(example, decide):
    body = {**REQUEST, "images": ["aGVsbG8="]}
    assert TestClient(example.app(decide)).post("/v1/systemone", json=body).status_code == 422
    strict = TestClient(example.app(decide, strict=True)).post("/v1/systemone", json=body)
    assert strict.status_code == 200 and strict.json() == json.loads(json.dumps(decide.systemone(REQUEST)))


def test_a_whole_request_is_answered_and_a_cut_one_is_refused_by_its_capacity(example, decide):
    """Under `whole`, the server answers a request that fits uncut as it
    always does, and refuses with a 422 naming the maximum context length one
    it could answer only by cutting the state, or its options to the
    checkpoint's budgets: the reason Decision Index's http engine reads as
    unsupported."""
    from dataclasses import replace

    wide = replace(decide, layout=replace(decide.layout, option_tokens=256, head_max_len=192, max_len=256))
    client = TestClient(example.app(wide, whole=True))
    fits = {"state": "short", "questions": {"q": {"type": "choice", "instructions": "Which?",
                                                  "criteria": {"a": "word " * 20, "b": "b"}}}}
    assert client.post("/v1/systemone", json=fits).json() == json.loads(json.dumps(wide.systemone(fits)))
    long_state = {**fits, "state": "word " * 400}
    for task, request in ((wide, long_state), (decide, fits)):
        assert task.systemone(request)["answers"]
        refused = TestClient(example.app(task, whole=True)).post("/v1/systemone", json=request)
        assert refused.status_code == 422
        assert "maximum context length" in refused.json()["error"]
