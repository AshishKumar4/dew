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
