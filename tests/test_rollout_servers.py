"""Rollout servers over a remote engine: token requests, reported likelihoods, weight reloads.

The vLLM and SGLang wire formats are mocked at the HTTP transport, so the
checks read the request the SDK actually sends and the response each engine
documents: token-id prompts, seeds, EOS controls, the field that returns the
sampled ids (vLLM's `return_tokens_as_token_ids` renders them into the
logprob tokens, SGLang's `return_token_ids` lists them on the choice) and one
reported log-probability per sampled token. The safetensors reload writes a
real Qwen2 export that `Pretrained.load` reads back, and posts the reload
calls to a real local HTTP endpoint in order.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inference import OpenAICompletion, OpenAIRolloutServer, SafetensorsReload
from dew.interop import Pretrained
from dew.sampling import Sampling

openai = pytest.importorskip("openai", reason="optional inference-clients extra")
httpx2 = import_module("httpx2")


FIXTURE = Path(__file__).parent / "fixtures/hf/qwen2-tiny"
EOS = 7


def engine(provider, answer):
    """An OpenAI client for `provider` whose transport answers each completion with `answer(request)`."""
    calls = []

    def network(request):
        calls.append(json.loads(request.content))
        return httpx2.Response(200, json={"id": "fixture", "model": "tiny", "created": 1,
                                          "object": "text_completion", "choices": [answer(calls[-1])]})

    client = openai.OpenAI(base_url="http://fixture/v1", api_key="fixture", max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(network)))
    return OpenAICompletion("tiny", client, provider=provider), calls


def choice(provider, ids, logprobs, reason):
    """One completion choice as `provider` reports sampled ids: rendered into the tokens, or listed."""
    if provider == "vllm":
        rendered, listed = [f"token_id:{token}" for token in ids], {}
    else:
        rendered, listed = [f"<{token}>" for token in ids], {"token_ids": list(ids)}
    return {"index": 0, "text": "", "finish_reason": reason, **listed,
            "logprobs": {"tokens": rendered, "token_logprobs": logprobs,
                         "top_logprobs": None, "text_offset": [0] * len(ids)}}


def pushed(variables, version):
    raise AssertionError("no weights are pushed in this check")


@pytest.mark.parametrize("provider", ["vllm", "sglang"])
def test_a_draw_is_the_reported_ids_and_behavior_likelihoods(provider):
    completion, calls = engine(provider, lambda _: choice(provider, [3, 5, EOS], [-.5, -1., -.25], "stop"))
    server = OpenAIRolloutServer(completion, Sampling(eos_token_ids=EOS), pushed, version=4)
    try:
        draw = server.submit([1, 2, 9], 8, key=11).result()
    finally:
        server.close()
    assert draw.prompt == (1, 2, 9) and draw.tokens == (3, 5, EOS) and draw.terminated
    assert draw.behavior_log_probs == (-.5, -1., -.25)
    assert draw.raw_log_probs is None and draw.version == 4
    sent = calls[0]
    assert sent["prompt"] == [[1, 2, 9]] and sent["seed"] == 11 and sent["max_tokens"] == 8
    assert sent["stop_token_ids"] == [EOS]
    assert sent["logprobs"] == 0 and sent["temperature"] == 1.0 and sent["top_k"] == -1
    returned = {"vllm": "return_tokens_as_token_ids", "sglang": "return_token_ids"}
    assert (
        sent[returned[provider]] is True
        and returned[{"vllm": "sglang", "sglang": "vllm"}[provider]] not in sent
    )


def test_a_draw_in_flight_across_a_push_keeps_its_submission_version():
    release = threading.Event()
    arrived = threading.Semaphore(0)

    def slow(_):
        arrived.release()
        assert release.wait(10)
        return choice("vllm", [3, EOS], [-.5, -.25], "stop")

    completion, _ = engine("vllm", slow)
    pushes = []
    server = OpenAIRolloutServer(
        completion, Sampling(eos_token_ids=EOS), lambda *push: pushes.append(push), version=1
    )
    try:
        early = server.submit([1], 4, key=0)
        assert arrived.acquire(timeout=10)
        server.load({"params": {}}, 2)
        late = server.submit([1], 4, key=1)
        release.set()
        assert (early.result().version, late.result().version) == (1, 2)
    finally:
        release.set()
        server.close()
    assert pushes == [({"params": {}}, 2)]


def test_sglang_ids_are_read_without_likelihoods():
    completion, _ = engine("sglang", lambda _: {"index": 0, "text": "", "finish_reason": "length",
                                                "logprobs": None, "token_ids": [3, 5]})
    result = completion([[1, 2]], 2, extra_body={"return_token_ids": True})
    assert result.tokens == ((3, 5),) and result.log_probs == (None,)


class Engine(BaseHTTPRequestHandler):
    """Records every request's path, query and body and answers like the engine it stands for.

    Like vLLM v0.30.0, `/pause` and `/resume` answer a status,
    `/reset_prefix_cache` answers 200 with `{"success": bool}` whether or not
    it reset, and `/update_weight_version` answers `{"success": true}`. Like
    SGLang v0.5.20, a weight update answers `success` in its body, with 400
    when it fails, and flushes the radix cache unless the request says
    otherwise. Like vLLM's NCCL weight transfer, `/get_world_size` reports one
    worker, and opening the group and starting or finishing an update answer
    200. The server's `failure` makes the reset or update
    fail: "status" with an error status, "body" with 200 and `success: false`.
    Every replica is its own server, so each keeps its own record.
    """

    def answer(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        body = json.loads(raw) if raw else None
        path, _, query = self.path.partition("?")
        record, failure = self.server.seen, self.server.failure
        record.append((path, query, body))
        if path in ("/pause", "/resume"):
            return self.answer(200, {"status": "paused" if path == "/pause" else "resumed"})
        if path == "/collective_rpc":
            return self.answer(200, [None])
        if path == "/admin/weight_version":
            return self.answer(200, {"weight_version": body["weight_version"]})
        if path == "/update_weight_version":
            return self.answer(200, {"success": True, "new_version": body["new_version"]})
        if path in ("/init_weight_transfer_engine", "/start_weight_update", "/finish_weight_update"):
            return self.answer(200, {"success": True})
        if path == "/update_weights" and failure is None:
            return self.answer(200, {"success": True})
        if path == "/reset_prefix_cache":
            if failure == "status":
                return self.answer(500, {"error": "engine dead"})
            return self.answer(200, {"success": failure != "body"})
        if failure is not None:
            return self.answer(400 if failure == "status" else 200,
                               {"success": False, "message": "Failed to update weights: shape mismatch",
                                "num_paused_requests": 0})
        self.server.flushed.append(body.get("flush_cache", True))
        self.answer(
            200, {"success": True, "message": "Succeeded to update model weights.", "num_paused_requests": 0}
        )
        return None

    def do_GET(self):
        path, _, query = self.path.partition("?")
        self.server.seen.append((path, query, None))
        if path == "/get_world_size":
            self.answer(200, {"world_size": 1})
        else:
            self.answer(404, {"error": f"no route {path}"})

    def log_message(self, *args):
        pass


@pytest.fixture
def replicas():
    """Start local engine stand-ins on demand; each has `url`, `seen`, `flushed` and `failure`."""
    started = []

    def start(count):
        for _ in range(count):
            server = HTTPServer(("127.0.0.1", 0), Engine)
            server.seen, server.flushed, server.failure = [], [], None
            server.url = f"http://127.0.0.1:{server.server_port}"
            threading.Thread(target=server.serve_forever, daemon=True).start()
            started.append(server)
        return started[-count:]

    yield start
    for server in started:
        server.shutdown()


def doubled(source):
    return jax.tree.map(
        lambda leaf: leaf * 2 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, source.variables
    )


def assert_served(directory, expected):
    served = Pretrained.load(directory, dtype="float32")
    for written, pushed in zip(jax.tree.leaves(served.variables), jax.tree.leaves(expected), strict=True):
        # Served in bfloat16: equal to the pushed weights at that precision.
        np.testing.assert_array_equal(np.asarray(written),
                                      np.asarray(jnp.asarray(pushed).astype(jnp.bfloat16).astype(jnp.float32)))


def paths(server):
    return [path for path, _, _ in server.seen]


def test_a_vllm_push_drains_reloads_resets_stamps_and_resumes_every_replica(tmp_path, replicas):
    source = Pretrained.load(FIXTURE, dtype="float32")
    changed = doubled(source)
    engines = replicas(3)
    SafetensorsReload(source, tmp_path / "served", tuple(server.url for server in engines), "vllm")(
        changed, 5
    )
    for server in engines:
        assert paths(server) == ["/pause", "/collective_rpc", "/reset_prefix_cache", "/update_weight_version",
                                 "/resume"]
        # In-flight requests finish on the old weights and no new one starts until the resume.
        assert server.seen[0][1] == "mode=wait"
        assert server.seen[1][2] == {"method": "reload_weights"}
        assert server.seen[3][2] == {"new_version": "5"}
    assert not (tmp_path / ".served.staging").exists()
    assert_served(tmp_path / "served", changed)


def test_an_sglang_push_loads_the_directory_stamps_and_flushes_in_one_call(tmp_path, replicas, monkeypatch):
    source = Pretrained.load(FIXTURE, dtype="float32")
    changed = doubled(source)
    monkeypatch.chdir(tmp_path)
    engines = replicas(2)
    SafetensorsReload(source, Path("served"), tuple(server.url for server in engines), "sglang")(changed, 2)
    for server in engines:
        assert paths(server) == ["/update_weights_from_disk"]
        body = server.seen[0][2]
        # The engine resolves the path in its own working directory, so it is sent absolute.
        assert body["model_path"] == str(tmp_path / "served") and body["weight_version"] == "2"
        # In-flight requests finish on the old weights instead of being aborted,
        # and no prefix they computed survives the load.
        assert body["abort_all_requests"] is False and server.flushed == [True]
    assert_served(tmp_path / "served", changed)
