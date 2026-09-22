"""Rollout servers over a remote engine: token requests, reported likelihoods, weight reloads.

The vLLM and SGLang wire formats are mocked at the HTTP transport, so the
checks read the request the SDK actually sends and the response each engine
documents: token-id prompts, seeds, EOS controls, the field that returns the
sampled ids (vLLM's `return_tokens_as_token_ids` renders them into the
logprob tokens, SGLang's `return_token_ids` lists them on the choice) and one
reported log-probability per sampled token. The safetensors reload writes a
real Qwen2 export that `load_pretrained` reads back, and posts the reload
calls to a real local HTTP endpoint in order.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

openai = pytest.importorskip("openai", reason="optional inference-clients extra")
import httpx2

from dew.inference import OpenAICompletion, OpenAIRolloutServer, SafetensorsReload
from dew.interop import load_pretrained
from dew.sampling import Sampling

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


def pushed(variables):
    raise AssertionError("no weights are pushed in this check")


@pytest.mark.parametrize("provider", ["vllm", "sglang"])
def test_a_draw_is_the_reported_ids_and_behavior_likelihoods(provider):
    completion, calls = engine(provider, lambda _: choice(provider, [3, 5, EOS], [-.5, -1., -.25], "stop"))
    server = OpenAIRolloutServer(completion, Sampling(eos_id=EOS), pushed, version=4)
    try:
        draw = server.submit([1, 2, 9], 8, seed=11).result()
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
    assert sent[returned[provider]] is True and returned[{"vllm": "sglang", "sglang": "vllm"}[provider]] not in sent


@pytest.mark.parametrize("provider", ["vllm", "sglang"])
def test_a_budget_stop_is_unterminated_and_a_disagreeing_reason_is_refused(provider):
    completion, _ = engine(provider, lambda _: choice(provider, [3, 5], [-.5, -1.], "length"))
    server = OpenAIRolloutServer(completion, Sampling(eos_id=EOS), pushed)
    try:
        assert not server.submit([1], 2, seed=0).result().terminated
        with pytest.raises(ValueError, match="disagrees"):
            server.submit([1], 4, seed=0).result()
    finally:
        server.close()


def test_listed_ids_that_disagree_with_the_likelihoods_are_refused():
    def misaligned(_):
        answer = choice("sglang", [3, 5], [-.5, -1.], "length")
        answer["token_ids"] = [3]
        return answer

    completion, _ = engine("sglang", misaligned)
    server = OpenAIRolloutServer(completion, Sampling(eos_id=EOS), pushed)
    try:
        with pytest.raises(ValueError, match="token_ids"):
            server.submit([1], 2, seed=0).result()
    finally:
        server.close()


def test_raw_engine_likelihoods_are_not_taken_for_a_transformed_policy():
    completion, _ = engine("vllm", lambda _: choice("vllm", [EOS], [0.], "stop"))
    with pytest.raises(ValueError, match="processed_logprobs"):
        OpenAIRolloutServer(completion, Sampling(temperature=.7, eos_id=EOS), pushed)
    OpenAIRolloutServer(completion, Sampling(temperature=.7, eos_id=EOS), pushed, processed_logprobs=True).close()


def test_sglang_serves_a_tempered_policy_but_no_filter():
    # SGLang reports log softmax(logits / T), before its top-k, top-p and min-p.
    completion, _ = engine("sglang", lambda _: choice("sglang", [EOS], [0.], "stop"))
    OpenAIRolloutServer(completion, Sampling(temperature=.7, eos_id=EOS), pushed).close()
    for filtering in (Sampling(top_k=20, eos_id=EOS), Sampling(temperature=.7, top_p=.9, eos_id=EOS),
                      Sampling(min_p=.05, eos_id=EOS)):
        with pytest.raises(ValueError, match="filter"):
            OpenAIRolloutServer(completion, filtering, pushed)
    with pytest.raises(ValueError, match="vLLM mode"):
        OpenAIRolloutServer(completion, Sampling(eos_id=EOS), pushed, processed_logprobs=True)


@pytest.mark.parametrize("provider", ["vllm", "sglang"])
def test_a_draw_in_flight_across_a_push_keeps_its_submission_version(provider):
    release = threading.Event()
    arrived = threading.Semaphore(0)

    def slow(_):
        arrived.release()
        assert release.wait(10)
        return choice(provider, [3, EOS], [-.5, -.25], "stop")

    completion, _ = engine(provider, slow)
    pushes = []
    server = OpenAIRolloutServer(completion, Sampling(eos_id=EOS), pushes.append, version=1)
    try:
        early = server.submit([1], 4, seed=0)
        assert arrived.acquire(timeout=10)
        server.load({"params": {}}, 2)
        late = server.submit([1], 4, seed=1)
        release.set()
        assert (early.result().version, late.result().version) == (1, 2)
    finally:
        release.set()
        server.close()
    assert pushes == [{"params": {}}]


def test_a_failed_push_keeps_the_version():
    def refused(variables):
        raise RuntimeError("/update_weights_from_disk answered 400")

    completion, _ = engine("sglang", lambda _: choice("sglang", [EOS], [0.], "stop"))
    server = OpenAIRolloutServer(completion, Sampling(eos_id=EOS), refused, version=3)
    try:
        with pytest.raises(RuntimeError, match="400"):
            server.load({"params": {}}, 4)
        assert server.version == 3 and server.submit([1], 1, seed=0).result().version == 3
    finally:
        server.close()


class Engine(BaseHTTPRequestHandler):
    """Records every POST path and body and answers like the engine it stands for.

    Like vLLM with a request in flight, a plain prefix-cache reset answers
    200 and resets nothing; only `reset_running_requests=true` resets. Like
    SGLang, a weight update that fails answers 400 with its message, and one
    that loads flushes the radix cache unless the request says otherwise.
    """

    seen: list = []
    reset: list = []
    flushed: list = []
    refuse = False

    def answer(self, status, body):
        self.send_response(status)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        body = json.loads(raw) if raw else None
        Engine.seen.append((self.path.split("?")[0], body))
        if self.path.startswith("/reset_prefix_cache"):
            if Engine.refuse:
                return self.answer(500, {})
            Engine.reset.append("reset_running_requests=true" in self.path)
        if self.path == "/update_weights_from_disk":
            if Engine.refuse:
                return self.answer(400, {"success": False, "message": "Failed to update weights: shape mismatch",
                                         "num_paused_requests": 0})
            Engine.flushed.append(body.get("flush_cache", True))
        self.answer(200, {"success": True, "message": "Succeeded to update model weights.",
                          "num_paused_requests": 0})

    def log_message(self, *args):
        pass


@pytest.fixture
def served():
    Engine.seen, Engine.reset, Engine.flushed, Engine.refuse = [], [], [], False
    engine = HTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{engine.server_port}"
    engine.shutdown()
    Engine.refuse = False


def doubled(source):
    return jax.tree.map(lambda leaf: leaf * 2 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, source.variables)


def assert_served(directory, expected):
    served = load_pretrained(directory, dtype="float32")
    for written, pushed in zip(jax.tree.leaves(served.variables), jax.tree.leaves(expected), strict=True):
        # Served in bfloat16: equal to the pushed weights at that precision.
        np.testing.assert_array_equal(np.asarray(written),
                                      np.asarray(jnp.asarray(pushed).astype(jnp.bfloat16).astype(jnp.float32)))


def test_a_vllm_reload_writes_the_policy_as_safetensors_then_asks_the_engine(tmp_path, served):
    source = load_pretrained(FIXTURE, dtype="float32")
    changed = doubled(source)
    SafetensorsReload(source, tmp_path / "served", served, "vllm")(changed)
    assert [path for path, _ in Engine.seen] == ["/collective_rpc", "/reset_prefix_cache"]
    assert Engine.seen[0][1] == {"method": "reload_weights"}
    # The reset preempts running requests, so no in-flight KV outlives the weights.
    assert Engine.reset == [True]
    assert not (tmp_path / ".served.staging").exists()
    assert_served(tmp_path / "served", changed)


def test_an_sglang_reload_loads_the_directory_and_flushes_the_radix_cache_in_one_call(tmp_path, served, monkeypatch):
    source = load_pretrained(FIXTURE, dtype="float32")
    changed = doubled(source)
    monkeypatch.chdir(tmp_path)
    SafetensorsReload(source, Path("served"), served, "sglang")(changed)
    assert [path for path, _ in Engine.seen] == ["/update_weights_from_disk"]
    body = Engine.seen[0][1]
    # The engine resolves the path in its own working directory, so it is sent absolute.
    assert body["model_path"] == str(tmp_path / "served")
    # In-flight requests finish on the old weights instead of being aborted,
    # and no prefix they computed survives the load.
    assert body["abort_all_requests"] is False and Engine.flushed == [True]
    assert_served(tmp_path / "served", changed)


@pytest.mark.parametrize(("provider", "refusal"), [("vllm", "reset_prefix_cache answered 500"),
                                                   ("sglang", "update_weights_from_disk answered 400.*shape mismatch")])
def test_a_refused_reload_fails_the_push(tmp_path, served, provider, refusal):
    source = load_pretrained(FIXTURE, dtype="float32")
    Engine.refuse = True
    with pytest.raises(RuntimeError, match=refusal):
        SafetensorsReload(source, tmp_path / "served", served, provider)(source.variables)

