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
import os
import socket
import subprocess
import sys
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


def pushed(variables, version):
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


def test_a_draw_in_flight_across_a_push_keeps_its_submission_version():
    release = threading.Event()
    arrived = threading.Semaphore(0)

    def slow(_):
        arrived.release()
        assert release.wait(10)
        return choice("vllm", [3, EOS], [-.5, -.25], "stop")

    completion, _ = engine("vllm", slow)
    pushes = []
    server = OpenAIRolloutServer(completion, Sampling(eos_id=EOS), lambda *push: pushes.append(push), version=1)
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
    assert pushes == [({"params": {}}, 2)]


def test_sglang_ids_are_read_without_likelihoods():
    completion, _ = engine("sglang", lambda _: {"index": 0, "text": "", "finish_reason": "length",
                                                "logprobs": None, "token_ids": [3, 5]})
    result = completion([[1, 2]], 2, extra_body={"return_token_ids": True})
    assert result.tokens == ((3, 5),) and result.log_probs == (None,)


def test_a_failed_push_keeps_the_version():
    def refused(variables, version):
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
    """Records every POST path, query and body and answers like the engine it stands for.

    Like vLLM v0.30.0, `/pause` and `/resume` answer a status,
    `/reset_prefix_cache` answers 200 with `{"success": bool}` whether or not
    it reset, and `/update_weight_version` answers `{"success": true}`. Like
    SGLang v0.5.20, a weight update answers `success` in its body, with 400
    when it fails, and flushes the radix cache unless the request says
    otherwise. Like rllm-model-gateway, `/admin/weight_version` answers the
    stamp it now holds. The server's `failure` makes the reset or update
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
        if path == "/reset_prefix_cache":
            if failure == "status":
                return self.answer(500, {"error": "engine dead"})
            return self.answer(200, {"success": failure != "body"})
        if failure is not None:
            return self.answer(400 if failure == "status" else 200,
                               {"success": False, "message": "Failed to update weights: shape mismatch",
                                "num_paused_requests": 0})
        self.server.flushed.append(body.get("flush_cache", True))
        self.answer(200, {"success": True, "message": "Succeeded to update model weights.", "num_paused_requests": 0})

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
    return jax.tree.map(lambda leaf: leaf * 2 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, source.variables)


def assert_served(directory, expected):
    served = load_pretrained(directory, dtype="float32")
    for written, pushed in zip(jax.tree.leaves(served.variables), jax.tree.leaves(expected), strict=True):
        # Served in bfloat16: equal to the pushed weights at that precision.
        np.testing.assert_array_equal(np.asarray(written),
                                      np.asarray(jnp.asarray(pushed).astype(jnp.bfloat16).astype(jnp.float32)))


def paths(server):
    return [path for path, _, _ in server.seen]


def test_a_vllm_push_drains_reloads_resets_stamps_and_resumes_every_replica(tmp_path, replicas):
    source = load_pretrained(FIXTURE, dtype="float32")
    changed = doubled(source)
    engines = replicas(3)
    SafetensorsReload(source, tmp_path / "served", tuple(server.url for server in engines), "vllm")(changed, 5)
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
    source = load_pretrained(FIXTURE, dtype="float32")
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


@pytest.mark.parametrize("provider", ["vllm", "sglang"])
def test_the_gateway_is_stamped_only_after_every_replica_serves_the_version(tmp_path, replicas, provider):
    source = load_pretrained(FIXTURE, dtype="float32")
    first, second, gateway = replicas(3)
    publish = SafetensorsReload(source, tmp_path / "served", (first.url, second.url), provider, gateway=gateway.url)
    publish(source.variables, 1)
    assert gateway.seen == [("/admin/weight_version", "", {"weight_version": 1})]
    second.failure = "body"
    with pytest.raises(RuntimeError, match=f"version 2 reached 1 of 2 replicas.*{second.url}"):
        publish(source.variables, 2)
    # One replica did not take version 2, so no call may be stamped with it.
    assert gateway.seen == [("/admin/weight_version", "", {"weight_version": 1})]
    # The replica that took it finished its whole sequence.
    assert paths(first)[-1] == ("/resume" if provider == "vllm" else "/update_weights_from_disk")


@pytest.mark.parametrize(("provider", "failure", "refusal"), [
    ("vllm", "status", "reset_prefix_cache answered 500"),
    ("vllm", "body", "reset_prefix_cache answered 200.*false"),
    ("sglang", "status", "update_weights_from_disk answered 400.*shape mismatch"),
    ("sglang", "body", "update_weights_from_disk answered 200.*shape mismatch"),
])
def test_a_refused_reload_fails_the_push(tmp_path, replicas, provider, failure, refusal):
    source = load_pretrained(FIXTURE, dtype="float32")
    (server,) = replicas(1)
    server.failure = failure
    with pytest.raises(RuntimeError, match=refusal):
        SafetensorsReload(source, tmp_path / "served", (server.url,), provider)(source.variables, 1)
    # A vLLM push that failed leaves the engine paused rather than serving half-loaded weights.
    assert "/resume" not in paths(server) and "/update_weight_version" not in paths(server)


WORKER = Path(__file__).with_name("weight_publication_worker.py")


def run_pool(directory, engine, processes=2, devices=2):
    """`processes` workers pushing one sharded tree to `engine`, and their reports in process order."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{probe.getsockname()[1]}"
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                   "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}
    outs = [directory / f"process{index}.json" for index in range(processes)]
    running = [subprocess.Popen([sys.executable, str(WORKER), "--out", str(out), "--coordinator", coordinator,
                                 "--processes", str(processes), "--process-id", str(index),
                                 "--directory", str(directory / "served"), "--engine", engine],
                                env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True) for index, out in enumerate(outs)]
    try:
        logs = [process.communicate(timeout=600)[0] for process in running]
    finally:
        for process in running:
            if process.poll() is None:
                process.kill()
    for index, process in enumerate(running):
        assert process.returncode == 0, f"process {index} exited {process.returncode}\n{logs[index]}"
    return [json.loads(out.read_text()) for out in outs]


@pytest.mark.distributed
def test_a_pool_publishes_its_sharded_policy_once_and_every_process_hears_a_failure(tmp_path, replicas):
    (engine,) = replicas(1)
    reports = run_pool(tmp_path, engine.url)
    assert all(report["processes"] == 2 and report["sharded"] and report["error"] is None for report in reports)
    # Process 0 alone writes and posts: one sequence reaches the engine, not one per process.
    assert paths(engine) == ["/pause", "/collective_rpc", "/reset_prefix_cache", "/update_weight_version", "/resume"]
    assert_served(tmp_path / "served", doubled(load_pretrained(FIXTURE, dtype="float32")))

    engine.seen.clear()
    engine.failure = "body"
    reports = run_pool(tmp_path, engine.url)
    # Process 0 raises the engine's refusal; process 1 hears it at the agreement point instead of hanging.
    assert "reset_prefix_cache answered 200" in reports[0]["error"]
    assert reports[1]["error"].startswith("PeerFailure") and "reset_prefix_cache" in reports[1]["error"]
    assert paths(engine) == ["/pause", "/collective_rpc", "/reset_prefix_cache"]
