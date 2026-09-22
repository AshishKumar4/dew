"""Rollout servers over a remote engine: token requests, reported likelihoods, weight reloads.

The vLLM wire format is mocked at the HTTP transport, so the checks read the
request the SDK actually sends and the response vLLM documents: token-id
prompts, seeds, EOS controls, `return_tokens_as_token_ids` and one reported
log-probability per sampled token. The safetensors reload writes a real Qwen2
export that `load_pretrained` reads back, and posts the reload calls to a
real local HTTP endpoint in order.
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


def vllm(choice):
    """An OpenAI client whose transport answers every completion with `choice`."""
    calls = []

    def network(request):
        calls.append(json.loads(request.content))
        return httpx2.Response(200, json={"id": "fixture", "model": "tiny", "created": 1,
                                          "object": "text_completion", "choices": [choice]})

    client = openai.OpenAI(base_url="http://fixture/v1", api_key="fixture", max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(network)))
    return OpenAICompletion("tiny", client, provider="vllm"), calls


def choice(ids, logprobs, reason):
    return {"index": 0, "text": "", "finish_reason": reason,
            "logprobs": {"tokens": [f"token_id:{token}" for token in ids], "token_logprobs": logprobs,
                         "top_logprobs": None, "text_offset": [0] * len(ids)}}


def pushed(variables):
    raise AssertionError("no weights are pushed in this check")


def test_a_draw_is_the_reported_ids_and_behavior_likelihoods():
    completion, calls = vllm(choice([3, 5, EOS], [-.5, -1., -.25], "stop"))
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
    assert sent["stop_token_ids"] == [EOS] and sent["return_tokens_as_token_ids"] is True
    assert sent["logprobs"] == 0 and sent["temperature"] == 1.0 and sent["top_k"] == -1


def test_a_budget_stop_is_unterminated_and_a_disagreeing_reason_is_refused():
    completion, _ = vllm(choice([3, 5], [-.5, -1.], "length"))
    server = OpenAIRolloutServer(completion, Sampling(eos_id=EOS), pushed)
    try:
        assert not server.submit([1], 2, seed=0).result().terminated
        with pytest.raises(ValueError, match="disagrees"):
            server.submit([1], 4, seed=0).result()
    finally:
        server.close()


def test_raw_engine_likelihoods_are_not_taken_for_a_transformed_policy():
    completion, _ = vllm(choice([EOS], [0.], "stop"))
    with pytest.raises(ValueError, match="processed_logprobs"):
        OpenAIRolloutServer(completion, Sampling(temperature=.7, eos_id=EOS), pushed)
    OpenAIRolloutServer(completion, Sampling(temperature=.7, eos_id=EOS), pushed, processed_logprobs=True).close()


class Engine(BaseHTTPRequestHandler):
    """Records every POST path and body; answers 200."""

    seen: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        Engine.seen.append((self.path, json.loads(body) if body else None))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def test_a_reload_writes_the_policy_as_safetensors_then_asks_the_engine(tmp_path):
    source = load_pretrained(FIXTURE, dtype="float32")
    changed = jax.tree.map(lambda leaf: leaf * 2 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
                           source.variables)
    Engine.seen = []
    engine = HTTPServer(("127.0.0.1", 0), Engine)
    thread = threading.Thread(target=engine.serve_forever, daemon=True)
    thread.start()
    try:
        reload = SafetensorsReload(source, tmp_path / "served", f"http://127.0.0.1:{engine.server_port}")
        reload(changed)
    finally:
        engine.shutdown()
    assert [path for path, _ in Engine.seen] == ["/collective_rpc", "/reset_prefix_cache"]
    assert Engine.seen[0][1] == {"method": "reload_weights"}
    assert not (tmp_path / ".served.staging").exists()
    served = load_pretrained(tmp_path / "served", dtype="float32")
    for written, expected in zip(jax.tree.leaves(served.variables), jax.tree.leaves(changed), strict=True):
        # Served in bfloat16: equal to the pushed weights at that precision.
        np.testing.assert_array_equal(np.asarray(written),
                                      np.asarray(jnp.asarray(expected).astype(jnp.bfloat16).astype(jnp.float32)))
