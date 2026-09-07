"""The HTTP adapter over a real socket: results, streams, errors, disconnects."""

import json
import socket
import time
from http.client import HTTPConnection
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.pretrained import load_pretrained
from dew.sampling import Sampling, generate
from dew.sampling.engine import Engine
from dew.serve import serve
from test_engine import initialized

FIXTURES = Path(__file__).parent / "fixtures" / "hf"


def post(address, payload, *, path="/generate"):
    connection = HTTPConnection(*address, timeout=120)
    connection.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, (json.loads(body) if body else None)


def stream_lines(address, payload):
    connection = HTTPConnection(*address, timeout=120)
    connection.request("POST", "/generate", body=json.dumps(payload))
    response = connection.getresponse()
    assert response.status == 200 and response.getheader("Content-Type") == "application/x-ndjson"
    lines = [json.loads(line) for line in response.read().decode().splitlines() if line]
    connection.close()
    return lines


@pytest.fixture(scope="module")
def served():
    model, params = initialized()
    with Engine(model, params, max_batch_size=2, max_pending_requests=1) as engine:
        with serve(engine) as server:
            yield engine, server, model, params


def test_results_match_batch_generation_and_health_reports_the_engine(served):
    engine, server, model, params = served
    sampling = Sampling(temperature=0.8, top_k=5, eos_id=(3, 9))
    expected = generate(model, params, jnp.array([[1, 2, 3], [4, 5, 6]]), 6, key=jax.random.key(4), sampling=sampling)
    status, body = post(server.address, {
        "input_ids": [[1, 2, 3], [4, 5, 6]], "max_new_tokens": 6, "seed": 4,
        "sampling": {"temperature": 0.8, "top_k": 5, "eos_id": [3, 9]}})
    assert status == 200 and body["version"] == engine.version.serial and body["text"] is None
    np.testing.assert_array_equal(body["result"]["tokens"], expected.tokens)
    np.testing.assert_array_equal(body["result"]["lengths"], expected.lengths)
    np.testing.assert_array_equal(body["result"]["terminated"], expected.terminated)
    np.testing.assert_allclose(body["result"]["behavior_log_probs"], expected.behavior_log_probs, atol=1e-6)
    np.testing.assert_allclose(body["result"]["raw_log_probs"], expected.raw_log_probs, atol=1e-6)
    status, health = post(server.address, {}, path="/health")
    assert status == 404
    connection = HTTPConnection(*server.address, timeout=30)
    connection.request("GET", "/health")
    response = connection.getresponse()
    health = json.loads(response.read())
    assert response.status == 200 and health["status"] == "ok" and health["versions"] == 1
    assert health["active"] == 0 and health["pending"] == 0


def test_streams_carry_events_before_the_result(served):
    engine, server, model, params = served
    sampling = Sampling(temperature=0.5, eos_id=9)
    expected = generate(model, params, jnp.array([[2, 3, 4]]), 8, key=jax.random.key(9), sampling=sampling)
    lines = stream_lines(server.address, {
        "input_ids": [[2, 3, 4]], "max_new_tokens": 8, "seed": 9, "stream": True,
        "sampling": {"temperature": 0.5, "eos_id": 9}})
    events, results = [line["event"] for line in lines if "event" in line], [line for line in lines if "result" in line]
    assert len(results) == 1 and lines[-1] is results[0]
    length = int(expected.lengths[0])
    assert [event["token"] for event in events] == np.asarray(expected.tokens[0, 3:3 + length]).tolist()
    assert [event["position"] for event in events] == list(range(length))
    np.testing.assert_allclose([event["raw_log_prob"] for event in events], np.asarray(expected.raw_log_probs[0, :length]), atol=1e-6)
    np.testing.assert_array_equal(results[0]["result"]["tokens"], expected.tokens)


def test_malformed_and_overloaded_requests_get_protocol_errors(served):
    engine, server, model, params = served
    bad = [
        ({"input_ids": [[1, 2]], "max_new_tokens": "3"}, "max_new_tokens"),
        ({"max_new_tokens": 3}, "prompt or input_ids"),
        ({"prompt": "hello", "max_new_tokens": 3}, "no tokenizer"),
        ({"input_ids": [[1, 2], [3]], "max_new_tokens": 3}, "one width"),
        ({"input_ids": [[1, 2]], "max_new_tokens": 3, "sampling": {"temperatur": 1.0}}, "unknown sampling"),
        ({"input_ids": [[1, 2]], "max_new_tokens": 3, "sampling": {"temperature": -1.0}}, "temperature"),
        ({"input_ids": [[1, 2]], "max_new_tokens": 3, "process": {"max_steps": 1}}, "canvas"),
        ({"input_ids": [[99, 2]], "max_new_tokens": 3}, "vocabulary"),
        ({"input_ids": [[1, 2]], "max_new_tokens": 3, "sampling": {"temperature": 1.0}, "process": {}}, "either"),
    ]
    for payload, fragment in bad:
        status, body = post(server.address, payload)
        assert status == 400 and fragment in body["error"], (payload, body)
    status, body = post(server.address, {"input_ids": [[1, 2]] * 3, "max_new_tokens": 3})
    assert status == 429 and "rows" in body["error"]
    connection = HTTPConnection(*server.address, timeout=30)
    connection.request("POST", "/generate", body=b"{", headers={"Content-Length": "1"})
    assert connection.getresponse().status == 400
    connection.close()


def test_a_disconnected_stream_cancels_its_request(served):
    engine, server, model, params = served
    raw = socket.create_connection(server.address, timeout=30)
    payload = json.dumps({"input_ids": [[1, 2, 3]], "max_new_tokens": 60, "seed": 1, "stream": True,
                          "sampling": {"temperature": 1.0}}).encode()
    raw.sendall(b"POST /generate HTTP/1.1\r\nHost: test\r\nContent-Length: %d\r\n\r\n" % len(payload) + payload)
    received = b""
    while b"\"event\"" not in received:
        chunk = raw.recv(4096)
        assert chunk, "the stream closed before its first event"
        received += chunk
    assert engine.stats().active == 1
    raw.close()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and engine.stats().active:
        time.sleep(0.05)
    assert engine.stats().active == 0 and engine.stats().allocated_rows == 0


def test_text_prompts_and_canvas_process_overrides_through_a_loaded_source():
    bundle = load_pretrained(FIXTURES / "diffusion-gemma-workflow", dtype="float32", attention_impl="xla",
                             max_seq_len=32)
    assert bundle.processor is not None
    prompts = ["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"]
    inputs = bundle.processor(prompts)
    expected = bundle.generate(inputs, 7, key=jax.random.key(11))
    with Engine(bundle, max_batch_size=2) as engine:
        with serve(engine, processor=bundle.processor) as server:
            status, body = post(server.address, {"prompt": prompts, "max_new_tokens": 7, "seed": 11})
            assert status == 200
            np.testing.assert_array_equal(body["result"]["tokens"], expected.tokens)
            np.testing.assert_array_equal(body["result"]["decoder_steps"], expected.decoder_steps)
            assert body["text"] == bundle.processor.decode(expected.tokens[:, 5:])
            lines = stream_lines(server.address, {
                "prompt": prompts[0], "max_new_tokens": 7, "seed": 11, "stream": True, "process": {"max_steps": 2}})
            spans = [line["event"] for line in lines if "event" in line]
            assert spans and all(span["row"] == 0 for span in spans)
            assert sum(span["decoder_steps"] for span in spans) == lines[-1]["result"]["decoder_steps"][0] <= 4
            status, body = post(server.address, {"prompt": prompts[0], "max_new_tokens": 3,
                                                 "sampling": {"temperature": 0.0}})
            assert status == 400 and "BlockProcess" in body["error"]
            status, body = post(server.address, {"prompt": prompts[0], "max_new_tokens": 3,
                                                 "process": {"entropy": 1.0}})
            assert status == 400 and "unknown process field" in body["error"]
