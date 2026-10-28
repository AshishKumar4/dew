"""Rollout servers over a remote engine: token requests, reported likelihoods, weight reloads.

The vLLM and SGLang wire formats are mocked at the HTTP transport, so the
checks read the request the SDK actually sends and the response each engine
documents: token-id prompts, seeds, EOS controls, the field that returns the
sampled ids (vLLM's `return_tokens_as_token_ids` renders them into the
logprob tokens, SGLang's `return_token_ids` lists them on the choice) and one
reported log-probability per sampled token. The safetensors reload writes a
real Qwen2 export that `Pretrained.load` reads back, and posts the reload
calls to a real local HTTP endpoint in order. The NCCL push runs on GPUs: a real
group to a replica worker on a second GPU, and a replica that refuses its join.
"""

import base64
import ctypes
import itertools
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inference import NCCLPush, OpenAICompletion, OpenAIRolloutServer, SafetensorsReload
from dew.inference.nccl import _DTYPES
from dew.interop import Pretrained
from dew.sampling import Sampling

openai = pytest.importorskip("openai", reason="optional inference-clients extra")
httpx2 = import_module("httpx2")


FIXTURE = Path(__file__).parent / "fixtures/hf/qwen2-tiny"
FP8 = Path(__file__).parent / "fixtures/hf/minimax-m2-fp8-tiny"
"""An FP8 source, whose export requantizes its E4M3 weights."""
EOS = 7


def engine(provider, answer):
    """An OpenAI client for `provider` whose transport answers each completion with `answer(request)`,
    a request that streams as SGLang streams it: an event a sampled id, the last with the choice's
    finish reason and the stop it matched."""
    calls = []

    def network(request):
        calls.append(json.loads(request.content))
        whole = answer(calls[-1])
        if not calls[-1].get("stream"):
            return httpx2.Response(200, json={"id": "fixture", "model": "tiny", "created": 1,
                                              "object": "text_completion", "choices": [whole]})
        logprobs, last = whole["logprobs"], len(whole["token_ids"]) - 1
        events = [{"id": "fixture", "model": "tiny", "created": 1, "object": "text_completion", "choices": [{
            "index": 0, "text": "", "token_ids": [token],
            "finish_reason": whole["finish_reason"] if index == last else None,
            **({"matched_stop": whole["matched_stop"]} if index == last and "matched_stop" in whole else {}),
            "logprobs": {"tokens": [logprobs["tokens"][index]], "top_logprobs": [], "text_offset": [-1],
                         "token_logprobs": [logprobs["token_logprobs"][index]]}}]}
            for index, token in enumerate(whole["token_ids"])]
        return httpx2.Response(200, text="".join(f"data: {json.dumps(event)}\n\n" for event in events)
                               + "data: [DONE]\n\n", headers={"Content-Type": "text/event-stream"})

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
    if provider == "vllm":
        assert sent["stop_token_ids"] == [EOS] and not sent.get("stream")
    else:
        # SGLang draws as the policy without its stops, streamed, and Dew cuts it at the first.
        assert "stop_token_ids" not in sent and sent["ignore_eos"] is True and sent["stream"] is True
    assert sent["logprobs"] == 0 and sent["temperature"] == 1.0 and sent["top_k"] == -1
    returned = {"vllm": "return_tokens_as_token_ids", "sglang": "return_token_ids"}
    assert (
        sent[returned[provider]] is True
        and returned[{"vllm": "sglang", "sglang": "vllm"}[provider]] not in sent
    )


CAPTURED = Path(__file__).parent / "fixtures/gateway"


def replayed(record, drawn):
    """A rollout server for `drawn`, one of `record`'s captured draws, whose transport answers each
    request with the engine's captured answer once Dew has sent the request it sent the engine."""
    exchanges = iter(drawn["exchanges"])

    def network(request):
        exchange = next(exchanges)
        assert json.loads(request.content) == exchange["request"], exchange["path"]
        if "stream" in exchange:
            return httpx2.Response(exchange["status"], text=exchange["stream"],
                                   headers={"Content-Type": "text/event-stream"})
        return httpx2.Response(exchange["status"], json=exchange["response"])

    client = openai.OpenAI(base_url="http://fixture/v1", api_key="fixture", max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(network)))
    server = OpenAIRolloutServer(OpenAICompletion(record["model"], client, provider=record["engine"]),
                                 Sampling(temperature=1.0, eos_token_ids=drawn["eos"]), pushed)
    return server, exchanges


def wire_log_probs(exchanges) -> list[float]:
    """The sampled ids' log-probabilities an engine sent over `exchanges`, in the order it sent them:
    a response's whole list, or each streamed event's in turn."""
    probabilities = []
    for exchange in exchanges:
        if "stream" in exchange:
            events = [json.loads(line.removeprefix("data: ")) for line in exchange["stream"].splitlines()
                      if line.startswith("data: {")]
        else:
            events = [exchange["response"]]
        for event in events:
            for choice in event["choices"]:
                probabilities += (choice.get("logprobs") or {}).get("token_logprobs") or []
    return probabilities


class Streamed(httpx2.SyncByteStream):
    """An event stream that records how much of it was read and whether it was closed."""

    def __init__(self, events: list[dict]):
        self.events, self.sent, self.closed = events, 0, False

    def __iter__(self):
        for event in self.events:
            self.sent += 1
            yield f"data: {json.dumps(event)}\n\n".encode()

    def close(self) -> None:
        self.closed = True


def sglang_event(ids, logprobs, reason=None) -> dict:
    """One streamed SGLang completion event, its ids and log-probabilities sliced as it slices them."""
    return {"id": "fixture", "model": "tiny", "created": 1, "object": "text_completion", "choices": [{
        "index": 0, "text": "", "token_ids": ids, "finish_reason": reason,
        "logprobs": {"tokens": [""] * len(logprobs), "token_logprobs": logprobs, "top_logprobs": [],
                     "text_offset": [-1] * len(logprobs)}}]}


def streamed_server(stream: Streamed):
    """An SGLang rollout server under a policy that stops at EOS, whose engine streams `stream`."""
    transport = httpx2.MockTransport(lambda request: httpx2.Response(
        200, stream=stream, headers={"Content-Type": "text/event-stream"}))
    client = openai.OpenAI(base_url="http://fixture/v1", api_key="fixture", max_retries=0,
                           http_client=httpx2.Client(transport=transport))
    return OpenAIRolloutServer(OpenAICompletion("tiny", client, provider="sglang"),
                               Sampling(eos_token_ids=EOS), pushed)


def test_an_sglang_stream_is_cut_at_the_stop_however_its_events_split_ids_and_likelihoods():
    """SGLang's events slice a draw's ids and log-probabilities apart: a stop arrives with ids after it
    and its log-probability an event later, under no id. The draw ends at the stop with each id's own
    likelihood, read by position, and the stream is closed there with its tail unread. An error event
    mid-stream (SGLang's abort) or a dropped connection is an infrastructure failure, the stream closed
    all the same."""
    stream = Streamed([sglang_event([3, 5], [-.5]), sglang_event([EOS, 11, 12], [-1.]),
                       sglang_event([], [-.25, -2., -3.]), sglang_event([13], [-4.]),
                       sglang_event([14], [-5.], "length")])
    server = streamed_server(stream)
    try:
        draw = server.submit([1, 2], 8, key=0).result()
    finally:
        server.close()
    assert draw.tokens == (3, 5, EOS) and draw.behavior_log_probs == (-.5, -1., -.25) and draw.terminated
    assert stream.closed and stream.sent < len(stream.events)

    from dew.inference import InfrastructureError

    failing = Streamed([sglang_event([3], [-.5]), {"error": {"message": "aborted", "type": "abort"}},
                        sglang_event([EOS], [-1.])])
    server = streamed_server(failing)
    try:
        with pytest.raises(InfrastructureError):
            server.submit([1, 2], 8, key=0).result()
    finally:
        server.close()
    assert failing.closed

    # A connection lost mid-stream is the engine's failure too, which openai 3.14 and later raise as
    # an API connection error.
    reset = Reset([sglang_event([3], [-.5]), sglang_event([5], [-1.])])
    server = streamed_server(reset)
    try:
        with pytest.raises(InfrastructureError):
            server.submit([1, 2], 8, key=0).result()
    finally:
        server.close()
    assert reset.closed


class Reset(Streamed):
    """An event stream whose connection drops after its first event."""

    def __iter__(self):
        for event in self.events[:1]:
            self.sent += 1
            yield f"data: {json.dumps(event)}\n\n".encode()
        raise httpx2.ReadError("connection reset by peer")


@pytest.mark.parametrize("provider", ["vllm", "sglang"])
def test_a_draw_ends_on_the_policys_stops_as_the_engine_drew_it(provider):
    """Draws Dew's rollout server made on vLLM 0.30.0 and SGLang 0.5.20 serving Qwen2.5-0.5B-Instruct
    (tools/gateway_capture.py), replayed request by request: Dew sends each engine the request it
    sent it then, and reads its answer as it did, SGLang's as the events it streamed. A policy with
    no stop runs to its budget past the model's own EOS ids on both engines, and a policy with one
    ends on it alone, drawn past the model's EOS ids where the policy names one of them or another
    id: vLLM stops on the policy's ids by itself, and SGLang, which would also end on its model's
    EOS, draws as the policy without stops while Dew cuts its stream at the policy's first."""
    record = json.loads((CAPTURED / f"{provider}_stops.json").read_text())
    native = set(record["native_eos"])
    unstopped, at_eos, past_eos = record["draws"]
    assert unstopped["eos"] is None and at_eos["eos"] in native and past_eos["eos"] not in native
    for drawn in record["draws"]:
        server, exchanges = replayed(record, drawn)
        try:
            draw = server.submit(record["prompt"], record["budget"], key=drawn["seed"]).result()
        finally:
            server.close()
        assert next(exchanges, None) is None
        assert (list(draw.tokens), draw.terminated) == (drawn["tokens"], drawn["terminated"])
        # The behavior likelihoods are the engine's own, in the order it sent them, up to the stop.
        assert list(draw.behavior_log_probs) == wire_log_probs(drawn["exchanges"])[:len(draw.tokens)]
        stops = () if drawn["eos"] is None else (drawn["eos"],)
        assert draw.terminated == bool(stops) and not any(token in stops for token in draw.tokens[:-1])
        assert draw.terminated or len(draw.tokens) == record["budget"]
    assert any(token in native for token in unstopped["tokens"][:-1])
    for stopped in (at_eos, past_eos):
        assert stopped["tokens"][-1] == stopped["eos"]
        assert any(token in native - {stopped["eos"]} for token in stopped["tokens"][:-1])


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


@pytest.mark.parametrize("status", [429, 503])
def test_an_engine_that_cannot_serve_fails_the_draw_with_an_infrastructure_error(status):
    """A 429 or 5xx, once the client's own retries are spent, is the engine's boundary failing,
    which a session source retries. An answer that breaks the draw's contract, a `stop` that
    ends on no EOS, is a broken engine and fails the draw with ValueError instead."""
    from dew.inference import InfrastructureError

    def refusing(request):
        return httpx2.Response(status, json={"error": {"message": "the engine is overloaded"}})

    client = openai.OpenAI(base_url="http://fixture/v1", api_key="fixture", max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(refusing)))
    broken, _ = engine("vllm", lambda _: choice("vllm", [3, 5], [-.5, -1.], "stop"))
    servers = [OpenAIRolloutServer(completion, Sampling(eos_token_ids=EOS), pushed)
               for completion in (OpenAICompletion("tiny", client, provider="vllm"), broken)]
    try:
        with pytest.raises(InfrastructureError) as caught:
            servers[0].submit([1, 2], 4, key=0).result(timeout=30)
        with pytest.raises(ValueError):
            servers[1].submit([1, 2], 4, key=0).result(timeout=30)
    finally:
        for server in servers:
            server.close()
    assert isinstance(caught.value.__cause__, openai.APIStatusError)


@pytest.mark.parametrize("provider", ["vllm", "sglang"])
def test_a_draw_a_stop_string_ends_is_cut_at_the_id_completing_it(provider):
    """Both engines take the policy's stop strings as the request's `stop`, match them on the text they
    decode, and say which one matched (vLLM's `stop_reason`, SGLang's `matched_stop`). vLLM keeps a
    speculative block's ids past the one completing the string (vllm/v1/engine/detokenizer.py:118-131
    at v0.30.0), so the draw ends at the first id whose text, by the policy's tokenizer, completes it:
    terminated, with the likelihoods of the ids it keeps. Without the tokenizer that id is unknown."""
    pieces = {3: "Hel", 5: "lo", 6: "!", 7: " How", 8: " are"}

    def decode(ids):
        return "".join(pieces[token] for token in ids)

    drawn = [3, 5, 6, 7, 8] if provider == "vllm" else [3, 5, 6]
    matched = {"stop_reason": "lo!"} if provider == "vllm" else {"matched_stop": "lo!"}
    completion, calls = engine(provider, lambda _: {
        **choice(provider, drawn, [-.5, -1., -.25, -2., -3.][:len(drawn)], "stop"), **matched})
    sampling = Sampling(temperature=1.0, eos_token_ids=EOS, stop=("lo!",))
    with pytest.raises(ValueError):
        OpenAIRolloutServer(completion, sampling, pushed)
    server = OpenAIRolloutServer(completion, sampling, pushed, decode=decode)
    try:
        draw = server.submit([1, 2], 8, key=11).result()
    finally:
        server.close()
    assert (draw.tokens, draw.behavior_log_probs, draw.terminated) == ((3, 5, 6), (-.5, -1., -.25), True)
    assert calls[0]["stop"] == ["lo!"]


def test_a_stop_strings_id_is_found_where_a_longer_prefix_does_not_extend_a_shorter_ones_text():
    """A tokenizer's text for a longer prefix need not extend a shorter one's: Transformers' cleanup
    writes "a" "." as "a." where "a" " " alone is "a ". A stop string the second id completes ends the
    draw there, though the whole draw's text no longer holds it, and ids that decode to nothing may sit
    inside a string; ids that don't decode to the string the engine named are refused."""
    pieces = {3: "a", 4: " ", 5: ".", 6: "", 7: "b"}

    def decode(ids):
        return "".join(pieces[token] for token in ids).replace(" .", ".")

    # Ids that decode to nothing (a special token, skipped) may sit inside the string they complete.
    for drawn, stop, kept in (([3, 4, 5], "a ", (3, 4)), ([3, 4, 5, 3, 4], "a ", (3, 4)),
                              ([3, *[6] * 20, 7, 5], "ab", (3, *[6] * 20, 7)), ([3, 5, 5], "a ", None)):
        completion, _ = engine("vllm", lambda _, drawn=drawn, stop=stop: {
            **choice("vllm", drawn, [-.5] * len(drawn), "stop"), "stop_reason": stop})
        server = OpenAIRolloutServer(completion, Sampling(temperature=1.0, eos_token_ids=EOS, stop=(stop,)),
                                     pushed, decode=decode)
        try:
            pending = server.submit([1, 2], 8, key=11)
            if kept is None:
                with pytest.raises(ValueError):
                    pending.result()
            else:
                assert pending.result().tokens == kept
        finally:
            server.close()


def test_sglang_ids_are_read_without_likelihoods():
    completion, _ = engine("sglang", lambda _: {"index": 0, "text": "", "finish_reason": "length",
                                                "logprobs": None, "token_ids": [3, 5]})
    result = completion([[1, 2]], 2, extra_body={"return_token_ids": True})
    assert result.tokens == ((3, 5),) and result.log_probs == (None,)


ORDER = itertools.count()
"""One clock for the engines' requests and the recorded collectives, so their order can be compared."""


class Engine(BaseHTTPRequestHandler):
    """Records every request's path, query and body and answers like the engine it stands for.

    Like vLLM v0.30.0, `/pause` and `/resume` answer a status,
    `/reset_prefix_cache` answers 200 with `{"success": true}`, and
    `/update_weight_version` answers `{"success": true}`. A replica of one
    worker answers `/get_world_size` with 1, and the NCCL weight-transfer
    calls with success. Like SGLang v0.5.20, a weight update answers
    `success` in its body and flushes the radix cache unless the request says
    otherwise. Every replica is its own server, so each keeps its own record,
    with each request's tick of `ORDER` in `order`, and a path in its
    `refuses` answers 500, as a failed engine call does.
    """

    def answer(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def recorded(self, body):
        path, _, query = self.path.partition("?")
        self.server.seen.append((path, query, body))
        self.server.order.append(next(ORDER))
        return path

    def do_GET(self):
        if self.recorded(None) == "/get_world_size":
            return self.answer(200, {"world_size": 1})
        return self.answer(404, {"error": "unknown"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        body = json.loads(raw) if raw else None
        path = self.recorded(body)
        if path in self.server.refuses:
            return self.answer(500, {"error": f"{path} failed"})
        if path in ("/pause", "/resume"):
            return self.answer(200, {"status": "paused" if path == "/pause" else "resumed"})
        if path == "/collective_rpc":
            return self.answer(200, [None])
        if path == "/update_weight_version":
            return self.answer(200, {"success": True, "new_version": body["new_version"]})
        if path in ("/reset_prefix_cache", "/init_weight_transfer_engine", "/start_weight_update",
                    "/update_weights", "/finish_weight_update"):
            return self.answer(200, {"success": True})
        self.server.flushed.append(body.get("flush_cache", True))
        self.answer(
            200, {"success": True, "message": "Succeeded to update model weights.", "num_paused_requests": 0}
        )
        return None

    def log_message(self, *args):
        pass


@pytest.fixture
def replicas():
    """Start local engine stand-ins on demand, each with `url`, `seen`, `order`, `flushed` and `refuses`."""
    started = []

    def start(count):
        for _ in range(count):
            server = HTTPServer(("127.0.0.1", 0), Engine)
            server.seen, server.order, server.flushed, server.refuses = [], [], [], ()
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


@pytest.mark.parametrize("refused", ["/collective_rpc", "/reset_prefix_cache", "/update_weight_version"])
def test_a_vllm_replica_that_fails_a_step_holds_every_replica_paused(refused, tmp_path, replicas):
    """One replica that fails the reload, the cache reset or the version stamp
    stops the push after that step on every replica: no replica goes on to a
    later step or resumes, so none serves weights another may not hold, and
    the push raises."""
    source = Pretrained.load(FIXTURE, dtype="float32")
    engines = replicas(3)
    engines[1].refuses = (refused,)
    push = SafetensorsReload(source, tmp_path / "served", tuple(server.url for server in engines), "vllm")
    with pytest.raises(RuntimeError):
        push(doubled(source), 5)
    steps = ["/pause", "/collective_rpc", "/reset_prefix_cache", "/update_weight_version"]
    for server in engines:
        assert paths(server) == steps[:steps.index(refused) + 1]


class UniqueId(ctypes.Structure):
    """`ncclUniqueId` (nccl.h)."""

    _fields_: ClassVar = [("internal", ctypes.c_byte * 128)]


class Worker:
    """A vLLM worker's side of the weight-transfer group on CUDA device `ordinal`, as vLLM's NCCL
    engine runs it (checked against v0.30.0): a blocking communicator at the rank its join names,
    which matches the sender's one-float32 all-reduce as it opens and receives each listed tensor
    by broadcast from rank 0 into device memory of its own, kept by name."""

    def __init__(self, library, ordinal):
        from dew.devices import primary_context

        self.nccl = ctypes.CDLL(str(library))
        self.cuda, self.context, _ = primary_context(ordinal)
        collective = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_void_p, ctypes.c_void_p]
        self.nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId,
                                               ctypes.c_int]
        self.nccl.ncclAllReduce.argtypes = collective
        self.nccl.ncclBroadcast.argtypes = collective
        self.nccl.ncclCommAbort.argtypes = [ctypes.c_void_p]
        self.cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self.cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.comm = ctypes.c_void_p()
        self.received = {}

    @staticmethod
    def checked(status):
        if status != 0:
            raise RuntimeError(f"NCCL or CUDA error {status}")

    def receive(self, shape, dtype, collective):
        """A tensor of `shape` and `dtype` that `collective` writes into device memory, on the host."""
        dtype = np.dtype(jnp.dtype(dtype))
        size, pointer = math.prod(shape) * dtype.itemsize, ctypes.c_uint64()
        self.checked(self.cuda.cuCtxSetCurrent(self.context))
        self.checked(self.cuda.cuMemAlloc_v2(ctypes.byref(pointer), max(size, 1)))
        try:
            self.checked(collective(pointer.value, math.prod(shape), _DTYPES[dtype.name]))
            self.checked(self.cuda.cuStreamSynchronize(None))
            host = np.empty(shape, dtype)
            self.checked(self.cuda.cuMemcpyDtoH_v2(host.ctypes.data, pointer.value, size))
        finally:
            self.checked(self.cuda.cuMemFree_v2(pointer.value))
        return host

    def join(self, init):
        uid = UniqueId.from_buffer_copy(base64.b64decode(init["nccl_unique_id_b64"]))
        self.checked(self.cuda.cuCtxSetCurrent(self.context))
        self.checked(self.nccl.ncclCommInitRank(ctypes.byref(self.comm), init["world_size"], uid,
                                                init["rank_offset"]))
        self.receive((1,), "float32", lambda pointer, count, dtype: self.nccl.ncclAllReduce(
            pointer, pointer, count, dtype, 0, self.comm, None))

    def update(self, info):
        for name, dtype, shape in zip(info["names"], info["dtype_names"], info["shapes"], strict=True):
            self.received[name] = self.receive(tuple(shape), dtype, lambda pointer, count, code:
                                               self.nccl.ncclBroadcast(pointer, pointer, count, code, 0,
                                                                       self.comm, None))

    def close(self):
        if self.comm.value is not None:
            self.checked(self.nccl.ncclCommAbort(self.comm))


def nccl_library():
    """The NCCL library JAX's CUDA wheels ship, which the push and a worker both load."""
    import importlib.util

    locations = importlib.util.find_spec("nvidia.nccl").submodule_search_locations
    return Path(next(iter(locations))) / "lib" / "libnccl.so.2"


class NCCLReplica(BaseHTTPRequestHandler):
    """A one-worker vLLM replica's weight-transfer endpoints over `server.worker`, logging each
    request's `ORDER` tick as it arrives and as it is answered, its path, query and body. A replica
    whose `server.refuses` names a path answers it 500 without acting on it."""

    def answer(self, status, body):
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def handle_one(self, body):
        path, _, query = self.path.partition("?")
        arrived = next(ORDER)
        if path in self.server.refuses:
            status, answer = 500, {}
        elif path == "/get_world_size":
            status, answer = 200, {"world_size": 1}
        else:
            if path == "/init_weight_transfer_engine":
                self.server.worker.join(body["init_info"])
            elif path == "/update_weights":
                self.server.worker.update(body["update_info"])
            status = 200
            answer = ({"status": "paused" if path == "/pause" else "resumed"} if path in ("/pause", "/resume")
                      else {"success": True})
        self.server.seen.append((arrived, next(ORDER), path, query, body))
        self.answer(status, answer)

    def do_GET(self):
        self.handle_one(None)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.handle_one(json.loads(raw) if raw else None)

    def log_message(self, *args):
        pass


def nccl_replicas(ordinal, count, refuses=()):
    """`count` replicas, each with a worker of its own on CUDA device `ordinal`."""
    started = []
    for _ in range(count):
        server = ThreadingHTTPServer(("127.0.0.1", 0), NCCLReplica)
        server.seen, server.refuses = [], refuses
        server.worker = Worker(nccl_library(), ordinal)
        server.url = f"http://127.0.0.1:{server.server_port}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
    return started


def test_an_fp8_sources_push_has_a_collective_for_every_tensor_it_publishes():
    """An FP8 source's push publication exports E4M3 weights, which vLLM's NCCL wrapper receives as
    `ncclFloat8e4m3`: every dtype the publication exports has an NCCL code, so no push pauses its
    replicas and then fails on a tensor it cannot send (ledger 1875)."""
    from dew.inference.rollouts import _published

    source = Pretrained.load(FP8, dtype="float32")
    published = _published(source, source.variables, jnp.dtype("bfloat16"), phase="fp8 push gather")
    dtypes = {str(tensor.dtype) for tensor in published.export().values()}
    assert "float8_e4m3fn" in dtypes and dtypes <= _DTYPES.keys(), dtypes


@pytest.mark.gpu(reason="an NCCL group joins two cuda devices")
@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("fixture", [FIXTURE, FP8], ids=["bfloat16", "fp8"])
def test_an_nccl_push_delivers_every_tensor_to_each_replica_on_another_gpu(fixture):
    """A push over real NCCL groups from the trainer's GPU to two one-worker replicas on a second
    GPU, each standing in for vLLM's weight-transfer endpoints with a worker that receives as
    vLLM's does, in a group of its own. Each replica is asked its world size and joins its group
    once, then pauses, starts the update, takes the listing of every tensor, finishes the update
    under the version, resets its prefix cache and resumes. Each group opens with the sender at
    rank 0 of two. The tensors go one after another to both groups, in chunks smaller than the
    policy: every replica receives the whole listing after every replica started the update and
    before any finished it, and each worker holds every listed tensor, the trainer's export of the
    pushed policy, an FP8 source's E4M3 weights among them."""
    engines = nccl_replicas(jax.devices()[1].local_hardware_id, 2)
    source = Pretrained.load(fixture, dtype="float32")
    changed = doubled(source)
    push = NCCLPush(source, tuple(server.url for server in engines), str(nccl_library()), chunk=4096,
                    timeout=120)
    try:
        push(changed, 3)
    finally:
        push.close()
        for server in engines:
            server.shutdown()
            server.worker.close()
    sent = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16) if jnp.issubdtype(leaf.dtype, jnp.floating)
                        else leaf, changed)
    exported = source.export(sent)
    assert len(exported) > 1 and sum(np.asarray(tensor).nbytes for tensor in exported.values()) > 4096
    listings = []
    for server in engines:
        assert [path for _, _, path, _, _ in server.seen] == [
            "/get_world_size", "/init_weight_transfer_engine", "/pause", "/start_weight_update",
            "/update_weights", "/finish_weight_update", "/reset_prefix_cache", "/resume"]
        world, joined, paused, _, update, finished, _, _ = server.seen
        assert world[3] == "include_dp=true" and paused[3] == "mode=wait"
        assert joined[4]["init_info"]["rank_offset"] == 1 and joined[4]["init_info"]["world_size"] == 2
        assert finished[4] == {"weight_version": "3"}
        listings.append(update[4]["update_info"])
        assert sorted(server.worker.received) == sorted(exported)
        for name, tensor in exported.items():
            np.testing.assert_array_equal(server.worker.received[name], np.asarray(tensor))
    assert listings[1] == listings[0] and listings[0]["names"] == sorted(listings[0]["names"])
    started = max(server.seen[3][1] for server in engines)
    received = [(server.seen[4][0], server.seen[4][1]) for server in engines]
    assert all(started < arrived and answered < min(server.seen[5][0] for server in engines)
               for arrived, answered in received)


@pytest.mark.gpu(reason="an NCCL group opens on a cuda device")
def test_a_push_whose_replica_refuses_its_join_aborts_its_group_rather_than_waiting():
    """The sender opens its NCCL group while the replica's HTTP join runs. A replica that refuses
    the join never enters the rendezvous, so the push aborts the group and raises the refusal
    within seconds, where a blocking open waited on that peer for ever. The aborted group is not
    kept: a second push asks the replica to join a new one."""
    from dew.inference.rollouts import InfrastructureError

    [server] = nccl_replicas(jax.devices()[0].local_hardware_id, 1, refuses=("/init_weight_transfer_engine",))
    source = Pretrained.load(FIXTURE, dtype="float32")
    push = NCCLPush(source, (server.url,), str(nccl_library()), timeout=30)
    raised = []

    def attempts():
        for _ in range(2):
            try:
                push(source.variables, 1)
            except RuntimeError as error:
                raised.append(error)

    pushing = threading.Thread(target=attempts, daemon=True)
    pushing.start()
    pushing.join(120)
    server.shutdown()
    assert not pushing.is_alive() and len(raised) == 2

    def causes(error):
        while error is not None:
            yield error
            error = error.__cause__

    assert all(any(isinstance(cause, InfrastructureError) for cause in causes(error)) for error in raised)
    assert [path for _, _, path, _, _ in server.seen] == [
        "/get_world_size", "/init_weight_transfer_engine"] * 2


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
