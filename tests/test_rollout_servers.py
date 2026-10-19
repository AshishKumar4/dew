"""Rollout servers over a remote engine: token requests, reported likelihoods, weight reloads.

The vLLM and SGLang wire formats are mocked at the HTTP transport, so the
checks read the request the SDK actually sends and the response each engine
documents: token-id prompts, seeds, EOS controls, the field that returns the
sampled ids (vLLM's `return_tokens_as_token_ids` renders them into the
logprob tokens, SGLang's `return_token_ids` lists them on the choice) and one
reported log-probability per sampled token. The safetensors reload writes a
real Qwen2 export that `Pretrained.load` reads back, and posts the reload
calls to a real local HTTP endpoint in order. The NCCL push posts its calls
to the same endpoints and drives a recording stand-in for its NCCL library.
"""

import base64
import itertools
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inference import NCCLPush, OpenAICompletion, OpenAIRolloutServer, SafetensorsReload
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


NATIVE = 2
"""The engine model's own EOS, which no policy here names."""


def vllm_stopping(request):
    """vLLM's answer to `request` over the draw 3, NATIVE, 5, EOS, 6, ...: it ends on its model's EOS
    unless the request ignores it, on a requested stop id either way, else at its budget."""
    stops = set(request.get("stop_token_ids") or ()) | (set() if request.get("ignore_eos") else {NATIVE})
    ids = []
    for token in [3, NATIVE, 5, EOS, 6, 6, 6, 6][:request["max_tokens"]]:
        ids.append(token)
        if token in stops:
            return choice("vllm", ids, [-.5] * len(ids), "stop")
    return choice("vllm", ids, [-.5] * len(ids), "length")


@pytest.mark.parametrize("eos", [EOS, None])
def test_a_vllm_draw_ends_on_the_policys_stops_alone(eos):
    """A policy's stops are the only ids that end a vLLM draw: its model's own EOS, which the policy
    does not name, is drawn past, and a policy with no stop runs to its budget."""
    completion, _ = engine("vllm", vllm_stopping)
    server = OpenAIRolloutServer(completion, Sampling(eos_token_ids=eos), pushed)
    try:
        draw = server.submit([1], 6, key=0).result(timeout=30)
    finally:
        server.close()
    assert draw.tokens == ((3, NATIVE, 5, EOS) if eos is not None else (3, NATIVE, 5, EOS, 6, 6))
    assert draw.terminated == (eos is not None)


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


class Collectives:
    """A recording stand-in for the NCCL library a push drives, which `NCCLPush` builds only when it
    holds none: each open makes the next numbered group, and each call is logged with its `ORDER`
    tick, group, element count and dtype, moving no data."""

    def __init__(self):
        self.calls, self.groups = [], 0

    def unique_id(self):
        return SimpleNamespace(internal=bytes(128))

    def open(self, uid, rank, world):
        self.groups += 1
        self.calls.append((next(ORDER), "open", self.groups, rank, world))
        return self.groups

    def all_reduce(self, comm, pointer, count, dtype):
        self.calls.append((next(ORDER), "all_reduce", comm, count, dtype))

    def broadcast(self, comm, pointer, count, dtype):
        self.calls.append((next(ORDER), "broadcast", comm, count, dtype))

    def settled(self, comms, failed):
        self.calls.append((next(ORDER), "settled", tuple(comms), None, None))

    def abort(self, comm):
        self.calls.append((next(ORDER), "abort", comm, None, None))


def test_an_nccl_push_makes_the_calls_its_replicas_and_groups_expect(replicas):
    """Two one-worker vLLM replicas behind a recording NCCL library. Each
    replica is asked its world size and joins its group once, then pauses,
    starts the update, takes the listing of every tensor, finishes the update
    under the version, resets its prefix cache and resumes. Each group opens
    with the sender at rank 0 of two and warms up with one float32, then
    receives every listed tensor in the listing's order. Every broadcast
    comes after every replica started the update and before any finished it,
    and no group is aborted."""
    source = Pretrained.load(FIXTURE, dtype="float32")
    engines = replicas(2)
    collectives = Collectives()
    push = NCCLPush(source, tuple(server.url for server in engines), library=__file__)
    push._library = collectives
    push(doubled(source), 3)

    listings = []
    for server in engines:
        assert paths(server) == ["/get_world_size", "/init_weight_transfer_engine", "/pause",
                                 "/start_weight_update", "/update_weights", "/finish_weight_update",
                                 "/reset_prefix_cache", "/resume"]
        world, joined, paused, _, update, finished, _, _ = server.seen
        assert world[1] == "include_dp=true" and paused[1] == "mode=wait"
        assert joined[2] == {"init_info": {"nccl_unique_id_b64": base64.b64encode(bytes(128)).decode(),
                                           "rank_offset": 1, "world_size": 2, "packed": False}}
        assert finished[2] == {"weight_version": "3"}
        listings.append(update[2]["update_info"])
    listing = listings[0]
    assert listings[1] == listing and listing["names"] == sorted(listing["names"])
    assert [call[1:] for call in collectives.calls if call[1] in ("open", "all_reduce")] == [
        ("open", 1, 0, 2), ("all_reduce", 1, 1, "float32"),
        ("open", 2, 0, 2), ("all_reduce", 2, 1, "float32")]
    assert [call[2:] for call in collectives.calls if call[1] == "broadcast"] == [
        (group, math.prod(shape), dtype)
        for shape, dtype in zip(listing["shapes"], listing["dtype_names"], strict=True) for group in (1, 2)]
    sent = [call[0] for call in collectives.calls if call[1] == "broadcast"]
    assert all(server.order[3] < min(sent) and max(sent) < server.order[5] for server in engines)
    assert not [call for call in collectives.calls if call[1] == "abort"]


@pytest.mark.gpu(reason="an NCCL group opens on a cuda device")
def test_a_push_whose_replica_refuses_its_join_aborts_its_group_rather_than_waiting():
    """The sender opens its NCCL group while the replica's HTTP join runs. A replica that refuses
    the join never enters the rendezvous, so the push aborts the group and raises within seconds,
    where a blocking open waited on that peer for ever."""
    import http.server
    import importlib.util

    locations = importlib.util.find_spec("nvidia.nccl").submodule_search_locations
    library = Path(next(iter(locations))) / "lib" / "libnccl.so.2"

    class Replica(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"world_size": 1}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Replica)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    source = Pretrained.load(FIXTURE, dtype="float32")
    push = NCCLPush(source, (f"http://127.0.0.1:{server.server_port}",), str(library), timeout=30)
    raised = []

    def attempt():
        try:
            push(source.variables, 1)
        except BaseException as error:
            raised.append(error)

    pushing = threading.Thread(target=attempt, daemon=True)
    pushing.start()
    pushing.join(120)
    server.shutdown()
    assert not pushing.is_alive() and len(raised) == 1 and isinstance(raised[0], RuntimeError)


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
