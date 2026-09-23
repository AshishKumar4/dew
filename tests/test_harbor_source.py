"""Harbor trials behind a recording gateway, as a rollout source.

The gateway's traces are the JSON rllm-model-gateway serves for a vLLM or
SGLang worker; Harbor is a stand-in executable that takes the same `trials
start` arguments and writes a `TrialResult`-shaped `result.json`.
"""

import json
import stat
import sys
import threading
import time

import httpx
import pytest

from dew.objectives.rl.harbor import HARBOR_KEY, Gateway, HarborSource, calls, outcome
from dew.objectives.rl.rollouts import Call, Status, Task


def trace(prompt, sampled, logprobs, reason="stop", *, started=0.0, version=None, **extra):
    return {"prompt_token_ids": prompt, "completion_token_ids": sampled, "logprobs": logprobs,
            "finish_reason": reason, "weight_version": version, "timestamp": started + 1.0, "latency_ms": 1000.0,
            **extra}


def test_calls_follow_submission_order_and_carry_the_gateway_stamp():
    # The first call finished last: stored order is completion order, not submission order.
    late = trace([1, 2, 3, 4, 5], [6, 7], [-.5, -.25], "stop", started=1.0, version=3)
    early = trace([1, 2], [3, 4], [-1., -2.], "tool_calls", started=0.0, version=None)
    early["latency_ms"] = 5000.0
    early["timestamp"] = 5.0
    assert calls([late, early], unstamped=2).calls == (
        Call((1, 2), (3, 4), (-1., -2.), "tool_calls", 2), Call((1, 2, 3, 4, 5), (6, 7), (-.5, -.25), "stop", 3))


def test_sglang_ids_are_read_from_the_raw_response():
    # SGLang's chat route lists ids under sglext, which the gateway keeps only in the raw response.
    recorded = trace([], [], [-.5, -.25], raw_response={"sglext": {"input_ids": [1, 2], "output_ids": [[5, 6]]}})
    assert calls([recorded], unstamped=0).calls == (Call((1, 2), (5, 6), (-.5, -.25), "stop", 0),)


@pytest.mark.parametrize("broken", [
    trace([], [3], [-.5]),  # no ids at all: the engine was not asked for them
    trace([1], [3, 4], [-.5]),  # one likelihood short
    trace([1], [3], [-.5], None),  # no finish reason
])
def test_a_trace_that_cannot_train_is_refused(broken):
    with pytest.raises(ValueError):
        calls([broken], unstamped=0)


# vLLM 0.30.0's reply to a prompt past max-model-len (renderers/params.py), as the gateway records it:
# no ids, no likelihoods, no finish reason, the error body kept in raw_response.
OVERFLOW = {"prompt_token_ids": [], "completion_token_ids": [], "logprobs": None, "finish_reason": None,
            "weight_version": None, "timestamp": 9.0, "latency_ms": 10.0,
            "raw_response": {"error": {"message": "This model's maximum context length is 2048 tokens. However, "
                                                  "you requested 16 output tokens and your prompt contains 4012 "
                                                  "input tokens, for a total of 4028 tokens.",
                                       "type": "BadRequestError", "param": "input_tokens", "code": 400}}}
BROKEN = {**OVERFLOW, "raw_response": {"error": {"message": "EngineCore died", "type": "InternalServerError",
                                                 "code": 500}}}


def test_engine_errors_are_events_not_calls():
    recorded = calls([trace([1, 2], [3], [-.5]), OVERFLOW], unstamped=0)
    assert recorded.calls == (Call((1, 2), (3,), (-.5,), "stop", 0),) and len(recorded.errors) == 1
    finished = result(rewards={"reward": 0})
    # An engine that refused an overflowing prompt truncated the rollout; any other engine error is infra.
    assert outcome(finished, recorded.calls, errors=recorded.errors)[0] is Status.TRUNCATED
    broken = calls([trace([1, 2], [3], [-.5]), BROKEN], unstamped=0)
    assert outcome(finished, broken.calls, errors=broken.errors)[0] is Status.INFRA_ERROR


STOP = Call((1,), (2,), (-.5,), "stop", 0)
CUT = Call((1,), (2,), (-.5,), "length", 0)
ABORTED = Call((1,), (2,), (-.5,), "abort", 0)


def result(exception=None, rewards=None):
    return {"exception_info": {"exception_type": exception, "exception_message": "boom"} if exception else None,
            "verifier_result": {"rewards": rewards} if rewards is not None else None}


@pytest.mark.parametrize(("trial", "records", "exit", "status", "reward"), [
    (result(rewards={"reward": 1}), (STOP,), None, Status.COMPLETED, 1.0),
    (result(rewards={"reward": 0}), (STOP,), None, Status.COMPLETED, 0.0),
    (result("NonZeroAgentExitCodeError", {"reward": 0}), (STOP,), None, Status.AGENT_ERROR, 0.0),
    (result("AgentTimeoutError", {"reward": 1}), (STOP,), None, Status.TRUNCATED, 1.0),
    (result("ContextWindowExceededError"), (STOP,), None, Status.TRUNCATED, None),
    (result(rewards={"reward": 0}), (STOP, CUT), None, Status.TRUNCATED, 0.0),
    (result(rewards={"reward": 0}), (STOP,), "LimitsExceeded", Status.TRUNCATED, 0.0),
    (result(rewards={"reward": 1}), (STOP, ABORTED), None, Status.INFRA_ERROR, 1.0),
    (result(rewards={"reward": 1}), (), None, Status.INFRA_ERROR, 1.0),
    (result("EnvironmentStartTimeoutError"), (), None, Status.INFRA_ERROR, None),
    (result("ApiConnectionClosedError", {"reward": 0}), (STOP,), None, Status.INFRA_ERROR, 0.0),
    (result("RewardFileNotFoundError"), (STOP,), None, Status.INFRA_ERROR, None),
    (result(rewards={"tests": 1, "style": 0}), (STOP,), None, Status.INFRA_ERROR, None),
    # The harness's model client gave up (litellm exception names, recorded by mini-swe-agent as its
    # exit status) and Harbor saw only a nonzero exit: an infrastructure fault, not a scored failure.
    *[(result("NonZeroAgentExitCodeError", {"reward": 0}), (STOP,), name, Status.INFRA_ERROR, 0.0)
      for name in ("APIConnectionError", "APIError", "InternalServerError", "ServiceUnavailableError",
                   "Timeout", "RateLimitError", "BadGatewayError")],
    (result("NonZeroAgentExitCodeError", {"reward": 0}), (STOP,), "ContextWindowExceededError",
     Status.TRUNCATED, 0.0),
    (result("NonZeroAgentExitCodeError", {"reward": 0}), (STOP,), "RepeatedFormatError", Status.AGENT_ERROR, 0.0),
])
def test_a_trial_is_scored_masked_or_retried_by_how_it_ended(trial, records, exit, status, reward):
    classified = outcome(trial, records, harness_exit=exit)
    assert classified[:2] == (status, reward)


HARBOR = '''\
import json, os, sys, time
arguments = sys.argv[1:]
def value(flag):
    return arguments[arguments.index(flag) + 1]
ae = [arguments[index + 1] for index, flag in enumerate(arguments) if flag == "--ae"]
agent = dict(entry.split("=", 1) for entry in ae)
trial = os.path.join(value("--trials-dir"), value("--trial-name"))
os.makedirs(trial)
with open(os.path.join(trial, "seen.json"), "w") as seen:
    json.dump({"arguments": arguments, "agent": agent}, seen)
if "stubborn" in value("-p"):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
if "slow" in value("-p") or "stubborn" in value("-p"):
    time.sleep(60)
with open(os.path.join(trial, "result.json"), "w") as out:
    json.dump({"exception_info": None, "verifier_result": {"rewards": {"reward": 1}}}, out)
'''


@pytest.fixture
def harbor(tmp_path):
    executable = tmp_path / "harbor"
    executable.write_text(f"#!{sys.executable}\n" + HARBOR)
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    return executable


def fake_gateway():
    """A gateway whose every session recorded one call sampling [10 + sample]."""
    asked = []

    def network(request):
        asked.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": 1})
        session = request.url.path.removeprefix("/sessions/").removesuffix("/traces")
        sample = int(session.rsplit(":", 1)[1])
        return httpx.Response(200, json=[trace([1, 2], [10 + sample], [-.5], version=7)])

    return Gateway("http://gateway", sandbox_url="http://172.17.0.1:9090",
                   client=httpx.Client(transport=httpx.MockTransport(network))), asked


def test_each_sample_is_its_own_trial_and_gateway_session(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, asked = fake_gateway()
    source = HarborSource(gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials",
                          environment={"MSWEA_API_KEY": "none"}, arguments=("-e", "docker"))
    try:
        futures = source.submit(Task("hello", {HARBOR_KEY: str(tmp_path / "task")}), 2, version=5)
        rollouts = [future.result(timeout=60) for future in futures]
    finally:
        source.close()
    assert [rollout.sample for rollout in rollouts] == [0, 1]
    assert {rollout.group for rollout in rollouts} == {rollouts[0].group}
    for rollout in rollouts:
        assert rollout.status is Status.COMPLETED and rollout.reward == 1.0 and rollout.task == "hello"
        # The session's own call, stamped by the gateway, came back to its own sample.
        assert rollout.calls == (Call((1, 2), (10 + rollout.sample,), (-.5,), "stop", 7),)
        seen = json.loads((tmp_path / "trials" / f"dew-{rollout.group}-{rollout.sample}" / "seen.json").read_text())
        session = f"hello:{rollout.group}:{rollout.sample}"
        assert seen["agent"] == {"MSWEA_API_KEY": "none",
                                 "OPENAI_BASE_URL": f"http://172.17.0.1:9090/sessions/{session}/v1"}
        assert seen["arguments"][:6] == ["trials", "start", "-a", "mini-swe-agent", "-m", "hosted_vllm/policy"]
        assert ("DELETE", f"/sessions/{session}") in asked


def test_a_cancelled_trial_is_interrupted_and_resolves_cancelled(tmp_path, harbor):
    (tmp_path / "slow").mkdir()
    gateway, asked = fake_gateway()
    source = HarborSource(gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials")
    try:
        (future,) = source.submit(Task("slow", {HARBOR_KEY: str(tmp_path / "slow")}), 1, version=0)
        deadline = time.monotonic() + 30
        while not list((tmp_path / "trials").glob("*/seen.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        began = time.monotonic()
        source.cancel([future])
        rollout = future.result(timeout=30)
    finally:
        source.close()
    assert rollout.status is Status.CANCELLED and time.monotonic() - began < 30
    # A cancelled session's traces leave the gateway too, not only a scored one's.
    assert ("DELETE", f"/sessions/slow:{rollout.group}:0") in asked


def test_a_harbor_that_ignores_the_interrupt_is_terminated_after_the_grace(tmp_path, harbor):
    (tmp_path / "stubborn").mkdir()
    gateway, _ = fake_gateway()
    source = HarborSource(gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials", grace=1.0)
    try:
        (future,) = source.submit(Task("stubborn", {HARBOR_KEY: str(tmp_path / "stubborn")}), 1, version=0)
        deadline = time.monotonic() + 30
        while not list((tmp_path / "trials").glob("*/seen.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        source.cancel([future])
        assert future.result(timeout=15).status is Status.CANCELLED
        # Cancelling a resolved future again is a no-op, not a signal to a reaped process group.
        source.cancel([future])
    finally:
        source.close()


def test_close_resolves_queued_trials_without_launching_them(tmp_path, harbor):
    (tmp_path / "slow").mkdir()
    gateway, _ = fake_gateway()
    source = HarborSource(gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials", workers=1)
    futures = source.submit(Task("slow", {HARBOR_KEY: str(tmp_path / "slow")}), 2, version=0)
    deadline = time.monotonic() + 30
    while not list((tmp_path / "trials").glob("*/seen.json")) and time.monotonic() < deadline:
        time.sleep(0.05)
    began = time.monotonic()
    closer = threading.Thread(target=source.close)
    closer.start()
    closer.join(20)
    assert not closer.is_alive() and time.monotonic() - began < 20
    assert [future.result(timeout=1).status for future in futures] == [Status.CANCELLED, Status.CANCELLED]
    # The queued sample never started a trial.
    assert len(list((tmp_path / "trials").glob("*/seen.json"))) == 1


def test_a_task_without_a_harbor_directory_is_refused(tmp_path, harbor):
    source = HarborSource(fake_gateway()[0], harbor=harbor, model="m/p", trials=tmp_path)
    try:
        with pytest.raises(ValueError, match="Harbor task directory"):
            source.submit(Task("nothing"), 1, version=0)
    finally:
        source.close()
