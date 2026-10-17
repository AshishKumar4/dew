"""Harbor trials behind a recording gateway, as a rollout source.

The gateway's traces are the JSON rllm-model-gateway serves for a vLLM or
SGLang worker; Harbor is a stand-in executable that takes the same `trials
start` arguments and writes a `TrialResult`-shaped `result.json`. A trial
that hangs past `trial_timeout` is interrupted and retried by a real
`RolloutScheduler` under a `Trainer`, and a child Harbor leaves in its
process group, which ignores SIGINT and SIGTERM and holds its output open,
is killed with the group, at the deadline and at exit.
"""

import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import jax
import numpy as np
import pytest

from dew.interop.harbor import HARBOR_KEY, Gateway, HarborSource, calls, outcome
from dew.objectives.rl.sessions import Call, Status, Task

KEY = jax.random.key(0)


def trace(prompt, sampled, logprobs, reason="stop", *, started=0.0, version=None, **extra):
    return {
        "prompt_token_ids": prompt,
        "completion_token_ids": sampled,
        "logprobs": logprobs,
        "finish_reason": reason,
        "weight_version": version,
        "timestamp": started + 1.0,
        "latency_ms": 1000.0,
        **extra,
    }


def test_calls_follow_submission_order_and_carry_the_gateway_stamp():
    # The first call finished last: stored order is completion order, not submission order.
    late = trace([1, 2, 3, 4, 5], [6, 7], [-.5, -.25], "stop", started=1.0, version=3, key=KEY)
    early = trace([1, 2], [3, 4], [-1., -2.], "tool_calls", started=0.0, version=None)
    early["latency_ms"] = 5000.0
    early["timestamp"] = 5.0
    assert calls([late, early], unstamped=2).calls == (
        Call((1, 2), (3, 4), (-1.0, -2.0), "tool_calls", 2),
        Call((1, 2, 3, 4, 5), (6, 7), (-0.5, -0.25), "stop", 3),
    )


def test_a_trace_carries_vllms_routed_experts_into_its_call():
    """vLLM's chat and completions choices carry `routed_experts`; the gateway
    keeps the raw response, so the call replays what the engine routed."""
    import base64
    import io

    import numpy as np

    record = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    buffer = io.BytesIO()
    np.save(buffer, record, allow_pickle=False)
    raw = {"choices": [{"routed_experts": base64.b64encode(buffer.getvalue()).decode()}]}
    (call,) = calls([trace([1, 2, 3], [4, 5], [-.5, -.25], raw_response=raw)], unstamped=0).calls
    assert call.routed_experts is not None
    np.testing.assert_array_equal(call.routed_experts, record)


FIXTURES = Path(__file__).parent / "fixtures/gateway"


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_a_live_gateway_session_is_the_engine_ids_and_likelihoods(engine):
    # A three-turn tool session captured live by tools/gateway_capture.py (Colab L4, Qwen2.5-0.5B-Instruct).
    # SGLang 0.5.20 lists the sampled ids as choices[0].response_token_ids,
    # which the gateway does not extract.
    traces = json.loads((FIXTURES / f"{engine}_session_traces.json").read_text())
    session = calls(traces, unstamped=0)
    assert len(session.calls) == 3 and not session.errors
    for call, recorded in zip(session.calls, traces, strict=True):
        choice = recorded["raw_response"]["choices"][0]
        assert call.sampled_ids == tuple(choice["response_token_ids"] if engine == "sglang"
                                         else recorded["completion_token_ids"])
        assert call.behavior_log_probs == tuple(entry["logprob"] for entry in choice["logprobs"]["content"])


def overflow(engine):
    return json.loads((FIXTURES / f"{engine}_overflow_trace.json").read_text())


def refusal(message, code=400):
    return {**trace([], [], None, None), "raw_response": {"error": {"message": message, "code": code}}}


@pytest.mark.parametrize(
    ("refused", "status"),
    [
        # Live refusals of a prompt past the context length, in each engine's own shape.
        (overflow("sglang"), Status.TRUNCATED),
        (overflow("vllm"), Status.TRUNCATED),
        # vLLM 0.30.0's input processor (_validate_prompt_len), reached by token-id prompts.
        (
            refusal(
                "The decoder prompt (length 4200) is longer than the maximum model length of 4096. "
                "Make sure that "
                "`max_model_len` is no smaller than the number of text tokens."
            ),
            Status.TRUNCATED,
        ),
        (
            refusal(
                "The decoder prompt (length 4096) plus the number of requested output tokens "
                "(at least 1) is longer "
                "than the maximum model length of 4096."
            ),
            Status.TRUNCATED,
        ),
        (refusal("EngineCore died", 500), Status.INFRA_ERROR),
    ],
)
def test_an_engine_error_is_an_event_that_decides_the_session(refused, status):
    recorded = calls(
        [trace([1, 2], [3], [-0.5]), {**refused, "timestamp": 9.0, "latency_ms": 10.0}], unstamped=0
    )
    assert recorded.calls == (Call((1, 2), (3,), (-.5,), "stop", 0),) and len(recorded.errors) == 1
    assert outcome(result(rewards={"reward": 0}), recorded.calls, errors=recorded.errors)[0] is status


def result(exception=None, rewards=None):
    return {
        "exception_info": {"exception_type": exception, "exception_message": "boom"} if exception else None,
        "verifier_result": {"rewards": rewards} if rewards is not None else None,
    }


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
    json.dump({"arguments": arguments, "agent": agent, "pid": os.getpid()}, seen)
if "stubborn" in value("-p"):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
if "slow" in value("-p") or "stubborn" in value("-p"):
    time.sleep(60)
if "lingering" in value("-p"):
    # A child inherits Harbor's output, ignores SIGINT and SIGTERM, then writes its pid. It stays
    # in Harbor's process group, or with `detached` leaves it.
    import subprocess
    subprocess.Popen([sys.executable, "-c", "import json, os, signal, sys, time\\n"
                      "signal.signal(signal.SIGINT, signal.SIG_IGN)\\n"
                      "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
                      "json.dump(os.getpid(), open(sys.argv[1], 'w'))\\n"
                      "time.sleep(60)", os.path.join(trial, "child.json")],
                     start_new_session="detached" in value("-p"))
    if "waits" in value("-p"):
        time.sleep(60)
    sys.exit(1)
if "once" in value("-p"):
    # The first trial of the task directory hangs, with its pid in the marker; the others finish.
    try:
        marker = os.open(os.path.join(value("-p"), "hung"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pass
    else:
        os.write(marker, str(os.getpid()).encode())
        os.close(marker)
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


def fake_gateway(unhealthy_checks=0, unreachable_models=0, version=7):
    """A gateway whose every session recorded one call sampling [10 + sample], stamped `version`.

    Its one worker is dead for the first `unhealthy_checks` health reads, like rllm-model-gateway
    started beside an engine still loading: its health loop marks the worker dead, and until the
    next check revives it every proxied call answers a traceless 500. Before its first probe the
    gateway reports the worker healthy whatever its state; `unreachable_models` makes the first
    proxied `GET /v1/models` answers that 500, like an engine that is not listening yet.
    """
    asked = []
    checks = [0]
    listings = [0]

    def network(request):
        asked.append((request.method, request.url.path))
        if request.url.path == "/health/workers":
            checks[0] += 1
            healthy = int(checks[0] > unhealthy_checks)
            return httpx.Response(200, json={"workers": [{"url": "http://engine", "healthy": bool(healthy)}],
                                             "healthy": healthy, "total": 1})
        if request.url.path == "/v1/models":
            listings[0] += 1
            if listings[0] <= unreachable_models:
                return httpx.Response(500, text="Internal Server Error")
            return httpx.Response(200, json={"object": "list", "data": [{"id": "policy"}]})
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": 1})
        session = request.url.path.removeprefix("/sessions/").removesuffix("/traces")
        sample = int(session.split(":")[2])
        return httpx.Response(200, json=[trace([1, 2], [10 + sample], [-.5], version=version)])

    return Gateway("http://gateway", sandbox_url="http://172.17.0.1:9090",
                   client=httpx.Client(transport=httpx.MockTransport(network))), asked


def test_the_gateway_stamp_is_checked():
    asked = []

    def network(request):
        asked.append(json.loads(request.content))
        return httpx.Response(200, json={"weight_version": 7 if len(asked) == 1 else 6})

    gateway = Gateway("http://gateway", client=httpx.Client(transport=httpx.MockTransport(network)))
    gateway.stamp(7)
    with pytest.raises(RuntimeError):
        gateway.stamp(8)
    assert asked == [{"weight_version": 7}, {"weight_version": 8}]


def test_each_sample_is_its_own_trial_and_gateway_session(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, asked = fake_gateway()
    source = HarborSource(gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials",
                          environment={"MSWEA_API_KEY": "none"}, arguments=("-e", "docker"))
    try:
        futures = source.submit(Task("hello", {HARBOR_KEY: str(tmp_path / "task")}), 2, version=5, key=KEY)
        rollouts = [future.result(timeout=60) for future in futures]
    finally:
        source.close()
    assert [rollout.sample for rollout in rollouts] == [0, 1]
    assert {rollout.group for rollout in rollouts} == {rollouts[0].group}
    for rollout in rollouts:
        assert rollout.status is Status.COMPLETED and rollout.reward == 1.0 and rollout.task == "hello"
        # The session's own call, stamped by the gateway, came back to its own sample.
        assert rollout.calls == (Call((1, 2), (10 + rollout.sample,), (-.5,), "stop", 7),)
        seen = json.loads(
            (tmp_path / "trials" / f"dew-{rollout.group}-{rollout.sample}" / "seen.json").read_text()
        )
        url = seen["agent"]["OPENAI_BASE_URL"]
        session = url.removeprefix("http://172.17.0.1:9090/sessions/").removesuffix("/v1")
        # A sample's session carries a secret token, so a sandbox cannot address its siblings' sessions.
        assert re.fullmatch(rf"hello:{rollout.group}:{rollout.sample}:[0-9a-f]{{32}}", session)
        assert seen["agent"] == {"MSWEA_API_KEY": "none", "OPENAI_BASE_URL": url}
        assert seen["arguments"][:6] == [
            "trials",
            "start",
            "-a",
            "mini-swe-agent",
            "-m",
            "hosted_vllm/policy",
        ]
        assert ("DELETE", f"/sessions/{session}") in asked


def test_a_cancelled_trial_is_interrupted_and_resolves_cancelled(tmp_path, harbor):
    (tmp_path / "slow").mkdir()
    gateway, asked = fake_gateway()
    source = HarborSource(gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials")
    try:
        (future,) = source.submit(Task("slow", {HARBOR_KEY: str(tmp_path / "slow")}), 1, version=0, key=KEY)
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
    assert any(method == "DELETE" and path.startswith(f"/sessions/slow:{rollout.group}:0:")
               for method, path in asked)


def test_a_harbor_that_ignores_the_interrupt_is_terminated_after_the_grace(tmp_path, harbor):
    (tmp_path / "stubborn").mkdir()
    gateway, _ = fake_gateway()
    source = HarborSource(
        gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials", grace=1.0
    )
    try:
        (future,) = source.submit(Task("stubborn", {HARBOR_KEY: str(tmp_path / "stubborn")}), 1,
                                  version=0, key=KEY)
        deadline = time.monotonic() + 30
        while not list((tmp_path / "trials").glob("*/seen.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        source.cancel([future])
        assert future.result(timeout=15).status is Status.CANCELLED
        # Cancelling a resolved future again is a no-op, not a signal to a reaped process group.
        source.cancel([future])
    finally:
        source.close()


def test_a_trial_past_its_timeout_is_interrupted_and_the_scheduler_retries_it(tmp_path, harbor):
    """Harbor bounds a trial's phases, not the whole trial. One trial hangs: `trial_timeout`
    interrupts it, its Harbor exits, it resolves INFRA_ERROR, and the scheduler's retry
    completes its group, so the step admits every task's group."""
    import optax

    from dew.data.dataset import Loader
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.rl import GRPOObjective, RolloutScheduler
    from dew.objectives.rl.scheduler import task_ids
    from dew.training import Layout, Trainer

    class Publisher:
        version = 0

        def load(self, variables, version):
            self.version = version

    hanging = tmp_path / "once"
    hanging.mkdir()
    gateway, _ = fake_gateway(version=None)
    source = HarborSource(gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials",
                          grace=1.0, trial_timeout=10.0)
    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=16, dtype="float32")
    objective = GRPOObjective(model, seq_len=7)
    tasks, records = jax.device_count(), []
    scheduler = RolloutScheduler(
        objective, source, Publisher(), width=8, rows=2 * tasks, groups=2, max_lag=0, ahead=0,
        tasks=lambda batch: [Task(task.id, {HARBOR_KEY: str(hanging)}) for task in task_ids(batch)],
        log=records.append)
    data = scheduler.tasks(Loader(train=lambda partition: iter(
        lambda: {"task_id": np.arange(tasks, dtype=np.int32)}, None), val=None, records=None, batch=tasks))
    try:
        trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(3), rollout=scheduler,
                          layout=Layout(min_shard=1, tolerance=1.0))
        state = trainer.fit(data, steps=1, log_every=1)
    finally:
        scheduler.close()
        source.close()
    assert int(state.step) == 1
    [record] = records
    assert record.groups == tasks and record.abandoned == 0 and record.resubmitted == {"infra_error": 1}
    # The hung trial's Harbor exited on the interrupt, long before its sleep ended, so of the
    # 2 * tasks + 1 trials only the hung one wrote no result.
    pid = int((hanging / "hung").read_text())
    assert not Path(f"/proc/{pid}").exists()
    assert len(list((tmp_path / "trials").glob("*/seen.json"))) == 2 * tasks + 1
    assert len(list((tmp_path / "trials").glob("*/result.json"))) == 2 * tasks


def ended(pid):
    """Whether process `pid` is gone or a zombie waiting to be reaped."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


@pytest.mark.parametrize("task", ["lingering-exits", "lingering-waits", "lingering-detached-waits"])
def test_a_trial_past_its_timeout_resolves_whatever_holds_its_output(tmp_path, harbor, task):
    """Harbor exits before the deadline (`exits`) or on its interrupt (`waits`), and a child that
    ignores SIGINT and SIGTERM holds its output open. In Harbor's process group, the escalation
    runs until the output closes and the SIGKILL ends the child. A child that left the group
    (`detached`) is beyond the group's signals, so reading stops a grace after the SIGKILL.
    Either way the trial resolves INFRA_ERROR within a few graces of its deadline, long before
    the child's minute-long sleep ends."""
    (tmp_path / task).mkdir()
    gateway, _ = fake_gateway()
    source = HarborSource(gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials",
                          grace=1.0, trial_timeout=2.0)
    began = time.monotonic()
    try:
        (future,) = source.submit(Task(task, {HARBOR_KEY: str(tmp_path / task)}), 1, version=0, key=KEY)
        rollout = future.result(timeout=30)
    finally:
        source.close()
    resolved = time.monotonic() - began
    (child,) = (tmp_path / "trials").glob("*/child.json")
    pid = json.loads(child.read_text())
    if "detached" in task:
        os.kill(pid, 9)
    else:
        assert ended(pid)
    assert rollout.status is Status.INFRA_ERROR and resolved < 20


def test_close_resolves_queued_trials_without_launching_them(tmp_path, harbor):
    (tmp_path / "slow").mkdir()
    gateway, _ = fake_gateway()
    source = HarborSource(
        gateway, harbor=harbor, model="hosted_vllm/policy", trials=tmp_path / "trials", workers=1
    )
    futures = source.submit(Task("slow", {HARBOR_KEY: str(tmp_path / "slow")}), 2, version=0, key=KEY)
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


def test_the_first_submission_waits_for_a_healthy_gateway_worker(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, asked = fake_gateway(unhealthy_checks=3)
    source = HarborSource(gateway, harbor=harbor, model="m/p", trials=tmp_path / "trials", ready_poll=0.01)
    try:
        (future,) = source.submit(Task("hello", {HARBOR_KEY: str(tmp_path / "task")}), 1, version=0, key=KEY)
        assert future.result(timeout=60).status is Status.COMPLETED
    finally:
        source.close()
    # No trial started until the gateway reported a healthy worker, and it asked only once per source.
    assert asked.index(("GET", "/v1/models")) < next(
        index for index, entry in enumerate(asked) if entry[1].endswith("/traces"))


def test_a_gateway_that_reports_healthy_before_its_engine_listens_is_waited_for(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, _ = fake_gateway(unreachable_models=3)
    source = HarborSource(gateway, harbor=harbor, model="m/p", trials=tmp_path / "trials", ready_poll=0.01)
    try:
        (future,) = source.submit(Task("hello", {HARBOR_KEY: str(tmp_path / "task")}), 1, version=0, key=KEY)
        assert future.result(timeout=60).status is Status.COMPLETED
    finally:
        source.close()
    # The model listing went through the gateway to the engine and answered before any trial ran.


def test_waiting_for_the_gateway_does_not_block_close_or_cancel(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, _ = fake_gateway(unhealthy_checks=10 ** 9)
    source = HarborSource(gateway, harbor=harbor, model="m/p", trials=tmp_path / "trials",
                          ready_timeout=3.0, ready_poll=0.01)
    waiting = threading.Thread(
        target=lambda: pytest.raises(
            RuntimeError, source.submit, Task("hello", {HARBOR_KEY: str(tmp_path / "task")}), 1, version=0,
            key=KEY)
    )
    waiting.start()
    time.sleep(0.3)
    began = time.monotonic()
    source.cancel([])
    source.close()
    # Neither waited out the readiness poll, which holds no lock the source's other calls need.
    assert time.monotonic() - began < 1.0
    waiting.join(10)


def test_a_gateway_with_no_healthy_worker_refuses_the_submission(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, _ = fake_gateway(unhealthy_checks=10 ** 9)
    source = HarborSource(gateway, harbor=harbor, model="m/p", trials=tmp_path / "trials",
                          ready_timeout=0.2, ready_poll=0.01)
    try:
        with pytest.raises(RuntimeError):
            source.submit(Task("hello", {HARBOR_KEY: str(tmp_path / "task")}), 1, version=0, key=KEY)
        assert not (tmp_path / "trials").exists()
    finally:
        source.close()


TRAINER = """
import sys, time
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {tests!r})
from test_harbor_source import KEY, fake_gateway
from dew.interop.harbor import HARBOR_KEY, HarborSource
from dew.objectives.rl.sessions import Task
root = Path({root!r})
source = HarborSource(fake_gateway()[0], harbor=root / "harbor", model="m/p",
                      trials=root / "trials", workers=1,
                      grace=2.0)
source.submit(Task({task!r}, {{HARBOR_KEY: str(root / {task!r})}}), 2, version=0, key=KEY)
while not list((root / "trials").glob({ready!r})):
    time.sleep(0.05)
raise KeyboardInterrupt  # the trainer is interrupted mid-trial and never calls close()
"""


@pytest.mark.parametrize("task", ["slow", "lingering-waits"])
def test_an_interrupted_trainer_kills_its_trials_and_starts_no_queued_one(tmp_path, harbor, task):
    """With `lingering-waits`, Harbor exits on the interrupt but a child in its group ignores
    SIGINT and SIGTERM; the exit hook kills the group, so the child dies with the trainer too."""
    (tmp_path / task).mkdir()
    script = tmp_path / "trainer.py"
    script.write_text(TRAINER.format(src=str(Path(__file__).resolve().parents[1] / "src"),
                                     tests=str(Path(__file__).parent), root=str(tmp_path), task=task,
                                     ready="*/child.json" if task.startswith("lingering") else "*/seen.json"))
    began = time.monotonic()
    trainer = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120,
                             env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert "KeyboardInterrupt" in trainer.stderr
    # The running trial was killed (its 60 s sleep never finished) and the queued one never launched.
    assert time.monotonic() - began < 30
    (seen,) = (tmp_path / "trials").glob("*/seen.json")
    assert not list((tmp_path / "trials").glob("*/result.json"))
    # The running trial died with its trainer rather than running on without it.
    pid = json.loads(seen.read_text())["pid"]
    deadline = time.monotonic() + 10
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    alive = Path(f"/proc/{pid}").exists()
    if alive:
        os.kill(pid, 9)
    assert not alive, "the interrupted trainer's running trial outlived it"
    if task.startswith("lingering"):
        (child,) = (tmp_path / "trials").glob("*/child.json")
        pid = json.loads(child.read_text())
        deadline = time.monotonic() + 10
        while not ended(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        lingering = not ended(pid)
        if lingering:
            os.kill(pid, 9)
        assert not lingering, "a child of the interrupted trainer's trial outlived it"
