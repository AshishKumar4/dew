"""Agent harnesses behind a recording gateway, as a rollout source: Harbor trials, and any agent
command under `Command`.

The gateway's traces are the JSON rllm-model-gateway serves for a vLLM or
SGLang worker; Harbor is a stand-in executable that takes the same `trials
start` arguments and writes a `TrialResult`-shaped `result.json`, and an
agent command is a stand-in CLI that records what it was given and leaves an
answer for its verifier. A trial
that hangs past `trial_timeout` is interrupted and retried by a real
`RolloutScheduler` under a `Trainer`, and a child Harbor leaves in its
process group, which ignores SIGINT and SIGTERM and holds its output open,
is killed with the group, at the deadline and at exit.
"""

import contextlib
import json
import os
import re
import signal
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

from dew.interop.gateway import Gateway, calls
from dew.interop.harbor import HARBOR_KEY, Harbor, outcome
from dew.interop.harness import Command, HarnessSource
from dew.objectives.rl.sessions import Call, Status, Task, Transcript


@pytest.mark.parametrize("slow_endpoint", ["/health/workers", "/v1/models"])
def test_gateway_readiness_rejects_an_answer_after_its_deadline(slow_endpoint):
    """A worker-health probe or its proxied model listing answers too late. Each network phase
    receives the remaining budget, and neither late answer establishes readiness."""
    probes = []

    def network(request):
        probes.append(request)
        if request.url.path == slow_endpoint:
            threading.Event().wait(timeout=0.1)
        return httpx.Response(200, json={"healthy": 1} if request.url.path == "/health/workers" else {})

    with httpx.Client(transport=httpx.MockTransport(network), timeout=30) as client:
        gateway = Gateway("http://gateway", client=client)
        with pytest.raises(RuntimeError):
            gateway.ready(timeout=0.05, poll=30)
    assert probes
    assert all(0 < request.extensions["timeout"]["read"] <= 0.05 for request in probes)


def test_gateway_readiness_checks_both_endpoints_within_its_budget():
    probes = []
    closed = []

    class UnreadBody(httpx.SyncByteStream):
        def __iter__(self):
            raise RuntimeError("readiness must not wait for a response body")

        def close(self):
            closed.append(self)

    def network(request):
        probes.append(request)
        return httpx.Response(200, stream=UnreadBody())

    with httpx.Client(transport=httpx.MockTransport(network), timeout=30) as client:
        Gateway("http://gateway", client=client).ready(timeout=10)
    assert [request.url.path for request in probes] == ["/health/workers", "/v1/models"]
    assert 0 < probes[1].extensions["timeout"]["read"] <= probes[0].extensions["timeout"]["read"] <= 10
    assert len(closed) == len(probes)


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
        assert call.tokens == tuple(choice["response_token_ids"] if engine == "sglang"
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
    assert outcome(result(rewards={"reward": 0}), recorded).status is status


@pytest.mark.parametrize(
    ("traces", "exception", "harness_exit"),
    [
        # An engine failure beside an overflow, in either order.
        ([refusal("EngineCore died", 500), overflow("vllm")], None, None),
        ([overflow("sglang"), refusal("EngineCore died", 500)], None, None),
        # An aborted call, then the harness's retry overflowing.
        ([trace([1, 2], [3], [-0.5], "abort"), overflow("vllm")], None, None),
        # A model client that failed, in a trial that also timed out.
        ([trace([1, 2], [3], [-0.5])], "AgentTimeoutError", "APIConnectionError"),
    ],
)
def test_an_infrastructure_failure_outranks_a_limit_the_trial_also_hit(traces, exception, harness_exit):
    """The scheduler retries an infra session and may train a truncated one, so
    a trial the infrastructure broke is infra even where it also overflowed or
    ran out of time."""
    recorded = calls([{**entry, "timestamp": 1.0 + index} for index, entry in enumerate(traces)], unstamped=0)
    judged = outcome(result(exception, rewards={"reward": 0}), recorded, harness_exit=harness_exit)
    assert judged.status is Status.INFRA_ERROR


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
if "stubborn" in value("-p") or "quiet" in value("-p"):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
if "quiet" in value("-p"):
    # Harbor closes its output and runs on, ignoring the interrupt: only the deadline's
    # escalation ends it.
    os.close(1)
    os.close(2)
    time.sleep(60)
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
if "leaves" in value("-p"):
    # A server it starts in its process group, its output elsewhere, which runs on as the trial ends.
    import subprocess
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(os.path.join(trial, "child.json"), "w") as out:
        json.dump(server.pid, out)
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


def harbor_trials(harbor, trials, *, model="hosted_vllm/policy", **options):
    """The stand-in Harbor as a harness, writing its trials under `trials`."""
    return Harbor(harbor=harbor, model=model, trials=trials, **options)


@pytest.fixture
def harbor(tmp_path):
    executable = tmp_path / "harbor"
    executable.write_text(f"#!{sys.executable}\n" + HARBOR)
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    return executable


def fake_gateway(unhealthy_checks=0, unreachable_models=0, version=7, extra=()):
    """A gateway whose every session recorded one call sampling [10 + sample], stamped `version`,
    and the traces `extra` after it.

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
            if checks[0] <= unhealthy_checks or listings[0] <= unreachable_models:
                return httpx.Response(500, text="Internal Server Error")
            return httpx.Response(200, json={"object": "list", "data": [{"id": "policy"}]})
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": 1})
        session = request.url.path.removeprefix("/sessions/").removesuffix("/traces")
        sample = int(session.split(":")[2])
        later = ({**entry, "timestamp": 2.0 + index} for index, entry in enumerate(extra))
        return httpx.Response(200, json=[trace([1, 2], [10 + sample], [-.5], version=version), *later])

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


def test_constructing_a_harness_source_or_a_publication_opens_nothing(tmp_path, harbor):
    """Constructing a source starts no trial worker and asks the gateway nothing, and constructing a
    publication stamps nothing (api spec A15): a source starts its workers at its first `submit`, and
    a publication stamps its launch version when the version is first read, before any draw is
    submitted under it, so a gateway an earlier run left ahead still cannot mislabel the first
    calls."""
    from dew.inference import Publication

    before = {thread.name for thread in threading.enumerate()}
    gateway, asked = fake_gateway()
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials"), gateway)
    assert not any(thread.name.startswith("dew-harness-trial") for thread in threading.enumerate()
                   if thread.name not in before)
    assert asked == []
    source.close()

    stamped = []
    publication = Publication(lambda variables, version: None, version=3, stamp=stamped.append)
    assert stamped == []
    assert publication.version == 3 and stamped == [3]
    assert publication.version == 3 and stamped == [3]
    publication.load({}, 4)
    assert publication.version == 4 and stamped == [3, 4]

    # A load before the version is first read leaves the launch stamp unsent.
    loaded = []
    early = Publication(lambda variables, version: None, version=3, stamp=loaded.append)
    early.load({}, 5)
    assert early.version == 5 and loaded == [5]

    # A first read while a load is publishing waits for it, so its stamp cannot land on the load's.
    pushing, release, raced = threading.Event(), threading.Event(), []

    def push(variables, version):
        pushing.set()
        release.wait()

    racing = Publication(push, version=0, stamp=raced.append)
    loader = threading.Thread(target=racing.load, args=({}, 1))
    loader.start()
    pushing.wait()
    reader = threading.Thread(target=lambda: raced.append(("read", racing.version)))
    reader.start()
    release.set()
    loader.join()
    reader.join()
    assert raced == [1, ("read", 1)]


def test_each_sample_is_its_own_trial_and_gateway_session(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, asked = fake_gateway()
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials", environment={"MSWEA_API_KEY": "none"},
                                         arguments=("-e", "docker")), gateway)
    try:
        futures = source.submit(Task("hello", {HARBOR_KEY: str(tmp_path / "task")}), 2, version=5, key=KEY)
        rollouts = [future.result() for future in futures]
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
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials"), gateway)
    try:
        (future,) = source.submit(Task("slow", {HARBOR_KEY: str(tmp_path / "slow")}), 1, version=0, key=KEY)
        deadline = time.monotonic() + 30
        while not list((tmp_path / "trials").glob("*/seen.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        began = time.monotonic()
        source.cancel([future])
        rollout = future.result()
    finally:
        source.close()
    assert rollout.status is Status.CANCELLED and time.monotonic() - began < 30
    # A cancelled session's traces leave the gateway too, not only a scored one's.
    assert any(method == "DELETE" and path.startswith(f"/sessions/slow:{rollout.group}:0:")
               for method, path in asked)


def test_a_harbor_that_ignores_the_interrupt_is_terminated_after_the_grace(tmp_path, harbor):
    (tmp_path / "stubborn").mkdir()
    gateway, _ = fake_gateway()
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials"), gateway, grace=1.0)
    try:
        (future,) = source.submit(Task("stubborn", {HARBOR_KEY: str(tmp_path / "stubborn")}), 1,
                                  version=0, key=KEY)
        deadline = time.monotonic() + 30
        while not list((tmp_path / "trials").glob("*/seen.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        source.cancel([future])
        assert future.result().status is Status.CANCELLED
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
    from dew.objectives.rl import Asynchronous, GRPOObjective, RolloutScheduler
    from dew.objectives.rl.scheduler import task_ids
    from dew.training import Layout, Trainer

    class Publisher:
        version = 0

        def load(self, variables, version):
            self.version = version

    hanging = tmp_path / "once"
    hanging.mkdir()
    gateway, _ = fake_gateway(version=None)
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials"), gateway, grace=1.0, trial_timeout=10.0)
    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=16, dtype="float32")
    objective = GRPOObjective(model, seq_len=7)
    tasks, records = jax.device_count(), []
    scheduler = RolloutScheduler(
        Asynchronous(source, Publisher(), objective, width=8, rows=2 * tasks, max_lag=0, ahead=0),
        tasks=lambda batch: [Task(task.id, {HARBOR_KEY: str(hanging)}) for task in task_ids(batch)],
        groups=2, log=records.append)
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


def test_a_finished_trial_ends_what_it_left_running_in_its_group(tmp_path, harbor):
    """Harbor finishes its trial and exits with a server it started still running in its process
    group, its output elsewhere: the trial completes, and the server ends with it."""
    (tmp_path / "leaves").mkdir()
    gateway, _ = fake_gateway()
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials"), gateway, grace=1.0)
    try:
        (future,) = source.submit(Task("leaves", {HARBOR_KEY: str(tmp_path / "leaves")}), 1, version=0,
                                  key=KEY)
        rollout = future.result()
    finally:
        source.close()
    (child,) = (tmp_path / "trials").glob("*/child.json")
    assert rollout.status is Status.COMPLETED and ended(json.loads(child.read_text()))


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
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials"), gateway, grace=1.0, trial_timeout=2.0)
    began = time.monotonic()
    try:
        (future,) = source.submit(Task(task, {HARBOR_KEY: str(tmp_path / task)}), 1, version=0, key=KEY)
        rollout = future.result()
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


def test_a_trial_that_closes_its_output_and_runs_on_is_bounded_by_its_deadline(tmp_path, harbor):
    """Harbor closes its output, ignores the interrupt and runs on: the deadline still holds past
    the output's close, its escalation's SIGTERM ends Harbor, and the trial resolves INFRA_ERROR
    within a few graces, long before Harbor's minute-long sleep ends."""
    task = "quiet"
    (tmp_path / task).mkdir()
    gateway, _ = fake_gateway()
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials"), gateway, grace=1.0, trial_timeout=2.0)
    began = time.monotonic()
    try:
        (future,) = source.submit(Task(task, {HARBOR_KEY: str(tmp_path / task)}), 1, version=0, key=KEY)
        rollout = future.result()
    finally:
        source.close()
    (seen,) = (tmp_path / "trials").glob("*/seen.json")
    assert ended(json.loads(seen.read_text())["pid"])
    assert rollout.status is Status.INFRA_ERROR and time.monotonic() - began < 20


def test_a_trial_without_a_descriptor_for_its_wake_pipe_is_infra_and_frees_its_worker(tmp_path, harbor):
    """A trial that cannot make its wake pipe, every descriptor taken, resolves an INFRA_ERROR session
    naming that OSError, which the scheduler retries (ledger 1875), and the source's one worker goes
    on to run the next trial once descriptors are free. In a process of its own, whose descriptor
    limit it lowers (`descriptorless_flow`)."""
    (tmp_path / "task").mkdir()
    here = Path(__file__).resolve().parent
    script = (f"import test_harbor_source as tests; "
              f"tests.descriptorless_flow({str(harbor)!r}, {str(tmp_path)!r})")
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         cwd=here, env={**os.environ,
                                        "PYTHONPATH": os.pathsep.join([str(here.parent / "src"), str(here)])})
    assert run.returncode == 0, run.stderr[-4000:]
    assert json.loads(run.stdout.splitlines()[-1]) == ["COMPLETED", "INFRA_ERROR", "OSError", "COMPLETED"]


def descriptorless_flow(harbor: str, root: str) -> None:
    """Run a trial, take every descriptor and submit one, free them and submit another, on one
    worker; print the first's and last's statuses, the middle one's and the error class its detail
    names."""
    import resource

    task = Task("hello", {HARBOR_KEY: str(Path(root) / "task")})
    source = HarnessSource(harbor_trials(harbor, Path(root) / "trials"), fake_gateway()[0], workers=1)
    try:
        (first,) = source.submit(task, 1, version=0, key=KEY)
        before = first.result()
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
        taken = []
        with contextlib.suppress(OSError):
            while True:
                taken.append(os.open(os.devnull, os.O_RDONLY))
        (starved,) = source.submit(task, 1, version=0, key=KEY)
        failed = starved.result()
        for descriptor in taken:
            os.close(descriptor)
        (after,) = source.submit(task, 1, version=0, key=KEY)
        rollout = after.result()
    finally:
        source.close()
    print(json.dumps([before.status.name, failed.status.name, failed.detail.split(":")[1].strip(),
                      rollout.status.name]))


def test_a_cancel_alone_resolves_a_trial_whose_output_a_detached_child_holds(tmp_path, harbor):
    """With no deadline, the cancel's escalation still bounds the trial: a grace after the SIGKILL,
    the output a child outside the group keeps open is left unread, and the trial resolves
    CANCELLED long before the child's minute-long sleep ends."""
    task = "lingering-detached-waits"
    (tmp_path / task).mkdir()
    gateway, _ = fake_gateway()
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials"), gateway, grace=1.0)
    try:
        (future,) = source.submit(Task(task, {HARBOR_KEY: str(tmp_path / task)}), 1, version=0, key=KEY)
        deadline = time.monotonic() + 30
        while not list((tmp_path / "trials").glob("*/child.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        began = time.monotonic()
        source.cancel([future])
        rollout = future.result()
        resolved = time.monotonic() - began
    finally:
        source.close()
        # A trial that never resolved has waited the child out in `close`.
        for child in (tmp_path / "trials").glob("*/child.json"):
            with contextlib.suppress(ProcessLookupError):
                os.kill(json.loads(child.read_text()), 9)
    assert rollout.status is Status.CANCELLED and resolved < 20


def test_close_resolves_queued_trials_without_launching_them(tmp_path, harbor):
    (tmp_path / "slow").mkdir()
    gateway, _ = fake_gateway()
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials"), gateway, workers=1)
    futures = source.submit(Task("slow", {HARBOR_KEY: str(tmp_path / "slow")}), 2, version=0, key=KEY)
    deadline = time.monotonic() + 30
    while not list((tmp_path / "trials").glob("*/seen.json")) and time.monotonic() < deadline:
        time.sleep(0.05)
    began = time.monotonic()
    closer = threading.Thread(target=source.close)
    closer.start()
    closer.join(20)
    assert not closer.is_alive() and time.monotonic() - began < 20
    assert [future.result().status for future in futures] == [Status.CANCELLED, Status.CANCELLED]
    # The queued sample never started a trial.
    assert len(list((tmp_path / "trials").glob("*/seen.json"))) == 1


def test_the_first_submission_waits_for_a_healthy_gateway_worker(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, asked = fake_gateway(unhealthy_checks=3)
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials", model="m/p"), gateway, ready_poll=0.01)
    try:
        (future,) = source.submit(Task("hello", {HARBOR_KEY: str(tmp_path / "task")}), 1, version=0, key=KEY)
        assert future.result().status is Status.COMPLETED
    finally:
        source.close()
    # No trial started until the gateway reported a healthy worker, and it asked only once per source.
    assert asked.index(("GET", "/v1/models")) < next(
        index for index, entry in enumerate(asked) if entry[1].endswith("/traces"))


def test_a_gateway_that_reports_healthy_before_its_engine_listens_is_waited_for(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, _ = fake_gateway(unreachable_models=3)
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials", model="m/p"), gateway, ready_poll=0.01)
    try:
        (future,) = source.submit(Task("hello", {HARBOR_KEY: str(tmp_path / "task")}), 1, version=0, key=KEY)
        assert future.result().status is Status.COMPLETED
    finally:
        source.close()
    # The model listing went through the gateway to the engine and answered before any trial ran.


def test_waiting_for_the_gateway_does_not_block_close_or_cancel(tmp_path, harbor):
    (tmp_path / "task").mkdir()
    gateway, _ = fake_gateway(unhealthy_checks=10 ** 9)
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials", model="m/p"), gateway,
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
    source = HarnessSource(harbor_trials(harbor, tmp_path / "trials", model="m/p"), gateway,
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
from dew.interop.harbor import HARBOR_KEY, Harbor
from dew.interop.harness import HarnessSource
from dew.objectives.rl.sessions import Task
root = Path({root!r})
source = HarnessSource(Harbor(harbor=root / "harbor", model="m/p", trials=root / "trials"), fake_gateway()[0],
                       workers=1, grace=2.0)
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
    trainer = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                             env={**os.environ, "JAX_PLATFORMS": "cpu"})
    # An uncaught KeyboardInterrupt ends CPython by SIGINT, so any other failure shows as itself.
    assert trainer.returncode == -signal.SIGINT, trainer.stderr[-2000:]
    # The running trial was killed (its 60 s sleep never finished) and the queued one never launched.
    assert time.monotonic() - began < 30
    (seen,) = (tmp_path / "trials").glob("*/seen.json")
    assert not list((tmp_path / "trials").glob("*/result.json"))
    # The running trial died with its trainer rather than running on without it. Orphaned by the
    # trainer, it is reparented to the namespace's init, which may leave it a zombie for longer.
    pid = json.loads(seen.read_text())["pid"]
    deadline = time.monotonic() + 10
    while not ended(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    alive = not ended(pid)
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


AGENT = """\
import json, os, sys, time
task, directory = sys.argv[1], os.getcwd()
with open("seen.json", "w") as seen:
    json.dump({"task": task, "cwd": directory, "url": os.environ["OPENAI_BASE_URL"],
               "key": os.environ.get("OPENAI_API_KEY")}, seen)
if task == "slow":
    time.sleep(60)
with open("answer.txt", "w") as answer:
    answer.write("4" if task != "wrong" else "5")
sys.exit(3 if task == "failing" else 0)
"""


def agent_command(tmp_path, verify=None):
    """`Command` over the stand-in agent CLI; the verifier scores 1 for a directory whose answer is 4,
    0 for any other, and nothing where the agent left none."""
    script = tmp_path / "agent.py"
    script.write_text(AGENT)

    def scored(task, directory, returncode):
        answer = directory / "answer.txt"
        return (answer.read_text() == "4") * 1.0 if answer.is_file() else None

    return Command(lambda task, directory: [sys.executable, str(script), task.id], verify or scored,
                   tmp_path / "work", environment={"OPENAI_API_KEY": "none"})


@pytest.mark.parametrize(("task", "status", "reward"), [
    ("adds", Status.COMPLETED, 1.0), ("wrong", Status.COMPLETED, 0.0), ("failing", Status.AGENT_ERROR, 1.0)])
def test_an_agent_command_runs_in_its_own_directory_against_its_own_session(tmp_path, task, status, reward):
    """Each sample runs the command once in a fresh directory, its model at its own gateway session
    and the environment it was given; the verifier scores what it left, and a command that exits
    nonzero is the agent's error, scored all the same."""
    gateway, asked = fake_gateway()
    source = HarnessSource(agent_command(tmp_path), gateway)
    try:
        rollouts = [future.result() for future in source.submit(Task(task), 2, version=3, key=KEY)]
    finally:
        source.close()
    for rollout in rollouts:
        assert (rollout.status, rollout.reward) == (status, reward)
        assert rollout.calls == (Call((1, 2), (10 + rollout.sample,), (-.5,), "stop", 7),)
        directory = tmp_path / "work" / f"dew-{rollout.group}-{rollout.sample}"
        seen = json.loads((directory / "seen.json").read_text())
        session = seen["url"].removeprefix("http://172.17.0.1:9090/sessions/").removesuffix("/v1")
        assert re.fullmatch(rf"{task}:{rollout.group}:{rollout.sample}:[0-9a-f]{{32}}", session)
        assert seen == {"task": task, "cwd": str(directory), "url": seen["url"], "key": "none"}
        assert ("DELETE", f"/sessions/{session}") in asked


@pytest.mark.parametrize("extra", [(), (overflow("vllm"),), (trace([1, 2, 10, 4], [5], [-.5], "length"),)],
                         ids=["ended", "refused", "length"])
def test_an_agent_command_the_verifier_cannot_score_is_infra_however_the_session_ended(tmp_path, extra):
    """A verifier that cannot score leaves the trial unscored, which is infra, retried and never
    trained, also where the engine refused a prompt or the last call ran out of tokens: a
    truncation would train or mask an invented outcome."""
    source = HarnessSource(agent_command(tmp_path, verify=lambda task, directory, code: None),
                           fake_gateway(extra=extra)[0])
    try:
        (unscored,) = source.submit(Task("adds"), 1, version=0, key=KEY)
        assert unscored.result().status is Status.INFRA_ERROR
    finally:
        source.close()


@pytest.mark.parametrize(("extra", "status"), [
    ((refusal("EngineCore died", 500),), Status.INFRA_ERROR), ((overflow("sglang"),), Status.TRUNCATED),
    ((trace([1, 2, 10, 4], [5], [-.5], "length"),), Status.TRUNCATED)], ids=["failed", "refused", "length"])
def test_an_agent_command_is_judged_by_what_the_engine_said_first(tmp_path, extra, status):
    """An engine failure makes a scored trial infra; a refused prompt or a last call stopped at its
    length limit truncates it, with its score."""
    source = HarnessSource(agent_command(tmp_path), fake_gateway(extra=extra)[0])
    try:
        (future,) = source.submit(Task("adds"), 1, version=0, key=KEY)
        rollout = future.result()
    finally:
        source.close()
    assert rollout.status is status and rollout.reward == (None if status is Status.INFRA_ERROR else 1.0)


def test_an_agent_command_in_a_relative_workspace_runs_what_it_put_in_its_directory(tmp_path, monkeypatch):
    """The command factory writes its agent into the trial's directory and runs it from there: the
    directory it is given resolves inside the trial's own working directory."""
    monkeypatch.chdir(tmp_path)

    def argv(task, directory):
        (directory / "agent.py").write_text(AGENT)
        return [sys.executable, str(directory / "agent.py"), task.id]

    def verify(task, directory, code):
        return float((directory / "answer.txt").read_text() == "4")

    command = Command(argv, verify, Path("work"))
    source = HarnessSource(command, fake_gateway()[0])
    try:
        (future,) = source.submit(Task("adds"), 1, version=0, key=KEY)
        rollout = future.result()
    finally:
        source.close()
    assert (rollout.status, rollout.reward) == (Status.COMPLETED, 1.0)


def test_a_cancelled_agent_command_is_interrupted_and_resolves_cancelled(tmp_path):
    source = HarnessSource(agent_command(tmp_path), fake_gateway()[0])
    try:
        (future,) = source.submit(Task("slow"), 1, version=0, key=KEY)
        deadline = time.monotonic() + 30
        while not list((tmp_path / "work").glob("*/seen.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        began = time.monotonic()
        source.cancel([future])
        rollout = future.result()
    finally:
        source.close()
    assert rollout.status is Status.CANCELLED and time.monotonic() - began < 30


def test_an_agent_commands_calls_keep_the_conversation_the_gateway_recorded(tmp_path):
    """The gateway records each call's chat messages and the message the engine answered, tool calls
    included, beside its ids: the call carries them as its transcript, its ids as recorded."""
    asked = [{"role": "system", "content": "Use the bash tool."}, {"role": "user", "content": "Add 2 and 2."}]
    command = {"name": "bash", "arguments": '{"command": "echo 4"}'}
    call = {"id": "call-0", "type": "function", "function": command}
    answered = {"role": "assistant", "content": None, "tool_calls": [call]}
    conversation = trace([1, 2, 10], [5, 6], [-.5, -.25], "tool_calls", messages=asked,
                         response_message=answered)
    source = HarnessSource(agent_command(tmp_path), fake_gateway(extra=[conversation])[0])
    try:
        (future,) = source.submit(Task("adds"), 1, version=0, key=KEY)
        rollout = future.result()
    finally:
        source.close()
    plain, spoken = rollout.calls
    assert plain.transcript is None
    assert spoken == Call((1, 2, 10), (5, 6), (-.5, -.25), "tool_calls", 0,
                          transcript=Transcript(tuple(asked), answered))


@pytest.mark.parametrize(("field", "value"),
                         [("messages", {}), ("messages", False), ("response_message", [])])
def test_a_trace_whose_transcript_is_not_the_gateways_json_leaves_the_trial_infra(tmp_path, field, value):
    """A gateway trace's messages are a list of objects and its response message an object; a trace
    holding anything else is unusable, as a trace without ids is, not a call without a transcript."""
    malformed = trace([1, 2, 10], [5, 6], [-.5, -.25], "stop",
                      **{"messages": [{"role": "user", "content": "hi"}], field: value})
    source = HarnessSource(agent_command(tmp_path), fake_gateway(extra=[malformed])[0])
    try:
        (future,) = source.submit(Task("adds"), 1, version=0, key=KEY)
        rollout = future.result()
    finally:
        source.close()
    assert rollout.status is Status.INFRA_ERROR


def test_an_agent_command_that_cannot_start_is_infra(tmp_path):
    """A command whose executable is missing raises OSError at its launch, the trial's own external
    boundary: the session is INFRA_ERROR naming it, which the scheduler retries, not a failed future
    that aborts the rollout (ledger 1875)."""
    command = Command(lambda task, directory: [str(tmp_path / "missing-agent")],
                      lambda task, directory, code: 1.0, tmp_path / "work")
    source = HarnessSource(command, fake_gateway()[0])
    try:
        (rollout,) = source.submit(Task("adds"), 1, version=0, key=KEY)
        failed = rollout.result()
    finally:
        source.close()
    assert failed.status is Status.INFRA_ERROR and "FileNotFoundError" in failed.detail


def test_harbor_judges_an_unreadable_result_as_infra(tmp_path):
    """A trial whose result.json is not JSON is an infrastructure failure of the trial, an
    INFRA_ERROR naming it, not a decode error that aborts the rollout (ledger 1875)."""
    from dew.interop.harness import Trial

    artifacts = tmp_path / "trial"
    artifacts.mkdir()
    (artifacts / "result.json").write_text("{not json")
    harbor = harbor_trials(sys.executable, tmp_path / "trials")
    judged = harbor.judge(Task("hello"), Trial((), artifacts), "", 0, calls([], unstamped=0))
    assert judged.status is Status.INFRA_ERROR and "result.json" in judged.detail
