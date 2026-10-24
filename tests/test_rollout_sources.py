"""Session sources over a rollout server: in-process environments and single-turn prompts.

The server is scripted: each draw is fixed ids at the version it was
submitted under, and a held draw waits for the test. The checks: a
multi-turn session records one call per turn with each call's own version,
and its calls merge into one packed chain; the failure policy separates
infrastructure failures (an environment's ERROR or OS error, Dew's own
sandbox past its wall clock or dead, a verifier's or reward's OS error, a
container runtime that starts no container, a draw from an unreachable
engine, a verifier's HTTP transport error reported as an
InfrastructureError) from truncations, which are still scored, and from
contract errors (a non-finite reward, an environment or sandbox worker that
breaks its protocol, a failing decode), which fail the rollout's future; a
cancelled session stops mid-draw and releases its environment; a server
that reports no raw likelihood records none, and its sessions still train;
and the prompt source scores decoded text and components per draw.
"""

import dataclasses
import json
import socket
import threading
from concurrent.futures import Future, wait
from contextlib import contextmanager
from dataclasses import asdict, replace

import jax
import numpy as np
import pytest
from episode_support import environment as sandbox
from tool_support import load

from dew.data.prompts import PromptRecord
from dew.inference import Draw, VLLMGenerateServer
from dew.objectives.rl.episodes import EpisodeStatus, Observation
from dew.objectives.rl.records import episode_from_record
from dew.objectives.rl.rollout import prompt_tasks
from dew.objectives.rl.sessions import Status, Task, Transcript, pack
from dew.objectives.rl.sources import EnvironmentSource, PromptSource
from dew.rl.sandbox import ContainerRunner, SandboxFleet
from dew.sampling import Sampling

CodeReward = load("train_rlvr", "examples").CodeReward

EOS = 9
SAMPLING = Sampling(temperature=1.0, eos_token_ids=EOS)
KEY = jax.random.key(7)


def words(key):
    return tuple(int(word) for word in jax.random.key_data(key))


class Server:
    """Draws `(5, EOS)`, or `tokens`, at the version served on submission; `hold` keeps draws pending."""

    def __init__(self, tokens=(5, EOS), raw=True, sampling=SAMPLING, hold=False, ending=None):
        self.version, self.tokens, self.raw, self._sampling, self.hold = 0, tokens, raw, sampling, hold
        self.ending = ending
        self.held, self.prompts, self.keys = [], [], []

    @property
    def sampling(self):
        return self._sampling

    def submit(self, prompt, max_new_tokens, *, key):
        self.prompts.append(tuple(prompt))
        self.keys.append(words(key))
        future = Future()
        ending = self.ending
        if ending is None:
            ending = "stop" if self.tokens and self.tokens[-1] == EOS else "length"
        draw = Draw(tuple(prompt), self.tokens, (-.5,) * len(self.tokens),
                    ending, self.version,
                    (-.4,) * len(self.tokens) if self.raw else None)
        if self.hold:
            self.held.append((future, draw))
        else:
            future.set_result(draw)
        return future


class Counter:
    """Appends each action and one tool id to the context; completes after `turns` steps, paying
    `paid[step]` for each step where given."""

    def __init__(self, turns=3, fail=None, on_step=None, paid=None):
        self.turns, self.fail, self.on_step, self.paid = turns, fail, on_step, paid
        self.context, self.steps = (1, 2), 0

    def reset(self):
        return Observation(self.context)

    def step(self, action):
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self.steps)
        if self.fail == "raise":
            raise ConnectionError("sandbox went away")
        if self.fail == "report":
            return Observation((), EpisodeStatus.ERROR, "container exited")
        if self.fail == "malformed":
            return {"context": [1, 2]}
        self.context = action.prompt + action.tokens + (4,)
        reward = None if self.paid is None else self.paid[self.steps - 1]
        if self.steps == self.turns:
            return Observation((), EpisodeStatus.COMPLETED, "tests passed", reward)
        return Observation(self.context, reward=reward)


def factory(entered, exited, **options):
    @contextmanager
    def environment(task, identity):
        entered.append((task.id, identity))
        try:
            yield Counter(**options)
        finally:
            exited.append((task.id, identity))
    return environment


def verifier(task, episode):
    return float(len(episode.transitions))


def source(server, environment, **options):
    options = {"max_prompt_tokens": 64, "max_new_tokens": 4, "max_turns": 5, "workers": 4, **options}
    return EnvironmentSource(server, environment, options.pop("verifier", verifier), **options)


def test_a_multi_turn_session_keeps_each_calls_version_and_packs_into_one_chain():
    server = Server()
    entered, exited = [], []

    def push(steps):
        server.version = steps  # a weight push lands after every tool step

    episodes = source(server, factory(entered, exited, on_step=push))
    rollout = episodes.submit(Task("3"), 1, version=0, key=KEY)[0].result(timeout=10)
    episodes.close()
    assert rollout.status == Status.COMPLETED and rollout.reward == 3.0
    assert [call.version for call in rollout.calls] == [0, 1, 2]
    # Turn t of sample 0 draws under fold_in(fold_in(key, 0), t), the key's own stream.
    assert server.keys == [words(jax.random.fold_in(jax.random.fold_in(KEY, 0), turn)) for turn in range(3)]
    assert [call.prompt for call in rollout.calls] == [
        (1, 2),
        (1, 2, 5, EOS, 4),
        (1, 2, 5, EOS, 4, 5, EOS, 4),
    ]
    assert entered == exited and entered[0][0] == "3" and rollout.task == "3"
    batch = pack([rollout], 16)
    assert batch["input_ids"].shape == (1, 16) and set(batch["text_segment_ids"][0].tolist()) == {0, 1}
    np.testing.assert_array_equal(batch["versions"][0][batch["response_mask"][0] > 0], [0, 0, 1, 1, 2, 2])


@pytest.mark.parametrize("credit", ["session", "turn"])
def test_step_rewards_credit_each_call_with_the_return_ahead_of_it(credit):
    """An environment that pays each sample its own reward at each of three steps: every call
    records the reward its step paid, and a session's return is the verifier's reward plus every
    step's. With `credit="turn"` each call's advantage is the session's return from that call on,
    centred on the group's members' at the same turn (`estimator="mean"`); with
    `credit="session"` every call takes its session's, the whole return centred."""
    paid = {0: (0.5, 0.0, 0.25), 1: (0.0, 0.25, 0.25), 2: (1.0, 0.0, 0.0), 3: (0.25, 0.25, 0.5)}

    @contextmanager
    def environment(task, identity):
        yield Counter(paid=paid[identity.sample])

    episodes = source(Server(), environment)
    rollouts = [future.result() for future in episodes.submit(Task("3"), 4, version=0, key=KEY)]
    episodes.close()
    assert [[call.reward for call in rollout.calls] for rollout in rollouts] == [
        list(paid[rollout.sample]) for rollout in rollouts]
    batch = pack(rollouts, 16, rows=4, estimator="mean", credit=credit)
    ahead = [[rollout.reward + sum(paid[rollout.sample][turn:]) for turn in range(3)] for rollout in rollouts]
    for index in range(len(rollouts)):
        for turn in range(3):
            at = turn if credit == "turn" else 0
            expected = ahead[index][at] - np.mean([returns[at] for returns in ahead])
            mine = (batch["session_index"] == index) & (batch["call_index"] == turn)
            assert mine.any() and (batch["advantages"][mine] == np.float32(expected)).all()


def test_a_session_ended_at_reset_takes_no_turn_in_the_turn_credit():
    """An environment whose sample 0 ends at its reset, scored 1 with no call, beside one that
    takes one call and scores 0: the call has no other call at its turn to compare with, so its
    turn advantage is zero, while session credit compares the two whole returns."""
    class Ended:
        def reset(self):
            return Observation((), EpisodeStatus.COMPLETED, "solved at reset")

    @contextmanager
    def environment(task, identity):
        yield Ended() if identity.sample == 0 else Counter(turns=1)

    episodes = source(Server(), environment, verifier=lambda task, episode: 1.0 - len(episode.transitions))
    ended, called = [future.result() for future in episodes.submit(Task("3"), 2, version=0, key=KEY)]
    episodes.close()
    assert (ended.calls, ended.reward, len(called.calls), called.reward) == ((), 1.0, 1, 0.0)
    for credit, expected in (("turn", 0.0), ("session", -0.5)):
        batch = pack([ended, called], 16, rows=1, estimator="mean", credit=credit)
        assert (batch["advantages"][batch["session_index"] == 1] == np.float32(expected)).all()


def test_each_call_lists_the_messages_the_observations_before_it_carried():
    """An environment whose reset and steps say, beside the ids, the chat messages their context
    added: each call's transcript lists the messages before it, the episode's journal record keeps
    them, and the ids the model read are the training record."""
    class Talking(Counter):
        def reset(self):
            return dataclasses.replace(super().reset(), messages=({"role": "user", "content": "count"},))

        def step(self, action):
            seen = super().step(action)
            return dataclasses.replace(seen, messages=({"role": "tool", "content": f"step {self.steps}"},))

    @contextmanager
    def environment(task, identity):
        yield Talking()

    recorded = []

    def scored(task, episode):
        recorded.append(episode)
        return verifier(task, episode)

    episodes = source(Server(), environment, verifier=scored)
    (rollout,) = [future.result() for future in episodes.submit(Task("3"), 1, version=0, key=KEY)]
    episodes.close()
    user, *tools = [{"role": "user", "content": "count"}, *({"role": "tool", "content": f"step {step}"}
                                                            for step in (1, 2))]
    assert [call.transcript for call in rollout.calls] == [
        Transcript((user,)), Transcript((user, tools[0])), Transcript((user, *tools))]
    assert [call.prompt for call in rollout.calls] == [
        (1, 2), (1, 2, 5, EOS, 4), (1, 2, 5, EOS, 4, 5, EOS, 4)]
    (episode,) = recorded
    assert episode_from_record(json.loads(json.dumps(asdict(episode)))) == episode


def test_every_call_of_a_multi_turn_session_carries_its_draws_engine_records():
    """Routing and the sampling support come from each draw, not the episode,
    so a multi-turn session replays what the engine routed and kept."""
    from dataclasses import replace

    from dew.objectives.rl.sessions import ROUTED_EXPERTS_KEY, SUPPORT_KEY

    class Recording(Server):
        def submit(self, prompt, max_new_tokens, *, key):
            draw = super().submit(prompt, max_new_tokens, key=key).result()
            done = Future()
            done.set_result(
                replace(
                    draw,
                    routed_experts=np.full((len(prompt) + 1, 2, 1), len(prompt), np.uint8),
                    support=((5, 6), (EOS,)),
                )
            )
            return done

    episodes = source(Recording(), factory([], []))
    session = episodes.submit(Task("3"), 1, version=0, key=KEY)[0].result(timeout=10)
    episodes.close()
    assert [int(call.routed_experts[0, 0, 0]) for call in session.calls] == [2, 5, 8]
    assert all(call.support == ((5, 6), (EOS,)) for call in session.calls)
    batch = pack([session], 16, support_capacity=16)
    assert ROUTED_EXPERTS_KEY in batch and (batch[SUPPORT_KEY] >= 0).sum() == 9


@pytest.mark.parametrize("fail, detail", [("raise", "ConnectionError: sandbox went away"),
                                          ("report", "container exited")])
def test_an_environment_failure_is_an_unscored_infra_error(fail, detail):
    entered, exited = [], []
    episodes = source(Server(), factory(entered, exited, fail=fail))
    rollout = episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=10)
    episodes.close()
    assert rollout.status == Status.INFRA_ERROR and rollout.reward is None and detail in rollout.detail
    assert len(rollout.calls) == 1 and exited == entered


def test_a_verifier_crash_is_an_infra_error():
    def crash(task, episode):
        raise TimeoutError("verifier sandbox timed out")

    episodes = source(Server(), factory([], []), verifier=crash)
    rollout = episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=10)
    episodes.close()
    assert rollout.status == Status.INFRA_ERROR and "verifier sandbox timed out" in rollout.detail


@pytest.mark.parametrize("mode, detail", [("hang", "TimeoutError"), ("crash", "ChildProcessError")])
def test_a_sandbox_past_its_wall_clock_or_dead_is_an_infra_error(mode, detail):
    """Dew's own sandbox bounds its session: its deadline and its worker's death come back as
    infrastructure failures, which the scheduler retries."""
    worker = sandbox(mode, wall_seconds=1)
    episodes = source(Server(), lambda task, identity: worker(identity))
    rollout = episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=30)
    episodes.close()
    assert rollout.status == Status.INFRA_ERROR and detail in rollout.detail and not rollout.calls


def test_a_container_runtime_that_starts_no_container_is_an_infra_error_in_both_sources():
    """The example's code reward on `ContainerRunner`, with `false` standing in for a runtime
    whose daemon is gone: the reward cannot run its program, which both sources retry."""
    cases, program = '[{"stdin": "", "stdout": "5"}]', "```python\nprint(5)\n```"
    with SandboxFleet(ContainerRunner("python:3.12-slim", runtime="false"), workers=1) as fleet:
        reward = CodeReward(fleet)
        task = Task("t", {"prompt": (1, 2), "record": PromptRecord("code", cases)})
        prompts = PromptSource(Server(), reward, decode=lambda ids: program, max_new_tokens=4)
        rollout = prompts.submit(task, 1, version=0, key=KEY)[0].result(timeout=30)
        prompts.close()
        episodes = source(Server(), factory([], []),
                          verifier=lambda task, episode: reward(program, PromptRecord("code", cases)))
        session = episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=30)
        episodes.close()
    for failed in (rollout, session):
        assert failed.status == Status.INFRA_ERROR and "InfrastructureError" in failed.detail


def unreachable():
    """The URL of a local port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


def test_an_http_transport_failure_is_an_infra_error_in_both_sources():
    """Two HTTP paths to a port nothing listens on. Dew's vLLM rollout server fails the draw with
    an InfrastructureError caused by httpx's ConnectError, and both sources retry the session. A
    verifier that judges over httpx reports its transport error, which is not an OSError, as an
    InfrastructureError, and is retried as well."""
    import httpx

    from dew.inference import InfrastructureError

    url = unreachable()
    engine = VLLMGenerateServer(url, Sampling(temperature=1.0, top_k=4, eos_token_ids=EOS),
                                lambda variables, version: None, timeout=10.0)
    try:
        episodes = source(engine, factory([], []))
        session = episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=30)
        episodes.close()
        [task] = prompt_tasks(prompt_batch())
        prompts = PromptSource(engine, lambda *_: 1.0, decode=str, max_new_tokens=4)
        rollout = prompts.submit(task, 1, version=0, key=KEY)[0].result(timeout=30)
        prompts.close()
        with pytest.raises(InfrastructureError) as caught:
            engine.submit((1, 2), 4, key=KEY).result(timeout=30)
    finally:
        engine.close()
    assert isinstance(caught.value.__cause__, httpx.ConnectError)
    for failed in (session, rollout):
        assert failed.status == Status.INFRA_ERROR and "InfrastructureError" in failed.detail

    def judged(task, episode):
        try:
            return float(httpx.post(f"{url}/judge", json={"task": task.id}, timeout=10.0).json()["score"])
        except httpx.TransportError as error:
            raise InfrastructureError(f"the judge is unreachable: {error}") from error

    episodes = source(Server(), factory([], []), verifier=judged)
    session = episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=30)
    episodes.close()
    assert session.status == Status.INFRA_ERROR and "the judge is unreachable" in session.detail
    assert len(session.calls) == 3 and session.reward is None


def test_a_contract_error_fails_the_rollouts_future_instead_of_being_retried():
    """A non-finite reward, an environment that returns no Observation, and a sandbox worker that
    answers with something other than JSON are broken code, not infrastructure: the future fails
    with the error, and the environment is still released."""
    def nan(task, episode):
        return float("nan")

    entered, exited = [], []
    episodes = source(Server(), factory(entered, exited), verifier=nan)
    with pytest.raises(ValueError):
        episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=10)
    episodes.close()
    broken = source(Server(), factory(entered, exited, fail="malformed"))
    with pytest.raises(TypeError):
        broken.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=10)
    broken.close()
    assert len(entered) == 2 and exited == entered
    worker = sandbox("garbage")
    garbled = source(Server(), lambda task, identity: worker(identity))
    with pytest.raises(ValueError):
        garbled.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=30)
    garbled.close()


def test_a_draw_that_breaks_its_contract_fails_both_sources_rollouts():
    """A draw future that fails with anything but an infrastructure error, as a native draw with
    non-finite likelihoods does, is a broken server or policy: both sources raise it."""
    class Broken(Server):
        def submit(self, prompt, max_new_tokens, *, key):
            future = Future()
            future.set_exception(ValueError("drawn likelihoods must be finite"))
            return future

    episodes = source(Broken(), factory([], []))
    with pytest.raises(ValueError):
        episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=10)
    episodes.close()
    [task] = prompt_tasks(prompt_batch())
    prompts = PromptSource(Broken(), lambda *_: 1.0, decode=str, max_new_tokens=4)
    with pytest.raises(ValueError):
        prompts.submit(task, 1, version=0, key=KEY)[0].result(timeout=10)
    prompts.close()


def test_tokens_after_eos_never_execute_an_environment_turn():
    """A completed server future is still untrusted: a token after EOS must fail admission before
    a tool sees it or a verifier assigns it reward, and the opened environment is released."""
    entered, exited, stepped, scored = [], [], [], []
    server = Server(tokens=(EOS, 7), ending="stop")

    def reward(task, episode):
        scored.append(episode)
        return 1.0

    episodes = source(server, factory(entered, exited, on_step=lambda _: stepped.append(True)),
                      verifier=reward)
    try:
        with pytest.raises(ValueError):
            episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=10)
    finally:
        episodes.close()
    assert entered and exited == entered
    assert not stepped and not scored


@pytest.mark.parametrize("ending", ["length", "stop"])
def test_empty_server_actions_fail_before_tools_or_rewards(ending):
    entered, exited, stepped, rewarded = [], [], [], []
    server = Server(tokens=(), ending=ending, sampling=replace(SAMPLING, stop=("finished",)))

    def reward(*args):
        rewarded.append(args)
        return 1.0

    episodes = source(server, factory(entered, exited, on_step=lambda _: stepped.append(True)),
                      verifier=reward)
    prompts = PromptSource(server, reward, decode=str, max_new_tokens=4)
    [task] = prompt_tasks(prompt_batch())
    try:
        for collector in (episodes, prompts):
            with pytest.raises(ValueError):
                collector.submit(task, 1, version=0, key=KEY)[0].result(timeout=10)
    finally:
        episodes.close()
        prompts.close()
    assert entered and exited == entered
    assert not stepped and not rewarded


def test_a_tool_call_finish_is_a_completed_single_turn_sample():
    [task] = prompt_tasks(prompt_batch())
    prompts = PromptSource(Server(ending="tool_calls"), lambda *_: 1.0, decode=str, max_new_tokens=4)
    try:
        session = prompts.submit(task, 1, version=0, key=KEY)[0].result(timeout=10)
    finally:
        prompts.close()
    assert session.status == Status.COMPLETED
    assert session.calls[0].terminated and session.reward == 1.0


def test_an_environment_exit_that_fails_after_broken_code_does_not_turn_it_into_a_retry():
    """The environment returns no Observation, then its exit fails with an OSError. The OSError
    that `with` raises carries the broken code's error, and the rollout fails with it."""
    @contextmanager
    def environment(task, identity):
        try:
            yield Counter(fail="malformed")
        finally:
            raise ConnectionError("the sandbox was gone before its release")

    episodes = source(Server(), environment)
    with pytest.raises(ConnectionError) as caught:
        episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=10)
    episodes.close()
    assert isinstance(caught.value.__context__, TypeError)


@pytest.mark.parametrize("server, options, calls, detail", [
    (Server(tokens=(5, 6, 7, 8)), {}, 1, "token limit"),
    (Server(), {"max_turns": 2}, 2, "turn limit"),
    (Server(), {"max_prompt_tokens": 6}, 2, "max_prompt_tokens"),
])
def test_limits_truncate_and_the_truncation_is_still_scored(server, options, calls, detail):
    episodes = source(server, factory([], []), **options)
    rollout = episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=10)
    episodes.close()
    assert rollout.status == Status.TRUNCATED and detail in rollout.detail
    assert len(rollout.calls) == calls and rollout.reward == float(calls)


def test_a_cancelled_session_stops_mid_draw_and_releases_its_environment():
    server = Server(hold=True)
    entered, exited = [], []
    episodes = source(server, factory(entered, exited))
    future = episodes.submit(Task("1"), 1, version=0, key=KEY)[0]
    while not server.held:
        threading.Event().wait(0.01)
    episodes.cancel([future])
    rollout = future.result(timeout=10)
    episodes.close()
    assert rollout.status == Status.CANCELLED and not rollout.calls and exited == entered


@pytest.mark.parametrize("raw", [True, False])
def test_a_transforming_server_with_or_without_raw_likelihoods_runs_and_records_what_it_reported(raw):
    """vLLM and SGLang report no raw likelihood. Their actions record none, the record reads
    back the same, and the session trains on its behavior likelihoods as any other."""
    seen = []

    def verifier(task, episode):
        seen.append(episode)
        return 1.0

    server = Server(raw=raw, sampling=Sampling(temperature=0.7, top_k=4, eos_token_ids=EOS))
    episodes = source(server, factory([], []), verifier=verifier)
    rollout = episodes.submit(Task("1"), 1, version=0, key=KEY)[0].result(timeout=10)
    episodes.close()
    assert rollout.status == Status.COMPLETED and len(rollout.calls) == 3
    [episode] = seen
    for turn in episode.transitions:
        assert turn.action.raw_log_probs == ((-.4, -.4) if raw else None)
        assert turn.action.behavior_log_probs == (-.5, -.5)
    assert episode_from_record(json.loads(json.dumps(asdict(episode), allow_nan=False))) == episode
    batch = pack([rollout], 16)
    np.testing.assert_array_equal(batch["behavior_log_probs"][batch["response_mask"] > 0], -.5)


def prompt_batch():
    def text(value):
        row = np.zeros((1, 8), np.int32)
        row[0, :len(value)] = list(value.encode())
        return row
    return {"prompt": np.asarray([[0, 0, 1, 2]], np.int32), "prompt_length": np.asarray([2], np.int32),
            "data_source": text("math"), "ground_truth": text("5"), "extra_info": text("")}


def test_the_prompt_source_scores_decoded_text_without_eos_and_masks_the_budget_hit():
    seen = []

    def reward(completion, record):
        seen.append((record.data_source, completion, record.ground_truth))
        return float(completion == record.ground_truth)

    [task] = prompt_tasks(prompt_batch())
    assert task.data["prompt"] == (1, 2)
    server = Server()
    prompts = PromptSource(server, reward, decode=lambda ids: " ".join(map(str, ids)), max_new_tokens=4)
    rollouts = [future.result(timeout=10) for future in prompts.submit(task, 2, version=0, key=KEY)]
    assert seen == [("math", "5", "5")] * 2
    assert server.keys == [words(jax.random.fold_in(KEY, sample)) for sample in range(2)]
    assert all(r.status == Status.COMPLETED and r.reward == 1.0 for r in rollouts)

    long = PromptSource(Server(tokens=(5, 6, 7, 8)), reward, decode=str, max_new_tokens=4)
    assert long.submit(task, 1, version=0, key=KEY)[0].result(timeout=10).status == Status.TRUNCATED
    prompts.close()
    long.close()


def test_the_prompt_sources_calls_carry_the_draws_routing_into_the_packed_batch():
    from dataclasses import replace

    from dew.objectives.rl.sessions import ROUTED_EXPERTS_KEY, pack

    class Routed(Server):
        def submit(self, prompt, max_new_tokens, *, key):
            future = super().submit(prompt, max_new_tokens, key=key)
            draw = future.result()
            routed = np.full((len(draw.prompt) + len(draw.tokens) - 1, 2, 1), 3, np.uint8)
            done = Future()
            done.set_result(replace(draw, routed_experts=routed))
            return done

    [task] = prompt_tasks(prompt_batch())
    prompts = PromptSource(Routed(), lambda *_: 1.0, decode=str, max_new_tokens=4)
    sessions = [future.result(timeout=10) for future in prompts.submit(task, 2, version=0, key=KEY)]
    prompts.close()
    batch = pack(sessions, 8)
    assert (batch[ROUTED_EXPERTS_KEY][:, :3] == 3).all()


def test_a_failed_draw_or_a_rewards_os_error_is_an_infra_error():
    class Failing(Server):
        def submit(self, prompt, max_new_tokens, *, key):
            future = Future()
            future.set_exception(ConnectionError("engine restarted"))
            return future

    def unreachable(*_):
        raise TimeoutError("reward sandbox timed out")

    [task] = prompt_tasks(prompt_batch())
    drawn = PromptSource(Failing(), lambda *_: 1.0, decode=str, max_new_tokens=4)
    rollout = drawn.submit(task, 1, version=0, key=KEY)[0].result(timeout=10)
    assert rollout.status == Status.INFRA_ERROR and "engine restarted" in rollout.detail
    scored = PromptSource(Server(), unreachable, decode=str, max_new_tokens=4)
    rollout = scored.submit(task, 1, version=0, key=KEY)[0].result(timeout=10)
    assert rollout.status == Status.INFRA_ERROR and "reward sandbox timed out" in rollout.detail
    assert len(rollout.calls) == 1
    drawn.close()
    scored.close()


class Interrupted(BaseException):
    pass


def unreadable(ids):
    raise ValueError("not a token id of this tokenizer")


def interrupted(ids):
    raise Interrupted


@pytest.mark.parametrize("decode, reward, failure", [
    (unreadable, 1.0, ValueError), (interrupted, 1.0, Interrupted), (str, float("nan"), ValueError)])
def test_a_non_finite_reward_or_a_failing_decode_fails_the_prompt_rollouts_future(decode, reward, failure):
    """Neither is an infrastructure failure to retry: the future resolves with the error instead of
    a session, and never hangs, whatever the error's class."""
    [task] = prompt_tasks(prompt_batch())
    prompts = PromptSource(Server(), lambda *_: reward, decode=decode, max_new_tokens=4)
    with pytest.raises(failure):
        prompts.submit(task, 1, version=0, key=KEY)[0].result(timeout=10)
    prompts.close()


def test_closing_the_prompt_source_cancels_the_rollouts_whose_rewards_never_started():
    """One reward thread, held in a reward, and a second draw's reward queued behind it: closing
    the source lets the running reward finish and cancels the queued rollout, so a scheduler waiting
    on it wakes, where it once waited on a future nothing would settle."""
    started, release = threading.Event(), threading.Event()

    def reward(text, record):
        started.set()
        release.wait(timeout=30)
        return 1.0

    prompts = PromptSource(Server(), reward, decode=lambda ids: "text", max_new_tokens=4, scorers=1)
    task = Task("t", {"prompt": (1, 2), "record": PromptRecord("rule", "1")})
    running, queued = prompts.submit(task, 2, version=0, key=KEY)
    assert started.wait(timeout=30)
    closing = threading.Thread(target=prompts.close)
    closing.start()
    done, _ = wait([queued], timeout=30)
    release.set()
    closing.join(timeout=30)
    assert done and queued.cancelled()
    assert running.result(timeout=30).status is Status.COMPLETED
