"""Session sources over a rollout server: in-process environments and single-turn prompts.

The server is scripted: each draw is fixed ids at the version it was
submitted under, and a held draw waits for the test. The checks: a
multi-turn session records one call per turn with each call's own version,
and its calls merge into one packed chain; the failure policy separates
infrastructure failures (environment errors, raised steps, verifier
crashes, failed draws) from truncations, which are still scored; a
cancelled session stops mid-draw and releases its environment; a server
that reports no raw likelihood under a transforming policy is refused; and
the prompt source scores decoded text and components per draw.
"""

import threading
from concurrent.futures import Future
from contextlib import contextmanager

import numpy as np
import pytest

from dew.inference import Draw
from dew.objectives.rl.episodes import EpisodeStatus, Observation
from dew.objectives.rl.sessions import Status, Task, pack
from dew.objectives.rl.sources import EnvironmentSource, PromptSource, prompt_tasks
from dew.sampling import Sampling

EOS = 9
SAMPLING = Sampling(temperature=1.0, eos_id=EOS)


class Server:
    """Draws `(5, EOS)`, or `tokens`, at the version served on submission; `hold` keeps draws pending."""

    def __init__(self, tokens=(5, EOS), raw=True, sampling=SAMPLING, hold=False):
        self.version, self.tokens, self.raw, self._sampling, self.hold = 0, tokens, raw, sampling, hold
        self.held, self.prompts = [], []

    @property
    def sampling(self):
        return self._sampling

    def submit(self, prompt, max_new_tokens, *, seed):
        self.prompts.append(tuple(prompt))
        future = Future()
        draw = Draw(tuple(prompt), self.tokens, (-.5,) * len(self.tokens),
                    (-.4,) * len(self.tokens) if self.raw else None, self.tokens[-1] == EOS, self.version)
        if self.hold:
            self.held.append((future, draw))
        else:
            future.set_result(draw)
        return future


class Counter:
    """Appends each action and one tool id to the context; completes after `turns` steps."""

    def __init__(self, turns=3, fail=None, on_step=None):
        self.turns, self.fail, self.on_step = turns, fail, on_step
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
        self.context = action.context + action.tokens + (4,)
        if self.steps == self.turns:
            return Observation((), EpisodeStatus.COMPLETED, "tests passed")
        return Observation(self.context)


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
    rollout = episodes.submit(Task("3"), 1, version=0)[0].result(timeout=10)
    episodes.close()
    assert rollout.status == Status.COMPLETED and rollout.reward == 3.0
    assert [call.version for call in rollout.calls] == [0, 1, 2]
    assert [call.prompt_ids for call in rollout.calls] == [(1, 2), (1, 2, 5, EOS, 4), (1, 2, 5, EOS, 4, 5, EOS, 4)]
    assert entered == exited and entered[0][0] == "3"
    batch = pack([rollout], 16)
    assert batch["input_ids"].shape == (1, 16) and set(batch["text_segment_ids"][0].tolist()) == {0, 1}
    np.testing.assert_array_equal(batch["versions"][0][batch["response_mask"][0] > 0], [0, 0, 1, 1, 2, 2])


@pytest.mark.parametrize("fail, detail", [("raise", "ConnectionError: sandbox went away"),
                                          ("report", "container exited")])
def test_an_environment_failure_is_an_unscored_infra_error(fail, detail):
    entered, exited = [], []
    episodes = source(Server(), factory(entered, exited, fail=fail))
    rollout = episodes.submit(Task("1"), 1, version=0)[0].result(timeout=10)
    episodes.close()
    assert rollout.status == Status.INFRA_ERROR and rollout.reward is None and detail in rollout.detail
    assert len(rollout.calls) == 1 and exited == entered


def test_a_verifier_crash_is_an_infra_error():
    def crash(task, episode):
        raise TimeoutError("verifier sandbox timed out")

    episodes = source(Server(), factory([], []), verifier=crash)
    rollout = episodes.submit(Task("1"), 1, version=0)[0].result(timeout=10)
    episodes.close()
    assert rollout.status == Status.INFRA_ERROR and "verifier sandbox timed out" in rollout.detail


@pytest.mark.parametrize("server, options, calls, detail", [
    (Server(tokens=(5, 6, 7, 8)), {}, 1, "token limit"),
    (Server(), {"max_turns": 2}, 2, "turn limit"),
    (Server(), {"max_prompt_tokens": 6}, 2, "max_prompt_tokens"),
])
def test_limits_truncate_and_the_truncation_is_still_scored(server, options, calls, detail):
    episodes = source(server, factory([], []), **options)
    rollout = episodes.submit(Task("1"), 1, version=0)[0].result(timeout=10)
    episodes.close()
    assert rollout.status == Status.TRUNCATED and detail in rollout.detail
    assert len(rollout.calls) == calls and rollout.reward == float(calls)


def test_a_cancelled_session_stops_mid_draw_and_releases_its_environment():
    server = Server(hold=True)
    entered, exited = [], []
    episodes = source(server, factory(entered, exited))
    future = episodes.submit(Task("1"), 1, version=0)[0]
    while not server.held:
        threading.Event().wait(0.01)
    episodes.cancel([future])
    rollout = future.result(timeout=10)
    episodes.close()
    assert rollout.status == Status.CANCELLED and not rollout.calls and exited == entered


def test_a_server_without_raw_likelihoods_under_a_transforming_policy_is_refused():
    server = Server(raw=False, sampling=Sampling(temperature=0.7, eos_id=EOS))
    episodes = source(server, factory([], []))
    with pytest.raises(Exception, match="no raw likelihoods"):
        episodes.submit(Task("1"), 1, version=0)[0].result(timeout=10)
    episodes.close()
    # At temperature one without filters the behavior likelihoods are the raw ones.
    episodes = source(Server(raw=False), factory([], []))
    assert episodes.submit(Task("1"), 1, version=0)[0].result(timeout=10).status == Status.COMPLETED
    episodes.close()


def prompt_batch():
    def text(value):
        row = np.zeros((1, 8), np.int32)
        row[0, :len(value)] = list(value.encode())
        return row
    return {"prompt": np.asarray([[0, 0, 1, 2]], np.int32), "prompt_length": np.asarray([2], np.int32),
            "data_source": text("math"), "ground_truth": text("5"), "extra_info": text("")}


def test_the_prompt_source_scores_decoded_text_without_eos_and_masks_the_budget_hit():
    seen = []

    def reward(source, completion, truth, info):
        seen.append((source, completion, truth))
        return float(completion == truth)

    [task] = prompt_tasks(prompt_batch())
    assert task.data["prompt"] == (1, 2)
    prompts = PromptSource(Server(), reward, decode=lambda ids: " ".join(map(str, ids)), max_new_tokens=4)
    rollouts = [future.result(timeout=10) for future in prompts.submit(task, 2, version=0)]
    assert seen == [("math", "5", "5")] * 2
    assert all(r.status == Status.COMPLETED and r.reward == 1.0 for r in rollouts)

    long = PromptSource(Server(tokens=(5, 6, 7, 8)), reward, decode=str, max_new_tokens=4)
    assert long.submit(task, 1, version=0)[0].result(timeout=10).status == Status.TRUNCATED
    prompts.close()
    long.close()


def test_a_failed_draw_or_reward_is_an_infra_error():
    class Failing(Server):
        def submit(self, prompt, max_new_tokens, *, seed):
            future = Future()
            future.set_exception(ConnectionError("engine restarted"))
            return future

    [task] = prompt_tasks(prompt_batch())
    drawn = PromptSource(Failing(), lambda *_: 1.0, decode=str, max_new_tokens=4)
    rollout = drawn.submit(task, 1, version=0)[0].result(timeout=10)
    assert rollout.status == Status.INFRA_ERROR and "engine restarted" in rollout.detail
    scored = PromptSource(Server(), lambda *_: float("nan"), decode=str, max_new_tokens=4)
    rollout = scored.submit(task, 1, version=0)[0].result(timeout=10)
    assert rollout.status == Status.INFRA_ERROR and "non-finite" in rollout.detail
    drawn.close()
    scored.close()


def test_a_prompt_source_failure_outside_the_reward_resolves_the_future_instead_of_hanging():
    # A decode that raises something other than Exception escapes the reward's handler.
    class Interrupted(BaseException):
        pass

    def decode(ids):
        raise Interrupted

    [task] = prompt_tasks(prompt_batch())
    prompts = PromptSource(Server(), lambda *_: 1.0, decode=decode, max_new_tokens=4)
    future = prompts.submit(task, 1, version=0)[0]
    with pytest.raises(Interrupted):
        future.result(timeout=10)
    prompts.close()
