"""OpenEnv environments as episode environments: one OpenEnv session per episode, Dew drawing each turn.

OpenEnv (huggingface/OpenEnv, the `openenv` distribution, 0.9) serves an
environment over a session a client opens, `EnvClient` or its synchronous
`.sync()`: `reset(**options)` and `step(action)` each answer a `StepResult`
holding the observation, the reward of the step and whether the episode is
done. OpenEnv draws no model turn; a trainer does. `OpenEnv` adapts one
session to the `Environment` an `EnvironmentSource` runs: the rollout server
draws each turn from the exact context ids, and the session steps on the
action the drawn ids stand for.

Three adapters are the caller's, since only it knows the model's chat format
and the environment's action type:

- `prompt(observation)` gives the first context's ids, and the chat messages
  they hold (`Said`).
- `reply(observation)` gives the ids the environment's answer appends after
  the model's turn, and the messages they hold. The turn's drawn ids, EOS
  included, stay as drawn: the next context is the turn's context, its drawn
  ids, then these.
- `action(ids)` gives the environment's action for a turn's drawn ids.
  Decoding them is the caller's.

Each step's reward becomes that turn's step reward (`Observation.reward`), so
a verifier that returns 0 trains on the environment's rewards alone, and a
`done` result completes the episode, a reset's too, before any turn. The
messages each observation holds become the episode's transcript
(`Observation.messages`).

Dew imports nothing of OpenEnv. `session(task, identity)` opens the caller's
client, a context manager whose exit closes the session, and the protocols
below are what Dew reads of it. OpenEnv's client speaks to its server over a
WebSocket, and a connection that drops while a reset or a step waits raises
`websockets.exceptions.ConnectionClosed`: that call raises it as an
`InfrastructureError`, an infrastructure failure the source retries.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from websockets.exceptions import ConnectionClosed

from dew._records import JSON
from dew.inference.rollouts import Draw, InfrastructureError
from dew.objectives.rl.episodes import EpisodeId, EpisodeStatus, Observation
from dew.objectives.rl.sessions import Task


class StepResult[ObservationT](Protocol):
    """What an OpenEnv session answers a reset or a step with (`openenv.core.StepResult`)."""

    @property
    def observation(self) -> ObservationT: ...

    @property
    def reward(self) -> float | None: ...

    @property
    def done(self) -> bool: ...


class Session[ObservationT, ActionT](Protocol):
    """One open OpenEnv session, as `EnvClient.sync()` serves it."""

    def reset(self, **options: JSON) -> StepResult[ObservationT]: ...

    def step(self, action: ActionT) -> StepResult[ObservationT]: ...


@dataclass(frozen=True)
class Said:
    """What an observation adds to the model's context: its `ids`, and the chat `messages` they hold."""

    ids: tuple[int, ...]
    messages: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class OpenEnv[ObservationT, ActionT]:
    """Runs each episode in its own OpenEnv session (an environment factory for `EnvironmentSource`).

    `session(task, identity)` opens it, `options(task, identity)` are its
    reset's keyword arguments, and `prompt`, `reply` and `action` are the
    adapters the module docstring describes.
    """

    session: Callable[[Task, EpisodeId], contextlib.AbstractContextManager[Session[ObservationT, ActionT]]]
    prompt: Callable[[ObservationT], Said]
    reply: Callable[[ObservationT], Said]
    action: Callable[[tuple[int, ...]], ActionT]
    options: Callable[[Task, EpisodeId], Mapping[str, JSON]] = field(default=lambda task, identity: {})

    @contextlib.contextmanager
    def __call__(self, task: Task, identity: EpisodeId) -> Iterator[_Episode[ObservationT, ActionT]]:
        with self.session(task, identity) as opened:
            yield _Episode(self, opened, self.options(task, identity))


@dataclass
class _Episode[ObservationT, ActionT]:
    """One episode's OpenEnv session as an `Environment`."""

    adapters: OpenEnv[ObservationT, ActionT]
    session: Session[ObservationT, ActionT]
    options: Mapping[str, JSON]

    def reset(self) -> Observation:
        with _dropped():
            answered = self.session.reset(**self.options)
        said = self.adapters.prompt(answered.observation)
        if answered.done:
            return Observation((), EpisodeStatus.COMPLETED, reward=answered.reward, messages=said.messages)
        return Observation(said.ids, reward=answered.reward, messages=said.messages)

    def step(self, action: Draw) -> Observation:
        sent = self.adapters.action(action.tokens)
        with _dropped():
            answered = self.session.step(sent)
        said = self.adapters.reply(answered.observation)
        if answered.done:
            return Observation((), EpisodeStatus.COMPLETED, reward=answered.reward, messages=said.messages)
        return Observation(action.prompt + action.tokens + said.ids, reward=answered.reward,
                           messages=said.messages)


@contextlib.contextmanager
def _dropped() -> Iterator[None]:
    """A session call whose connection drops raises an infrastructure failure."""
    try:
        yield
    except ConnectionClosed as error:
        raise InfrastructureError(f"the OpenEnv session's connection dropped: {error}") from error


__all__ = ["OpenEnv", "Said", "Session", "StepResult"]
