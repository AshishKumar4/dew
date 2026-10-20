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
`done` result completes the episode. The messages each observation holds
become the episode's transcript (`Observation.messages`). Dew imports nothing
of OpenEnv: `session(task, identity)` opens the caller's client, a context
manager whose exit closes the session, and the protocols below are what Dew
reads of it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from dew.objectives.rl.episodes import Action, EpisodeId, EpisodeStatus, Observation
from dew.objectives.rl.sessions import Task


class StepResult(Protocol):
    """What an OpenEnv session answers a reset or a step with (`openenv.core.StepResult`)."""

    @property
    def observation(self) -> object: ...

    @property
    def reward(self) -> float | None: ...

    @property
    def done(self) -> bool: ...


class Session(Protocol):
    """One open OpenEnv session, as `EnvClient.sync()` serves it."""

    def reset(self, **options: object) -> StepResult: ...

    def step(self, action: object) -> StepResult: ...


@dataclass(frozen=True)
class Said:
    """What an observation adds to the model's context: its `ids`, and the chat `messages` they hold."""

    ids: tuple[int, ...]
    messages: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class OpenEnv:
    """Runs each episode in its own OpenEnv session (an environment factory for `EnvironmentSource`).

    `session(task, identity)` opens it, `options(task, identity)` are its
    reset's keyword arguments, and `prompt`, `reply` and `action` are the
    adapters the module docstring describes.
    """

    session: Callable[[Task, EpisodeId], contextlib.AbstractContextManager[Session]]
    prompt: Callable[[object], Said]
    reply: Callable[[object], Said]
    action: Callable[[tuple[int, ...]], object]
    options: Callable[[Task, EpisodeId], Mapping[str, object]] = field(default=lambda task, identity: {})

    @contextlib.contextmanager
    def __call__(self, task: Task, identity: EpisodeId) -> Iterator[_Episode]:
        with self.session(task, identity) as opened:
            yield _Episode(self, opened, self.options(task, identity))


@dataclass
class _Episode:
    """One episode's OpenEnv session as an `Environment`."""

    adapters: OpenEnv
    session: Session
    options: Mapping[str, object]

    def reset(self) -> Observation:
        result = self.session.reset(**self.options)
        said = self.adapters.prompt(result.observation)
        return Observation(said.ids, messages=said.messages)

    def step(self, action: Action) -> Observation:
        result = self.session.step(self.adapters.action(action.tokens))
        said = self.adapters.reply(result.observation)
        if result.done:
            return Observation((), EpisodeStatus.COMPLETED, reward=result.reward, messages=said.messages)
        return Observation(action.context + action.tokens + said.ids, reward=result.reward,
                           messages=said.messages)


__all__ = ["OpenEnv", "Said", "Session", "StepResult"]
