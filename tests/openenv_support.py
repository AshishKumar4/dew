"""A guessing game served by OpenEnv's own server, for the OpenEnv source's tests.

Run as a program, it serves the game on the port its first argument names, and a step stalls
after creating the file its second argument names, as a server whose connection drops mid-step.
"""

import sys
import time
from pathlib import Path

from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State


class Guess(Action):
    number: int


class Feedback(Observation):
    hint: str = ""


class Guessing(Environment):
    """Guess the target; a wrong guess costs 0.1 and says higher or lower, the right one pays 1. A
    target below 0 has nothing to guess: the reset is already done."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    stalled: Path | None = None

    def __init__(self):
        super().__init__()
        self.target, self.steps = 0, 0

    def reset(self, seed=None, episode_id=None, **options):
        self.target, self.steps = options["target"], 0
        if self.target < 0:
            return Feedback(hint="nothing to guess", done=True)
        return Feedback(hint="guess a number from 0 to 9")

    def step(self, action, timeout_s=None, **kwargs):
        self.steps += 1
        if self.stalled is not None:
            self.stalled.touch()
            time.sleep(3600)
        if action.number == self.target:
            return Feedback(hint="right", done=True, reward=1.0)
        return Feedback(hint="higher" if action.number < self.target else "lower", reward=-0.1)

    @property
    def state(self):
        return State(step_count=self.steps)


def app():
    return create_fastapi_app(Guessing, Guess, Feedback, max_concurrent_envs=8)


if __name__ == "__main__":
    import uvicorn

    Guessing.stalled = Path(sys.argv[2])
    uvicorn.run(app(), host="127.0.0.1", port=int(sys.argv[1]), log_level="error")
