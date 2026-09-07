"""Train through real subprocess tools, optionally pausing inside a final call."""

import itertools
import json
from pathlib import Path
import sys

import jax
import numpy as np

from dew.data import Dataset
from dew.objectives.rl import EpisodeJournal, SandboxLimits, SubprocessEnvironment
from dew.objectives.rl.records import episode_record
from test_tool_episodes import build


def main() -> None:
    directory = Path(sys.argv[1])
    mode = sys.argv[2]
    directory.mkdir(parents=True, exist_ok=True)
    command = (sys.executable, str(Path(__file__).with_name("sandbox_square_worker.py")), mode,
               str(directory / "calls.jsonl"), str(directory / "ready"))
    environment = SubprocessEnvironment(command, SandboxLimits(wall_seconds=60.))
    records = []
    trainer, rollout = build(environment, journal=EpisodeJournal(str(directory / "journal")), record=records.append)
    trainer.rollout = rollout
    data = Dataset(train=lambda: itertools.repeat({"task_id": np.arange(jax.device_count(), dtype=np.int32)}),
                   val=None, records=None, batch=jax.device_count())
    state = trainer.fit(data, steps=1, log_every=1)
    np.save(directory / "parameters.npy", np.asarray(state.params["params"]["table"]))
    (directory / "episodes.json").write_text(json.dumps([episode_record(episode) for episode in records]))
    assert int(state.updates) == 1


if __name__ == "__main__":
    main()
