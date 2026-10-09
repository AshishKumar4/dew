"""Train through real subprocess tools, optionally pausing inside a final call."""

import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
from episode_support import build

from dew.data import Dataset
from dew.objectives.rl import EpisodeJournal
from dew.rl.sandbox import SandboxLimits, SubprocessEnvironment


def main() -> None:
    directory = Path(sys.argv[1])
    mode = sys.argv[2]
    directory.mkdir(parents=True, exist_ok=True)
    command = (sys.executable, str(Path(__file__).with_name("sandbox_square_worker.py")), mode,
               str(directory / "calls.jsonl"), str(directory / "ready"))
    environment = SubprocessEnvironment(command, SandboxLimits(wall_seconds=60.))
    records = []
    trainer, rollout = build(
        environment, journal=EpisodeJournal(str(directory / "journal")), record=records.append
    )
    trainer.rollout = rollout
    data = Dataset(
        train=lambda partition: itertools.repeat({"task_id": np.arange(jax.device_count(), dtype=np.int32)}),
        val=None,
        records=None,
        batch=jax.device_count(),
    )
    state = trainer.fit(data, steps=1, log_every=1)
    np.save(directory / "parameters.npy", np.asarray(state.variables["params"]["table"]))
    (directory / "episodes.json").write_text(json.dumps([asdict(episode) for episode in records]))
    assert int(state.updates) == 1


if __name__ == "__main__":
    main()
