"""Run the same task cohort on one process or a real two-process CPU pool."""

from dataclasses import asdict, replace
from contextlib import contextmanager
import itertools
import json
from pathlib import Path
import sys

import jax
import numpy as np


def main() -> None:
    rank, processes = int(sys.argv[1]), int(sys.argv[2])
    coordinator, output, mode = sys.argv[3], Path(sys.argv[4]), sys.argv[5]
    if processes > 1:
        jax.distributed.initialize(coordinator_address=coordinator, num_processes=processes,
                                   process_id=rank, local_device_ids=[0], initialization_timeout=30)
    from dew.artifacts import collective_host
    from dew.data import Dataset
    from dew.objectives.rl import EpisodeStatus, Observation
    from dew.training.distributed import shard_batch
    from test_tool_episodes import Harness, SquareSession, build

    harness = Harness()
    records = []

    class VariableTurns(SquareSession):
        def step(self, action):
            if mode == "error" and self.identity.task == 1:
                raise OSError("rank-local tool failed")
            result = super().step(action)
            if self.identity.task == 1 and result.status == EpisodeStatus.RUNNING:
                return Observation(result.context, EpisodeStatus.TRUNCATED, result.detail)
            return result

    @contextmanager
    def environment(identity):
        try:
            yield VariableTurns(identity, harness)
        finally:
            harness.closed.append(identity)

    trainer, rollout = build(environment=environment, record=records.append)
    if mode == "mismatch" and rank == 1:
        rollout = replace(rollout, max_turns=rollout.max_turns + 1)
    trainer.rollout = rollout
    width = 2 // processes
    task_ids = np.arange(2, dtype=np.int32)[rank * width:(rank + 1) * width]
    data = Dataset(train=lambda: itertools.repeat({"task_id": task_ids}), val=None, records=2, batch=2)
    state = None
    failed = None
    try:
        state = trainer.fit(data, steps=1, log_every=1)
    except BaseException as error:
        if mode == "ok":
            raise
        failed = f"{type(error).__name__}: {error}"
    if mode == "ok":
        assert state is not None
        parameters = collective_host(state.params, phase="episode pool test parameters")
        np.savez(output.with_suffix(".npz"), allow_pickle=False, **{
            jax.tree_util.keystr(path): np.asarray(value)
            for path, value in jax.tree_util.tree_flatten_with_path(parameters)[0]})
        batch = rollout.project(records)
        arrays = collective_host(shard_batch(trainer.device_mesh, batch), phase="episode pool test batch")
        np.savez(output.with_suffix(".batch.npz"), **arrays)
    public = []
    for episode in records:
        row = asdict(episode)
        row.pop("_binding_id")
        for transition in row["transitions"]:
            transition["action"].pop("_binding_id")
        public.append(row)
    output.write_text(json.dumps({
        "episodes": public, "error": failed, "opened": len(harness.opened),
        "closed": len(harness.closed), "updates": int(state.updates) if state is not None else None,
    }))
    if processes > 1:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
