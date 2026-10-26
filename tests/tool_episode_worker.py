"""Run the same task cohort on one process or a real two-process CPU pool."""

import itertools
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np


def main() -> None:
    rank, processes = int(sys.argv[1]), int(sys.argv[2])
    coordinator, output, mode = sys.argv[3], Path(sys.argv[4]), sys.argv[5]
    if processes > 1:
        jax.distributed.initialize(coordinator_address=coordinator, num_processes=processes,
                                   process_id=rank, local_device_ids=[0], initialization_timeout=30)
    from episode_support import Harness, SquareSession, build, episode_scheduler

    from dew._coordination import collective_host
    from dew.data.dataset import Loader
    from dew.objectives.rl import EpisodeStatus, Observation
    from dew.training.distributed import shard_batch

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

    trainer, collector = build(environment=environment, record=records.append)
    if mode == "mismatch" and rank == 1:
        collector = replace(collector, max_turns=collector.max_turns + 1)
    scheduler, trained = episode_scheduler(collector), []

    def rollout(state, batch, key):
        """The scheduler's call, keeping the batch it hands the step."""
        trained.append(scheduler(state, batch, key))
        return trained[-1]

    trainer.rollout = rollout
    width = 2 // processes
    task_ids = np.arange(2, dtype=np.int32)[rank * width:(rank + 1) * width]
    data = Loader(
        train=lambda partition: itertools.repeat({"task_id": task_ids}), val=None, records=2, batch=2
    )
    state = None
    failed = failure = None
    try:
        state = trainer.fit(data, steps=1, log_every=1)
    except BaseException as error:
        if mode == "ok":
            raise
        failed, failure = f"{type(error).__name__}: {error}", type(error).__name__
    if mode == "ok":
        assert state is not None
        parameters = collective_host(state.variables, phase="episode pool test parameters")
        np.savez(output.with_suffix(".npz"), **{
            jax.tree_util.keystr(path): np.asarray(value)
            for path, value in jax.tree_util.tree_flatten_with_path(parameters)[0]})
        arrays = collective_host(shard_batch(trainer.device_mesh, trained[0]),
                                 phase="episode pool test batch")
        np.savez(output.with_suffix(".batch.npz"), **arrays)
    public, bindings = [], set()
    for episode in records:
        row = asdict(episode)
        bindings.add(row.pop("_binding_id"))
        public.append(row)
    output.write_text(json.dumps({
        "episodes": public, "bindings": sorted(bindings), "error": failed, "failure": failure,
        "opened": len(harness.opened),
        "closed": len(harness.closed), "updates": int(state.updates) if state is not None else None,
    }))
    if processes > 1:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
