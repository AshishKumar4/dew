"""Real CPU readers of one native or PPO episode share."""

from dataclasses import replace

import jax
import numpy as np
from episode_support import Harness, build, episode_scheduler
from process_support import serve_mode, worker_parser

from dew.data.dataset import DataPartition
from dew.training import MeshSpec


def replicated_readers(args):
    harness, verified = Harness(), []

    def verifier(episode):
        return verified.append(episode) or 0.

    if args.mode == "ppo":
        from test_ppo import build_ppo

        trainer, rollout = build_ppo(verifier)
        rollout = replace(rollout, episodes=replace(rollout.episodes, environment=harness))
    else:
        trainer, collector = build(harness, verifier=verifier)
        rollout = episode_scheduler(collector)
    trainer.mesh = MeshSpec(tensor=args.processes)
    state = trainer.place()[0]
    partition = DataPartition.of(trainer.device_mesh)
    refused = False
    try:
        rollout(state, {"task_id": np.array([31], np.int32)}, jax.random.key(23))
    except ValueError:
        refused = True
    return {"refused": refused, "readers": partition.readers, "shares": partition.count,
            "opened": len(harness.opened), "closed": len(harness.closed), "verified": len(verified)}


if __name__ == "__main__":
    modes = {"native": replicated_readers, "ppo": replicated_readers}
    serve_mode(modes, worker_parser(modes, __doc__).parse_args())
