"""Real CPU readers of one episode share and its PPO targets."""

from dataclasses import asdict, fields, replace

import jax
import numpy as np
from episode_support import EOS, Harness, build
from process_support import serve_mode, worker_parser

from dew.coordination import from_every_process
from dew.data.dataset import DataPartition
from dew.objectives.base import Step
from dew.objectives.rl import EpisodeRollout, EpisodeStatus, PPORollout
from dew.objectives.rl.records import episode_from_record
from dew.training import MeshSpec, Trainer


def native(args):
    harness, verified = Harness(), []
    trainer, rollout = build(harness, verifier=lambda episode: verified.append(episode) or 0.)
    trainer.mesh = MeshSpec(tensor=args.processes)
    state = trainer.place()[0]
    partition = DataPartition.of(trainer.device_mesh)
    refused = False
    try:
        rollout.collect(state, {"task_id": np.array([31], np.int32)}, jax.random.key(23))
    except ValueError:
        refused = True
    return {"refused": refused, "readers": partition.readers, "shares": partition.count,
            "opened": len(harness.opened), "closed": len(harness.closed), "verified": len(verified)}


def ppo(args):
    from test_ppo import build_ppo

    trainer, rollout = build_ppo()
    tasks, key = {"task_id": np.array([31], np.int32)}, jax.random.key(23)
    episodes = rollout.episodes.collect(trainer.place()[0], tasks, key)
    records = from_every_process([asdict(episode) for episode in episodes])[0]
    cohort = tuple(episode_from_record(record) for record in records)
    if args.mode == "single_action":
        first, *others = cohort
        turn = first.transitions[0]
        action = replace(turn.action, tokens=(EOS,), raw_log_probs=(-.1,), behavior_log_probs=(-.1,))
        first = replace(first, transitions=(replace(turn, action=action),), status=EpisodeStatus.COMPLETED)
        cohort = (first, *(replace(episode, status=EpisodeStatus.TRUNCATED) for episode in others))

    class CapturedEpisodes(EpisodeRollout):
        def collect(self, state, batch, key):
            return cohort

    # Replay one collected share on its readers without re-executing its tools.
    captured = CapturedEpisodes(**{field.name: getattr(rollout.episodes, field.name)
                                  for field in fields(rollout.episodes)})
    trainer = Trainer(trainer.objective, trainer.optimizer, key=trainer.key,
                      mesh=MeshSpec(tensor=args.processes))
    state = trainer.place()[0]
    partition = DataPartition.of(trainer.device_mesh)
    rollout = PPORollout(rollout.objective, captured, gamma=rollout.gamma, lam=rollout.lam)
    try:
        batch = rollout(state, tasks, key)
    except ValueError:
        return {"refused": True, "readers": partition.readers}
    keep = batch["response_mask"] != 0
    info = Step(state.microstep, key, state.ema)
    gradients = jax.grad(lambda variables: rollout.objective.scalar_loss(variables, batch, info)[0])(
        state.variables)
    return {"refused": False, "readers": partition.readers,
            **{name: batch[name][keep].tolist() for name in ("advantages", "returns", "old_values")},
            "actor_gradient": np.asarray(gradients["params"]["policy"]["table"]).tolist()}


if __name__ == "__main__":
    modes = {"native": native, "ppo": ppo, "single_action": ppo}
    serve_mode(modes, worker_parser(modes, __doc__).parse_args())
