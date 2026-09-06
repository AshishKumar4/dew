"""A bounded FlowGRPO job for the real-process rollout ownership test."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import jax
import numpy as np


def main() -> None:
    rank, processes = int(sys.argv[1]), int(sys.argv[2])
    coordinator, output = sys.argv[3], Path(sys.argv[4])
    if processes > 1:
        jax.distributed.initialize(coordinator_address=coordinator, num_processes=processes,
                                   process_id=rank, local_device_ids=[0], initialization_timeout=30)

    import optax
    from dew.artifacts import collective_host
    from dew.data import Dataset
    from dew.diffusion import FlowMatchingScheduler, FlowMatchPredictionTransform, Process
    from dew.inputs import Field, InputSpec
    from dew.nn.backbones.dit import SimpleDiT
    from dew.objectives.rl import FlowGRPOObjective, FlowRollout
    from dew.training import MeshSpec, Trainer
    from dew.training.distributed import shard_batch

    inputs = InputSpec(Field("image", (4, 4, 1)))
    model = SimpleDiT(output_channels=1, patch_size=2, emb_features=8,
                      num_layers=1, num_heads=2, mlp_ratio=2)
    process = Process(FlowMatchingScheduler(shift=2), FlowMatchPredictionTransform())
    objective = FlowGRPOObjective(model, process, inputs, guidance=None, beta=0.1, steps=3)

    def reward(images, context):
        if jax.process_index() != 0:
            raise RuntimeError("the test's reward service is only available on rank zero")
        return -np.square(images.mean(axis=(1, 2, 3)) - context["target"])

    rollout = FlowRollout(objective, reward, groups=3, steps=3)
    trainer = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(70),
                      rollout=rollout, mesh=MeshSpec())
    global_batch = {"image": np.zeros((4, 4, 4, 1), np.uint8),
                    "target": np.asarray([-0.5, -0.1, 0.1, 0.5], np.float32)}
    width = 4 // processes
    local = {name: value[rank * width:(rank + 1) * width] for name, value in global_batch.items()}
    initial = trainer.place()[0]
    rolled = rollout(initial, shard_batch(trainer.device_mesh, local), jax.random.key(71))
    reassembled = shard_batch(trainer.device_mesh, rolled)
    scores = objective.log_probs(initial.params, reassembled)
    compared = collective_host((scores, reassembled["old_log_probs"], reassembled["rewards"]),
                                phase="flow test likelihoods")
    error = float(np.max(np.abs(compared[0] - compared[1])))
    data = Dataset(train=lambda: itertools.repeat(local), val=None, records=4, batch=4)
    final = trainer.fit(data, steps=1, log_every=1)
    change = float(optax.tree.norm(jax.tree.map(
        lambda a, b: a - b, final.params["params"], initial.params["params"])))
    frozen = all(np.array_equal(np.asarray(a), np.asarray(b))
                 for a, b in zip(jax.tree.leaves(final.ema), jax.tree.leaves(initial.ema)))
    params = collective_host(final.params["params"], phase="flow test parameters")
    np.savez(output.with_suffix(".npz"), **{
        jax.tree_util.keystr(path): np.asarray(value)
        for path, value in jax.tree_util.tree_flatten_with_path(params)[0]})
    output.write_text(json.dumps({
        "local_rewards": np.asarray(rolled["rewards"]).tolist(),
        "global_rewards": np.asarray(compared[2]).tolist(),
        "density_error": error, "parameter_change": change,
        "updates": int(final.updates), "reference_unchanged": frozen,
    }))
    if processes > 1:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
