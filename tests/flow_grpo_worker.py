"""A bounded FlowGRPO job for real-process ownership and callback precision."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import jax
import numpy as np

from dew.telemetry.records import RECORD_TYPES


def reward_service():
    """The test's reward, which rank zero alone serves, and the first values and captions it saw,
    kept outside the code under test."""
    import jax

    callback_rewards, callback_captions = [], []

    def reward(images, context):
        if jax.process_index() != 0:
            raise RuntimeError("the test's reward service is only available on rank zero")
        value = 1_000_000 - 0.01 * np.square(
            images.mean(axis=(1, 2, 3), dtype=np.float64) - np.asarray(context["target"], np.float64))
        if not callback_rewards:
            callback_rewards.append(value.copy())
            callback_captions.append([str(caption) for caption in context["caption"]])
        return value

    return reward, callback_rewards, callback_captions


def main() -> None:
    rank, processes = int(sys.argv[1]), int(sys.argv[2])
    coordinator, output = sys.argv[3], Path(sys.argv[4])
    if processes > 1:
        jax.distributed.initialize(coordinator_address=coordinator, num_processes=processes,
                                   process_id=rank, local_device_ids=[0], initialization_timeout=30)
    import optax

    from dew.artifacts import ImageGrid
    from dew.coordination import collective_host
    from dew.data.dataset import Loader
    from dew.diffusion import FlowMatchingScheduler, FlowMatchPredictionTransform, Process
    from dew.inputs import CharTable, Condition, Field, InputSpec
    from dew.nn.backbones.dit import SimpleDiT
    from dew.objectives.rl import FlowGRPOObjective, FlowRollout
    from dew.training import MeshSpec, Trainer
    from dew.training.distributed import shard_batch, split_fields, with_host

    inputs = InputSpec(Field("image", (4, 4, 1)), {
        "textcontext": Condition(CharTable.from_pretrained(tokens=3, features=4))})
    model = SimpleDiT(output_channels=1, patch_size=2, emb_features=8,
                      num_layers=1, num_heads=2, mlp_ratio=2)
    process = Process(FlowMatchingScheduler(shift=2), FlowMatchPredictionTransform())
    objective = FlowGRPOObjective(model, process, inputs, guidance=None, beta=0.1, steps=3)
    reward, callback_rewards, callback_captions = reward_service()

    class PixelMean:
        name = "pixel_mean"
        reads = ImageGrid

        def __init__(self):
            self.rows = 0

        def __call__(self, artifact, batch):
            images = np.asarray(artifact.images)
            self.rows += images.shape[0]
            return float(images.sum(dtype=np.float64)), images.size

        def merge(self, left, right):
            return left[0] + right[0], left[1] + right[1]

        def finalize(self, values):
            return values[0] / values[1]

    class Capture:
        def __init__(self):
            self.scalars = {}
            self.preview_rows = 0

        def log(self, scalars, step):
            self.scalars.update(scalars)

        def artifact(self, artifact, step):
            if isinstance(artifact, RECORD_TYPES):
                return
            self.preview_rows += artifact.images.shape[0]

    tracker, metric = Capture(), PixelMean()
    rollout = FlowRollout(objective, reward, groups=3, steps=3)
    trainer = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(70),
                      rollout=rollout, mesh=MeshSpec(), tracker=tracker)
    captions = ["red", "blue", "green", "gold"]
    # The captions stay on each process's host beside the placed rows, as a run's data keeps them.
    global_batch = {**inputs.tokenize(captions), "target": np.asarray([-0.5, -0.1, 0.1, 0.5], np.float32),
                    "caption": np.asarray(captions)}
    width = 4 // processes
    local = jax.tree.map(lambda value: value[rank * width:(rank + 1) * width], global_batch)
    initial = trainer.place()[0]
    refusal = None
    if processes > 1:
        try:
            rollout(initial, local, jax.random.key(71))
        except ValueError as error:
            refusal = str(error)
    device, host = split_fields(local)
    rolled = rollout(initial, with_host(shard_batch(trainer.device_mesh, device), host), jax.random.key(71))
    reassembled = shard_batch(trainer.device_mesh, rolled)
    scores = objective.log_probs(initial.variables, reassembled)
    compared = collective_host((scores, reassembled["old_log_probs"], reassembled["rewards"],
                                reassembled["advantages"]), phase="flow test likelihoods")
    data = Loader(
        train=lambda partition: itertools.repeat(local),
        val=lambda partition: iter((local,)),
        records=4,
        batch=4,
    )
    final = trainer.fit(data, steps=1, log_every=1, eval_every=1, metrics=(metric,), preview=True)
    change = float(optax.tree.norm(jax.tree.map(
        lambda a, b: a - b, final.variables["params"], initial.variables["params"])))
    frozen = all(np.array_equal(np.asarray(a), np.asarray(b))
                 for a, b in zip(jax.tree.leaves(final.ema), jax.tree.leaves(initial.ema), strict=True))
    params = collective_host(final.variables["params"], phase="flow test parameters")
    np.savez(output.with_suffix(".npz"), **{
        jax.tree_util.keystr(path): np.asarray(value)
        for path, value in jax.tree_util.tree_flatten_with_path(params)[0]})
    output.write_text(json.dumps({
        "local_rewards": np.asarray(rolled["rewards"]).tolist(),
        "callback_rewards": callback_rewards[0].tolist() if callback_rewards else None,
        "callback_captions": callback_captions[0] if callback_captions else None,
        "global_rewards": np.asarray(compared[2]).tolist(),
        "global_advantages": np.asarray(compared[3]).tolist(),
        "global_scores": np.asarray(compared[0]).tolist(),
        "global_old_log_probs": np.asarray(compared[1]).tolist(), "parameter_change": change,
        "updates": int(final.updates), "reference_unchanged": frozen,
        "metric_rows": metric.rows, "preview_rows": tracker.preview_rows,
        "validation_mean": tracker.scalars.get("val/pixel_mean"),
        "x64_enabled": jax.config.jax_enable_x64, "local_refusal": refusal,
    }))
    if processes > 1:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
