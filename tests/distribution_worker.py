"""One member of a `dew launch` pool training a tiny decoder on a named mesh.

tests/test_distribution.py starts it through the launcher, so the process
joins exactly as a multi-node run does: `prepare_process` reads the
coordinator, count and rank `dew launch` left in the environment. Every
process feeds its own rows of one packed global batch, trains a few steps,
and process 0 writes the losses and, for every fsdp group of the mesh, the
processes its devices live on. The same script on one process of eight
devices is the reference the pool is compared with.
"""

import argparse
import json
from pathlib import Path

import numpy as np

VOCAB = 64
SEQ_LEN = 16
BATCH = 8
TINY_SHARD = 256


def packed_batch() -> dict[str, np.ndarray]:
    """Two documents a row, the boundary moving from row to row, and the tail
    of the last row padding (segment 0)."""
    rng = np.random.default_rng(0)
    text = rng.integers(1, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)
    segments = np.ones((BATCH, SEQ_LEN + 1), np.int32)
    positions = np.zeros((BATCH, SEQ_LEN + 1), np.int32)
    for row in range(BATCH):
        boundary = int(rng.integers(3, SEQ_LEN - 2))
        segments[row, boundary:] = 2
        positions[row, :boundary] = np.arange(boundary)
        positions[row, boundary:] = np.arange(SEQ_LEN + 1 - boundary)
    segments[-1, -3:] = 0
    return {"text": text, "text_segment_ids": segments, "text_positions": positions}


def fsdp_group_processes(mesh) -> list[list[int]]:
    """For every fsdp group of `mesh`, the process indices its devices sit on."""
    devices = np.moveaxis(mesh.devices, mesh.axis_names.index("fsdp"), -1)
    return [sorted({device.process_index for device in group})
            for group in devices.reshape(-1, devices.shape[-1])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mesh", required=True,
                        help="MeshSpec fields as JSON, e.g. '{\"fsdp\": 4, \"replicas\": 2}'")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    from dew.training.runtime import prepare_process

    prepare_process()

    import jax
    import optax

    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import Layout, MeshSpec, Trainer
    from dew.training.distributed import shard_batch

    model = models.build("causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=2,
                         num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN)
    trainer = Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-2), key=jax.random.key(0),
                      mesh=MeshSpec(**json.loads(args.mesh)), layout=Layout(min_shard=TINY_SHARD),
                      checkpoints=None, tracker=None)
    rows = BATCH // jax.process_count()
    mine = {name: leaf[jax.process_index() * rows:(jax.process_index() + 1) * rows]
            for name, leaf in packed_batch().items()}
    state, _, _ = trainer.place()
    batch = shard_batch(trainer.device_mesh, mine)
    step = trainer.compile(state, batch)
    losses = []
    for _ in range(args.steps):
        state, loss, _, _, _ = step(state, batch)
        losses.append(float(loss))
    if jax.process_index() == 0:
        args.out.write_text(json.dumps({
            "processes": jax.process_count(),
            "devices": jax.device_count(),
            "mesh": {axis: int(size) for axis, size in trainer.device_mesh.shape.items()},
            "fsdp_groups": fsdp_group_processes(trainer.device_mesh),
            "losses": losses,
        }))


if __name__ == "__main__":
    main()
