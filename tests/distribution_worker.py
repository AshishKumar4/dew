"""One member of a `dew launch` pool training a tiny decoder on a named mesh.

tests/test_distribution.py starts it through the launcher, so the process
joins exactly as a multi-node run does: `prepare_process` reads the
coordinator, count and rank `dew launch` left in the environment. The pool
trains a few steps through `Trainer.fit` on one packed global batch, each
process reading the share of it the mesh gives it (`data_partition`), and
process 0 writes the losses, the partition and, for every fsdp group of the
mesh, the processes its devices sit on. The same script on one process is
the reference the pool is compared with.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

VOCAB = 64
SEQ_LEN = 16
BATCH = 8
TINY_SHARD = 256


def packed_batch(seed: int = 0) -> dict[str, np.ndarray]:
    """Two documents a row, the boundary moving from row to row, and the tail
    of the last row padding (segment 0)."""
    rng = np.random.default_rng(seed)
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


class Losses:
    """The tracker the fit reports to: the loss of every step, in order, and
    the step each was logged at."""

    def __init__(self):
        self.losses: list[float] = []
        self.steps: list[int] = []

    def log(self, scalars, step):
        if "train/loss" in scalars:
            self.losses.append(float(scalars["train/loss"]))
            self.steps.append(int(step))

    def artifact(self, value, step):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mesh", required=True,
                        help="MeshSpec fields as JSON, e.g. '{\"fsdp\": 4, \"replicas\": 2}'")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--fail-at", type=int, default=None,
                        help="process 1's loader raises when asked for this batch (0-based)")
    parser.add_argument("--checkpoints", type=Path, default=None,
                        help="the run's checkpoint directory, resumed from when it holds one")
    parser.add_argument("--step-seconds", type=float, default=0.0,
                        help="how long the loader takes over each batch, to slow the run down")
    parser.add_argument("--vary", action="store_true",
                        help="a new packed batch every step, from a stream a checkpoint can put back")
    args = parser.parse_args()

    from dew.training.runtime import prepare_process

    prepare_process()

    import jax
    import optax
    from jax.experimental import multihost_utils

    from dew.checkpoints import Checkpoints
    from dew.data import Dataset
    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import Layout, MeshSpec, Trainer, data_partition
    from dew.training.distributed import shard_batch

    def share(partition):
        """A loader's share: the rows of one packed batch this partition's
        readers read, whole in every other dimension, for ever; or on
        process 1 with --fail-at, until that batch."""
        rows = partition.rows(BATCH)
        mine = {name: leaf[partition.index * rows:(partition.index + 1) * rows]
                for name, leaf in packed_batch().items()}
        read = 0
        while True:
            if read == args.fail_at and jax.process_index() == 1:
                raise RuntimeError(f"injected failure reading batch {read}")
            read += 1
            time.sleep(args.step_seconds)
            yield mine

    class Stream:
        """A partition's share of a new packed batch at every step, drawn from
        the step's index, and where it stands, so a resumed run reads on."""

        def __init__(self, partition):
            self.partition, self.read = partition, 0

        def __iter__(self):
            return self

        def __next__(self):
            time.sleep(args.step_seconds)
            rows = self.partition.rows(BATCH)
            batch = packed_batch(self.read)
            self.read += 1
            return {name: leaf[self.partition.index * rows:(self.partition.index + 1) * rows]
                    for name, leaf in batch.items()}

        def get_state(self) -> bytes:
            return str(self.read).encode()

        def set_state(self, state: bytes) -> None:
            self.read = int(state)

    model = models.build("causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=2,
                         num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN)
    losses = Losses()
    trainer = Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-2), key=jax.random.key(0),
                      mesh=MeshSpec(**json.loads(args.mesh)), layout=Layout(min_shard=TINY_SHARD),
                      checkpoints=None if args.checkpoints is None else Checkpoints(str(args.checkpoints)),
                      tracker=losses)
    trainer.fit(Dataset(train=Stream if args.vary else share, val=None, records=None, batch=BATCH),
                steps=args.steps, log_every=1)

    # A leaf whose second dimension the sequence axis splits, placed from the
    # share and gathered back whole.
    mesh = trainer.device_mesh
    partition = data_partition(mesh)
    wide = np.arange(BATCH * SEQ_LEN, dtype=np.float32).reshape(BATCH, SEQ_LEN)
    rows = partition.rows(BATCH)
    placed = shard_batch(mesh, {"wide": wide[partition.index * rows:(partition.index + 1) * rows]})
    gathered = multihost_utils.process_allgather(placed["wide"], tiled=True)
    if jax.process_index() == 0:
        args.out.write_text(json.dumps({
            "processes": jax.process_count(),
            "devices": jax.device_count(),
            "mesh": {axis: int(size) for axis, size in mesh.shape.items()},
            "fsdp_groups": fsdp_group_processes(mesh),
            "partition": {"count": partition.count, "readers": partition.readers},
            "placed_whole": bool(np.array_equal(np.asarray(gathered), wide)),
            "losses": losses.losses,
            "loss_steps": losses.steps,
        }))


if __name__ == "__main__":
    main()
