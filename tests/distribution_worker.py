"""One member of a `dew launch` pool training a tiny decoder on a named mesh.

tests/test_distribution.py starts it through the launcher, so the process
joins exactly as a multi-node run does: `prepare_process` reads the
coordinator, count and rank `dew launch` left in the environment. The pool
trains a few steps through `Trainer.fit` on one packed global batch, each
process reading the share of it the mesh gives it (`DataPartition.of`), and
process 0 writes the losses, the partition and, for every fsdp group of the
mesh, the processes its devices sit on. The same script on one process is
the reference the pool is compared with, and with `--float64` the float64
truth both are held to (check 1, `test_distribution.launches_as_exact`):
`--draws` trains that many runs in the one run's place, each from the run's
one initial state on a packed batch of its own (the draw's seed, from
`--first-draw`), and writes each one's losses and the dtype its loss is
computed in.
"""

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path

import numpy as np

VOCAB = 64
SEQ_LEN = 16
BATCH = 8
TINY_SHARD = 256


def packed_batch(seed: int = 0):
    """Two documents a row, the boundary moving from row to row, and the tail
    of the last row padding (segment 0)."""
    from model_support import two_documents

    rng = np.random.default_rng(seed)
    text = rng.integers(1, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)
    return {"text": text, "text_coordinates": two_documents(rng, BATCH, SEQ_LEN + 1)}


def fsdp_group_processes(mesh) -> list[list[int]]:
    """For every fsdp group of `mesh`, the process indices its devices sit on."""
    devices = np.moveaxis(mesh.devices, mesh.axis_names.index("fsdp"), -1)
    return [sorted({device.process_index for device in group})
            for group in devices.reshape(-1, devices.shape[-1])]


def rows_of(partition, batch):
    """The rows of `batch` this partition's readers read, whole in every other dimension."""
    import jax

    rows = partition.rows(BATCH)
    return jax.tree.map(lambda leaf: leaf[partition.index * rows:(partition.index + 1) * rows], batch)


def drawn(seed: int):
    """A loader's share of the packed batch `seed` draws, for ever."""
    def batches(partition):
        mine = rows_of(partition, packed_batch(seed))
        while True:
            yield mine

    return batches


class Stream:
    """A partition's share of a new packed batch at every step, drawn from
    the step's index, each after `step_seconds`, and where it stands, so a
    resumed run reads on."""

    def __init__(self, partition, step_seconds: float = 0.0):
        self.partition, self.step_seconds, self.read = partition, step_seconds, 0

    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(self.step_seconds)
        batch = packed_batch(self.read)
        self.read += 1
        return rows_of(self.partition, batch)

    def get_state(self) -> bytes:
        return str(self.read).encode()

    def set_state(self, state: bytes) -> None:
        self.read = int(state)


def state_digest(state) -> str:
    """One hash of every leaf of the trained state whole, in a fixed order, so
    two runs' states compare by it."""
    import jax
    from jax.experimental import multihost_utils

    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        whole = np.asarray(multihost_utils.process_allgather(
            jax.random.key_data(leaf) if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf,
            tiled=True))
        digest.update(jax.tree_util.keystr(path).encode() + whole.tobytes())
    return digest.hexdigest()


def placed_whole(mesh, partition) -> bool:
    """Whether a leaf whose second dimension the sequence axis splits, placed
    from this process's share, gathers back whole."""
    from jax.experimental import multihost_utils

    from dew.distributed.batches import shard_batch

    wide = np.arange(BATCH * SEQ_LEN, dtype=np.float32).reshape(BATCH, SEQ_LEN)
    rows = partition.rows(BATCH)
    placed = shard_batch(mesh, {"wide": wide[partition.index * rows:(partition.index + 1) * rows]})
    gathered = multihost_utils.process_allgather(placed["wide"], tiled=True)
    return bool(np.array_equal(np.asarray(gathered), wide))


def loss_dtype(trainer, state, batch) -> str:
    """The dtype `trainer`'s objective computes its loss in on a run's own
    trained `state`, as its step traces it: the tracker logs the loss as a
    float, which says no dtype."""
    import jax

    from dew.objectives.base import Step

    step = Step(state.microstep, jax.random.key(0), state.ema)
    with jax.set_mesh(trainer.device_mesh):
        loss = jax.eval_shape(lambda variables: trainer.objective.loss(variables, batch, step)[0].mean()[0],
                              state.variables)
    return loss.dtype.name


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


def copied(state):
    """`state` copied, for a fit that consumes the state it steps."""
    import jax

    return jax.tree.map(lambda leaf: jax.numpy.copy(leaf) if isinstance(leaf, jax.Array) else leaf, state)


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
    parser.add_argument("--draws", type=int, default=0,
                        help="runs in the one run's place, each on a packed batch of its own from the one "
                             "initial state")
    parser.add_argument("--first-draw", type=int, default=0, help="the first of those draws' seeds")
    parser.add_argument("--float64", action="store_true",
                        help="the float64 model under x64, each draw from the float32 initial state widened")
    args = parser.parse_args()

    from dew.training.runtime import prepare_process

    prepare_process()

    import jax
    from placement_parity import widened_start
    from sharded import split_axes

    if args.float64:
        jax.config.update("jax_enable_x64", val=True)

    from dew.checkpoints import Checkpoints
    from dew.data.dataset import DataPartition, Loader
    from dew.objectives.lm import LMObjective
    from dew.optim import OptimConfig
    from dew.registry import models
    from dew.training import Layout, MeshSpec, Trainer

    def share(partition):
        """A loader's share: the rows of one packed batch this partition's
        readers read, for ever; or on process 1 with --fail-at, until that
        batch."""
        mine = rows_of(partition, packed_batch())
        read = 0
        while True:
            if read == args.fail_at and jax.process_index() == 1:
                raise RuntimeError(f"injected failure reading batch {read}")
            read += 1
            time.sleep(args.step_seconds)
            yield mine

    def trainer_of(dtype, **options):
        """The run's trainer over the model computing in `dtype` (None: the model's own, float32)."""
        model = models.build("causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=2,
                             num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN, dtype=dtype)
        # The optimizer as a config, which keys the step (`Trainer._step_key`): each draw's fit reads the step
        # the first traced and lowered (`candidate_for`), where a built optimizer had each trace its own.
        return Trainer(LMObjective(model, SEQ_LEN), OptimConfig(optimizer="adam", learning_rate=1e-2),
                       key=jax.random.key(0),
                       mesh=MeshSpec(**json.loads(args.mesh)), layout=Layout(min_shard=TINY_SHARD), **options)

    losses = Losses()
    trainer = trainer_of("float64" if args.float64 else None, tracker=losses,
                         checkpoints=None if args.checkpoints is None else Checkpoints(str(args.checkpoints)))
    stream = functools.partial(Stream, step_seconds=args.step_seconds)

    def run(train, **options):
        """One run of the trainer on `train`'s shares (`Trainer.fit`): its trained state, and the losses
        it logged with the steps they were logged at."""
        before = len(losses.losses)
        state = trainer.fit(Loader(train=train, val=None, records=None, batch=BATCH), steps=args.steps,
                            log_every=1, **options)
        return state, {"losses": losses.losses[before:], "loss_steps": losses.steps[before:]}

    if args.draws:
        # Every judged run is a draw: from the one initial state on a packed batch of its own,
        # the record's state the first draw's. Each fit is handed a copy, as it consumes the state it steps.
        initial = None
        if args.float64:
            # The float32 run's start, drawn by a trainer of the float32 model, its optimizer built over the
            # run's length as each fit builds the run's (`Trainer.scheduled`), widened.
            narrowed = trainer_of(None)
            narrowed.scheduled(args.steps)
            initial = widened_start(narrowed)
        seeds = range(args.first_draw, args.first_draw + args.draws)
        runs = (run(drawn(seed), **({} if initial is None else {"state": copied(initial)})) for seed in seeds)
        state, logged = next(runs)
        first, draws = {}, [logged["losses"], *(later["losses"] for _, later in runs)]
        computed = loss_dtype(trainer, state, packed_batch(args.first_draw))
    else:
        state, first = run(stream if args.vary else share)
        draws, computed = [], None
    mesh = trainer.device_mesh
    partition = DataPartition.of(mesh)
    # Every process gathers, collectives both, before process 0 writes.
    whole, digest = placed_whole(mesh, partition), state_digest(state)
    if jax.process_index() == 0:
        args.out.write_text(json.dumps({
            "processes": jax.process_count(),
            "devices": jax.device_count(),
            "mesh": {axis: int(size) for axis, size in mesh.shape.items()},
            "fsdp_groups": fsdp_group_processes(mesh),
            "split_axes": sorted(split_axes(state.variables["params"])),
            "partition": {"count": partition.count, "readers": partition.readers},
            "placed_whole": whole,
            **first,
            "state_digest": digest,
            "draws": draws,
            "loss_dtype": computed,
        }))


if __name__ == "__main__":
    main()
