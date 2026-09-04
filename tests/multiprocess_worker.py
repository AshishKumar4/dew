"""One process of a real multi-process run, driven by tests/test_multiprocess.py.

This is a file and not an argument to `python -c` because grain's worker
processes re-exec the path they were started from, and "<stdin>" is not a path
they can import. Every mode records what it saw into --out and asserts nothing,
so the invariants stay in the test that reads the files back. The test imports
this module as well, which is what keeps its single-process reference run and
the spawned processes the same run in two topologies.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from dew.data import Loading

RES = 8
BATCH = 8
# The test model's parameters are far below the production shard threshold, so
# lower it or fsdp > 1 would silently mean "everything replicated".
TINY = 256


def global_images(count: int = BATCH) -> np.ndarray:
    """The global batch every topology trains on, in the range data arrives in."""
    return np.random.default_rng(0).integers(0, 256, size=(count, RES, RES, 3)).astype(np.uint8)


def row_marked_batch(count: int = BATCH) -> np.ndarray:
    """A global batch whose row i is filled with i.

    Which rows of the global batch reached which process's devices is otherwise
    only visible in the loss.
    """
    return np.stack([np.full((RES, RES, 3), row, np.uint8) for row in range(count)])


def checkpoint_dir(base: str | Path, name: str) -> Path:
    """Where a run named by one word keeps its checkpoints."""
    return Path(base) / name


def make_objective():
    """Squared error against the input through the real DiT, no randomness.

    dew is imported here rather than at module scope because a JAX backend
    opened before jax.distributed.initialize() would pin the process to its own
    devices, and this module is imported before the pool is joined.
    """
    import jax.numpy as jnp
    import optax
    from dew.artifacts import Representations
    from dew.inputs import unit_range
    from dew.nn.backbones.dit import SimpleDiT
    from dew.objectives.base import Aux, EMASpec, Objective

    class Reconstruction(Objective):
        artifact = Representations

        def __init__(self):
            self.model = SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2,
                                   mlp_ratio=1)
            self.ema = EMASpec(decay=optax.constant_schedule(0.999))

        def init(self, key):
            return self.model.init(key, jnp.ones((1, RES, RES, 3)), jnp.zeros((1,)))

        def loss(self, params, batch, step):
            data = unit_range(batch["image"])
            preds = self.model.apply(params, data, jnp.zeros((data.shape[0],), jnp.float32))
            return jnp.mean((preds - data) ** 2), Aux({})

        def evaluate(self, params, batch, step):
            # The validation split here is a token split whose contents the
            # objective has no use for: what the pass exercises is its length.
            return Representations(features=jnp.zeros((1, 1)), labels=jnp.zeros((1,), jnp.int32))

    return Reconstruction()


def build_trainer(name, checkpoint_base, fsdp=1, tracker=None):
    """The trainer a recipe would build, at the smallest size that still shards."""
    import jax
    import optax
    from dew.training import Checkpoints, Layout, MeshSpec, Trainer

    return Trainer(
        make_objective(), optax.adam(1e-3), key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp), layout=Layout(min_shard=TINY),
        checkpoints=Checkpoints(str(checkpoint_dir(checkpoint_base, name)), keep=4),
        tracker=tracker)


class LossRecorder:
    def __init__(self):
        self.losses = []

    def log(self, scalars, step):
        if "train/loss" in scalars:
            self.losses.append(scalars["train/loss"])

    def artifact(self, value, step):
        pass


class Data:
    def __init__(self, train, val=None, batch=BATCH, records=None):
        self._train, self.val, self.batch, self.records = train, val, batch, records

    def train(self):
        return self._train()

    @property
    def steps_per_epoch(self):
        return None if self.records is None else self.records // self.batch


def as_numpy(tree):
    """A pytree of plain arrays.

    In a multi-process run no single process addresses every shard, so values
    have to be gathered before numpy can read them. A global array only
    gathers with tiled=True, which replicates it and keeps its shape; a fully
    addressable one would be stacked or concatenated instead, so a single
    process takes the other branch.
    """
    import jax

    if jax.process_count() == 1:
        return jax.tree.map(np.asarray, tree)
    from jax.experimental import multihost_utils
    return multihost_utils.process_allgather(tree, tiled=True)


def indexed_loader(records: int, batch: int = BATCH):
    """A checkpointable source whose batches say which records they hold.

    Sharded by process, as every grain loader in dew is, so a process's
    position names its own shard and a resume has to hand it back to that
    process and no other.
    """
    import grain.python as pygrain

    class ToImage(pygrain.MapTransform):
        def map(self, index):
            return {"image": np.full((RES, RES, 3), index, np.uint8)}

    return pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, records, 1),
        sampler=pygrain.IndexSampler(num_records=records, shuffle=False, seed=0,
                                     num_epochs=1,
                                     shard_options=pygrain.ShardByJaxProcess()),
        operations=[ToImage(), pygrain.Batch(batch, drop_remainder=True)],
        loading=Loading(workers=0),
    )


def batch_records(batch) -> list[int]:
    """The record ids a batch out of `indexed_loader` holds."""
    return [int(value) for value in np.asarray(batch["image"])[:, 0, 0, 0]]


class BlockUntilKilled:
    """Hands out `limit` batches and then waits to be killed.

    A preemption has to land mid-epoch, between two checkpoints. The loop
    spends its waiting time asking the source for a batch, so stopping there
    leaves the process in the state a preempted run dies in, and makes the step
    it dies on the same on every machine. The wait is bounded so a test that
    never kills anything fails rather than hangs.
    """

    def __init__(self, loader, limit: int, marker: Path, timeout: float = 300.0):
        self._iterator = iter(loader)
        self._limit = limit
        self._marker = marker
        self._timeout = timeout
        self._handed = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._handed >= self._limit:
            self._marker.write_text(str(self._handed))
            deadline = time.monotonic() + self._timeout
            while time.monotonic() < deadline:
                time.sleep(0.02)
            raise RuntimeError(
                f"alive {self._timeout}s after handing out {self._handed} batches")
        batch = next(self._iterator)
        self._handed += 1
        return batch

    def get_state(self):
        return self._iterator.get_state()

    def set_state(self, state):
        self._iterator.set_state(state)


def sharding_facts(params) -> dict:
    """The layout the parameters actually landed in."""
    import jax

    leaves = jax.tree.leaves(params)
    return {
        "leaves": len(leaves),
        "specs": sorted({str(leaf.sharding.spec) for leaf in leaves}),
        "device_counts": sorted({len(leaf.sharding.device_set) for leaf in leaves}),
        "fully_addressable": sorted({bool(leaf.is_fully_addressable) for leaf in leaves}),
    }


def params_dict(params) -> dict:
    """Parameter leaves keyed by their path in the tree, as plain arrays."""
    import jax

    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    return as_numpy({jax.tree_util.keystr(keys): leaf for keys, leaf in flat})


def dump_params(path: Path, params) -> None:
    np.savez(path, **params_dict(params))


class YearAhead(datetime):
    """A clock a year off, for every process but the first in topology mode.

    Two processes on one machine read the same second nearly always, so a
    run name that came from each process's own clock would agree in a test
    and split on a pod. Skewing one clock a year makes the source visible.
    """

    @classmethod
    def now(cls, tz=None):
        return datetime.now(tz) + timedelta(days=400)


def mode_topology(args) -> dict:
    import jax
    from dew.training import runtime
    from dew.training.distributed import MeshSpec, batch_sharding, build_mesh, shard_batch

    if args.process_id > 0:
        runtime.datetime = YearAhead
    mesh = build_mesh(MeshSpec(fsdp=args.fsdp_size))
    rows = BATCH // args.processes
    local = row_marked_batch()[args.process_id * rows:(args.process_id + 1) * rows]
    sharded = shard_batch(batch_sharding(mesh), {"image": local})["image"]
    return {
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "mesh_shape": {axis: int(size) for axis, size in mesh.shape.items()},
        "mesh_devices": int(mesh.devices.size),
        "mesh_process_indices": sorted(
            {int(device.process_index) for device in mesh.devices.flatten()}),
        "batch_shape": list(sharded.shape),
        "addressable_shards": len(sharded.addressable_shards),
        "local_rows": sorted(
            {int(value) for shard in sharded.addressable_shards
             for value in np.asarray(shard.data)[:, 0, 0, 0]}),
        "run_timestamp": runtime.run_timestamp(),
        "own_year": runtime.datetime.now().year,
    }


def mode_data(args) -> dict:
    """One pass over the held-out split, which ends by itself."""
    import jax
    from dew.data import TokenWindows, local_batch

    data = TokenWindows(path=args.tokens, seq_len=args.seq_len, val_batches=None,
                        loading=Loading(workers=args.workers, threads=1,
                                        read_buffer=8, worker_buffer=1)).load(batch=BATCH)
    records, batches = [], 0
    for batch in data.val():
        window = np.asarray(batch["text"])
        # The corpus is a token ramp, so a window's first token names its record.
        records.extend(int(row[0]) // args.seq_len for row in window)
        batches += 1
    return {
        "process_index": jax.process_index(),
        "records": records,
        "batches": batches,
        "local_batch_size": local_batch(data.batch),
        "global_batch_size": data.batch,
        "train_len": data.records,
    }


def mode_packed(args) -> dict:
    import jax
    from dew.data import PackedTokens, local_batch

    data = PackedTokens(path=args.tokens, seq_len=args.seq_len, val_batches=None,
                        loading=Loading(workers=args.workers,
                                        worker_buffer=1)).load(batch=BATCH)
    documents, windows = set(), 0
    for batch in data.val():
        text = np.asarray(batch["text"])
        # Every document is one token value repeated, so the values in a
        # window name the documents packed into it. Padding and the eos that
        # closes a document are both zero and name nothing.
        documents.update(int(value) for value in text[text > 0])
        windows += len(text)
    return {
        "process_index": jax.process_index(),
        "documents": sorted(documents),
        "windows": windows,
        "local_batch_size": local_batch(data.batch),
    }


def restored_state(checkpoints):
    """The step and this process's data position a run directory holds."""
    if checkpoints.latest is None:
        return None, None
    _, position = checkpoints.restore()
    return checkpoints.latest, None if position is None else position.decode()


def mode_steps(args) -> dict:
    """`--steps` more steps of the executable fit runs, from the directory's state.

    Driven step by step here rather than through fit so every process records
    the losses it computed; fit reports them through the tracker on process 0
    alone.
    """
    from dew.training.distributed import shard_batch

    trainer = build_trainer(args.name, args.run_dir, args.fsdp_size)
    restored, _ = restored_state(trainer.checkpoints)
    state, _, _ = trainer.place()
    rows = BATCH // args.processes
    images = global_images()[args.process_id * rows:(args.process_id + 1) * rows]
    batch = shard_batch(trainer.batch_sharding, {"image": images})
    compiled = trainer.compile(state, batch)
    losses = []
    for _ in range(args.steps):
        state, _, loss, _, _ = compiled(state, None, batch)
        losses.append(float(as_numpy(loss)))
    if args.save and args.steps:
        trainer.checkpoints.save(int(as_numpy(state.step)), state, None)
        trainer.checkpoints.wait()
    dump_params(args.out.with_suffix(".npz"), state.params)
    return {
        "losses": losses,
        "step": int(as_numpy(state.step)),
        "restored_step": restored,
        "checkpoint_path": trainer.checkpoints.directory,
        "sharding": sharding_facts(state.params),
        "mesh_shape": {axis: int(size) for axis, size in trainer.device_mesh.shape.items()},
    }


class Batches:
    """A metric whose pass score is how many batches the pass scored."""
    name = "batches"

    def __init__(self):
        from dew.artifacts import Representations
        self.reads = Representations

    def __call__(self, artifact, batch):
        return 1.0

    def reduce(self, values):
        return float(np.sum(values))


def mode_fit(args) -> dict:
    trainer = build_trainer(args.name, args.run_dir, args.fsdp_size)
    # Where the checkpoint on disk left this run, read before fit trains past it.
    restored_step, restored = restored_state(trainer.checkpoints)
    rows = BATCH // args.processes
    loader = indexed_loader(args.records, rows)
    if args.block_after:
        loader = BlockUntilKilled(loader, args.block_after, Path(args.marker))
    val, available = None, None
    scored = []
    evaluate = trainer.objective.evaluate
    trainer.objective.evaluate = lambda *a: scored.append(1) or evaluate(*a)
    if args.tokens:
        # The packed token split, whose documents are strided over the
        # processes before packing. The objective's evaluation ignores the
        # batch's contents, so the split only has to shard.
        from dew.data import PackedTokens

        data = PackedTokens(path=args.tokens, seq_len=args.seq_len, val_batches=args.val_steps,
                            loading=Loading(workers=args.workers)).load(batch=BATCH)
        val = data.val
        available = sum(1 for _ in data.val())
    state = trainer.fit(Data(lambda: iter(loader), val=val, records=args.records),
                        steps=args.steps, log_every=1,
                        eval_every=args.steps if args.tokens else None,
                        checkpoint_every=args.save_every, metrics=(Batches(),))
    dump_params(args.out.with_suffix(".npz"), state.params)
    _, final_position = restored_state(trainer.checkpoints)
    return {
        "step": int(as_numpy(state.step)),
        "restored_step": restored_step,
        "restored_dataset_state": restored,
        "checkpoint_path": trainer.checkpoints.directory,
        "written_steps": sorted(int(step) for step in trainer.checkpoints._open().all_steps()),
        "dataset_state": final_position,
        "val_available": available,
        "val_batches": None if available is None else len(scored),
    }


MODES = {"topology": mode_topology, "data": mode_data, "packed": mode_packed,
         "steps": mode_steps, "fit": mode_fit}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coordinator", help="host:port of the jax.distributed service")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--fsdp-size", type=int, default=1)
    parser.add_argument("--name", default="worker")
    parser.add_argument("--run-dir")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--records", type=int, default=BATCH * 16)
    parser.add_argument("--val-steps", type=int, default=0,
                        help="validation batches per pass, over the packed split of --tokens")
    parser.add_argument("--block-after", type=int,
                        help="batches to hand out before waiting to be killed")
    parser.add_argument("--marker", help="file written once the source blocks")
    parser.add_argument("--tokens", help="directory holding train.bin and val.bin")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.coordinator:
        # What mpirun puts in the environment, which is how a pod run finds
        # its pool: jax's OMPI detector reads the size and the ranks from it,
        # and JAX_COORDINATOR_ADDRESS names the coordinator. The join and the
        # rendezvous right after it are then the recipes' own.
        os.environ.update({
            "OMPI_MCA_orte_hnp_uri": f"0.0;tcp://{args.coordinator}",
            "OMPI_COMM_WORLD_SIZE": str(args.processes),
            "OMPI_COMM_WORLD_RANK": str(args.process_id),
            "OMPI_COMM_WORLD_LOCAL_RANK": str(args.process_id),
            "JAX_COORDINATOR_ADDRESS": args.coordinator,
        })
    from dew.training.runtime import prepare_process

    prepare_process(multi_host=bool(args.coordinator))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(MODES[args.mode](args)))


if __name__ == "__main__":
    main()
