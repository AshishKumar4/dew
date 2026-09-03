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
import time
from pathlib import Path

import numpy as np

RES = 8
BATCH = 8
# The test model's parameters are far below the production shard threshold, so
# lower it or fsdp_size > 1 would silently mean "everything replicated".
TINY = 256


def global_images(count: int = BATCH) -> np.ndarray:
    """The global batch every topology trains on, in the range data arrives in."""
    return np.random.default_rng(0).integers(
        0, 256, size=(count, RES, RES, 3)).astype(np.float32)


def row_marked_batch(count: int = BATCH) -> np.ndarray:
    """A global batch whose row i is filled with i.

    Which rows of the global batch reached which process's devices is otherwise
    only visible in the loss.
    """
    return np.stack([np.full((RES, RES, 3), row, np.float32) for row in range(count)])


def checkpoint_dir(base: str | Path, name: str) -> Path:
    """Where SimpleTrainer.checkpoint_path() puts a run named by one word."""
    return Path(base) / name


def build_trainer(name, checkpoint_base, fsdp_size=1, load=None, **kwargs):
    """The trainer a recipe would build, at the smallest size that still shards.

    dew is imported here rather than at module scope because a JAX backend
    opened before jax.distributed.initialize() would pin the process to its own
    devices, and this module is imported before the pool is joined.
    """
    import jax
    import optax
    from dew.diffusion.transforms import get_diffusion_preset
    from dew.inputs import DiffusionInputConfig
    from dew.nn.backbones.dit import SimpleDiT
    from dew.training import ObjectiveTrainer

    schedule, _, transform = get_diffusion_preset("edm")
    return ObjectiveTrainer(
        model=SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2,
                        mlp_ratio=1),
        optimizer=optax.adam(1e-3),
        noise_schedule=schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image", sample_data_shape=(RES, RES, 3), conditions=[]),
        rngs=jax.random.PRNGKey(0),
        name=name,
        wandb_config=None,
        distributed_training=True,
        fsdp_size=fsdp_size,
        fsdp_min_param_size=TINY,
        checkpoint_base_path=str(checkpoint_base),
        load_from_checkpoint=load,
        **kwargs,
    )


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


def run_losses(trainer, steps, images):
    """Per-step losses from the real compiled step over this process's rows.

    `images` is this process's slice of the global batch; shard_batch is what
    assembles the slices into the one global array the step is compiled for.
    """
    from dew.training.distributed import shard_batch

    train_step = trainer._define_train_step(batch_size=len(images))
    state, rng = trainer.state, trainer.rngstate
    losses = []
    for _ in range(steps):
        state, loss, _, rng, _ = train_step(
            state, rng, shard_batch(trainer.batch_sharding, {"image": images}))
        losses.append(float(as_numpy(loss)))
    return state, rng, losses


def indexed_loader(records: int, batch: int = BATCH):
    """A checkpointable source whose batches say which records they hold.

    Sharded by process, as every grain loader in dew is, so a process's
    position names its own shard and a resume has to hand it back to that
    process and no other.
    """
    import grain.python as pygrain

    class ToImage(pygrain.MapTransform):
        def map(self, index):
            return {"image": np.full((RES, RES, 3), index, np.float32)}

    return pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, records, 1),
        sampler=pygrain.IndexSampler(num_records=records, shuffle=False, seed=0,
                                     num_epochs=1,
                                     shard_options=pygrain.ShardByJaxProcess()),
        operations=[ToImage(), pygrain.Batch(batch, drop_remainder=True)],
        worker_count=0,
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


def mode_topology(args) -> dict:
    import jax
    from dew.training.distributed import batch_sharding, build_mesh, shard_batch

    mesh = build_mesh(args.fsdp_size)
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
    }


def mode_data(args) -> dict:
    import jax
    from dew.data.dataloaders import get_token_dataset_grain

    tokens = Path(args.tokens)
    data = get_token_dataset_grain(
        str(tokens / "train.bin"), str(tokens / "val.bin"),
        batch_size=BATCH, seq_len=args.seq_len, num_epochs=1,
        worker_count=args.workers, read_thread_count=1, read_buffer_size=8,
        worker_buffer_size=1)

    records, batches = [], 0
    for batch in data["train"]():
        window = np.asarray(batch["text"])
        # The corpus is a token ramp, so a window's first token names its record.
        records.extend(int(row[0]) // args.seq_len for row in window)
        batches += 1
    return {
        "process_index": jax.process_index(),
        "records": records,
        "batches": batches,
        "local_batch_size": data["local_batch_size"],
        "global_batch_size": data["global_batch_size"],
        "train_len": data["train_len"],
    }


def mode_packed(args) -> dict:
    import jax
    from dew.data.dataloaders import get_packed_token_dataset_grain

    tokens = Path(args.tokens)
    data = get_packed_token_dataset_grain(
        str(tokens / "train.bin"), str(tokens / "val.bin"),
        batch_size=BATCH, seq_len=args.seq_len, num_epochs=1,
        worker_count=args.workers, worker_buffer_size=1)

    documents, windows = set(), 0
    for batch in data["train"]():
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
        "local_batch_size": data["local_batch_size"],
        "train_len": data["train_len"],
    }


def mode_steps(args) -> dict:
    trainer = build_trainer(args.name, args.run_dir, args.fsdp_size, load=args.load)
    rows = BATCH // args.processes
    images = global_images()[args.process_id * rows:(args.process_id + 1) * rows]
    state, rng, losses = run_losses(trainer, args.steps, images)
    trainer.state, trainer.rngstate = state, rng
    if args.save:
        trainer.save(epoch=0, step=args.steps)
        trainer.wait_for_checkpoints()
    dump_params(args.out.with_suffix(".npz"), state.params)
    return {
        "losses": losses,
        "step": int(as_numpy(state.step)),
        "restored_step": trainer.latest_step,
        "checkpoint_path": trainer.checkpoint_path(),
        "sharding": sharding_facts(state.params),
        "mesh_shape": {axis: int(size) for axis, size in trainer.mesh.shape.items()},
    }


def scored_batches():
    """An EvaluationMetric whose epoch score is how many batches the pass scored."""
    from dew.eval.common import EvaluationMetric

    return EvaluationMetric(function=lambda artifacts, batch: 1.0, name="batches",
                            reducer=np.sum)


def mode_fit(args) -> dict:
    trainer = build_trainer(args.name, args.run_dir, args.fsdp_size, load=args.load,
                            eval_metrics=[scored_batches()])
    # Where the checkpoint on disk left this run, read before fit trains past it.
    restored, restored_step = trainer.dataset_state, trainer.latest_step
    rows = BATCH // args.processes
    loader = indexed_loader(args.records, rows)
    if args.block_after:
        loader = BlockUntilKilled(loader, args.block_after, Path(args.marker))
    data = {"train": lambda: loader, "train_len": args.records,
            "local_batch_size": rows, "global_batch_size": BATCH}
    available = None
    if args.tokens:
        # The packed token split, whose documents are strided over the
        # processes before packing. The sampler that validation runs ignores
        # an unconditional batch's contents, so the split only has to shard.
        from dew.data.dataloaders import get_packed_token_dataset_grain

        tokens = Path(args.tokens)
        data["val"] = get_packed_token_dataset_grain(
            str(tokens / "train.bin"), str(tokens / "val.bin"), batch_size=BATCH,
            seq_len=args.seq_len, num_epochs=1, worker_count=args.workers)["val"]
        available = sum(1 for _ in data["val"]())
        # 200 sampler steps per validation batch is the production default and
        # pure overhead here.
        trainer.objective.diffusion_steps = 4
    state = trainer.fit(data, training_steps_per_epoch=args.steps, epochs=1,
                        val_steps_per_epoch=args.val_steps,
                        checkpoint_every_steps=args.save_every)
    trainer.wait_for_checkpoints()
    dump_params(args.out.with_suffix(".npz"), state.params)
    return {
        "step": int(as_numpy(state.step)),
        "restored_step": restored_step,
        "restored_dataset_state": None if restored is None else restored.decode(),
        "checkpoint_path": trainer.checkpoint_path(),
        "written_steps": sorted(int(step) for step in trainer.checkpointer.all_steps()),
        "dataset_state": trainer.dataset_state.decode(),
        "val_available": available,
        "val_batches": (None if available is None
                        else int(trainer.best_val_metrics["val/batches"])),
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
    parser.add_argument("--load", help="checkpoint directory to resume from")
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
        import jax
        from jax.experimental import multihost_utils

        jax.distributed.initialize(
            coordinator_address=args.coordinator, num_processes=args.processes,
            process_id=args.process_id)
        # One collective while the processes are still in lockstep. CPU
        # collectives rendezvous through the coordinator with a 30 second
        # deadline, and the first one otherwise falls inside a checkpoint
        # barrier, by which time the processes are as far apart as their model
        # init and compile times. On a machine under load that is more than 30
        # seconds and the run dies in gloo rather than in anything under test.
        multihost_utils.sync_global_devices("worker ready")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(MODES[args.mode](args)))


if __name__ == "__main__":
    main()
