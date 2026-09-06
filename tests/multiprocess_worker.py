"""One process of a real multi-process run, driven by tests/test_multiprocess.py.

This is a file and not an argument to `python -c` because grain's worker
processes re-exec the path they were started from, and "<stdin>" is not a path
they can import. Every mode records what it saw into --out and asserts nothing,
so the invariants stay in the test that reads the files back. The test imports
this module as well, so its single-process reference run and the spawned
processes are the same run in two topologies.
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
# the threshold is lowered; at the production value fsdp > 1 would replicate
# every parameter.
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

    dew is imported here, not at module scope, because a JAX backend opened
    before jax.distributed.initialize() would pin the process to its own
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


def build_trainer(name, checkpoint_base, fsdp=1, tracker=None, local_dir=None,
                  local_every=None):
    """The trainer a recipe would build, at the smallest size that still shards."""
    import jax
    import optax
    from dew.training import Checkpoints, Layout, MeshSpec, Trainer

    return Trainer(
        make_objective(), optax.adam(1e-3), key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp), layout=Layout(min_shard=TINY),
        checkpoints=Checkpoints(str(checkpoint_dir(checkpoint_base, name)), keep=4,
                                local_directory=local_dir, local_every=local_every),
        tracker=tracker)


class LossRecorder:
    def __init__(self):
        self.losses = []

    def log(self, scalars, step):
        if "train/loss" in scalars:
            self.losses.append(scalars["train/loss"])

    def artifact(self, value, step):
        pass


class ScoreRecorder(LossRecorder):
    """Keeps the validation scores and the artifacts a pass produced."""

    def __init__(self):
        super().__init__()
        self.scores: dict = {}
        self.drawn: list = []

    def log(self, scalars, step):
        super().log(scalars, step)
        self.scores.update({name: float(value) for name, value in scalars.items()
                            if name.startswith("val/")})

    def artifact(self, value, step):
        images = getattr(value, "images", None)
        self.drawn.append({"type": type(value).__name__,
                           "shape": [] if images is None else list(np.shape(images)),
                           "captions": list(getattr(value, "captions", ()))})


def jepa_objective():
    """The smallest real JEPA: an encoder, a predictor and one target block."""
    from dew.inputs import Field
    from dew.nn.backbones.jepa import JepaPredictor
    from dew.objectives.jepa import JepaEncoder, JepaObjective, multi_block_mask

    patch = 2
    grid = (RES // patch, RES // patch)
    return JepaObjective(
        JepaEncoder(patch_size=patch, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1),
        JepaPredictor(grid=grid, emb_features=32, predictor_features=16,
                      num_layers=1, num_heads=2, mlp_ratio=1),
        multi_block_mask(grid, num_targets=1, scale=(0.2, 0.5)),
        sample=Field("image", (RES, RES, 3)))


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
    never kills anything fails on the timeout.
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
    from dew.training.distributed import MeshSpec, build_mesh, shard_batch

    if args.process_id > 0:
        runtime.datetime = YearAhead
    mesh = build_mesh(MeshSpec(fsdp=args.fsdp_size))
    rows = BATCH // args.processes
    local = row_marked_batch()[args.process_id * rows:(args.process_id + 1) * rows]
    sharded = shard_batch(mesh, {"image": local})["image"]
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


def restored_state(trainer):
    """The step and this process's data position the trainer would resume
    from, restored the way `fit` restores: onto the mesh, which is the one
    way a local checkpoint can be read on a pool."""
    import jax

    assert trainer.checkpoints is not None
    if trainer.checkpoints.latest is None:
        return None, None
    state, _, position = trainer.place()
    return int(jax.device_get(state.step)), None if position is None else position.decode()


def mode_steps(args) -> dict:
    """`--steps` more steps of the executable fit runs, from the directory's state.

    Driven step by step here, not through fit, so every process records
    the losses it computed; fit reports them through the tracker on process 0
    alone.
    """
    from dew.training.distributed import shard_batch

    trainer = build_trainer(args.name, args.run_dir, args.fsdp_size)
    checkpoints = trainer.checkpoints
    assert checkpoints is not None
    restored = checkpoints.latest
    state, _, _ = trainer.place()
    rows = BATCH // args.processes
    images = global_images()[args.process_id * rows:(args.process_id + 1) * rows]
    batch = shard_batch(trainer.device_mesh, {"image": images})
    compiled = trainer.compile(state, batch)
    losses = []
    for _ in range(args.steps):
        state, _, loss, _, _ = compiled(state, None, batch)
        losses.append(float(as_numpy(loss)))
    if args.save and args.steps:
        checkpoints.save(int(as_numpy(state.step)), state, None)
        checkpoints.wait()
    dump_params(args.out.with_suffix(".npz"), state.params)
    return {
        "losses": losses,
        "step": int(as_numpy(state.step)),
        "restored_step": restored,
        "checkpoint_path": checkpoints.directory,
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

    def merge(self, accumulated, contribution):
        return accumulated + contribution

    def finalize(self, accumulated):
        return accumulated


def mode_fit(args) -> dict:
    trainer = build_trainer(args.name, args.run_dir, args.fsdp_size,
                            local_dir=args.local_dir, local_every=args.local_every)
    checkpoints = trainer.checkpoints
    assert checkpoints is not None
    # Where the checkpoint on disk left this run, read before fit trains past it.
    restored_step, restored = restored_state(trainer)
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
    _, final_position = restored_state(trainer)
    return {
        "step": int(as_numpy(state.step)),
        "restored_step": restored_step,
        "restored_dataset_state": restored,
        "restored_from": None if restored_step is None else checkpoints.source(restored_step),
        "checkpoint_path": checkpoints.directory,
        "written_steps": sorted(int(step) for step in checkpoints._open().all_steps()),
        "local_path": None if args.local_dir is None else checkpoints.local_path,
        "local_steps": None if args.local_dir is None else sorted(
            int(step) for step in checkpoints._open_local().all_steps()),
        "dataset_state": final_position,
        "val_available": available,
        "val_batches": None if available is None else len(scored),
    }


FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROMPTS = ["a red bird", "two cats on a mat", "a harbour at dawn", "rain on the roof",
           "bread and jam", "birds at dawn", "a short note", "the sun set"]


class GlobalMean:
    """A metric that reads a whole batch field with numpy.

    The trainer gathers every field before metrics run on process zero, so
    the scalar covers the complete coordinated global batch.
    """

    name = "global_mean"

    def __init__(self):
        from dew.artifacts import ImageGrid

        self.reads = ImageGrid

    def __call__(self, artifact, batch) -> tuple[float, int]:
        values = np.asarray(batch["image"], np.float64)
        return float(values.sum()), values.size

    def merge(self, accumulated, contribution):
        return accumulated[0] + contribution[0], accumulated[1] + contribution[1]

    def finalize(self, accumulated):
        return accumulated[0] / accumulated[1]


def mode_validate(args) -> dict:
    """A real diffusion validation pass in the pool, scored by the clip metric.

    The artifact and the batch the metric reads are shards of global arrays on
    every process here, which numpy cannot read at all, so this is the topology
    a validation that gathers nothing dies in. CLIP and its tokenizer come from
    the committed tiny fixture, so nothing downloads.
    """
    import jax
    import optax
    from dew.data import Dataset
    from dew.diffusion import presets
    from dew.eval import clip
    from dew.inputs import Condition, Field, InputSpec
    from dew.inputs.encoders import CLIPText
    from dew.nn.backbones.dit import SimpleDiT
    from dew.objectives.diffusion import DiffusionObjective
    from dew.training import Layout, MeshSpec, Trainer

    tiny = str(FIXTURES / "clip" / "tiny")
    encoder = CLIPText.from_pretrained(tiny)
    rows = BATCH // args.processes
    mine = slice(args.process_id * rows, (args.process_id + 1) * rows)
    batch = {"image": global_images()[mine],
             "text": {name: value[mine] for name, value in encoder.tokenize(PROMPTS).items()}}
    objective = DiffusionObjective(
        SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1),
        presets.Flow()(),
        InputSpec(Field("image", (RES, RES, 3)), {"textcontext": Condition(encoder)}),
        guidance=None, steps=2)
    scored = ScoreRecorder()
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=args.fsdp_size), layout=Layout(min_shard=TINY),
                      checkpoints=None, tracker=scored)
    data = Dataset(train=lambda: iter([batch] * args.steps), val=lambda: iter([batch]),
                   records=BATCH * args.steps, batch=BATCH)
    state = trainer.fit(data, steps=args.steps, log_every=1, eval_every=args.steps,
                        metrics=(clip(modelname=tiny), GlobalMean()))
    return {
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "step": int(as_numpy(state.step)),
        "scores": scored.scores,
        "drawn": scored.drawn,
    }


def mode_tracked(args) -> dict:
    """A JEPA run in the pool with a tracker attached.

    The tracker draws on process zero, so its artifact has to arrive already
    complete: a renderer that reached for a shard of a global array would take
    this run down, and one that gathered from the one drawing process would
    wedge the pool in a collective the others never enter.
    """
    import jax
    import optax
    from dew.data import Dataset
    from dew.training import Layout, MeshSpec, Trainer
    from dew.training.tracker import render

    drawn: list = []

    class Drawing(LossRecorder):
        def artifact(self, value, step):
            payload = render(value)
            drawn.append({
                "type": type(value).__name__,
                "rendered": payload is not NotImplemented,
                "features": [int(size) for size in np.shape(value.features)],
                "std": float(np.std(np.asarray(value.features, np.float32))),
            })

    objective = jepa_objective()
    rows = BATCH // args.processes
    mine = slice(args.process_id * rows, (args.process_id + 1) * rows)
    batch = {"image": global_images()[mine],
             "label": np.arange(BATCH, dtype=np.int32)[mine]}
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=args.fsdp_size), layout=Layout(min_shard=TINY),
                      checkpoints=None, tracker=Drawing())
    data = Dataset(train=lambda: iter([batch] * args.steps), val=lambda: iter([batch]),
                   records=BATCH * args.steps, batch=BATCH)
    state = trainer.fit(data, steps=args.steps, log_every=1, eval_every=args.steps)
    return {"process_index": jax.process_index(), "drawn": drawn,
            "step": int(as_numpy(state.step))}


SEQ_LEN = 15
VOCAB = 64


def token_batch() -> np.ndarray:
    """The global token batch the pipeline runs train on: BATCH rows of
    SEQ_LEN + 1 ids."""
    return np.random.default_rng(0).integers(0, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)


def pipeline_trainer(stage: int, microbatches, fsdp: int):
    """A four-layer decoder on the stage axis, the same model and seed on
    every topology."""
    import jax
    import optax
    import dew.nn.backbones.causal_transformer  # registers the model built below
    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import Layout, MeshSpec, Trainer

    model = models.build("causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=4,
                         num_heads=4, num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN)
    return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
                   mesh=MeshSpec(fsdp=fsdp, stage=stage, microbatches=microbatches),
                   layout=Layout(min_shard=TINY), checkpoints=None, tracker=None)


def pipeline_losses(trainer, rows, steps: int):
    """`steps` steps of the compiled step over this process's `rows` of the
    token batch, with the losses and the final parameters."""
    from dew.training.distributed import shard_batch

    state, _, _ = trainer.place()
    batch = shard_batch(trainer.device_mesh, {"text": rows})
    compiled = trainer.compile(state, batch)
    losses = []
    for _ in range(steps):
        state, _, loss, _, _ = compiled(state, None, batch)
        losses.append(float(loss))
    return losses, state


def mode_pipeline(args) -> dict:
    """`--steps` steps of a two-stage pipeline in the pool: each process
    holds its rows of the batch and half of every stage's devices."""
    import jax

    rows = BATCH // args.processes
    mine = token_batch()[args.process_id * rows:(args.process_id + 1) * rows]
    trainer = pipeline_trainer(args.stage_size, args.microbatches, args.fsdp_size)
    losses, state = pipeline_losses(trainer, mine, args.steps)
    dump_params(args.out.with_suffix(".npz"), state.params)
    return {
        "process_index": jax.process_index(),
        "losses": losses,
        "sharding": sharding_facts(state.params),
        "mesh_shape": {axis: int(size) for axis, size in trainer.device_mesh.shape.items()},
    }


def mode_evaluation_contract(args) -> dict:
    """Tiny all-rank numerical scoring with root-only metrics and previews."""
    import jax
    import jax.numpy as jnp
    import optax
    from dew.artifacts import Representations, host
    from dew.data import Dataset
    from dew.objectives.base import Aux, Objective
    from dew.training import Trainer

    rank = jax.process_index()
    events = []
    closed = []

    class Numerical(Objective):
        artifact = Representations

        def init(self, key):
            return {"params": {"offset": jnp.zeros(())}}

        def loss(self, params, batch, step):
            return jnp.mean((batch["x"] + params["params"]["offset"]) ** 2), Aux({})

        def evaluate(self, params, batch, step):
            events.append(["score", np.asarray(jax.random.key_data(step.key)).tolist()])
            features = jax.jit(lambda x, key: x + jax.random.normal(key, x.shape))(
                batch["x"], step.key)
            global_artifact = Representations(features=features, labels=batch["x"][:, 0])
            if failure in ("deleted_first", "deleted_later", "mismatched_plan"):
                local = jax.device_put(np.ones((3, 1), np.float32), jax.local_devices()[0])
                if rank == 0 and failure != "mismatched_plan":
                    local.delete()
                local_artifact = Representations(features=local, labels=np.arange(3))
                if failure == "mismatched_plan":
                    return local_artifact if rank == 0 else global_artifact
                return ((local_artifact, global_artifact) if failure == "deleted_first"
                        else (global_artifact, local_artifact))
            if failure == "deleted_batch" and rank == 0:
                batch["a_metadata"].delete()
            return global_artifact

        def preview(self, params, batch, step, *, scored=None):
            events.append(["preview", np.asarray(jax.random.key_data(step.key)).tolist()])
            if failure == "deleted_preview":
                local = jax.device_put(np.ones((3, 1), np.float32), jax.local_devices()[0])
                if rank == 0:
                    local.delete()
                return Representations(features=local, labels=np.arange(3))
            features = jax.jit(lambda x, key: x + jax.random.normal(key, x.shape))(
                batch["x"], step.key)
            preview = host(Representations(features=features, labels=batch["x"][:, 0]))
            if rank == 0 and failure == "preview":
                raise ValueError("preview decoding failed")
            return preview

    class Mean:
        name, reads = "mean", Representations

        def __call__(self, artifact, batch):
            if rank != 0:
                raise AssertionError("host metric executed off root")
            if failure == "metric":
                raise ValueError("host metric failed")
            values = np.asarray(artifact.features)
            assert np.array_equal(artifact.labels, np.arange(8))
            assert batch["a_metadata"] == 7 and batch["a_python"] == 9
            return float(values.sum()), values.size

        def merge(self, accumulated, contribution):
            return accumulated[0] + contribution[0], accumulated[1] + contribution[1]

        def finalize(self, accumulated):
            if failure == "finalize":
                raise ValueError("metric finalization failed")
            return accumulated[0] / accumulated[1]

    class Drawing:
        def log(self, scalars, step):
            if failure == "log":
                raise ValueError("tracker logging failed")

        def artifact(self, artifact, step):
            assert rank == 0
            if failure == "render":
                raise ValueError("tracker rendering failed")
            assert np.shape(artifact.features) == (8, 1)

    def batches():
        try:
            count = 0 if failure == "empty" else (
                1 if failure == "next" or (rank == 0 and failure == "uneven") else 2)
            for _ in range(count):
                yield {"a_metadata": np.asarray(7), "a_python": 9,
                       "x": np.arange(rank * 4, (rank + 1) * 4, dtype=np.float32)[:, None]}
            if failure == "next" and rank == 1:
                raise OSError("iterator next failed")
        finally:
            closed.append(failure)

    def validation():
        if failure == "construct" and rank == 1:
            raise OSError("iterator construction failed")
        return batches()

    objective = Numerical()
    trainer = Trainer(objective, optax.sgd(.01), key=jax.random.key(37))
    state, _, _ = trainer.place()
    data = Dataset(train=batches, val=validation, records=8, batch=8)
    results = {}
    for failure in ("normal", "repeat", "untracked", "preview_only", "uneven", "empty",
                    "no_consumer", "mismatch", "duplicates", "metric", "preview", "finalize",
                    "log", "render", "construct", "next", "deleted_first", "deleted_later",
                    "deleted_batch", "deleted_preview", "mismatched_plan"):
        events.clear()
        trainer.tracker = Drawing() if rank == 0 and failure not in ("untracked", "no_consumer") else None
        metric = Mean()
        if failure == "mismatch" and rank == 1:
            metric.name = "another_metric"
        scoring = () if failure in ("preview_only", "no_consumer") else (metric,)
        if failure == "duplicates" and rank == 0:
            scoring = (metric, metric)
        try:
            result = trainer._evaluate(state, data, scoring, trainer.device_mesh, 0)
            results[failure] = {"scores": result, "events": list(events)}
        except (ValueError, RuntimeError, OSError) as error:
            results[failure] = {"error": str(error), "events": list(events)}
    return {"results": results, "closed": closed}


def mode_evaluation_replicas(args) -> dict:
    """Record counts follow logical rows rather than sequence/stage replicas."""
    import jax
    import jax.numpy as jnp
    import optax
    from dew.artifacts import Representations, host
    from dew.data import Dataset
    from dew.objectives.base import Aux, Objective
    from dew.training import MeshSpec, Trainer

    class Rows(Objective):
        def init(self, key):
            return {"params": {"offset": jnp.zeros(())}}

        def loss(self, params, batch, step):
            return params["params"]["offset"] ** 2, Aux({})

        def evaluate(self, params, batch, step):
            return Representations(features=batch["x"], labels=batch["x"][:, 0])

    class Count:
        name, reads = "count", Representations

        def __call__(self, artifact, batch):
            assert batch["a_metadata"] == 7 and batch["a_python"] == 9
            return len(artifact.features)

        def merge(self, accumulated, contribution):
            return accumulated + contribution

        def finalize(self, accumulated):
            return float(accumulated)

    mesh = MeshSpec(stage=2) if args.name == "stage" else MeshSpec(sequence=2)
    trainer = Trainer(Rows(), optax.sgd(.01), mesh=mesh, key=jax.random.key(0))
    state, _, _ = trainer.place()
    batch = {"a_metadata": np.asarray(7), "a_python": 9,
             "x": np.arange(3, dtype=np.float32)[:, None]}
    data = Dataset(train=lambda: iter([batch]), val=lambda: iter([batch]), records=3, batch=3)
    measured = trainer._evaluate(state, data, (Count(),), trainer.device_mesh, 0)
    unconsumed = trainer._evaluate(state, data, (), trainer.device_mesh, 0)
    # Plain host stays usable on root alone for local arrays outside evaluation.
    local = None
    if jax.process_index() == 0:
        local = host(jax.device_put(np.arange(3), jax.local_devices()[0])).tolist()
    return {"measured": measured, "no_consumer": unconsumed, "local": local}


MODES = {"topology": mode_topology, "data": mode_data, "packed": mode_packed,
         "steps": mode_steps, "fit": mode_fit, "validate": mode_validate,
         "tracked": mode_tracked, "pipeline": mode_pipeline,
         "evaluation_contract": mode_evaluation_contract,
         "evaluation_replicas": mode_evaluation_replicas}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coordinator", help="host:port of the jax.distributed service")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--fsdp-size", type=int, default=1)
    parser.add_argument("--stage-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int)
    parser.add_argument("--name", default="worker")
    parser.add_argument("--run-dir")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--local-dir", help="every process's local checkpoint directory")
    parser.add_argument("--local-every", type=int)
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
        # What mpirun puts in the environment, and what a pod run finds its
        # pool by: jax's OMPI detector reads the size and the ranks from it,
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
