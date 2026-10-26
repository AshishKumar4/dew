#!/usr/bin/env python3
"""Time the data pipeline on its own.

Iterates a dataset's training stream for a fixed number of steps and reports
throughput and per-step latency. No model is built, so when a training step
looks slow this says whether the loader is the reason.

The dataset is the chain a run reads, named as `dew data export` names one (a
Python file's `data`, a chain's `.json` record or a store's URL), so the knobs
here are the knobs a run has:

    python tools/benchmark_data.py --steps 100 --data run.py
    python tools/benchmark_data.py matrix --output data-matrix.json --work matrix-work
    python tools/benchmark_data.py caching --workload bpe --read indexed --output caching.json

The matrix runs each arm, Dew's reader and plain Grain over the same steps, in
a fresh interpreter over the same produced files.
"""

import contextlib
import cProfile
import functools
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

# Reading data needs no accelerator, and claiming one costs seconds of startup.
# An explicit JAX_PLATFORMS=cuda selects the same-session A100 measurement.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tyro
from absl import flags

from dew.data import Loading
from dew.data.dataset import Forwarding

# grain's worker processes read absl flags, and a plain script never parses them
flags.FLAGS.mark_as_parsed()

if TYPE_CHECKING:
    from dew.batch import Batch
    from dew.data.pipeline import Pipeline


def flowers(batch: int) -> "Pipeline":
    """Oxford Flowers 2.1.1 in a run's order, decoded by Dew at a reduced scale no smaller than 128
    pixels, resized to 128 and augmented on the accelerator, in batches of `batch`."""
    import tensorflow_datasets as tfds

    from dew.data import TFDS, images, presets

    source = TFDS("oxford_flowers102", version="2.1.1", decoders={"image": tfds.decode.SkipDecoding()})
    return source.shuffle(0).repeat().pipe(presets.images, 128, batch=batch, augment=True, draft=True,
                                           rejects=False, host=(images.EncodeCaptions(),))._chain()


@dataclass(frozen=True)
class Benchmark:
    data: str | None = None
    """The chain to time: a Python file whose `data` is one, a chain's `.json` record, or a store's
    URL read in batches of `batch` (`dew.cli.main.chain_of`); none times Oxford Flowers (`flowers`)."""
    batch: int = 32
    """Global batch size, where the chain states none."""
    steps: int = 100
    """Steps to time."""
    warmup: int = 5
    """Steps to run before timing starts."""
    read: Literal["memory", "indexed", "streamed"] = "indexed"
    loading: Loading | None = None
    """How to load the chain, turned on by its own flags (`--loading.workers 8`), which takes it over
    the chain's own `loading(...)`; none times the chain as a run reads it, under its own."""


def measure(batches: Iterable, steps: int, warmup: int) -> list[float]:
    """Wall time per batch after warmup, in seconds."""
    latencies = []
    # islice pulls exactly the batches that are timed: the loop never reads
    # one batch past the last measured step.
    taken = itertools.islice(batches, warmup + steps)
    last = time.perf_counter()
    for index, _ in enumerate(taken):
        now = time.perf_counter()
        if index >= warmup:
            latencies.append(now - last)
        last = now
    return latencies


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def report(latencies: list[float], batch_size: int) -> None:
    if not latencies:
        raise SystemExit("the loader ran out of batches before the first measured step")
    print(f"steps measured:   {len(latencies)}")
    print(f"samples/sec:      {len(latencies) * batch_size / sum(latencies):.1f}")
    print(f"step latency p50: {percentile(latencies, 0.5) * 1e3:.2f} ms")
    print(f"step latency p95: {percentile(latencies, 0.95) * 1e3:.2f} ms")


def benchmark(args: list[str]) -> Benchmark:
    """The benchmark `args` ask for, a `--loading.` flag turning on the tool's own loading."""
    from dew.config import flag_selected

    return tyro.cli(tyro.conf.CascadeSubcommandArgs[Benchmark],
                    args=flag_selected(args, {"loading": "loading"}))


def main(config: Benchmark) -> None:
    from dew.cli.main import chain_of
    from dew.data.compile import compile

    chain = flowers(config.batch) if config.data is None else chain_of(config.data, "train")
    if not chain.batched:
        chain = chain.repeat().batch(config.batch)
    dataset = compile(chain, read=config.read, loading=config.loading)
    print(f"{config.data or 'oxford_flowers102'}: {dataset.records} records, "
          f"batch {dataset.batch} across every process")
    from dew.data.dataset import DataPartition
    from dew.training import MeshSpec

    # The share this process reads on the plain data-parallel mesh.
    source = dataset.train(DataPartition.of(MeshSpec().build()))
    try:
        report(measure(source, config.steps, config.warmup), dataset.batch)
    finally:
        primary = sys.exception()
        try:
            close = getattr(source, "close", None)
            if close is not None:
                close()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f"Data iterator cleanup failed: {error!r}")




@dataclass(frozen=True)
class Measurement:
    """How an arm is timed and what it reads, the same for a matrix and each worker it starts: its
    seed, the batches it times after its warmup, the training steps and profiled stage batches,
    its loading workers, and its batch, tokens, sequence length and image size."""

    seed: int = 17
    steps: int = 128
    warmup: int = 8
    train_steps: int = 16
    stage_batches: int = 16
    workers: int = 2
    batch: int = 32
    tokens: int = 16384
    seq_len: int = 128
    image_size: int = 128


@dataclass(frozen=True)
class Matrix:
    output: Path = Path("data-matrix.json")
    work: Path = Path("data-matrix")
    measurement: Annotated[Measurement, tyro.conf.OmitArgPrefixes] = field(default_factory=Measurement)
    engines: tuple[Literal["dew", "dew_memory", "grain", "grain_prepacked"], ...] = ("dew", "grain")
    workloads: tuple[Literal["image", "text"], ...] = ("image", "text")
    cold: bool = True
    """Also open each arm cold: a fresh copy of its inputs, never provisioned, out of the page cache."""
    drops: bool = False
    """Read Dew's arms as chains whose steps can drop records (`Worker.drops`)."""


@dataclass(frozen=True)
class Worker:
    engine: Literal["dew", "dew_memory", "grain", "grain_prepacked"]
    workload: Literal["image", "text"]
    source: Path
    output: Path
    measurement: Annotated[Measurement, tyro.conf.OmitArgPrefixes] = field(default_factory=Measurement)
    cold: bool = False
    drops: bool = False
    """Dew's corpus read as a chain whose steps can drop records, as a web corpus's decode or a
    target count before a pack can, though these keep every record: its loading workers hand on
    rows and the reading process lays them out and batches them (row 883)."""


def file_fact(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def write_images(paths: list[Path], count: int) -> None:
    """`count` encoded 192x256 images, dealt round-robin into the ArrayRecord files `paths`."""
    import cv2
    import numpy as np
    from array_record.python.array_record_module import ArrayRecordWriter

    from dew.data.sources.codec import pack_dict_of_byte_arrays

    writers = [ArrayRecordWriter(str(path), "group_size:1") for path in paths]
    try:
        pixels = np.arange(192 * 256 * 3).reshape(192, 256, 3)
        for index in range(count):
            value = ((pixels + index * 7) % 256).astype(np.uint8)
            _, png = cv2.imencode(".png", value[..., ::-1])
            writers[index % len(writers)].write(pack_dict_of_byte_arrays({"jpg": png.tobytes(), "txt": b""}))
    finally:
        for writer in writers:
            writer.close()


def synthetic_inputs(config: Matrix) -> dict:
    measured = config.measurement
    import numpy as np


    image = config.work / "image"
    text = config.work / "text"
    image.mkdir(parents=True, exist_ok=True)
    text.mkdir(parents=True, exist_ok=True)
    path = image / "images.array_record"
    write_images([path], 4096)
    # No labels in the encoded layout: the image step uses a genuine per-image
    # squared prediction loss, so all loaders read exactly these same bytes.
    lengths = np.random.default_rng(measured.seed).integers(17, 98, 18000)
    documents = [np.concatenate(((np.arange(size - 1) + index) % 200 + 4, [255])).astype(np.int32)
                 for index, size in enumerate(lengths)]
    from dew.data import Files, Memory
    from dew.data.sinks import export

    export(Memory([{"text": document} for document in documents]), str(text), format="array_record",
           length_field="text", metadata={"token_store": True, "tokenizer": "byte", "vocab_size": 256,
                                          "dtype": "int32", "eos_id": 255})
    Files(str(image), format="array_record").provision()
    return {
        "image": {**file_fact(path), "rows": 4096, "source": str(image)},
        "text": {**file_fact(text / MANIFEST), "rows": len(documents),
                 "tokens": int(sum(lengths)), "source": str(text)},
    }


MANIFEST = "_dew_manifest.json"
"""A committed export's manifest, which names its shards and their length indexes."""


def inputs_of(source: Path, workload: str) -> tuple[str, ...]:
    """The files `workload`'s inputs under `source` are made of, without what a reader provisioned
    beside them: the image records, or the token export's manifest and the files it names."""
    if workload == "image":
        return ("images.array_record",)
    files = json.loads((source / MANIFEST).read_text())["files"]
    return (MANIFEST, *(entry["path"] for entry in files), *(entry["lengths"]["path"] for entry in files))


def evicted_copy(source: Path, workload: str, into: Path) -> Path:
    """`workload`'s input files under `source` copied to `into`, written through and dropped from the
    page cache, so a reader opens a store it has never provisioned from bytes it reads from disk.
    Clean pages leave the cache on POSIX_FADV_DONTNEED without root, where dropping every cache
    needs it."""
    shutil.rmtree(into, ignore_errors=True)
    into.mkdir(parents=True)
    for name in inputs_of(source, workload):
        (into / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, into / name)
        with (into / name).open("rb+") as copied:
            os.fsync(copied.fileno())
            os.posix_fadvise(copied.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return into


def objective_for(config: Worker):
    measured = config.measurement
    import jax.numpy as jnp
    from flax import linen as nn

    from dew.inputs import Field, InputSpec
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.objectives.supervised import Supervised

    if config.workload == "image":
        class ImageMean(nn.Module):
            @nn.compact
            def __call__(self, image):
                values = jnp.mean(image.astype(jnp.float32) / 255, axis=(1, 2))
                return nn.Dense(3)(values)

        def loss(outputs, batch):
            target = jnp.mean(batch["image"].astype(jnp.float32) / 255, axis=(1, 2))
            return jnp.mean(jnp.square(outputs - target), axis=-1)

        return Supervised(ImageMean(), loss, inputs=InputSpec(
            Field("image", (measured.image_size, measured.image_size, 3))))
    return LMObjective(CausalTransformer(vocab_size=256, emb_features=32, num_layers=1,
        num_heads=2, mlp_features=64, max_seq_len=measured.seq_len),
        seq_len=measured.seq_len)


def dew_reader(config: Worker, objective, loading: Loading):
    measured = config.measurement
    from dew.data import Files, images, presets
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition
    from dew.data.sources import TokenFiles

    if config.drops:
        chain = _dropping(config)
    elif config.workload == "image":
        # Every record trains, as the Grain arm reads them all, through the steps it runs.
        stored = Files(str(config.source), format="array_record")
        records = stored.shuffle(measured.seed).repeat()
        chain = records.pipe(presets.images, measured.image_size, batch=measured.batch, draft=True,
                             rejects=False, caption=images.caption(), host=(images.EncodeCaptions(),))
    else:
        # Packed in the store's order, then ordered, as a run reads a packed token store.
        packed = TokenFiles(str(config.source)).pipe(presets.lm, None, seq_len=measured.seq_len, batch=None,
                                                     pack="first-fit", bins=32)
        chain = packed.shuffle(measured.seed).repeat().batch(measured.batch)
    read = "memory" if config.engine == "dew_memory" else "indexed"
    data = compile(chain, read=read, provision=True, loading=loading, seed=measured.seed,
                   objective=objective.data())
    assert data.plan is not None, "a compiled run states its plan"
    plan = data.plan.record()
    return lambda: iter(data.train(DataPartition())), {
        "records": data.records, "read": read, "batch_policy": "fixed rows",
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "lowering": plan["lowering"], "exact": plan["exact"],
    }


def _dropping(config: Worker):
    """The workload's corpus as a chain whose steps can drop records and keep every one here: an image
    decode that rejects what doesn't decode, or a target count before the pack."""
    measured = config.measurement
    from dew.data import Files, presets
    from dew.data.pipeline import Pipeline
    from dew.data.sources import TokenFiles
    from dew.data.steps.text import CountTargets

    if config.workload == "image":
        rows = Pipeline(Files(str(config.source), format="array_record")).shuffle(measured.seed).repeat()
        return rows.pipe(presets.images, measured.image_size, batch=measured.batch, rejects=True)
    documents = Pipeline(TokenFiles(str(config.source))).shuffle(measured.seed).repeat()
    documents = documents.filter_map(CountTargets())
    return documents.pipe(presets.lm, None, seq_len=measured.seq_len, batch=measured.batch, bins=32)


def grain_options(rows, config: Worker):
    measured = config.measurement
    import grain.python as grain

    if measured.workers:
        rows = rows.mp_prefetch(grain.MultiprocessingOptions(
            num_workers=measured.workers, per_worker_buffer_size=2))
    return rows


def grain_reader(config: Worker, objective, loading: Loading):
    measured = config.measurement
    import grain.experimental as gx
    import grain.python as grain

    from dew.data import images
    from dew.data.images import EncodeCaptions
    from dew.data.steps.generic import Coordinated

    read_options = grain.ReadOptions(loading.threads, loading.read_buffer)
    if config.workload == "image":
        source = grain.ArrayRecordDataSource([str(config.source / "images.array_record")])
        rows = grain.MapDataset.source(source).shuffle(seed=measured.seed).repeat()
        # The images preset's own record steps, applied by Grain alone.
        rows = rows.map(images.DecodeImage("image", at_least=measured.image_size, rejects=False))
        rows = rows.map(images.Resize(measured.image_size)).map(images.caption())
        rows = rows.batch(measured.batch, drop_remainder=True).map(EncodeCaptions())
        stream = grain_options(rows.to_iter_dataset(read_options), config)
    elif config.workload == "text" and config.engine == "grain_prepacked":
        import numpy as np

        from dew.data.sources import TokenFiles

        source = TokenFiles(str(config.source)).tokens[:]
        ends = np.flatnonzero(source == 255) + 1
        documents = [{"text": np.asarray(source[start:end], np.int32)}
                     for start, end in zip(np.concatenate(([0], ends[:-1])), ends, strict=True)]
        rows = grain.MapDataset.source(documents).to_iter_dataset(read_options)
        packed = gx.FirstFitPackIterDataset(rows, length_struct={"text": measured.seq_len + 1},
                                           num_packing_bins=32, shuffle_bins=False)
        # A produced packing order is repeated and shuffled by both Dew loaders.
        stored = list(packed)
        rows = grain.MapDataset.source(stored).shuffle(seed=measured.seed).repeat()
        # Grain's packed columns as the objective and the trainer read them, as Dew's read lays them.
        rows = rows.batch(measured.batch, drop_remainder=True).map(Coordinated(("text",), None))
        stream = grain_options(rows.to_iter_dataset(read_options), config)
    else:
        from dew.data.packing import PackedSource
        from dew.data.pipeline import Pack
        from dew.data.sources import TokenFiles

        source = TokenFiles(str(config.source))
        source.provision()
        windows = PackedSource.planned(source, Pack(measured.seq_len + 1, bins=32)).records("indexed")
        rows = grain.MapDataset.source(windows).map(
            lambda row: {name: value for name, value in row.items() if not name.startswith("_dew_")})
        rows = rows.shuffle(seed=measured.seed).repeat().batch(measured.batch, drop_remainder=True)
        stream = grain_options(rows.map(Coordinated(("text",), None)).to_iter_dataset(read_options), config)
    reference = ("prepacked in-memory native-Grain first-fit list" if config.engine == "grain_prepacked" else
                 "planned first-fit windows read from token store" if config.workload == "text" else
                 "native Grain transforms")
    return lambda: iter(stream), {"read": "indexed", "batch_policy": "fixed rows", "reference": reference}


def batch_counts(batch, objective) -> dict:
    """A batch's rows and its real tokens as the objective counts them (`real_lengths`), the count
    the trainer's throughput reads, among the slots its inputs hold; an objective that counts no
    tokens (an image's) counts the rows it marks valid."""
    import numpy as np

    rows = len(next(iter(batch.values())))
    if objective.counts_tokens:
        tokens = int(np.sum(objective.real_lengths(batch)))
        return {"rows": rows, "tokens": tokens, "padded_tokens": int(np.size(batch["text"][:, :-1]))}
    valid = np.asarray(batch["valid"]) if "valid" in batch else np.ones(rows, bool)
    return {"rows": rows, "tokens": int(np.sum(valid)), "padded_tokens": int(np.size(valid))}


def signature(batch) -> tuple:
    import jax

    return tuple((jax.tree_util.keystr(path), tuple(value.shape), str(value.dtype))
                 for path, value in jax.tree_util.tree_leaves_with_path(batch))


class ResidentMemory:
    """Sample parent and aggregate loader-worker RSS, including cold materialization."""
    def __init__(self):
        self.parent_peak = self.workers_peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        import psutil

        process = psutil.Process()
        while not self.stop.is_set():
            self.parent_peak = max(self.parent_peak, process.memory_info().rss)
            rss = 0
            for child in process.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess):
                    rss += child.memory_info().rss
            self.workers_peak = max(self.workers_peak, rss)
            self.stop.wait(0.02)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def loader_measure(factory, config: Worker, objective) -> dict:
    measured = config.measurement
    first = None
    latencies, counts = [], []
    started = steady_started = time.perf_counter()
    with contextlib.closing(factory()) as reader:
        for index in range(measured.warmup + measured.steps):
            before = time.perf_counter()
            if index == measured.warmup:
                steady_started = before
            batch = next(reader)
            seconds = time.perf_counter() - before
            if first is None:
                first = time.perf_counter() - started
            if index >= measured.warmup:
                latencies.append(seconds)
                counts.append(batch_counts(batch, objective))
        # Counted batches are consumer work too: do not turn time spent counting
        # into free prefetch time and overstate steady pipeline throughput.
        seconds = time.perf_counter() - steady_started
    return {"first_batch_seconds": first, "batches": len(latencies),
            "batches_per_second": len(latencies) / seconds,
            "records_per_second": sum(row["rows"] for row in counts) / seconds,
            "tokens_per_second": sum(row["tokens"] for row in counts) / seconds,
            "latency_p50_ms": percentile(latencies, 0.5) * 1000,
            "latency_p95_ms": percentile(latencies, 0.95) * 1000}


class Counted(Forwarding):
    def __init__(self, parent, objective):
        self._source = parent
        self.objective = objective
        self.counts = queue.Queue()

    def __iter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self._source, name)

    def __next__(self):
        batch = next(self._source)
        counts = batch_counts(batch, self.objective)
        # An objective that counts no tokens, an image's, declares no FLOP count for its rows.
        counts["model_flops"] = (self.objective.training_flops(None, batch)
                                 if self.objective.counts_tokens and counts["tokens"] else None)
        self.counts.put(counts)
        return batch


def training_measure(factory, objective, config: Worker) -> dict:
    timed = config.measurement
    import jax
    import optax

    from dew.distributed.batches import DevicePrefetchIterator
    from dew.telemetry.instrumentation import model_flops_utilization
    from dew.telemetry.profile import capture_options
    from dew.training import Trainer

    trainer = Trainer(objective, optax.adam(1e-4), key=0, checkpoints=None, tracker=None)
    state, _, _ = trainer.place()
    jax.block_until_ready(state)
    compiled, compiles, measured, flops = {}, [], [], {}
    counted = Counted(factory(), objective)
    index = 0
    signals, completed_seconds, segment_started = [], 0.0, None
    with DevicePrefetchIterator(counted, trainer.device_mesh) as reader:
        while len(measured) < timed.train_steps:
            start = time.perf_counter()
            batch = next(reader)
            waited = time.perf_counter() - start
            counts = counted.counts.get_nowait()
            key = signature(batch)
            cold_shape = key not in compiled
            if cold_shape:
                jax.block_until_ready(state)
                if segment_started is not None:
                    completed_seconds += time.perf_counter() - segment_started
                    segment_started = None
                before = time.perf_counter()
                compiled[key] = trainer.compile(state, batch)
                flops[key] = None if trainer.step_flops is None else trainer.step_flops.hardware
                compiles.append({"signature": key, "seconds": time.perf_counter() - before,
                                 "flops": flops[key]})
            before = time.perf_counter()
            # The step's commit telemetry (its sixth output) is not a signal this run reads.
            state, loss, _, finite, accepted = compiled[key](state, batch)[:5]
            dispatched = time.perf_counter() - before
            signals.append((loss, finite, accepted))
            if cold_shape or index < timed.warmup:
                jax.block_until_ready((state, loss, finite, accepted))
            if index >= timed.warmup and not cold_shape:
                if segment_started is None:
                    segment_started = start
                measured.append({**counts, "wait_seconds": waited,
                    "dispatch_seconds": dispatched, "signal_index": len(signals) - 1,
                    "hlo_flops": flops[key]})
            index += 1
        jax.block_until_ready(state)
        if segment_started is not None:
            completed_seconds += time.perf_counter() - segment_started
        timeline = None
        if jax.default_backend() != "cpu":
            # After the timed steps, so the trace's own overhead is not in them, as `benchmark_step`
            # traces. A shape first seen here compiles inside the window, and its idle counts.
            with tempfile.TemporaryDirectory() as directory:
                with jax.profiler.trace(directory, profiler_options=capture_options()):
                    for _ in range(timed.train_steps):
                        batch = next(reader)
                        counted.counts.get_nowait()
                        key = signature(batch)
                        if key not in compiled:
                            compiled[key] = trainer.compile(state, batch)
                        state = compiled[key](state, batch)[0]
                    jax.block_until_ready(state)
                timeline = device_idle(directory)
    resolved = jax.device_get(signals)
    if any(not bool(finite) or not bool(accepted) for _, finite, accepted in resolved):
        raise ValueError("the real small-model update was nonfinite or not accepted")
    seconds = completed_seconds
    padded = sum(row["padded_tokens"] for row in measured)
    hlo_flops = None if any(row["hlo_flops"] is None for row in measured) else sum(
        row["hlo_flops"] for row in measured)
    model_flops = None if any(row["model_flops"] is None for row in measured) else sum(
        row["model_flops"] for row in measured)
    return {"steps": len(measured), "data_wait_ms": statistics.mean(
                row["wait_seconds"] for row in measured) * 1000,
            "step_ms": (seconds - sum(row["wait_seconds"] for row in measured)) * 1000 / len(measured),
            "dispatch_ms": statistics.mean(row["dispatch_seconds"] for row in measured) * 1000,
            "records_per_second": sum(row["rows"] for row in measured) / seconds,
            "tokens_per_second": sum(row["tokens"] for row in measured) / seconds,
            "padding_fraction": None if padded == 0 else 1 - sum(
                row["tokens"] for row in measured) / padded,
            "compile_count": len(compiles), "compiles": compiles, "device_timeline": timeline,
            "losses": [float(resolved[row["signal_index"]][0]) for row in measured],
            "scope": "real asynchronous Trainer.compile updates; compilation boundaries excluded",
            "model_mfu_bf16_peak": model_flops_utilization(model_flops, seconds),
            "model_flops_scope": "objective.training_flops on real host lengths and layout; "
                                 "custom image-mean model and frozen legacy API have no analytic count",
            "hlo_utilization_bf16_peak": model_flops_utilization(hlo_flops, seconds),
            "utilization_scope": "optimized executable work, including padding, over dense bf16 peak; "
                                 "not an analytic real-token MFU"}


def device_idle(directory: str) -> dict[str, float]:
    """The share of the traced steps' device window no kernel ran in, from the accelerator's own
    timeline (`trace_window.window_split`, summed over its devices), and the part of it a batch's
    host-to-device copy ended, the device waiting for data."""
    from trace_window import device_events, window_totals

    planes, _ = device_events(directory)
    if not planes:
        raise ValueError(f"the trace under {directory} holds no accelerator kernels")
    split = window_totals([(event.name, event.start_ns, event.end_ns) for event in events]
                          for events in planes.values())
    return {"device_idle_fraction": (split["idle_input"] + split["idle_host"]) / split["window"],
            "device_input_idle_fraction": split["idle_input"] / split["window"]}


def stage_profile(factory, batches: int = 16) -> list[dict]:
    profile = cProfile.Profile()
    with contextlib.closing(factory()) as reader, profile:
        for _ in range(batches):
            next(reader)
    profile.create_stats()
    return [{"file": Path(key[0]).name, "line": key[1], "function": key[2],
             "self_seconds": value[2], "cumulative_seconds": value[3], "calls": value[1]}
            for key, value in sorted(profile.stats.items(), key=lambda pair: pair[1][3], reverse=True)[:20]]


def matrix_worker(config: Worker) -> dict:
    measured = config.measurement
    import jax

    if config.cold:
        config = replace(config, source=evicted_copy(config.source, config.workload,
                                                     config.output.with_suffix(".inputs")))
    objective = objective_for(config)
    loading = Loading(workers=measured.workers, worker_buffer=2)
    with ResidentMemory() as rss:
        started = time.perf_counter()
        factory, policy = {"dew": dew_reader, "dew_memory": dew_reader, "grain": grain_reader,
                           "grain_prepacked": grain_reader}[
            config.engine](config, objective, loading)
        opened = time.perf_counter() - started
        loader = loader_measure(factory, config, objective)
        result = {"engine": config.engine, "workload": config.workload, "config": asdict(config),
            "policy": policy, "cold_open_seconds": opened, "loader": loader,
            "loading": asdict(loading),
            "device": jax.devices()[0].device_kind,
            "cold_scope": ("fresh process and cache home over a fresh copy of the inputs, never provisioned, "
                           "evicted from the page cache" if config.cold else
                           "fresh process and loader objects; shared OS page cache after input preparation"),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                "cpu": platform.processor(), "packages": {name: importlib.metadata.version(name)
                    for name in ("jax", "jaxlib", "numpy", "grain", "transformers")}}}
        result["first_batch_from_open_seconds"] = opened + result["loader"]["first_batch_seconds"]
        if measured.train_steps:
            result["training"] = training_measure(factory, objective, config)
        result["stages"] = stage_profile(factory, measured.stage_batches) if measured.stage_batches else []
        result["stage_scope"] = (f"separate untimed {measured.stage_batches}-batch parent-thread cProfile; "
                                 "excludes worker CPU" if measured.stage_batches else "not recorded")
    result["parent_rss_peak_bytes"] = rss.parent_peak
    result["worker_rss_peak_bytes"] = rss.workers_peak
    config.output.write_text(json.dumps(result, default=str, indent=2) + "\n")
    return result


def comparison(rows: list[dict]) -> list[dict]:
    """The data acceptance checks for each workload, Dew's reader against plain Grain over the same
    steps: steady throughput within 5%, resident memory, parent and workers, at most Grain's and
    10%, a warm first batch within 10 s of opening and a cold one within 30 s, and the device idle
    under 5% of its traced training steps' timeline, where it has an accelerator to trace. The
    training loop's own wait in `next(reader)` is reported beside it, never checked: it overlaps
    the updates still running on the device."""
    verdicts = []
    for workload in sorted({row["workload"] for row in rows}):
        arms = {row["engine"]: row for row in rows if row["workload"] == workload and not row.get("cold")}
        cold = {row["engine"]: row for row in rows if row["workload"] == workload and row.get("cold")}
        if not {"dew", "grain"} <= arms.keys():
            continue
        failed = {name: row["error"] for name, row in arms.items() if "error" in row}
        if failed:
            verdicts.append({"workload": workload, "cause": failed})
            continue
        dew, grain = arms["dew"], arms["grain"]
        metric = "records_per_second" if workload == "image" else "tokens_per_second"
        resident = {name: arm["parent_rss_peak_bytes"] + arm["worker_rss_peak_bytes"]
                    for name, arm in arms.items()}
        verdict = {"workload": workload, "metric": metric,
                   "dew_over_grain": dew["loader"][metric] / grain["loader"][metric],
                   "rss_over_grain": resident["dew"] / resident["grain"],
                   "first_batch_from_open_seconds": dew["first_batch_from_open_seconds"]}
        if "first_batch_from_open_seconds" in cold.get("dew", {}):
            verdict["cold_first_batch_from_open_seconds"] = cold["dew"]["first_batch_from_open_seconds"]
        if "training" in dew:
            training = dew["training"]
            verdict["host_data_wait_fraction"] = training["data_wait_ms"] / (
                training["data_wait_ms"] + training["step_ms"])
            verdict.update(training["device_timeline"] or {})
        verdict["checks"] = {
            "within_5_percent_of_grain": verdict["dew_over_grain"] >= 0.95,
            "rss_within_grain_plus_10_percent": verdict["rss_over_grain"] <= 1.10,
            "warm_first_batch_under_10_seconds": verdict["first_batch_from_open_seconds"] < 10,
            **({"device_idle_under_5_percent": verdict["device_idle_fraction"] < 0.05}
               if "device_idle_fraction" in verdict else {}),
            **({"cold_first_batch_under_30_seconds": verdict["cold_first_batch_from_open_seconds"] < 30}
               if "cold_first_batch_from_open_seconds" in verdict else {})}
        verdicts.append(verdict)
    return verdicts


def run_matrix(config: Matrix) -> None:
    config.work.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    prepared = config.work / "inputs.json"
    inputs = json.loads(prepared.read_text()) if prepared.is_file() else synthetic_inputs(config)
    rows = []
    cases: list[tuple[Literal["image", "text"], Literal["dew", "dew_memory", "grain", "grain_prepacked"],
                      bool]] = [(workload, engine, False) for workload in config.workloads
                                for engine in config.engines]
    if "text" in config.workloads and "grain" in config.engines and "grain_prepacked" not in config.engines:
        cases.append(("text", "grain_prepacked", False))
    if config.cold:
        cases.extend((workload, engine, True) for workload in config.workloads for engine in config.engines)
    for number, (workload, engine, cold) in enumerate(cases):
        output = config.work / f"{number}-{workload}-{engine}{'-cold' if cold else ''}.json"
        source = Path(inputs[workload]["source"])
        # A cold arm times its open and first batches only.
        timed = (replace(config.measurement, steps=4, warmup=0, train_steps=0, stage_batches=0) if cold
                 else config.measurement)
        worker = Worker(engine, workload, source.resolve(), output.resolve(), timed, cold=cold,
                        drops=config.drops)
        args = [sys.executable, str(Path(__file__).resolve()), "worker"]
        # The measurement's flags are its fields' own (`OmitArgPrefixes`).
        flags = {name: value for name, value in asdict(worker).items() if name != "measurement"}
        for name, value in {**asdict(timed), **flags}.items():
            flag = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                # tyro reads a bool as a flag and its --no- form.
                args.append(flag if value else "--no-" + flag[2:])
            elif value is not None:
                args.extend([flag, str(value)])
        environment = {**os.environ, "PYTHONPATH": str(root / "src")}
        if cold:
            # A cache home of its own, so nothing an earlier arm provisioned or compiled is found.
            environment["XDG_CACHE_HOME"] = str(output.with_suffix(".cache").resolve())
        with output.with_suffix(".log").open("w") as log:
            done = subprocess.run(args, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        row = ({"engine": engine, "workload": workload, "exit_code": done.returncode,
                "error": output.with_suffix(".log").read_text()[-4000:]}
               if done.returncode else json.loads(output.read_text()))
        rows.append({**row, "cold": cold})
        result = {"schema": 1, "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "inputs": inputs, "rows": rows,
            "comparisons": comparison(rows), "config": asdict(config)}
        config.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        arm = f"{workload}/{engine}{' cold' if cold else ''}"
        print(f"{arm}: " + (f"exit {done.returncode}" if done.returncode else
            f"{row['loader']['records_per_second']:.1f} rows/s, "
            f"{row['loader']['tokens_per_second']:.1f} tokens/s, "
            f"first batch {row['first_batch_from_open_seconds']:.2f} s"), flush=True)
    if any("error" in row for row in rows):
        raise RuntimeError(f"matrix contains failed arms; see {config.output}")



@dataclass(frozen=True)
class Scaling:
    """Streamed reads of one image corpus in `shards` files, interleaving `cycles` shards at a time."""

    output: Path = Path("data-scaling.json")
    work: Path = Path("data-matrix")
    shards: int = 16
    records: int = 4096
    cycles: tuple[int, ...] = (1, 2, 4, 8)
    steps: int = 64
    warmup: int = 4
    batch: int = 32
    image_size: int = 128


def run_scaling(config: Scaling) -> None:
    """Throughput of a streamed image read as it interleaves more shards: decoding and resizing run
    inside each shard's stream, so a wider cycle decodes more shards at once."""
    from dew.data import Files, images
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition

    directory = config.work / "shards"
    if not directory.is_dir():
        directory.mkdir(parents=True)
        write_images([directory / f"images-{index:02d}.array_record" for index in range(config.shards)],
                     config.records)
    rows = []
    for cycle in config.cycles:
        chain = (Files(str(directory), format="array_record").read("streamed")
                 .map(images.DecodeImage("image", at_least=config.image_size, rejects=False))
                 .map(images.Resize(config.image_size)).interleave(cycle).repeat().batch(config.batch))
        data = compile(chain, loading=Loading(workers=0))
        worker = Worker("dew", "image", directory, config.output, Measurement(
            steps=config.steps, warmup=config.warmup, batch=config.batch, image_size=config.image_size))
        reader = functools.partial(data.train, DataPartition())
        measured = loader_measure(reader, worker, objective_for(worker))
        rows.append({"cycle": cycle, **measured})
        print(f"cycle {cycle}: {measured['records_per_second']:.1f} rows/s", flush=True)
    base = rows[0]["records_per_second"]
    config.output.write_text(json.dumps({"schema": 1, "config": asdict(config), "rows": rows, "speedup": {
        str(row["cycle"]): row["records_per_second"] / base for row in rows}}, indent=2, default=str) + "\n")


@dataclass(frozen=True)
class Caching:
    """A finite pass over `records` records through a chain's leading map, read to its end, run each
    time (`computed`), run while filling a cache of it (`fill`) and read back from the generation that
    fill published (`replay`), each twice and a first computed pass discarded as warmup, every pass a
    fresh compile: what a cache costs and saves where its steps are a decode (`images`: PNGs of 256
    pixels decoded and resized to 128), a learned tokenizer (`bpe`: a 4,000-token byte-level BPE
    trained here over the documents of 700 words) or a field selection (`fields`). Every pass must read
    each record once, and every arm the records the computed pass read, each its id and a digest of
    what the map made (`Digest`). It qualifies which steps are `Costly` (`dew.data.pipeline`)."""

    workload: Literal["images", "bpe", "fields"] = "bpe"
    read: Literal["memory", "indexed"] = "indexed"
    records: int = 2000
    batch: int = 32
    output: Path = Path("data-caching.json")


@dataclass(frozen=True)
class Digest:
    """A record as its id and a CRC of every other field it holds, in name order: what a pass read, which
    arms that read the same records alike read the same of."""

    def __call__(self, row: "Batch") -> "Batch":
        import zlib

        import numpy as np

        crc = 0
        for name in sorted(row):
            if name != "id":
                crc = zlib.crc32(np.ascontiguousarray(np.asarray(row[name])).tobytes(), crc)
        return {"id": row["id"], "digest": np.uint32(crc)}


def caching_chain(config: Caching, directory: Path) -> "tuple[list[dict], Callable[[str | None], Pipeline]]":
    """The workload's records in memory and the chain of its leading map over them."""
    import numpy as np

    from dew.data import Memory, images, text
    from dew.data.pipeline import Pipeline
    from dew.data.steps.generic import Fields

    rng = np.random.default_rng(0)
    if config.workload == "images":
        import cv2

        rows = [{"id": np.int32(index), "image": cv2.imencode(".png", rng.integers(0, 255, (256, 256, 3),
                                                                                np.uint8))[1].tobytes()}
                for index in range(config.records)]
        steps = (images.DecodeImage("image", rejects=False), images.Resize(128))
    else:
        words = np.asarray(["the", "of", "and", "to", "in", "is", "that", "for", "it", "as", "with",
                            "was", "on", "be", "by", "this", "are", "from", "at", "or", "an", "have",
                            "not", "which", "data", "model", "train", "token", "layer", "weight", "cache",
                            "shard", "record", "image", "text", "batch", "step", "loss", "value"])
        rows = [{"id": np.int32(index), "text": " ".join(rng.choice(words, 700))}
                for index in range(config.records)]
        steps = (Fields(("id", "text")),)
        if config.workload == "bpe":
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers
            from transformers import PreTrainedTokenizerFast

            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            tokenizer.train_from_iterator((row["text"] for row in rows[:200]), trainers.BpeTrainer(
                vocab_size=4000, special_tokens=["[UNK]", "</s>"]))
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="</s>").save_pretrained(directory)
            steps = (text.tokenize(str(directory)),)
    def chain(cache: str | None) -> Pipeline:
        """The steps over the records, cached after them where `cache` names one, in one finite pass:
        a cache holds what they make whole (a tokenized document's ids), and the batch reads each
        record's id and the digest of what they made (`Digest`)."""
        mapped = Pipeline(Memory(rows))
        for step in steps:
            mapped = mapped.map(step)
        cached = mapped if cache is None else mapped.cache(cache)
        return cached.map(Digest(), identity="digest@1").repeat(1).batch(config.batch, drop_remainder=False)

    return rows, chain


def run_caching(config: Caching) -> None:
    import numpy as np

    from dew.batch import VALID_ROWS
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition

    with tempfile.TemporaryDirectory() as work:
        _, chain = caching_chain(config, Path(work) / "tokenizer")
        root = str(Path(work) / "cache")

        def passed(cache: str | None) -> tuple[float, list[tuple[int, int]]]:
            """One pass read to its end: its seconds, and each record it read as its id and digest."""
            started = time.perf_counter()
            loader = compile(chain(cache), read=config.read, loading=Loading(workers=0, cache=False))
            stream = loader.train(DataPartition())
            try:
                # A short last batch is padded to the batch's rows, its padding marked (`VALID_ROWS`).
                read = [(int(row), int(digest)) for batch in stream
                        for row, digest, valid in zip(batch["id"], batch["digest"],
                                                      batch.get(VALID_ROWS, np.ones(len(batch["id"]), bool)),
                                                      strict=True) if valid]
            finally:
                stream.close()
            if len(read) != config.records:
                raise ValueError(f"a pass read {len(read)} of the {config.records} records")
            return time.perf_counter() - started, read

        _, expected = passed(None)
        rows: dict[str, list[float]] = {"computed": [], "fill": [], "replay": []}

        def timed(arm: str, cache: str | None) -> None:
            seconds, read = passed(cache)
            if read != expected:
                raise ValueError(f"the {arm} pass read other records than the computed one")
            rows[arm].append(seconds)

        timed("computed", None)
        timed("computed", None)
        timed("fill", root)
        if not any(Path(root).rglob("CURRENT")):
            raise ValueError("the fill published no generation for the replay to read")
        timed("replay", root)
        timed("replay", root)
        shutil.rmtree(root)
        timed("fill", root)
    result = {"schema": 1, "config": asdict(config), "host": platform.processor() or platform.machine(),
              "cpus": os.cpu_count(), "seconds": rows}
    config.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["matrix"]:
        run_matrix(tyro.cli(Matrix, args=sys.argv[2:]))
    elif sys.argv[1:2] == ["worker"]:
        matrix_worker(tyro.cli(Worker, args=sys.argv[2:]))
    elif sys.argv[1:2] == ["scaling"]:
        run_scaling(tyro.cli(Scaling, args=sys.argv[2:]))
    elif sys.argv[1:2] == ["caching"]:
        run_caching(tyro.cli(Caching, args=sys.argv[2:]))
    else:
        main(benchmark(sys.argv[1:]))
