#!/usr/bin/env python3
"""A small real workflow of Dew, run by the interpreter of a clean environment its wheel was installed into.

    <environment>/bin/python -P tools/bounds_workflow.py --scratch DIR [--extra NAME ...] [--report FILE]

`-P` keeps this file's directory off `sys.path`, and the caller runs it from outside the checkout, so nothing
of the checkout is importable: every Dew module and every dependency is the installed one. Each step is one
thing a user does with the library, checked against a value computed another way:

- every module the wheel installed imports;
- the console entry point and a subcommand's help (tyro);
- a grouped-query decoder's jitted forward, an Engram's bucket primes (sympy) and a Gaussian cutoff (scipy);
- a language model trained through grain's worker processes over a token store exported as ArrayRecord;
- a checkpoint saved and resumed: the restored state is the saved one, and the run resumed from step 2 ends at
  step 4 bit for bit as an uninterrupted run of four does, its optimizer state and data position included;
- a decoder directory written with transformers' config and safetensors, loaded, saved and loaded again;
- records read compressed (zstandard) and as Parquet (pyarrow), a file and an HTTP url (aiohttp) fetched, an
  image decoded (opencv) and resized.

With `--extra`, the flow of each named extra follows (`FLOWS`): one public feature of that extra, run with the
extra installed and nothing else. `tools/check_bounds.py` makes the environment and runs this;
`tools/smoke_wheel.py` runs the same file in the one a user's pip makes. Every step runs, and the exit status
is 1 if any failed.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import http.server
import importlib
import importlib.metadata
import json
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

DISTRIBUTION = "dewml"
"""The name the wheel installs under."""
STEPS: dict[str, Callable[[Path], None]] = {}
"""Each base step by its name, in the order it runs (`step`)."""
FLOWS: dict[str, Callable[[Path], None]] = {}
"""Each extra's flow by the extra's name (`flow`): what its feature does with the extra installed."""
SELECTED: tuple[str, ...] = ()
"""The extras this run selects, which `imports` reads (`main`)."""


def step(function: Callable[[Path], None]) -> Callable[[Path], None]:
    """Register `function` as the base step named for it; it takes a directory of its own to write in."""
    STEPS[function.__name__] = function
    return function


def flow(extra: str) -> Callable[[Callable[[Path], None]], Callable[[Path], None]]:
    """Register a function as `extra`'s flow."""
    def register(function: Callable[[Path], None]) -> Callable[[Path], None]:
        FLOWS[extra] = function
        return function
    return register


def run(*command: str | Path, **options) -> subprocess.CompletedProcess[str]:
    """`command` run to its end, its output captured; a failure carries both streams' tails, which a bare
    `CalledProcessError` leaves out."""
    done = subprocess.run([str(part) for part in command], capture_output=True, text=True, check=False,
                          **options)
    if done.returncode:
        raise RuntimeError(f"{' '.join(map(str, command))} exited {done.returncode}:\n"
                           f"{done.stdout[-1500:]}\n{done.stderr[-2500:]}")
    return done


def requirements(selected: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """The distributions the wheel requires of every install and of the `selected` extras (the first), and
    those only an extra not selected requires (the second), by canonical name, from the wheel's own
    metadata."""
    extras = importlib.metadata.metadata(DISTRIBUTION).get_all("Provides-Extra") or []
    needed: set[str] = set()
    optional: set[str] = set()
    for requirement in map(Requirement, importlib.metadata.requires(DISTRIBUTION) or ()):
        name = canonicalize_name(requirement.name)
        marker = requirement.marker
        if marker is None or any(marker.evaluate({"extra": extra}) for extra in ("", *selected)):
            needed.add(name)
        elif any(marker.evaluate({"extra": extra}) for extra in extras):
            optional.add(name)
    return needed, optional - needed


@step
def imports(scratch: Path) -> None:
    """Every module the wheel installed imports. One that cannot for want of a package only an extra this run
    did not select requires is listed and passed over, the missing module named by the exception
    (`ModuleNotFoundError.name`) and matched to the wheel's own requirements; any other failure, a package
    that is there and incompatible among them, fails the step."""
    files = importlib.metadata.distribution(DISTRIBUTION).files or ()
    modules = sorted(".".join(path.with_suffix("").parts).removesuffix(".__init__")
                     for path in files if path.suffix == ".py" and path.parts[0] == "dew")
    assert modules, "the wheel lists no module of dew"
    absent = requirements(SELECTED)[1]
    skipped, failed = [], []
    for name in modules:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name and canonicalize_name(error.name.split(".")[0]) in absent:
                skipped.append(f"{name} needs {error.name}")
            else:
                failed.append(f"{name}: {error!r}")
        except Exception as error:  # the failure is reported with its module, and fails the step
            failed.append(f"{name}: {error!r}")
    print(f"imported {len(modules) - len(skipped) - len(failed)} of {len(modules)} modules; passed over for "
          f"an extra not selected: {skipped}")
    assert not failed, "modules that fail to import:\n" + "\n".join(failed)


@step
def command_line(scratch: Path) -> None:
    """The console entry point answers `--help`, and so does one of its subcommands (tyro)."""
    entry = Path(sys.executable).parent / "dew"
    for arguments in (["--help"], ["export", "--help"]):
        assert "dew" in run(entry, *arguments, cwd=scratch).stdout, arguments


@step
def decoder(scratch: Path) -> None:
    """A grouped-query decoder's jitted forward is finite and as wide as its vocabulary; an Engram's bucket
    primes are the primes above its vocabulary (sympy); the Gaussian cutoff of a row is its mean plus its
    deviation times the quantile (scipy)."""
    import jax
    import jax.numpy as jnp
    import scipy.special

    from dew.nn.backbones import CausalTransformer
    from dew.nn.engram import Engram
    from dew.nn.mlp import gaussian_cutoff

    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=4, num_kv_heads=2,
                              mlp_features=64, max_seq_len=16)
    ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)
    logits = jax.jit(model.apply)(model.init(jax.random.key(0), ids), ids)
    assert logits.shape == (2, 8, 64) and bool(jnp.all(jnp.isfinite(logits))), logits.shape
    engram = Engram(layer_ids=(1,), num_embeddings=(), max_ngram_size=2, vocab_size=64, n_heads=2,
                    head_dim=4, compressed_vocab_size=32)
    assert engram.primes == (((67, 71),),), engram.primes
    row = np.arange(8, dtype=np.float32).reshape(1, 8)
    expected = row.mean() + row.std() * np.sqrt(2) * scipy.special.erfinv(2 * 0.95 - 1)
    cutoff = np.asarray(gaussian_cutoff(jnp.asarray(row), 0.95))
    np.testing.assert_allclose(cutoff, [[expected]], rtol=1e-5)


def token_stores(scratch: Path) -> tuple[str, str]:
    """Two token stores, a train and a validation one, exported as ArrayRecord from four symbols repeated."""
    from dew.data import Memory
    from dew.data.sinks import export

    stores = {split: str(scratch / f"tokens-{split}") for split in ("train", "val")}
    for split, repeats in (("train", 64), ("val", 16)):
        export(Memory([{"text": np.tile(np.arange(1, 5, dtype=np.int32), repeats)}]), stores[split],
               format="array_record", length_field="text",
               metadata={"token_store": True, "tokenizer": "symbols", "vocab_size": 8, "dtype": "int32"})
    return stores["train"], stores["val"]


def token_data(train: str, validation: str, *, workers: int = 0):
    """A run's data over the stores: windows of eight tokens, four a batch, read from memory through
    `workers` grain worker processes."""
    from dew.data import Loading, presets
    from dew.data.compile import compile
    from dew.data.sources import TokenFiles

    chain = TokenFiles(train).shuffle(0).repeat().pipe(presets.token_windows, seq_len=8, batch=4)
    held = TokenFiles(validation).pipe(presets.token_windows, seq_len=8, batch=4, records=4)
    return compile(chain, validation=held, read="memory",
                   loading=Loading(workers=workers, threads=1, read_buffer=2))


def tiny_trainer(checkpoints=None, tracker=None):
    """A one-layer language model's trainer, seeded, with `checkpoints` where a run keeps them and `tracker`
    where it reports."""
    import jax
    import jax.numpy as jnp
    import optax

    from dew import Trainer
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=16, dtype=jnp.float32, attention_impl="xla")
    return Trainer(LMObjective(model, seq_len=8, ema_decay=None), optax.adam(1e-2), key=jax.random.key(0),
                   checkpoints=checkpoints, tracker=tracker)


def tracked_fit(scratch: Path, tracker) -> None:
    """Two steps of the tiny language model reported through `tracker`."""
    train, validation = token_stores(scratch)
    state = tiny_trainer(tracker=tracker).fit(token_data(train, validation), steps=2, log_every=1)
    assert int(state.step) == 2


def raw(leaf) -> np.ndarray:
    """`leaf` as an array a comparison can read: a typed PRNG key as its key data."""
    import jax
    import jax.numpy as jnp

    return np.asarray(jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf)


def assert_state_equal(actual, expected, label: str) -> None:
    """Every leaf of two train states equal, bit for bit, the same structure."""
    import jax

    assert jax.tree.structure(actual) == jax.tree.structure(expected), label
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(raw(left), raw(right), err_msg=label)


@step
def training(scratch: Path) -> None:
    """A language model trains two steps through `Trainer.fit` with a grain worker process reading its token
    store; its parameters are finite and have moved."""
    import jax
    import jax.numpy as jnp

    train, validation = token_stores(scratch)
    before = jax.tree.leaves(tiny_trainer().place()[0].variables["params"])
    state = tiny_trainer().fit(token_data(train, validation, workers=1), steps=2, log_every=1)
    assert int(state.step) == 2, int(state.step)
    after = jax.tree.leaves(state.variables["params"])
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in after)
    assert any(not np.array_equal(np.asarray(new), np.asarray(old))
               for new, old in zip(after, before, strict=True)), "no parameter moved"


@step
def checkpoint(scratch: Path) -> None:
    """A run checkpointed at step 2 restores what it saved, and resumed to step 4 ends where an uninterrupted
    run of four does, bit for bit: the parameters, the optimizer's moments, the key and the step, the data
    position among them (a resume that read its batches from the start could not match)."""
    from dew.checkpoints import Checkpoints

    train, validation = token_stores(scratch)
    baseline = tiny_trainer().fit(token_data(train, validation), steps=4, log_every=1)
    directory = str(scratch / "run")
    with Checkpoints(directory) as writer:
        prefix = tiny_trainer(writer).fit(token_data(train, validation), steps=2, checkpoint_every=1,
                                          log_every=1)
    with Checkpoints(directory) as reader:
        fresh = tiny_trainer(reader)
        restored, _, position = fresh.place()
        assert int(restored.step) == int(prefix.step) == 2
        assert position, "the checkpoint holds no data position"
        assert_state_equal(restored, prefix, "the restored state is the saved one")
        resumed = fresh.fit(token_data(train, validation), steps=4, checkpoint_every=1, log_every=1)
    assert int(resumed.step) == 4
    assert_state_equal(resumed, baseline, "the resumed run is the uninterrupted one")


def llama_directory(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """A tiny Llama decoder's checkpoint directory at `path`, written as a Hub repository holds one: its
    config.json by transformers' config class, its tensors by safetensors, from a seed. Returns the tensors
    and the vocabulary."""
    import safetensors.numpy
    import transformers

    fields = {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
              "head_dim": 8, "intermediate_size": 64, "vocab_size": 64, "max_position_embeddings": 32,
              "tie_word_embeddings": False}
    transformers.AutoConfig.for_model("llama", **fields).save_pretrained(path)
    rng = np.random.default_rng(0)
    hidden, heads, kv, size, inner, vocab = (fields[key] for key in (
        "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
        "vocab_size"))

    def matrix(rows: int, columns: int) -> np.ndarray:
        return (rng.standard_normal((rows, columns)) / np.sqrt(columns)).astype(np.float32)

    tensors = {"model.embed_tokens.weight": matrix(vocab, hidden), "lm_head.weight": matrix(vocab, hidden),
               "model.norm.weight": np.ones(hidden, np.float32)}
    for layer in range(fields["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        tensors |= {f"{prefix}.self_attn.q_proj.weight": matrix(heads * size, hidden),
                    f"{prefix}.self_attn.k_proj.weight": matrix(kv * size, hidden),
                    f"{prefix}.self_attn.v_proj.weight": matrix(kv * size, hidden),
                    f"{prefix}.self_attn.o_proj.weight": matrix(hidden, heads * size),
                    f"{prefix}.mlp.gate_proj.weight": matrix(inner, hidden),
                    f"{prefix}.mlp.up_proj.weight": matrix(inner, hidden),
                    f"{prefix}.mlp.down_proj.weight": matrix(hidden, inner),
                    f"{prefix}.input_layernorm.weight": np.ones(hidden, np.float32),
                    f"{prefix}.post_attention_layernorm.weight": np.ones(hidden, np.float32)}
    safetensors.numpy.save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    return tensors, vocab


def logits_of(loaded, vocab: int) -> np.ndarray:
    """A loaded decoder's logits over fixed ids, finite and as wide as its vocabulary."""
    ids = (np.arange(16, dtype=np.int32).reshape(2, 8) * 5) % vocab
    logits = np.asarray(loaded.model.apply(loaded.variables, ids))
    assert logits.shape == (2, 8, vocab) and np.isfinite(logits).all(), logits.shape
    return logits


@step
def huggingface(scratch: Path) -> None:
    """A decoder directory loads, gives finite logits, saves, and loads again to the same logits; the tensors
    it saved are the tensors written, and its config reads back through transformers."""
    import jax.numpy as jnp
    import safetensors.numpy
    import transformers

    from dew.interop import Pretrained

    source, saved = scratch / "source", scratch / "saved"
    source.mkdir()
    tensors, vocab = llama_directory(source)
    loaded = Pretrained.load(source, dtype=jnp.float32, param_dtype=jnp.float32)
    before = logits_of(loaded, vocab)
    loaded.save(saved)
    again = Pretrained.load(saved, dtype=jnp.float32, param_dtype=jnp.float32)
    np.testing.assert_array_equal(logits_of(again, vocab), before)
    written = {name: tensor for file in sorted(saved.glob("*.safetensors"))
               for name, tensor in safetensors.numpy.load_file(file).items()}
    assert written.keys() == tensors.keys(), sorted(written.keys() ^ tensors.keys())
    for name, tensor in tensors.items():
        np.testing.assert_array_equal(written[name], tensor, err_msg=name)
    assert transformers.AutoConfig.from_pretrained(saved).model_type == "llama"


class Quiet(http.server.SimpleHTTPRequestHandler):
    """The standard library's file handler without its access log."""

    def log_message(self, format, *args):
        pass


@contextlib.contextmanager
def served(directory: Path) -> Iterator[str]:
    """`directory` served over HTTP on a loopback port by the standard library's server; the address."""
    handler = functools.partial(Quiet, directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join()


@step
def records(scratch: Path) -> None:
    """Records read back as written: a zstandard-compressed JSON lines file and a Parquet file (pyarrow); a
    file and an HTTP url fetched byte for byte (aiohttp); an image decoded and resized to the pixels it was
    drawn."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zstandard
    from PIL import Image

    from dew.data import images
    from dew.data.sources import Files
    from dew.data.sources.urls import Fetch

    rows = [{"id": index, "text": f"row {index}"} for index in range(12)]
    lines = "\n".join(json.dumps(row) for row in rows).encode()
    (scratch / "rows.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(lines))
    pq.write_table(pa.table({"id": [row["id"] for row in rows], "text": [row["text"] for row in rows]}),
                   scratch / "rows.parquet")
    assert list(Files(str(scratch / "rows.jsonl.zst")).records("memory")) == rows
    assert [dict(row) for row in Files(str(scratch / "rows.parquet")).records("memory")] == rows
    Image.fromarray(np.full((16, 16, 3), 9, np.uint8)).save(scratch / "flat.png")
    (scratch / "payload").write_bytes(b"source bytes")
    with served(scratch) as address:
        fetched = Fetch(output="image")({"url": f"{address}/flat.png"})
        assert Fetch()({"url": f"{address}/payload"})["bytes"] == b"source bytes"
    assert Fetch()({"url": (scratch / "payload").as_uri()})["bytes"] == b"source bytes"
    decoded = images.DecodeImage("image").map(fetched)
    np.testing.assert_array_equal(decoded["image"], np.full((16, 16, 3), 9, np.uint8))
    resized = images.Resize(8).map(decoded)["image"]
    assert resized.shape == (8, 8, 3) and (resized == 9).all()


def byte_tokenizer():
    """A byte-level BPE over all 256 bytes as a transformers tokenizer, an id a byte, built here from
    `tokenizers` alone: any text has a spelling."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    alphabet = bytes_to_unicode()
    backend = Tokenizer(models.BPE({alphabet[byte]: byte for byte in range(256)}, [], unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token=alphabet[0])


@flow("profile")
def profile(scratch: Path) -> None:
    """A profile captured around a jitted program exports its native reports (xprof)."""
    import jax
    import jax.numpy as jnp

    import dew

    with dew.Profiler(scratch), jax.profiler.TraceAnnotation("bounds_marker"):
        jax.jit(lambda x: jnp.sin(x @ x))(jnp.ones((32, 32), jnp.float32)).block_until_ready()
    [capture] = sorted(scratch.glob("capture-*"))
    record = json.loads((capture / "manifest.json").read_text())
    assert record["body_status"] == record["capture_status"] == record["export_status"] == "completed", record
    assert list(capture.glob("plugins/profile/*/*.xplane.pb")), "the capture wrote no XSpace"
    assert any(report["status"] == "completed" for report in record["reports"]), record["reports"]


@flow("av")
def av(scratch: Path) -> None:
    """A clip muxed by moviepy's ffmpeg is cut by `PickClip` and decoded: each frame of the window is the flat
    grey of its own index, and the window's sound is the track's samples (moviepy)."""
    import moviepy.config as moviepy_config

    from dew.data import Memory, audio, video

    fps, rate, frames, step_grey = 25, 16000, 50, 5
    pixels = np.stack([np.full((32, 32, 3), step_grey * index, np.uint8) for index in range(frames)])
    pcm = np.repeat(np.arange(frames, dtype=np.int16) * 100, rate // fps)
    (scratch / "clip.rgb").write_bytes(pixels.tobytes())
    (scratch / "clip.pcm").write_bytes(pcm.tobytes())
    run(moviepy_config.FFMPEG_BINARY, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "32x32", "-r", str(fps), "-i", scratch / "clip.rgb", "-f", "s16le", "-ar", str(rate),
        "-ac", "1", "-i", scratch / "clip.pcm", "-c:v", "ffv1", "-c:a", "pcm_s16le", scratch / "clip.avi")
    chain = (Memory([{"mp4": str(scratch / "clip.avi"), "id": 0}])
             .random_filter_map(video.PickClip(8 / fps, margin=2 / fps))
             .filter_map(video.DecodeVideo(8, fps=fps, size=32))
             .filter_map(audio.DecodeAudio(rate, context=2 / fps))
             .select_columns("clip", "video", "audio", "audio_valid"))
    [record] = list(chain.repeat(1))
    start = float(record["clip"][0])
    assert record["video"].shape == (8, 32, 32, 3)
    seen = [round(int(frame[16, 16, 0]) / step_grey) for frame in record["video"]]
    assert seen == [int(fps * (start + index / fps) + 1e-5) for index in range(8)], seen
    assert record["audio"].shape == (12 * (rate // fps),) and record["audio_valid"].all()


@flow("wandb")
def wandb(scratch: Path) -> None:
    """A fit reported to an offline Weights & Biases run leaves the run's summary and its datastore."""
    import os

    from dew.training import WandbTracker

    os.environ["WANDB_DIR"] = str(scratch)
    with WandbTracker("bounds", offline=True) as tracker:
        tracked_fit(scratch, tracker)
        assert tracker.run.summary["FitEnded"]["status"] == "completed"
    assert [path for path in scratch.glob("wandb/offline-run-*/*.wandb") if path.stat().st_size]


@flow("mlflow")
def mlflow(scratch: Path) -> None:
    """A fit reported to a local MLflow file store is read back with MLflow's own client."""
    import os

    from dew.training import MLflowTracker

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    with MLflowTracker("bounds", "tiny-fit", uri=(scratch / "mlruns").as_uri()) as tracker:
        tracked_fit(scratch, tracker)
        client, run_id = tracker.run
    assert [point.step for point in client.get_metric_history(run_id, "train/loss")] == [1, 2]
    assert client.get_run(run_id).info.status == "FINISHED"


@flow("tensorboard")
def tensorboard(scratch: Path) -> None:
    """A fit reported to TensorBoard's event files is read back with its own accumulator."""
    from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

    from dew.training import TensorBoardTracker

    with TensorBoardTracker(scratch / "events") as tracker:
        tracked_fit(scratch, tracker)
    reader = EventAccumulator(str(scratch / "events"), size_guidance={SCALARS: 0})
    reader.Reload()
    points = reader.Scalars("train/loss")
    assert [point.step for point in points] == [1, 2] and all(np.isfinite(p.value) for p in points)


@flow("hpo")
def hpo(scratch: Path) -> None:
    """Optuna's sampler is given the finished trials and draws the next point from the space, the same each
    time it is asked."""
    from dew.config.sweep import OptunaSearch
    from dew.telemetry.records import TrialFinished

    space = {"rate": [0.1, 0.01, 0.001], "depth": [1, 2, 3]}
    finished = [TrialFinished(index, f"trial-{index}", {"rate": space["rate"][index % 3],
                                                        "depth": space["depth"][index % 3]}, float(index))
                for index in range(4)]
    point = OptunaSearch(seed=0)(space, finished)
    assert point.keys() == space.keys() and all(point[key] in space[key] for key in space), point
    assert OptunaSearch(seed=0)(space, finished) == point


@flow("streaming")
def streaming(scratch: Path) -> None:
    """A Hugging Face `datasets` table is read as a Dew store, its rows as they were written."""
    import datasets

    from dew.data.sources import from_hf

    rows = [{"id": index, "text": f"row {index}"} for index in range(8)]
    stored = from_hf(datasets.Dataset.from_list(rows), identity="rows@1")
    assert [dict(row) for row in stored.records("memory")] == rows


@flow("delta")
def delta(scratch: Path) -> None:
    """A Delta table written twice reads as its latest version and, pinned, as its first (deltalake)."""
    import pyarrow as pa
    from deltalake import write_deltalake

    from dew.data.sources.delta import Delta

    url = str(scratch / "table")
    write_deltalake(url, pa.table({"id": [0, 1, 2]}))
    write_deltalake(url, pa.table({"id": [3, 4]}), mode="append")
    assert sorted(row["id"] for row in Delta(url).records("memory")) == [0, 1, 2, 3, 4]
    assert sorted(row["id"] for row in Delta(url, version=0).records("memory")) == [0, 1, 2]


PROSE = ("The river runs past the old mill and through the town, where the children wait on the bridge to "
         "watch the boats go by in the evening. Their parents talk about the harvest and the weather, and "
         "about the market that opens on the square at the end of every week. When the light goes, the lamps "
         "come on along the water and the town settles down for the night.")


@flow("curation")
def curation(scratch: Path) -> None:
    """DataTrove's Gopher quality filter, over spaCy's word splitting, keeps a paragraph and rejects a
    fragment."""
    from dew.data import Rejected, text

    step_ = text.Quality("gopher")
    kept = {"id": 0, "text": PROSE}
    assert step_(kept) == kept
    assert isinstance(step_({"id": 1, "text": "Click here."}), Rejected)


@flow("eval-harness")
def eval_harness(scratch: Path) -> None:
    """lm-evaluation-harness scores a continuation through `DewLM` over a saved run: a finite log-likelihood
    and a greedy flag, one pair a request."""
    import dataclasses as dc

    import jax
    import optax
    from lm_eval.api.instance import Instance

    importlib.import_module("dew.nn.backbones")  # registers the models
    from dew import Trainer
    from dew.checkpoints import Checkpoints
    from dew.data import ByteTokenizer
    from dew.eval.harness import DewLM
    from dew.inference import RunProcessor
    from dew.objectives.lm import LMObjective, Samples
    from dew.registry import ModelConfig
    from dew.sampling import Sampling

    fields = {"vocab_size": 256, "emb_features": 16, "num_layers": 1, "num_heads": 2, "head_dim": 8,
              "mlp_features": 32, "max_seq_len": 16}
    config = ModelConfig("causal_transformer", {**fields, "dtype": "float32", "attention_impl": "reference"})
    sampling = Sampling(temperature=0, eos_token_ids=255)
    objective = LMObjective(config.build(), 8, ema_decay=0.9,
                            samples=Samples([1, 2, 3], 4, sampling=sampling),
                            processor=RunProcessor(ByteTokenizer()))
    train, validation = token_stores(scratch)
    run_directory = scratch / "run"
    with Checkpoints(str(run_directory), keep=1) as checkpoints:
        Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), checkpoints=checkpoints).fit(
            token_data(train, validation), steps=2, log_every=100, checkpoint_every=2)
    (run_directory / "run.json").write_text(json.dumps({
        "objective": "lm", "model": dc.asdict(config), "tokenizer": "byte", "max_new_tokens": 4,
        "sampling": dc.asdict(sampling), "ema_decay": 0.9, "data": {"seq_len": 8}}))
    [(likelihood, greedy)] = DewLM.from_run(str(run_directory), batch_size=2).loglikelihood(
        [Instance(request_type="loglikelihood", doc={}, arguments=("the cat", " sat"), idx=0)])
    assert np.isfinite(likelihood) and likelihood < 0 and isinstance(greedy, bool), (likelihood, greedy)


@flow("tfds")
def tfds(scratch: Path) -> None:
    """A dataset prepared as ArrayRecords by TensorFlow Datasets' builder reads back through `TFDS`,
    row for row."""
    import tensorflow_datasets as tensorflow_datasets

    from dew.data import TFDS

    class BoundsPassages(tensorflow_datasets.core.GeneratorBasedBuilder):
        VERSION = tensorflow_datasets.core.Version("1.0.0")

        def _info(self):
            return tensorflow_datasets.core.DatasetInfo(builder=self, disable_shuffling=True, features=(
                tensorflow_datasets.features.FeaturesDict({"text": tensorflow_datasets.features.Text()})))

        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            for index in range(6):
                yield index, {"text": f"passage {index}"}

    builder = BoundsPassages(data_dir=str(scratch), file_format="array_record")
    builder.download_and_prepare(download_config=tensorflow_datasets.download.DownloadConfig(
        try_download_gcs=False))
    stored = TFDS("bounds_passages", path=str(builder.data_dir), split="train").records("indexed")
    assert [stored[index]["text"] for index in range(len(stored))] == [f"passage {i}" for i in range(6)]


@flow("torch")
def torch_flow(scratch: Path) -> None:
    """A decoder whose weights are a PyTorch pickle (`pytorch_model.bin`) loads to the logits its safetensors
    twin gives."""
    import os
    import shutil

    import jax.numpy as jnp
    import torch

    from dew.interop import Pretrained

    os.environ["XDG_CACHE_HOME"] = str(scratch / "cache")
    safe, pickled = scratch / "safe", scratch / "pickled"
    safe.mkdir()
    pickled.mkdir()
    tensors, vocab = llama_directory(safe)
    shutil.copy(safe / "config.json", pickled / "config.json")
    torch.save({name: torch.from_numpy(tensor) for name, tensor in tensors.items()},
               pickled / "pytorch_model.bin")
    options = {"dtype": jnp.float32, "param_dtype": jnp.float32}
    expected = logits_of(Pretrained.load(safe, **options), vocab)
    np.testing.assert_array_equal(logits_of(Pretrained.load(pickled, **options), vocab), expected)


@flow("diffusers")
def diffusers(scratch: Path) -> None:
    """The diffusers repository Dew's single-file route takes a checkpoint's configs from is the one diffusers
    names for the model type its key map infers (diffusers, torch)."""
    import torch
    from diffusers.loaders.single_file_utils import DIFFUSERS_DEFAULT_PIPELINE_PATHS

    from dew.interop import single_file

    xl = {"conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias": torch.zeros(4)}
    v2_key = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
    v2 = {v2_key: torch.zeros(4, 1024)}
    for checkpoint, kind in ((xl, "xl_base"), (v2, "v2")):
        assert single_file.config_repo(checkpoint) == (
            DIFFUSERS_DEFAULT_PIPELINE_PATHS[kind]["pretrained_model_name_or_path"]), kind


@flow("wan")
def wan(scratch: Path) -> None:
    """Wan's prompt cleaning repairs a mis-decoded prompt (ftfy) and collapses its whitespace before it
    tokenizes."""
    from dew.inputs.diffusion import WanConditioner

    tokenizer = byte_tokenizer()
    conditioner = WanConditioner(tower=None, tokenizer=tokenizer, params={}, checkpoint="", tokens=32)
    encoded = conditioner.tokenize(["A  caf\u00c3\u00a9\n  on the   table"])
    kept = np.asarray(encoded["attention_mask"][0], bool)
    assert tokenizer.decode(np.asarray(encoded["input_ids"][0])[kept]) == "A caf\u00e9 on the table"


@flow("inference-clients")
def inference_clients(scratch: Path) -> None:
    """Ollama's and OpenAI's official clients, over a mocked transport, answer a completion through Dew's
    adapters."""
    import httpx
    import httpx2
    import ollama
    import openai

    from dew.inference import OllamaCompletion, OpenAICompletion
    from dew.sampling import Sampling

    def served_by(module, body):
        return module.MockTransport(lambda request: module.Response(200, json=body))

    local = ollama.Client(host="http://fixture", transport=served_by(httpx, {"response": "ok"}))
    assert OllamaCompletion("tiny", local)("prompt", 5, sampling=Sampling(temperature=0)).texts == ("ok",)
    body = {"id": "fixture", "model": "tiny", "created": 1, "object": "text_completion",
            "choices": [{"index": 0, "text": "ok", "finish_reason": None}]}
    remote = openai.OpenAI(base_url="http://fixture/v1", api_key="fixture", max_retries=0,
                           http_client=httpx2.Client(transport=served_by(httpx2, body)))
    assert OpenAICompletion("tiny", remote)("prompt", 5, sampling=Sampling(temperature=0)).texts == ("ok",)


@flow("environments")
def environments(scratch: Path) -> None:
    """An episode in an OpenEnv environment served by OpenEnv's own server and reached by its client: Dew's
    adapter resets it, steps it on a drawn guess, and ends the episode on the right one."""
    import socket
    import time

    import uvicorn
    from openenv.core import GenericEnvClient
    from openenv.core.env_server.http_server import create_fastapi_app
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State

    from dew.inference.rollouts import Draw
    from dew.interop.openenv import OpenEnv, Said
    from dew.objectives.rl import Task
    from dew.objectives.rl.episodes import EpisodeId, EpisodeStatus

    class Guess(Action):
        number: int

    class Feedback(Observation):
        hint: str = ""

    class Guessing(Environment):
        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self):
            super().__init__()
            self.target = 0

        def reset(self, seed=None, episode_id=None, **options):
            self.target = options["target"]
            return Feedback(hint="guess a number from 0 to 9")

        def step(self, action, timeout_s=None, **kwargs):
            if action.number == self.target:
                return Feedback(hint="right", done=True, reward=1.0)
            return Feedback(hint="higher" if action.number < self.target else "lower", reward=-0.1)

        @property
        def state(self):
            return State(step_count=0)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = create_fastapi_app(Guessing, Guess, Feedback, max_concurrent_envs=2)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    running = threading.Thread(target=server.run, daemon=True)
    running.start()
    while not server.started and running.is_alive():
        time.sleep(0.05)

    @contextlib.contextmanager
    def session(task, identity):
        with GenericEnvClient(base_url=f"http://127.0.0.1:{port}").sync() as opened:
            yield opened

    environment = OpenEnv(session, prompt=lambda observation: Said((1,), ()),
                          reply=lambda observation: Said((2,), ()), action=lambda ids: {"number": ids[0] - 3},
                          options=lambda task, identity: {"target": int(task.id)})
    try:
        with environment(Task("7"), EpisodeId(task=0, attempt=0, sample=0, seed=(0,))) as episode:
            assert episode.reset().context == (1,)
            wrong = episode.step(Draw((1,), (6,), (0.0,), "stop", 0))
            assert (wrong.status, wrong.reward, wrong.context) == (EpisodeStatus.RUNNING, -0.1, (1, 6, 2))
            right = episode.step(Draw((1, 6, 2), (10,), (0.0,), "stop", 0))
            assert (right.status, right.reward) == (EpisodeStatus.COMPLETED, 1.0)
    finally:
        server.should_exit = True
        running.join()


@flow("guided")
def guided(scratch: Path) -> None:
    """A regex compiled over a byte-level tokenizer's vocabulary (outlines-core) lets exactly its strings
    through the token automaton, byte by byte, and allows the end token where the pattern is complete."""
    from dew.sampling import guided as grammars

    end = 256
    grammar = grammars.regex(byte_tokenizer(), "[0-9]{3}-[a-z]{2}", end, vocab_size=end + 1)
    transitions, classes = np.asarray(grammar.transitions), np.asarray(grammar.classes)

    def following(state: int, token: int) -> int:
        return int(transitions[state, classes[token]])

    assert following(0, ord("a")) < 0 and following(0, end) < 0
    state = 0
    for byte in b"123-ab":
        state = following(state, byte)
        assert state >= 0, chr(byte)
    assert following(state, end) >= 0 and following(state, ord("c")) < 0


def flat_image() -> np.ndarray:
    """A two-column image whose left column is dark and right column light, so a flip is seen."""
    image = np.zeros((4, 2, 3), np.uint8)
    image[:, 1] = 200
    return image


@flow("augment")
def augment(scratch: Path) -> None:
    """An albumentations flip with probability one, seeded from the record's generator, flips the image."""
    import albumentations

    from dew.data import images

    stage = images.Albumentations(albumentations.Compose([albumentations.HorizontalFlip(p=1.0)]))
    flipped = stage.random_map({"image": flat_image()}, np.random.default_rng(0))["image"]
    np.testing.assert_array_equal(flipped, flat_image()[:, ::-1])


@flow("vision")
def vision(scratch: Path) -> None:
    """A torchvision v2 flip with probability one, seeded from the record's generator, flips the image."""
    from torchvision.transforms import v2

    from dew.data import images

    stage = images.Torchvision(v2.RandomHorizontalFlip(p=1.0))
    flipped = stage.random_map({"image": flat_image()}, np.random.default_rng(0))["image"]
    np.testing.assert_array_equal(flipped, flat_image()[:, ::-1])


@flow("torchax")
def torchax(scratch: Path) -> None:
    """A model no Dew family computes, a random GPT-NeoX transformers saved, loads through the torchax
    fallback to the logits transformers' own forward gives (torch, torchax)."""
    import jax.numpy as jnp
    import torch
    import transformers

    from dew.interop import Pretrained

    torch.manual_seed(0)
    config = transformers.GPTNeoXConfig(num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
                                        intermediate_size=256, vocab_size=256, max_position_embeddings=64)
    model = transformers.GPTNeoXForCausalLM(config).eval()
    model.save_pretrained(scratch / "neox")
    ids = np.random.RandomState(0).randint(0, 256, (4, 16)).astype(np.int32)
    with torch.no_grad():
        expected = model(torch.from_numpy(ids.astype(np.int64))).logits.float().numpy()
    loaded = Pretrained.load(scratch / "neox", fallback="torchax", dtype="float32")
    got = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(got, expected, atol=2e-6)


@dataclasses.dataclass(frozen=True)
class Outcome:
    """One step's result: its name, whether it passed, and where it failed."""

    name: str
    passed: bool
    error: str = ""
    distributions: tuple[str, ...] = ()
    """The distributions, as `name==version`, that the failing traceback's frames were in."""


def distributions_of(error: BaseException) -> tuple[str, ...]:
    """The installed distributions that own the modules of `error`'s traceback frames, each `name==version`:
    where to look first, not a cause."""
    owners = importlib.metadata.packages_distributions()
    found: set[str] = set()
    for frame, _ in traceback.walk_tb(error.__traceback__):
        for name in owners.get(str(frame.f_globals.get("__name__", "")).split(".")[0], ()):
            found.add(f"{name}=={importlib.metadata.version(name)}")
    return tuple(sorted(found))


def attempt(name: str, function: Callable[[Path], None], scratch: Path) -> Outcome:
    """`function` run in its own directory under `scratch`; the step's outcome, a failure's traceback kept."""
    directory = scratch / name.replace(" ", "-")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        function(directory)
    except Exception as error:  # the failure is the result: reported by name, and the run exits 1
        error_text = "".join(traceback.format_exception(error))[-6000:]
        return Outcome(name, passed=False, error=error_text, distributions=distributions_of(error))
    return Outcome(name, passed=True)


def main(arguments: list[str] | None = None) -> int:
    global SELECTED
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scratch", type=Path, required=True, help="a directory to write in")
    parser.add_argument("--extra", action="append", default=[], choices=sorted(FLOWS),
                        help="an extra the environment installed, whose flow runs after the base steps")
    parser.add_argument("--report", type=Path, help="write each step's outcome here as JSON")
    args = parser.parse_args(arguments)
    SELECTED = tuple(args.extra)
    args.scratch.mkdir(parents=True, exist_ok=True)
    import dew

    assert "site-packages" in dew.__file__, f"dew is imported from {dew.__file__}, not an installed wheel"
    outcomes = [attempt(name, function, args.scratch) for name, function in STEPS.items()]
    outcomes += [attempt(f"extra {extra}", FLOWS[extra], args.scratch) for extra in args.extra]
    for outcome in outcomes:
        print(f"{'ok' if outcome.passed else 'FAILED'} {outcome.name}")
        if not outcome.passed:
            print(outcome.error)
    if args.report:
        args.report.write_text(json.dumps([dataclasses.asdict(outcome) for outcome in outcomes], indent=1))
    return 0 if all(outcome.passed for outcome in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
