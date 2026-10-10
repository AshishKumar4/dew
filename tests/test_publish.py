"""What a run publishes, and how many times.

Publishing is the only step a recipe takes after `fit` that talks to a
service, so the two facts that matter on a pod are checked here without one:
it happens once per run, not once per process, and a checkpoint directory
that is a bucket URI is referenced, not uploaded, with the run spec beside it.
"""

import json
import shutil
import sys
import types
from pathlib import Path

import jax
import pytest

import dew.io
from dew.checkpoints import FROZEN_STORE, RUN_FILE, Checkpoints


class Artifact:
    """wandb's artifact, reduced to what publish does to one."""

    def __init__(self, name, type):
        self.name, self.type = name, type
        self.dirs, self.files, self.references = [], [], []

    def add_dir(self, directory, name=None):
        self.dirs.append((directory, name))

    def add_file(self, path, name=None):
        self.files.append((path, name))

    def add_reference(self, uri, name=None):
        self.references.append((uri, name))


class Run:
    """The tracker's run: what was logged and what was linked."""

    def __init__(self):
        self.logged, self.linked = [], []

    def log_artifact(self, artifact, aliases):
        self.logged.append((artifact, tuple(aliases)))
        return artifact

    def link_artifact(self, artifact, target_path, aliases):
        self.linked.append((artifact, target_path, tuple(aliases)))


class Tracker:
    def __init__(self):
        self.run = Run()


@pytest.fixture
def wandb(monkeypatch):
    """`import wandb` inside publish, without wandb."""
    module = types.ModuleType("wandb")
    module.Artifact = Artifact
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def test_a_local_checkpoint_is_uploaded_with_its_run_spec(tmp_path, wandb):
    run = tmp_path / "flowers"
    step = run / "step_6"
    step.mkdir(parents=True)
    (step / "params").write_text("weights")
    (run / RUN_FILE).write_text("{}")
    tracker = Tracker()

    logged = dew.io.publish(str(step), "flowers", tracker=tracker, aliases=("v1",))

    assert logged is not None
    assert logged.dirs == [(str(step), "step_6")] and logged.references == []
    assert logged.files == [(str(run / RUN_FILE), RUN_FILE)], "the run spec rides with the weights"
    assert tracker.run.logged[0][1] == ("latest", "v1")
    assert tracker.run.linked[0][1] == f"{dew.io.REGISTRY}/flowers"


def test_a_bucket_checkpoint_is_referenced_rather_than_uploaded(monkeypatch, tmp_path, wandb):
    """A pod writes its checkpoints to the bucket, and os.path.exists on a
    gs:// path is False, so the spec is found through the path's own parent
    and the reference carries it with no local read."""
    class Uri:
        """The three things publish asks a path: its parent, a child, and
        whether it is there. A real gs:// path would need credentials."""

        def __init__(self, uri):
            self.uri = str(uri)

        @property
        def parent(self):
            return Uri(self.uri.rsplit("/", 1)[0])

        def __truediv__(self, name):
            return Uri(f"{self.uri}/{name}")

        @property
        def name(self):
            return self.uri.rsplit("/", 1)[-1]

        def exists(self):
            return True

        def read_text(self):
            """The step's Orbax metadata, recording one stored collection."""
            return json.dumps({"custom_metadata": {"frozen": {"variables": {"frozen": "f00d"}}}})

        def __str__(self):
            return self.uri

    monkeypatch.setattr(dew.io.epath, "Path", Uri)
    tracker = Tracker()

    logged = dew.io.publish("gs://dew-runs/flowers/step_6", "flowers", tracker=tracker)

    assert logged is not None
    assert logged.dirs == [] and logged.files == []
    assert logged.references == [("gs://dew-runs/flowers/step_6", "step_6"),
                                 (f"gs://dew-runs/flowers/{RUN_FILE}", RUN_FILE),
                                 (f"gs://dew-runs/flowers/{FROZEN_STORE}/f00d", f"{FROZEN_STORE}/f00d")]


def test_only_process_zero_publishes(monkeypatch, tmp_path, wandb):
    """Every process holds the same checkpoint, so a publish per process is a
    duplicate upload of one artifact, not a part of the run."""
    step = tmp_path / "flowers" / "step_6"
    step.mkdir(parents=True)
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    tracker = Tracker()

    assert dew.io.publish(str(step), "flowers", tracker=tracker) is None
    assert tracker.run.logged == [] and tracker.run.linked == []


class FileUri:
    """A file:// URI over a local path, standing in for a bucket path, which
    `epath` reads but this test cannot reach; wandb references it as it is."""

    def __init__(self, uri):
        self.uri = str(uri)
        self.local = Path(self.uri.removeprefix("file://"))

    @property
    def name(self):
        return self.local.name

    @property
    def parent(self):
        return FileUri(f"file://{self.local.parent}")

    def __truediv__(self, name):
        return FileUri(f"file://{self.local / name}")

    def exists(self):
        return self.local.exists()

    def read_text(self):
        return self.local.read_text()

    def __str__(self):
        return self.uri


def downloaded(artifact, target: Path) -> Path:
    """What wandb's download writes for `artifact`: each manifest entry at
    its path in the artifact, a local file's bytes or a file:// reference's."""
    for name, entry in artifact.manifest.entries.items():
        source = entry.local_path or str(entry.ref).removeprefix("file://")
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target / name)
    return target


@pytest.mark.parametrize("reference", [False, True], ids=["upload", "uri"])
def test_a_published_lora_run_downloads_to_a_run_that_loads_and_restores(tmp_path, monkeypatch, reference):
    """A LoRA run's step, uploaded or referenced by URI (a file:// one, as a
    bucket's) through wandb's own artifact, downloads with its run spec and the stored base it
    records into a directory that `Adapter.from_run` and `Checkpoints.restore`
    read with the run itself gone: the adapter computes what the run trained,
    and the restored state is the run's, bitwise."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import wandb as library

    from dew.data import Dataset, Loading
    from dew.lora import Adapter, LoRA
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    monkeypatch.setenv("WANDB_MODE", "offline")
    base = CausalTransformer(vocab_size=16, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                             max_seq_len=16, dtype="float32", attention_impl="xla")
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(
        base, base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)), key=1)
    objective = LMObjective(adapter.model, seq_len=8, ema_decay=None, variables=adapter.variables)
    rows = [{"text": (np.arange(9, dtype=np.int32) + index) % 16} for index in range(8)]
    run = tmp_path / "run"
    checkpoints = Checkpoints(str(run))
    state = Trainer(objective, optax.sgd(1.0), key=0, checkpoints=checkpoints).fit(
        Dataset.from_records(rows, batch=8, loading=Loading(workers=0, threads=1, read_buffer=1)),
        steps=2, checkpoint_every=1, log_every=100)
    checkpoints.wait()
    original, _ = checkpoints.restore()
    step = checkpoints.path(2)
    with monkeypatch.context() as patched:
        if reference:
            patched.setattr(dew.io.epath, "Path", FileUri)
        logged = dew.io.publish(f"file://{step}" if reference else step, "lora", tracker=Tracker())
    assert isinstance(logged, library.Artifact)
    assert {name.split("/")[0] for name in logged.manifest.entries} == {"2", FROZEN_STORE, RUN_FILE}

    local = downloaded(logged, tmp_path / "download")
    shutil.rmtree(run)
    rebuilt = Adapter.from_run(local)
    tokens = jnp.arange(1, 9)[None, :]
    np.testing.assert_array_equal(np.asarray(rebuilt.model.apply(rebuilt.variables, tokens)),
                                  np.asarray(objective.model.apply(state.variables, tokens)))
    restored, _ = Checkpoints(str(local)).restore()
    for left, right in zip(jax.tree.leaves(original), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
