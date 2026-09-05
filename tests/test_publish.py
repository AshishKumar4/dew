"""What a run publishes, and how many times.

Publishing is the one thing a recipe does after `fit` that talks to a service,
so the two facts that matter on a pod are checked here without one: it happens
once per run rather than once per process, and a checkpoint directory that is
a bucket URI is referenced rather than uploaded, with the run spec beside it.
"""

import sys
import types

import jax
import pytest

import dew.io
from dew.checkpoints import RUN_FILE


class Artifact:
    """wandb's artifact, reduced to what publish does to one."""

    def __init__(self, name, type):
        self.name, self.type = name, type
        self.dirs, self.files, self.references = [], [], []

    def add_dir(self, directory):
        self.dirs.append(directory)

    def add_file(self, path):
        self.files.append(path)

    def add_reference(self, uri):
        self.references.append(uri)


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
    assert logged.dirs == [str(step)] and logged.references == []
    assert logged.files == [str(run / RUN_FILE)], "the run spec rides with the weights"
    assert tracker.run.logged[0][1] == ("latest", "v1")
    assert tracker.run.linked[0][1] == f"{dew.io.REGISTRY}/flowers"


def test_a_bucket_checkpoint_is_referenced_rather_than_uploaded(monkeypatch, tmp_path, wandb):
    """A pod writes its checkpoints to the bucket, and os.path.exists on a
    gs:// path is False, so the spec used to be dropped from every pod
    publish and the upload had nothing local to read."""
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

        def exists(self):
            return True

        def __str__(self):
            return self.uri

    monkeypatch.setattr(dew.io.epath, "Path", Uri)
    tracker = Tracker()

    logged = dew.io.publish("gs://dew-runs/flowers/step_6", "flowers", tracker=tracker)

    assert logged is not None
    assert logged.dirs == [] and logged.files == []
    assert logged.references == ["gs://dew-runs/flowers/step_6",
                                 f"gs://dew-runs/flowers/{RUN_FILE}"]


def test_only_process_zero_publishes(monkeypatch, tmp_path, wandb):
    """Every process holds the same checkpoint, so a publish per process is a
    duplicate upload of one artifact, not a part of the run."""
    step = tmp_path / "flowers" / "step_6"
    step.mkdir(parents=True)
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    tracker = Tracker()

    assert dew.io.publish(str(step), "flowers", tracker=tracker) is None
    assert tracker.run.logged == [] and tracker.run.linked == []
