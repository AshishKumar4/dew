"""What a run publishes, and how many times.

Publishing is the only step a recipe takes after `fit` that talks to a
service, so the two facts that matter on a pod are checked here without one:
it happens once per run, not once per process, and a checkpoint directory
that is a bucket URI is referenced, not uploaded, with the run spec beside it.
"""

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
    module.created = []

    def artifact(*, name, type):
        created = Artifact(name, type)
        module.created.append(created)
        return created

    module.Artifact = artifact
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


class TrackedRun(Run):
    """The run `wandb.init` opens for a config's run: what it logs, ignored, and its artifacts."""

    def __init__(self):
        super().__init__()
        self.summary, self.history = {}, []
        self.config = types.SimpleNamespace(update=lambda value, **_: self.configuration.update(value))
        self.configuration = {}
        self.exit_codes = []
        self.preempting = False

    def define_metric(self, *_, **__):
        pass

    def log(self, value, **_):
        self.history.append(value)

    def mark_preempting(self):
        self.preempting = True

    def finish(self, exit_code=0, **_):
        self.exit_codes.append(exit_code)


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

    def is_file(self):
        return self.local.is_file()

    def is_dir(self):
        return self.local.is_dir()

    def iterdir(self):
        return [FileUri(f"file://{entry}") for entry in self.local.iterdir()]

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
    import jax.numpy as jnp
    import numpy as np
    import optax
    import wandb as library

    from dew.data import Loading, Memory
    from dew.data.compile import compile
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
        compile(Memory(rows).shuffle(0).repeat().batch(8), read="memory",
                loading=Loading(workers=0, threads=1, read_buffer=1)),
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


class RankedRun(Run):
    """A W&B run as `promote` reads it: where it lives, its summary and, in a sweep, the sweep's id."""

    def __init__(self, summary, *, sweep_id=None, id="current"):
        super().__init__()
        self.summary, self.sweep_id, self.id = summary, sweep_id, id
        self.entity, self.project = "team", "flowers"


class Other:
    """Another run of the project or sweep, as `wandb.Api` lists it."""

    def __init__(self, id, summary):
        self.id, self.summary = id, summary


class Api:
    """`wandb.Api`, holding the project's runs, a sweep's and the artifacts published, by their paths:
    what `ranked` lists and what `downloaded` fetches, each download wandb's own (`downloaded`)."""

    def __init__(self, project, sweeps, published, target):
        self.project, self.sweeps, self.published, self.target = project, sweeps, published, target
        self.asked = []

    def runs(self, path, filters=None):
        self.asked.append(("runs", path, filters))
        return self.project

    def sweep(self, path):
        self.asked.append(("sweep", path))
        return types.SimpleNamespace(runs=self.sweeps[path])

    def artifact(self, reference, type=None):
        name, alias = reference.rsplit(":", 1)
        return self.fetched(self.published[(name, alias)])

    def run(self, path):
        logged = [self.fetched(artifact) for artifact in self.published[path]]
        return types.SimpleNamespace(logged_artifacts=lambda: logged)

    def fetched(self, logged):
        def download(root=None):
            return downloaded(logged, self.target)

        return types.SimpleNamespace(type=logged.type, download=download)


def test_a_sweeps_best_run_is_promoted_to_the_registry_and_loads_in_one_call_from_it(tmp_path, monkeypatch):
    """FlaxDiff's W&B workflow on a tiny LM run. `promote` ranks the run by its summary's metric among
    the best three of its project's runs (those whose summary holds the metric) or of its sweep's: ranked
    first it publishes, aliased "best" in the registry; second, it publishes with no alias; past the
    third, or tied with the third, it publishes nothing. A configured run's `trainer.promote` names the
    rule and requires its W&B sink. `dew.pipeline("wandb:...")` then loads the promoted run in one call,
    from the registry by its alias and from the W&B run by its path, as the task the run directory itself
    loads: its continuation is the run's, token for token."""
    import numpy as np
    import optax
    import wandb as library

    import dew
    from dew.config import TrainerConfig
    from dew.data import Loading, Memory
    from dew.data.compile import compile
    from dew.io import Promotion
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.sampling import Sampling
    from dew.training import Trainer, Wandb

    monkeypatch.setenv("WANDB_MODE", "offline")
    model = CausalTransformer(vocab_size=16, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=16, dtype="float32", attention_impl="xla")
    rows = [{"text": (np.arange(9, dtype=np.int32) + index) % 16} for index in range(8)]
    run = tmp_path / "run"
    checkpoints = Checkpoints(str(run))
    Trainer(LMObjective(model, seq_len=8), optax.sgd(1.0), key=0, checkpoints=checkpoints).fit(
        compile(Memory(rows).shuffle(0).repeat().batch(8), read="memory",
                loading=Loading(workers=0, threads=1, read_buffer=1)), steps=2, log_every=100)
    checkpoints.wait()
    step = checkpoints.path(2)
    project = [Other("a", {"val/loss": 1.0}), Other("b", {"val/loss": 2.0}), Other("c", {"val/loss": 3.0}),
               Other("d", {}), Other("current", {"val/loss": 0.1})]
    published = {}
    api = Api(project, {"team/flowers/sweep": project[1:3]}, published, tmp_path / "download")
    monkeypatch.setattr(library, "Api", lambda: api)

    def promoted(value, *, sweep_id=None, top=3):
        tracker = Tracker()
        tracker.run = RankedRun({"val/loss": value}, sweep_id=sweep_id)
        logged = dew.io.promote(str(step), "tiny", tracker=tracker, metric="val/loss", top=top)
        return logged, tracker.run.linked

    first, linked = promoted(0.5)
    assert isinstance(first, library.Artifact) and linked == [(first, "wandb-registry-model/tiny", ("best",))]
    second, linked = promoted(1.5)
    assert second is not None and linked == [(second, "wandb-registry-model/tiny", ())]
    assert promoted(2.5, top=2) == (None, []) and promoted(3.0) == (None, [])
    assert api.asked[-1] == ("runs", "team/flowers", None)
    # In a sweep the run is ranked among the sweep's runs alone, ahead of both.
    swept, linked = promoted(1.5, sweep_id="sweep", top=1)
    assert swept is not None and linked[0][2] == ("best",)
    assert api.asked[-1] == ("sweep", "team/flowers/sweep")
    with pytest.raises(ValueError):
        TrainerConfig(steps=1, promote=Promotion("val/loss"))
    configured = TrainerConfig(steps=1, wandb=Wandb(project="flowers"), promote=Promotion("val/loss", top=2))
    assert configured.promote == Promotion("val/loss", top=2)

    published.update({("wandb-registry-model/tiny", "best"): first, "team/flowers/current": [first]})
    prompt = np.arange(1, 5, dtype=np.int32)[None, :]
    greedy = Sampling(temperature=0)
    own = dew.pipeline(str(run))(prompt, 4, key=0, sampling=greedy).tokens
    for source in ("wandb:wandb-registry-model/tiny:best", "wandb:team/flowers/current"):
        shutil.rmtree(tmp_path / "download", ignore_errors=True)
        loaded = dew.pipeline(source)
        np.testing.assert_array_equal(np.asarray(loaded(prompt, 4, key=0, sampling=greedy).tokens),
                                      np.asarray(own))
