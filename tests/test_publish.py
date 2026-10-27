"""What a run publishes, and how many times.

Publishing is the only step a recipe takes after `fit` that talks to a
service, so the two facts that matter on a pod are checked here without one:
it happens once per run, not once per process, and a checkpoint directory
that is a bucket URI is referenced, not uploaded, with the run spec beside it.
"""

import dataclasses
import json
import shutil
import sys
import time
import types
from pathlib import Path

import jax
import pytest

import dew.io
from dew.checkpoints import FROZEN_STORE, RUN_FILE, Checkpoints
from dew.io import Lease


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


class Stored:
    """A version as the registry holds it: its metadata and its aliases."""

    def __init__(self, metadata, aliases):
        self.metadata, self.aliases = dict(metadata), list(aliases)


class Version:
    """A version of a registry collection as `wandb.Api().artifacts` lists it: a snapshot, detached from
    the registry, of its metadata and aliases when it was read, whose `save` writes the snapshot's
    aliases back, `on_save` run first (another client's writes in between)."""

    def __init__(self, registry, path, index, stored: Stored):
        self.registry, self.path, self.index = registry, path, index
        self.metadata, self.aliases = dict(stored.metadata), list(stored.aliases)
        self.version = f"v{index}"

    def save(self):
        self.registry.on_save()
        self.registry.write(self.path, self.index, self.aliases)


class Registry:
    """W&B's model registry, its collections by path, each of the versions linked to it in order. An
    alias is on one version of a collection: writing it to one takes it from the rest."""

    def __init__(self):
        self.collections: dict[str, list[Stored]] = {}
        self.on_save = lambda: None

    def link(self, artifact, path, aliases):
        versions = self.collections.setdefault(path, [])
        versions.append(Stored(artifact.metadata, ()))
        self.write(path, len(versions) - 1, aliases)

    def write(self, path, index, aliases):
        for at, stored in enumerate(self.collections[path]):
            stored.aliases = (list(aliases) if at == index
                              else [alias for alias in stored.aliases if alias not in aliases])

    def read(self, path):
        """Each version of `path` as a fresh read gives it: snapshots."""
        held = self.collections.get(path, [])
        return [Version(self, path, index, stored) for index, stored in enumerate(held)]

    def best(self, path):
        """The metadata of the one version of `path` that holds the best alias."""
        (holder,) = [stored for stored in self.collections[path] if "best" in stored.aliases]
        return holder.metadata


class RankedRun(Run):
    """A W&B run as `promote` reads it: where it lives, its summary and, in a sweep, the sweep's id;
    what it links reaches `registry`."""

    def __init__(self, summary, *, sweep_id=None, id="current", registry=None):
        super().__init__()
        self.summary, self.sweep_id, self.id = summary, sweep_id, id
        self.entity, self.project = "team", "flowers"
        self.registry = registry

    def link_artifact(self, artifact, target_path, aliases):
        super().link_artifact(artifact, target_path, aliases)
        if self.registry is not None:
            self.registry.link(artifact, target_path, aliases)


class Other:
    """Another run of the project or sweep, as `wandb.Api` lists it."""

    def __init__(self, id, summary):
        self.id, self.summary = id, summary


class Api:
    """`wandb.Api`, holding the project's runs, a sweep's and the artifacts published, by their paths:
    what `ranked` lists and what `downloaded` fetches, each download wandb's own (`downloaded`)."""

    def __init__(self, project, sweeps, published, target, registry=None):
        self.project, self.sweeps, self.published, self.target = project, sweeps, published, target
        self.registry = Registry() if registry is None else registry
        self.asked = []

    def artifacts(self, type, path):
        return self.registry.read(path)

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


def tiny_run(tmp_path):
    """A tiny LM run's directory, trained two steps, and the step it ended on."""
    import numpy as np
    import optax

    from dew.data import Loading, Memory
    from dew.data.compile import compile
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    model = CausalTransformer(vocab_size=16, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                              max_seq_len=16, dtype="float32", attention_impl="xla")
    rows = [{"text": (np.arange(9, dtype=np.int32) + index) % 16} for index in range(8)]
    run = tmp_path / "run"
    checkpoints = Checkpoints(str(run))
    Trainer(LMObjective(model, seq_len=8), optax.sgd(1.0), key=0, checkpoints=checkpoints).fit(
        compile(Memory(rows).shuffle(0).repeat().batch(8), read="memory",
                loading=Loading(workers=0, threads=1, read_buffer=1)), steps=2, log_every=100)
    checkpoints.wait()
    return run, checkpoints.path(2)


def test_a_sweeps_best_run_is_promoted_to_the_registry_and_loads_in_one_call_from_it(tmp_path, monkeypatch):
    """FlaxDiff's W&B workflow on a tiny LM run. `promote` ranks the run by its summary's metric among
    the best three of its project's runs (those whose summary holds a finite score) or of its sweep's:
    within the three it publishes into the registry collection, its score its metadata, and the
    collection's best alias rests on its best version; past the third, or tied with the third, it
    publishes nothing. `dew.pipeline("wandb:...")` then loads the promoted run in one call, from the
    registry by its alias and from the W&B run by its path, as the task the run directory itself loads:
    its continuation is the run's, token for token."""
    import numpy as np
    import wandb as library

    import dew
    from dew.sampling import Sampling

    monkeypatch.setenv("WANDB_MODE", "offline")
    run, step = tiny_run(tmp_path)
    project = [Other("a", {"val/loss": 1.0}), Other("b", {"val/loss": 2.0}), Other("c", {"val/loss": 3.0}),
               Other("d", {}), Other("e", {"val/loss": float("nan")}), Other("current", {"val/loss": 0.1})]
    published = {}
    api = Api(project, {"team/flowers/sweep": project[1:3]}, published, tmp_path / "download")
    monkeypatch.setattr(library, "Api", lambda: api)

    def promoted(value, *, sweep_id=None, top=3):
        tracker = Tracker()
        tracker.run = RankedRun({"val/loss": value}, sweep_id=sweep_id, registry=api.registry)
        logged = dew.io.promote(str(step), "tiny", tracker=tracker, metric="val/loss", top=top,
                                lease=Lease(str(tmp_path / "tiny.lease")))
        return logged, tracker.run.linked

    first, linked = promoted(0.5)
    assert isinstance(first, library.Artifact) and linked == [(first, "wandb-registry-model/tiny", ())]
    assert api.registry.best("wandb-registry-model/tiny") == {"val/loss": 0.5}
    second, linked = promoted(1.5)
    assert second is not None and linked == [(second, "wandb-registry-model/tiny", ())]
    assert api.registry.best("wandb-registry-model/tiny") == {"val/loss": 0.5}
    assert promoted(2.5, top=2) == (None, []) and promoted(3.0) == (None, [])
    assert api.asked[-1] == ("runs", "team/flowers", None)
    # In a sweep the run is ranked among the sweep's runs alone, ahead of both.
    swept, _ = promoted(0.25, sweep_id="sweep", top=1)
    assert swept is not None and api.asked[-1] == ("sweep", "team/flowers/sweep")
    assert api.registry.best("wandb-registry-model/tiny") == {"val/loss": 0.25}

    published.update({("wandb-registry-model/tiny", "best"): swept, "team/flowers/current": [swept]})
    prompt = np.arange(1, 5, dtype=np.int32)[None, :]
    greedy = Sampling(temperature=0)
    own = dew.pipeline(str(run))(prompt, 4, key=0, sampling=greedy).tokens
    for source in ("wandb:wandb-registry-model/tiny:best", "wandb:team/flowers/current"):
        shutil.rmtree(tmp_path / "download", ignore_errors=True)
        loaded = dew.pipeline(source)
        np.testing.assert_array_equal(np.asarray(loaded(prompt, 4, key=0, sampling=greedy).tokens),
                                      np.asarray(own))


@pytest.mark.parametrize("mode", ["min", "max"])
def test_a_run_of_no_finite_score_is_never_promoted(tmp_path, monkeypatch, mode):
    """A diverged run, its summary's score NaN or infinite, is refused before it publishes, in either
    ranking mode, and a competitor of no finite score ranks nowhere: the registry's best stays the
    finite run's."""
    import wandb as library

    monkeypatch.setenv("WANDB_MODE", "offline")
    _, step = tiny_run(tmp_path)
    api = Api([Other("diverged", {"val/loss": float("nan")})], {}, {}, tmp_path / "download")
    monkeypatch.setattr(library, "Api", lambda: api)

    def promoted(value):
        tracker = Tracker()
        tracker.run = RankedRun({"val/loss": value}, registry=api.registry)
        return dew.io.promote(str(step), "tiny", tracker=tracker, metric="val/loss", mode=mode, top=1,
                              lease=Lease(str(tmp_path / "tiny.lease")))

    assert promoted(2.0) is not None
    for broken in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            promoted(broken)
    assert api.registry.best("wandb-registry-model/tiny") == {"val/loss": 2.0}
    assert len(api.registry.collections["wandb-registry-model/tiny"]) == 1


def test_a_promoter_between_anothers_read_and_write_cannot_publish_or_crown(tmp_path, monkeypatch):
    """A promoter that comes while another holds the collection's lease, between the holder's fresh read
    of the collection and its write of the alias, publishes and crowns nothing: it waits on the lease and
    gives up once its wait passes. Once the holder has written, it promotes, and the alias it writes is
    the best of every version, read afresh. Snapshots of the collection are detached: a write from one
    is its own aliases, whatever was written since it was read."""
    import wandb as library

    monkeypatch.setenv("WANDB_MODE", "offline")
    _, step = tiny_run(tmp_path)
    api = Api([], {}, {}, tmp_path / "download")
    monkeypatch.setattr(library, "Api", lambda: api)
    url = str(tmp_path / "shared.lease")

    def promoted(value, run_id, wait):
        tracker = Tracker()
        tracker.run = RankedRun({"val/loss": value}, id=run_id, registry=api.registry)
        return dew.io.promote(str(step), "shared", tracker=tracker, metric="val/loss", top=3,
                              lease=Lease(url, hold=60.0, wait=wait))

    def meanwhile():
        api.registry.on_save = lambda: None
        with pytest.raises(TimeoutError):
            promoted(1.0, "late", wait=0.5)
        assert len(api.registry.collections["wandb-registry-model/shared"]) == 1

    api.registry.on_save = meanwhile
    assert promoted(2.0, "first", wait=0.5) is not None
    assert api.registry.best("wandb-registry-model/shared") == {"val/loss": 2.0}
    assert promoted(1.0, "late", wait=0.5) is not None
    assert len(api.registry.collections["wandb-registry-model/shared"]) == 2
    assert api.registry.best("wandb-registry-model/shared") == {"val/loss": 1.0}


def test_a_promoter_that_dies_holding_the_lease_is_followed_by_one_that_crowns_afresh(tmp_path,
                                                                                     monkeypatch):
    """A collection left as an interrupted promoter left it, its alias written on its worse version and
    its lease never released: the next promoter waits until the dead holder's hold passes, takes the
    lease over and crowns the best of every version from a fresh read, though its own is worse."""
    import wandb as library

    from dew.data.storage import Store

    monkeypatch.setenv("WANDB_MODE", "offline")
    _, step = tiny_run(tmp_path)
    api = Api([], {}, {}, tmp_path / "download")
    monkeypatch.setattr(library, "Api", lambda: api)
    path = "wandb-registry-model/shared"
    api.registry.link(types.SimpleNamespace(metadata={"val/loss": 1.0}), path, ())
    api.registry.link(types.SimpleNamespace(metadata={"val/loss": 2.0}), path, ("best",))
    url = str(tmp_path / "shared.lease")
    Store(url).put_if_absent(json.dumps({"holder": "dead", "until": time.time() + 1.5}).encode())
    tracker = Tracker()
    tracker.run = RankedRun({"val/loss": 3.0}, id="next", registry=api.registry)
    began = time.monotonic()
    assert dew.io.promote(str(step), "shared", tracker=tracker, metric="val/loss", top=3,
                          lease=Lease(url, hold=60.0, wait=30.0)) is not None
    assert time.monotonic() - began >= 1.0
    assert len(api.registry.collections[path]) == 3
    assert api.registry.best(path) == {"val/loss": 1.0}
    assert json.loads(Path(url).read_text())["holder"] is None


def test_a_sweeps_configured_trials_promote_into_their_shared_collection(tmp_path, monkeypatch):
    """Two trials of a sweep, each a run of its own name and directory, configured to promote into one
    registry collection (`Promotion.collection`): the second, better than the first, takes the
    collection's best alias, which a downstream `dew.pipeline` follows. A promotion with no W&B sink is
    refused."""
    import autoencoder_experiment
    import wandb as library

    from dew.config import TrainerConfig
    from dew.io import Promotion
    from dew.training import Wandb

    with pytest.raises(ValueError):
        TrainerConfig(steps=1, promote=Promotion("trial/score", "flowers"))
    monkeypatch.setenv("WANDB_MODE", "offline")
    project: list = []
    api = Api(project, {}, {}, tmp_path / "download")
    monkeypatch.setattr(library, "Api", lambda: api)

    class Trial(TrackedRun):
        def __init__(self, name, score):
            super().__init__()
            self.summary = {"trial/score": score}
            self.id, self.sweep_id, self.entity, self.project = name, None, "team", "flowers"

        def link_artifact(self, artifact, target_path, aliases):
            super().link_artifact(artifact, target_path, aliases)
            api.registry.link(artifact, target_path, aliases)

    built = autoencoder_experiment.run()
    for name, score in (("trial-0", 2.0), ("trial-1", 1.0)):
        trial = Trial(name, score)
        monkeypatch.setattr(library, "init", lambda trial=trial, **_: trial)
        trainer = dataclasses.replace(built.trainer, name=name, steps=2, checkpoint_every=None,
                                      checkpoint_final=True, checkpoint_dir=str(tmp_path / "runs"),
                                      wandb=Wandb(project="flowers", offline=True),
                                      promote=Promotion("trial/score", "flowers"))
        dataclasses.replace(built, trainer=trainer, validation=None).run()
        project.append(Other(name, {"trial/score": score}))
    assert len(api.registry.collections["wandb-registry-model/flowers"]) == 2
    assert api.registry.best("wandb-registry-model/flowers") == {"trial/score": 1.0}
