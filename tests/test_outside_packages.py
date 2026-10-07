"""Classes from packages outside Dew, named by their import paths: built,
recorded and loaded back with nothing registered, each checked in a fresh
process against a toy package on the path. A record imports such a package
only once it is imported or the loader trusts it."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _install(site: Path, package: str, source: str) -> None:
    """Lay `package` out under `site`, its `models` module holding `source`."""
    (site / package).mkdir(parents=True)
    (site / package / "__init__.py").write_text("")
    (site / package / "models.py").write_text(source)


def _run(site: Path, program: str) -> subprocess.CompletedProcess:
    path = os.pathsep.join([str(ROOT / "src"), str(site)])
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path}
    return subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                          timeout=300, check=False)


PACKAGE_MODEL = '''
from dataclasses import dataclass

import flax.linen as nn


class Activation:
    pass


@dataclass(frozen=True)
class ScaledTanh(Activation):
    scale: float = 1.0


class PackageMLP(nn.Module):
    features: int = 4
    activation: Activation = ScaledTanh()

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)
'''


def test_a_packages_model_and_its_own_kind_build_and_record_by_import_path(tmp_path):
    """The model builds by its path once its package is trusted, which imports
    the module the record names; its field typed with the package's own base
    class takes a record of a class derived from it, and the model records
    both by path."""
    _install(tmp_path, "toypackage", PACKAGE_MODEL)
    done = _run(tmp_path, "import sys\n"
                          "from dew.config import ModelConfig\n"
                          "from dew.registry import import_trusted, models\n"
                          "assert 'toypackage' not in sys.modules\n"
                          "record = {'features': 7, 'activation': {'class': 'toypackage.models:ScaledTanh',\n"
                          "                                        'fields': {'scale': 2.0}}}\n"
                          "import_trusted(record, ('toypackage',))\n"
                          "built = models.build('toypackage.models:PackageMLP', record)\n"
                          "print(type(built).__name__, built.features, built.activation.scale)\n"
                          "saved = ModelConfig.from_model(built)\n"
                          "print(saved.name, saved.fields['activation'])\n"
                          "print(saved.build() == built)\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == [
        "PackageMLP 7 2.0",
        "toypackage.models:PackageMLP {'class': 'toypackage.models:ScaledTanh', 'fields': {'scale': 2.0}}",
        "True"]


def test_a_path_naming_nothing_is_refused_by_what_it_names(tmp_path):
    _install(tmp_path, "toypackage", PACKAGE_MODEL)
    done = _run(tmp_path, "import toypackage.models\n"
                          "from dew.registry import models\n"
                          "for name in ('toypackage.models:Nope', 'no_such_model'):\n"
                          "    try:\n"
                          "        models[name]\n"
                          "    except (KeyError, ValueError) as error:\n"
                          "        print(type(error).__name__, str(error)[:60])\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == [
        "ValueError toypackage.models has no Nope, which 'toypackage.models:Nope",
        "KeyError \"no model named 'no_such_model'; known: causal_transformer, "]


PACKAGE_OBJECTIVE = '''
import jax
import jax.numpy as jnp

from dew.objectives.base import Objective


class Scalar:
    """The package's saved task: the one weight a run of Shift trained."""

    def __init__(self, value):
        self.value = value

    @classmethod
    def from_run(cls, directory, *, ema=None, step=None, mesh=None, layout=None, dtype=None,
                 param_dtype=None):
        from dew.checkpoints import Checkpoints
        variables = Checkpoints(directory).variables(ema=ema, step=step, mesh=mesh, layout=layout)
        return cls(float(variables["params"]["w"]))


class Shift(Objective):
    saved_task = Scalar

    def init(self, key, variables=None):
        return {"params": {"w": jnp.zeros(())}}

    def loss(self, variables, batch, step):
        return jnp.mean((variables["params"]["w"] - batch["x"]) ** 2)

    def inference_record(self):
        return {"objective": "toypackage.models:Shift"}

    def pipeline(self, state, *, ema=None) -> Scalar:
        return Scalar(float(self._pipeline_weights(state, ema)["params"]["w"]))
'''


def test_dew_pipeline_loads_a_packages_run_only_when_trusted(tmp_path):
    """A package's objective declares its task the way Dew's do. A run of it,
    as a Hub download would hold it, is refused in a process that has not
    imported the package, naming the module and the flag; trusted, it loads
    by the objective's import path as that task, equal to the task
    `pipeline` returns in place."""
    _install(tmp_path, "toypackage", PACKAGE_OBJECTIVE)
    run = tmp_path / "run"
    train = ("import jax, numpy as np, optax\n"
             "from dew import Checkpoints, Trainer\n"
             "from dew.data import Dataset, Loading\n"
             "from toypackage.models import Shift\n"
             "data = Dataset.from_records({'x': np.full((8,), 3.0, np.float32)}, batch=8,\n"
             "                            loading=Loading(workers=0, threads=1, read_buffer=1))\n"
             "trainer = Trainer(Shift(), optax.sgd(0.5), key=jax.random.key(0),\n"
             f"                  checkpoints=Checkpoints({str(run)!r}))\n"
             "state = trainer.fit(data, steps=3, log_every=3, checkpoint_every=3)\n"
             "trainer.checkpoints.wait()\n"
             "print(trainer.objective.pipeline(state).value)\n")
    trained = _run(tmp_path, train)
    assert trained.returncode == 0, trained.stderr[-2000:]
    refused = _run(tmp_path, f"import dew\ndew.pipeline({str(run)!r})\n")
    assert refused.returncode != 0
    assert ("names toypackage.models, which is outside Dew and not imported" in refused.stderr
            and "trust=('toypackage',), --trust toypackage" in refused.stderr)
    load = ("import dew\n"
            f"task = dew.pipeline({str(run)!r}, trust=('toypackage',))\n"
            "print(type(task).__module__, type(task).__name__, task.value)\n")
    loaded = _run(tmp_path, load)
    assert loaded.returncode == 0, loaded.stderr[-2000:]
    module, name, value = loaded.stdout.split()
    assert (module, name) == ("toypackage.models", "Scalar")
    assert float(value) == float(trained.stdout.split()[-1])


PACKAGE_RUN = '''
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct

from dew.config import Prepared, RunConfig, TrainerConfig
from dew.data import Dataset, Loading
from dew.objectives.base import Objective, mean_of_totals, merge_totals
from dew.training import DEFAULT_RULES, Layout, MeshSpec


@struct.dataclass
class Activity:
    """The package's own scoring artifact: each row's mean firing rate."""

    rates: jax.Array


@dataclasses.dataclass(frozen=True)
class MeanRate:
    """The pass's mean rate, read from the package's artifact by its type."""

    name = "mean_rate"
    reads = Activity

    def __call__(self, artifact, batch, /):
        return float(np.sum(artifact.rates)), float(np.shape(artifact.rates)[0])

    merge = staticmethod(merge_totals)
    finalize = staticmethod(mean_of_totals)


class Population(nn.Module):
    """Neurons whose weights split along the package's own logical axis."""

    neurons: int = 64

    @nn.compact
    def __call__(self, x):
        init = nn.with_logical_partitioning(nn.initializers.normal(), ("inputs", "neurons"))
        return jax.nn.sigmoid(x @ self.param("w", init, (x.shape[-1], self.neurons)))


class Rate(Objective):
    """Drives every neuron's rate toward `target`."""

    def __init__(self, model, target):
        self.model, self.target = model, target

    def init(self, key, variables=None):
        return self.model.init(key, jnp.zeros((1, 8)))

    def loss(self, variables, batch, step):
        return jnp.mean((self.model.apply(variables, batch["x"]) - self.target) ** 2)

    def evaluate(self, params, batch, step):
        return Activity(jnp.mean(self.model.apply(params, batch["x"]), axis=-1))


@dataclasses.dataclass(frozen=True)
class ActivityRun(RunConfig):
    """The package's own kind of run, whose layout places its axis."""

    target: float = 0.25
    trainer: TrainerConfig = dataclasses.field(default_factory=lambda: TrainerConfig(
        mesh=MeshSpec(fsdp=2), layout=Layout(rules=(*DEFAULT_RULES, ("neurons", "fsdp")), min_shard=0)))

    def prepare(self):
        rows = np.random.default_rng(0).normal(size=(64, 8)).astype(np.float32)
        data = Dataset.from_records({"x": rows}, batch=self.trainer.batch_size, validation={"x": rows[:20]},
                                    loading=Loading(workers=0, threads=1, read_buffer=1))
        objective = Rate(self.model.build(), self.target)
        return Prepared(self, lambda name: self.train(objective, data, name=name, metrics=(MeanRate(),)))
'''

TRAIN_RUN = '''
import dataclasses, sys
from dew.config import ModelConfig, TrainerConfig
from toypackage.models import ActivityRun
default = ActivityRun().trainer
run = ActivityRun(model=ModelConfig("toypackage.models:Population", {"neurons": 64}),
                  trainer=dataclasses.replace(default, steps=4, batch_size=8, eval_every=4, log_every=1,
                                              checkpoint_dir=sys.argv[1], name="activity",
                                              compilation_cache_dir=None, multi_host=False))
state = run.run()
print(tuple(state.variables["params"]["w"].sharding.spec))
'''


def test_a_packages_own_run_artifact_metric_and_axis_train_from_its_record_in_a_new_process(tmp_path):
    """A package's run class builds its objective and data in `prepare`, its
    objective scores into an artifact of its own that its metric reads by
    type, and its layout places the package's own logical axis on the mesh,
    with nothing in Dew naming any of it. A new process trains the run again
    from its `run.json` alone once it trusts the package, to the same losses,
    and refuses it until then."""
    import json

    _install(tmp_path, "toypackage", PACKAGE_RUN)
    devices = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    path = os.pathsep.join([str(ROOT / "src"), str(tmp_path)])

    def run(*argv):
        return subprocess.run([sys.executable, *argv], capture_output=True, text=True, timeout=600,
                              check=False, cwd=tmp_path,
                              env={**devices, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path})

    def scalars(directory, name):
        rows = [json.loads(row)["scalars"] for row in
                (directory / "activity" / "tracking" / "scalars.jsonl").read_text().splitlines()]
        return [row[name] for row in rows if name in row]

    trained = run("-c", TRAIN_RUN, str(tmp_path / "first"))
    assert trained.returncode == 0, trained.stderr[-3000:]
    assert trained.stdout.splitlines()[-1] == "(None, 'fsdp')"
    assert len(scalars(tmp_path / "first", "train/loss")) == 4
    assert 0.0 < scalars(tmp_path / "first", "val/mean_rate")[-1] < 1.0

    record = str(tmp_path / "first" / "activity" / "run.json")
    again = ("-m", "dew.cli.main", "train", record, "--set", f"trainer.checkpoint_dir={tmp_path / 'again'}")
    refused = run(*again)
    assert refused.returncode != 0 and "--trust toypackage" in refused.stderr
    retrained = run(*again, "--trust", "toypackage")
    assert retrained.returncode == 0, retrained.stderr[-3000:]
    assert scalars(tmp_path / "again", "train/loss") == scalars(tmp_path / "first", "train/loss")
    assert scalars(tmp_path / "again", "val/mean_rate") == scalars(tmp_path / "first", "val/mean_rate")
