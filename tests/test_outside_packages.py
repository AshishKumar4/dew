"""Classes from packages outside Dew, named by their import paths: built,
recorded and loaded back with nothing registered, each checked in a fresh
process against a toy package on the path."""

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
    """The model builds by its path, importing the package only then; its
    field typed with the package's own base class takes a record of a class
    derived from it, and the model records both by path."""
    _install(tmp_path, "toypackage", PACKAGE_MODEL)
    done = _run(tmp_path, "import sys\n"
                          "from dew.config import ModelConfig\n"
                          "from dew.registry import models\n"
                          "assert 'toypackage' not in sys.modules\n"
                          "record = {'features': 7, 'activation': {'class': 'toypackage.models:ScaledTanh',\n"
                          "                                        'fields': {'scale': 2.0}}}\n"
                          "built = models.build('toypackage.models:PackageMLP', record)\n"
                          "print(type(built).__name__, built.features, built.activation.scale)\n"
                          "saved = ModelConfig.from_model(built)\n"
                          "print(saved.architecture, saved.config['activation'])\n"
                          "print(saved.build() == built)\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == [
        "PackageMLP 7 2.0",
        "toypackage.models:PackageMLP {'class': 'toypackage.models:ScaledTanh', 'fields': {'scale': 2.0}}",
        "True"]


def test_a_path_naming_nothing_is_refused_by_what_it_names(tmp_path):
    _install(tmp_path, "toypackage", PACKAGE_MODEL)
    done = _run(tmp_path, "from dew.registry import models\n"
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


def test_dew_pipeline_loads_a_packages_objectives_saved_task(tmp_path):
    """A package's objective declares its task the way Dew's do, and a run of
    it trained in one process loads, by the objective's import path, as that
    task in another that has not imported the package, equal to the task
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
    load = ("import sys\n"
            "import dew\n"
            f"task = dew.pipeline({str(run)!r})\n"
            "print(type(task).__module__, type(task).__name__, task.value)\n")
    loaded = _run(tmp_path, load)
    assert loaded.returncode == 0, loaded.stderr[-2000:]
    module, name, value = loaded.stdout.split()
    assert (module, name) == ("toypackage.models", "Scalar")
    assert float(value) == float(trained.stdout.split()[-1])
