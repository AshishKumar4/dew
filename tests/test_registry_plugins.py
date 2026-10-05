"""Packages outside Dew register into its tables through the `dew.plugins` entry-point group."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _install(site: Path, package: str, source: str, *, entry: str | None = None) -> None:
    """Lay `package` out under `site` as an installed distribution that names itself a Dew plugin."""
    (site / package).mkdir(parents=True)
    (site / package / "__init__.py").write_text("")
    (site / package / "models.py").write_text(source)
    info = site / f"{package}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(f"[dew.plugins]\n{package} = {entry or package}\n")


def _run(site: Path, program: str) -> subprocess.CompletedProcess:
    path = os.pathsep.join([str(ROOT / "src"), str(site)])
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path}
    return subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                          timeout=300)


PLUGIN_MODEL = '''
import flax.linen as nn

from dew.registry import models


@models("plugin_mlp")
class PluginMLP(nn.Module):
    features: int = 4

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)
'''


def test_a_plugin_model_is_found_by_name_without_importing_the_plugin(tmp_path):
    _install(tmp_path, "toyplugin", PLUGIN_MODEL)
    done = _run(tmp_path, "import sys\n"
                          "from dew.registry import models\n"
                          "assert 'toyplugin' not in sys.modules\n"
                          "built = models.build('plugin_mlp', {'features': 7})\n"
                          "print(type(built).__module__, built.features)\n"
                          "print(models.name_of(type(built)))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["toyplugin.models 7", "plugin_mlp"]


def test_an_unknown_name_still_raises_with_plugins_installed(tmp_path):
    _install(tmp_path, "toyplugin", PLUGIN_MODEL)
    done = _run(tmp_path, "from dew.registry import models\n"
                          "try:\n"
                          "    models['no_such_model']\n"
                          "except KeyError as error:\n"
                          "    print('refused', 'plugin_mlp' in str(error))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "refused True"


def test_a_plugin_entry_naming_a_missing_package_raises(tmp_path):
    _install(tmp_path, "toyplugin", PLUGIN_MODEL, entry="not_installed_anywhere")
    done = _run(tmp_path, "from dew.registry import models\n"
                          "models['plugin_mlp']\n")
    assert done.returncode != 0
    assert "not_installed_anywhere" in done.stderr and "dew.plugins" in done.stderr


PLUGIN_KIND = '''
from dataclasses import dataclass

from dew.registry import Registry, share

activations = share(Registry("activation", record="kind"))


class Activation:
    pass


@activations("scaled_tanh")
@dataclass(frozen=True)
class ScaledTanh(Activation):
    scale: float = 1.0
'''

PLUGIN_MODEL_WITH_KIND = '''
import flax.linen as nn

from dew.registry import models

from toyplugin.kinds import Activation


@models("plugin_act")
class PluginAct(nn.Module):
    activation: Activation

    @nn.compact
    def __call__(self, x):
        return x
'''


def test_a_plugin_kind_rebuilds_from_a_record_and_writes_back_its_kind(tmp_path):
    _install(tmp_path, "toyplugin", PLUGIN_MODEL_WITH_KIND)
    (tmp_path / "toyplugin" / "kinds.py").write_text(PLUGIN_KIND)
    done = _run(tmp_path, "from dew.registry import models\n"
                          "from dew.config import _to_json\n"
                          "record = {'activation': {'kind': 'scaled_tanh', 'scale': 2.0}}\n"
                          "built = models.build('plugin_act', record)\n"
                          "print(type(built.activation).__name__, built.activation.scale)\n"
                          "print(_to_json(built.activation, type(built).__annotations__['activation']))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["ScaledTanh 2.0", "{'kind': 'scaled_tanh', 'scale': 2.0}"]


def test_a_second_shared_table_of_one_kind_is_refused():
    import pytest

    from dew.registry import Registry, models, share
    assert share(models) is models
    with pytest.raises(ValueError, match="already shared"):
        share(Registry("model"))


PLUGIN_OBJECTIVE = '''
import jax
import jax.numpy as jnp

from dew.objectives.base import Objective
from dew.registry import objectives


class Scalar:
    """The plugin's saved task: the one weight a run of Shift trained."""

    def __init__(self, value):
        self.value = value

    @classmethod
    def from_run(cls, directory, *, ema=None, step=None, mesh=None, layout=None, dtype=None,
                 param_dtype=None):
        from dew.checkpoints import Checkpoints
        variables = Checkpoints(directory).variables(ema=ema, step=step, mesh=mesh, layout=layout)
        return cls(float(variables["params"]["w"]))


@objectives("shift")
class Shift(Objective):
    saved_task = Scalar

    def init(self, key, variables=None):
        return {"params": {"w": jnp.zeros(())}}

    def loss(self, variables, batch, step):
        return jnp.mean((variables["params"]["w"] - batch["x"]) ** 2)

    def inference_record(self):
        return {"objective": "shift"}
'''


def test_dew_pipeline_loads_a_plugin_objectives_saved_task(tmp_path):
    _install(tmp_path, "toyplugin", PLUGIN_OBJECTIVE)
    run = tmp_path / "run"
    train = ("import jax, numpy as np, optax\n"
             "from dew import Checkpoints, Trainer\n"
             "from dew.data import Dataset, Loading\n"
             "from toyplugin.models import Shift\n"
             "data = Dataset.from_records({'x': np.full((8,), 3.0, np.float32)}, batch=8,\n"
             "                            loading=Loading(workers=0, threads=1, read_buffer=1))\n"
             "trainer = Trainer(Shift(), optax.sgd(0.5), key=jax.random.key(0),\n"
             f"                  checkpoints=Checkpoints({str(run)!r}))\n"
             "state = trainer.fit(data, steps=3, log_every=3, checkpoint_every=3)\n"
             "trainer.checkpoints.wait()\n"
             "print(float(state.variables['params']['w']))\n")
    trained = _run(tmp_path, train)
    assert trained.returncode == 0, trained.stderr[-2000:]
    load = ("import sys\n"
            "import dew\n"
            f"task = dew.pipeline({str(run)!r})\n"
            "print(type(task).__module__, type(task).__name__, task.value)\n")
    loaded = _run(tmp_path, load)
    assert loaded.returncode == 0, loaded.stderr[-2000:]
    module, name, value = loaded.stdout.split()
    assert (module, name) == ("toyplugin.models", "Scalar")
    assert float(value) == float(trained.stdout.split()[-1])
