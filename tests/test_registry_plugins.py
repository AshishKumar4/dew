"""Packages outside Dew register into its tables through the `dew.plugins`
entry-point group, each checked in a fresh process against a toy plugin
installed as a distribution."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dew.registry import Registry, models

ROOT = Path(__file__).resolve().parents[1]


def _install(site: Path, package: str, source: str, *, entry: str | None = None) -> None:
    """Lay `package` out under `site` as an installed distribution whose
    `dew.plugins` entry names `entry` (the package itself by default),
    whose `__init__` imports its `models` module, as a plugin's
    registering module does."""
    (site / package).mkdir(parents=True)
    (site / package / "__init__.py").write_text("from . import models  # noqa: F401\n")
    (site / package / "models.py").write_text(source)
    info = site / f"{package}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(f"[dew.plugins]\n{package} = {entry or package}\n")


def _run(site: Path, program: str) -> subprocess.CompletedProcess:
    path = os.pathsep.join([str(ROOT / "src"), str(site)])
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path}
    return subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                          timeout=300, check=False)


PLUGIN_MODEL = '''
"""A plugin's models. A docstring may show a registration without making one:

    @models("phantom_mlp")
    class Phantom(nn.Module): ...
"""
import flax.linen as nn

from dew.registry import models as register


@register("plugin_mlp")
class PluginMLP(nn.Module):
    features: int = 4

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)
'''


def test_a_plugin_model_is_found_by_name_without_importing_the_plugin(tmp_path):
    """The plugin registers through an alias of `models`, which no reading
    of its source would match; loading its entry is what registers it, and
    only once a name misses Dew's own index."""
    _install(tmp_path, "toyplugin", PLUGIN_MODEL)
    done = _run(tmp_path, "import sys\n"
                          "from dew.registry import models\n"
                          "models['simple_dit']\n"
                          "assert 'toyplugin' not in sys.modules\n"
                          "built = models.build('plugin_mlp', {'features': 7})\n"
                          "print(type(built).__module__, built.features)\n"
                          "print(models.name_of(type(built)))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["toyplugin.models 7", "plugin_mlp"]


def test_a_registration_shown_in_a_docstring_is_no_member(tmp_path):
    """The plugin's docstring shows `@models("phantom_mlp")`. It names no
    member: the lookup raises, and the names it lists are only the ones
    importing the plugin registered."""
    _install(tmp_path, "toyplugin", PLUGIN_MODEL)
    done = _run(tmp_path, "from dew.registry import models\n"
                          "try:\n"
                          "    models['phantom_mlp']\n"
                          "except KeyError as error:\n"
                          "    known = str(error).split('known:')[1]\n"
                          "    print('refused', 'plugin_mlp' in known, 'phantom' in known)\n"
                          "print('phantom_mlp' in models)\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["refused True False", "False"]


BROKEN = 'raise RuntimeError("the plugin is broken")\n'


def test_a_broken_plugin_breaks_no_lookup_dew_answers(tmp_path):
    """An entry that names a missing module and one whose import raises.
    Every name Dew registers still builds, and `in` and `get` answer for
    a name nobody registers without raising."""
    _install(tmp_path, "brokenplugin", BROKEN)
    _install(tmp_path, "missingplugin", PLUGIN_MODEL, entry="not_installed_anywhere")
    done = _run(tmp_path, "from dew.registry import models, objectives, schedules\n"
                          "fields = {'patch_size': 2, 'emb_features': 8, 'num_layers': 1, 'num_heads': 2}\n"
                          "print(type(models.build('simple_dit', fields)).__name__)\n"
                          "print(objectives['lm'].__name__, schedules['cosine'].__name__)\n"
                          "print('no_such_model' in models, models.get('no_such_model'))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["SimpleDiT", "LMObjective Cosine", "False None"]


def test_a_broken_plugin_is_named_by_a_lookup_that_still_misses(tmp_path):
    """A name nobody registers raises, naming each entry that failed to
    import and why, the first failure chained."""
    _install(tmp_path, "brokenplugin", BROKEN)
    _install(tmp_path, "missingplugin", PLUGIN_MODEL, entry="not_installed_anywhere")
    done = _run(tmp_path, "from dew.registry import models\n"
                          "models['plugin_mlp']\n")
    assert done.returncode != 0
    error = done.stderr.splitlines()[-1]
    assert error.startswith("KeyError") and "no model named 'plugin_mlp'" in error
    assert "dew.plugins" in error
    assert "brokenplugin = brokenplugin (RuntimeError: the plugin is broken)" in error
    assert "missingplugin = not_installed_anywhere (ModuleNotFoundError" in error
    assert "The above exception was the direct cause" in done.stderr


def test_an_unknown_name_still_raises_with_plugins_installed(tmp_path):
    _install(tmp_path, "toyplugin", PLUGIN_MODEL)
    done = _run(tmp_path, "from dew.registry import models\n"
                          "try:\n"
                          "    models['no_such_model']\n"
                          "except KeyError as error:\n"
                          "    print('refused', 'plugin_mlp' in str(error), 'dew.plugins' in str(error))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "refused True False"


PLUGIN_KIND = '''
from dataclasses import dataclass

from dew.registry import Registry

activations = Registry("activation").share()


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
    """The kind's table is shared in the module that defines its base class,
    so the model's field, typed with that class, finds it."""
    _install(tmp_path, "toyplugin", PLUGIN_MODEL_WITH_KIND)
    (tmp_path / "toyplugin" / "kinds.py").write_text(PLUGIN_KIND)
    done = _run(tmp_path, "from dew.registry import Registry, models\n"
                          "from dew.registry import to_record\n"
                          "record = {'activation': {'name': 'scaled_tanh', 'fields': {'scale': 2.0}}}\n"
                          "built = models.build('plugin_act', record)\n"
                          "print(type(built.activation).__name__, built.activation.scale)\n"
                          "print(to_record(built.activation, type(built).__annotations__['activation']))\n"
                          "print([table.kind for table in Registry.shared()][-1])\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["ScaledTanh 2.0", "{'name': 'scaled_tanh', 'fields': {'scale': 2.0}}",
                                        "activation"]


def test_a_kind_has_one_shared_table():
    """Sharing a table again returns it; a second table of a shared kind is
    refused, and the shared tables are a tuple no caller writes to."""
    assert models.share() is models
    with pytest.raises(ValueError, match="already shared"):
        Registry("model").share()
    shared = Registry.shared()
    assert isinstance(shared, tuple) and sum(table is models for table in shared) == 1


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

    def pipeline(self, state, *, ema=None) -> Scalar:
        return Scalar(float(self._pipeline_weights(state, ema)["params"]["w"]))
'''


def test_dew_pipeline_loads_a_plugin_objectives_saved_task(tmp_path):
    """A plugin objective declares its task the way Dew's do, and a run of
    it trained in one process loads as that task in another that has not
    imported the plugin, equal to the task `pipeline` returns in place."""
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
    assert (module, name) == ("toyplugin.models", "Scalar")
    assert float(value) == float(trained.stdout.split()[-1])
