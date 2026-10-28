"""Classes from packages outside Dew, named by their import paths: built,
recorded and loaded back with nothing registered, each checked in a fresh
process against a toy package on the path. A record imports such a package
only once it is imported or the loader trusts it."""

import json
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


PACKAGE_OBJECTIVE = '''
import jax
import jax.numpy as jnp

from dew.objectives.base import Objective


class Scalar:
    """The package's saved task: the one weight a run of Shift trained."""

    def __init__(self, value):
        self.value = value

    @classmethod
    def from_saved(cls, run, *, ema=None, mesh=None, layout=None, dtype=None, param_dtype=None):
        variables = run.variables(ema=ema, mesh=mesh, layout=layout)
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
    imported the package, and the refusal imports none of it; trusted, it
    loads by the objective's import path as that task, equal to the task
    `pipeline` returns in place."""
    _install(tmp_path, "toypackage", PACKAGE_OBJECTIVE)
    run = tmp_path / "run"
    train = ("import jax, numpy as np, optax\n"
             "from dew import Checkpoints, Trainer\n"
             "from dew.data import Loading, Memory\n"
             "from toypackage.models import Shift\n"
             "rows = Memory({'x': np.full((8,), 3.0, np.float32)}).read('memory')\n"
             "data = rows.shuffle().repeat().batch(8).loading(Loading(workers=0, threads=1, read_buffer=1))\n"
             "trainer = Trainer(Shift(), optax.sgd(0.5), key=jax.random.key(0),\n"
             f"                  checkpoints=Checkpoints({str(run)!r}))\n"
             "state = trainer.fit(data, steps=3, log_every=3, checkpoint_every=3)\n"
             "trainer.checkpoints.wait()\n"
             "print(trainer.objective.pipeline(state).value)\n")
    trained = _run(tmp_path, train)
    assert trained.returncode == 0, trained.stderr[-2000:]
    refuse = ("import json, sys\nimport dew\n"
              "try:\n"
              f"    dew.pipeline({str(run)!r})\n"
              "except ValueError:\n"
              "    print(json.dumps({'refused': True, 'imported': 'toypackage' in sys.modules}))\n")
    refused = _run(tmp_path, refuse)
    assert refused.returncode == 0, refused.stderr[-2000:]
    assert json.loads(refused.stdout.splitlines()[-1]) == {"refused": True, "imported": False}
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
from dew.data import Loading, Memory
from dew.mesh import DEFAULT_RULES
from dew.objectives.base import Objective, mean_of_totals, merge_totals
from dew.training import Layout, MeshSpec


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
        loading = Loading(workers=0, threads=1, read_buffer=1)
        data = Memory({"x": rows}).read("memory").shuffle().repeat().batch(8).loading(loading)
        held = Memory({"x": rows[:20]}).read("memory").batch(8).loading(loading)
        objective = Rate(self.model.build(), self.target)
        return Prepared(self, lambda name: self.train(objective, data, name=name, metrics=(MeanRate(),),
                                                      validation=held))
'''

TRAIN_RUN = '''
import dataclasses, sys
from dew.config import TrainerConfig
from dew.registry import ModelConfig
from toypackage.models import ActivityRun
default = ActivityRun().trainer
run = ActivityRun(model=ModelConfig("toypackage.models:Population", {"neurons": 64}),
                  trainer=dataclasses.replace(default, steps=4, eval_every=4, log_every=1,
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
    # Refused before anything trains: the run writes nothing of its own.
    assert refused.returncode != 0 and not (tmp_path / "again").exists()
    retrained = run(*again, "--trust", "toypackage")
    assert retrained.returncode == 0, retrained.stderr[-3000:]
    assert scalars(tmp_path / "again", "train/loss") == scalars(tmp_path / "first", "train/loss")
    assert scalars(tmp_path / "again", "val/mean_rate") == scalars(tmp_path / "first", "val/mean_rate")
