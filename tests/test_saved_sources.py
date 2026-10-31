"""A run that starts from a saved Dew run (`--pretrained <run directory>`, ledger 2713): its source is the
release `dew export` writes of the run's checkpoint (`dew.interop.saved_sources`), or a diffusion run's
own sampling task, at the step its selection pins, which its record names."""

import dataclasses
from pathlib import Path

import jax
import numpy as np
import pytest
from process_support import run_pool
from run_kinds import decision, diffusion, trained

from dew.batch import FROZEN
from dew.config import ObjectiveConfig, SourceOptions
from dew.config.diffusion import DEFAULT_MODEL_CONFIG, DiffusionRunConfig
from dew.decision.config import DecisionRunConfig
from dew.inference import TextToImage
from dew.inference.tasks import SavedRun
from dew.interop.pretrained import load_diffusion_source
from dew.interop.saved_sources import export_run, released_run
from dew.lora import LoRA
from dew.registry import ModelConfig
from dew.sampling.solvers import Euler


@pytest.mark.mesh(devices=8)
def test_a_decision_run_starts_from_a_saved_decision_run_as_from_its_export(tmp_path, monkeypatch):
    """A decision run named on the command line by a saved decision run (`--pretrained <run>`,
    `--source.step latest`) starts from the weights its export does (`dew export`):
    one step from each, alike in every other option, lands every leaf bit for bit alike, and its record
    names the run it started from and the step it read. Its record resumes on a host whose cache lacks
    the export (another node's, ledger 2774). A step beside a source that is no run is refused."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    first = decision(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    exported = tmp_path / "exported"
    export_run(SavedRun.pinned(str(source)), exported)

    def started(pretrained, name: str, *flags: str):
        run = DecisionRunConfig.cli(["--pretrained", str(pretrained), *flags], default=first)
        return trained(run, tmp_path, name, trainer={"checkpoint_final": True})[0]

    from_run = started(source, "from_run", "--source.step", "latest")
    from_export = started(exported, "from_export")
    leaves_alike(from_run.variables, from_export.variables)
    recorded = DecisionRunConfig.load(str(tmp_path / "runs" / "from_run"))
    assert recorded.pretrained == str(source)
    assert recorded.source.step == SavedRun.pinned(str(source)).step
    # Named, the run resumes from its own checkpoint (`TrainerConfig.resumes`), reading its source without
    # weights, on a host whose cache holds no release.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "another_cache"))
    trained(recorded, tmp_path, "from_run",
            trainer={"name": "from_run", "steps": 2, "checkpoint_final": True})
    assert "2" in {path.name for path in (tmp_path / "runs" / "from_run").iterdir()}
    assert released_run(source, step="latest") == released_run(source)
    with pytest.raises(ValueError):
        released_run(exported, step="latest")
    with pytest.raises(ValueError):
        load_diffusion_source(str(source))


def leaves_alike(tree, alike) -> None:
    for leaf, other in zip(jax.tree.leaves(jax.device_get(tree)), jax.tree.leaves(jax.device_get(alike)),
                           strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(other), strict=True)


@pytest.mark.mesh(devices=8)
def test_a_diffusion_run_starts_from_a_saved_diffusion_run_as_from_its_task(tmp_path):
    """A diffusion run that starts from a saved diffusion run (`--pretrained <run>`, `--source.step latest`)
    starts from the weights the run's own task restores (`TextToImage.from_run`), bit for bit, and samples
    by that task's policy where it states none. It trains a step, and its record names the run it started
    from and the step it read; an adapter over it trains its factors over the run's frozen weights, and a
    run that trained an adapter starts another with its base frozen. The source is at the run's own size,
    another refused, and a language model's loader refuses a diffusion run, which has no release."""
    first = diffusion(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    run = dataclasses.replace(first, pretrained=str(source), source=SourceOptions(step="latest"), text=None,
                              model=ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)),
                              objective=ObjectiveConfig("diffusion", {}))
    task = TextToImage.from_run(str(source))

    objective = run.build()
    leaves_alike(objective.variables["params"], task.variables["params"])
    assert (objective.steps, objective.solver) == (task.steps, task.solver) == (2, Euler())
    trained(run, tmp_path, "from_run")
    recorded = DiffusionRunConfig.load(str(tmp_path / "runs" / "from_run"))
    assert recorded.pretrained == str(source)
    assert recorded.source.step == SavedRun.pinned(str(source)).step
    adapted, _ = trained(run, tmp_path, "adapted", lora=LoRA(rank=2, modules=("ada_proj",)))
    assert FROZEN in adapted.variables
    factors = sum(leaf.size for leaf in jax.tree.leaves(adapted.variables["params"]))
    assert 0 < factors < sum(leaf.size for leaf in jax.tree.leaves(task.variables["params"]))

    # A run that trained an adapter starts another with its base frozen and its factors training.
    trained(first, tmp_path, "adapted_first", trainer={"checkpoint_final": True},
            lora=LoRA(rank=2, modules=("ada_proj",)))
    from_adapted, _ = trained(dataclasses.replace(run, pretrained=str(tmp_path / "runs" / "adapted_first")),
                              tmp_path, "from_adapted")
    assert FROZEN in from_adapted.variables
    assert sum(leaf.size for leaf in jax.tree.leaves(from_adapted.variables["params"])) == factors

    sized = load_diffusion_source(str(source), dtype="float32", size=task.inputs.sample.shape[:-1])
    assert sized.inputs.sample == task.inputs.sample
    with pytest.raises(ValueError):
        load_diffusion_source(str(source), dtype="float32", size=(8, 8))
    with pytest.raises(ValueError):
        released_run(source)


WORKER = Path(__file__).with_name("saved_export_worker.py")


@pytest.mark.distributed
@pytest.mark.mesh
def test_a_pool_exports_a_model_spanning_its_processes_a_shard_at_a_time(tmp_path):
    """Two real processes export a decoder whose kernels span both as a saved run's release (`export_run`,
    ledger 2874), in a score of shards: each reads one tensor home at a time, together, so neither host's
    allocations during the export peak at half the model, where the pool read the whole tree home
    first. The publishing process writes the files one process writes of the same weights unplaced,
    byte for byte; the other writes the release's small files alone, no weights. Outside the export's
    scope the write is refused rather than left waiting on the other process."""
    publisher, reader = run_pool("exported", tmp_path / "pool", processes=2, script=WORKER, devices=2,
                                 fsdp_size=4, run_dir=str(tmp_path))
    for report in (publisher, reader):
        assert report["processes"] == 2 and report["spanning"] > 0
        assert report["refused_outside"]
        assert report["largest_leaf"] <= report["peak"] < report["whole"] / 2
    assert publisher["differing"] == {}
    assert publisher["release"] == publisher["alone"] == reader["alone"] and len(publisher["release"]) > 10
    assert reader["release"] == {} and reader["small_files"] == publisher["small_files"]

