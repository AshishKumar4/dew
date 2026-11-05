"""A run that starts from a saved Dew run (`--pretrained <run directory>`, ledger 2713): its source is the
release `dew export` writes of the run's checkpoint (`dew.interop.saved_sources`), or a diffusion run's
own sampling task, at the step its selection pins, which its record names."""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from process_support import run_pool
from run_kinds import decision, diffusion, lm, trained

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


@pytest.mark.mesh(devices=8)
def test_a_run_started_from_a_saved_run_resumes_its_step_after_that_run_trains_on(tmp_path):
    """A named run started from a saved Dew run (`--pretrained <run>`) at the step its selection names by
    default (a diffusion run's, its latest) records that step; when the saved run has trained on, so that
    its selection names a later one, the same command resuming takes the step it read before it pins
    (`RunConfig.BUILT`, `source.step`, `resumed`), and its source compares alike with the record's, so
    it trains on past its checkpoint, sampling by the policy it recorded (row 3864); the command naming
    the later step keeps it, and its resume is refused."""
    from dew.config import earlier_run

    first = diffusion(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"name": "first", "checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    read = SavedRun.pinned(str(source)).step
    command = dataclasses.replace(first, pretrained=str(source), text=None,
                                  model=ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)),
                                  objective=ObjectiveConfig("diffusion", {}))
    trained(command, tmp_path, "after", trainer={"name": "after", "checkpoint_final": True})
    trained(first, tmp_path, "first", trainer={"name": "first", "steps": 2, "checkpoint_final": True})
    later = SavedRun.pinned(str(source)).step
    assert later != read
    again = dataclasses.replace(command, trainer=dataclasses.replace(
        first.trainer, name="after", checkpoint_dir=str(tmp_path / "runs")))
    saved = earlier_run(json.loads((tmp_path / "runs" / "after" / "run.json").read_text()))
    recorded = DiffusionRunConfig.read(saved)
    assert recorded.source.step == read
    assert again.resumed().pinned().source == recorded.source
    stated = dataclasses.replace(again, source=SourceOptions(step=later))
    assert stated.resumed().pinned().source.step == later
    with pytest.raises(ValueError):
        dataclasses.replace(recorded, source=SourceOptions(step=later)).check_resume(saved)
    trained(command, tmp_path, "after", trainer={"name": "after", "steps": 2, "checkpoint_final": True})
    assert "2" in {path.name for path in (tmp_path / "runs" / "after").iterdir()}


@pytest.mark.mesh(devices=8)
def test_a_saved_decision_runs_step_reads_the_calibration_fitted_on_it_after_a_later_one_is(tmp_path):
    """A decision run calibrated at its first step and trained on, calibrated again at its second, reads
    each step with the calibration fitted on it (`Decide.save` keeps each, `CALIBRATIONS`, and a latest an
    earlier Dew kept alone by its own weights before replacing it): a reader pinned to the first step, a
    run started from it among them, loads it after the second was fitted, where only the latest was kept
    and refused it (rows 3865, 3877)."""
    import shutil

    from dew.decision.calibration import Calibration
    from dew.decision.task import Decide
    from dew.registry import from_record

    first = decision(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"name": "first", "checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    read = SavedRun.pinned(str(source)).step
    # The run as a Dew that kept no calibration by step left it: its latest alone, which the next fit
    # keeps by its own weights before it replaces it (row 3877).
    shutil.rmtree(source / "calibrations")
    trained(first, tmp_path, "first", trainer={"name": "first", "steps": 2, "checkpoint_final": True})
    later = SavedRun.pinned(str(source)).step
    assert later != read
    tasks = {step: Decide.from_saved(SavedRun.pinned(str(source), step)) for step in (read, later)}
    from dew.decision.task import step_calibrations

    for step, task in tasks.items():
        assert task.weights is not None and task.weights.step == step
        (kept,) = step_calibrations(str(source), step)
        assert task.calibration == from_record(Calibration, json.loads(kept.read_text())["calibration"])
    # A run started from the first step trains, as `dew train --pretrained <run> --source.step` starts it
    # (row 3880).
    pinned = DecisionRunConfig.cli(["--pretrained", str(source), "--source.step", str(read)], default=first)
    trained(pinned, tmp_path, "from_first", trainer={"checkpoint_final": True})
    assert DecisionRunConfig.load(str(tmp_path / "runs" / "from_first")).source.step == read
    # The release of the first step is written again where its own calibration changes, the run's latest
    # as it was (`released_run`'s fingerprint holds the step's calibrations).
    release = released_run(source, step=read)
    (kept,) = step_calibrations(str(source), read)
    changed = json.loads(kept.read_text())
    changed["name"] = "recalibrated"
    kept.write_text(json.dumps(changed))
    assert released_run(source, step=read) != release


def leaves_alike(tree, alike) -> None:
    for leaf, other in zip(jax.tree.leaves(jax.device_get(tree)), jax.tree.leaves(jax.device_get(alike)),
                           strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(other), strict=True)


@pytest.mark.mesh(devices=8)
def test_a_diffusion_run_starts_from_a_saved_diffusion_run_as_from_its_task(tmp_path):
    """A diffusion run that starts from a saved diffusion run (`--pretrained <run>`, `--source.step latest`)
    starts from the weights the run's own task restores (`TextToImage.from_run`), bit for bit, and samples
    by that task's policy where it states none. It trains a step, and its record names the run it started
    from and the step it read; an adapter over it trains its factors over the run's frozen weights. The
    source is at the run's own size, another refused, and a language model's loader refuses a diffusion
    run, which has no release. A run that trained an adapter starts another with its base frozen
    (`test_a_diffusion_run_starts_from_an_adapted_run_with_its_base_frozen`; the two took 135 s as one
    test, ledger 3899)."""
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

    sized = load_diffusion_source(str(source), dtype="float32", size=task.inputs.sample.shape[:-1])
    assert sized.inputs.sample == task.inputs.sample
    with pytest.raises(ValueError):
        load_diffusion_source(str(source), dtype="float32", size=(8, 8))
    with pytest.raises(ValueError):
        released_run(source)


@pytest.mark.mesh(devices=8)
def test_a_diffusion_run_starts_from_an_adapted_run_with_its_base_frozen(tmp_path):
    """A run that trained an adapter starts another (`--pretrained <run>`) with its base frozen and its
    factors training: as many as the adapter it starts from trained."""
    first = diffusion(tmp_path, drawn=False)
    adapter, _ = trained(first, tmp_path, "adapted_first", trainer={"checkpoint_final": True},
                         lora=LoRA(rank=2, modules=("ada_proj",)))
    run = dataclasses.replace(first, pretrained=str(tmp_path / "runs" / "adapted_first"),
                              source=SourceOptions(step="latest"), text=None,
                              model=ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)),
                              objective=ObjectiveConfig("diffusion", {}))
    from_adapted, _ = trained(run, tmp_path, "from_adapted")
    assert FROZEN in from_adapted.variables
    factors = sum(leaf.size for leaf in jax.tree.leaves(adapter.variables["params"]))
    assert 0 < sum(leaf.size for leaf in jax.tree.leaves(from_adapted.variables["params"])) == factors


WORKER = Path(__file__).with_name("saved_export_worker.py")


@pytest.mark.distributed
@pytest.mark.mesh
def test_a_pool_exports_a_model_spanning_its_processes_a_shard_at_a_time(tmp_path):
    """Two real processes export a decoder whose kernels span both, every layer's attention extended by
    DEX, as a saved run's release (`export_run`, ledger 2874), in a score of shards, each output
    projection folded as it is written: each reads one tensor home at a time, together, so neither host's
    resident pages rise during the export by half the model (2.0 to 5.5 MB of 31 MB on armada), where the
    pool reading the whole tree home first rises past half (23 to 39 MB), and the same measure over that
    tree held home rises at least half of it on each host, so the measure sees the bytes in that run
    (`saved_export_worker.resident_growth`, a decoder of 256 KiB tensors): tracemalloc never saw the
    buffers a tensor is read home into, and held a 1.3 MB decoder's export to its own fixed cost, which
    CI's process state crossed (ledger 3794). The publishing process writes the tensors one process
    writes of the same weights unplaced, byte for byte, though the restored tree's key order may place
    one in another shard; the other writes the release's small files alone, no weights. Outside the
    export's scope the write is refused rather than left waiting on the other process."""
    publisher, reader = run_pool("exported", tmp_path / "pool", processes=2, script=WORKER, devices=2,
                                 fsdp_size=4, run_dir=str(tmp_path))
    for report in (publisher, reader):
        assert report["processes"] == 2 and report["spanning"] > 0
        assert report["refused_outside"]
        assert report["exported"] < report["whole"] / 2 <= report["held"]
    assert publisher["differing"] == {} and len(publisher["release"]) > 10
    assert publisher["alone"] == reader["alone"]
    assert reader["release"] == {} and reader["small_files"] == publisher["small_files"]


@pytest.mark.distributed
@pytest.mark.mesh
def test_a_pool_reads_a_saved_runs_release_through_one_publisher(tmp_path, monkeypatch):
    """Two real processes sharing Dew's cache read a saved language model run as a source (`released_run`,
    ledger 2992): the release is written once, the first process publishing it and the other reading
    every tensor as it does, writing none, each host's resident pages rising inside the export by less
    than half the release, written in shards a fraction of it, where the same measure over the release
    held home rises at least half of it (`saved_export_worker.resident_growth`): at one shard, the 5 GB
    default, the writer held the whole release, which tracemalloc did not see (ledger 3794). Both read
    the one release. `dew export` to a destination they share writes it once
    likewise, the tensors one process's export writes."""
    from dew.cli.main import main
    from dew.registry import ModelConfig

    run = dataclasses.replace(lm(tmp_path, drawn=False), model=ModelConfig(
        "causal_transformer", {"emb_features": 256, "num_layers": 4, "num_heads": 4, "dtype": "float32"}))
    trained(run, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "shared_cache"))
    publisher, reader = run_pool("released", tmp_path / "pool", processes=2, script=WORKER, devices=2,
                                 fsdp_size=4, run_dir=str(source))
    assert publisher["processes"] == 2 and publisher["release"] == reader["release"]
    assert publisher["release_files"] == reader["release_files"] and publisher["release_files"]
    whole = sum(path.stat().st_size for path in Path(publisher["release"]).glob("*.safetensors"))
    assert [scope["writes"] for scope in publisher["read"]] == [True]
    assert [scope["writes"] for scope in reader["read"]] == [False]
    for report in (publisher, reader):
        assert report["read"][0]["growth"] < whole / 2 <= report["held"]

    assert publisher["export_code"] == reader["export_code"] == 0
    assert [scope["writes"] for scope in reader["exported"]] == [False]
    assert main(["export", str(source), str(tmp_path / "alone")]) == 0
    assert exported_tensors(source.parent / "exported") == exported_tensors(tmp_path / "alone")


def exported_tensors(directory: Path) -> dict[str, bytes]:
    """Each tensor an export's weight files hold, by name, as bytes beside its dtype and shape."""
    from safetensors.numpy import load_file

    return {name: (str(array.dtype), array.shape, array.tobytes())
            for path in sorted(directory.glob("*.safetensors")) for name, array in load_file(path).items()}

