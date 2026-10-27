"""The run classes' command lines: the registries as subcommands, and a run
that rebuilds from what it logged."""

import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import pytest
from diffusion_support import images_of

from dew.config import RunConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.config.jepa import JepaRunConfig
from dew.config.lm import LMRunConfig
from dew.data import Files
from dew.data.dataset import Loader
from dew.diffusion.presets import Flow
from dew.optim import ScheduleSpec
from dew.registry import import_path, record_argument
from dew.sampling import Heun
from dew.training import MeshSpec

pytestmark = pytest.mark.mesh

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse(cls, args):
    return cls.cli(args)


def test_the_flags_pick_a_preset_and_a_solver_and_set_sets_the_chains_settings():
    config = parse(DiffusionRunConfig, [
        "preset:flow", "--preset.shift", "3.0", "--objective.solver", '{"class": "heun"}',
        "--objective.guidance", '{"class": "dew.sampling.guidance:CFG", "fields": {"scale": 2.5}}',
        "--trainer.steps", "10", "--trainer.mesh.fsdp", "2",
        "--model", "simple_dit", "--model.scan_order", "hilbert"])

    assert config.preset == Flow(shift=3.0)
    assert config.objective.fields["solver"]["class"] == import_path(Heun)
    assert config.objective.fields["guidance"]["fields"]["scale"] == 2.5
    assert config.trainer.mesh == MeshSpec(fsdp=2)
    assert config.model.fields["scan_order"] == "hilbert"
    # The data's settings are its chain's calls', set by name (`dew.data.records.override`).
    chained = dataclasses.replace(config, data=images_of("tfds://oxford_flowers102", 32, batch=16))
    set_ = chained.assigned(["data.images.size=64", "data.images.batch=8"])
    assert set_.data.facts()["fields"]["image"] == [64, 64, 3] and set_.data.facts()["batch"] == 8


def test_set_points_a_recipes_chain_at_another_store():
    """`--set data.source.<field>` sets a field of the chain's store, as a recipe's command line points
    its default chain at the caller's own data: a token store's URL, and a TFDS store's path through
    the holdout it is read under; a field the store does not have is refused."""
    from dew.data import presets
    from dew.data.sources import source
    from dew.data.sources.tokens import TokenFiles

    windows = TokenFiles("data/shakespeare-byte").pipe(presets.lm, None, seq_len=8, batch=2, pack=None)
    config = LMRunConfig.cli(["--set", "data.source.url=data/mine", "--set", "data.lm.seq_len=16"],
                             default=LMRunConfig(data=windows))
    assert config.data.source.url == "data/mine" and config.data.facts()["seq_len"] == 16
    train, _ = source("tfds://oxford_flowers102").holdout(records=128)
    flowers = DiffusionRunConfig(data=train.pipe(presets.images, 32, batch=16))
    moved = flowers.assigned(["data.source.path=/data/tfds"])
    assert moved.data.source.source.path == "/data/tfds"
    with pytest.raises(ValueError):
        flowers.assigned(["data.source.colour=red"])


def test_a_chains_settings_are_set_together():
    """Settings of one chain valid only together are one point: an augmented images chain's size and
    staging size move as a pair, read back once."""
    from dew.data import presets
    from dew.data.sources import source

    flowers = source("tfds://oxford_flowers102").pipe(presets.images, 64, staging_size=64, augment=True,
                                                     batch=8)
    moved = DiffusionRunConfig(data=flowers).assigned(["data.images.size=128",
                                                        "data.images.staging_size=128"])
    assert moved.data.facts()["fields"]["image"] == [128, 128, 3]


def test_a_point_sets_a_runs_fields_and_its_chains_settings_together():
    """A run's own fields and its chain's settings are one point, built once: audio conditioning and
    clips that carry audio are valid only together, and a point that sets both reads as both."""
    from dew.data import presets
    from dew.inputs import AudioCondition

    clips = Files("clips", format="files").pipe(presets.clips, 2 / 25, frames=2, fps=25, size=8, batch=8)
    audio = DiffusionRunConfig(text=None, audio=AudioCondition()).to_dict()["audio"]
    moved = DiffusionRunConfig(data=clips).override({"text": None, "audio": audio,
                                                     "data.clips.sample_rate": 16000})
    assert moved.audio is not None and moved.data is not None and "audio" in moved.data.facts()["fields"]


def test_a_python_run_whose_chain_is_unrecorded_trains_from_dew_train(tmp_path):
    """`dew train run.py` trains a run whose chain holds a step only Python has (a lambda) as the file
    built it, and its run.json says why it can't be rebuilt from the record."""
    from dew.cli.main import main

    (tmp_path / "unrecorded_run.py").write_text(
        "import dataclasses\n"
        "import numpy as np\n"
        "from autoencoder_experiment import FEATURES, run as experiment\n"
        "from dew.data import Memory\n"
        "rows = [{'x': np.full((FEATURES,), index / 32, np.float32)} for index in range(32)]\n"
        "run = dataclasses.replace(experiment(), validation=None,\n"
        "                          data=Memory(rows).map(lambda row: row).shuffle().repeat().batch(8))\n")
    assert main(["train", str(tmp_path / "unrecorded_run.py"), "--set", "trainer.steps=1",
                 "--set", f"trainer.checkpoint_dir={tmp_path / 'runs'}", "--set", "trainer.eval_every=None",
                 "--set", "trainer.checkpoint_every=1"]) == 0
    record = json.loads((tmp_path / "runs" / "unrecorded_run" / "run.json").read_text())
    assert set(record["fields"]["data"]) == {"unrecorded"}
    with pytest.raises(ValueError):
        RunConfig.load(str(tmp_path / "runs" / "unrecorded_run"))


def profiled(report_file: Path) -> dict:
    """A profile's report, held to the capture it names: the steps the capture's training annotations
    hold are exactly the two after its warmup of two, the steps its log covers."""
    from dew.training.profiling import traced_steps

    report = json.loads(report_file.read_text())
    assert traced_steps(Path(report["capture"])) == (2, 3) and tuple(report["traced"]) == (2, 3)
    assert (report["steps"], report["warmup"]) == (2, 2)
    assert report["step_ms"] > 0 and report["input_wait_ms"] >= 0 and report["compile_s"] > 0
    assert (report["device_idle_ms"] is None) == (report["attribution"] is not None)
    assert (Path(report["capture"]) / "reports").is_dir()
    return report


def test_dew_profile_trains_a_run_afresh_and_reports_where_its_window_spent_the_time(tmp_path):
    """`dew profile run.py --warmup 2 --steps 2 --out DIR` trains the run afresh in a directory of its
    own, writing no checkpoint, traces the two steps after its warmup, its capture's training
    annotations those steps alone, and writes the report of the window, the one log interval those steps
    make: its step and input-wait times, the step's compile, and the device time XProf attributes to its
    ops or why it attributes none, beside the capture's own reports. A second profile into the same
    directory is refused."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    import autoencoder_experiment

    from dew.checkpoints import Checkpoints
    from dew.cli.main import main

    out = tmp_path / "profile"
    command = ["profile", autoencoder_experiment.__file__, "--warmup", "2", "--steps", "2", "--out", str(out)]
    assert main(command) == 0
    assert profiled(out / "report.json")["directory"] == str(out)
    assert Checkpoints(str(out / "autoencoder_experiment")).latest is None
    with pytest.raises(ValueError):
        main(command)


def test_dew_profile_with_no_output_prepares_its_process_before_naming_its_directory(tmp_path):
    """`dew profile run.py` with no `--out` profiles into profiles/<the run's name>/<the time it starts>,
    the time agreed across the pool once the process is prepared: the run's XLA flags reach XLA, which
    reads them as JAX opens its backend, so an XLA dump the run asks for is written."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    dump = tmp_path / "dump"
    flags = f"--xla_dump_to={dump}"
    (tmp_path / "dumped.py").write_text(
        "import dataclasses\n"
        "from autoencoder_experiment import run as experiment\n"
        "def run():\n"
        "    built = experiment()\n"
        f"    trainer = dataclasses.replace(built.trainer, xla_flags={flags!r})\n"
        "    return dataclasses.replace(built, trainer=trainer)\n")
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")])}
    done = subprocess.run([sys.executable, "-m", "dew.cli.main", "profile", str(tmp_path / "dumped.py"),
                           "--warmup", "2", "--steps", "2"],
                          cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    (report_file,) = (tmp_path / "profiles" / "dumped").glob("*/report.json")
    profiled(report_file)
    assert any(dump.iterdir())


def test_a_configured_run_is_profiled_without_its_ranking_cooldowns_or_validation(tmp_path):
    """A configured run that ranks its checkpoints by its training loss, branches into cooldowns and
    validates a split of its own is profiled without them: its profile writes no checkpoint, best or
    otherwise, and runs no cooldown, which would want the validation the profile drops."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    import autoencoder_experiment

    from dew.checkpoints import Checkpoints
    from dew.training import Cooldowns

    run = autoencoder_experiment.run()
    run = dataclasses.replace(run, trainer=dataclasses.replace(
        run.trainer, name="ranked", best="train/loss", cooldowns=Cooldowns(at=(2,), length=2)))
    report = run.profile(str(tmp_path / "profile"), warmup=2, steps=2)
    assert report is not None and report.traced == (2, 3)
    with Checkpoints(str(tmp_path / "profile" / "ranked")) as checkpoints:
        assert checkpoints.latest is None and checkpoints.best is None


@pytest.mark.parametrize("built", ["in Python", "as a run states them"])
def test_a_run_trainer_fit_trains_directly_says_why_no_record_rebuilds_it(tmp_path, built):
    """A run that `Trainer.fit` trains directly writes a run.json no record rebuilds, naming each part
    of the trainer a run's record states otherwise: its objective, built in Python, always, and an
    optax optimizer and a JAX key where it was given them in place of an OptimConfig and an integer
    seed. `RunConfig.load` refuses it, saying each."""
    import jax
    import optax
    from test_run_budgets import OptimConfig, Pulled, chain, records

    from dew.checkpoints import Checkpoints
    from dew.training import Trainer

    (x,) = records(0)
    optimizer, key = ((optax.adam(0.1), jax.random.key(0)) if built == "in Python"
                      else (OptimConfig(optimizer="adam", learning_rate=0.1), 0))
    Trainer(Pulled(), optimizer, key=key, checkpoints=Checkpoints(str(tmp_path))).fit(
        chain(x), steps=1, checkpoint_every=1)
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["rebuildable"] is False
    assert len(record["reasons"]) == (3 if built == "in Python" else 1)
    with pytest.raises(ValueError) as refused:
        RunConfig.load(str(tmp_path))
    assert all(reason in str(refused.value) for reason in record["reasons"])


def test_an_lm_run_over_images_is_refused():
    with pytest.raises(ValueError):
        LMRunConfig(data=images_of("tfds://oxford_flowers102", 32)).prepare()


def test_a_spec_field_the_dataset_lacks_is_a_command_line_error():
    with pytest.raises(SystemExit):
        parse(RunConfig, ["--data.seq-len", "8"])


@pytest.mark.parametrize("name", ["diffusion", "lm", "jepa"])
def test_a_run_config_round_trips_through_its_json_record(name):
    from dew.data import presets
    from dew.data.sources.tokens import TokenFiles

    cls = {"diffusion": DiffusionRunConfig, "lm": LMRunConfig, "jepa": JepaRunConfig}[name]
    args = {"diffusion": ["preset:karras", "--preset.sigma-data", "0.6",
                          "--objective.unconditional-prob", "0.2"],
            "lm": ["--max-new-tokens", "4", "--objective.ema-decay", "0.9"],
            "jepa": ["--probe-classes", "7", "--objective.momentum", "0.9", "0.99"]}[name]
    data = (TokenFiles("d").pipe(presets.lm, None, seq_len=64, batch=8, pack=None) if name == "lm"
            else images_of("tfds://oxford_flowers102", 32))
    config = dataclasses.replace(parse(cls, [*args, "--trainer.steps", "5"]), data=data)

    record = json.loads(json.dumps(config.record()))

    assert RunConfig.read(record).record() == record
    preset = "lm" if name == "lm" else "images"
    assert record["fields"]["data"]["stages"][-1]["pipe"] == f"dew.data.presets:{preset}"


def test_a_run_over_url_datasets_round_trips_through_its_json_record():
    """A diffusion run may train on any hub table of urls; its record has to
    name the sources, so the run and its solvers rebuild from run.json."""
    from dew.data import presets

    config = DiffusionRunConfig(data=Files("hf://datasets/user/images/train.parquet", format="parquet").pipe(
        presets.images, 64, batch=32))

    record = json.loads(json.dumps(config.to_dict()))

    assert DiffusionRunConfig.from_dict(record).to_dict() == record
    assert record["data"]["source"]["fields"]["url"] == "hf://datasets/user/images/train.parquet"


def test_steps_and_epochs_are_one_choice():
    with pytest.raises(SystemExit):
        parse(RunConfig, ["--trainer.steps", "5", "--trainer.epochs", "1"])
    config = parse(RunConfig, ["--trainer.epochs", "2"])
    data = Loader(train=lambda partition: iter(()), val=None, records=100, batch=10)
    assert config.trainer.total_steps(data) == 20
    with pytest.raises(ValueError):
        config.trainer.total_steps(
            Loader(train=lambda partition: iter(()), val=None, records=None, batch=10)
        )


def test_a_runs_updates_are_its_microsteps_over_its_accumulation():
    """What moves once an update, a JEPA target's average, ramps over the run's updates: an epoch of
    64 records in batches of 8 is 8 steps, and accumulating 4 of them commits 2 updates."""
    from dew.config import TrainerConfig
    from dew.data.dataset import Loader

    loader = Loader(lambda partition: iter(()), None, batch=8, records=64)
    assert TrainerConfig(epochs=1, accumulation=4).total_updates(loader) == 2


def test_the_cli_sets_a_token_budget_the_datas_token_batches_divide():
    config = parse(RunConfig, ["--trainer.update-tokens", "98304", "--trainer.steps", "1"])
    assert config.trainer.update_tokens == 98304
    assert RunConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


@pytest.mark.parametrize("size", [(16, 16), (16, 8)], ids=["square", "wide-grid"])
def test_a_jepa_run_without_probes_trains_and_validates_its_loss(tmp_path, size):
    """A checkpointed JEPA recipe validates without class probes or a representation preview, over
    square images and over images twice as tall as they are wide, whose patch grid is as tall."""
    import numpy as np

    from dew.config import TrainerConfig
    from dew.data import TFDS, Loading, images, presets
    from dew.registry import ModelConfig

    root = REPO_ROOT / "tests" / "fixtures" / "tfds" / "dew_images" / "1.0.0"
    pictures = TFDS("dew_images", path=str(root), split="all")
    train, held = pictures.holdout(records=4)

    def prepared(chain):
        return (chain.pipe(presets.images, 16, batch=None).map(images.Resize(size)).batch(8)
                .read("memory").loading(Loading(workers=0)))

    run = JepaRunConfig(
        model=ModelConfig("jepa_encoder", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                           "num_heads": 2, "mlp_ratio": 2, "dtype": "float32",
                                           "attention_impl": "xla"}),
        predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
        data=prepared(train), validation=prepared(held),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path), steps=1, log_every=1,
                              eval_every=1, checkpoint_every=None))
    trained = run.prepare().train("jepa-loss")
    assert int(trained.updates) == 1
    reports = [json.loads(line)["scalars"] for line in
               (tmp_path / "jepa-loss" / "tracking" / "scalars.jsonl").read_text().splitlines()]
    assert any("val/loss" in report and np.isfinite(report["val/loss"]) for report in reports)
    assert not any("probe" in name for report in reports for name in report)


def test_a_jepa_run_trains_its_lora_factors_and_ramps_their_momentum_over_its_updates(tmp_path):
    """`--lora` adapts the run's encoder: its factors train under `params` over a frozen draw of it.
    The target's momentum ramps over the run's updates, its steps over its accumulation: falling from
    1 to 0 over four steps of two, it holds the first update's target where it starts and averages the
    second's halfway, so a factor that starts at zero ends half the online one."""
    import jax
    import numpy as np

    from dew.batch import FROZEN
    from dew.config import ObjectiveConfig, TrainerConfig
    from dew.data import TFDS, Loading, presets
    from dew.lora import LoRA
    from dew.registry import ModelConfig

    images = TFDS("dew_images", path=str(REPO_ROOT / "tests" / "fixtures" / "tfds" / "dew_images" / "1.0.0"),
                  split="all")
    run = JepaRunConfig(
        model=ModelConfig("jepa_encoder", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                           "num_heads": 2, "mlp_ratio": 2, "dtype": "float32",
                                           "attention_impl": "xla"}),
        objective=ObjectiveConfig("jepa", {"momentum": (1.0, 0.0)}),
        predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
        lora=LoRA(rank=2, modules=("up_proj",)),
        data=images.pipe(presets.images, 16, batch=8).read("memory").loading(Loading(workers=0)),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path), steps=4, accumulation=2, log_every=1,
                              eval_every=None, checkpoint_every=None))
    trained = run.prepare().train("jepa-lora")
    assert int(trained.updates) == 2
    factors = trained.variables["params"]["context_encoder"]
    assert factors and all(path[-1].key in ("lora_A", "lora_B")
                           for path, _ in jax.tree_util.tree_leaves_with_path(factors))
    assert jax.tree.leaves(trained.variables[FROZEN]["context_encoder"])
    for path, online in jax.tree_util.tree_leaves_with_path(factors):
        if path[-1].key == "lora_B":
            target = trained.ema["params"]["context_encoder"]
            for key in path:
                target = target[key.key]
            assert np.any(np.asarray(online))
            np.testing.assert_array_equal(np.asarray(target), 0.5 * np.asarray(online))


def test_param_groups_on_the_command_line_read_their_schedules_as_records():
    """A group's schedule is the record that names it, as the run's record
    writes it, and the config's own rate is one value: a number, or a schedule's record that steps
    per epoch by its `every`."""
    from dew.optim import Cosine, OneCycle, ParamGroup
    groups = [{"name": "delays", "patterns": ["*/delay"], "bounds": [0, 24],
               "schedule": {"class": "cosine",
                            "fields": {"peak_value": 0.1, "warmup_steps": 0, "every": 40}}},
              {"name": "rest", "patterns": ["*"],
               "b1": {"class": "one_cycle",
                      "fields": {"peak_value": 0.85, "init_value": 0.95, "end_value": 0.95}}}]
    rate = {"class": "one_cycle", "fields": {"peak_value": 5e-3, "every": 40}}
    config = parse(RunConfig, ["--optim.param-groups", json.dumps(groups),
                               "--optim.learning-rate", json.dumps(rate)])
    assert config.optim.learning_rate == OneCycle(peak_value=5e-3, every=40)
    assert parse(RunConfig, ["--optim.learning-rate", "3e-4"]).optim.learning_rate == 3e-4
    assert config.optim.param_groups == (
        ParamGroup("delays", ("*/delay",), schedule=Cosine(peak_value=0.1, warmup_steps=0, every=40),
                   bounds=(0.0, 24.0)),
        ParamGroup("rest", ("*",), b1=OneCycle(peak_value=0.85, init_value=0.95, end_value=0.95)))


@dataclasses.dataclass(frozen=True)
class _Schedules(RunConfig):
    """A recipe with a named set of schedules, as sparx's surrogate and delay widths."""

    schedules: Annotated[Mapping[str, ScheduleSpec], record_argument(Mapping[str, ScheduleSpec])] = \
        dataclasses.field(default_factory=dict)


def test_a_mapping_of_class_records_reads_from_one_flag_and_round_trips_its_record():
    """#45: each schedule is the record that names it, read from one JSON
    object on the command line and written back by the run's record."""
    from dew.optim import Exponential, Linear

    given = {"sigma": {"class": "linear", "fields": {"peak_value": 0.5, "warmup_steps": 0, "end_value": 0.1}},
             "delay": {"class": "exponential", "fields": {"init_value": 10.0, "end_value": 1.0, "every": 4}}}
    config = parse(_Schedules, ["--schedules", json.dumps(given)])

    assert config.schedules == {"sigma": Linear(peak_value=0.5, warmup_steps=0, end_value=0.1),
                                "delay": Exponential(init_value=10.0, end_value=1.0, every=4)}
    assert _Schedules.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert parse(_Schedules, []).schedules == {}
