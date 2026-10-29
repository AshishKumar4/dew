"""The rest of the common run layer's options, each set on `dew train` for every kind of run and
observed where it acts, in the steps, checkpoints, scalars and records the run writes (row 1901):
a pause, a split, a ranking, a cooldown, the final save, the run's length in epochs, examples and
tokens, a batch ramp, a lowering for a GPU, a step compiled ahead and one handed to the run, and the
options a kind's data or mesh refuses. The
options each kind takes from Python and the rest of the command line are tests/test_run_options.py's.
Nothing here reads a config's fields back."""

import json
from pathlib import Path

import pytest
from test_run_options import KINDS, scalars

from dew.checkpoints import Checkpoints
from dew.cli.main import main
from dew.training._lowering import Lowered

pytestmark = pytest.mark.mesh(devices=8)

COUNTED = {"lm", "decision"}
"""The kinds whose objectives count tokens (`Objective.counts_tokens`), which a token budget reads."""

COUNTED_RECORDS = {"lm", "diffusion", "jepa"}
"""The kinds whose training data states its record count, which an epoch reads; a decision run's
questions stream through a filter, whose count no read states before it ends."""

EXAMPLES = {"lm": 2, "decision": 16, "diffusion": 16, "jepa": 16}
"""Examples a run of each kind reads in a few steps: an LM's example is a document, many windows long;
the other kinds' are a batch's rows."""

GLOBAL_ORDER = {"lm", "diffusion", "jepa"}
"""The kinds whose training data is compiled to one global record order, which a batch ramp cuts
its growing batches out of; a decision run's shares cut their own."""

MESH = {"tensor": 2}
"""Eight devices, two a tensor and four row shards of a batch of eight (`MeshSpec`'s data axis takes
the rest): two rows each, which a split in two and a ramp's first batch of four still divide."""


def trained(kind: str, root: Path, name: str, **settings: object) -> Path:
    """`dew train` over `kind`'s run.json under `root`, its options `settings` (`--set path=value`, a
    mapping as JSON), at most as many steps as it is given: the run's directory."""
    record, data = root / f"record-{name}", root / name
    data.mkdir()
    KINDS[kind](data, drawn=False).save(str(record))
    given = {"trainer.name": name, "trainer.checkpoint_dir": str(root / "runs"), "trainer.log_every": "1",
             "trainer.eval_every": "None", "trainer.checkpoint_every": "None", "trainer.mesh": MESH,
             "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None", "val_metrics": "[]",
             **settings}
    flags = [flag for path, value in given.items()
             for flag in ("--set", f"{path}={json.dumps(value) if isinstance(value, dict) else value}")]
    assert main(["train", str(record / "run.json"), *flags]) == 0
    return root / "runs" / name


def saved(run: Path) -> list[int]:
    """The steps the run saved a checkpoint at."""
    return sorted(int(path.name) for path in run.iterdir() if path.name.isdigit())


def recorded(run: Path, kind: str) -> list[dict]:
    """The values of the records of `kind` the run wrote (`tracking/records.jsonl`)."""
    rows = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    return [row["value"] for row in rows if row["type"] == kind]


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_run_pauses_splits_and_ranks_as_set(kind, tmp_path):
    """Four steps paused halfway (`stop`), each batch stepped as two microbatches (`split`), a
    checkpoint each step, ranked as the run asks, the highest training loss best (`best`): the run
    saves steps one and two and stops there, its compiled step's plan splits in two, and its best
    checkpoint is the saved step whose logged training loss is the higher, as no default ranking
    picks."""
    best = {"metric": "train/loss", "mode": "max"}
    run = trained(kind, tmp_path, "paused", **{"trainer.steps": 4, "trainer.stop": {"fraction": 0.5},
                                                 "trainer.split": 2, "trainer.checkpoint_every": 1,
                                                 "trainer.keep": 4, "trainer.best": best})
    assert saved(run) == [1, 2]
    assert {plan["plan"]["split"] for plan in recorded(run, "StepCompiled")} == {2}
    losses = {step: row["train/loss"] for step, row in scalars(run).items() if step in (1, 2)}
    assert Checkpoints(str(run)).resolve("best") == max(losses, key=losses.__getitem__)


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_run_cools_down_and_skips_its_final_save_as_set(kind, tmp_path):
    """Four steps with a cooldown branched at the second (`cooldowns`) and no checkpoint, the final
    save off (`checkpoint_final`): the second step's scalars hold the cooled branch's scores, and the
    run saves nothing."""
    run = trained(kind, tmp_path, "cooled", **{"trainer.steps": 4, "trainer.checkpoint_final": "False",
                                                 "trainer.cooldowns": {"at": [2], "length": 1}})
    assert any(name.startswith("cooldown_") for name in scalars(run)[2])
    assert saved(run) == []


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_run_is_as_long_as_its_epochs_and_examples(kind, tmp_path):
    """A run of two epochs ends at twice the step one epoch ends at (`epochs`), where the kind's data
    states its record count, and is refused where it does not; one of twice the examples ends at twice
    the step (`examples`)."""
    if kind in COUNTED_RECORDS:
        once = saved(trained(kind, tmp_path, "once", **{"trainer.epochs": 1}))[-1]
        twice = saved(trained(kind, tmp_path, "twice", **{"trainer.epochs": 2}))[-1]
        assert twice == 2 * once
    else:
        with pytest.raises(ValueError):
            trained(kind, tmp_path, "once", **{"trainer.epochs": 1})
    few = saved(trained(kind, tmp_path, "few", **{"trainer.examples": EXAMPLES[kind]}))[-1]
    assert saved(trained(kind, tmp_path, "more", **{"trainer.examples": 2 * EXAMPLES[kind]}))[-1] == 2 * few


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_token_budget_ends_its_run_or_is_refused(kind, tmp_path):
    """A budget of one token ends a run of a kind that counts tokens at its first step (`tokens`); a
    kind whose objective counts none refuses it."""
    if kind in COUNTED:
        assert saved(trained(kind, tmp_path, "budget", **{"trainer.tokens": 1})) == [1]
    else:
        with pytest.raises(ValueError):
            trained(kind, tmp_path, "budget", **{"trainer.tokens": 1})


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_batch_ramp_compiles_a_step_for_each_batch_it_reads(kind, tmp_path):
    """A batch ramp from four rows to the batch's eight over the first eight examples (`batch_ramp`)
    reads batches of two sizes, and compiles a step for each; a kind whose shares cut their own batches
    refuses it."""
    ramp = {"start": 4, "increment": 4, "samples": 8}
    if kind not in GLOBAL_ORDER:
        with pytest.raises(TypeError):
            trained(kind, tmp_path, "ramped", **{"trainer.steps": 4, "trainer.batch_ramp": ramp})
        return
    run = trained(kind, tmp_path, "ramped", **{"trainer.steps": 4, "trainer.batch_ramp": ramp})
    assert len(recorded(run, "StepCompiled")) == 2


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_run_lowered_for_a_gpu_lowers_its_step_and_trains_nothing(kind, tmp_path):
    """Set to lower for an A100 (`lower_for`), a kind's run lowers its first step for that GPU on this
    CPU and ends with status 0, saving and stepping nothing."""
    with pytest.raises(Lowered) as lowered:
        trained(kind, tmp_path, "lowered", **{"trainer.steps": 2, "trainer.lower_for": "sm80"})
    assert lowered.value.code == 0 and lowered.value.generation == "sm80"
    assert not any(path.name.isdigit() for path in (tmp_path / "runs" / "lowered").iterdir())


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_run_handed_a_step_compiled_ahead_for_other_devices_refuses_it(kind, tmp_path):
    """A kind's run compiled ahead for eight A100s it lacks (`compile_ahead`) writes its first step and
    ends with status 0; the same run handed that step (`ahead`) on this host's CPUs reaches the step's
    own check, which refuses it before the run trains, saving and stepping nothing (train 286's review).
    Which refusal is the program's is tests/test_compile_ahead.py's to show, on the GPU a step runs on."""
    from dew.training.ahead import CompiledAhead

    step = tmp_path / "step.dew-aot"
    target = {"generation": "sm80", "devices": 8, "path": str(step)}
    with pytest.raises(CompiledAhead) as ended:
        trained(kind, tmp_path, "ahead", **{"trainer.steps": 2, "trainer.compile_ahead": target})
    assert ended.value.code == 0 and step.exists()
    with pytest.raises(ValueError):
        trained(kind, tmp_path, "handed", **{"trainer.steps": 2, "trainer.ahead": str(step)})
    assert not any(path.name.isdigit() for path in (tmp_path / "runs" / "handed").iterdir())


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("option", ["microbatches", "update_tokens"])
def test_an_option_a_kinds_mesh_or_data_cannot_take_is_refused(kind, option, tmp_path):
    """Microbatches fed through a stage axis the mesh does not have (`microbatches`), and an update's
    token count over batches of rows (`update_tokens`), are refused before the run trains."""
    value = {"microbatches": 2, "update_tokens": 64}[option]
    with pytest.raises(ValueError):
        trained(kind, tmp_path, option, **{"trainer.steps": 2, f"trainer.{option}": value})


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_run_is_tracked_on_wandb_offline_and_publishes_only_there(kind, tmp_path, monkeypatch):
    """A run tracked on W&B offline (`wandb`) writes its W&B run where W&B is told to keep it; publishing
    its final checkpoint (`publish`) without W&B to publish to is refused before the run trains."""
    import wandb

    # W&B settles its directory once a process; each case starts it afresh under its own.
    wandb.teardown()
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    trained(kind, tmp_path, "tracked", **{"trainer.steps": 2,
                                            "trainer.wandb": {"project": "dew-tests", "offline": True}})
    assert list((tmp_path / "wandb").glob("offline-run-*"))
    with pytest.raises(ValueError):
        trained(kind, tmp_path, "published", **{"trainer.steps": 2, "trainer.publish": "True"})
