"""recipes/decision/train.py end to end on Laya's toy checkpoint: a CSV in, a calibrated run out."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dew.decision import Choice, Decide, DecisionTable, Noul

pytestmark = pytest.mark.mesh

RECIPE = Path(__file__).parents[1] / "recipes" / "decision" / "train.py"
TINY = Path(__file__).parent / "fixtures" / "laya" / "tiny"
QWEN = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"


@pytest.fixture(scope="module")
def recipe():
    spec = importlib.util.spec_from_file_location("decision_recipe", RECIPE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_csv_fine_tunes_a_laya_checkpoint_into_a_calibrated_run(recipe, tmp_path):
    table = tmp_path / "tickets.csv"
    tickets = [("charged twice for March", "billing"), ("the site is down", "technical"),
               ("how much is the pro plan", "sales"), ("refund my last invoice", "billing"),
               ("error 500 on login", "technical"), ("can I upgrade", "sales")] * 4
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(tickets)
    # The command line `laya-train --data tickets.csv` is, with this run's sizes.
    config = recipe.DecisionRunConfig.cli([
        "--data.path", str(table), "--data.held-out", "0.5", "--data.question", "team",
        "--data.loading.workers", "0", "--pretrained", str(TINY), "--objective.loss", json.dumps(
            {"class": "dew.decision.scoring:Combined",
             "fields": {"terms": [[1.0, {"class": "log_loss"}], [0.5, {"class": "brier"}]]}}),
        "--trainer.name", "tickets", "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.batch-size", "8", "--trainer.steps", "2", "--trainer.eval-every", "2",
        "--trainer.checkpoint-every", "2", "--trainer.log-every", "1", "--trainer.multi-host", "False",
        "--trainer.compilation-cache-dir", "None", "--model.dtype", "float32"])
    assert isinstance(config.data, DecisionTable)
    recipe.main(config)
    run = tmp_path / "runs" / "tickets"
    records = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    assert next(record["value"]["summary"]["loss"] for record in records
                if record["type"] == "RunRecord") == "log_loss+0.5*brier"
    assert (run / "decide.json").is_file()
    decide = Decide.from_run(str(run))
    answer = decide("I was billed twice", {"team": Choice("Which team?", ["billing", "technical", "sales"])})
    assert answer["team"].choice in ("billing", "technical", "sales")
    assert decide.name == "tickets" and set(decide.calibration.temperatures.types) == {"choice"}


def _module(name: str):
    path = RECIPE.parent / f"{name}.py"
    sys.path.insert(0, str(RECIPE.parent))
    spec = importlib.util.spec_from_file_location(f"decision_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HUB_SOURCES = ("open-jev", "typed-decisions", "gliclass", "banking77", "clinc150", "wanli", "hellaswag",
               "arc", "boolq", "gsm8k")


def _rows(path: Path, count: int, *, joint: bool) -> list[dict]:
    """Labelled requests as JSON lines: a choice with a soft target, and with `joint` a noul beside it."""
    team = {"type": "choice", "instructions": "Which team?",
            "criteria": {"billing": "payments", "technical": "outages", "sales": "pricing"}}
    rows = []
    for index in range(count):
        row = {"state": f"ticket {index}: charged twice for plan {index % 5}",
               "questions": {"team": team}, "targets": {"team": [0.7, 0.2, 0.1]}}
        if joint:
            row["questions"]["urgent"] = {"type": "noul", "instructions": "Is it urgent?"}
            row["answers"] = {"urgent": "true" if index % 2 else "false"}
        rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def _flags(rows: Path, index: Path, runs: Path, held: int) -> list[str]:
    off = [flag for source in HUB_SOURCES for flag in (f"--mixture.{source}.weight", "0")]
    return [*off, "--mixture.rows.path", str(rows), "--mixture.rows.weight", "1",
            "--mixture.rows.held", str(held), "--decontaminate", str(index), "--trainer.name", "run",
            "--trainer.checkpoint-dir", str(runs), "--trainer.batch-size", "8", "--trainer.steps", "2",
            "--trainer.eval-every", "2", "--trainer.checkpoint-every", "2", "--trainer.log-every", "1",
            "--trainer.multi-host", "False", "--trainer.compilation-cache-dir", "None",
            "--data.loading.workers", "0", "--model.dtype", "float32", "--optim.schedule.warmup-steps", "1"]


def _summary(run: Path) -> dict:
    records = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    return next(record["value"]["summary"] for record in records if record["type"] == "RunRecord")


def test_the_encoder_recipe_trains_a_fresh_head_on_a_decontaminated_mixture(tmp_path):
    """The encoder run class end to end on a tiny ModernBERT: its own rows, one
    of which an evaluation item shares, which the index drops before training;
    a soft target; the temperatures fitted on the rows held back."""
    from dew.data.text import HFTokenizer
    from dew.interop.pretrained import PretrainedDecoder
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.base import part
    from dew.registry import from_record

    encoder = _module("encoder")
    contamination = _module("contamination")
    laya = Decide.from_pretrained(TINY, attention_impl="xla")
    backbone = tmp_path / "encoder"
    encoded = from_record(CausalTransformer, laya.model.backbone)
    PretrainedDecoder.from_model(encoded, part(laya.variables, "backbone"),
                                 tokenizer=HFTokenizer(str(TINY / "tokenizer"))).save(backbone)
    rows = _rows(tmp_path / "rows.jsonl", 24, joint=False)
    index = tmp_path / "index.npz"
    contamination.Overlaps.of([rows[0]["state"]]).save(index)
    config = encoder.EncoderRun.cli([*_flags(tmp_path / "rows.jsonl", index, tmp_path / "runs", 12),
                                     "--pretrained", str(backbone), "--revision", "None",
                                     "--max-len", "96", "--head-max-len", "64", "--option-tokens", "16"])
    encoder.main(config)
    summary = _summary(tmp_path / "runs" / "run")
    decontaminated = summary["decontamination"]
    assert decontaminated["rows"]["checked"] + decontaminated["rows (held out)"]["checked"] == 24
    assert sum(entry["dropped"] for entry in decontaminated.values()) == 1
    assert summary["loss"] == "log_loss+0.5*brier+rps"
    decide = Decide.from_run(str(tmp_path / "runs" / "run"))
    assert set(decide.calibration.temperatures.types) == {"choice"}
    team = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
    assert decide("charged twice", {"team": team})["team"].choice in team.options


def test_the_clef_recipe_trains_a_joint_head_under_lora_on_a_frozen_decoder(tmp_path):
    """The Clef run class end to end on the tiny Qwen 3.5: LoRA on its attention,
    Gated DeltaNet and MLP projections, a fresh joint head reading every
    question of a row, recomputed blocks; the run reloads as a task."""
    clef = _module("clef")
    contamination = _module("contamination")
    _rows(tmp_path / "rows.jsonl", 24, joint=True)
    index = tmp_path / "index.npz"
    contamination.Overlaps.of(["nothing shared with these rows at all"]).save(index)
    config = clef.ClefRun.cli([*_flags(tmp_path / "rows.jsonl", index, tmp_path / "runs", 8),
                               "--pretrained", str(QWEN), "--revision", "None", "--lora.rank", "4",
                               "--head.width", "24", "--head.heads", "2", "--head.feedforward", "40",
                               "--head.layers", "1", "--head.routing-layers", "1", "--max-len", "1024"])
    clef.main(config)
    decide = Decide.from_run(str(tmp_path / "runs" / "run"))
    team = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
    answers = decide("charged twice", {"team": team, "urgent": Noul("Is it urgent?")})
    assert answers["team"].choice in team.options and 0 <= answers["urgent"].noul <= 1
