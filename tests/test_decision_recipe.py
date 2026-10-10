"""A decision run end to end on Laya's toy checkpoint: a CSV in, a calibrated run out."""

import csv
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from dew.data.text import HFTokenizer
from dew.decision import Choice, Decide, DecisionTable, Example, Noul
from dew.decision.config import DecisionRunConfig
from dew.decision.scoring import ScoringRule
from dew.registry import from_record

pytestmark = pytest.mark.mesh

TINY = Path(__file__).parent / "fixtures" / "laya" / "tiny"


def test_a_csv_fine_tunes_a_laya_checkpoint_into_a_calibrated_run(tmp_path):
    table = tmp_path / "tickets.csv"
    tickets = [("charged twice for March", "billing"), ("the site is down", "technical"),
               ("how much is the pro plan", "sales"), ("refund my last invoice", "billing"),
               ("error 500 on login", "technical"), ("can I upgrade", "sales")] * 4
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(tickets)
    # The command line `laya-train --data tickets.csv` is, with this run's sizes.
    config = DecisionRunConfig.cli([
        "--data.path", str(table), "--data.held-out", "0.5", "--data.question", "team",
        "--data.loading.workers", "0", "--pretrained", str(TINY), "--objective.loss", json.dumps(
            {"class": "dew.decision.scoring:Combined",
             "fields": {"terms": [[1.0, {"class": "log_loss"}], [0.5, {"class": "brier"}]]}}),
        "--trainer.name", "tickets", "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.batch-size", "8", "--trainer.steps", "2", "--trainer.eval-every", "2",
        "--trainer.checkpoint-every", "2", "--trainer.log-every", "1", "--trainer.multi-host", "False",
        "--trainer.compilation-cache-dir", "None", "--model.dtype", "float32"])
    assert isinstance(config.data, DecisionTable)
    assert from_record(ScoringRule, config.objective.fields["loss"]).name == "log_loss+0.5*brier"
    config.run()
    run = tmp_path / "runs" / "tickets"
    assert (run / "decide.json").is_file()
    decide = Decide.from_run(str(run))
    answer = decide("I was billed twice", {"team": Choice("Which team?", ["billing", "technical", "sales"])})
    assert answer["team"].choice in ("billing", "technical", "sales")
    assert decide.name == "tickets" and set(decide.calibration.temperatures.types) == {"choice"}


RECIPES = Path(__file__).parents[1] / "recipes" / "decision"
QWEN = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"


def _module(name: str):
    """A module of recipes/decision, which imports its siblings by name."""
    sys.path.insert(0, str(RECIPES))
    spec = importlib.util.spec_from_file_location(f"decision_{name}", RECIPES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _requests(count: int, *, joint: bool) -> list[dict]:
    """Labelled requests as rows: a choice with a soft target, and with `joint` a noul beside it."""
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
    return rows


def _mixture(tmp_path: Path, rows: list[dict], evaluated: list[str], held: int) -> Path:
    """`rows` written as the only set of a mixture by sources.py, decontaminated against `evaluated`."""
    sources, contamination = _module("sources"), _module("contamination")
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    default = sources.Mixture()
    off = {entry.name: dataclasses.replace(getattr(default, entry.name), weight=0.0)
           for entry in dataclasses.fields(default)}
    mine = sources.Rows(weight=1.0, held=held, path=str(tmp_path / "rows.jsonl"))
    mixture = sources.Mixture(**{**off, "rows": mine})
    out = tmp_path / "mixture"
    items = [contamination.Evaluated((text,)) for text in evaluated]
    sources.write(mixture, out, contamination.Overlaps.of(items))
    return out


def _flags(mixture: Path, runs: Path) -> list[str]:
    return ["--mixture.root", str(mixture), "--trainer.name", "run", "--trainer.checkpoint-dir", str(runs),
            "--trainer.batch-size", "8", "--trainer.steps", "2", "--trainer.eval-every", "2",
            "--trainer.checkpoint-every", "2", "--trainer.log-every", "1", "--trainer.multi-host", "False",
            "--trainer.compilation-cache-dir", "None", "--data.loading.workers", "0",
            "--model.dtype", "float32", "--optim.schedule.warmup-steps", "1", "--revision", "None"]


def _summary(run: Path) -> dict:
    records = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    return next(record["value"]["summary"] for record in records if record["type"] == "RunRecord")


def test_the_encoder_recipe_trains_a_fresh_head_on_a_decontaminated_mixture(tmp_path):
    """The encoder recipe end to end on a tiny ModernBERT: sources.py writes a
    set of soft-labelled rows, one of which an evaluation item shares and is
    dropped; the run trains a fresh head on the rest and fits its temperature
    on the rows held back."""
    from dew.interop.pretrained import PretrainedDecoder
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.base import part

    rows = _requests(24, joint=False)
    mixture = _mixture(tmp_path, rows, [rows[0]["state"]], held=12)
    made = json.loads((mixture / "mixture.json").read_text())
    assert sum(entry["dropped"] for entry in made["decontamination"].values()) == 1
    assert made["rows"]["rows"]["train"] + made["rows"]["rows"]["held"] == 23
    laya = Decide.from_pretrained(TINY, attention_impl="xla")
    backbone = tmp_path / "encoder"
    encoded = from_record(CausalTransformer, laya.model.backbone)
    PretrainedDecoder.from_model(encoded, part(laya.variables, "backbone"),
                                 tokenizer=HFTokenizer(str(TINY / "tokenizer"))).save(backbone)
    encoder = _module("encoder")
    config = DecisionRunConfig.cli([*_flags(mixture, tmp_path / "runs"), "--pretrained", str(backbone),
                                    "--max-len", "96", "--head-max-len", "64", "--option-tokens", "16"],
                                   default=encoder.run_config())
    assert from_record(ScoringRule, config.objective.fields["loss"]).name == "log_loss+0.5*brier+rps"
    config.run()
    summary = _summary(tmp_path / "runs" / "run")
    assert summary["mixture"] == {"rows": 1.0} and summary["made"]["rows"] == made["rows"]
    decide = Decide.from_run(str(tmp_path / "runs" / "run"))
    assert set(decide.calibration.temperatures.types) == {"choice"}
    team = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
    assert decide("charged twice", {"team": team})["team"].choice in team.options
    hard = [{**rows[1], "targets": {}, "answers": {"team": "technical"}},
            {**rows[2], "targets": {}, "answers": {"team": "billing"}}]
    (mixture / "hard.held.jsonl").write_text("".join(json.dumps(row) + "\n" for row in hard))
    made["rows"]["hard"] = {"train": 0, "held": len(hard)}
    (mixture / "mixture.json").write_text(json.dumps(made))
    benchmark = _module("benchmark")
    scored = benchmark.held(tmp_path / "runs" / "run", mixture)
    assert set(scored) == {"sources", "macro"}
    assert scored["sources"]["rows"]["rows"] == made["rows"]["rows"]["held"]
    assert scored["sources"]["hard"]["rows"] == 2
    for source, metrics in scored["sources"].items():
        assert set(metrics) == {"rows", "questions", "log_loss", "infinite", "accuracy"}
        expected = []
        for line in (mixture / f"{source}.held.jsonl").read_text().splitlines():
            example = Example.of(json.loads(line))
            probabilities = decide(example.state, example.questions)["team"].probabilities
            target = example.distribution("team")
            expected.append(-sum(t * math.log(p) for t, p in zip(target, probabilities, strict=True)))
        assert metrics["log_loss"] == pytest.approx(sum(expected) / len(expected))
        assert math.isfinite(metrics["log_loss"]) and metrics["infinite"] == 0
        assert 0 <= metrics["accuracy"] <= 1
    for metric in ("log_loss", "accuracy"):
        assert scored["macro"][metric] == pytest.approx(
            sum(value[metric] for value in scored["sources"].values()) / 2)


@pytest.mark.parametrize("param_dtype", ["float32", "bfloat16"])
def test_the_clef_recipe_trains_a_joint_head_under_lora_on_a_frozen_decoder(tmp_path, param_dtype):
    """The Clef recipe end to end on the tiny Qwen 3.5: LoRA on its attention,
    Gated DeltaNet and MLP projections, a fresh joint head reading every
    question of a row; an example whose questions alone pass the row's 1,024
    tokens is dropped and counted; the run reloads as a task."""
    from flax.traverse_util import flatten_dict

    from dew.objectives.base import FROZEN

    rows = _requests(24, joint=True)
    rows.append({**rows[0], "questions": {**rows[0]["questions"], "urgent": {
        "type": "noul", "instructions": "Is it urgent? " + "Consider every detail. " * 300}}})
    mixture = _mixture(tmp_path, rows, ["nothing these rows share at all"], held=8)
    clef = _module("clef")
    storage = [] if param_dtype == "float32" else ["--param-dtype", param_dtype]
    config = DecisionRunConfig.cli([*_flags(mixture, tmp_path / "runs"), "--pretrained", str(QWEN),
                                    "--lora.rank", "4", "--head.width", "24", "--head.heads", "2",
                                    "--head.feedforward", "40", "--head.layers", "1",
                                    "--head.routing-layers", "1", "--max-len", "1024", *storage],
                                   default=clef.run_config())
    config.run()
    assert sum(_summary(tmp_path / "runs" / "run")["unfit"].values()) == 1
    decide = Decide.from_run(str(tmp_path / "runs" / "run"))
    frozen = flatten_dict(decide.variables[FROZEN])
    kernels = [value for path, value in frozen.items() if path[-1] == "kernel"]
    assert kernels and all(str(value.dtype) == param_dtype for value in kernels)
    moving = flatten_dict(decide.variables["params"])
    factors = [value for path, value in moving.items() if path[-1] in ("lora_A", "lora_B")]
    head = [value for path, value in moving.items() if path[0] == "head"]
    assert factors and head and all(str(value.dtype) == "float32" for value in [*factors, *head])
    assert decide.weights.step == 2
    journal = tmp_path / "runs" / "run" / "tracking" / "scalars.jsonl"
    scalars = [json.loads(line)["scalars"] for line in journal.read_text().splitlines()]
    losses = [entry["train/loss"] for entry in scalars if "train/loss" in entry]
    assert len(losses) == 2 and all(math.isfinite(float(loss)) for loss in losses)
    team = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
    answers = decide("charged twice", {"team": team, "urgent": Noul("Is it urgent?")})
    assert answers["team"].choice in team.options and 0 <= answers["urgent"].noul <= 1
    assert math.isfinite(-sum(math.log(value) for value in answers["team"].probabilities))
