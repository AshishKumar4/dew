"""recipes/decision/train.py end to end on Laya's toy checkpoint: a CSV in, a calibrated run out."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dew.decision import Choice, Decide, DecisionTable

pytestmark = pytest.mark.mesh

RECIPE = Path(__file__).parents[1] / "recipes" / "decision" / "train.py"
TINY = Path(__file__).parent / "fixtures" / "laya" / "tiny"


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
