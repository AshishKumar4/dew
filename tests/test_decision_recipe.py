"""A decision run end to end on Laya's toy checkpoint: a CSV in, a calibrated run out."""

import csv
import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from dew.config.data import Split
from dew.data import Memory
from dew.decision import Choice, Decide, Decisions
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
        "--data.source", str(table), "--data.validation", json.dumps({"class": "dew.data.holdout:HoldOut",
            "fields": {"fraction": 0.5, "by": "random", "most": 400}}), "--data.question", "team",
        "--loading.workers", "0", "--pretrained", str(TINY), "--objective.loss", json.dumps(
            {"class": "dew.decision.scoring:Combined",
             "fields": {"terms": [[1.0, {"class": "log_loss"}], [0.5, {"class": "brier"}]]}}),
        "--trainer.name", "tickets", "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.batch-size", "8", "--trainer.steps", "2", "--trainer.eval-every", "2",
        "--trainer.checkpoint-every", "2", "--trainer.log-every", "1", "--trainer.multi-host", "False",
        "--trainer.compilation-cache-dir", "None", "--model.dtype", "float32"])
    assert isinstance(config.data, Decisions)
    assert from_record(ScoringRule, config.objective.fields["loss"]).name == "log_loss+0.5*brier"
    config.run()
    run = tmp_path / "runs" / "tickets"
    assert (run / "decide.json").is_file()
    decide = Decide.from_run(str(run))
    answer = decide("I was billed twice", {"team": Choice("Which team?", ["billing", "technical", "sales"])})
    assert answer["team"].choice in ("billing", "technical", "sales")
    assert decide.name == "tickets" and set(decide.calibration.temperatures.types) == {"choice"}


QWEN = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"


@dataclasses.dataclass(frozen=True)
class _NamedSplits(Memory):
    """Memory rows with native named splits, as a Hub or TFDS source has them."""

    splits: Mapping[str, tuple] = dataclasses.field(default_factory=dict)

    def with_split(self, name: str) -> "_NamedSplits":
        return dataclasses.replace(self, values=self.splits[name])


def test_a_named_validation_split_reads_the_source_s_own_split():
    """`Split("validation")` opens the training source's own named split through its
    `with_split`, as every recipe opens one, and a source with no native splits refuses it."""
    train = tuple({"text": f"train {index}", "label": "ab"[index % 2]} for index in range(4))
    held = tuple({"text": f"held {index}", "label": "a"} for index in range(2))
    named = _NamedSplits(train, splits={"train": train, "validation": held})
    _, evaluated, _ = Decisions(source=named, validation=Split("validation"), question="team").pipelines()
    assert list(evaluated.source.source.records("memory")) == list(held)
    with pytest.raises(ValueError):
        Decisions(source=Memory(train), validation=Split("validation"), question="team").pipelines()


def test_a_checkpoint_loops_at_the_passes_and_over_the_block_asked():
    """--loop-steps sets the passes of Ouro's tiny looped decoder, leaving its
    weights as they were; --loop-layers gives the tiny Qwen 3.5, which does not
    loop, a loop over a block of its own layers; a block with no passes, or
    passes with no block, on a decoder that does not loop is refused."""
    from dew.decision.config import looping
    from dew.interop import Pretrained
    from dew.nn.backbones.decoder_stack import Loop

    ouro = Pretrained.load(Path(__file__).parent / "fixtures" / "hf" / "ouro-tiny", dtype="float32")
    once = looping(ouro, 1, None)
    assert once.model.loop == Loop(1, exit_gate=True) and ouro.model.loop == Loop(3, exit_gate=True)
    assert once.variables is ouro.variables
    qwen = Pretrained.load(QWEN, dtype="float32")
    retrofit = looping(qwen, 2, (1, 2))
    assert retrofit.model.language_model.loop == Loop(2, step_norm=False, layers=(1, 2))
    assert qwen.model.language_model.loop is None
    assert retrofit.variables is qwen.variables
    for steps, layers in ((2, None), (None, (1, 2))):
        with pytest.raises(ValueError):
            looping(qwen, steps, layers)


@pytest.mark.parametrize("fields,checkpoint,message", [
    ({"loop_backprop_steps": 1}, QWEN, "has no loop"),
    ({"loop_remat": True}, QWEN, "has no loop"),
    ({"loop_backprop_steps": 1}, TINY, "Laya checkpoint"),
    ({"loop_remat": True}, TINY, "Laya checkpoint"),
    ({"remat": "full"}, TINY, "Laya checkpoint"),
    ({"remat": "unknown"}, QWEN, "remat names one of"),
])
def test_a_decision_run_refuses_memory_options_without_their_backbone(fields, checkpoint, message):
    with pytest.raises(ValueError):
        DecisionRunConfig(pretrained=str(checkpoint), **fields).prepare()
