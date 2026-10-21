"""A decision run end to end on Laya's toy checkpoint: a CSV in, a calibrated run out."""

import csv
import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from dew.data import Files, Loading, Memory
from dew.decision import Choice, Decide, decisions
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
    # The command line over a table's chain: half its rows held out by the hash of their content.
    train, held = Files(str(table), format="csv").holdout(0.5)

    def asked(rows):
        return rows.pipe(decisions, question="team").batch(8).loading(Loading(workers=0))

    config = DecisionRunConfig.cli([
        "--pretrained", str(TINY), "--objective.loss", json.dumps(
            {"class": "dew.decision.scoring:Combined",
             "fields": {"terms": [[1.0, {"class": "log_loss"}], [0.5, {"class": "brier"}]]}}),
        "--trainer.name", "tickets", "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.steps", "2", "--trainer.eval-every", "2",
        "--trainer.checkpoint-every", "2", "--trainer.log-every", "1", "--trainer.multi-host", "False",
        "--trainer.compilation-cache-dir", "None", "--model.dtype", "float32"],
        default=DecisionRunConfig(data=asked(train), validation=asked(held)))
    assert config.data.facts()["batch"] == 8
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
    """A holdout of the store's own named split (`holdout(split="validation")`) opens it through the
    source's `with_split`, and the decisions over it read that split's rows; a source with no native
    splits refuses it."""
    from dew.data.compile import iterate

    train = tuple({"text": f"train {index}", "label": "ab"[index % 2]} for index in range(4))
    held = tuple({"text": f"held {index}", "label": "a"} for index in range(2))
    named = _NamedSplits(train, splits={"train": train, "validation": held})
    _, evaluated = named.holdout(split="validation")
    rows = iterate(evaluated.pipe(decisions, options=("a", "b"), question="team"), read="memory")
    assert [row["example"].state for row in rows] == [row["text"] for row in held]
    with pytest.raises(ValueError):
        Memory(train).holdout(split="validation")


def test_a_tables_own_integer_labels_name_their_options_by_value():
    """A flat table whose options are its label column's own values: every
    label names the option of its value, the integers 2 and 10 included,
    rather than the option at that position. The options sort as text."""
    from dew.data.compile import held as held_reader
    from dew.data.dataset import DataPartition
    from dew.data.text import ByteTokenizer
    from dew.decision import Specials
    from dew.decision.layout import StateFirstLayout
    from dew.decision.objective import DecisionObjective
    from dew.nn.backbones.causal_transformer import CausalTransformer

    rows = [{"text": f"row {index}", "label": 10 if index < 3 else 2} for index in range(8)]
    layout = StateFirstLayout(max_len=64, head_max_len=32, option_tokens=4)
    objective = DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                          max_seq_len=layout.max_len),
        tokenizer=ByteTokenizer(), specials=Specials(begin=None, separator=10, marker=0, marker_text="\x00",
                                                     pad=255),
        layout=layout, shuffle_options=False)
    table = Memory(rows).pipe(decisions).read("memory")
    (batch,) = list(held_reader(table.batch(8), objective=objective.data())(DataPartition()))
    scored = batch["scored"].ravel()
    labels = [int(label) for label, real in zip(batch["labels"].ravel(), scored, strict=True) if real]
    # Options ("10", "2"): the three tens are option 0 and the five twos option 1.
    assert sorted(labels) == [0, 0, 0, 1, 1, 1, 1, 1]


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
