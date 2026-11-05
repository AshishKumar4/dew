"""A decision ablation workflow as a standing scenario (row 1890): a scaled-down real base read
by its real tokenizer (`real_tokenizers.backbone_reading`, GPT-2's), a task of sixty-four options whose
examples each ask two questions, trained through `dew train` in each cell of a pairwise covering set of
the workflow's decision axes (the head and its layout, context through cross-attention, consistency
groups) and run bounds (an examples budget, a share of one). Each cell holds the conservation contract
(docs/design/testing.md), observed from the run's own counts: its bound counts source examples, whatever
rows its layout makes of each; the examples it reads are counted once each, as trained or as refused under
the reason the run's own layout refuses them for, so none goes uncounted; and its saved task holds each
axis the cell sets and answers over every option its data asks."""

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from real_tokenizers import backbone_reading
from reference_error import assert_fp32_reduction_bound
from run_kinds import scalars

import dew
from dew._cli.main import main
from dew.config import ObjectiveConfig
from dew.data import Files
from dew.data.dataset import Loading
from dew.decision import Choice, Example, decisions
from dew.decision.config import DecisionRunConfig, JointShape, Letters
from dew.decision.data import Group
from dew.decision.head import DecisionHead, JointSchemaHead, LabelReadout
from dew.registry import ModelConfig

INTENTS = tuple(f"intent_{index}" for index in range(64))
"""The task's options, named as BANKING77's and CLINC150's intents are."""

ASKED = Choice("Which intent does the customer's message express?", list(INTENTS))
AGAIN = Choice("Which of these does the message ask for?", list(reversed(INTENTS)))
"""The intent asked again, its options in another order: the consistency group's other member."""

ALIKE = Choice("Which team handles it?", [
    f"please route this message on to the one support team here that usually handles {name}"
    for name in ("billing", "refunds")])
"""A question whose two options read alike once a question row cuts each to `OPTION_TOKENS`: they part only
past their fourteenth token."""

LONG = Choice("Which intent does the customer's message express? " + "Read the message whole. " * 2000,
              list(INTENTS[:2]))
"""A question that alone passes every head's `MAX_LEN`, which no row holds."""

MAX_LEN = {"joint": 3072, "question": 2048, "letters": 2048}
"""Each head's window: a joint row lays both questions' options out in one schema, a question row one."""

OPTION_TOKENS, BATCH = 10, 8
CONTEXT_LAYERS = "0"
SHARE = 0.5


def example(index: int, extra: Mapping[str, Choice] | None = None) -> Example:
    """The `index`th message, its intent asked twice, the two questions one consistency group, beside a
    context of the customer's account, and `extra` questions asked beside them."""
    intent = INTENTS[(7 * index) % len(INTENTS)]
    same = {name: {key: key for key in INTENTS} for name in ("intent", "again")}
    asked = {"intent": ASKED, "again": AGAIN, **(extra or {})}
    return Example(f"message {index}: my card shows a problem with {intent}, please help", asked,
                   {"intent": intent, "again": intent, **{name: question.options[0]
                                                          for name, question in (extra or {}).items()}},
                   context={"plan": ("basic", "pro")[index % 2], "region": f"r{index % 3}"},
                   groups=(Group("intent", same),))


def line(each: Example) -> dict:
    """An example as a line of a decision set (`Example.of` reads it back): its questions in the wire
    format, its groups by name."""
    return {"state": each.state,
            "questions": {name: question.wire() for name, question in each.questions.items()},
            "answers": dict(each.answers), "context": each.context,
            "groups": {group.name: {name: dict(keys) for name, keys in group.members.items()}
                       for group in each.groups}}


PLANTED = {"unfit": {"long": LONG}, "options_alike": {"team": ALIKE}}
"""The questions an example is planted with to be refused, by the reason it is."""


def corpus(first: int, count: int, planted: Mapping[int, str]) -> tuple[list[Example], dict[str, str]]:
    """`count` examples from `first`, those `planted` at their offsets asking their reason's question
    beside their own; and each planted example's state by the reason it was planted with."""
    examples = [example(first + offset, PLANTED[planted[offset]] if offset in planted else None)
                for offset in range(count)]
    return examples, {examples[offset].state: reason for offset, reason in planted.items()}


TRAIN, HELD = corpus(0, 48, {5: "unfit", 17: "options_alike", 29: "unfit", 40: "options_alike"}), \
    corpus(1000, 8, {3: "unfit", 5: "options_alike"})
"""The training and held-out examples and the reasons planted in each: none at a split's end, whose
refusal no later kept row would carry its consumption on (`Rejected.consumed`)."""


@dataclasses.dataclass(frozen=True)
class Cell:
    """One cell of the covering set: the head (`joint`, Laya's `question` head, `letters`), whether the
    backbone reads each example's context, the consistency term's weight, and the bound kind."""

    head: str
    context: bool
    consistency: float
    bound: str

    @property
    def name(self) -> str:
        return f"{self.head}-{'context' if self.context else 'plain'}-{self.consistency}-{self.bound}"

    @property
    def joint(self) -> bool:
        return self.head == "joint"


CELLS = tuple(Cell(head, context=context, consistency=consistency, bound=bound)
              for head, context, consistency, bound in (
                  ("joint", False, 0.0, "examples"), ("joint", True, 0.5, "share"),
                  ("question", False, 0.5, "share"), ("question", True, 0.0, "examples"),
                  ("letters", False, 0.5, "examples"), ("letters", True, 0.0, "share")))
"""Six cells, in which every pair of two axes' values meets (`test_the_cells_cover_every_pair`): the head
by each of the others, and each two of context, consistency and bound."""

HEADS = {"joint": JointShape(width=32, routing_layers=1, layers=1, heads=2, feedforward=64), "question": None,
         "letters": Letters()}
HEAD_KINDS = {"joint": JointSchemaHead, "question": DecisionHead, "letters": LabelReadout}


def test_the_cells_cover_every_pair():
    """Every pair of values of two axes meets in some cell, the head's three each of the others' two."""
    axes = [{getattr(cell, field.name) for cell in CELLS} for field in dataclasses.fields(Cell)]
    assert [len(values) for values in axes] == [3, 2, 2, 2]
    for one, other in [(a, b) for a in range(4) for b in range(a + 1, 4)]:
        met = {(dataclasses.astuple(cell)[one], dataclasses.astuple(cell)[other]) for cell in CELLS}
        assert met == {(a, b) for a in axes[one] for b in axes[other]}


@pytest.fixture(scope="module")
def backbone(tmp_path_factory) -> Path:
    """A tiny Qwen3.5 reading GPT-2's tokenizer (`backbone_reading`)."""
    return backbone_reading("gpt2", tmp_path_factory.mktemp("gpt2") / "backbone")


def run_of(cell: Cell, backbone: Path, root: Path) -> DecisionRunConfig:
    """The cell's run over the scenario's examples, written under `root` as JSONL decision sets, before its
    bound and its command line's trainer."""
    def read(examples: list[Example], name: str):
        path = root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(line(each)) + "\n" for each in examples))
        return (Files(str(path), format="jsonl").pipe(decisions).read("memory").batch(BATCH)
                .loading(Loading(workers=0)))
    return DecisionRunConfig(
        pretrained=str(backbone), model=ModelConfig("causal_transformer", {"dtype": "float32"}),
        head=HEADS[cell.head], max_len=MAX_LEN[cell.head],
        option_tokens=None if cell.joint else OPTION_TOKENS,
        objective=ObjectiveConfig("decision", {"consistency": cell.consistency}), calibrate=False,
        data=read(TRAIN[0], "train"), validation=read(HELD[0], "held"))


def refused_in(cell: Cell, planted: Mapping[str, str]) -> dict[str, int]:
    """The examples of `planted` the cell's layout refuses, by reason: a joint row lays its options out in one
    schema, which no option budget cuts, so it refuses no alike options."""
    reasons = [reason for reason in planted.values() if reason != "options_alike" or not cell.joint]
    return {reason: reasons.count(reason) for reason in set(reasons)}


def kept(cell: Cell) -> int:
    """The training examples the cell's layout keeps, which a decision run fits before its bound reads."""
    return len(TRAIN[0]) - sum(refused_in(cell, TRAIN[1]).values())


def trained(cell: Cell, backbone: Path, root: Path) -> Path:
    """The run directory `dew train` writes for the cell, bounded to every training example its layout keeps
    (`examples`) or to a share of them (`stop`), its context read through cross-attention where the cell reads
    one."""
    record, runs = root / "record", root / "runs"
    run_of(cell, backbone, root).save(str(record))
    given = {"trainer.name": cell.name, "trainer.checkpoint_dir": str(runs), "trainer.log_every": "1",
             "trainer.eval_every": "None", "trainer.checkpoint_every": "None", "trainer.multi_host": "False",
             "trainer.compilation_cache_dir": "None", "trainer.examples": str(kept(cell)),
             **({"trainer.stop": f'{{"fraction": {SHARE}}}'} if cell.bound == "share" else {}),
             **({"context_layers": CONTEXT_LAYERS} if cell.context else {})}
    assert main(["train", str(record / "run.json"),
                 *(part for path, value in given.items() for part in ("--set", f"{path}={value}"))]) == 0
    return runs / cell.name


@pytest.mark.parametrize("cell", CELLS, ids=lambda cell: cell.name)
def test_a_cell_of_the_workflow_holds_the_conservation_contract(cell, backbone, tmp_path):
    run = trained(cell, backbone, tmp_path)
    task = dew.pipeline(str(run))
    logged = scalars(run)
    last = logged[max(logged)]
    consumed = int(last["train/examples_consumed"])
    # Each planted example is refused for its planted reason by the run's own layout, every other one fits:
    # its question rows apart, its joint row whole.
    refusal = {}
    for split in (TRAIN, HELD):
        for each in split[0]:
            rows = task.layout.rows(task.tokenizer, task.specials, each.state, each.questions)
            refused = {task.layout.refusal(row, task.context_length) for row in rows} - {None}
            refusal[each.state] = refused.pop() if refused else None
    planted = {state: reason for split in (TRAIN, HELD) for state, reason in split[1].items()
               if reason != "options_alike" or not cell.joint}
    assert {state: reason for state, reason in refusal.items() if reason} == planted
    # Its question layouts lay an example out as two rows or more, its joint one as one: a bound counting
    # rows would end the run at half the examples it is given or sooner.
    first = example(0)
    assert len(task.layout.rows(task.tokenizer, task.specials, first.state, first.questions)) == (
        1 if cell.joint else 2)
    # The bound counts source examples, the ones the layout keeps, which it fits before the bound reads them:
    # all of them, one pass, or the first step's batch to reach the share's.
    if cell.bound == "examples":
        assert consumed == kept(cell)
    else:
        clock = [int(logged[step]["train/examples_consumed"]) for step in sorted(logged)
                 if "train/examples_consumed" in logged[step]]
        stride = max(later - earlier for earlier, later in zip([0, *clock], clock, strict=False))
        wanted = math.ceil(SHARE * kept(cell))
        assert wanted <= consumed < wanted + stride
    # The examples read are counted once each, as trained or as refused under the reason the layout refuses
    # them for: a bound of every kept example reads one pass, whose counts add to the examples written; a
    # share's counts reach as far as its reading ahead went.
    refused = {key.rsplit("/", 1)[1]: int(value) for key, value in last.items()
               if key.startswith("data/rejected/")}
    delivered = sum(int(value) for key, value in last.items() if key.startswith("data/rows/"))
    assert refused == refused_in(cell, TRAIN[1])
    if cell.bound == "examples":
        assert delivered == consumed
        assert delivered + sum(refused.values()) == len(TRAIN[0])
    else:
        assert consumed <= delivered <= kept(cell)
    # The held-out examples are each scored or refused, every question of each scored one: the run's final
    # validation, which a share pauses before, scores alike options as a server answers them, so it refuses
    # only an example no window holds.
    if cell.bound == "examples":
        [scored] = [int(value) for values in logged.values() for key, value in values.items()
                    if key.endswith("scored_questions")]
        assert scored == sum(len(each.questions) for each in HELD[0] if HELD[1].get(each.state) != "unfit")
    # The saved task holds each axis the cell sets, and answers over every option its data asks.
    assert isinstance(task.model.head, HEAD_KINDS[cell.head])
    assert (task.context_length is not None) == cell.context
    saved = json.loads((run / "run.json").read_text())["fields"]
    assert saved["objective"]["fields"]["consistency"] == cell.consistency
    assert (saved["trainer"]["stop"] is not None) == (cell.bound == "share")
    if cell.head == "letters":
        # Its readout letters every option a question of its data asks: its data's examples state no
        # option count, so it holds every label its tokenizer spells as one token (`Letters.count`).
        assert len(task.model.head.label_ids) >= len(INTENTS)
    [answer] = task([(example(7).state, {"intent": ASKED})],
                    contexts=[example(7).context if cell.context else None])
    choice = answer["intent"]
    assert choice.options == INTENTS
    probabilities = np.asarray(choice.probabilities, np.float64)
    # A softmax's probabilities sum to one within an fp32 sum of their magnitudes.
    assert_fp32_reduction_bound(np.sum(probabilities), 1.0, np.sum(np.abs(probabilities)), len(INTENTS))
