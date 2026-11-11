"""A decision run whose implicit loop's pass count grows with its data's phases (row 5480(a)'s
acceptance): the tiny Qwen 3.5, its rows batched by their tokens after a step that drops some, so each
phase reads per share (`dew.data.dataset.StepPhases`), two sources in two phases on the step clock, its
block [1, 3) run once in the first and twice in the second (`DecoderRunConfig.loop_steps`, `Trainer`'s
`phase_passes`)."""

import csv
from pathlib import Path

import jax
import numpy as np
import pytest
from run_kinds import MostlyKept, trained

from dew.config import ModelConfig
from dew.data import Files, Loading, phases
from dew.decision import Decide, decisions
from dew.decision.config import DecisionRunConfig

QWEN35 = Path(__file__).parent / "fixtures" / "hf" / "qwen35-native-tiny"
"""The tiny Qwen 3.5, with its tokenizer, which lays a decision's rows out."""
MAX_LEN = 256
"""A decision row's tokens: room for the tokenizer's chat template, the instructions, the options and a
ticket (at 64 every row was refused as too long, and the read went on without one)."""
TOKENS = 8 * MAX_LEN
"""A batch's token budget: eight rows, which the run's eight row shards divide."""
STEPS = (1, 2)
"""The block's passes a phase."""
PHASE = 2
"""The step the first phase ends at."""
TEAMS = ("billing", "technical", "sales")
"""The team a ticket asks for, its question's options, which a table read by index states."""


def tickets(root: Path, name: str, team: str) -> Files:
    """A source of 32 tickets, each its own text, labelled by its team: its rows are this source's."""
    table = root / f"{name}.csv"
    asked = [(f"{name} charged twice for March", "billing"), (f"{name} the site is down", "technical"),
             (f"{name} how much is the pro plan", "sales"), (f"{name} refund my invoice", team)]
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows((f"{text}, ticket {index}", label) for index in range(8) for text, label in asked)
    return Files(str(table), format="csv")


def asked(rows: Files):
    """`rows` as decision rows, some drawn out, read on past their end, batched by their tokens: a read
    per share."""
    return (rows.pipe(decisions, question="team", options=TEAMS).read("indexed")
            .random_filter_map(MostlyKept()).repeat()
            .batch(tokens=TOKENS).loading(Loading(workers=0)))


def run(root: Path, steps, data) -> DecisionRunConfig:
    del root
    return DecisionRunConfig(pretrained=str(QWEN35),
                             model=ModelConfig("causal_transformer", {"dtype": "float32"}), max_len=MAX_LEN,
                             data=data, validation=None, loop_layers=(1, 3), loop_steps=steps,
                             loop_implicit=True)


def phased(root: Path):
    return phases((asked(tickets(root, "first", "billing")), PHASE),
                  (asked(tickets(root, "second", "sales")), None))


def parameters(state) -> dict:
    return {jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables["params"])}


def theta(state) -> np.ndarray:
    (found,) = [leaf for name, leaf in parameters(state).items() if "loop_theta" in name]
    return found


def test_a_decision_run_whose_passes_grow_by_phase_is_its_phases_runs_and_resumes_as_one(tmp_path):
    """The run's first phase is a one-pass run's over the first source alone, bit for bit (its rows that
    source's, its steps at one pass), its implicit loop's θ untouched
    while every step runs one pass (its gradient zero) and moved from the second phase's first step; stopped
    after step 3, inside its second phase, and resumed, it lands where the run nobody stopped lands; its saved
    run loads with the last phase's two passes. A count list whose length is not the data's phases is
    refused before anything loads."""
    with pytest.raises(ValueError, match="one a phase"):
        run(tmp_path, (1, 2, 3), phased(tmp_path))
    data = phased(tmp_path)
    first, _ = trained(run(tmp_path, STEPS, data), tmp_path, "first", trainer={"steps": PHASE})
    alone, _ = trained(run(tmp_path, 1, asked(tickets(tmp_path, "first", "billing"))), tmp_path, "alone",
                       trainer={"steps": PHASE})
    assert parameters(first).keys() == parameters(alone).keys()
    for name, leaf in parameters(first).items():
        np.testing.assert_array_equal(leaf, parameters(alone)[name], err_msg=name)
    assert not np.any(theta(first)), "θ's gradient is zero while each step runs one pass"

    whole, _ = trained(run(tmp_path, STEPS, data), tmp_path, "whole", trainer={"steps": 4})
    assert np.any(theta(whole)), "θ moves once a step runs two passes"
    trained(run(tmp_path, STEPS, data), tmp_path, "stopped", trainer={"steps": 3, "checkpoint_every": 1})
    resumed, _ = trained(run(tmp_path, STEPS, data), tmp_path, "stopped",
                         trainer={"steps": 4, "checkpoint_every": 1})
    assert int(resumed.step) == 4
    for name, leaf in parameters(whole).items():
        np.testing.assert_array_equal(leaf, parameters(resumed)[name], err_msg=name)
    loaded = Decide.from_run(str(tmp_path / "runs" / "stopped"))
    assert loaded.model.backbone.loop.steps == STEPS[-1]

