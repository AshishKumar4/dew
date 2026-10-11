"""A batch's real options compacted (`DecisionInputs.option_slots`): Clef's joint head computes those alone.

A batch pads every row to the most questions times the most options, and a
row's real options are a few of them. The joint head computes each option on
its own but for its weight in its question's summary, so over the compacted
options it scores every real option as it does over the grid: to float32
rounding against the grid, bit for bit at any number of option slots.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.data import DataPartition
from dew.data.text import ByteTokenizer, HFTokenizer
from dew.decision import (
    Choice,
    DecisionInputs,
    DecisionObjective,
    Example,
    JointLayout,
    JointSchemaHead,
    Question,
    Specials,
)
from dew.decision.clef import ClefHead
from dew.decision.layout import compacted
from dew.decision.task import laid_out
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import VALID_ROWS, Step

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "clef" / "tiny"
BACKBONE = FIXTURES / "hf" / "qwen38-dense-tiny"
CASES = json.loads((FIXTURES / "laya" / "cases.json").read_text())


def clef_batch():
    """Clef's tiny head, its output table, the five cases laid out in one
    batch and their states padded to it."""
    tokenizer = HFTokenizer(str(BACKBONE), local_files_only=True)
    rows = []
    for request in CASES.values():
        questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
        rows.extend(JointLayout().rows(tokenizer, Specials.of(tokenizer), request["state"], questions))
    inputs = DecisionInputs.collate(rows, 0)
    with np.load(TINY / "states.npz") as states:
        length = inputs.tokens.shape[1]
        padded = np.stack([np.pad(states[case], ((0, length - len(states[case])), (0, 0))) for case in CASES])
    table = load_file(BACKBONE / "model.safetensors")["lm_head.weight"]
    return ClefHead.load(TINY, attention_impl="xla"), table, inputs, padded


def with_slots(inputs: DecisionInputs, slots: int | None) -> DecisionInputs:
    """`inputs` with each row's real options compacted to `slots`, or none."""
    if slots is None:
        return inputs.replace(option_slots=None, slotted=None)
    indexed = [compacted(row, slots) for row in np.asarray(inputs.options)]
    return inputs.replace(option_slots=jnp.asarray(np.stack([index for index, _ in indexed])),
                          slotted=jnp.asarray(np.stack([real for _, real in indexed])))


def test_a_rows_real_options_compact_in_order_and_a_row_over_the_cap_is_refused():
    options = np.zeros((3, 4), bool)
    options[0, :2] = options[2, :3] = True
    indices, real = compacted(options, 6)
    assert indices.tolist() == [0, 1, 8, 9, 10, 0] and real.tolist() == [True] * 5 + [False]
    with pytest.raises(ValueError, match="a row of 5 options outgrows its 4 option slots"):
        compacted(options, 4)


def test_clefs_head_scores_compacted_options_as_it_scores_the_grid():
    """Clef's trained tiny head over the five cases: every real option's
    logit as over the grid to float32 rounding, and bit for bit at two
    numbers of option slots; a slot with no real option scores zero, which
    the model's logits mask (`DecisionModel.logits`). The head's parameters
    do not depend on the number of slots."""
    loaded, table, inputs, states = clef_batch()
    most = int(np.max(np.sum(np.asarray(inputs.options).reshape(len(states), -1), axis=1)))
    assert most < inputs.options.shape[1] * inputs.options.shape[2]

    def scores(slots):
        laid = with_slots(inputs, slots)
        return np.asarray(loaded.head.apply({"params": loaded.params}, states, laid, table))

    grid, fewest, more = scores(None), scores(most), scores(most + 5)
    real = np.asarray(inputs.options)
    np.testing.assert_allclose(fewest[real], grid[real], rtol=0, atol=1e-5 * np.abs(grid[real]).max())
    np.testing.assert_array_equal(fewest[real], more[real])
    assert not np.any(fewest[~real]) and not np.any(more[~real])


def ragged() -> list[Example]:
    """Rows of the Qwen3.5-0.8B decision mixture's shapes: one question of a hundred options
    (LEDGAR), seventeen of three (ContractNLI) and one of twenty-four (CLINC150)."""
    clauses = Choice("Which clause?", [f"c{index}" for index in range(100)])
    hypotheses = {f"h{index}": Choice(f"Hypothesis {index}?", ["yes", "no", "unsaid"]) for index in range(17)}
    intents = Choice("Which intent?", [f"i{index}" for index in range(24)])
    return [Example("a clause on fees", {"clause": clauses}, {"clause": "c7"}),
            Example("a contract", hypotheses, dict.fromkeys(hypotheses, "unsaid")),
            Example("book a flight", {"intent": intents}, {"intent": "i3"})]


def test_a_ragged_batch_trains_on_compacted_options_as_on_the_grid():
    """A batch of 100, 17 x 3 and 24 real options in a 17 x 100 grid: the
    loss and every parameter's gradient, backbone's and head's, as over the
    grid to float32 rounding, and the loss reads only the real options."""
    tokenizer = ByteTokenizer()
    specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    objective = DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                          max_seq_len=16384, attention_impl="xla"),
        tokenizer=tokenizer, specials=specials, layout=JointLayout(max_len=16384),
        head=JointSchemaHead(hidden_size=32, width=24, routing_layers=2, layers=1, heads=2, feedforward=40,
                             attention_impl="xla"), shuffle_options=False)
    examples = ragged()
    data = objective.dataset(examples, batch=len(examples), validation=examples)
    (batch,) = list(data.val(DataPartition()))
    batch = {name: value for name, value in batch.items() if name != VALID_ROWS}
    options = np.asarray(batch["options"])
    assert options.shape[1:] == (17, 100)
    assert sorted(options.reshape(len(examples), -1).sum(axis=1).tolist()) == [24, 51, 100]
    indexed = [compacted(row, 128) for row in options]
    slotted = {**batch, "option_slots": np.stack([index for index, _ in indexed]),
               "slotted": np.stack([real for _, real in indexed])}
    assert laid_out(slotted).option_slots.shape == (3, 128)
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(0), None)

    def loss(held, rows):
        return objective.scalar_loss(held, rows, step)[0]

    (grid, grid_gradient), (fewer, fewer_gradient) = (jax.value_and_grad(loss)(variables, rows)
                                                      for rows in (batch, slotted))
    assert float(fewer) == pytest.approx(float(grid), rel=1e-6)
    scale = max(float(jnp.abs(leaf).max()) for leaf in jax.tree.leaves(grid_gradient))

    def close(path, got, want):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0, atol=1e-5 * scale,
                                   err_msg=jax.tree_util.keystr(path))

    jax.tree_util.tree_map_with_path(close, fewer_gradient, grid_gradient)

    # Whatever the head gives a slot with no real option, the loss and its
    # gradients are the same bits: the model's logits mask those slots.
    class Noisy(JointSchemaHead):
        def __call__(self, states, inputs, table=None, train=False):
            scores = super().__call__(states, inputs, table, train)
            return scores + jnp.where(inputs.options, 0.0, 1e6)

    head = objective.model.head
    objective.model = dataclasses.replace(objective.model, head=Noisy(**{
        field.name: getattr(head, field.name) for field in dataclasses.fields(head)
        if field.init and field.name not in ("parent", "name")}))
    noisy_loss, noisy_gradient = jax.value_and_grad(loss)(variables, slotted)
    assert float(noisy_loss) == float(fewer)
    jax.tree.map(np.testing.assert_array_equal, noisy_gradient, fewer_gradient)
