"""A batch's real options compacted (`DecisionInputs.option_slots`): Clef's joint head computes those alone.

A batch pads every row to the most questions times the most options, and a
row's real options are a few of them. The joint head computes each option on
its own but for its weight in its question's summary, so over the compacted
options it scores every real option as it does over the grid, to float32
rounding, whatever the number of option slots: XLA's products sum in an
order that may follow how many rows they multiply.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, distance
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
    logit as over the grid at two numbers of option slots, to float32
    rounding (on armada's CPUs they moved by up to 6.6e-7 in logits of about 0.6
    at two numbers of slots); a slot with no real option scores zero,
    which the model's logits mask (`DecisionModel.logits`). The head's
    parameters do not depend on the number of slots, so a checkpoint trained
    at one serves at another."""
    loaded, table, inputs, states = clef_batch()
    most = int(np.max(np.sum(np.asarray(inputs.options).reshape(len(states), -1), axis=1)))
    assert most < inputs.options.shape[1] * inputs.options.shape[2]

    def scores(slots):
        laid = with_slots(inputs, slots)
        return np.asarray(loaded.head.apply({"params": loaded.params}, states, laid, table))

    grid, fewest, more = scores(None), scores(most), scores(most + 5)
    real = np.asarray(inputs.options)
    for compact in (fewest, more):
        np.testing.assert_allclose(compact[real], grid[real], rtol=0, atol=1e-5 * np.abs(grid[real]).max())
        assert not np.any(compact[~real])
    drawn = [jax.eval_shape(loaded.head.init, jax.random.key(0), states, with_slots(inputs, slots), table)
             for slots in (None, most, most + 5)]
    assert drawn[0] == drawn[1] == drawn[2]


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
    """A batch of 100, 17 x 3 and 24 real options in a 17 x 100 grid, which
    the encoding compacts into 128 slots: the loss and every parameter's
    gradient, backbone's and head's, as over the grid to float32 rounding,
    and the loss reads only the real options."""
    tokenizer = ByteTokenizer()
    specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    examples = ragged()
    # Rows as long as the longest needs, since each attention here holds its
    # whole [rows, heads, queries, keys] score on the CPU.
    layout = JointLayout(max_len=16384)
    length = max(len(row.tokens) for example in examples
                 for row in layout.rows(tokenizer, specials, example.state, example.questions))
    objective = DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=16, num_layers=1, num_heads=1, mlp_features=24,
                          max_seq_len=length, attention_impl="xla"),
        tokenizer=tokenizer, specials=specials, layout=JointLayout(max_len=length),
        head=JointSchemaHead(hidden_size=16, width=16, routing_layers=2, layers=1, heads=1, feedforward=24,
                             attention_impl="xla"), shuffle_options=False)
    data = objective.dataset(examples, batch=len(examples), validation=examples)
    (slotted,) = list(data.val(DataPartition()))
    slotted = {name: value for name, value in slotted.items() if name != VALID_ROWS}
    options = np.asarray(slotted["options"])
    assert options.shape[1:] == (17, 100)
    assert sorted(options.reshape(len(examples), -1).sum(axis=1).tolist()) == [24, 51, 100]
    # The encoding compacts the 100 real options of its widest row into 128 slots.
    assert laid_out(slotted).option_slots.shape == (3, 128)
    batch = {name: value for name, value in slotted.items() if name not in ("option_slots", "slotted")}
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(0), None)

    def loss(held, rows):
        return objective.scalar_loss(held, rows, step)[0]

    (grid, grid_gradient), (fewer, fewer_gradient) = (jax.value_and_grad(loss)(variables, rows)
                                                      for rows in (batch, slotted))
    # Float64 truth: the grid's step in float64. The compacted step is held
    # to it as the grid is, per tensor (tests/reference_error.py): no looser
    # rule for the head than for any other port.
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: leaf.astype(jnp.float64)
                            if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, (variables, batch))
        truth, truth_gradient = jax.value_and_grad(loss)(*wide)
        truth, truth_gradient = float(truth), jax.tree.map(np.asarray, truth_gradient)
    # A gradient that is zero to float32's resolution (a key bias's, which
    # softmax cancels) holds only rounding on either path, whose ratio is
    # noise: it must stay at that resolution of the step's largest gradient.
    floor = float(np.finfo(np.float32).eps) * max(distance(leaf, 0) for leaf in jax.tree.leaves(truth_gradient))

    def exact(path, got, want, true):
        label = jax.tree_util.keystr(path)
        if distance(true, 0) <= floor:
            assert distance(got, true) <= floor, (label, distance(got, true), floor)
        elif distance(want, true) == 0:
            assert distance(got, true) == 0, label
        else:
            assert_as_exact_as_the_reference(got, want, true, label)

    exact((), float(fewer), float(grid), truth)
    jax.tree_util.tree_map_with_path(exact, fewer_gradient, grid_gradient, truth_gradient)

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
