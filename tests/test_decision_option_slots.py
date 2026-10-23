"""A batch's real options compacted (`DecisionInputs.option_slots`): the joint head computes those alone.

A batch pads every row to the most questions times the most options, and a
row's real options are a few of them. The joint head computes each option on
its own but for its weight in its question's summary, so over the compacted
options it scores every real option as it does over the grid, to float32
rounding, whatever the number of option slots: XLA's products sum in an
order that may follow how many rows they multiply.
"""

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from decision_support import decision_rows
from reference_error import assert_as_exact, widened, with_gradients
from safetensors.numpy import load_file

from dew.data.dataset import DataPartition
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
from dew.decision.joint_checkpoint import StoredJointHead
from dew.decision.layout import compacted
from dew.decision.task import laid_out
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import VALID_ROWS, Step

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "joint_checkpoint" / "tiny"
BACKBONE = FIXTURES / "hf" / "qwen38-dense-tiny"
CASES = json.loads((FIXTURES / "laya" / "cases.json").read_text())


def stored_batch():
    """The stored tiny joint head, its output table, the five cases laid out in one
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
    return StoredJointHead.load(TINY, attention_impl="xla"), table, inputs, padded


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
    with pytest.raises(ValueError):
        compacted(options, 4)


def test_the_stored_joint_head_scores_compacted_options_as_it_scores_the_grid():
    """The stored tiny joint head over the five cases: every real option's
    logit as over the grid at two numbers of option slots, to float32
    rounding (on armada's CPUs they moved by up to 6.6e-7 in logits of about 0.6
    at two numbers of slots); a slot with no real option scores zero,
    which the model's logits mask (`DecisionModel.logits`). The head's
    parameters do not depend on the number of slots, so a checkpoint trained
    at one serves at another."""
    loaded, table, inputs, states = stored_batch()
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


SLOT_SEEDS = 512
"""The compacted-options site's declared draws (`seeds_for`, paired): each draw is DRAW_COPIES sets of
the ragged rows' states and answers from its seed, over the objective initialized from key 0."""

DRAW_COPIES = 4
"""The sets of ragged rows a draw packs: each parameter's gradient sums over all of them, so one
draw's rounding is many rows', whose ratio to the reference's varies less across draws than one
set's, heavy-tailed in the head's attention query and key gradients."""

ALGEBRAIC_ZEROS = ("to_k", "scorer_out")
"""The head's modules whose bias gradient is zero by algebra: a key's bias shifts every key's logit
of a query alike, and the scorer's shared bias every option's, so the softmax and the loss do not
move with either (their float64 values are residue)."""


def ragged(seed: int | Sequence[int] = 0) -> list[Example]:
    """Ragged rows: one question of forty options, six of three and one of twenty-four, their states
    and answers drawn from `seed` at fixed lengths, so every draw packs alike. A window's grid is its
    bound's questions times forty slots, of which a row holds 40, 18 or 24 real options; compacted, a
    window keeps 128 slots (`OPTION_TILE`), fewer than its grid holds, the dropped ones options no row
    has."""
    rng = np.random.default_rng(seed)

    def state(prefix):
        return prefix + " " + "".join(rng.choice(list("abcdefghij"), 12))

    clauses = Choice("Which clause?", [f"c{index}" for index in range(40)])
    hypotheses = {f"h{index}": Choice(f"Hypothesis {index}?", ["yes", "no", "unsaid"]) for index in range(6)}
    intents = Choice("Which intent?", [f"i{index}" for index in range(24)])
    return [Example(state("a clause on"), {"clause": clauses}, {"clause": f"c{rng.integers(40)}"}),
            Example(state("a contract"), hypotheses,
                    {name: str(rng.choice(["yes", "no", "unsaid"])) for name in hypotheses}),
            Example(state("book a flight"), {"intent": intents}, {"intent": f"i{rng.integers(24)}"})]


def ragged_objective() -> DecisionObjective:
    """A small decoder and joint head over the ragged rows, a window as long as their longest row."""
    tokenizer = ByteTokenizer()
    specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    # Rows as long as the longest needs, since each attention here holds its
    # whole [rows, heads, queries, keys] score on the CPU.
    layout = JointLayout(max_len=16384)
    length = max(len(row.tokens) for example in ragged()
                 for row in layout.rows(tokenizer, specials, example.state, example.questions))
    return DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=16, num_layers=1, num_heads=1, mlp_features=24,
                          max_seq_len=length, attention_impl="xla"),
        tokenizer=tokenizer, specials=specials, layout=JointLayout(max_len=length),
        head=JointSchemaHead(hidden_size=16, width=16, routing_layers=2, layers=1, heads=1, feedforward=24,
                             attention_impl="xla"), shuffle_options=False)


def ragged_batches(objective: DecisionObjective, seed: int = 0, copies: int = 1):
    """The ragged rows of `seed`, `copies` sets of them each drawn apart, packed into windows of a
    grid as many questions deep as a window holds, with their options compacted into the slots its
    bounds allow (`slotted`) and without (`batch`)."""
    examples = ragged(seed) if copies == 1 else [example for copy in range(copies)
                                                 for example in ragged([seed, copy])]
    data = decision_rows(examples, len(examples), examples, objective=objective.data())
    (slotted,) = list(data.val(DataPartition()))
    slotted = {name: value for name, value in slotted.items() if name != VALID_ROWS}
    batch = {name: value for name, value in slotted.items() if name not in ("option_slots", "slotted")}
    return slotted, batch


def stepped(objective: DecisionObjective, compact: bool):
    """The forward over the compacted options or over the grid, each real option's logit (0 at a slot
    with none) and the loss, and every parameter's gradient under the draw's cotangent
    (`reference_error.with_gradients`): the forward's every option a row of its own."""
    step = Step(jnp.asarray(0), jax.random.key(0), None)

    def run(variables, grid, slotted, cotangent):
        rows = slotted if compact else grid

        def forward(held):
            inputs = laid_out(rows, objective.bounds.segments)
            logits = jnp.where(inputs.options, objective.model.logits(held, inputs), 0.0)
            return logits, objective.scalar_loss(held, rows, step)[0]

        return with_gradients(forward)(variables, cotangent)
    return jax.jit(run)


SCORER_MARGIN = 1e-4
"""The least gap a draw admits between a scorer feature's two sides at a real option (`scorer_gap`):
a hundred times the fp32 rounding of `repeated - options` (seed 225 of the unmargined draw rounded a
7.8e-7 difference to 0, which flips `abs`'s slope and moved every gradient by its upstream value), so
float32 and float64 take one slope. What it excludes, the slope at a difference that rounds to zero,
is `abs`'s own convention and no compaction's, held exactly by
tests/test_decision_joint_checkpoint.py::test_a_scorer_tie_moves_neither_side_through_the_difference;
the joint head's scores themselves are held to its release's by
test_the_stored_joint_head_matches_its_release_under_the_float64_rule."""


def scorer_inputs(objective: DecisionObjective):
    """The features `scorer_hidden` takes (`nn.intercept_methods`) in the objective's forward of
    `(variables, inputs)`, compiled once for every draw: the first two quarters of their width are
    the scorer's `repeated` and `options`."""
    def features(variables, inputs):
        seen = []

        def record(next_fun, args, kwargs, context):
            if context.method_name == "__call__" and context.module.name == "scorer_hidden":
                seen.append(args[0])
            return next_fun(*args, **kwargs)

        with nn.intercept_methods(record):
            objective.model.logits(variables, inputs)
        return seen[0]
    return jax.jit(features)


def scorer_gap(objective: DecisionObjective, features, variables, grid) -> float:
    """The least gap, over the grid's real option slots and the scorer's width, between the two sides
    of its `abs(repeated - options)` feature in float64, read off `features` (`scorer_inputs`)."""
    with jax.enable_x64(new_val=True):
        wide_variables, wide_grid = widened((variables, grid))
        inputs = laid_out(wide_grid, objective.bounds.segments)
        taken = np.asarray(features(wide_variables, inputs))
    width = taken.shape[-1] // 4
    difference = taken[..., :width] - taken[..., width:2 * width]
    options = np.asarray(inputs.options).reshape(difference.shape[:-1])
    real = np.broadcast_to(options[..., None], difference.shape)
    above, below = difference[real & (difference >= 0)], difference[real & (difference < 0)]
    return float(min(np.min(above, initial=np.inf), np.min(-below, initial=np.inf)))


def zeroed(tree):
    """`tree` with the algebraic zeros' bias gradients (`ALGEBRAIC_ZEROS`) exactly zero."""
    def exact(path, leaf):
        names = [getattr(entry, "key", None) for entry in path]
        return leaf * 0 if names[-1:] == ["bias"] and any(name in ALGEBRAIC_ZEROS for name in names) else leaf
    return jax.tree_util.tree_map_with_path(exact, tree)


def test_a_ragged_batch_trains_on_compacted_options_as_on_the_grid():
    """The ragged rows' packing compacts its real options into the slots its bounds allow, fewer than
    its grid's, and each real option's logit, the loss and every parameter's gradient, backbone's and
    head's, over them are on par with the grid's from float64 (check 1, over draws of the rows and of a
    cotangent on each real option): the compaction only reorders the sums over the grid's option
    slots. The loss reads only the real options."""
    objective = ragged_objective()
    slotted, _ = ragged_batches(objective)
    options = np.asarray(slotted["options"])
    bounds = objective.bounds
    assert options.shape[1:] == (bounds.questions, 40)
    starts = np.asarray(slotted["spans"])[..., 0]
    owners = np.where(slotted["questions"], np.take_along_axis(slotted["segments"], starts, axis=1), 0)
    totals = [int(options[window][owners[window] == row].sum()) for window in range(len(options))
              for row in range(1, int(np.max(slotted["segments"][window])) + 1)]
    assert sorted(totals) == [18, 24, 40]
    # A window's real options compact into the slots its bounds allow, a multiple of 128, fewer than its
    # grid's.
    capacity = -(-bounds.options // 128) * 128
    assert laid_out(slotted, bounds.segments).option_slots.shape == (len(options), capacity)
    assert capacity < bounds.questions * 40

    variables = objective.init(jax.random.key(0))
    features = scorer_inputs(objective)

    def draw(seed):
        """A seed's rows and a cotangent drawn apart from them (each real option's, the loss's 1), from
        the first of its candidate rows whose scorer gap clears SCORER_MARGIN. The candidates are
        ragged's seeds past 0, which the structure above reads."""
        for attempt in range(64):
            index = 1 + 64 * seed + attempt
            slotted, grid = ragged_batches(objective, index, DRAW_COPIES)
            if scorer_gap(objective, features, variables, grid) > SCORER_MARGIN:
                options = np.asarray(grid["options"]).astype(bool)
                readout = np.random.default_rng([0x5107, index]).standard_normal(options.shape, np.float32)
                return variables, grid, slotted, (readout * options, np.float32(1.0))
        raise AssertionError(f"no candidate of seed {seed} has its scorer gap above {SCORER_MARGIN}")

    grid = stepped(objective, compact=False)
    # The truth of the algebraic zeros is exactly zero, which each of their rows is held to.
    assert_as_exact(stepped(objective, compact=True), grid, draw, "compacted options logits and gradient",
                    seeds=SLOT_SEEDS, truth_fn=lambda *inputs: zeroed(grid(*inputs)))

    step = Step(jnp.asarray(0), jax.random.key(0), None)

    def loss(held, rows):
        return objective.scalar_loss(held, rows, step)[0]

    fewer, fewer_gradient = jax.value_and_grad(loss)(variables, slotted)
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
