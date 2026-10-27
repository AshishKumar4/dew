"""A batch's real options compacted (`DecisionInputs.option_slots`): the joint head computes those alone.

A batch pads every row to the most questions times the most options, and a
row's real options are a few of them. The joint head computes each option on
its own but for its weight in its question's summary, so over the compacted
options it scores every real option as near float64 as it does over the
grid, whatever the number of option slots: XLA's products sum in an order
that may follow how many rows they multiply.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from decision_support import decision_rows
from flax import linen as nn
from reference_error import assert_as_exact, widened, with_gradients
from safetensors.numpy import load_file

from dew.batch import VALID_ROWS, part, thaw
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
from dew.decision.layout import compacted
from dew.decision.task import laid_out
from dew.interop import Pretrained
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step

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
    return Pretrained.load(TINY, dtype="float32", attention_impl="xla"), table, inputs, padded


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
        return np.asarray(loaded.model.apply(loaded.variables, states, laid, table))

    grid, fewest, more = scores(None), scores(most), scores(most + 5)
    real = np.asarray(inputs.options)
    for compact in (fewest, more):
        np.testing.assert_allclose(compact[real], grid[real], rtol=0, atol=1e-5 * np.abs(grid[real]).max())
        assert not np.any(compact[~real])
    drawn = [jax.eval_shape(loaded.model.init, jax.random.key(0), states, with_slots(inputs, slots), table)
             for slots in (None, most, most + 5)]
    assert drawn[0] == drawn[1] == drawn[2]


CLAUSES, HYPOTHESES, INTENTS = 24, 6, 12
"""The ragged rows' real options: one question of `CLAUSES`, `HYPOTHESES` of three and one of `INTENTS`.
At 40, 6 and 24 a window was 1504 tokens and 13 questions of 40 slots deep, and the site's 512 draws took
201-266 s on armada (ledger 1851); here 1240 tokens and 10 of 24, a draw and its two arms 0.16 s against
0.24."""


def ragged() -> list[Example]:
    """Ragged rows: one question of `CLAUSES` options, `HYPOTHESES` of three and one of `INTENTS`. A
    window's grid is its bound's questions times `CLAUSES` slots, more than the 128 (`OPTION_TILE`) its
    real options compact into."""
    clauses = Choice("Which clause?", [f"c{index}" for index in range(CLAUSES)])
    hypotheses = {f"h{index}": Choice(f"Hypothesis {index}?", ["yes", "no", "unsaid"])
                  for index in range(HYPOTHESES)}
    intents = Choice("Which intent?", [f"i{index}" for index in range(INTENTS)])
    return [Example("a clause on fees", {"clause": clauses}, {"clause": "c7"}),
            Example("a contract", hypotheses, dict.fromkeys(hypotheses, "unsaid")),
            Example("book a flight", {"intent": intents}, {"intent": "i3"})]


OPTION_SEEDS = 256
"""The compacted-options site's declared draws, sized by `seeds_for` paired on armada at 6a398cc82 (both
arms read the same head, states and cotangent, so their upstream rounding is shared): the head's scalar
parameters, the prior's logit scale among them, take their gradient's error from one sum over every
option, whose ratio to the reference's varies widely from draw to draw."""

ALGEBRAIC_ZEROS = ("to_k",)
"""The head's modules whose bias gradient is zero by algebra: a key's bias shifts every key's logit of a
query alike, which the attention's softmax does not move with (its float64 value is residue)."""

SCORER_MARGIN = 1e-4
"""The least gap a draw admits between the two sides of the scorer's `abs(repeated - options)` feature at
a real option (`scorer_gap`), in float64: a hundred times the float32 rounding of the difference, so
float32 and float64 take one slope of `abs` (JAX's is +1 at zero, so a small negative difference float32
rounds to zero flips its slope and moves every gradient behind it by twice its upstream value, which is
no compaction's)."""


def ragged_objective(examples) -> DecisionObjective:
    """A small decoder and joint head over `examples`, a window as long as their longest row."""
    tokenizer = ByteTokenizer()
    specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    # Rows as long as the longest needs, since each attention here holds its
    # whole [rows, heads, queries, keys] score on the CPU.
    layout = JointLayout(max_len=16384)
    length = max(len(row.tokens) for example in examples
                 for row in layout.rows(tokenizer, specials, example.state, example.questions))
    return DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=16, num_layers=1, num_heads=1, mlp_features=24,
                          max_seq_len=length, attention_impl="xla"),
        tokenizer=tokenizer, specials=specials, layout=JointLayout(max_len=length),
        head=JointSchemaHead(hidden_size=16, width=16, routing_layers=2, layers=1, heads=1, feedforward=24,
                             attention_impl="xla"), shuffle_options=False)


def scored(objective: DecisionObjective, inputs: DecisionInputs):
    """The joint head's every real option's logit (0 at a slot with none) and, under the draw's cotangent
    on them, the gradients of its parameters, of the backbone's states it reads and of the vocabulary
    rows (`reference_error.with_gradients`), compiled once for every draw."""
    def logits(head, states, table):
        found = objective.model.head.apply(head, states, inputs, table)
        return jnp.where(inputs.options, found, 0.0)
    return jax.jit(with_gradients(logits))


def scorer_features(objective: DecisionObjective, inputs: DecisionInputs):
    """The features `scorer_hidden` reads (`nn.intercept_methods`) in the joint head's forward of
    `(head, states, table)` over `inputs`, compiled once for every draw: their first two quarters are
    the scorer's `repeated` and `options`."""
    def features(head, states, table):
        seen = []

        def record(next_fun, args, kwargs, context):
            if context.method_name == "__call__" and context.module.name == "scorer_hidden":
                seen.append(args[0])
            return next_fun(*args, **kwargs)

        with nn.intercept_methods(record):
            objective.model.head.apply(head, states, inputs, table)
        return seen[0]
    return jax.jit(features)


def scorer_gap(features, options, *drawn) -> float:
    """The least gap, over the real option slots and the scorer's width, between the two sides of its
    `abs(repeated - options)` feature in float64, read off `features` (`scorer_features`)."""
    with jax.enable_x64(new_val=True):
        taken = np.asarray(features(*widened(drawn)))
    width = taken.shape[-1] // 4
    difference = taken[..., :width] - taken[..., width:2 * width]
    real = np.broadcast_to(np.asarray(options).reshape(difference.shape[:-1])[..., None], difference.shape)
    return float(np.min(np.abs(difference[real])))


def zeroed(tree):
    """`tree` with the algebraic zeros' bias gradients (`ALGEBRAIC_ZEROS`) exactly zero."""
    def exact(path, leaf):
        names = [getattr(entry, "key", None) for entry in path]
        return leaf * 0 if names[-1:] == ["bias"] and any(name in ALGEBRAIC_ZEROS for name in names) else leaf
    return jax.tree_util.tree_map_with_path(exact, tree)


def test_a_ragged_batch_trains_on_compacted_options_as_on_the_grid():
    """Rows of `CLAUSES`, `HYPOTHESES` x 3 and `INTENTS` real options packed into windows of a grid as
    many questions deep as a window holds, which the packing compacts into the slots its bounds allow.
    The compaction is the joint head's alone (the backbone reads the rows' tokens, never their option
    slots), so the head is what check 1 holds: over draws of the backbone's states and of a cotangent
    on each real option, its every real option's logit and the gradients of its parameters, of the
    states and of the vocabulary rows it reads are on par with the grid's from float64, the compaction
    only reordering the sums over the grid's option slots. The backbone's own gradient is its backward
    of those states' gradient, one program on both. The loss reads only the real options."""
    examples = ragged()
    objective = ragged_objective(examples)
    data = decision_rows(examples, len(examples), examples, objective=objective.data())
    (slotted,) = list(data.val(DataPartition()))
    slotted = {name: value for name, value in slotted.items() if name != VALID_ROWS}
    options = np.asarray(slotted["options"])
    bounds = objective.bounds
    assert options.shape[1:] == (bounds.questions, CLAUSES)
    starts = np.asarray(slotted["spans"])[..., 0]
    owners = np.where(slotted["questions"], np.take_along_axis(slotted["segments"], starts, axis=1), 0)
    totals = [int(options[window][owners[window] == row].sum()) for window in range(len(options))
              for row in range(1, int(np.max(slotted["segments"][window])) + 1)]
    assert sorted(totals) == sorted([CLAUSES, 3 * HYPOTHESES, INTENTS])
    # A window's real options compact into the slots its bounds allow, a multiple of 128.
    capacity = -(-bounds.options // 128) * 128
    compact = laid_out(slotted, bounds.segments)
    assert compact.option_slots.shape == (len(options), capacity) and capacity < bounds.questions * CLAUSES
    batch = {name: value for name, value in slotted.items() if name not in ("option_slots", "slotted")}
    grid = laid_out(batch, bounds.segments)
    variables = objective.init(jax.random.key(0))
    head = thaw(part(variables, "head"))
    table = objective.model.table(variables)
    states_shape = (*np.shape(slotted["tokens"]), objective.model.head.hidden_size)
    features = scorer_features(objective, grid)

    def draw(seed):
        """A seed's backbone states and cotangent on each real option, from the first of its candidates
        whose scorer gap clears SCORER_MARGIN."""
        for attempt in range(64):
            rng = np.random.default_rng([0x5107, seed, attempt])
            states = rng.standard_normal(states_shape, np.float32)
            if scorer_gap(features, grid.options, head, states, table) > SCORER_MARGIN:
                readout = rng.standard_normal(options.shape, np.float32) * options
                return head, states, table, readout
        raise AssertionError(f"no candidate of seed {seed} has its scorer gap above {SCORER_MARGIN}")

    on_grid = scored(objective, grid)
    # The truth of the algebraic zeros is exactly zero, which each of their rows is held to.
    assert_as_exact(scored(objective, compact), on_grid, draw, "compacted options logits and gradients",
                    seeds=OPTION_SEEDS, truth_fn=lambda *inputs: zeroed(on_grid(*inputs)))

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
