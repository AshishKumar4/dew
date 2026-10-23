"""A decision that reads a context beside its state, through gated cross-attention layers attached to a
pretrained backbone: trained as an adapter, counted in its training work, and served, each request's
answer read from its own context."""

import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from decision_support import decision_rows
from model_support import flat_tree
from recording import RecordingTracker
from reference_error import assert_as_exact

from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import Choice, Decide, Example
from dew.decision.layout import Specials
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.decision.questions import ChoiceAnswer
from dew.interop import Pretrained
from dew.nn.cross_attention import CrossAttention, crossing
from dew.objectives.base import FROZEN, thaw
from dew.training import Trainer

pytestmark = pytest.mark.mesh

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"

ROWS = [("one two three", "a", "a first note, longer than a short context holds"),
        ("four five", "b", None),
        ("six seven eight nine", "c", "a second note of some length"),
        ("ten eleven", "a", "third"),
        ("twelve thirteen", "b", "a fourth note"),
        ("fourteen", "c", None)] * 4
"""States, their option, and a note a decision may read beside the state: none for some, and some
longer than a short context holds."""


SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
"""A byte tokenizer's specials, over the fixture's 256-token vocabulary."""

QUESTION = {"pick": Choice("Which one?", ["a", "b", "c"])}

WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "note", "first"]

CONTEXT_SEEDS = 16
"""Check 1's declared seeds for the batched answers with their contexts."""


def conditioned(length: int = 16, layers: tuple[int, ...] = (0, 1)) -> DecisionObjective:
    """A decision over the tiny Llama with fresh cross-attention before `layers`, its context up to
    `length` tokens, its rows laid out by bytes."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    return DecisionObjective(source.with_cross_attention(CrossAttention(layers, length), key=0),
                             tokenizer=ByteTokenizer(), specials=SPECIALS)


def examples() -> list[Example]:
    return [Example(text, QUESTION, {"pick": label}, context=note) for text, label, note in ROWS]


def opened(variables, value: float = 1.0):
    """`variables` with every cross-attention gate at `value`, so the blocks' reading joins the stream."""
    tree = thaw(variables)

    def gate(path, leaf):
        return np.full_like(np.asarray(leaf), value) if "alpha_" in str(path[-1]) else leaf

    return {**tree, "params": jax.tree_util.tree_map_with_path(gate, tree["params"])}


def test_a_backbone_with_fresh_context_layers_scores_every_row_as_its_source_does():
    """The cross-attention blocks start from zero gates, so a decision over the backbone with them
    scores a row, with its context or without, exactly as the backbone alone does; the blocks train
    and the source's weights are held frozen."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    plain = DecisionObjective(source, tokenizer=ByteTokenizer(), specials=SPECIALS)
    crossed = conditioned(8, (0,))
    assert crossed.context_length == 8 and plain.context_length is None
    assert crossed.variables is not None
    trained = jax.tree_util.tree_leaves_with_path(crossed.variables["params"]["backbone"])
    assert trained and all(crossing(tuple(entry.key for entry in path)) for path, _ in trained)
    assert FROZEN in crossed.variables

    asked = Decide(plain.model, plain.init(jax.random.key(0)), plain.layout, plain.tokenizer, plain.specials)
    variables = crossed.init(jax.random.key(0))
    variables = {**variables, "params": {**variables["params"], "head": asked.variables["params"]["head"]}}
    reading = Decide(crossed.model, variables, crossed.layout, crossed.tokenizer, crossed.specials)
    alone = asked("one two three", QUESTION)["pick"]
    for context in (None, "a first note"):
        read = reading("one two three", QUESTION, context=context)["pick"]
        assert isinstance(read, ChoiceAnswer) and isinstance(alone, ChoiceAnswer)
        np.testing.assert_array_equal(read.probabilities, alone.probabilities)


def test_a_conditioned_run_logs_its_training_work_and_serves_with_its_context():
    """Examples with a context trained through context layers: every logged step reports its
    training work, and beyond what its batch's rows cost without their contexts by the
    cross-attention's work over them (`test_the_context_layers_work_is_counted_in_a_conditioned_batch`
    holds the count itself), far past the float32 sum the step logs it by. The gates move off zero and
    every frozen leaf stays as it was; the run's task answers a request otherwise for another context."""
    objective = conditioned()
    data = decision_rows(examples(), 8, objective=objective.data())
    batches = list(itertools.islice(data.train(DataPartition()), 2))
    plain = [objective.training_flops(None, {name: value for name, value in batch.items()
                                             if name != "context_lengths"}) for batch in batches]
    tracker = RecordingTracker()
    trainer = Trainer(objective, optax.adam(1e-3), key=0, tracker=tracker)
    initial = trainer.initial_state()
    state = trainer.fit(data, steps=2, log_every=1)
    logged = [scalars["train/flops"] for _, scalars in tracker.scalars if "train/loss" in scalars]
    assert len(logged) == 2 and all(work > without for work, without in zip(logged, plain, strict=True))
    for before, after in zip(jax.tree.leaves(initial.variables[FROZEN]),
                             jax.tree.leaves(state.variables[FROZEN]), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    gates = [np.asarray(leaf) for path, leaf in flat_tree(thaw(state.variables)["params"]).items()
             if "alpha_" in path]
    assert gates and all(np.all(gate != 0) for gate in gates)

    task = objective.pipeline(state)
    assert isinstance(task, Decide)
    read = [task("one two three", QUESTION, context=context)["pick"] for context in ("a first note", "ten")]
    assert all(isinstance(answer, ChoiceAnswer) for answer in read)
    assert not np.array_equal(read[0].probabilities, read[1].probabilities)


def test_the_context_layers_work_is_counted_in_a_conditioned_batch():
    """The objective's training work for a batch whose rows carry a context is the frozen backbone's,
    three passes of its cross-attention blocks over each row's real context tokens
    (`CrossAttending.cross_flops`), and three of its head; a batch without a context counts no
    cross-attention work."""
    from dew.nn.flops import counted_flops, frozen_training_flops, parameter_flops

    objective = conditioned(8, (0,))
    lengths = np.asarray([5, 7])
    # Two windows of one row each, one question of three options a row.
    batch = {"segments": (np.arange(8) < lengths[:, None]).astype(np.int32),
             "spans": np.zeros((2, 1, 2), np.int32), "questions": np.ones((2, 1), bool),
             "options": np.ones((2, 1, 3), bool), "context_lengths": np.asarray([3, 0], np.int32)}
    np.testing.assert_array_equal(objective.real_lengths(batch), lengths)
    backbone = objective.model.backbone
    cross = backbone.cross_flops(lengths, np.asarray([3, 0]))
    assert cross > 0
    head = objective.model.head.decision_flops(lengths, np.ones(2, np.int64), np.full(2, 3, np.int64))
    frozen = frozen_training_flops(counted_flops(backbone, lengths), parameter_flops(backbone, lengths))
    assert objective.training_flops(None, batch) == frozen + 3 * (cross + head)
    plain = {name: value for name, value in batch.items() if name != "context_lengths"}
    assert objective.training_flops(None, plain) == frozen + 3 * head


def test_a_batch_answers_each_request_from_its_own_context():
    """With the gates open, a request's answer reads its context: another context, or none, answers it
    otherwise. In a batch, each request's every option probability is on par from float64 with its
    answer alone (check 1), so each row reads its own context and not a neighbour's."""
    objective = conditioned(16, (0, 1))
    variables = opened(objective.init(jax.random.key(0)))
    task = Decide(objective.model, variables, objective.layout, objective.tokenizer, objective.specials)
    answers = [task("one two three", QUESTION, context=context)["pick"]
               for context in ("a first note", "ten", None)]
    probabilities = [np.asarray(answer.probabilities) for answer in answers
                     if isinstance(answer, ChoiceAnswer)]
    assert len(probabilities) == len(answers)
    assert not np.array_equal(probabilities[0], probabilities[1])
    assert not np.array_equal(probabilities[0], probabilities[2])

    wide = {}

    def requests(seed):
        """Four requests of one question, two of them on one state, each with a context drawn from
        `seed`, or none."""
        rng = np.random.default_rng([0xC0E7, int(seed)])
        states = [" ".join(rng.choice(WORDS, 6)) for _ in range(3)]
        contexts = [" ".join(rng.choice(WORDS, int(rng.integers(1, 6)))) for _ in range(3)]
        return [(states[0], QUESTION), (states[0], QUESTION), (states[1], QUESTION), (states[2], QUESTION)], [
            contexts[0], contexts[1], None, contexts[2]]

    def scored(reading: Decide, seed, *, together: bool) -> np.ndarray:
        asked, contexts = requests(seed)
        answered = (reading.batch(asked, contexts=contexts) if together else
                    [reading.batch([one], contexts=[context])[0]
                     for one, context in zip(asked, contexts, strict=True)])
        return np.concatenate([answer.probabilities for answers, _ in answered
                               for answer in answers.values()])

    def truth(seed):
        if "task" not in wide:
            model = objective.model
            exact = DecisionModel(model.backbone.clone(dtype=jnp.float64),
                                  model.head.clone(dtype=jnp.float64))
            widened = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64)
                                   if np.issubdtype(np.asarray(leaf).dtype, np.floating) else leaf, variables)
            wide["task"] = Decide(exact, widened, objective.layout, objective.tokenizer, objective.specials)
        return scored(wide["task"], seed, together=False)

    def draw(seed):
        # The draw is its seed; the requests are made from it, their text no array the check widens.
        return (np.int64(seed),)

    assert_as_exact(lambda seed: scored(task, seed, together=True),
                    lambda seed: scored(task, seed, together=False), draw,
                    "batched answers with their contexts", seeds=CONTEXT_SEEDS, truth_fn=truth)


def test_a_request_with_images_reads_its_context_beside_them():
    """Over a backbone that reads images (a `MultimodalTransformer`), the blocks attach to its language
    model. With their gates shut a request with images and a context is answered exactly as with its
    images alone; with them open its context moves the answer."""
    from PIL import Image

    from dew.data.text import HFTokenizer
    from dew.decision import JointLayout

    directory = LLAMA.parent / "qwen38-dense-tiny"
    source = Pretrained.load(directory, dtype="float32", attention_impl="xla")
    tokenizer = HFTokenizer(str(directory), local_files_only=True)
    layout = JointLayout(max_len=1024)
    objective = DecisionObjective(source.with_cross_attention(CrossAttention((0,), 16), key=0),
                                  tokenizer=tokenizer, layout=layout)
    variables = objective.init(jax.random.key(0))
    images = [Image.fromarray(array) for array in np.load(directory / "images.npy")]

    def read(gates, context):
        task = Decide(objective.model, gates, layout, tokenizer, objective.specials,
                      processor=source.processor)
        answer = task("one two three", QUESTION, images=images, context=context)["pick"]
        assert isinstance(answer, ChoiceAnswer)
        return np.asarray(answer.probabilities)

    np.testing.assert_array_equal(read(variables, "a first note"), read(variables, None))
    widened = opened(variables)
    assert not np.array_equal(read(widened, "a first note"), read(widened, "ten"))
