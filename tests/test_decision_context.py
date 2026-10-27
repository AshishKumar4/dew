"""A decision that reads a context beside its state, through gated cross-attention layers attached to a
pretrained backbone: trained as an adapter, counted in its training work, and served, each request's
answer read from its own context."""

import dataclasses
import itertools
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from decision_support import decision_rows
from model_support import flat_tree
from recording import RecordingTracker
from reference_error import as_given, assert_as_exact, widened, without_ynn

from dew.batch import FROZEN, thaw
from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import Choice, Decide, Example
from dew.decision.images import Images
from dew.decision.layout import DecisionInputs, Specials
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.decision.questions import ChoiceAnswer
from dew.interop import Pretrained
from dew.nn.cross_attention import ContextEncoder, CrossAttention, crossing
from dew.nn.multimodal import MultimodalTransformer
from dew.training import Trainer

pytestmark = pytest.mark.mesh

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"

ROWS = [("one two three", "a", "a first note, longer than most"),
        ("four five", "b", None),
        ("six seven eight nine", "c", "a second note of some length"),
        ("ten eleven", "a", "third"),
        ("twelve thirteen", "b", "a fourth note"),
        ("fourteen", "c", None)] * 4
"""States, their option, and a note a decision may read beside the state: none for some, each within
`conditioned`'s context."""


SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
"""A byte tokenizer's specials, over the fixture's 256-token vocabulary."""

QUESTION = {"pick": Choice("Which one?", ["a", "b", "c"])}

WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "note", "first"]

CONTEXT_SEEDS = 8
"""Check 1's declared seeds for the batched answers with their contexts, sized paired (`seeds_for`
with the batch as `dew_fn`, 2026-10-23 on armada, both arms without YNNPACK's fusions)."""


def conditioned(length: int = 32, layers: tuple[int, ...] = (0, 1),
                encoder: ContextEncoder = "table") -> DecisionObjective:
    """A decision over the tiny Llama with fresh cross-attention before `layers`, its context up to
    `length` tokens and encoded as `encoder` says, its rows laid out by bytes."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    return DecisionObjective(source.with_cross_attention(CrossAttention(layers, length), key=0),
                             tokenizer=ByteTokenizer(), specials=SPECIALS, context_encoder=encoder)


def examples() -> list[Example]:
    return [Example(text, QUESTION, {"pick": label}, context=note) for text, label, note in ROWS]


def in_float64(task: Decide) -> Decide:
    """`task` computing in float64: its backbone, a multimodal one's language model too, and its head,
    over its weights widened (`reference_error.widened`), for a truth under `jax.enable_x64`."""
    backbone = task.model.backbone.clone(dtype=jnp.float64)
    if isinstance(backbone, MultimodalTransformer):
        backbone = backbone.clone(language_model=backbone.language_model.clone(dtype=jnp.float64))
    model = DecisionModel(backbone, task.model.head.clone(dtype=jnp.float64))
    return dataclasses.replace(task, model=model, variables=widened(task.variables))


COMPILED = {}
"""Each model's forward (`DecisionModel.logits`), compiled once without YNNPACK's fusions
(`without_ynn`): the numerical checks here compare two Dew programs."""


def scored(task: Decide, asked: Sequence[tuple[str, str | None]], images=()) -> np.ndarray:
    """The real options' logits of `asked`'s requests, each a state and a context, laid out by `task`
    and collated into one pass (each request's images beside it where given, one request a pass),
    through its model compiled without YNNPACK's fusions."""
    forward = COMPILED.setdefault(id(task.model), without_ynn(task.model.logits))
    if images:
        [(state, context)] = asked
        assert task.processor is not None and task.layout.images
        media = Images.of(task.processor, images, task.processor.image_text())
        [row] = task._encoded(state, QUESTION, media.tokens, context=context)
        inputs = DecisionInputs.collate([row], task.specials.pad, context_length=task.context_length)
        logits = forward(task.variables, inputs, media=media.inputs(task.processor, row))
    else:
        # Laid out as `Decide.batch` lays a batch out (`Decide.rows`).
        rows = [row for own in task.rows([(state, QUESTION) for state, _ in asked],
                                         contexts=[context for _, context in asked]) for row in own]
        inputs = DecisionInputs.collate(rows, task.specials.pad, context_length=task.context_length)
        logits = forward(task.variables, inputs)
    return np.asarray(logits)[np.asarray(inputs.options)]


def drawn(seed) -> tuple[str, str]:
    """A seed's state and context, of the fixture's words."""
    rng = np.random.default_rng([0xC0E8, int(seed)])
    return " ".join(rng.choice(WORDS, 6)), " ".join(rng.choice(WORDS, int(rng.integers(1, 6))))


FRESH_SEEDS = 8
"""Check 1's declared seeds for the shut gates' answers against the source's, and beside images, sized
paired (`seeds_for` with the shut gates' answers as `dew_fn`, 2026-10-23 on armada)."""


def opened(variables, value: float = 1.0):
    """`variables` with every cross-attention gate at `value`, so the blocks' reading joins the stream."""
    tree = thaw(variables)

    def gate(path, leaf):
        return np.full_like(np.asarray(leaf), value) if "alpha_" in str(path[-1]) else leaf

    return {**tree, "params": jax.tree_util.tree_map_with_path(gate, tree["params"])}


def windows_of(objective: DecisionObjective, rows, count: int = 6) -> list[dict]:
    """The first `count` training windows `objective`'s data packs of `rows`, three rows a window at
    most: a window's packing alone, the bounds a backbone's cross-attention reads not yet lifted."""
    from dew.decision.layout import Bounds

    held = DecisionObjective.bounds

    def three(self) -> Bounds:
        return dataclasses.replace(held.func(self), segments=3)

    DecisionObjective.bounds = property(three)
    try:
        reader = decision_rows(rows, None, objective=objective.data(), tokens=objective.layout.max_len)
        return list(itertools.islice(reader.train(DataPartition()), count))
    finally:
        DecisionObjective.bounds = held


@pytest.mark.parametrize("grouped", [False, True], ids=["rows", "unit"])
def test_rows_that_carry_a_context_pack_several_a_window_each_beside_its_own(grouped):
    """Rows that carry a context pack several a window as rows without one do, each row's context
    laid end to end along the context in the order the rows stand and numbered as `segments` numbers
    its row (`context_segments`, 0 for padding), a row with no context holding none: every row's
    context is its own example's, under a layout of a row an example and under a unit of rows that
    train together (row 1885)."""
    from dew.decision import Group, GroupedExamples

    objective = conditioned()
    if grouped:
        objective = DecisionObjective(objective.model.backbone, tokenizer=ByteTokenizer(), specials=SPECIALS,
                                      context_encoder="table", consistency=0.5)
        pairs = [(examples()[index], examples()[index + 1]) for index in range(0, 12, 2)]
        rows = [GroupedExamples(pair, groups=(Group("pick", {f"{member}.pick": {option: option for option in
                                                                                  "abc"}
                                                             for member in range(2)}),)) for pair in pairs]
        owned = [example for pair in pairs for example in pair]
    else:
        rows = owned = examples()
    notes = {text: note for text, _, note in ROWS}
    several = 0
    for window in windows_of(objective, rows):
        segments, contexts = np.asarray(window["segments"]), np.asarray(window["context_segments"])
        tokens, context = np.asarray(window["tokens"]), np.asarray(window["context"])
        for batch_row in range(len(segments)):
            standing = [number for number in np.unique(segments[batch_row]) if number > 0]
            several += len(standing) > 1
            assert set(np.unique(contexts[batch_row])) - {0} <= set(standing)
            for number in standing:
                text = bytes(tokens[batch_row][segments[batch_row] == number].astype(np.uint8)).decode(
                    errors="replace")
                (state,) = {example.state for example in owned if example.state in text}
                read = bytes(context[batch_row][contexts[batch_row] == number].astype(np.uint8)).decode()
                assert (notes[state] or "") in read and (read == "") == (notes[state] is None), (state, read)
    assert several, "no window held more than one row"


def test_a_backbone_with_fresh_context_layers_scores_every_row_as_its_source_does():
    """The cross-attention blocks start from zero gates, so a decision over the backbone with them
    answers a row, with its context or without, as the backbone alone does: on par from float64 (check
    1), the two being programs of their own. The blocks train and the source's weights are held
    frozen."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    plain = DecisionObjective(source, tokenizer=ByteTokenizer(), specials=SPECIALS)
    crossed = conditioned(32, (0,))
    assert crossed.context_length == 32 and plain.context_length is None
    assert crossed.variables is not None
    trained = jax.tree_util.tree_leaves_with_path(crossed.variables["params"]["backbone"])
    assert trained and all(crossing(tuple(entry.key for entry in path)) for path, _ in trained)
    assert FROZEN in crossed.variables

    asked_task = Decide(plain.model, plain.init(jax.random.key(0)), plain.layout, plain.tokenizer,
                        plain.specials)
    variables = crossed.init(jax.random.key(0))
    head = asked_task.variables["params"]["head"]
    variables = {**variables, "params": {**variables["params"], "head": head}}
    reading = Decide(crossed.model, variables, crossed.layout, crossed.tokenizer, crossed.specials)
    wide = {}

    def truth(seed):
        with jax.enable_x64(new_val=True):
            if "task" not in wide:
                wide["task"] = in_float64(asked_task)
            return scored(wide["task"], [(drawn(seed)[0], None)])

    for read in (as_given(lambda seed: scored(reading, [(drawn(seed)[0], None)])),
                 as_given(lambda seed: scored(reading, [drawn(seed)]))):
        assert_as_exact(read, as_given(lambda seed: scored(asked_task, [(drawn(seed)[0], None)])),
                        lambda seed: (np.int64(seed),), "shut gates against the source", seeds=FRESH_SEEDS,
                        truth_fn=truth)


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
    otherwise. A batch lays each request out with its own context (`Decide.rows`, which `Decide.batch`
    scores), each request's rows exactly its lone call's; and in one pass of four, each request's every
    option logit is on par from float64 with its pass alone (check 1), so each row reads its own
    context and not a neighbour's."""
    objective = conditioned(32, (0, 1))
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
        return [(states[0], contexts[0]), (states[0], contexts[1]), (states[1], None),
                (states[2], contexts[2])]

    asked = requests(0)
    together = [row.context for own in task.rows([(state, QUESTION) for state, _ in asked],
                                                 contexts=[context for _, context in asked]) for row in own]
    alone = [row.context for state, context in asked
             for row in task.rows([(state, QUESTION)], contexts=[context])[0]]
    assert together == alone and len(set(together)) == len(asked)

    def read(reading: Decide, seed, *, together: bool) -> np.ndarray:
        asked = requests(seed)
        if together:
            return scored(reading, asked)
        return np.concatenate([scored(reading, [one]) for one in asked])

    def truth(seed):
        with jax.enable_x64(new_val=True):
            if "task" not in wide:
                wide["task"] = in_float64(task)
            return read(wide["task"], seed, together=False)

    def draw(seed):
        # The draw is its seed; the requests are made from it, their text no array the check widens.
        return (np.int64(seed),)

    assert_as_exact(as_given(lambda seed: read(task, seed, together=True)),
                    as_given(lambda seed: read(task, seed, together=False)), draw,
                    "batched answers with their contexts", seeds=CONTEXT_SEEDS, truth_fn=truth)


def test_a_request_with_images_reads_its_context_beside_them():
    """Over a backbone that reads images (a `MultimodalTransformer`), the blocks attach to its language
    model. With their gates shut a request with images and a context is answered as with its images
    alone, on par from float64 (check 1); with them open its context moves the answer."""
    from PIL import Image

    from dew.data.text import HFTokenizer
    from dew.decision import JointLayout

    directory = LLAMA.parent / "qwen38-dense-tiny"
    source = Pretrained.load(directory, dtype="float32", attention_impl="xla")
    tokenizer = HFTokenizer(str(directory), local_files_only=True)
    layout = JointLayout(max_len=1024)
    objective = DecisionObjective(source.with_cross_attention(CrossAttention((0,), 32), key=0),
                                  tokenizer=tokenizer, layout=layout)
    variables = objective.init(jax.random.key(0))
    images = [Image.fromarray(array) for array in np.load(directory / "images.npy")]

    def task(gates) -> Decide:
        return Decide(objective.model, gates, layout, tokenizer, objective.specials,
                      processor=source.processor)

    shut, wide = task(variables), {}

    def truth(seed):
        with jax.enable_x64(new_val=True):
            if "task" not in wide:
                wide["task"] = in_float64(shut)
            return scored(wide["task"], [(drawn(seed)[0], None)], images)

    assert_as_exact(as_given(lambda seed: scored(shut, [drawn(seed)], images)),
                    as_given(lambda seed: scored(shut, [(drawn(seed)[0], None)], images)),
                    lambda seed: (np.int64(seed),), "shut gates beside images", seeds=FRESH_SEEDS,
                    truth_fn=truth)
    reading = task(opened(variables))
    assert not np.array_equal(scored(reading, [("one two three", "a first note")], images),
                              scored(reading, [("one two three", "ten")], images))


def test_a_backbone_encoded_context_is_the_backbones_own_pass_over_it():
    """With `context_encoder="backbone"` the cross-attention reads the backbone's final states over the
    context, its own pass (`encode_context`): the decision states are the backbone's over the row with
    that conditioning, exactly; they differ from the table encoder's, both read the same weights, and
    the encoder rides the task's settings, so a reload reads its context the same way."""
    from dew.decision.task import DecideSettings
    from dew.inference.tasks import saved_settings
    from dew.nn.cross_attention import encode_context
    from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS

    table, encoded = conditioned(32, (0, 1)), conditioned(32, (0, 1), "backbone")
    assert encoded.model.context_encoder == "backbone" and table.model.context_encoder == "table"
    variables = opened(encoded.init(jax.random.key(0)))
    tasks = [Decide(objective.model, variables, objective.layout, objective.tokenizer, objective.specials)
             for objective in (table, encoded)]
    rows = tasks[1].rows([("one two three", QUESTION)], contexts=["a first note"])[0]
    inputs = DecisionInputs.collate(rows, SPECIALS.pad, context_length=32)
    states = [np.asarray(task.model.states(variables, inputs)) for task in tasks]
    model = encoded.model
    backbone = {name: value["backbone"] for name, value in variables.items() if "backbone" in value}
    context = encode_context(model.backbone, backbone, inputs.context, inputs.context_lengths,
                             encoder="backbone")[CONTEXT]
    direct = model.backbone.apply(backbone, inputs.tokens, coordinates=inputs.coordinates(),
                                  conditioning={CONTEXT: context.astype(model.backbone.context_dtype),
                                                CONTEXT_LENGTHS: inputs.context_lengths},
                                  method="hidden_states")
    np.testing.assert_array_equal(states[1], np.asarray(direct))
    assert not np.array_equal(states[0], states[1])
    settings = encoded.task_settings()
    assert settings.context_encoder == "backbone"
    record = {name: getattr(settings, name) for name in ("head", "layout", "specials", "tokenizer",
                                                          "processor", "context_encoder")}
    assert saved_settings(record, DecideSettings).context_encoder == "backbone"
    del record["context_encoder"]
    assert saved_settings(record, DecideSettings).context_encoder == "table"
    assert DecisionObjective(tasks[1]).model.context_encoder == "backbone"


def test_a_context_its_cross_attention_cannot_hold_is_refused_never_cut():
    """A context of more tokens than the cross-attention reads is refused, by a task and over the wire
    (ValueError, which a server turns into the request's own refusal), and refuses a run's data where
    its fitting reads the record; one of exactly that many is read whole."""
    from dew.decision.rows import Fit

    objective = conditioned(16, (0,))
    task = Decide(objective.model, objective.init(jax.random.key(0)), objective.layout, objective.tokenizer,
                  objective.specials)
    exact, over = "x" * 16, "x" * 17
    [row] = task._encoded("one two", QUESTION, context=exact)
    assert row.context is not None and len(row.context) == 16
    with pytest.raises(ValueError):
        task("one two", QUESTION, context=over)
    wire = {"pick": {"type": "choice", "instructions": "Which one?", "criteria": ["a", "b", "c"]}}
    [refused] = task.systemone([{"state": "one two", "questions": wire, "context": over}])
    assert isinstance(refused, ValueError)
    fit = Fit(objective.layout, objective.tokenizer, objective.specials, objective.model.head,
              objective.context_length)
    assert not isinstance(fit.map({"example": Example("one two", QUESTION, {"pick": "a"}, context=exact)}),
                          type(None))
    with pytest.raises(ValueError):
        fit.map({"example": Example("one two", QUESTION, {"pick": "a"}, context=over)})


def test_the_batched_engine_reads_a_requests_context():
    """The batched engine (`WireScoring`) reads a request's context, as `Decide` does: each answer's
    probabilities those of the task's own call with that context, to the bit, the one program; under
    `strict` the context is dropped, as Jev's fields alone are read."""
    objective = conditioned(32, (0, 1))
    variables = opened(objective.init(jax.random.key(0)))
    task = Decide(objective.model, variables, objective.layout, objective.tokenizer, objective.specials)
    wire = {"pick": {"type": "choice", "instructions": "Which one?", "criteria": ["a", "b", "c"]}}
    bodies = [{"state": "one two three", "questions": wire, "context": "a first note"},
              {"state": "one two three", "questions": wire}]
    answers = task.systemone(bodies)
    for body, answer in zip(bodies, answers, strict=True):
        alone = task.systemone(body)
        assert isinstance(answer, dict) and answer["answers"] == alone["answers"]
    with_context, without = answers
    assert isinstance(with_context, dict) and isinstance(without, dict)
    assert with_context["answers"] != without["answers"]
    assert task._read_wire(bodies[0], strict=True)[3] is None
