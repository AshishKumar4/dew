"""A decision that reads a context beside its state, through gated cross-attention layers attached to a
pretrained backbone: trained as an adapter, counted in its training work, and served, each request's
answer read from its own context."""

import copy
import dataclasses
import functools
import itertools
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from decision_support import decision_rows
from model_support import attending, flat_tree
from recording import RecordingTracker
from reference_error import as_given, assert_as_exact, widened, without_ynn

from dew.batch import FROZEN, part, thaw, unboxed
from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import Choice, Decide, Example
from dew.decision.config import CrossContext
from dew.decision.images import Images
from dew.decision.inputs import DecisionInputs
from dew.decision.layout import SegmentLayout, Specials
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.decision.questions import ChoiceAnswer
from dew.inputs import CharTable
from dew.interop import Pretrained
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.cross_attention import ContextEncoder, CrossAttention, crossing
from dew.nn.multimodal import MultimodalTransformer
from dew.training import Trainer, TrainerAdvanced

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
                encoder: ContextEncoder = "table", segments: int | None = None) -> DecisionObjective:
    """A decision over the tiny Llama with fresh cross-attention before `layers`, its context up to
    `length` tokens and encoded as `encoder` says, its rows laid out by bytes, at most `segments` rows
    a window where given."""
    source = Pretrained.load(LLAMA, dtype="float32")
    return DecisionObjective(source.with_cross_attention(CrossAttention(layers, length), key=0),
                             tokenizer=ByteTokenizer(), specials=SPECIALS, context_encoder=encoder,
                             segments=segments)


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
        # Laid out as a call over requests lays them out (`Decide.rows`).
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
    """The first `count` windows `objective`'s data packs of `rows` as its evaluation reads them, in
    order, each window as many rows as its bounds allow."""
    reader = decision_rows(rows, None, rows, objective=objective.data(), tokens=objective.layout.max_len)
    return list(itertools.islice(reader.val(DataPartition()), count))


@attending("reference")
@pytest.mark.parametrize("grouped", ["rows", "unit", "unit beside rows"])
def test_rows_that_carry_a_context_pack_several_a_window_each_beside_its_own(grouped):
    """Rows that carry a context pack several a window as rows without one do, each row's context
    laid end to end along the context in the order the rows stand and numbered as `segments` numbers
    its row (`context_segments`, 0 for padding), a row with no context holding none: every row's
    context is its own example's, under a layout of a row an example, under a unit of rows that
    train together, and under a run of consistency whose ungrouped examples stand in rows of their own
    beside its units (row 1885; the last crashed where a lone row had no `context_row`)."""
    from dew.decision import Group, GroupedExamples

    objective = conditioned(segments=3)
    if grouped != "rows":
        objective = DecisionObjective(objective.model.backbone, tokenizer=ByteTokenizer(), specials=SPECIALS,
                                      context_encoder="table", consistency=0.5, segments=3)
        pairs = [(examples()[index], examples()[index + 1]) for index in range(0, 12, 2)]
        rows = [GroupedExamples(pair, groups=(Group("pick", {f"{member}.pick": {option: option for option in
                                                                                  "abc"}
                                                             for member in range(2)}),)) for pair in pairs]
        owned = [example for pair in pairs for example in pair]
        if grouped == "unit beside rows":
            # Examples of no group, each a row of its own beside the units.
            alone = examples()[12:18]
            rows, owned = [item for pair in zip(rows, alone, strict=True) for item in pair], owned + alone
    else:
        rows = owned = examples()
    notes = {text: note for text, _, note in ROWS}
    several = 0
    for window in windows_of(objective, rows):
        assert "context_segments" in window, sorted(window)
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


@attending("reference")
def test_a_unit_whose_contexts_outgrow_a_windows_trains_its_examples_apart():
    """Grouped examples train as one unit where a window holds their rows' contexts end to end
    (`DecisionObjective.context_window`), and apart where their contexts outgrow it, as where their
    tokens do: each fits a window alone, so none is cut or left out."""
    from dew.decision import Group, GroupedExamples
    from dew.decision.rows import Fit

    backbone = conditioned().model.backbone
    objective = DecisionObjective(backbone, tokenizer=ByteTokenizer(), specials=SPECIALS,
                                  context_encoder="table", consistency=0.5, segments=3, context_window=32)
    [fit] = [step for step in objective.data().train if isinstance(step, Fit)]
    group = (Group("pick", {f"{member}.pick": {option: option for option in "abc"} for member in range(2)}),)
    noted = examples()
    # Notes of 29 and 27 tokens outgrow a window of 32; one of 5 beside none does not.
    long, short = (GroupedExamples(pair, groups=group)
                   for pair in ((noted[0], noted[2]), (noted[3], noted[5])))
    assert "decision_unit" in fit.map({"example": short})
    apart = fit.map({"example": long})
    assert "decision_unit" not in apart and len(apart["decision_rows"]) == 2


PACKED_CONTEXT_SEEDS = {"table": 128, "backbone": 256}
"""Check 1's declared seeds for packed rows reading their own contexts against each row alone, by encoder,
sized paired (`seeds_for` with the packed windows as `dew_fn`) on armada: the table's under fp32 sums
(ledger 1698, rows/1698-fp32-sums at 9f68891e0d, size-ctx), 8 before them; the backbone's at base
7853cfdf6 (job 20261109062353-1a4d7ce5), whose 8 left the verdict's 40 Inconclusive (a pooled ratio's
interval [0.755, 2.255])."""


def packed_arms(packed: DecisionObjective, alone: DecisionObjective, rows):
    """The arms of check 1 between `packed`'s windows of `rows` and `alone`'s windows of one row each:
    Dew's, the reference's, the parameters' draw (every cross-attention gate open) and the float64 truth,
    each arm every row's real options' logits through the unmodified objective, keyed by the row's own
    tokens' text and its context's, each question owned by the row the packer placed it in
    (`question_rows`)."""
    from dew.decision.task import laid_out

    windows = {name: windows_of(objective, rows, count=12)
               for name, objective in (("packed", packed), ("alone", alone))}
    assert max(int(np.max(window["segments"])) for window in windows["packed"]) > 1, "no window packed two"

    def scored(objective, batches, variables) -> dict[str, np.ndarray]:
        """Each row's real options' logits, keyed by its tokens' text."""
        forward = COMPILED.setdefault((id(objective.model), "packed"), without_ynn(
            lambda variables, inputs: objective.model.logits(variables, inputs)))
        logits = {}
        for batch in batches:
            inputs = laid_out(batch, objective.bounds.segments, spans_in_context=objective.spans_in_context,
                              attention=objective.layout.attention)
            held = np.asarray(forward(variables, inputs))
            options, owners = np.asarray(batch["options"]), np.asarray(batch["question_rows"])
            # Rows that carry no context read none.
            contexts = np.asarray(batch.get("context_segments", np.zeros((len(owners), 0), np.int32)))
            context = np.asarray(batch.get("context", contexts))
            for window, segments in enumerate(np.asarray(batch["segments"])):
                for row in range(1, int(segments.max()) + 1):
                    own = np.asarray(batch["tokens"])[window][segments == row]
                    read = context[window][contexts[window] == row]
                    asked = owners[window] == row
                    key = b"\0".join(bytes(part.astype(np.uint8)) for part in (own, read))
                    real = options[window][asked]
                    logits.setdefault(key.decode(errors="replace"), held[window][asked][real])
        return dict(sorted(logits.items()))

    def arm(objective, batches):
        return as_given(lambda variables: scored(objective, batches, variables))

    # A draw of every parameter, the backbone's too: `init` alone takes the objective's held checkpoint
    # as its tree, the same at every seed (`Objective.init`), which leaves check 1 one rounding draw; a
    # draw's leaves are its arrays, unboxed of their axes, as a loaded tree's are.
    initialized = jax.jit(lambda key: unboxed(packed.init(key, variables={})))
    wide = copy.copy(alone)
    wide.model = DecisionModel(alone.model.backbone.clone(dtype=jnp.float64),
                               alone.model.head.clone(dtype=jnp.float64), alone.model.context_encoder)

    @as_given
    def truth(variables):
        with jax.enable_x64(new_val=True):
            return scored(wide, windows["alone"], widened(variables))

    def draw(seed):
        return (opened(initialized(jax.random.key(seed))),)

    return arm(packed, windows["packed"]), arm(alone, windows["alone"]), draw, truth


def assert_packed_as_alone(packed: DecisionObjective, alone: DecisionObjective, rows, what: str, seeds: int):
    """Check 1 (over drawn parameters, float64 truth): `packed`'s windows of `rows` decide each row's real
    options as `alone`'s windows of one row each do (`packed_arms`)."""
    dew, reference, draw, truth = packed_arms(packed, alone, rows)
    assert_as_exact(dew, reference, draw, what, seeds=seeds, truth_fn=truth)


def packed_contexts_read_as_alone(encoder: str):
    """Rows that carry a context, three packed a window, decide each row's options as the row does in a
    window of its own beside its context alone, through the unmodified objective, its cross-attention
    open (every gate 1): each row's tokens read its own context and no other row's, the real options'
    logits on par from float64 (check 1, over drawn parameters)."""
    packed, alone = conditioned(encoder=encoder, segments=3), conditioned(encoder=encoder, segments=1)
    assert_packed_as_alone(packed, alone, examples()[:12], f"packed contexts by {encoder}",
                           PACKED_CONTEXT_SEEDS[encoder])


@attending("reference")
@pytest.mark.check1_shards(seeds=128, per_task=32)
def test_packed_rows_each_read_their_own_context_through_the_token_table_as_each_does_alone():
    """Packed rows read their own contexts through the token table as each row alone does
    (`packed_contexts_read_as_alone`), its 128 seeds drawn by CI's map tasks (`check1_shards`)."""
    packed_contexts_read_as_alone("table")


@attending("reference")
@pytest.mark.check1_shards(seeds=256, per_task=32)
def test_packed_rows_each_read_their_own_context_through_the_backbone_as_each_does_alone():
    """Packed rows read their own contexts through the backbone's own pass over each context as its own
    document as each row alone does (`packed_contexts_read_as_alone`), its 256 seeds drawn by CI's map
    tasks (`check1_shards`)."""
    packed_contexts_read_as_alone("backbone")


@attending("reference")
def test_a_backbone_with_fresh_context_layers_scores_every_row_as_its_source_does():
    """The cross-attention blocks start from zero gates, so a decision over the backbone with them
    answers a row, with its context or without, as the backbone alone does: on par from float64 (check
    1), the two being programs of their own. The blocks train and the source's weights are held
    frozen."""
    source = Pretrained.load(LLAMA, dtype="float32")
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

    @as_given
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


@attending("reference")
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


@attending("reference")
def test_a_decisions_context_is_its_inputs_condition_and_its_check_reads_it():
    """A decision whose backbone reads a context holds it as its inputs' condition (`InputSpec`,
    `DecisionModel.context_condition`): the backbone's token table as the run says, or its own pass where
    rows read their schema in context; a batch missing its `context` field is refused by the check every
    objective's batches meet (`InputSpec.check`); one whose backbone reads none holds none."""
    from dew.decision.model import BackboneContext

    for encoder in ("table", "backbone"):
        held = conditioned(16, (0,), encoder).inputs.conditions
        assert list(held) == ["context"] and held["context"].encoder == BackboneContext(encoder)
    objective = conditioned(16, (0,))
    batch = next(iter(decision_rows(examples(), 8, objective=objective.data()).train(DataPartition())))
    objective.inputs.check(batch)
    with pytest.raises(ValueError):
        objective.inputs.check({name: value for name, value in batch.items() if name != "context"})
    source = Pretrained.load(LLAMA, dtype="float32")
    plain = DecisionObjective(source, tokenizer=ByteTokenizer(), specials=SPECIALS)
    assert dict(plain.inputs.conditions) == {}


@dataclasses.dataclass(frozen=True, eq=False)
class Captionless(CharTable):
    """A character table that declares it reads no captions (`ConditionEncoder.reads_captions`)."""

    reads_captions = False


@dataclasses.dataclass(frozen=True, eq=False)
class Conditioning(CharTable):
    """A character table that declares a denoiser's conditioning keyword (`ConditionEncoder.keyword`)."""

    keyword = "conditioning"


@dataclasses.dataclass(frozen=True, eq=False)
class Wide(CharTable):
    """A character table whose states are the tiny Laya backbone's width, 128 features."""

    @classmethod
    def from_pretrained(cls, checkpoint: str = "char_table", **settings) -> "Wide":
        return super().from_pretrained(checkpoint, **{"features": 128, **settings})


def configured(tokens: int = 12, *, trained: bool = False) -> DecisionObjective:
    """A decision over the tiny Llama whose context is encoded by a configured encoder (`CharTable`, a
    character table of 16 features) in place of the backbone's own, read by fresh cross-attention before
    layer 0 at that width, its windows packing rows as a decision's do."""

    table = CharTable.from_pretrained(tokens=tokens, features=16)
    source = Pretrained.load(LLAMA, dtype="float32")
    blocks = CrossAttention((0,), tokens, context_features=16)
    return DecisionObjective(source.with_cross_attention(blocks, key=0), tokenizer=ByteTokenizer(),
                             specials=SPECIALS, context_encoder=table, context_trained=trained)


@attending("reference")
def test_a_context_read_by_a_configured_encoder_trains_saves_and_resumes_exactly(tmp_path):
    """A decision whose context a configured encoder reads (`CharTable`, any `ConditionEncoder`): its
    rows carry the encoder's tokens of the context, its model reads the encoder's states, a step counts
    the encoder's work; a run trains, saves, and its task loads the encoder from the run's record and
    weights and answers as the trained task does, to the bit; a resume reads on exactly. A run's context
    was only ever the backbone's own (6141)."""
    from dew.batch import FROZEN
    from dew.checkpoints import Checkpoints

    objective = configured()
    assert isinstance(objective.inputs.conditions["context"].encoder, CharTable)
    data = decision_rows(examples(), 8, objective=objective.data())
    batch = next(iter(data.train(DataPartition())))
    codes = CharTable.from_pretrained(tokens=12, features=16).tokenize([note for *_, note in ROWS if note])
    notes = {tuple(ids[mask != 0].tolist()) for ids, mask in zip(codes["input_ids"], codes["attention_mask"],
                                                                  strict=True)}
    # Each packed row's context its encoder's tokens of its note (`context_segments`, a row's document).
    context, owners = np.asarray(batch["context"]), np.asarray(batch["context_segments"])
    held = {tuple(row[owned == document].tolist()) for row, owned in zip(context, owners, strict=True)
            for document in set(owned.tolist()) - {0}}
    assert held and held <= notes
    directory = str(tmp_path / "run")
    trainer = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=Checkpoints(directory, keep=2))
    state = trainer.fit(data, steps=2, log_every=1, checkpoint_every=1)
    trainer.checkpoints.wait()
    trained = objective.pipeline(state)
    loaded = Decide.from_run(directory)
    assert isinstance(loaded.model.context.encoder, CharTable)
    for context in ("a first note", "ten"):
        got = loaded("one two three", QUESTION, context=context)["pick"].probabilities
        want = trained("one two three", QUESTION, context=context)["pick"].probabilities
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    again = Trainer(configured(), optax.adam(1e-3), key=0, checkpoints=Checkpoints(directory, keep=2),
                    advanced=TrainerAdvanced(data_resume="exact"))
    resumed = again.fit(dataclasses.replace(data, resume="exact"), steps=3, log_every=1, checkpoint_every=1)
    # Every weight, the encoder's among them, and the optimizer's state as a run of three steps leaves them.
    whole = Trainer(configured(), optax.adam(1e-3), key=0).fit(data, steps=3, log_every=1)
    # The encoder's weights held frozen, as a configured encoder's are unless trained (`context_trained`).
    assert "context" in resumed.variables[FROZEN]
    for got, want in zip(jax.tree.leaves((resumed.variables, resumed.opt_state)),
                         jax.tree.leaves((whole.variables, whole.opt_state)), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # A note longer than the encoder's own width is cut by its tokenization, and the cut is counted.
    note = "a note longer than twelve characters"
    [row] = loaded._encoded("one two three", QUESTION, context=note)
    assert (row.context_given, row.cuts["context"]) == (1 + len(note), 1 + len(note) - 12)


@attending("reference")
def test_a_clip_context_encoder_trains_serves_and_counts_each_rows_context_alone():
    """A decision whose context a pretrained text tower reads (`CLIPText`, 77 tokens a text): a window of
    hundreds of context positions reaches it a row's context at a time, each at the tower's own width, so a
    step trains and a request is answered; its work is the tower's over each row's context alone
    (`conditioning_flops`), not over a window's contexts together. A window's whole context reached the
    tower past its 77 positions, and its work was counted over the window (theme 22's head review)."""
    from test_text_encoders import TINY

    from dew.interop.encoders import CLIPText

    tower = CLIPText.from_pretrained(str(TINY))
    width = tower.conditioning_flops(tower.tokenize([""])).features
    source = Pretrained.load(LLAMA, dtype="float32")
    blocks = CrossAttention((0,), 16, context_features=width)
    objective = DecisionObjective(source.with_cross_attention(blocks, key=0), tokenizer=ByteTokenizer(),
                                  specials=SPECIALS, context_encoder=tower)
    data = decision_rows(examples(), 8, objective=objective.data())
    batch = next(iter(data.train(DataPartition())))
    assert np.asarray(batch["context"]).shape[1] > 77
    owners = np.asarray(batch["context_segments"])
    alone = sum(tower.conditioning_flops(tower.tokenize([""]) | {
        "attention_mask": (np.arange(77)[None, :] < np.sum(row == document)).astype(np.int32)}).flops
        for row in owners for document in set(row.tolist()) - {0})
    assert alone > 0 and objective._encoder_flops(batch) == alone
    # Frozen, as it is unless trained, the tower runs forward alone: a step counts its work once where a
    # trained one counts three passes.
    trained = DecisionObjective(source.with_cross_attention(blocks, key=0), tokenizer=ByteTokenizer(),
                                specials=SPECIALS, context_encoder=tower, context_trained=True)
    assert trained.training_flops(None, batch) - objective.training_flops(None, batch) == 2 * alone
    state = Trainer(objective, optax.adam(1e-3), key=0).fit(data, steps=1, log_every=1)
    task = objective.pipeline(state)
    assert set(task("one two three", QUESTION, context="a first note")) == {"pick"}


@attending("xla")
def test_a_configured_context_encoders_bundle_carries_it_and_its_trained_weights():
    """A task whose context a configured encoder reads is a bundle a run starts from
    (`PretrainedDecision.of`) carrying the encoder bound to its trained weights, so the task it declares
    again (`Decide.released`) answers as the task, to the bit. The bundle dropped the encoder, and a run
    from it read its contexts with the backbone's table (theme 22's head review)."""
    from dew.data.text import HFTokenizer
    from dew.decision.sources import PretrainedDecision

    directory = LLAMA.parent / "qwen38-dense-tiny"
    source = Pretrained.load(directory, dtype="float32")
    tokenizer = HFTokenizer(str(directory), local_files_only=True)
    layout = SegmentLayout.joint_row(max_len=1024)
    blocks = CrossAttention((0,), 12, context_features=16)
    objective = DecisionObjective(source.with_cross_attention(blocks, key=0), tokenizer=tokenizer,
                                  layout=layout, context_encoder=CharTable.from_pretrained(tokens=12))
    drawn = opened(objective.init(jax.random.key(0)))
    # Weights the encoder was not drawn with, as a run's training leaves them.
    trained = {**drawn, "params": {**drawn["params"], "context": jax.tree.map(
        lambda leaf: leaf + 1.0, drawn["params"]["context"])}}
    task = Decide(objective.model, trained, layout, tokenizer, objective.specials, processor=source.processor)
    bundle = PretrainedDecision.of(task)
    again = Decide.released(bundle)
    for context in ("a first note", "ten"):
        np.testing.assert_array_equal(
            np.asarray(again("one two three", QUESTION, context=context)["pick"].probabilities),
            np.asarray(task("one two three", QUESTION, context=context)["pick"].probabilities))
    # A load's transform of the bundle's trees (`map_variables`: a placement, a cast, shapes alone) reaches
    # the encoder's weights as the backbone's and the head's, which the task declared again reads: they were
    # left as they were (row 7471).
    halved = Decide.released(bundle.map_variables(lambda tree: jax.tree.map(lambda leaf: leaf / 2, tree)))
    for name in ("backbone", "head", "context"):
        for got, want in zip(jax.tree.leaves(part(halved.variables, name)),
                             jax.tree.leaves(part(again.variables, name)), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want) / 2)


@attending("reference")
def test_a_configured_context_encoder_beside_a_schema_read_in_context_is_refused():
    """A layout that reads its schema as the row's context has the backbone's own pass encode it, the
    example's context after it in the backbone's tokens, so a configured encoder would read nothing: the
    objective is refused as it is built, where it tokenized contexts in the encoder's vocabulary for the
    backbone to read (theme 22's head review)."""
    from test_decision_schema_stream import LAYOUT as IN_CONTEXT

    source = Pretrained.load(LLAMA, dtype="float32")
    blocks = CrossAttention((0,), 64, context_features=16)
    with pytest.raises(ValueError):
        DecisionObjective(source.with_cross_attention(blocks, key=0), tokenizer=ByteTokenizer(),
                          specials=SPECIALS, layout=IN_CONTEXT, context_encoder=CharTable.from_pretrained())


def test_a_configured_encoder_reads_each_packed_rows_context_alone():
    """A window's rows' contexts, packed end to end, reach a configured encoder each alone
    (`dew.decision.model._apart`): an encoder whose states mix its tokens (a running sum) gives each token
    the state its document alone gives it, zeros where no document is, as the backbone's own pass reads
    each document alone. An encoder over the window's whole context mixed its rows'."""
    from dew.decision.model import _apart

    def running(tokens, lengths):
        real = jnp.arange(tokens.shape[1])[None, :] < lengths[:, None]
        return jnp.cumsum(jnp.where(real, tokens, 0), axis=1)[:, :, None].astype(jnp.float32)

    tokens = jnp.asarray([[3, 4, 5, 6, 7, 0], [8, 9, 1, 0, 0, 0]], jnp.int32)
    segments = jnp.asarray([[1, 1, 2, 2, 2, 0], [1, 2, 2, 0, 0, 0]], jnp.int32)
    states = np.asarray(_apart(running, tokens, segments, 3, 4))[:, :, 0]
    np.testing.assert_array_equal(states, [[3, 7, 5, 11, 18, 0], [8, 9, 10, 0, 0, 0]])


@attending("reference")
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


@attending("reference")
def test_a_batch_answers_each_request_from_its_own_context():
    """With the gates open, a request's answer reads its context: another context, or none, answers it
    otherwise. A call over requests lays each out with its own context (`Decide.rows`, which it
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

    @as_given
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


@attending("xla")
def test_a_request_with_images_reads_its_context_beside_them():
    """Over a backbone that reads images (a `MultimodalTransformer`), the blocks attach to its language
    model. With their gates shut a request with images and a context is answered as with its images
    alone, on par from float64 (check 1); with them open its context moves the answer."""
    from PIL import Image

    from dew.data.text import HFTokenizer

    directory = LLAMA.parent / "qwen38-dense-tiny"
    source = Pretrained.load(directory, dtype="float32")
    tokenizer = HFTokenizer(str(directory), local_files_only=True)
    layout = SegmentLayout.joint_row(max_len=1024)
    objective = DecisionObjective(source.with_cross_attention(CrossAttention((0,), 32), key=0),
                                  tokenizer=tokenizer, layout=layout)
    variables = objective.init(jax.random.key(0))
    images = [Image.fromarray(array) for array in np.load(directory / "images.npy")]

    def task(gates) -> Decide:
        return Decide(objective.model, gates, layout, tokenizer, objective.specials,
                      processor=source.processor)

    shut, wide = task(variables), {}

    @as_given
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
    # A caller of a context past the room, with images, is told it was cut, as one without them is.
    with pytest.warns(UserWarning):
        reading("one two three", QUESTION, images=images, context="a long note " * 20)


@attending("xla")
def test_a_context_reading_tasks_bundle_carries_its_cross_attention_and_refuses_to_export(tmp_path):
    """A task whose backbone reads its decision context through cross-attention is a bundle a run starts
    from (`PretrainedDecision.of`), its backbone its source family's carrying the cross-attention, so the
    task it declares again (`Decide.released`) is the task, model and every weight, its context encoded
    as the task encodes it (row 4797: a backbone-encoded context came back read through the token table);
    its release, which no family writes, is refused as it is written (row 4776's review: the bundle was
    refused as it was made, and a run from such a run's native release could not start)."""
    from dew.data.text import HFTokenizer
    from dew.decision.sources import PretrainedDecision

    directory = LLAMA.parent / "qwen38-dense-tiny"
    source = Pretrained.load(directory, dtype="float32")
    tokenizer = HFTokenizer(str(directory), local_files_only=True)
    layout = SegmentLayout.joint_row(max_len=1024)
    objective = DecisionObjective(source.with_cross_attention(CrossAttention((0,), 32), key=0),
                                  tokenizer=tokenizer, layout=layout, context_encoder="backbone")
    task = Decide(objective.model, opened(objective.init(jax.random.key(0))), layout, tokenizer,
                  objective.specials, processor=source.processor)
    bundle = PretrainedDecision.of(task)
    again = Decide.released(bundle)
    assert again.model == task.model
    assert jax.tree.structure(again.variables) == jax.tree.structure(task.variables)
    for read, held in zip(jax.tree.leaves(again.variables), jax.tree.leaves(task.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(read), np.asarray(held), strict=True)
    with pytest.raises(ValueError):
        bundle.save(tmp_path / "saved")
    assert not (tmp_path / "saved" / "config.json").exists()


@attending("reference")
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


@attending("reference")
def test_a_context_past_its_cross_attentions_room_is_cut_and_its_caller_told():
    """A context of more tokens than the cross-attention reads keeps its first ones, as a context beside a
    schema does (one policy, `with_context`): a task answers from them, a served answer states the
    request's context tokens and those it read, whatever rows its questions take, and a run's data counts
    the cut (`data/cut/.../context`). It was refused, and refused a run's data inside its fitting (train
    985's review, F5); a served answer of three question rows summed their cuts, reporting 12 kept where
    every row read 16, and one that forbids cuts answered from a cut context (train 986's review); a
    Python caller's context was cut silently (train 990's review)."""
    from dew.decision.rows import Fit

    objective = conditioned(16, (0,))
    task = Decide(objective.model, objective.init(jax.random.key(0)), objective.layout, objective.tokenizer,
                  objective.specials)
    exact, over = "x" * 16, "x" * 17 + "y"
    [whole] = task._encoded("one two", QUESTION, context=exact)
    [cut] = task._encoded("one two", QUESTION, context=over)
    assert whole.context == cut.context and cut.cuts == {"context": 2}
    # A caller of the call is told too, as a served caller is in its answer.
    with pytest.warns(UserWarning):
        read = task("one two", QUESTION, context=over)["pick"].probabilities.tolist()
    assert read == task("one two", QUESTION, context=exact)["pick"].probabilities.tolist()
    wire = {name: {"type": "choice", "instructions": "Which one?", "criteria": ["a", "b", "c"]}
            for name in ("pick", "other", "third")}
    served = task.systemone({"state": "one two", "questions": wire, "context": over})
    assert (served["usage"]["context_tokens"], served["usage"]["context_kept"]) == (18, 16)
    # A caller that forbids every cut is refused one of its context, as one of its state.
    with pytest.raises(ValueError):
        task.systemone({"state": "one two", "questions": wire, "context": over}, whole=True)
    fit = Fit(objective.layout, objective.tokenizer, objective.specials, objective.model.head,
              objective.context_length)
    fitted = fit.map({"example": Example("one two", QUESTION, {"pick": "a"}, context=over)})
    assert isinstance(fitted, dict) and fitted["_dew_cut"] == {"context": (1, 2)}


@attending("reference")
def test_a_backbone_without_cross_attention_refuses_a_requests_context():
    """A task over a backbone with no cross-attention refuses a request that brings a context, by a call
    and over the wire (its own refusal, ValueError), rather than answering it without the context; the
    same request without one is answered (docs/design/training.md, context)."""
    source = Pretrained.load(LLAMA, dtype="float32")
    objective = DecisionObjective(source, tokenizer=ByteTokenizer(), specials=SPECIALS)
    task = Decide(objective.model, objective.init(jax.random.key(0)), objective.layout, objective.tokenizer,
                  objective.specials)
    assert isinstance(task("one two", QUESTION)["pick"], ChoiceAnswer)
    with pytest.raises(ValueError):
        task("one two", QUESTION, context="a first note")
    wire = {"pick": {"type": "choice", "instructions": "Which one?", "criteria": ["a", "b", "c"]}}
    refused, answered = task.systemone([{"state": "one two", "questions": wire, "context": "a first note"},
                                        {"state": "one two", "questions": wire}])
    assert isinstance(refused, ValueError) and not isinstance(answered, Exception)


@attending("reference")
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


@attending("reference")
def test_a_configured_context_encoder_is_frozen_unless_trained_and_counted_as_it_runs():
    """A configured context encoder's weights are held frozen (`FROZEN`) and run forward alone, as a
    diffusion run's text towers do: a step leaves them as drawn; trained (`context_trained=True`) they are
    parameters a step moves. They trained always, their Adam state beside them (theme 22's review)."""
    from dew.batch import FROZEN

    for trained in (False, True):
        objective = configured(trained=trained)
        data = decision_rows(examples(), 8, objective=objective.data())
        drawn = objective.model.context.encoder.params
        # Two steps: the cross-attention's gate opens from zero on the first, so the second reaches the
        # encoder.
        state = Trainer(objective, optax.adam(1e-1), key=0).fit(data, steps=2, log_every=1)
        held = state.variables["params" if trained else FROZEN]["context"]
        moved = [bool(np.any(np.asarray(after) != np.asarray(before)))
                 for after, before in zip(jax.tree.leaves(held), jax.tree.leaves(drawn), strict=True)]
        assert any(moved) == trained
        assert "context" not in state.variables.get(FROZEN if trained else "params", {})


def test_a_runs_configured_context_encoder_computes_in_its_dtype_and_reads_text(tmp_path):
    """A decision run's context's text encoder (`CrossContext.text`) builds in the run's compute dtype, as a
    diffusion run's text condition is, a pretrained tower (`clip_text`) among them, and trains a step with it
    frozen; an encoder that reads no text into the states context layers read (an audio tower, a denoiser's
    conditioner) and a diffusion run's field or unconditional text are refused. The run built the tower in
    float32 and took any registered encoder, which failed at its first trace (theme 22's review). Its task
    loaded computes its encoder as the run did, or in the dtype the load asks for."""
    from run_kinds import decision, trained
    from test_text_encoders import TINY

    from dew.batch import FROZEN
    from dew.inputs.spec import TextCondition
    from dew.registry import ModelConfig

    text = TextCondition(encoder="clip_text", checkpoint=str(TINY))
    context = CrossContext(layers=(0,), length=32, text=text)
    run = dataclasses.replace(decision(tmp_path, drawn=False), context=context, calibrate=False,
                              model=ModelConfig(
                                  functools.partial(CausalTransformer, dtype="bfloat16")))
    objective = run.prepare().objective
    assert jnp.dtype(objective.model.context.encoder.dtype) == jnp.dtype("bfloat16")
    state, _ = trained(run, tmp_path, "clip", trainer={"checkpoint_final": True})
    assert "context" in state.variables[FROZEN] and "context" not in state.variables["params"]
    # Loaded, the task's encoder computes as the run's did, or in the dtype the load asks for, as its
    # backbone does: a load's dtype bound the backbone and head alone (row 7611).
    for asked, computed in ((None, "bfloat16"), ("float32", "float32")):
        loaded = Decide.from_run(str(tmp_path / "runs" / "clip"), dtype=asked)
        assert jnp.dtype(loaded.model.context.encoder.dtype) == jnp.dtype(computed)
        assert jnp.dtype(loaded.model.backbone.dtype) == jnp.dtype(computed)
    # A character table is taken; one differing from it in its declarations alone (`reads_captions`,
    # `keyword`) is refused, as an audio tower and a denoiser's conditioner are.
    table = TextCondition(encoder="char_table", checkpoint="char_table")
    tabled = dataclasses.replace(run, context=dataclasses.replace(context, text=table))
    assert isinstance(tabled.prepare().objective.model.context.encoder, CharTable)
    for refused in (dataclasses.replace(text, field="caption"), dataclasses.replace(text, unconditional="x"),
                    dataclasses.replace(table, encoder=f"{__name__}:Captionless"),
                    dataclasses.replace(table, encoder=f"{__name__}:Conditioning")):
        with pytest.raises(ValueError):
            dataclasses.replace(context, text=refused)


def test_a_run_from_a_saved_run_holds_its_configured_encoders_weights_frozen_or_trained(tmp_path):
    """A run started from a saved run whose task configured a context encoder (`--pretrained <run>`,
    `PretrainedDecision.context`) starts from that run's encoder weights, which its training moved from the
    encoder's checkpoint, and holds them as it says: frozen (`FROZEN`) by default, trained (`params`) under
    `--context-text-trained`, which such a start takes without `--context-text`; a run whose start carries
    none refuses it. The started tree is read after a step at a learning rate of zero, which moves no
    weight. A run from a release could not train its encoder, and the cell that stood for it read a tree's
    structure alone (train 1003's review, F1)."""
    from run_kinds import decision, trained
    from test_text_encoders import TINY

    from dew.config import SourceOptions
    from dew.inputs.spec import TextCondition
    from dew.optim import OptimConfig

    text = TextCondition(encoder="clip_text", checkpoint=str(TINY))
    first = dataclasses.replace(decision(tmp_path, drawn=False), context=CrossContext(layers=(0,), length=32,
                                                                                       text=text),
                                context_text_trained=True, calibrate=False, optim=OptimConfig(lr=1e-2))
    # Two steps: the cross-attention's gate opens from zero on the first, so the second reaches the encoder.
    state, _ = trained(first, tmp_path, "first", trainer={"steps": 2,
                                                          "checkpoint_final": True})
    held = jax.tree.leaves(state.variables["params"]["context"])
    checkpoint = jax.tree.leaves(text.build(dtype="float32").encoder.params)
    assert any(np.any(np.asarray(one) != np.asarray(two)) for one, two in zip(held, checkpoint, strict=True))
    start = dataclasses.replace(decision(tmp_path, drawn=False), pretrained=str(tmp_path / "runs" / "first"),
                                source=SourceOptions(step="latest"), calibrate=False,
                                optim=OptimConfig(lr=0.0))
    for trains in (False, True):
        second, _ = trained(dataclasses.replace(start, context_text_trained=trains), tmp_path,
                            f"second-{trains}")
        kept, other = ("params", FROZEN) if trains else (FROZEN, "params")
        for got, want in zip(jax.tree.leaves(second.variables[kept]["context"]), held, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert "context" not in second.variables.get(other, {})
    with pytest.raises(ValueError):
        dataclasses.replace(decision(tmp_path, drawn=False), context_text_trained=True).prepare()


def test_a_shared_cross_attention_reads_a_configured_encoder_of_the_backbones_width(tmp_path):
    """A shared context form (`CrossContext.form`) reads a configured encoder's states in the layer's own
    attention, at the backbone's width: an encoder of that width (`Wide`, 128 features as the tiny Laya
    backbone) trains a step, and one of another width (char_table's 16) is refused by the layer's own width
    check. Every configured encoder was refused under the shared form (row 7613)."""
    from run_kinds import decision, trained

    from dew.inputs.spec import TextCondition

    context = CrossContext(layers=(0,), length=32, form="shared",
                           text=TextCondition(encoder=f"{__name__}:Wide", checkpoint="char_table"))
    run = dataclasses.replace(decision(tmp_path, drawn=False), context=context, calibrate=False)
    state, _ = trained(run, tmp_path, "shared")
    assert "context" in state.variables[FROZEN]
    table = TextCondition(encoder="char_table", checkpoint="char_table")
    narrow = dataclasses.replace(run, context=dataclasses.replace(context, text=table))
    with pytest.raises(ValueError):
        trained(narrow, tmp_path, "narrow")


@pytest.mark.mesh(devices=8)
def test_a_configured_encoder_is_placed_by_its_own_axes_frozen_or_trained(tmp_path):
    """A run's configured context encoder (a CLIP tower) is placed on a mesh with a tensor axis by the axes
    its towers declare (`DecisionObjective.tower_axes`), in the collection the run holds it in, frozen or
    trained, some of its weights split over tensor as a diffusion run's frozen towers are: its weights are
    no module's draw, and the layout replicated them (the base review of fcc9f98cc, F3)."""
    from placement_cases import LAYOUT, MIN_SHARD
    from run_kinds import decision
    from test_text_encoders import TINY

    from dew.distributed.placement import Layout as Placement
    from dew.inputs.spec import TextCondition
    from dew.mesh import TENSOR_AXIS, mesh_axes

    text = TextCondition(encoder="clip_text", checkpoint=str(TINY))
    for trains in (False, True):
        run = dataclasses.replace(decision(tmp_path, drawn=False),
                                  context=CrossContext(layers=(0,), length=32, text=text),
                                  context_text_trained=trains, calibrate=False)
        objective = run.prepare().objective
        trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), mesh=LAYOUT,
                          advanced=TrainerAdvanced(layout=Placement(min_shard=MIN_SHARD)), tracker=None)
        trainer.device_mesh = LAYOUT.build(jax.devices()[:8])
        placed = trainer.shardings(jax.eval_shape(trainer.initial_state)).variables
        held = placed["params" if trains else FROZEN]["context"]
        assert any(TENSOR_AXIS in mesh_axes(entry) for sharding in jax.tree.leaves(held)
                   for entry in sharding.spec)
