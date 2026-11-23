"""A run's validation reads in its training's shapes (row 2012). Each validation split once bound its own
option width, measured from its own questions, so a run of many splits compiled a forward program a
width; the run's static widths are now measured over its training and every split (`run_widths`) and
bound into each before any compiles, so every split's batch has the shape training's has. A scoring
rule charges a batch at its fixed shape with its unanswered rows masked, one jitted program a shape,
where indexing the answered rows first compiled a program a count of them (a run of 101 differently
sized validation splits compiled 1,999 programs before and 12 after)."""

import numpy as np

from dew.artifacts import Decisions
from dew.data import Memory
from dew.data.compile import fitted
from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import Choice, DecisionObjective, Example, decisions
from dew.decision.layout import SegmentLayout, Specials
from dew.decision.scoring import Brier, LogLoss, RankedProbability, Spherical
from dew.nn.backbones.causal_transformer import CausalTransformer

SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
LAYOUT = SegmentLayout.state_first_row(max_len=96)


def asked(options: int, count: int) -> list[dict]:
    """`count` examples of one question of `options` options, each answered by its first."""
    question = Choice("Which?", [f"o{index}" for index in range(options)])
    return [{"example": Example(f"state {index}", {"q": question}, {"q": "o0"})} for index in range(count)]


def test_every_validation_split_reads_in_its_trainings_shapes():
    """A run training on a question of 3 options and validating on splits of 2 and 7 reads every split's
    batch in one shape, its option width the widest any of them asks, as training's batch is: one
    forward program for all of them."""
    backbone = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2, mlp_features=48,
                                 max_seq_len=128, attention_impl="xla")
    objective = DecisionObjective(backbone, tokenizer=ByteTokenizer(), specials=SPECIALS, layout=LAYOUT,
                                  shuffle_options=False)

    def chain(options: int, count: int):
        return Memory(asked(options, count)).pipe(decisions).read("memory").batch(4).workers(0)

    loader, readers = fitted(chain(3, 8), {"two": chain(2, 4), "seven": chain(7, 4)},
                             objective=objective.data())
    stream = loader.train(DataPartition())
    try:
        trained = next(stream)
    finally:
        stream.close()
    shapes = {}
    for split, reader in readers.items():
        opened = reader(DataPartition())
        try:
            shapes[split] = {name: np.shape(value) for name, value in next(iter(opened)).items()
                             if name in ("options", "tokens", "questions")}
        finally:
            opened.close()
    assert shapes["two"] == shapes["seven"] == {name: np.shape(trained[name]) for name in shapes["two"]}
    assert shapes["two"]["options"][-1] >= 7


def test_a_run_measures_its_widths_over_examples_it_reads_once_each():
    """A decision run over examples its read holds in memory, with no step that states their widths, has
    its objective measure them as it lays the examples out, over training and validation alike: the run
    binds the widest question either asks (7 options), and lays out each example once, the widths'
    measure and the run's read being one pass."""
    import collections

    encoded: collections.Counter = collections.Counter()

    class Counting(ByteTokenizer):
        def encode(self, text, *, add_special_tokens=True):
            encoded[text] += 1
            return super().encode(text, add_special_tokens=add_special_tokens)

    backbone = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2, mlp_features=48,
                                 max_seq_len=128, attention_impl="xla")
    objective = DecisionObjective(backbone, tokenizer=Counting(), specials=SPECIALS, layout=LAYOUT,
                                  shuffle_options=False)

    def chain(split: str, options: int, count: int):
        question = Choice("Which?", [f"o{index}" for index in range(options)])
        return Memory([{"example": Example(f"{split} state {index}", {"q": question}, {"q": "o0"})}
                       for index in range(count)]).batch(4)

    _, readers = fitted(chain("train", 3, 8), {"val": chain("val", 7, 4)}, objective=objective.data())
    states = [text for text in encoded if "state" in text]
    assert len(states) == 12 and all(encoded[text] == 1 for text in states)
    opened = readers["val"](DataPartition())
    try:
        assert np.shape(next(iter(opened))["options"])[-1] >= 7
    finally:
        opened.close()


def test_a_scoring_rules_total_reads_its_answered_rows_alone():
    """A rule's total over a batch is the same whatever its unanswered rows hold, NaN among them: they
    are masked at the batch's shape, never read; and two batches of one shape with different counts of
    answered rows each give their own answered rows' total."""
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet(np.ones(5), (4, 3))
    options = np.ones((4, 3, 5), bool)
    labels = rng.integers(0, 5, (4, 3))
    targets = np.eye(5)[labels]
    ordinal = np.arange(12).reshape(4, 3) % 2 == 0
    scored = np.arange(12).reshape(4, 3) % 3 != 0
    poisoned = np.where(scored[..., None], probabilities, np.nan)
    for rule in (LogLoss(), Brier(), Spherical(), RankedProbability()):
        clean = rule(Decisions(probabilities, options, labels, targets, ordinal, scored), {})
        dirty = rule(Decisions(poisoned, options, labels, targets, ordinal, scored), {})
        assert clean == dirty and np.isfinite(clean[0])
        fewer = scored & (np.arange(12).reshape(4, 3) < 6)
        alone = rule(Decisions(probabilities, options, labels, targets, ordinal, fewer), {})
        kept = rule(Decisions(np.where(fewer[..., None], probabilities, np.nan), options, labels, targets,
                              ordinal, fewer), {})
        assert alone == kept


def test_packed_validation_splits_read_in_one_shape_and_one_bound():
    """Packed validation splits whose windows hold 1 to 4 documents each, of 1 to 7 windows, read in the
    run's batch of 4 every batch the same: each marks its rows (`VALID_ROWS`), a split's whole batches and
    its short last one alike, and declares the most documents a window of any split holds
    (`Coordinates.max_segments`), so a run's evaluation compiles one program for them all, where each
    split's own most and its short batch's unmarked rows compiled one apiece (row 12474)."""
    from dew.batch import VALID_ROWS
    from dew.data.pipeline import ObjectiveData, Pipeline

    def split(windows: int, per_window: int) -> Pipeline:
        size = 9 // per_window
        return Pipeline(Memory(tuple({"text": np.arange(size, dtype=np.int32)}
                                     for _ in range(windows * per_window)))).pack(9)

    train = Pipeline(Memory(tuple({"text": np.arange(9, dtype=np.int32)} for _ in range(16)))).pack(9)
    train = train.batch(4)
    shapes = {(1, 1), (3, 2), (7, 3), (4, 4), (5, 1)}
    _, readers = fitted(train, {f"{windows}x{per}": split(windows, per) for windows, per in shapes},
                        objective=ObjectiveData(), steps=2)
    assert readers is not None
    seen = set()
    for reader in readers.values():
        opened = reader(DataPartition())
        try:
            for batch in opened:
                laid = batch["text_coordinates"]
                seen.add((tuple(sorted(batch)), np.shape(batch["text"]), laid.max_segments))
                assert VALID_ROWS in batch, sorted(batch)
        finally:
            opened.close()
    assert len(seen) == 1 and next(iter(seen))[2] == 4, seen
