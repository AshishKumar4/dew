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
from dew.decision.layout import Specials, StateFirstLayout
from dew.decision.scoring import Brier, LogLoss, RankedProbability, Spherical
from dew.nn.backbones.causal_transformer import CausalTransformer

SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
LAYOUT = StateFirstLayout(max_len=96)


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
