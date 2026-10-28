"""A run's validation reports per split what a pass's decisions are (row 2034): its macro F1 over the
classes its questions ask, its Brier score and the count of questions it scored, and, where asked, every
batch's decisions, so a pass's metrics pool exactly offline across batches and splits."""

import jax
import numpy as np

from dew.artifacts import Decisions
from dew.data import Memory
from dew.data.compile import held
from dew.data.text import ByteTokenizer
from dew.decision import Choice, DecisionObjective, Example, MacroF1, ScoredQuestions, decisions
from dew.decision.layout import Specials, StateFirstLayout
from dew.decision.rows import class_id
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.training.evaluation import Evaluation

SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)


def drawn(seed: int, rows: int) -> tuple[Decisions, np.ndarray, np.ndarray]:
    """`rows` rows of two questions, `a` over options x, y, z and `b` over x, y (the same keys, other
    classes), some unanswered, with each scored question's predicted and gold class names."""
    rng = np.random.default_rng(seed)
    questions = (("a", ("x", "y", "z")), ("b", ("x", "y")))
    probabilities = rng.dirichlet(np.ones(3), (rows, 2))
    options = np.zeros((rows, 2, 3), bool)
    classes = np.zeros((rows, 2, 3, 2), np.int32)
    for slot, (name, keys) in enumerate(questions):
        options[:, slot, :len(keys)] = True
        classes[:, slot, :len(keys)] = [class_id(name, key) for key in keys]
    probabilities = np.where(options, probabilities, 0.0)
    labels = np.stack([rng.integers(0, 3, rows), rng.integers(0, 2, rows)], axis=1)
    scored = rng.random((rows, 2)) < 0.8
    names = np.array([[f"{name}/{key}" for key in (*keys, "")[:3]] for name, keys in questions])
    real = np.where(options, probabilities, -1.0)
    predicted = np.take_along_axis(names[None].repeat(rows, 0), real.argmax(-1)[..., None], -1)[..., 0]
    gold = np.take_along_axis(names[None].repeat(rows, 0), labels[..., None], -1)[..., 0]
    artifact = Decisions(probabilities=probabilities, options=options, labels=labels,
                         targets=np.eye(3)[labels], ordinal=np.zeros((rows, 2), bool), scored=scored,
                         classes=classes)
    return artifact, predicted[scored], gold[scored]


def test_macro_f1_pools_each_class_over_a_pass_and_keeps_questions_classes_apart():
    """Macro F1 over three batches merged is each class's F1 over the pooled questions, averaged over the
    classes they are or are answered as, two questions' options of the same key two classes; the count
    of scored questions is the pool's."""
    metric, count = MacroF1(), ScoredQuestions()
    batches = [drawn(seed, rows) for seed, rows in ((0, 40), (1, 25), (2, 33))]
    accumulated = metric(batches[0][0], {})
    counted = count(batches[0][0], {})
    for artifact, _, _ in batches[1:]:
        accumulated = metric.merge(accumulated, metric(artifact, {}))
        counted = count.merge(counted, count(artifact, {}))
    predicted = np.concatenate([guess for _, guess, _ in batches])
    gold = np.concatenate([truth for _, _, truth in batches])

    def key(name: str) -> int:
        high, low = class_id(*name.split("/"))
        return (high << 32) | (low & 0xFFFFFFFF)

    # Each class's true and false positives and its misses, counted over the pooled questions.
    pooled = {key(name): [int(np.sum((predicted == name) & (gold == name))),
                          int(np.sum((predicted == name) & (gold != name))),
                          int(np.sum((predicted != name) & (gold == name)))]
              for name in set(predicted) | set(gold)}
    assert {name: counts.tolist() for name, counts in accumulated.items()} == pooled and len(pooled) == 5
    assert metric.finalize(accumulated) == np.mean(
        [2 * true / (2 * true + false + missed) for true, false, missed in accumulated.values()])
    assert count.finalize(counted) == len(gold)


def test_a_pass_writes_every_batchs_decisions_where_asked(tmp_path):
    """A pass asked to keep its decisions writes every batch's, its real rows alone, so the pooled files
    hold one row a record of the split and score as the pass's own metrics do."""
    backbone = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2, mlp_features=48,
                                 max_seq_len=128, attention_impl="xla")
    objective = DecisionObjective(backbone, tokenizer=ByteTokenizer(), specials=SPECIALS,
                                  layout=StateFirstLayout(max_len=96), shuffle_options=False)
    question = Choice("Which?", ["red", "green", "blue"])
    colours = ("red", "green", "blue")
    examples = [{"example": Example(f"state {index}", {"q": question}, {"q": colours[index % 3]})}
                for index in range(10)]
    # A batch every device count the suite runs on divides, its last short of it.
    reader = held(Memory(examples).pipe(decisions).read("memory").batch(8).workers(0),
                  objective=objective.data())
    variables = objective.init(jax.random.key(0))
    evaluation = Evaluation.run(objective, variables, reader, key=0, metrics=[ScoredQuestions(), MacroF1()],
                                split="val", scored_to=tmp_path)
    files = sorted(tmp_path.glob("val-*.npz"))
    assert [path.name for path in files] == ["val-00000.npz", "val-00001.npz"]
    pooled = [np.load(path) for path in files]
    assert sum(int(np.sum(arrays["scored"])) for arrays in pooled) == 10
    assert evaluation.scores["val/scored_questions"] == 10
    classes = np.concatenate([arrays["classes"][arrays["scored"]] for arrays in pooled])
    assert len({tuple(lanes) for row in classes for lanes in row.tolist()}) == 3
