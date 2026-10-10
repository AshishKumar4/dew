"""Decision examples share the indexed Grain path, including fit and resume."""

import dataclasses
import json

import numpy as np
import pytest
from etils import epath

from dew.data import DataPartition
from dew.data.dataset import Loading
from dew.data.text import ByteTokenizer
from dew.decision import Choice, DecisionMixture, DecisionObjective, Example, JointLayout, Noul, Specials
from dew.decision.layout import MarkerLayout, StateFirstLayout
from dew.nn.backbones.causal_transformer import CausalTransformer

TOKENIZER = ByteTokenizer()
SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
LAYOUT = JointLayout(max_len=1024, max_state_tokens=32)
QUESTION = Choice("Which team?", {"billing": "payments", "technical": "outages"})


def _examples(count=8):
    return [Example(f"ticket {index}", {"team": QUESTION}, {"team": index % 2}) for index in range(count)]


def _write(root, sets):
    from dew.decision.data import write_examples

    root.mkdir(parents=True, exist_ok=True)
    recorded = {name: write_examples(root, name, examples, shard_size=3)
                for name, (_, examples) in sets.items()}
    made = {"weights": {name: weight for name, (weight, _) in sets.items()}, "sets": recorded}
    (root / "mixture.json").write_text(json.dumps(made))
    return DecisionMixture(str(root))


def _read(mixture, layout=LAYOUT):
    return mixture.read(layout=layout, tokenizer=TOKENIZER, specials=SPECIALS)


def _objective(layout=LAYOUT, *, bucket=None):
    backbone = CausalTransformer(vocab_size=256, emb_features=8, num_layers=1, num_heads=2,
                                 mlp_features=16, max_seq_len=2048)
    return DecisionObjective(backbone, tokenizer=TOKENIZER, specials=SPECIALS, layout=layout,
                             shuffle_options=False, bucket=bucket)


def test_prepared_records_keep_json_and_held_bytes_and_declared_shape(tmp_path):
    rows = _examples()
    rows.append(Example("unlabelled", {"team": QUESTION}))
    rows.append(Example("multi", {"team": QUESTION, "urgent": Noul("Urgent?")}, {"urgent": "true"}))
    mixture = _write(tmp_path, {"tickets": (1.0, rows)})
    held_bytes = b'{"state":"held", "questions":{"q":{"type":"noul","instructions":"Q?"}},"answers":{"q":"true"}}\n'
    (tmp_path / "tickets.held.jsonl").write_bytes(held_bytes)
    weighted, held, unfit = _read(mixture)
    source = weighted["tickets"].examples
    assert len(source) == 9 and source.width == 2 and source.questions == 2
    assert [source.example(index)[0] for index in range(len(source))] == [*rows[:8], rows[-1]]
    assert held == [Example.of(json.loads(held_bytes))] and unfit == {"tickets": 0, "held out": 0}
    assert (tmp_path / "tickets.held.jsonl").read_bytes() == held_bytes
    assert len(list(tmp_path.glob("tickets-*.array_record"))) == 4


def test_fit_is_an_index_before_grain_and_counts_only_unfit_rows(tmp_path):
    rows = _examples(4)
    long = Example("oversized", {"team": Choice("x" * 2000, ["a", "b"])}, {"team": 0})
    mixture = _write(tmp_path, {"tickets": (1.0, [rows[0], long, *rows[1:]])})
    (tmp_path / "tickets.held.jsonl").write_text(json.dumps({"state": long.state, "questions": {
        "team": long.questions["team"].wire()}, "answers": {"team": 0}}) + "\n")
    weighted, held, unfit = _read(mixture)
    source = weighted["tickets"].examples
    assert source.kept.tolist() == [0, 2, 3, 4]
    assert [source.example(index)[0] for index in range(len(source))] == rows
    assert held == [] and unfit == {"tickets": 1, "held out": 1}
    data = _objective().dataset(weighted, batch=2, loading=Loading(threads=1))
    stream = data.train(DataPartition())
    try:
        assert next(stream)["tokens"].shape == (2, 1024)
    finally:
        stream.close()


def test_an_empty_fitting_set_is_refused_by_name(tmp_path):
    bad = Example("x", {"team": Choice("x" * 2000, ["a", "b"])}, {"team": 0})
    mixture = _write(tmp_path, {"fine": (1.0, _examples()), "oversized": (1.0, [bad])})
    with pytest.raises(ValueError, match="oversized.*no.*fitting"):
        _read(mixture)
    mixture = dataclasses.replace(mixture, weights={"oversized": 0})
    assert set(_read(mixture)[0]) == {"fine"}


@pytest.mark.parametrize("layout", [LAYOUT, StateFirstLayout(max_len=96, head_max_len=48, option_tokens=8),
                                    MarkerLayout(max_len=96, head_max_len=48, option_tokens=8)])
def test_sidecar_lengths_match_layout_and_questions_follow_labels_order(tmp_path, layout):
    rows = [Example({"text": "é" * 100}, {"team": QUESTION, "urgent": Noul("Urgent?")},
                    {"urgent": "true"}, {"team": (0.3, 0.7)}), *_examples(3)]
    mixture = _write(tmp_path, {"tickets": (1.0, rows)})
    source = _read(mixture, layout)[0]["tickets"].examples
    expected = ([(row, tuple(row.questions)) for row in rows] if layout.joint else
                [(row, (name,)) for row in rows for name in row.labels()])
    assert [source.example(index) for index in range(len(source))] == expected
    lengths = [len(layout.rows(TOKENIZER, SPECIALS, row.state,
                               {name: row.questions[name] for name in names})[0].tokens)
               for row, names in expected]
    assert source.lengths.tolist() == lengths
    assert list(tmp_path.glob("lengths/*/tickets.npy"))


def test_a_warm_sidecar_does_not_decode_or_lay_out_the_training_set(tmp_path, monkeypatch):
    mixture = _write(tmp_path, {"tickets": (1.0, _examples(32))})
    _read(mixture)

    def refuse(*args, **kwargs):
        raise AssertionError("a warm indexed mixture must not decode examples at startup")

    monkeypatch.setattr(Example, "of", refuse)
    monkeypatch.setattr(JointLayout, "rows", refuse)
    weighted, held, unfit = _read(mixture)
    assert len(weighted["tickets"].examples) == 32 and held == [] and unfit["tickets"] == 0
    assert _objective(bucket=8).dataset(weighted, batch=4, loading=Loading(threads=1)).records == 32


def test_shares_and_saved_positions_use_the_existing_global_stream(tmp_path):
    mixture = _write(tmp_path, {"large": (3.0, _examples(32)), "small": (1.0, [
        Example("zz", {"team": QUESTION}, {"team": 1})] * 3)})
    layout = StateFirstLayout(max_len=96, head_max_len=48, option_tokens=8)
    weighted = _read(mixture, layout)[0]
    data = _objective(layout).dataset(weighted, batch=8, loading=Loading(threads=1))
    stream = data.train(DataPartition())
    resumed = data.train(DataPartition())
    try:
        batch = next(stream)
        assert sum(int(tokens[0]) == ord("z") for tokens in batch["tokens"]) == 2
        state = stream.get_state()
        expected = next(stream)
        resumed.set_state(state)
        next_batch = next(resumed)
        for name, value in expected.items():
            np.testing.assert_array_equal(next_batch[name], value)
        changed = dataclasses.replace(layout, max_len=104)
        other = _objective(changed).dataset(_read(mixture, changed)[0], batch=8,
                                           loading=Loading(threads=1)).train(DataPartition())
        try:
            with pytest.raises(ValueError, match="order"):
                other.set_state(state)
        finally:
            other.close()
    finally:
        stream.close()
        resumed.close()


def test_remote_shards_are_copied_once_by_content(tmp_path, monkeypatch):
    local = tmp_path / "remote"
    _write(local, {"tickets": (1.0, _examples())})
    original = epath.Path
    monkeypatch.setattr(epath, "Path", lambda path: original(
        str(path).replace("gs://bucket/mix", str(local))))
    mixture = DecisionMixture("gs://bucket/mix", cache=str(tmp_path / "cache"))
    weighted = _read(mixture)[0]
    source = weighted["tickets"].examples
    assert source.example(0)[0] == _examples()[0]
    cached = list((tmp_path / "cache").rglob("*.array_record"))
    assert len(cached) == 3
    before = {path: path.stat().st_mtime_ns for path in cached}
    assert _read(mixture)[0]["tickets"].examples.example(0)[0] == _examples()[0]
    assert {path: path.stat().st_mtime_ns for path in cached} == before
