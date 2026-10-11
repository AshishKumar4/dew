"""Recipe encodings against the frozen pre-migration loader behavior."""

import functools
import itertools

import numpy as np
import pytest

from dew.data import ByteTokenizer, DataPartition, Loading
from dew.data.recipes import Chat, Prompts, Text, load
from dew.data.sources.memory import Memory
from dew.decision import Choice, DecisionObjective, Example, Specials, StateFirstLayout, Weighted
from dew.decision.data import Decisions
from dew.nn.backbones.causal_transformer import CausalTransformer

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


def _objective(bucket=None):
    return DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=8, num_layers=1, num_heads=2, mlp_features=16),
        tokenizer=ByteTokenizer(), specials=Specials(0, 1, 2, "<mark>", 3),
        layout=StateFirstLayout(max_len=128), shuffle_options=True, bucket=bucket)


def _assert_batches(left, right, count=4):
    try:
        for ours, expected in zip(itertools.islice(left, count), itertools.islice(right, count), strict=True):
            assert set(ours) == set(expected)
            for key in ours:
                np.testing.assert_array_equal(ours[key], expected[key], err_msg=key)
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("bucket", [None, 16])
def test_decision_objective_data_matches_legacy_augmented_and_unaugmented_batches(bucket):
    examples = [Example(f"state {index}", {"class": Choice("Pick", ["yes", "no"])}, {"class": index % 2})
                for index in range(40)]
    objective = _objective(bucket)
    recipe = Decisions(examples, validation=examples[:5])
    ours = load(recipe, batch=4, read="memory", objective=objective.data(),
                seed=7, loading=READ)
    old = objective.dataset(examples, validation=examples[:5], batch=4, seed=7, loading=READ)
    _assert_batches(ours.train(DataPartition()), old.train(DataPartition()))
    _assert_batches(ours.val(DataPartition()), old.val(DataPartition()))


def test_text_windows_keep_document_continuations_and_global_resume():
    documents = ({"text": "abcdefgh"}, {"text": "ijklmnopqrstuvwx"})
    loaded = load(Text(documents, seq_len=3), batch=2, read="memory", loading=READ)
    assert loaded.records == 7
    reader = loaded.train(DataPartition())
    first = next(reader)["text"]
    assert first.shape == (2, 4)
    state = reader.get_state()
    expected = next(reader)["text"]
    resumed = loaded.train(DataPartition())
    resumed.set_state(state)
    np.testing.assert_array_equal(next(resumed)["text"], expected)
    reader.close()
    resumed.close()


def test_prompt_default_drops_long_tasks_and_explicit_truncation_keeps_the_tail():
    rows = ({"prompt": [1, 2]}, {"prompt": [3, 4, 5, 6, 7]}, {"prompt": [8, 9]})
    filtered = load(Prompts(rows, "byte", max_prompt_len=3), batch=1, read="memory", loading=READ)
    assert filtered.records == 2
    truncated = load(Prompts(rows, "byte", max_prompt_len=3, long="truncate"),
                     batch=3, read="memory", loading=READ)
    reader = truncated.train(DataPartition())
    assert sorted(next(reader)["prompt"].tolist()) == [[0, 1, 2], [0, 8, 9], [5, 6, 7]]
    reader.close()


def test_chat_default_truncates_and_chunking_preserves_later_assistant_tokens():
    rows = ({"messages": [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello " * 20}]},) * 8
    tokenizer = "tests/fixtures/tokenizers/tiny-chat"
    truncated = load(Chat(rows, tokenizer, seq_len=15), batch=1, read="memory", loading=READ)
    chunked = load(Chat(rows, tokenizer, seq_len=15, long="chunk"), batch=1, read="memory", loading=READ)
    assert chunked.records > truncated.records
    reader = chunked.train(DataPartition())
    assert any(np.any(batch["text_roles"] == 3) for batch in itertools.islice(reader, chunked.records))
    reader.close()
