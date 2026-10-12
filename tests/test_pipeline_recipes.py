"""Recipe encodings against the frozen pre-migration loader behavior."""

import itertools
import re

import numpy as np
import pytest

from dew.data import ByteTokenizer, DataPartition, Loading
from dew.data.recipes import Chat, Prompts, Text, load
from dew.decision import Choice, DecisionObjective, Example, Specials, StateFirstLayout
from dew.decision.data import Decisions
from dew.nn.backbones.causal_transformer import CausalTransformer

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


def _objective(bucket=None):
    return DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=8, num_layers=1, num_heads=2, mlp_features=16),
        tokenizer=ByteTokenizer(), specials=Specials(0, 1, 2, "<mark>", 3),
        layout=StateFirstLayout(max_len=128), shuffle_options=True, bucket=bucket)


@pytest.mark.parametrize("bucket", [None, 16])
def test_decision_branches_keep_labels_aligned_with_option_order_and_cut_both_alike(bucket):
    examples = [Example(f"state {index}", {"class": Choice("Pick", ["yes", "no"])}, {"class": index % 2})
                for index in range(40)]
    objective = _objective(bucket)
    recipe = Decisions(examples, validation=examples[:5])
    ours = load(recipe, batch=4, read="memory", objective=objective.data(),
                seed=7, loading=READ)

    def width(batch) -> int:
        longest = int(batch["valid"].sum(axis=1).max())
        return 128 if bucket is None else min(128, -(-longest // bucket) * bucket)

    reader = ours.train(DataPartition())
    try:
        for batch in itertools.islice(reader, 8):
            for tokens, valid, labels in zip(batch["tokens"], batch["valid"], batch["labels"], strict=True):
                text = ByteTokenizer().decode(tokens[valid])
                matched = re.search(r"state (\d+)", text)
                assert matched is not None
                index = int(matched[1])
                order = ("yes", "no") if text.index("yes") < text.index("no") else ("no", "yes")
                assert order[int(labels[0])] == ("yes" if index % 2 == 0 else "no")
            assert batch["tokens"].shape[1] == width(batch)
    finally:
        reader.close()
    held = ours.val(DataPartition())
    try:
        batch = next(held)
        assert batch["tokens"].shape[1] == width(batch)
        np.testing.assert_array_equal(batch["labels"][:, 0], [0, 1, 0, 1])
    finally:
        held.close()


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


def test_a_cached_token_holdout_caches_the_documents_before_the_split(tmp_path):
    """A recipe's cache holds the deterministic steps before its token split,
    so a cached memory read of a token holdout holds out what the uncached
    read holds out."""
    import json
    from uuid import uuid4

    from dew.data import HoldOut
    from dew.data.cache import Cache

    (tmp_path / "text.jsonl").write_text(
        "".join(json.dumps({"text": f"document {index} " * 6}) + "\n" for index in range(8)))
    recipe = Text(source=str(tmp_path / "text.jsonl"), seq_len=8, token_holdout=HoldOut(fraction=0.25))

    def held(cache):
        loaded = load(recipe, batch=2, read="memory", cache=cache, loading=READ)
        assert loaded.val is not None
        return [{name: np.asarray(value).tolist() for name, value in batch.items()}
                for batch in loaded.val(DataPartition())]

    assert held(Cache(f"memory://{uuid4().hex}")) == held(None)
