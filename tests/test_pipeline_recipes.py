"""Recipe encodings against the frozen pre-migration loader behavior."""

import itertools
import re

import numpy as np

from dew.config.data import ChatData, PromptData, TextData, loader
from dew.data import ByteTokenizer, Loading
from dew.data.dataset import DataPartition
from dew.decision import Choice, DecisionObjective, Example, Specials, StateFirstLayout
from dew.decision.data import Decisions
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import VALID_ROWS

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


def _objective():
    return DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=8, num_layers=1, num_heads=2, mlp_features=16),
        tokenizer=ByteTokenizer(), specials=Specials(0, 1, 2, "<mark>", 3),
        layout=StateFirstLayout(max_len=128), shuffle_options=True)


def _rows(batch):
    """Each packed row of a batch's real windows: its text and its one question's label."""
    real = batch.get(VALID_ROWS, np.ones(len(batch["tokens"]), bool))
    names = ("segments", "tokens", "spans", "labels", "questions")
    columns = zip(*(batch[name] for name in names), strict=True)
    for (segments, tokens, spans, labels, asked), kept in zip(columns, real, strict=True):
        for slot in np.flatnonzero(asked & kept):
            yield ByteTokenizer().decode(tokens[segments == segments[spans[slot, 0]]]), int(labels[slot])


def test_decision_branches_keep_labels_aligned_with_option_order_in_packed_windows():
    examples = [Example(f"state {index}", {"class": Choice("Pick", ["yes", "no"])}, {"class": index % 2})
                for index in range(40)]
    recipe = Decisions(examples, validation=examples[:5])
    ours = loader(recipe, batch=4, read="memory", objective=_objective().data(), seed=7, loading=READ)

    def aligned(text: str, label: int) -> int:
        matched = re.search(r"state (\d+)", text)
        assert matched is not None
        index = int(matched[1])
        order = ("yes", "no") if text.index("yes") < text.index("no") else ("no", "yes")
        assert order[label] == ("yes" if index % 2 == 0 else "no")
        return index

    reader = ours.train(DataPartition())
    try:
        for batch in itertools.islice(reader, 8):
            assert batch["tokens"].shape == (4, 128)
            for text, label in _rows(batch):
                aligned(text, label)
    finally:
        reader.close()
    held = ours.val(DataPartition())
    try:
        indices = [aligned(text, label) for batch in held for text, label in _rows(batch)]
        assert sorted(indices) == list(range(5))
    finally:
        held.close()


def test_text_windows_keep_document_continuations_and_global_resume():
    documents = ({"text": "abcdefgh"}, {"text": "ijklmnopqrstuvwx"})
    loaded = loader(TextData(documents, seq_len=3), batch=2, read="memory", loading=READ)
    assert loaded.records == 8
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
    filtered = loader(PromptData(rows, "byte", max_prompt_len=3), batch=1, read="memory", loading=READ)
    assert filtered.records == 2
    truncated = loader(PromptData(rows, "byte", max_prompt_len=3, long="truncate"),
                     batch=3, read="memory", loading=READ)
    reader = truncated.train(DataPartition())
    assert sorted(next(reader)["prompt"].tolist()) == [[0, 1, 2], [0, 8, 9], [5, 6, 7]]
    reader.close()


def test_chat_default_truncates_and_chunking_preserves_later_assistant_tokens():
    rows = ({"messages": [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello " * 20}]},) * 8
    tokenizer = "tests/fixtures/tokenizers/tiny-chat"
    truncated = loader(ChatData(rows, tokenizer, seq_len=15), batch=1, read="memory", loading=READ)
    chunked = loader(ChatData(rows, tokenizer, seq_len=15, long="chunk"), batch=1, read="memory",
                     loading=READ)
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

    from dew.config.data import HoldOut
    from dew.data.cache import Cache

    (tmp_path / "text.jsonl").write_text(
        "".join(json.dumps({"text": f"document {index} " * 6}) + "\n" for index in range(8)))
    recipe = TextData(source=str(tmp_path / "text.jsonl"), seq_len=8, token_holdout=HoldOut(fraction=0.25))

    def held(cache):
        loaded = loader(recipe, batch=2, read="memory", cache=cache, loading=READ)
        assert loaded.val is not None
        return [{name: np.asarray(value).tolist() for name, value in batch.items()}
                for batch in loaded.val(DataPartition())]

    assert held(Cache(f"memory://{uuid4().hex}")) == held(None)
