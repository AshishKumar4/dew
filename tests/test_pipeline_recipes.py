"""Preset encodings against the frozen pre-migration loader behavior."""

import itertools
import json
import re

import numpy as np

from dew.batch import VALID_ROWS
from dew.data import ByteTokenizer, Loading, Memory, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.decision import Choice, DecisionObjective, Example, Specials, StateFirstLayout, decisions
from dew.nn.backbones.causal_transformer import CausalTransformer

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
    def rows(chosen):
        return Memory([{"example": example} for example in chosen]).pipe(decisions)

    ours = compile(rows(examples).shuffle(7).repeat().batch(4), validation=rows(examples[:5]).batch(4),
                   read="memory", objective=_objective().data(), seed=7, loading=READ)

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
    documents = Memory([{"text": "abcdefgh"}, {"text": "ijklmnopqrstuvwx"}])
    chain = documents.pipe(presets.lm, "byte", seq_len=3, batch=None, pack=None).shuffle(0).repeat().batch(2)
    loaded = compile(chain, read="memory", loading=READ)
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


def test_prompt_default_keeps_a_long_tasks_tail_and_explicit_dropping_filters_it():
    """An overlong prompt keeps its tail by default (post-training P62); `long="drop"` filters it."""
    rows = Memory([{"prompt": [1, 2]}, {"prompt": [3, 4, 5, 6, 7]}, {"prompt": [8, 9]}])

    def loaded(batch, **options):
        # Rendered, then ordered: the prompts a drop keeps are the records an epoch reads.
        chain = rows.pipe(presets.prompts, "byte", max_prompt_len=3, batch=None, **options)
        return compile(chain.shuffle(0).repeat().batch(batch), read="memory", loading=READ)

    assert loaded(1, long="drop").records == 2
    reader = loaded(3).train(DataPartition())
    assert sorted(next(reader)["prompt"].tolist()) == [[0, 1, 2], [0, 8, 9], [5, 6, 7]]
    reader.close()


def test_chat_default_chunks_and_preserves_later_assistant_tokens_that_truncation_cuts():
    """A long conversation is chunked by default (post-training P37), keeping every turn, where an
    explicit `long="truncate"` cuts the later assistant tokens away."""
    rows = Memory([{"messages": [{"role": "user", "content": "hi"},
                                 {"role": "assistant", "content": "hello " * 20}]}] * 8)

    def loaded(**options):
        chain = rows.pipe(presets.sft, "tests/fixtures/tokenizers/tiny-chat", seq_len=15, batch=None, bins=8,
                          **options)
        return compile(chain.shuffle(0).repeat().batch(1), read="memory", loading=READ)

    truncated, chunked = loaded(long="truncate"), loaded()
    assert chunked.records > truncated.records
    reader = chunked.train(DataPartition())
    assert any(np.any(batch["text_roles"] == 3) for batch in itertools.islice(reader, chunked.records))
    reader.close()


def test_a_cached_token_holdout_holds_out_what_the_uncached_one_does(tmp_path):
    """A cache of the tokenized documents before a token split, read into memory, holds out what the
    uncached read holds out."""
    from uuid import uuid4

    from dew.data import Files, text
    from dew.data.cache import Cache
    from dew.data.compile import held

    (tmp_path / "text.jsonl").write_text(
        "".join(json.dumps({"text": f"document {index} " * 6}) + "\n" for index in range(8)))
    documents = Files(str(tmp_path / "text.jsonl")).map(text.tokenize("byte", eos=True))

    def scored(cache):
        tokens = documents if cache is None else documents.cache(cache)
        _, part = tokens.holdout(tokens=0.25)
        windows = part.pipe(presets.lm, None, seq_len=8, batch=2, pack=None).read("memory").loading(READ)
        return [{name: np.asarray(value).tolist() for name, value in batch.items()}
                for batch in held(windows)(DataPartition())]

    assert scored(Cache(f"memory://{uuid4().hex}")) == scored(None)


