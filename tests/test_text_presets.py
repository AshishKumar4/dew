"""The text presets prepare what the text and chat recipes do, from the recipes' own settings."""

from pathlib import Path

import numpy as np
import pytest
from token_support import write_token_corpus

from dew.config.data import ChatData, TextData, loader
from dew.data import Loading, Memory, presets, text
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.sources.tokens import TokenFiles

CHAT = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat"
LOADING = Loading(workers=0, threads=1, read_buffer=2)


def documents(count: int = 24) -> Memory:
    """`count` short passages of differing lengths."""
    return Memory([{"text": f"passage {index}: " + "to be or not to be " * (1 + index % 4)}
                   for index in range(count)])


def same_batches(expected, built, count: int = 4) -> None:
    """The first `count` training batches of two loaders, equal field by field."""
    wanted, made = expected.train(DataPartition()), built.train(DataPartition())
    try:
        for _ in range(count):
            left, right = next(wanted), next(made)
            assert left.keys() == right.keys()
            for name in left:
                np.testing.assert_array_equal(right[name], left[name])
    finally:
        wanted.close()
        made.close()


@pytest.mark.parametrize("read", ["memory", "indexed"])
@pytest.mark.parametrize("pack", [None, "first-fit", "concat"])
def test_the_lm_preset_prepares_what_the_text_recipe_does(read, pack):
    """Raw passages, tokenized by bytes with an EOS and laid out by each of the recipe's layouts:
    fixed windows, first-fit packing in eight open rows, or concatenation. The preset's batches are
    the recipe's; its plan is too where the chain orders its records where the recipe does (an
    index read; a memory read orders the recipe's after its fixed steps)."""
    recipe = TextData(documents(), tokenizer="byte", seq_len=16, pack=pack, packing_bins=8,
                      token_holdout=None)
    expected = loader(recipe, batch=4, read=read, loading=LOADING)
    chain = documents().shuffle(0).repeat().pipe(presets.lm, "byte", seq_len=16, batch=4, eos=True,
                                                 pack=pack, bins=8)
    built = compile(chain, read=read, loading=LOADING)
    if read != "memory":
        assert built.plan.identity() == expected.plan.identity()
    same_batches(expected, built)


@pytest.mark.parametrize("read", ["memory", "indexed"])
@pytest.mark.parametrize("long", ["chunk", "truncate", "drop"])
def test_the_sft_preset_prepares_what_the_chat_recipe_does(read, long):
    """Conversations of differing lengths rendered by a chat template, each one longer than a row
    cut into pieces, cut at its end or dropped, as the recipe's `long` says."""
    conversations = Memory([{"messages": [{"role": "user", "content": "say " + "more " * index},
                                          {"role": "assistant", "content": "done " * (1 + index % 5)}]}
                            for index in range(24)])
    recipe = ChatData(conversations, tokenizer=str(CHAT), seq_len=48, long=long, packing_bins=8)
    expected = loader(recipe, batch=2, read=read, loading=LOADING)
    chain = conversations.shuffle(0).repeat().pipe(presets.sft, str(CHAT), seq_len=48, batch=2, long=long,
                                                   bins=8)
    built = compile(chain, read=read, loading=LOADING)
    if read != "memory":
        assert built.plan.identity() == expected.plan.identity()
    same_batches(expected, built)


def test_a_concatenated_conversation_keeps_its_target_across_a_row_end():
    """Conversations each cut to fit a row and then concatenated: a row end that cuts one keeps its
    last token again at the next row's start, so every next-token target of every conversation is
    in some row, in one piece."""
    from dew.data import chat

    conversations = Memory([{"messages": [{"role": "user", "content": "hello there " * (1 + index)},
                                          {"role": "assistant", "content": "hi " * 3}]}
                            for index in range(5)])
    rendered = [len(np.asarray(row["text"])) for row in conversations.map(chat.template(str(CHAT)))]
    rows = list(conversations.pipe(presets.sft, str(CHAT), seq_len=20, batch=1, long="truncate",
                                   pack="concat"))
    segments = [np.asarray(row["text_segment_ids"])[0] for row in rows]
    held = sum(int(np.count_nonzero((ids[:-1] == ids[1:]) & (ids[:-1] > 0))) for ids in segments)
    assert held == sum(min(length, 21) - 1 for length in rendered)


@pytest.mark.parametrize("pack", [None, "first-fit"])
def test_a_token_store_is_prepared_as_the_text_recipe_prepares_it(tmp_path, pack):
    """A token store of documents each ended by EOS: fixed windows of its stream two apart
    (`token_windows`), or its documents packed first-fit (`lm` with no tokenizer)."""
    stream = np.concatenate([np.asarray([*range(1, 4 + index % 5), 0], np.int64) for index in range(40)])
    write_token_corpus(tmp_path, stream.astype(np.uint16), None, eos_id=0)
    recipe = TextData(str(tmp_path), seq_len=8, stride=2 if pack is None else None, pack=pack,
                      packing_bins=8)
    expected = loader(recipe, batch=4, read="indexed", loading=LOADING)
    store = TokenFiles(str(tmp_path))
    ordered = store.shuffle(0).repeat()
    chain = (ordered.pipe(presets.token_windows, 8, stride=2, batch=4) if pack is None
             else ordered.pipe(presets.lm, None, seq_len=8, batch=4, bins=8))
    built = compile(chain, read="indexed", loading=LOADING)
    same_batches(expected, built)


def test_a_token_store_that_states_no_document_ends_is_not_packed(tmp_path):
    """A store with neither a document index nor an EOS token reads as one document, whose pieces
    no producer made, so packing its documents is refused; its fixed windows still read."""
    write_token_corpus(tmp_path, np.arange(64, dtype=np.uint16), None)
    store = TokenFiles(str(tmp_path))
    with pytest.raises(ValueError):
        store.pipe(presets.lm, None, seq_len=8, batch=4)
    rows = list(store.pipe(presets.token_windows, 8, batch=4))
    assert np.asarray(rows[0]["text"]).shape == (4, 9)


def test_the_lm_preset_tokenizes_a_field_of_another_name():
    """Passages under "body" are tokenized into "text", as `text.tokenize(field=)` writes them."""
    passages = Memory([{"body": "abc " * 6}])
    rows = list(passages.pipe(presets.lm, "byte", seq_len=8, batch=1, field="body", pack=None))
    assert np.asarray(rows[0]["text"])[0].tolist()[:4] == text.ByteTokenizer().encode("abc ")
