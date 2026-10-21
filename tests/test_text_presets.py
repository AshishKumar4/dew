"""The text presets prepare what the text and chat recipes do, from the recipes' own settings."""

from pathlib import Path

import jax
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
    """The first `count` training batches of two loaders, equal field by field, a packed batch's
    coordinates included."""
    wanted, made = expected.train(DataPartition()), built.train(DataPartition())
    try:
        for _ in range(count):
            left, right = next(wanted), next(made)
            assert left.keys() == right.keys()
            jax.tree.map(np.testing.assert_array_equal, right, left)
    finally:
        wanted.close()
        made.close()


@pytest.mark.parametrize("pack", [None, "first-fit", "concat"])
def test_the_lm_preset_prepares_what_the_text_recipe_does(pack):
    """Raw passages, tokenized by bytes with an EOS and laid out by each of the recipe's layouts:
    fixed windows, first-fit packing in eight open rows, or concatenation, read by index: the
    preset's plan and batches are the recipe's. (A memory read orders the recipe's windows, after
    its fixed steps, and the chain's documents, where its shuffle is written.)"""
    read = "indexed"
    recipe = TextData(documents(), tokenizer="byte", seq_len=16, pack=pack, packing_bins=8,
                      token_holdout=None)
    expected = loader(recipe, batch=4, read=read, loading=LOADING)
    chain = documents().shuffle(0).repeat().pipe(presets.lm, "byte", seq_len=16, batch=4, eos=True,
                                                 pack=pack, bins=8)
    built = compile(chain, read=read, loading=LOADING)
    assert built.plan.identity() == expected.plan.identity()
    same_batches(expected, built)


@pytest.mark.parametrize("long", ["chunk", "truncate", "drop"])
def test_the_sft_preset_prepares_what_the_chat_recipe_does(long):
    """Conversations of differing lengths rendered by a chat template, each one longer than a row
    cut into pieces, cut at its end or dropped, as the recipe's `long` says, read by index."""
    read = "indexed"
    conversations = Memory([{"messages": [{"role": "user", "content": "say " + "more " * index},
                                          {"role": "assistant", "content": "done " * (1 + index % 5)}]}
                            for index in range(24)])
    recipe = ChatData(conversations, tokenizer=str(CHAT), seq_len=48, long=long, packing_bins=8)
    expected = loader(recipe, batch=2, read=read, loading=LOADING)
    chain = conversations.shuffle(0).repeat().pipe(presets.sft, str(CHAT), seq_len=48, batch=2, long=long,
                                                   bins=8)
    built = compile(chain, read=read, loading=LOADING)
    assert built.plan.identity() == expected.plan.identity()
    same_batches(expected, built)


def test_a_concatenated_conversation_keeps_its_target_across_a_row_end():
    """Conversations that each fit a row, concatenated: the first fills less than a row, so the
    second is cut where that row ends, and the next row starts with the cut's last token again;
    every next-token target of every conversation is in some row, in one piece."""
    from dew.data import chat

    conversations = Memory([{"messages": [{"role": "user", "content": "hello there " * (1 + index)},
                                          {"role": "assistant", "content": "hi " * 3}]}
                            for index in range(5)])
    rendered = [len(np.asarray(row["text"])) for row in conversations.map(chat.template(str(CHAT)))]
    # Rows one token longer than the longest conversation: each fits, and none fills a row alone.
    seq_len = max(rendered)
    rows = list(conversations.pipe(presets.sft, str(CHAT), seq_len=seq_len, batch=1, long="truncate",
                                   pack="concat"))
    segments = [np.asarray(row["text_coordinates"].segments)[0] for row in rows]
    held = sum(int(np.count_nonzero((ids[:-1] == ids[1:]) & (ids[:-1] > 0))) for ids in segments)
    assert len(rows) > 1
    assert held == sum(length - 1 for length in rendered)


def store_of_documents(directory) -> np.ndarray:
    """A token store of 40 short documents, each ended by EOS (0); the stream it holds."""
    stream = np.concatenate([np.asarray([*range(1, 4 + index % 5), 0], np.int64) for index in range(40)])
    write_token_corpus(directory, stream.astype(np.uint16), None, eos_id=0)
    return stream


def test_a_token_stores_windows_are_the_text_recipes(tmp_path):
    """A token store's stream cut into fixed windows two apart, shuffled and repeated as the
    recipe orders them, read by index: the windows written first on the store read its stream by
    index, so the plan and batches are the recipe's."""
    store_of_documents(tmp_path)
    expected = loader(TextData(str(tmp_path), seq_len=8, stride=2), batch=4, read="indexed", loading=LOADING)
    chain = TokenFiles(str(tmp_path)).windows(8, stride=2).shuffle(0).repeat().batch(4)
    built = compile(chain, read="indexed", loading=LOADING)
    assert built.plan.identity() == expected.plan.identity()
    same_batches(expected, built)


def test_the_token_windows_preset_reads_a_stores_windows_as_the_text_recipe(tmp_path):
    """The token windows preset over a token store shuffled and repeated: the recipe's plan and
    batches; recorded and read back with its stride overridden, windows that far apart."""
    from dew.data.records import Unrecorded, decode, encode, override

    store_of_documents(tmp_path)
    expected = loader(TextData(str(tmp_path), seq_len=8, stride=2), batch=4, read="indexed", loading=LOADING)
    chain = TokenFiles(str(tmp_path)).shuffle(0).repeat().pipe(presets.token_windows, seq_len=8, batch=4,
                                                               stride=2)
    built = compile(chain, read="indexed", loading=LOADING)
    assert built.plan.identity() == expected.plan.identity()
    same_batches(expected, built)
    written = encode(chain)
    assert not isinstance(written, Unrecorded)
    wider = decode(override(written, "token_windows.stride", 4))
    recipe = loader(TextData(str(tmp_path), seq_len=8, stride=4), batch=4, read="indexed", loading=LOADING)
    assert compile(wider, read="indexed", loading=LOADING).plan.identity() == recipe.plan.identity()


def test_a_token_stores_documents_are_packed_whole(tmp_path):
    """A token store's documents, ended by EOS, packed first-fit with no tokenizer: each row holds
    whole documents, so its targets are its documents' own, and together the rows of a pass hold
    every target of every document."""
    stream = store_of_documents(tmp_path)
    rows = list(TokenFiles(str(tmp_path)).pipe(presets.lm, None, seq_len=8, batch=1, bins=8))
    segments = [np.asarray(row["text_coordinates"].segments)[0] for row in rows]
    held = sum(int(np.count_nonzero((ids[:-1] == ids[1:]) & (ids[:-1] > 0))) for ids in segments)
    documents = np.split(stream, np.flatnonzero(stream == 0)[:-1] + 1)
    assert held == sum(len(document) - 1 for document in documents)


@pytest.mark.parametrize("composed", ["alone", "concatenated", "mixed"])
def test_a_token_store_that_states_no_document_ends_is_not_packed(tmp_path, composed):
    """A store with neither a document index nor an EOS token reads as one document, whose pieces
    no producer made, so packing its documents is refused, alone or among other sources; its
    stream still reads as fixed windows."""
    from dew.data import concat, mix

    write_token_corpus(tmp_path, np.arange(64, dtype=np.uint16), None)
    store = TokenFiles(str(tmp_path))
    chained = {"alone": store, "concatenated": concat([store]), "mixed": mix([store])}[composed]
    with pytest.raises(ValueError):
        chained.pipe(presets.lm, None, seq_len=8, batch=4)
    rows = list(store.pipe(presets.lm, None, seq_len=8, batch=4, pack=None))
    assert np.asarray(rows[0]["text"]).shape == (4, 9)


def test_the_lm_preset_tokenizes_a_field_of_another_name():
    """Passages under "body" are tokenized into "text", as `text.tokenize(field=)` writes them."""
    passages = Memory([{"body": "abc " * 6}])
    rows = list(passages.pipe(presets.lm, "byte", seq_len=8, batch=1, field="body", pack=None))
    assert np.asarray(rows[0]["text"])[0].tolist()[:4] == text.ByteTokenizer().encode("abc ")
