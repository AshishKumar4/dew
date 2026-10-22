"""The text presets: documents tokenized and packed, conversations rendered and packed, and a token
store's windows and documents, written out as the public steps."""

from pathlib import Path

import jax
import numpy as np
from token_support import write_token_corpus

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


def store_of_documents(directory) -> tuple[str, np.ndarray]:
    """A token store of 40 short documents, each ended by EOS (0): its url and the stream it holds."""
    stream = np.concatenate([np.asarray([*range(1, 4 + index % 5), 0], np.int64) for index in range(40)])
    return write_token_corpus(directory, stream, eos_id=0).train, stream


def test_the_token_windows_preset_reads_a_stores_windows_as_the_windows_verb_written_first(tmp_path):
    """The token windows preset over a token store shuffled and repeated: the plan and batches of the
    store's windows two apart written first and then shuffled, repeated and batched, read by index;
    recorded and read back with its stride overridden, windows that far apart."""
    from dew.data.records import Unrecorded, decode, encode, override

    url, _ = store_of_documents(tmp_path)

    def windows(stride):
        return compile(TokenFiles(url).windows(8, stride=stride).shuffle(0).repeat().batch(4), read="indexed",
                       loading=LOADING)

    chain = TokenFiles(url).shuffle(0).repeat().pipe(presets.token_windows, seq_len=8, batch=4, stride=2)
    built = compile(chain, read="indexed", loading=LOADING)
    assert built.plan.identity() == windows(2).plan.identity()
    same_batches(windows(2), built)
    written = encode(chain)
    assert not isinstance(written, Unrecorded)
    wider = decode(override(written, "token_windows.stride", 4))
    assert compile(wider, read="indexed", loading=LOADING).plan.identity() == windows(4).plan.identity()


def test_a_token_stores_documents_are_packed_whole(tmp_path):
    """A token store's documents, ended by EOS, packed first-fit with no tokenizer: each row holds
    whole documents, so its targets are its documents' own, and together the rows of a pass hold
    every target of every document."""
    url, stream = store_of_documents(tmp_path)
    rows = list(TokenFiles(url).pipe(presets.lm, None, seq_len=8, batch=1, bins=8))
    segments = [np.asarray(row["text_coordinates"].segments)[0] for row in rows]
    held = sum(int(np.count_nonzero((ids[:-1] == ids[1:]) & (ids[:-1] > 0))) for ids in segments)
    documents = np.split(stream, np.flatnonzero(stream == 0)[:-1] + 1)
    assert held == sum(len(document) - 1 for document in documents)


def test_the_lm_preset_tokenizes_a_field_of_another_name():
    """Passages under "body" are tokenized into "text", as `text.tokenize(field=)` writes them."""
    passages = Memory([{"body": "abc " * 6}])
    rows = list(passages.pipe(presets.lm, "byte", seq_len=8, batch=1, field="body", pack=None))
    assert np.asarray(rows[0]["text"])[0].tolist()[:4] == text.ByteTokenizer().encode("abc ")


def test_exported_tokens_read_back_as_a_token_store_and_pack(tmp_path):
    """Documents exported as tokens by bytes, each ended by EOS: the store states the tokenizer and
    its vocabulary, and packing it reads every next-token target of every document."""
    rows = documents(6)
    manifest = text.export_tokens(rows, str(tmp_path / "tokens"), "byte")
    assert manifest["metadata"]["tokenizer"] == "byte"
    store = TokenFiles(str(tmp_path / "tokens"))
    assert (store.stated()["vocab_size"], store.stated()["eos_id"]) == (256, 255)
    packed = list(store.pipe(presets.lm, None, seq_len=8, batch=1, bins=8))
    segments = [np.asarray(row["text_coordinates"].segments)[0] for row in packed]
    held = sum(int(np.count_nonzero((ids[:-1] == ids[1:]) & (ids[:-1] > 0))) for ids in segments)
    lengths = [len(str(row["text"]).encode()) + 1 for row in rows]
    assert held == sum(length - 1 for length in lengths)
