"""The SFT data path: chat rendering, the packed role column, and refusals.

`render_conversation` must produce TRL's ids and TRL's assistant mask for the
same template and conversation. The reference is tests/fixtures/rl/chat.npz,
written by tools/parity_chat.py from TRL 1.12's SFT path with
`assistant_only_loss=True` on the tiny tokenizer under
tests/fixtures/tokenizers/tiny-chat. `Chat` packs those roles beside
the ids, and the packer and chunker keep the four per-token fields aligned.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from dew.data import Files, Loading, presets
from dew.data.chat import ROLES_KEY, Conversation, Role, render_conversation
from dew.data.compile import compile
from dew.data.dataset import DataPartition

TOKENIZER = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rl" / "chat.npz"

SYSTEM = "Be brief."
SHORT = [
    {"role": "user", "content": "What is two plus two?"},
    {"role": "assistant", "content": "Two plus two is four."},
]
MEDIUM = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Name a color."},
    {"role": "assistant", "content": "Blue."},
]
LONG = [
    {"role": "user", "content": "Count to three."},
    {"role": "assistant", "content": "One two three."},
    {"role": "user", "content": "Again."},
    {"role": "assistant", "content": "One two three."},
]
# Rendered lengths under the fixture tokenizer: 37, 53 and 72 tokens. The
# exact-bin test below packs them into windows of 96, so a length change
# re-derives its bins and this triple is asserted there, not here.


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURE, allow_pickle=True))


def write_parquet(path, conversations):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"prompt": conversations}), path / "chat.parquet")
    return str(path / "chat.parquet")


def render(tokenizer, conversation):
    return render_conversation(tokenizer, Conversation.parse(conversation), "tiny-chat")


# --- parity against TRL ----------------------------------------------------

def test_render_matches_trl_ids_and_assistant_mask(tokenizer, reference):
    """Dew's prefix rendering against TRL 1.12's SFT mask.

    TRL's mask marks assistant tokens of the input ids; Dew's weights count
    targets of the shifted row, so the assistant targets are TRL's mask from
    position 1 on. Largest observed difference: 0, both arrays exact.
    """
    conversation = json.loads(str(reference["conversation"]))
    ids, roles = render(tokenizer, conversation)

    np.testing.assert_array_equal(np.asarray(ids), reference["input_ids"])
    assistant = (np.asarray(roles[1:]) == Role.ASSISTANT).astype(np.int64)
    difference = float(np.abs(assistant - reference["assistant_mask"][1:]).max())
    assert difference == 0.0, f"largest difference against TRL: {difference}"


# --- packing -----------------------------------------------------------------

def packed(path, *, window, bins, batch, read="memory", loading=None, records=None, **settings):
    """The conversations at `path` packed by the sft preset into `window`-token rows, `bins` open, by
    its other `settings`, as a run reads them in batches of `batch`, and validating on the first
    `records` rows they pack."""
    def rows(**options):
        return Files(path).pipe(presets.sft, str(TOKENIZER), seq_len=window - 1, bins=bins, **settings,
                                **options)

    order = rows(batch=None).shuffle(0, window=64 if read == "streamed" else None).repeat()
    return compile(order.batch(batch), validation=rows(batch=batch, records=records), read=read,
                   loading=loading or Loading(workers=0))


@pytest.mark.parametrize("read", ["memory", "streamed"])
def test_a_packed_sft_batch_keeps_ids_roles_and_documents_aligned(tmp_path, tokenizer, read):
    """Ids, roles, segment ids and positions pack together, by the memory read's planner or
    Grain's packer as rows stream; the bins hold whole conversations first-fit, padding
    reads segment 0 and role 0."""
    window = 96
    path = write_parquet(tmp_path, [SHORT, MEDIUM, LONG])
    rendered = [render(tokenizer, conversation) for conversation in (SHORT, MEDIUM, LONG)]
    assert [len(ids) for ids, _ in rendered] == [37, 53, 72], (
        "the fixture tokenizer changed; re-derive the bins below")
    (a_ids, a_roles), (b_ids, b_roles), (c_ids, c_roles) = rendered

    data = packed(path, window=window, bins=2, batch=2, read=read, records=8, long="chunk")
    assert data.val is not None
    batches = list(data.val(DataPartition()))
    assert len(batches) == 1
    batch = batches[0]

    assert set(batch) == {"text", ROLES_KEY, "text_coordinates"}, (
        "one Coordinates value serves every aligned field")
    assert batch["text"].shape == (2, window)
    assert batch[ROLES_KEY].shape == (2, window)
    assert batch["text"].dtype == np.int32 and batch[ROLES_KEY].dtype == np.int8
    # First fit packs 37 + 53 into the first window and 72 into the second.
    np.testing.assert_array_equal(batch["text"][0], np.concatenate([a_ids, b_ids, np.zeros(6)]))
    np.testing.assert_array_equal(batch[ROLES_KEY][0],
                                  np.concatenate([a_roles, b_roles, np.zeros(6)]))
    np.testing.assert_array_equal(batch["text_coordinates"].segments[0],
                                  [1] * 37 + [2] * 53 + [0] * 6)
    np.testing.assert_array_equal(batch["text_coordinates"].positions[0],
                                  list(range(37)) + list(range(53)) + [0] * 6)
    np.testing.assert_array_equal(batch["text"][1], np.concatenate([c_ids, np.zeros(24)]))
    np.testing.assert_array_equal(batch[ROLES_KEY][1], np.concatenate([c_roles, np.zeros(24)]))
    np.testing.assert_array_equal(batch["text_coordinates"].segments[1], [1] * 72 + [0] * 24)
    np.testing.assert_array_equal(batch["text_coordinates"].positions[1], list(range(72)) + [0] * 24)


@pytest.mark.parametrize("options", [{"long": "chunk"}, {}], ids=["chunk", "default"])
def test_overlong_conversations_chunk_with_roles_aligned(tmp_path, tokenizer, options):
    """A conversation longer than the window is cut like a document, into
    chunks that share their boundary token, and the chunks reassemble to the
    render, ids and roles together, as the chat data does by default (P37)."""
    window = 64
    conversation = SHORT * 5
    path = write_parquet(tmp_path, [conversation])
    ids, roles = render(tokenizer, conversation)
    assert len(ids) > window
    chunks = -(-(len(ids) - 1) // (window - 1))

    data = packed(path, window=window, bins=1, batch=1, records=4, **options)
    assert data.records == chunks, "one bin puts every chunk in a window of its own"
    assert data.val is not None
    batches = list(data.val(DataPartition()))
    assert len(batches) == chunks
    # Chunk k holds the render from k * (window - 1), its first id and role the last of chunk k - 1,
    # then padding: 0 under segment 0.
    for index, batch in enumerate(batches):
        start = index * (window - 1)
        size = min(window, len(ids) - start)
        for field, rendered in (("text", ids), (ROLES_KEY, roles)):
            expected = np.zeros(window, batch[field].dtype)
            expected[:size] = rendered[start:start + size]
            np.testing.assert_array_equal(batch[field][0], expected)
        np.testing.assert_array_equal(batch["text_coordinates"].segments[0],
                                      [1] * size + [0] * (window - size))
        np.testing.assert_array_equal(batch["text_coordinates"].positions[0],
                                      list(range(size)) + [0] * (window - size))


@pytest.mark.slow
def test_the_train_stream_runs_end_to_end(tmp_path, tokenizer):
    """The four columns an SFT step reads, out of real worker processes: each
    row holds whole conversations, one segment each, positions restarting
    inside them, and the assistant turns marked."""
    path = write_parquet(tmp_path, [SHORT, MEDIUM, LONG])
    rendered = {tuple(render(tokenizer, conversation)[0].tolist())
                for conversation in (SHORT, MEDIUM, LONG)}
    columns = ("text", ROLES_KEY, "text_coordinates")
    data = packed(path, window=96, bins=2, batch=2, loading=Loading(workers=2), records=8, long="chunk")

    batch = next(data.train(DataPartition()))

    assert set(columns) <= set(batch)
    laid = batch["text_coordinates"]
    assert all(value.shape == (2, 96) for value in
               (batch["text"], batch[ROLES_KEY], laid.segments, laid.positions))
    assert batch["text"].dtype == np.int32 and batch[ROLES_KEY].dtype == np.int8
    for row in range(2):
        segments = laid.segments[row]
        for segment in range(1, int(segments.max()) + 1):
            span = segments == segment
            assert tuple(batch["text"][row][span].tolist()) in rendered
            np.testing.assert_array_equal(laid.positions[row][span],
                                          np.arange(int(span.sum())))
    # Every conversation ends on an assistant turn, so a batch that lost the
    # roles crossing back from a worker would carry no targets at all.
    assert (batch[ROLES_KEY] == Role.ASSISTANT).any()
