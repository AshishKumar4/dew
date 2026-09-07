"""The SFT data path: chat rendering, the packed role column, and refusals.

`render_conversation` must produce TRL's ids and TRL's assistant mask for the
same template and conversation. The reference is tests/fixtures/rl/chat.npz,
written by tools/parity_chat.py from TRL 1.12's SFT path with
`assistant_only_loss=True` on the tiny tokenizer under
tests/fixtures/tokenizers/tiny-chat. `ChatMessages` packs those roles beside
the ids, and the packer and chunker keep the four per-token fields aligned.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from dew.data import ChatMessages, Loading
from dew.data.chat import ROLES_KEY, Conversation, Role, render_conversation

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


def test_the_fixture_names_its_reference(reference):
    """The parity claim is checkable: template, versions and TRL's own
    template checks travel with the arrays."""
    assert str(reference["template"]) == "trl_qwen3_training"
    assert str(reference["trl_version"]) and str(reference["transformers_version"])
    assert bool(reference["prefix_preserving"]) and bool(reference["stop_token_trained"])
    roles = {message["role"] for message in json.loads(str(reference["conversation"]))}
    assert roles == {"system", "user", "assistant", "tool"}


# --- packing -----------------------------------------------------------------

def test_a_packed_sft_batch_carries_four_aligned_fields(tmp_path, tokenizer):
    """Ids, roles, segment ids and positions pack together; the bins hold
    whole conversations first-fit, padding reads segment 0 and role 0."""
    window = 96
    path = write_parquet(tmp_path, [SHORT, MEDIUM, LONG])
    rendered = [render(tokenizer, conversation) for conversation in (SHORT, MEDIUM, LONG)]
    assert [len(ids) for ids, _ in rendered] == [37, 53, 72], (
        "the fixture tokenizer changed; re-derive the bins below")
    (a_ids, a_roles), (b_ids, b_roles), (c_ids, c_roles) = rendered

    data = ChatMessages(tokenizer=str(TOKENIZER), path=path, val_path=path,
                        seq_len=window - 1, packing_bins=2,
                        loading=Loading(workers=0)).load(batch=2)
    assert data.val is not None
    batches = list(data.val())
    assert len(batches) == 1
    batch = batches[0]

    assert batch["text"].shape == (2, window)
    assert batch[ROLES_KEY].shape == (2, window)
    assert batch["text"].dtype == np.int32 and batch[ROLES_KEY].dtype == np.int8
    # First fit packs 37 + 53 into the first window and 72 into the second.
    np.testing.assert_array_equal(batch["text"][0], np.concatenate([a_ids, b_ids, np.zeros(6)]))
    np.testing.assert_array_equal(batch[ROLES_KEY][0],
                                  np.concatenate([a_roles, b_roles, np.zeros(6)]))
    np.testing.assert_array_equal(batch["text_segment_ids"][0],
                                  [1] * 37 + [2] * 53 + [0] * 6)
    np.testing.assert_array_equal(batch["text_positions"][0],
                                  list(range(37)) + list(range(53)) + [0] * 6)
    np.testing.assert_array_equal(batch["text"][1], np.concatenate([c_ids, np.zeros(24)]))
    np.testing.assert_array_equal(batch[ROLES_KEY][1], np.concatenate([c_roles, np.zeros(24)]))
    np.testing.assert_array_equal(batch["text_segment_ids"][1], [1] * 72 + [0] * 24)
    np.testing.assert_array_equal(batch["text_positions"][1], list(range(72)) + [0] * 24)


def test_overlong_conversations_chunk_with_roles_aligned(tmp_path, tokenizer):
    """A conversation longer than the window is cut like a document, and the
    chunks reassemble to the render, ids and roles together."""
    window = 64
    conversation = SHORT * 5
    path = write_parquet(tmp_path, [conversation])
    ids, roles = render(tokenizer, conversation)
    assert len(ids) > window
    chunks = -(-len(ids) // window)

    data = ChatMessages(tokenizer=str(TOKENIZER), path=path, val_path=path,
                        seq_len=window - 1, packing_bins=1,
                        loading=Loading(workers=0)).load(batch=1)
    assert data.records == chunks
    assert data.val is not None
    batches = list(data.val())
    assert len(batches) == chunks
    np.testing.assert_array_equal(
        np.concatenate([batch["text"][0] for batch in batches])[:len(ids)], ids)
    np.testing.assert_array_equal(
        np.concatenate([batch[ROLES_KEY][0] for batch in batches])[:len(ids)], roles)


@pytest.mark.slow
def test_the_train_stream_runs_end_to_end(tmp_path):
    """The shuffled, endless train path emits the same four fields, rendered
    inside real worker processes."""
    path = write_parquet(tmp_path, [SHORT, MEDIUM, LONG])
    data = ChatMessages(tokenizer=str(TOKENIZER), path=path, seq_len=95,
                        packing_bins=2, loading=Loading(workers=2)).load(batch=2)

    batch = next(data.train())

    assert batch["text"].shape == (2, 96) and batch[ROLES_KEY].shape == (2, 96)


def test_records_count_chunks_not_conversations(tmp_path, tokenizer):
    """A run turns records into steps, so one conversation in several chunks
    reports the chunk count."""
    conversation = SHORT * 5
    path = write_parquet(tmp_path, [conversation])
    ids, _ = render(tokenizer, conversation)
    chunks = -(-len(ids) // 64)
    data = ChatMessages(tokenizer=str(TOKENIZER), path=path, val_path=path,
                        seq_len=63, packing_bins=1,
                        loading=Loading(workers=0)).load(batch=1)

    assert data.records == chunks
    assert data.val is not None
    assert len(list(data.val())) == chunks, "the length is not the pass it counts"


def test_a_malformed_message_is_refused(tokenizer):
    """A message is an object with a string role; its content is text,
    typed parts, or nothing."""
    with pytest.raises(ValueError, match="an object"):
        render(tokenizer, ["just a string"])
    with pytest.raises(ValueError, match="content is a list"):
        render(tokenizer, [{"role": "user", "content": 4}])
    with pytest.raises(ValueError, match="role as a string"):
        render(tokenizer, [{"content": "hi"}])


def test_an_empty_conversation_is_refused(tokenizer):
    with pytest.raises(ValueError, match="empty"):
        render(tokenizer, [])


def test_a_conversation_starting_with_the_assistant_is_refused(tokenizer):
    """The opening header cannot be separated from the completion through the
    template, so an assistant-first row fails with its position."""
    with pytest.raises(ValueError, match="first message"):
        render(tokenizer, list(reversed(SHORT)))


def test_chat_messages_needs_a_parquet_path():
    with pytest.raises(ValueError, match="--data.path"):
        ChatMessages(tokenizer=str(TOKENIZER)).load(batch=8)


def test_an_empty_parquet_file_is_refused(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"prompt": []}), tmp_path / "empty.parquet")
    with pytest.raises(ValueError, match="holds no conversations"):
        ChatMessages(tokenizer=str(TOKENIZER),
                     path=str(tmp_path / "empty.parquet")).load(batch=8)
