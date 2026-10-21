"""Structured conversations: tool calls, tool responses and typed content
through a real chat template, with the ids and the mask read back.

Two fixture tokenizers render the same conversation. `tiny-tools` carries
a ChatML template that reads tool-call ids, the id and name a tool response
answers, over a BPE model with every byte. Text parts join before rendering;
rendered JSON tokenizes to distinct ids. `tiny-chat` carries TRL's Qwen3
training template, which reads `tool_calls` and `reasoning_content` and
merges a run of tool responses into one block. Every expectation here is
either a string written by hand and tokenized by the fixture tokenizer, or
transformers' own generation-marker mask over the same template; nothing
reads a field back out of the parsed value.
"""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from hub_support import native_builder
from transformers import AutoTokenizer

from dew.data import Files, Loading, presets
from dew.data.chat import ROLES_KEY, Conversation, Role, render_conversation
from dew.data.compile import held
from dew.data.dataset import Checkpointable, DataPartition

TOKENIZERS = Path(__file__).resolve().parent / "fixtures" / "tokenizers"
TOOLS_TOKENIZER = TOKENIZERS / "tiny-tools"
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather now.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}
CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": [{"type": "text", "text": "Weather in Paris and Athens?"}]},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
        {"id": "call_2", "type": "function",
         "function": {"name": "get_weather", "arguments": {"city": "Athens"}}},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "22"},
    {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "25"},
    {"role": "user", "content": "Thanks, which is warmer?"},
    {"role": "assistant", "content": "Athens, at 25.", "reasoning_content": "25 > 22"},
]

TOOLS_TEXT = (
    '<|im_start|>system\nBe brief.\n\nTools:\n' + json.dumps(WEATHER) + '<|im_end|>\n'
    '<|im_start|>user\nWeather in Paris and Athens?<|im_end|>\n'
    '<|im_start|>assistant\n'
    '<tool_call id="call_1">get_weather {"city": "Paris"}</tool_call>'
    '<tool_call id="call_2">get_weather {"city": "Athens"}</tool_call><|im_end|>\n'
    '<|im_start|>tool\n<tool_response id="call_1" name="get_weather">22</tool_response><|im_end|>\n'
    '<|im_start|>tool\n<tool_response id="call_2" name="get_weather">25</tool_response><|im_end|>\n'
    '<|im_start|>user\nThanks, which is warmer?<|im_end|>\n'
    '<|im_start|>assistant\nAthens, at 25.<|im_end|>\n'
)
"""What the tiny-tools template renders for CONVERSATION, written by hand
from its source: every id, name and argument in place, the text part's
text, no content for the tool-call turn."""


def validated(source, seq_len: int):
    """`source`'s conversations rendered and packed one to a row of `seq_len + 1` tokens by the sft
    preset, each one longer cut into pieces, read once in memory to validate on, a row a batch."""
    rows = source.pipe(presets.sft, str(TOOLS_TOKENIZER), seq_len=seq_len, batch=1, bins=1, long="chunk")
    return held(rows.read("memory").loading(Loading(workers=0)))


@pytest.fixture(scope="module")
def tools_tokenizer():
    return AutoTokenizer.from_pretrained(TOOLS_TOKENIZER)


def render(tokenizer, messages, tools=None, where="test"):
    return render_conversation(tokenizer, Conversation.parse(messages, tools, where), where)


def reference_messages():
    """Plain-string HF inputs for the fixed structured fixture."""
    messages = json.loads(json.dumps(CONVERSATION))
    messages[1]["content"] = "Weather in Paris and Athens?"
    messages[2]["tool_calls"][0]["function"]["arguments"] = {"city": "Paris"}
    return messages


def test_structured_packing_resumes_exact_tokens_and_masks(tmp_path, tools_tokenizer):
    """A saved packer position restores the pending typed conversation.

    Each row has different completion text, so replaying a previous row or
    losing an assistant part changes the expected tokens or target mask.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows, expected = [], []
    for answer in ("First answer.", "Second answer.", "Third answer."):
        messages = parquet_conversation()
        messages[-1]["content"] = answer
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = [{"type": "text", "text": content[:2]},
                                      {"type": "text", "text": content[2:]}]
        rows.append({"prompt": messages, "tools": json.dumps([WEATHER])})
        desired = TOOLS_TEXT.replace('Athens, at 25.<|im_end|>', answer + '<|im_end|>')
        hf_rows = reference_messages()
        hf_rows[-1]["content"] = answer
        hf = tools_tokenizer.apply_chat_template(
            hf_rows, tools=[WEATHER], return_dict=True, return_assistant_tokens_mask=True)
        expected.append((tools_tokenizer.encode(desired, add_special_tokens=False),
                         hf["assistant_masks"]))
    path = tmp_path / "sft.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    stream = validated(Files(str(path)), 511)(DataPartition())
    assert isinstance(stream, Checkpointable)

    first = next(stream)
    position = stream.get_state()
    remaining = list(stream)
    restored = validated(Files(str(path)), 511)(DataPartition())
    assert isinstance(restored, Checkpointable)
    restored.set_state(position)
    resumed = list(restored)

    assert len(resumed) == len(remaining) == 2
    for uninterrupted, restarted in zip(remaining, resumed, strict=True):
        assert jax.tree.structure(uninterrupted) == jax.tree.structure(restarted)
        for left, right in zip(jax.tree.leaves(uninterrupted), jax.tree.leaves(restarted), strict=True):
            np.testing.assert_array_equal(left, right)
    for batch, (ids, assistant_mask) in zip([first, *resumed], expected, strict=True):
        np.testing.assert_array_equal(batch["text"][0, :len(ids)], ids)
        np.testing.assert_array_equal(batch["text_roles"][0, :len(ids)] == Role.ASSISTANT,
                                      assistant_mask)
        np.testing.assert_array_equal(batch["text_coordinates"].positions[0, :len(ids)], np.arange(len(ids)))
        np.testing.assert_array_equal(batch["text_coordinates"].segments[0, :len(ids)], 1)
        np.testing.assert_array_equal(batch["text_roles"][0, len(ids):], 0)


# --- the tool template ---------------------------------------------------------

def test_arguments_render_from_their_parsed_object(tools_tokenizer):
    """A JSON-string argument and the same object as a dict render to the
    same tokens, and a different argument renders differently."""
    as_string = json.loads(json.dumps(CONVERSATION))
    as_string[2]["tool_calls"][1]["function"]["arguments"] = '{"city": "Athens"}'
    as_dict = json.loads(json.dumps(CONVERSATION))
    as_dict[2]["tool_calls"][0]["function"]["arguments"] = {"city": "Paris"}
    elsewhere = json.loads(json.dumps(CONVERSATION))
    elsewhere[2]["tool_calls"][0]["function"]["arguments"] = '{"city": "Oslo"}'

    base = render(tools_tokenizer, CONVERSATION, [WEATHER])[0].tolist()
    assert render(tools_tokenizer, as_string, [WEATHER])[0].tolist() == base
    assert render(tools_tokenizer, as_dict, [WEATHER])[0].tolist() == base
    assert render(tools_tokenizer, elsewhere, [WEATHER])[0].tolist() != base


def test_flat_tool_calls_read_like_nested_ones(tools_tokenizer):
    """The Hermes layout, `{id, name, arguments}` without a `function`
    wrapper, renders to the tokens the nested layout renders to."""
    flat = json.loads(json.dumps(CONVERSATION))
    flat[2]["tool_calls"] = [
        {"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}},
        {"id": "call_2", "name": "get_weather", "arguments": {"city": "Athens"}},
    ]

    assert render(tools_tokenizer, flat, [WEATHER])[0].tolist() == render(
        tools_tokenizer, CONVERSATION, [WEATHER])[0].tolist()


def test_typed_parts_render_their_text(tools_tokenizer):
    """Text parts concatenate in order without separators or list repr."""
    as_text = json.loads(json.dumps(CONVERSATION))
    as_text[1]["content"] = "Weather in Paris and Athens?"
    two_parts = json.loads(json.dumps(CONVERSATION))
    two_parts[1]["content"] = [{"type": "text", "text": "Weather in Paris"},
                               {"type": "text", "text": " and Athens?"}]

    expected = tools_tokenizer.encode(TOOLS_TEXT, add_special_tokens=False)
    assert render(tools_tokenizer, as_text, [WEATHER])[0].tolist() == expected
    assert render(tools_tokenizer, two_parts, [WEATHER])[0].tolist() == expected


# --- the Qwen3 training template -----------------------------------------------

# --- packing -------------------------------------------------------------------

def parquet_conversation():
    """CONVERSATION in a shape parquet holds: string content throughout
    (arrow cannot mix text and parts in one column) and JSON-string
    arguments (a struct cannot vary its fields per call)."""
    messages = json.loads(json.dumps(CONVERSATION))
    messages[1]["content"] = "Weather in Paris and Athens?"
    messages[2]["tool_calls"][1]["function"]["arguments"] = '{"city": "Athens"}'
    return messages


def test_rows_without_a_tools_column_render_plain(tmp_path, tools_tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as pq

    messages = parquet_conversation()
    path = tmp_path / "chat.parquet"
    pq.write_table(pa.table({"prompt": [messages]}), path)
    ids, roles = render(tools_tokenizer, messages)

    batch = next(iter(validated(Files(str(path)), 511)(DataPartition())))

    np.testing.assert_array_equal(batch["text"][0][:len(ids)], ids)
    np.testing.assert_array_equal(batch[ROLES_KEY][0][:len(ids)], roles)


def test_jsonl_rows_pack_the_way_the_same_rows_do_in_parquet(tmp_path, tools_tokenizer):
    """A `.jsonl` path is the third reader: one JSON object per line, the
    form a chat corpus is published in, under the same `messages` column and
    with the same tools beside it."""
    messages = parquet_conversation()
    path = tmp_path / "chat.jsonl"
    path.write_text("".join(
        json.dumps({"messages": messages, "tools": [WEATHER]}) + "\n" for _ in range(2)))
    ids, roles = render(tools_tokenizer, messages, [WEATHER])

    batch = next(iter(validated(Files(str(path)), 1023)(DataPartition())))
    np.testing.assert_array_equal(batch["text"][0][:len(ids)], ids)
    np.testing.assert_array_equal(batch[ROLES_KEY][0][:len(ids)], roles)
    # Two conversations, one window: the second starts where the first ends.
    np.testing.assert_array_equal(batch["text"][0][len(ids):2 * len(ids)], ids)
    np.testing.assert_array_equal(batch["text_coordinates"].segments[0][len(ids):2 * len(ids)], 2)


def test_a_prepared_hub_split_packs_the_same_conversations(tmp_path, monkeypatch, tools_tokenizer):
    from types import SimpleNamespace

    import datasets as hf_datasets
    from huggingface_hub import HfApi

    from dew.data import Hub

    messages = parquet_conversation()
    split = hf_datasets.Dataset.from_dict(
        {"messages": [messages], "tools": [json.dumps([WEATHER])], "id": ["row-0"]})
    builder = native_builder({"train": split, "test": split}, tmp_path, tmp_path / "arrow")
    builder.download_and_prepare()
    monkeypatch.setattr(HfApi, "dataset_info", lambda self, repo_id, **kwargs: SimpleNamespace(sha="abc123"))
    monkeypatch.setattr(hf_datasets, "load_dataset_builder", lambda *args, **kwargs: builder)
    source = Hub("acme/chat", revision="abc123", cache_dir=str(tmp_path / "arrow"))
    reader = validated(dataclasses.replace(source, split="test"), 1023)(DataPartition())
    try:
        batch = next(reader)
    finally:
        reader.close()
    ids, roles = render(tools_tokenizer, messages, [WEATHER])
    np.testing.assert_array_equal(batch["text"][0][:len(ids)], ids)
    np.testing.assert_array_equal(batch[ROLES_KEY][0][:len(ids)], roles)


# --- refusals ------------------------------------------------------------------
