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

import json
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from dew.data import ChatMessages, Checkpointable, Loading, Prompts
from dew.data.chat import ROLES_KEY, Conversation, Role, render_conversation

TOKENIZERS = Path(__file__).resolve().parent / "fixtures" / "tokenizers"
TOOLS_TOKENIZER = TOKENIZERS / "tiny-tools"
CHAT_TOKENIZER = TOKENIZERS / "tiny-chat"

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


@pytest.fixture(scope="module")
def tools_tokenizer():
    return AutoTokenizer.from_pretrained(TOOLS_TOKENIZER)


@pytest.fixture(scope="module")
def chat_tokenizer():
    return AutoTokenizer.from_pretrained(CHAT_TOKENIZER)


def render(tokenizer, messages, tools=None, where="test"):
    return render_conversation(tokenizer, Conversation.parse(messages, tools, where), where)


def reference_messages():
    """Plain-string HF inputs for the fixed structured fixture."""
    messages = json.loads(json.dumps(CONVERSATION))
    messages[1]["content"] = "Weather in Paris and Athens?"
    messages[2]["tool_calls"][0]["function"]["arguments"] = {"city": "Paris"}
    return messages


def reference_mask(tokenizer):
    """HF generation-marker masks from independently specified text inputs."""
    rendered = tokenizer.apply_chat_template(
        reference_messages(), tools=[WEATHER],
        tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    return np.asarray(rendered["input_ids"]), np.asarray(rendered["assistant_masks"])


@pytest.mark.parametrize("tokenizer_path", [TOOLS_TOKENIZER, CHAT_TOKENIZER])
@pytest.mark.parametrize("storage", ["records", "parquet"])
def test_prompt_sources_keep_structured_chat(tmp_path, tokenizer_path, storage):
    """Prompt generation keeps the SFT conversation and its tool schemas.

    HF text inputs independently specify the expected tokens, including
    reasoning metadata in Qwen and tool ids/names in the tools template.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    messages = parquet_conversation()
    for message in messages:
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
    row = {"prompt": messages, "tools": json.dumps([WEATHER]), "ground_truth": "25"}
    if storage == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "prompts.parquet"
        pq.write_table(pa.Table.from_pylist([row]), path)
        source = Prompts(tokenizer=str(tokenizer_path), path=str(path),
                         max_prompt_len=1024, loading=Loading(workers=0))
    else:
        source = Prompts(tokenizer=str(tokenizer_path), records=(json.dumps(row),),
                         max_prompt_len=1024, loading=Loading(workers=0))

    batch = next(source.load(batch=1).train())

    expected = tokenizer.apply_chat_template(
        reference_messages(), tools=[WEATHER], return_dict=False, add_generation_prompt=True)
    np.testing.assert_array_equal(batch["prompt"][0, -len(expected):], expected)
    assert int(batch["prompt_length"][0]) == len(expected)
    np.testing.assert_array_equal(batch["prompt"][0, :-len(expected)], 0)
    if tokenizer_path == TOOLS_TOKENIZER:
        assert tokenizer.decode(expected) == TOOLS_TEXT + '<|im_start|>assistant\n'


def test_prompt_media_is_refused_before_truncation():
    row = {"prompt": [{"role": "user", "content": [
        {"type": "image", "url": "file:///map.png"},
        {"type": "text", "text": "Weather in Paris?"}]}]}
    spec = Prompts(tokenizer=str(TOOLS_TOKENIZER), records=(json.dumps(row),),
                   max_prompt_len=4, loading=Loading(workers=0))
    with pytest.raises(ValueError, match=r"row 0 message 0.*image.*processor"):
        next(spec.load(batch=1).train())


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
    spec = ChatMessages(tokenizer=str(TOOLS_TOKENIZER), path=str(path), val_path=str(path),
                        seq_len=511, packing_bins=1, val_batches=None,
                        loading=Loading(workers=0))
    data = spec.load(batch=1)
    assert data.val is not None
    stream = data.val()
    assert isinstance(stream, Checkpointable)

    first = next(stream)
    position = stream.get_state()
    remaining = list(stream)
    fresh = spec.load(batch=1)
    assert fresh.val is not None
    restored = fresh.val()
    assert isinstance(restored, Checkpointable)
    restored.set_state(position)
    resumed = list(restored)

    assert len(resumed) == len(remaining) == 2
    for uninterrupted, restarted in zip(remaining, resumed, strict=True):
        for key in uninterrupted:
            np.testing.assert_array_equal(uninterrupted[key], restarted[key])
    for batch, (ids, assistant_mask) in zip([first, *resumed], expected, strict=True):
        np.testing.assert_array_equal(batch["text"][0, :len(ids)], ids)
        np.testing.assert_array_equal(batch["text_roles"][0, :len(ids)] == Role.ASSISTANT,
                                      assistant_mask)
        np.testing.assert_array_equal(batch["text_positions"][0, :len(ids)], np.arange(len(ids)))
        np.testing.assert_array_equal(batch["text_segment_ids"][0, :len(ids)], 1)
        np.testing.assert_array_equal(batch["text_roles"][0, len(ids):], 0)


def decode(tokenizer, ids, roles, role):
    return tokenizer.decode(ids[roles == role])


# --- the tool template ---------------------------------------------------------

def test_tool_calls_render_to_the_template_ids(tools_tokenizer):
    """The ids are the hand-written render tokenized, and the assistant mask
    is transformers' generation-marker mask: both calls with their ids and
    arguments, a string argument and a dict argument alike, and the text
    part's text, with the tool-call turn contributing no content."""
    conversation = Conversation.parse(CONVERSATION, [WEATHER])

    ids, roles = render_conversation(tools_tokenizer, conversation, "tiny-tools")

    expected = tools_tokenizer.encode(TOOLS_TEXT, add_special_tokens=False)
    assert ids.tolist() == expected
    hf_ids, hf_mask = reference_mask(tools_tokenizer)
    np.testing.assert_array_equal(ids, hf_ids)
    np.testing.assert_array_equal((roles == Role.ASSISTANT).astype(np.int64), hf_mask)
    assert decode(tools_tokenizer, ids, roles, Role.ASSISTANT) == (
        '<tool_call id="call_1">get_weather {"city": "Paris"}</tool_call>'
        '<tool_call id="call_2">get_weather {"city": "Athens"}</tool_call><|im_end|>\n'
        'Athens, at 25.<|im_end|>\n')
    assert decode(tools_tokenizer, ids, roles, Role.TOOL) == (
        '<|im_start|>tool\n<tool_response id="call_1" name="get_weather">22</tool_response><|im_end|>\n'
        '<|im_start|>tool\n<tool_response id="call_2" name="get_weather">25</tool_response><|im_end|>\n')
    assert "<unk>" not in tools_tokenizer.decode(ids)


def test_ids_and_names_change_the_tool_span(tools_tokenizer):
    """A response answering another call, or another function, renders to
    different tokens inside the tool span and nowhere else."""
    ids, roles = render(tools_tokenizer, CONVERSATION, [WEATHER])
    relinked = json.loads(json.dumps(CONVERSATION))
    relinked[3]["tool_call_id"] = "call_9"
    renamed = json.loads(json.dumps(CONVERSATION))
    renamed[4]["name"] = "get_wind"

    for changed in (relinked, renamed):
        other_ids, other_roles = render(tools_tokenizer, changed, [WEATHER])
        assert other_ids.tolist() != ids.tolist()
        assert decode(tools_tokenizer, other_ids, other_roles, Role.ASSISTANT) == decode(
            tools_tokenizer, ids, roles, Role.ASSISTANT)
        assert decode(tools_tokenizer, other_ids, other_roles, Role.USER) == decode(
            tools_tokenizer, ids, roles, Role.USER)
    assert decode(tools_tokenizer, *render(tools_tokenizer, relinked, [WEATHER]), Role.TOOL).count(
        'id="call_9"') == 1


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


def test_tool_schemas_render_into_the_system_span(tools_tokenizer):
    """With schemas the system block carries their JSON; without, it is the
    plain system message, and the assistant span is the same either way."""
    with_tools = render(tools_tokenizer, CONVERSATION, [WEATHER])
    without = render(tools_tokenizer, CONVERSATION)

    assert json.dumps(WEATHER) in decode(tools_tokenizer, *with_tools, Role.SYSTEM)
    assert decode(tools_tokenizer, *without, Role.SYSTEM) == '<|im_start|>system\nBe brief.<|im_end|>\n'
    assert decode(tools_tokenizer, *with_tools, Role.ASSISTANT) == decode(
        tools_tokenizer, *without, Role.ASSISTANT)


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


@pytest.mark.parametrize("body", ["{{ message.content }}",
                                  "{% if message.content is string %}{{ message.content }}{% endif %}"])
def test_text_parts_reach_string_templates_as_text(body):
    """Both direct interpolation and string-guarded templates retain text.

    Before the fix they trained list repr and empty bodies respectively.
    The desired text and the HF assistant mask use plain-string messages.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOOLS_TOKENIZER)
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message.role + '\n' }}"
        "{% if message.role == 'assistant' %}{% generation %}"
        + body + "{{ '<|im_end|>\n' }}{% endgeneration %}"
        "{% else %}" + body + "{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Two "},
                                      {"type": "text", "text": "plus two?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Four"},
                                           {"type": "text", "text": "."}]},
    ]
    desired = [dict(role="user", content="Two plus two?"),
               dict(role="assistant", content="Four.")]
    expected_text = ("<|im_start|>user\nTwo plus two?<|im_end|>\n"
                     "<|im_start|>assistant\nFour.<|im_end|>\n")

    ids, roles = render(tokenizer, messages)

    assert ids.tolist() == tokenizer.encode(expected_text, add_special_tokens=False)
    expected = tokenizer.apply_chat_template(
        desired, return_dict=True, return_assistant_tokens_mask=True)
    np.testing.assert_array_equal(roles == Role.ASSISTANT, expected["assistant_masks"])
    assert decode(tokenizer, ids, roles, Role.ASSISTANT) == 'Four.<|im_end|>\n'


@pytest.mark.parametrize("parts, expected_text", [
    ([{"type": "image", "url": "file:///map.png"}], "<image>"),
    ([{"type": "text", "text": "Read this map"},
      {"type": "image", "url": "file:///map.png"}], "Read this map<image>"),
])
def test_media_content_requires_a_processor(tools_tokenizer, parts, expected_text):
    """Text tokenization refuses media without changing the stored parts."""
    conversation = Conversation.parse([{"role": "user", "content": parts}])
    with pytest.raises(ValueError, match=r"row 7 message 0.*image.*processor"):
        render_conversation(tools_tokenizer, conversation, "chat.parquet row 7")
    # A processor can still consume the structured content after refusal.
    template = ("{% for part in messages[0].content %}"
                "{% if part.type == 'image' %}{{ '<image>' }}"
                "{% elif part.type == 'text' %}{{ part.text }}{% endif %}{% endfor %}")
    raw_ids = tools_tokenizer.apply_chat_template(
        conversation.rows(), chat_template=template, return_dict=False)
    assert raw_ids == tools_tokenizer.encode(expected_text, add_special_tokens=False)


# --- the Qwen3 training template -----------------------------------------------

def test_the_qwen_template_merges_parallel_tool_responses(chat_tokenizer):
    """Two tool responses in a row render as one user block, so the second
    response's render rewrites the first's closing tokens. The ids and the
    assistant mask match transformers over the same template, and the tool
    span is the merged block."""
    conversation = Conversation.parse(CONVERSATION, [WEATHER])

    ids, roles = render_conversation(chat_tokenizer, conversation, "tiny-chat")

    hf_ids, hf_mask = reference_mask(chat_tokenizer)
    np.testing.assert_array_equal(ids, hf_ids)
    np.testing.assert_array_equal((roles == Role.ASSISTANT).astype(np.int64), hf_mask)
    assert decode(chat_tokenizer, ids, roles, Role.TOOL) == (
        '<|im_start|>user\n<tool_response>\n22\n</tool_response>\n'
        '<tool_response>\n25\n</tool_response><|im_end|>\n')
    assert decode(chat_tokenizer, ids, roles, Role.ASSISTANT).startswith(
        '<think>\n\n</think>\n\n<tool_call>\n')


def test_extra_keys_reach_the_template(chat_tokenizer):
    """`reasoning_content` is not a typed field, and the Qwen template reads
    it: the assistant span carries the thinking block, and dropping the key
    empties it."""
    ids, roles = render(chat_tokenizer, CONVERSATION, [WEATHER])
    silent = json.loads(json.dumps(CONVERSATION))
    del silent[6]["reasoning_content"]
    silent_ids, silent_roles = render(chat_tokenizer, silent, [WEATHER])

    assert decode(chat_tokenizer, ids, roles, Role.ASSISTANT).endswith(
        '<think>\n25 > 22\n</think>\n\nAthens, at 25.<|im_end|>\n')
    assert decode(chat_tokenizer, silent_ids, silent_roles, Role.ASSISTANT).endswith(
        '<think>\n\n</think>\n\nAthens, at 25.<|im_end|>\n')


# --- packing -------------------------------------------------------------------

def parquet_conversation():
    """CONVERSATION in a shape parquet holds: string content throughout
    (arrow cannot mix text and parts in one column) and JSON-string
    arguments (a struct cannot vary its fields per call)."""
    messages = json.loads(json.dumps(CONVERSATION))
    messages[1]["content"] = "Weather in Paris and Athens?"
    messages[2]["tool_calls"][1]["function"]["arguments"] = '{"city": "Athens"}'
    return messages


def test_structured_rows_pack_with_their_tools(tmp_path, tools_tokenizer):
    """A parquet row with a `tools` column packs to the ids and roles the
    renderer produces, with every other message field arriving as a parquet
    null and reading as absent."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    messages = parquet_conversation()
    path = tmp_path / "chat.parquet"
    pq.write_table(pa.table({"prompt": [messages], "tools": [json.dumps([WEATHER])]}), path)
    ids, roles = render(tools_tokenizer, messages, [WEATHER])
    window = 512

    data = ChatMessages(tokenizer=str(TOOLS_TOKENIZER), path=str(path), val_path=str(path),
                        seq_len=window - 1, packing_bins=1,
                        loading=Loading(workers=0)).load(batch=1)
    assert data.val is not None
    batches = list(data.val())

    assert len(batches) == 1
    batch = batches[0]
    np.testing.assert_array_equal(batch["text"][0][:len(ids)], ids)
    np.testing.assert_array_equal(batch[ROLES_KEY][0][:len(ids)], roles)
    np.testing.assert_array_equal(batch["text"][0][len(ids):], 0)
    np.testing.assert_array_equal(batch[ROLES_KEY][0][len(ids):], 0)
    np.testing.assert_array_equal(batch["text_segment_ids"][0][:len(ids)], 1)
    assert tools_tokenizer.decode(batch["text"][0][batch[ROLES_KEY][0] == Role.ASSISTANT]) == (
        decode(tools_tokenizer, ids, roles, Role.ASSISTANT))


def test_rows_without_a_tools_column_render_plain(tmp_path, tools_tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as pq

    messages = parquet_conversation()
    path = tmp_path / "chat.parquet"
    pq.write_table(pa.table({"prompt": [messages]}), path)
    ids, roles = render(tools_tokenizer, messages)

    data = ChatMessages(tokenizer=str(TOOLS_TOKENIZER), path=str(path), val_path=str(path),
                        seq_len=511, packing_bins=1, loading=Loading(workers=0)).load(batch=1)
    assert data.val is not None
    batch = next(iter(data.val()))

    np.testing.assert_array_equal(batch["text"][0][:len(ids)], ids)
    np.testing.assert_array_equal(batch[ROLES_KEY][0][:len(ids)], roles)


# --- refusals ------------------------------------------------------------------

def test_a_template_refusal_names_the_row(tools_tokenizer):
    """The template concatenates user content with a closing delimiter.
    Null content fails with the source row and the template's words."""
    silent = json.loads(json.dumps(CONVERSATION))
    silent[5]["content"] = None
    with pytest.raises(ValueError, match="row 7: the chat template refused"):
        render(tools_tokenizer, silent, [WEATHER], where="chat.parquet row 7")


def test_a_role_the_template_does_not_read_is_refused(tools_tokenizer):
    """The tools template has no developer branch, so the turn renders to
    nothing; that is refused, not silently trained around. Without tools,
    since a tools block renders before any message and would count as the
    first message's tokens whatever its role."""
    with_developer = [{"role": "developer", "content": "Answer in French."}] + CONVERSATION
    with pytest.raises(ValueError, match=r"message 0 \(developer\) to no tokens"):
        render(tools_tokenizer, with_developer)


def test_an_assistant_turn_rewritten_by_later_turns_is_refused(tools_tokenizer):
    """A template that keeps thinking only on the last assistant turn
    renders an earlier turn differently once followed; masking it from the
    whole conversation would drop its thinking silently, so it fails."""
    tokenizer = AutoTokenizer.from_pretrained(TOOLS_TOKENIZER)
    tokenizer.chat_template = (
        "{%- for message in messages %}"
        "{%- if message.role == 'assistant' %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- if loop.last and message.reasoning_content is string %}"
        "{{- '<think>' + message.reasoning_content + '</think>' }}{%- endif %}"
        "{{- message.content + '<|im_end|>\\n' }}"
        "{%- else %}"
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
        "{%- endif %}{%- endfor %}"
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}")
    messages = [
        {"role": "user", "content": "Two plus two?"},
        {"role": "assistant", "content": "Four.", "reasoning_content": "2 + 2"},
        {"role": "user", "content": "And three?"},
        {"role": "assistant", "content": "Five."},
    ]

    with pytest.raises(ValueError, match=r"message 1 \(assistant\) differently"):
        render(tokenizer, messages)
    ids, roles = render(tokenizer, messages[2:])
    assert decode(tokenizer, ids, roles, Role.ASSISTANT) == 'Five.<|im_end|>\n'


def test_a_misplaced_structured_field_is_refused(tools_tokenizer):
    user_call = [{"role": "user", "content": "hi",
                  "tool_calls": [{"name": "get_weather", "arguments": {}}]}]
    with pytest.raises(ValueError, match="only an assistant message calls tools"):
        render(tools_tokenizer, user_call)
    user_id = [{"role": "user", "content": "hi", "tool_call_id": "call_1"}]
    with pytest.raises(ValueError, match="only a tool response carries tool_call_id"):
        render(tools_tokenizer, user_id)


def test_a_malformed_tool_call_is_refused(tools_tokenizer):
    def with_call(call):
        messages = json.loads(json.dumps(CONVERSATION[:3]))
        messages[2]["tool_calls"] = [call]
        return messages

    with pytest.raises(ValueError, match="arguments that are not JSON"):
        render(tools_tokenizer, with_call({"id": "c", "name": "get_weather", "arguments": "{oops"}))
    with pytest.raises(ValueError, match="arguments that are not an object"):
        render(tools_tokenizer, with_call({"id": "c", "name": "get_weather", "arguments": "[1]"}))
    with pytest.raises(ValueError, match=r"unknown fields \['index'\]"):
        render(tools_tokenizer, with_call(
            {"id": "c", "index": 0, "function": {"name": "get_weather", "arguments": {}}}))
    with pytest.raises(ValueError, match="a tool call's name is a string"):
        render(tools_tokenizer, with_call({"id": "c", "arguments": {}}))
    with pytest.raises(ValueError, match="a content part's type is a string"):
        render(tools_tokenizer, [{"role": "user", "content": [{"text": "hi"}]}])
    with pytest.raises(ValueError, match="tools is not JSON"):
        render(tools_tokenizer, CONVERSATION, "{oops")
