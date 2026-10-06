"""Generated tokenizer inputs preserve their encoded IDs and rendered chat."""

import pytest
from test_tools import FIXTURES, load
from transformers import AutoTokenizer


@pytest.mark.parametrize("name", ["tiny-chat", "tiny-tools"])
def test_tokenizers_regenerate_the_committed_ids_and_chat_and_refuse_template_drift(tmp_path, name):
    """Copy template inputs, retrain BPE, and detect changed role markers in both text and IDs."""
    load("make_tiny_tokenizer").main(["--out", str(tmp_path)])
    committed = FIXTURES / "tokenizers" / name
    for file in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert (tmp_path / name / file).read_bytes() == (committed / file).read_bytes()
    expected = AutoTokenizer.from_pretrained(committed, local_files_only=True)
    messages = [{"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1",
                 "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "22"},
                {"role": "assistant", "content": "Thanks!"}]
    for tokenize in (False, True):
        actual = AutoTokenizer.from_pretrained(tmp_path / name, local_files_only=True)
        settings = {"tokenize": tokenize, "add_generation_prompt": True}
        reference = expected.apply_chat_template(messages, **settings)
        assert actual.apply_chat_template(messages, **settings) == reference
        actual.chat_template = actual.chat_template.replace("<|im_start|>", "<|im_end|>")
        with pytest.raises(AssertionError):
            assert actual.apply_chat_template(messages, **settings) == reference
