"""A joint row's conversation markup is its backbone's own: the tokenizer's chat template renders the
system message, the user's turn around the state and the schema, and the assistant's opening
(`dew.decision.layout.Framing`), so every markup token is that tokenizer's own special token, with no
other family's markup spelled out as text. A tokenizer with no template frames the row as plain text.
Qwen3.5's template with thinking off writes the release's rows exactly
(tests/test_decision_joint_checkpoint.py::test_the_joint_layout_lays_out_what_the_release_does)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dew._records import JSON
from dew.data.text import ByteTokenizer, HFTokenizer
from dew.decision import Choice, JointLayout, Noul
from dew.decision.layout import JOINT_PREFILL, JOINT_SYSTEM, Framing, Specials

FIXTURES = Path(__file__).parent / "fixtures"

QUESTIONS = {"pick": Choice("Which one?", ["a", "b", "c"]), "holds": Noul("Does it hold?")}

LOCAL = {"qwen3.5": FIXTURES / "hf" / "qwen38-dense-tiny", "llama4": FIXTURES / "hf" / "llama4-native-tiny"}
"""Tokenizers with a chat template of their own: Qwen3.5's ChatML with a thinking switch, and Llama 4's
headers, which hold no ChatML token."""

HUB = {"dory": FIXTURES / "hf" / "dory-1.0-4b", "nemotron-h": FIXTURES / "hf" / "nemotron-h-4b"}
"""Checkpoints whose tokenizer is read from the Hub at the commit their `source.json` pins."""

CHATML = ("<|im_start|>", "<|im_end|>")


def pinned(directory: Path) -> HFTokenizer:
    source = json.loads((directory / "source.json").read_text())
    return HFTokenizer(f"{source['repo']}@{source['revision']}")


def assert_markup_is_its_own(tokenizer: HFTokenizer) -> None:
    """A joint row over `tokenizer`: each special token the template's framing writes is that token's
    one id in the row, as often as the framing writes it, and no ChatML spelled as text where the
    vocabulary has no ChatML token."""
    layout = JointLayout()
    [row] = layout.rows(tokenizer, Specials.of(tokenizer), "one two three", QUESTIONS)
    before, after = layout.framing.around(tokenizer)
    from transformers import PreTrainedTokenizerFast

    loaded = tokenizer.tokenizer
    assert isinstance(loaded, PreTrainedTokenizerFast)
    vocabulary = loaded.get_added_vocab()
    written = {token: (before + after).count(token) for token in vocabulary if token in before + after}
    assert written, "the template frames the row with no special token of its own"
    for token, count in written.items():
        assert list(row.tokens).count(vocabulary[token]) >= count, token
    text = tokenizer.decode(row.tokens)
    for markup in CHATML:
        if markup not in vocabulary:
            assert markup not in text
    # The row opens with the framing's own tokens and closes with them (a tiny vocabulary reads the
    # words as unknowns, so the tokens, not the text, are what holds).
    opening = tokenizer.encode(f"{before}STATE:\n", add_special_tokens=False)
    closing = tokenizer.encode("\n" + after, add_special_tokens=False)
    assert list(row.tokens[:len(opening)]) == opening
    assert list(row.tokens[-len(closing):]) == closing


@pytest.mark.parametrize("name", sorted(LOCAL))
def test_a_joint_rows_markup_is_its_tokenizers_own_special_tokens(name):
    assert_markup_is_its_own(HFTokenizer(str(LOCAL[name]), local_files_only=True))


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(HUB))
def test_a_joint_rows_markup_is_its_hub_tokenizers_own_special_tokens(name):
    assert_markup_is_its_own(pinned(HUB[name]))


def test_a_tokenizer_without_a_template_frames_a_row_as_plain_text():
    """A byte vocabulary has no chat template, so its row is the system message, the user's content
    and the decisions' heading as plain text, with no markup of any family; so is a templated
    tokenizer's row where the framing asks for no template."""
    tokenizer = ByteTokenizer()
    before, after = Framing().around(tokenizer)
    assert before.startswith(JOINT_SYSTEM) and after.endswith(JOINT_PREFILL)
    specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    [row] = JointLayout().rows(tokenizer, specials, "one two three", QUESTIONS)
    assert not any(markup in tokenizer.decode(row.tokens) for markup in CHATML)
    templated = HFTokenizer(str(LOCAL["qwen3.5"]), local_files_only=True)
    assert Framing(template=False).around(templated) == (before, after)


@pytest.mark.parametrize("name", sorted(LOCAL))
def test_an_image_stands_as_its_processors_own_image_part(name):
    """A row's image is the text its processor's own chat template writes for an image part, which
    holds the processor's own image token, and which the processor expands to the image's tokens: the
    processed row is longer than the written text's own tokens."""
    from transformers import AutoProcessor

    from dew.data.chat import image_text

    processor = AutoProcessor.from_pretrained(str(LOCAL[name]), local_files_only=True)
    written = image_text(processor, where=name)
    assert processor.image_token in written
    picture = Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    expanded = processor(text=[written], images=[picture])
    own = processor.tokenizer(written, add_special_tokens=False)["input_ids"]
    assert np.asarray(expanded["input_ids"]).shape[-1] > len(own)


def test_a_framing_reads_a_chat_template_of_its_own_and_its_variables_and_its_record_reloads_them():
    """A framing's own Jinja chat template frames a row in place of the tokenizer's (a base model's,
    which has none, or a templated one's, overridden), its variables reach the template, and the
    layout's record carries both, so a run that reloads its layout frames its rows alike."""
    from dew.data.chat import CONTENT
    from dew.registry import from_record, to_record

    template = ("{% for message in messages %}[{{ message.role }}{{ ' ' + mark if mark else '' }}]"
                "{{ message.content }}{% endfor %}{% if add_generation_prompt %}[reply]{% endif %}")
    templated = HFTokenizer(str(LOCAL["qwen3.5"]), local_files_only=True)
    framing = Framing(chat_template=template, options={"mark": "x"})
    before, after = framing.around(templated)
    direct = templated.tokenizer.apply_chat_template(
        [{"role": "system", "content": JOINT_SYSTEM}, {"role": "user", "content": CONTENT}],
        chat_template=template, mark="x", tokenize=False, add_generation_prompt=True)
    assert isinstance(direct, str) and before + CONTENT + after == direct + JOINT_PREFILL
    assert not any(markup in before + after for markup in CHATML)
    layout = JointLayout(framing=framing)
    reloaded = from_record(JointLayout, to_record(layout, JointLayout))
    assert reloaded == layout
    specials = Specials.of(templated)
    rows = [laid.rows(templated, specials, "one two", QUESTIONS) for laid in (reloaded, layout)]
    assert rows[0] == rows[1]



def test_a_framing_refuses_what_it_would_otherwise_leave_unread():
    """A template variable that names one of the template call's own controls, a framing's own
    template over a tokenizer that renders none, and a framing's own template with templates off are
    each refused, rather than overridden or left unread; a template's variables keep their order."""
    with pytest.raises(ValueError):
        Framing(options={"enable_thinking": True})
    with pytest.raises(ValueError):
        Framing(options={"chat_template": "{{ messages }}"})
    with pytest.raises(ValueError):
        Framing(options={"tokenizer_kwargs": {}})
    with pytest.raises(ValueError):
        Framing(chat_template="{{ messages }}", template=False)
    with pytest.raises(ValueError):
        Framing(chat_template="{{ messages }}").around(ByteTokenizer())
    walked = ("{% for message in messages %}{{ message.content }}{% endfor %}"
              "{% for key, value in labels.items() %}<{{ key }}>{% endfor %}")
    from dew.data.chat import CONTENT, framed

    templated = HFTokenizer(str(LOCAL["qwen3.5"]), local_files_only=True)
    with pytest.raises(ValueError):
        framed(templated.tokenizer, JOINT_SYSTEM, where="direct", options={"tokenizer_kwargs": {}})
    rendered = []
    orders: list[dict[str, JSON]] = [{"z": 1, "a": 2}, {"a": 2, "z": 1}]
    for labels in orders:
        before, after = Framing(chat_template=walked, options={"labels": labels}).around(templated)
        direct = templated.tokenizer.apply_chat_template(
            [{"role": "system", "content": JOINT_SYSTEM}, {"role": "user", "content": CONTENT}],
            chat_template=walked, labels=labels, tokenize=False, add_generation_prompt=True)
        assert isinstance(direct, str) and before + CONTENT + after == direct + JOINT_PREFILL
        rendered.append(after)
    assert rendered[0] != rendered[1]
