"""A joint row's conversation markup is its backbone's own: the tokenizer's chat template renders the
system message, the user's turn around the state and the schema, and the assistant's opening
(`dew.decision.layout.Framing`), so every markup token is that tokenizer's own special token, with no
other family's markup spelled out as text. A tokenizer with no template frames the row as plain text.
Qwen3.5's template with thinking off writes the release's rows exactly
(tests/test_decision_joint_checkpoint.py::test_the_joint_layout_lays_out_what_the_release_does)."""

import json
from pathlib import Path

import pytest

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
    assert JOINT_SYSTEM in text and text.rstrip().endswith(JOINT_PREFILL)


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
    assert (before, after) == (f"{JOINT_SYSTEM}\n\n", f"\n\n{JOINT_PREFILL}")
    specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    [row] = JointLayout().rows(tokenizer, specials, "one two three", QUESTIONS)
    assert not any(markup in tokenizer.decode(row.tokens) for markup in CHATML)
    templated = HFTokenizer(str(LOCAL["qwen3.5"]), local_files_only=True)
    assert Framing(template=False).around(templated) == (before, after)


@pytest.mark.parametrize(("name", "written"), [("qwen3.5", "<|vision_start|><|image_pad|><|vision_end|>"),
                                               ("llama4", "<|image|>")])
def test_an_image_stands_as_its_processors_own_image_part(name, written):
    """A row's image is the text its processor's own chat template writes for an image part, which
    the processor expands to the image's tokens: Qwen-VL's vision span, Llama 4's image token."""
    from transformers import AutoProcessor

    from dew.data.chat import image_text

    processor = AutoProcessor.from_pretrained(str(LOCAL[name]), local_files_only=True)
    assert image_text(processor, where=name) == written
