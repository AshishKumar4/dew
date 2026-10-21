"""A dataset's facts (`Dataset.facts`), read without opening a reader: what its stores state, then what
each step makes of them, in order. They say what the first batch the chain makes holds."""

from pathlib import Path

import numpy as np
import pytest
from token_support import write_token_corpus

from dew.data import Loading, Memory, from_iter, mix, presets, text
from dew.data.compile import compile
from dew.data.dataset import DataPartition

LOADING = Loading(workers=0, threads=1, read_buffer=1)


def _first(chain):
    """The shapes of the first batch `chain` makes, each field's without the batch."""
    reader = compile(chain, read="indexed", loading=LOADING).train(DataPartition())
    try:
        return {name: list(np.shape(value)[1:]) for name, value in next(reader).items()}
    finally:
        reader.close()


def _images(count: int = 8) -> Memory:
    return Memory([{"image": np.random.default_rng(index).integers(0, 256, (12, 9, 3), np.uint8),
                    "label": index} for index in range(count)])


@pytest.mark.parametrize("augment", [False, True], ids=["plain", "augmented"])
def test_the_images_presets_facts_are_its_first_batchs_image(augment):
    """Images resized to eight-pixel squares, augmented on the device or not, four a batch: the
    facts say so, as the first batch holds them."""
    chain = _images().shuffle(0).repeat().pipe(presets.images, 8, batch=4, augment=augment)
    facts = chain.facts()
    assert facts["batch"] == 4
    assert facts["fields"]["image"] == [8, 8, 3] == _first(chain)["image"]


@pytest.mark.parametrize("pack", ["first-fit", None])
def test_the_lm_presets_facts_are_its_tokenizer_and_its_rows(pack):
    """Documents tokenized by bytes and laid out in rows of sixteen inputs, packed or cut into
    windows: the facts name the tokenizer, its vocabulary and the rows the first batch holds."""
    documents = Memory([{"text": "word " * (3 + index)} for index in range(12)])
    chain = documents.shuffle(0).repeat().pipe(presets.lm, "byte", seq_len=16, batch=2, pack=pack)
    facts = chain.facts()
    assert (facts["tokenizer"], facts["seq_len"], facts["packed"]) == ("byte", 16, pack is not None)
    assert facts["vocab_size"] == 256
    assert facts["fields"]["text"] == [17] == _first(chain)["text"]


def test_a_rectangular_resize_states_its_height_width_and_the_decoded_channels():
    """Images decoded to RGB and resized to six by ten: the facts say so, as the first batch holds
    them; resized without a decode, their channels are unknown, so their shape is."""
    from dew.data import images

    decoded = _images().filter_map(images.DecodeImage("image")).map(images.Resize((6, 10))).repeat().batch(2)
    assert decoded.facts()["fields"]["image"] == [6, 10, 3] == _first(decoded)["image"]
    resized = _images().map(images.Resize((6, 10))).repeat().batch(2)
    assert "image" not in resized.facts().get("fields", {})


def test_a_rename_after_a_preset_moves_its_facts_with_it():
    """Images renamed after the preset: the facts hold the pixels under the new name, none under
    the old."""
    chain = _images().shuffle(0).repeat().pipe(presets.images, 8, batch=4).rename_columns({"image": "pixels"})
    fields = chain.facts()["fields"]
    assert "image" not in fields
    assert fields["pixels"] == [8, 8, 3] == _first(chain)["pixels"]


def _shifted(row):
    return {**row, "image": row["image"][:4]}


def test_a_step_that_says_nothing_of_what_it_makes_leaves_the_fields_unknown():
    """A step of the caller's after the preset that changes an image's shape and declares nothing:
    the facts claim no field's shape, though they keep the batch."""
    chain = _images().shuffle(0).repeat().pipe(presets.images, 8, batch=4).map(_shifted)
    facts = chain.facts()
    assert "fields" not in facts
    assert facts["batch"] == 4


def test_a_mixtures_facts_are_those_its_corpora_share():
    """Two token corpora of one tokenizer, one in rows of sixteen and one of eight: the mixture's
    facts keep the tokenizer and drop the row length the two state differently."""
    def corpus(seq_len, name):
        rows = Memory([{"text": "word " * (3 + index)} for index in range(6)])
        return rows.named(name).shuffle(0).repeat().pipe(presets.lm, "byte", seq_len=seq_len, batch=2)

    alike = mix([corpus(16, "a"), corpus(16, "b")]).facts()
    assert (alike["tokenizer"], alike["seq_len"], alike["fields"]["text"]) == ("byte", 16, [17])
    apart = mix([corpus(16, "a"), corpus(8, "b")]).facts()
    assert apart["tokenizer"] == "byte"
    assert "seq_len" not in apart and "text" not in apart.get("fields", {})


def _unopenable():
    raise AssertionError("facts() opened the store")


def test_facts_open_no_reader():
    """A store that fails as soon as it is read: its chain's facts come without reading it."""
    stream = from_iter(_unopenable, name="unopenable")
    facts = stream.pipe(presets.lm, "byte", seq_len=8, batch=2).facts()
    assert facts["seq_len"] == 8


def _held_fields(chain, read: str = "indexed"):
    """Each field the first batch of `chain` holds with its shape, without the batch."""
    reader = compile(chain, read=read, loading=LOADING).train(DataPartition())
    try:
        return {name: list(np.shape(value)[1:]) for name, value in next(reader).items()
                if not isinstance(value, (str, bytes))}
    finally:
        reader.close()


def test_the_prompt_presets_facts_are_its_first_batchs_rows():
    """Prompt rows rendered to eight ids with their reward columns sixteen bytes: every field the
    facts shape is the first batch's, the rendered prompt and its reward columns among them."""
    rows = Memory([{"prompt": "hi", "ground_truth": "an answer", "data_source": "rule"} for _ in range(4)])
    chain = rows.repeat().pipe(presets.prompts, "byte", max_prompt_len=8, batch=2, reward_bytes=16)
    facts = chain.facts()
    held = _held_fields(chain)
    assert facts["fields"]["prompt"] == [8] == held["prompt"]
    assert facts["fields"]["ground_truth"] == [16] == held["ground_truth"]
    assert all(held[name] == shape for name, shape in facts["fields"].items())


def test_the_preference_presets_facts_are_its_first_batchs_rows():
    """Chosen and rejected completions laid out in rows of twelve: the facts shape the ids and the
    completion mask as the first batch holds them."""
    rows = Memory([{"chosen": [1, 2, 3], "rejected": [4, 5]} for _ in range(4)])
    chain = rows.repeat().pipe(presets.preferences, seq_len=12, batch=2)
    facts = chain.facts()
    held = _held_fields(chain)
    assert all(held[name] == shape for name, shape in facts["fields"].items())
    assert facts["fields"]["input_ids"] == [2, 12] and facts["seq_len"] == 12


def test_a_tokenized_stores_facts_name_its_end_of_document():
    """Documents tokenized by bytes and ended by EOS, as the byte tokenizer ends them by default:
    the facts name its EOS; not ended, none."""
    documents = Memory([{"text": "a word"}])
    assert documents.map(text.tokenize("byte", eos=True)).facts()["eos_id"] == 255
    assert documents.map(text.tokenize("byte")).facts()["eos_id"] == 255
    assert documents.map(text.tokenize("byte", eos=False)).facts()["eos_id"] is None


def test_a_holdouts_part_states_what_its_store_states(tmp_path):
    """A token store held out in part: each part's facts name the store's tokenizer and vocabulary,
    without counting the store's rows, which neither part holds all of."""
    from dew.data.sources import TokenFiles

    corpus = write_token_corpus(tmp_path, np.arange(1, 100, dtype=np.uint16), eos_id=0)
    train, held = TokenFiles(corpus.train).holdout(records=1)
    for part in (train, held):
        facts = part.facts()
        assert (facts["tokenizer"], facts["vocab_size"]) == ("byte", 256)
        assert "rows" not in facts


def test_the_sft_presets_facts_are_its_tokenizer_and_its_rows():
    """Conversations rendered by the byte tokenizer's template and packed into rows of sixteen
    inputs: the facts name the tokenizer and the rows' length, and shape the ids and speakers as
    the first batch holds them."""
    tokenizer = str(Path(__file__).parent / "fixtures" / "tokenizers" / "tiny-chat")
    conversations = Memory([{"messages": [{"role": "user", "content": "hi " * (1 + index)},
                                          {"role": "assistant", "content": "hello"}]} for index in range(6)])
    chain = conversations.repeat().pipe(presets.sft, tokenizer, seq_len=16, batch=2)
    facts = chain.facts()
    held = _held_fields(chain)
    assert (facts["tokenizer"], facts["seq_len"]) == (tokenizer, 16)
    assert facts["fields"]["text"] == [17] == held["text"]
    assert facts["fields"]["text_roles"] == [17] == held["text_roles"]


def test_a_store_not_present_states_nothing_and_the_presets_facts_stand(tmp_path):
    """A TFDS split never prepared here and a directory of clips not yet fetched state no facts of
    their own, so a chain over either says what its steps make: the images preset's squares, the
    video preset's frames, and no class names or rows."""
    from dew.data import TFDS, Files
    from dew.data.sources.av_utils import FPS

    images = TFDS("never_prepared", path=str(tmp_path / "tfds")).pipe(presets.images, 16, batch=4).facts()
    assert images["fields"]["image"] == [16, 16, 3] and "class_names" not in images and "rows" not in images
    clips = Files(str(tmp_path / "clips"), format="files").pipe(
        presets.video, 4 / FPS, frames=4, fps=FPS, size=16, batch=2).facts()
    assert clips["fields"]["video"] == [4, 16, 16, 3]


def test_host_flips_jitters_crops_and_captions_keep_the_images_shape_their_steps_make():
    """A host flip, colour jitter and fixed or drawn caption leave each image as its shape was; a
    centre crop makes it the crop's square. A run over such a chain builds its model from these."""
    from dew.data import images

    pictures = Memory([{"image": np.zeros((20, 24, 3), np.uint8), "label": 0}])
    augmented = pictures.pipe(presets.images, 16, batch=4, augment=(images.Flip(), images.ColorJitter()),
                              caption=images.caption(templates=("a {}",), classes=("rose",)))
    assert augmented.facts()["fields"]["image"] == [16, 16, 3]
    cropped = pictures.pipe(presets.images, 16, batch=None, caption=images.caption("a photo")).map(
        images.CenterCrop(8))
    assert cropped.facts()["fields"]["image"] == [8, 8, 3]
