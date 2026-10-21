"""A dataset's facts (`Dataset.facts`), read without opening a reader: what its stores state, then what
each step makes of them, in order. They say what the first batch the chain makes holds."""

import numpy as np
import pytest

from dew.data import Loading, Memory, from_iter, mix, presets
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
