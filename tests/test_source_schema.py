"""The one owner each piece of a source's schema has in dew.interop: the groups its checked aliases
join, the aliases a load has already checked, the collection a layer scalar lives in, a natively
built Gemma 4 tower's tensor names, and a wrapper's placeholder ids."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.interop import hf_decoders
from dew.interop.hf_decoders import alias_groups, checked_groups, read_decoder, translate_wrapper_config
from dew.nn.backbones.decoder_block import LAYER_SCALAR_COLLECTIONS, layer_scalar_mode
from dew.nn.vision import tower_variables
from dew.nn.vision.gemma4 import (
    gemma4_projector_layouts,
    gemma4_tower_layouts,
    translate_gemma4_projector_weights,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def test_aliases_chained_or_joined_make_one_group_per_tensor():
    """Pairs chaining a -> b -> c, and pairs meeting at one name from two sides, join into one group
    each; a name no pair names stays out."""
    groups = alias_groups([("a", "b"), ("b", "c"), ("x", "z"), ("y", "z")])
    assert sorted(map(sorted, groups)) == [["a", "b", "c"], ["x", "y", "z"]]
    assert checked_groups([("a", "b"), ("c", "b")])["c"] == frozenset("abc")


class Reads(dict):
    """A tensor table that counts the reads of each name."""

    def __init__(self, tensors):
        super().__init__(tensors)
        self.reads: dict[str, int] = {}

    def __getitem__(self, name):
        self.reads[name] = self.reads.get(name, 0) + 1
        return super().__getitem__(name)


def tied_fixture():
    """Llama's tiny decoder tied: its head a copy of its embedding."""
    directory = FIXTURES / "llama-tiny"
    config = json.loads((directory / "config.json").read_text())
    tensors = load_file(directory / "model.safetensors")
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].copy()
    return hf_decoders.translate_config({**config, "tie_word_embeddings": True}), tensors


def test_a_reader_reads_a_copy_the_load_already_checked_no_more():
    """A tied head the load checked equal to its embedding before decoding (`_decoded`) is taken as
    equal: the reader drops it without reading it; a bare reader checks it itself, once."""
    record, tensors = tied_fixture()
    head, embedding = "lm_head.weight", "model.embed_tokens.weight"
    bare = Reads(tensors)
    read_decoder(bare, record, "llama")
    assert bare.reads[head] == 1
    checked = Reads(tensors)
    read_decoder(checked, record, "llama", checked=checked_groups([(head, embedding)]))
    assert checked.reads.get(head, 0) == 0
    assert checked.reads.get(embedding, 0) == bare.reads[embedding] - 1


def test_a_differing_copy_nothing_checked_is_refused():
    record, tensors = tied_fixture()
    tensors["lm_head.weight"] = tensors["lm_head.weight"] + 1
    with pytest.raises(ValueError):
        read_decoder(tensors, record, "llama")


@pytest.mark.parametrize("mode", ["trainable", "frozen"])
def test_a_layer_scalars_mode_is_the_collection_that_holds_it(mode):
    """The collection a tree keeps a layer scalar in names its mode, the one the block creates it in
    (`LAYER_SCALAR_COLLECTIONS`); a tree holding none names none."""
    tree = {"params": {"layers_0": {}}, "constants": {"layers_0": {}}}
    tree[LAYER_SCALAR_COLLECTIONS[mode]]["layers_0"]["layer_scalar"] = np.ones((1,), np.float32)
    assert layer_scalar_mode(tree, ("layers_0", "layer_scalar")) == mode
    assert layer_scalar_mode({"params": {"layers_0": {}}}, ("layers_0", "layer_scalar")) is None


def test_a_gemma4_tower_and_embedder_write_back_what_their_readers_read():
    """Every leaf of a Gemma 4 tower and its embedder writes back under the name its reader reads
    into that leaf (`gemma4_tower_layouts`), and reading those tensors again gives the tree bit for
    bit: the names, shapes and transposes are the readers' own."""
    directory = FIXTURES / "gemma4-vision-tiny"
    tensors = load_file(directory / "model.safetensors")
    tower, _ = tower_variables("gemma4", tensors, "float32")
    written = {layout.name: layout.export(tower) for layout in gemma4_tower_layouts(tower)}
    assert set(written) == set(tensors)
    again, _ = tower_variables("gemma4", written, "float32")
    jax.tree.map(np.testing.assert_array_equal, again, tower)
    projector, _ = translate_gemma4_projector_weights(load_file(directory / "projector.safetensors"))
    projected = {layout.name: layout.export({"params": projector})
                 for layout in gemma4_projector_layouts(projector)}
    jax.tree.map(np.testing.assert_array_equal, translate_gemma4_projector_weights(projected)[0], projector)


@pytest.mark.parametrize("stated, video, audio", [({}, 258884, 258881),
                                                  ({"video_token_id": 7, "audio_token_id": 9}, 7, 9)])
def test_a_gemma4_wrapper_record_holds_its_video_and_further_placeholders(stated, video, audio):
    """A Gemma 4 wrapper's record holds its video placeholder and the further ids the decoder treats
    as placeholders, Gemma4Config's defaults where the config leaves them out, whether or not it
    reads audio; the model and the processor read them there."""
    config = json.loads((FIXTURES / "gemma4-tiny-mm" / "config.json").read_text())
    config = {key: value for key, value in config.items() if key not in ("video_token_id", "audio_token_id")}
    record = translate_wrapper_config({**config, **stated})
    assert record["video_token_id"] == video
    assert record["extra_placeholder_ids"] == (video, audio)
