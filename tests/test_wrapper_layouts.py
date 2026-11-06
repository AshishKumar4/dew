"""A multimodal wrapper built in Dew written in its family's release layout (`dew.interop.wrappers`): the
config and tensors its declaration derives open in transformers' class with nothing missing or left over,
and read back as the model they were written from."""

from pathlib import Path

import numpy as np
import pytest
from model_support import flat_tree

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import WRAPPER_LAYOUTS
from dew.interop.layouts import source_layout

FIXTURES = Path(__file__).parent / "fixtures" / "hf"
WRITTEN = ("gemma3-tiny-mm", "gemma4-tiny-mm", "llama4-tiny-mm", "qwen35-tiny-mm", "qwen35-moe-native-tiny")
"""A release of each wrapper family whose layout writes it (`WRAPPER_LAYOUTS`), its media a vision tower
and its projector."""


@pytest.mark.parametrize("name", WRITTEN)
def test_a_wrapper_writes_its_release_from_its_own_fields_and_reads_back_as_itself(name, tmp_path):
    """A wrapper read from its release, written from its model and variables alone
    (`PretrainedDecoder.from_model`), opens in transformers' class of its family with no tensor missing,
    left over or misshapen, and Dew reads it back as the same model, every parameter its own."""
    import transformers

    loaded = Pretrained.load(str(FIXTURES / name), dtype="float32", attention_impl="reference")
    layout = source_layout(loaded.model)
    assert layout, layout.reason
    PretrainedDecoder.from_model(loaded.model, loaded.variables, tokenizer="byte").save(tmp_path)
    architecture = WRAPPER_LAYOUTS[loaded.model.family].architecture
    _, report = getattr(transformers, architecture).from_pretrained(tmp_path, output_loading_info=True)
    assert not any(report[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys")), report
    again = Pretrained.load(str(tmp_path), dtype="float32", attention_impl="reference")
    assert again.model == loaded.model
    written, read = flat_tree(loaded.variables), flat_tree(again.variables)
    assert written.keys() == read.keys()
    for path, leaf in written.items():
        assert np.array_equal(np.asarray(read[path]), np.asarray(leaf)), path


def test_a_wrapper_whose_release_is_its_own_code_is_unwritable_with_its_reason():
    """LFM3-VL's release is its own modeling code (LiquidAI/LFM3-VL-2B), which transformers has no class
    for and no layout of Dew's writes (`WRAPPER_LAYOUTS`): its model is `Unwritable`, falsy and naming why,
    before any file is written."""
    loaded = Pretrained.load(str(FIXTURES / "lfm3-vl-tiny"), dtype="float32", attention_impl="reference")
    layout = source_layout(loaded.model)
    assert not layout and "lfm3_vl" in layout.reason
