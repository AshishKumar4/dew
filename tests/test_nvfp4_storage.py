"""Weight-only compressed storage and the activation formats that it cannot compute."""

import json
from pathlib import Path

import numpy as np
import pytest

from dew.interop import codecs
from dew.interop.safetensors_io import read_weights

FIXTURE = Path(__file__).parent / "fixtures" / "codecs" / "storage"


@pytest.fixture(scope="module", params=["fp8", "nvfp4"])
def checkpoint(request):
    directory = FIXTURE / request.param
    with np.load(directory / "reference.npz") as reference:
        return directory, dict(reference)


def test_weight_only_storage_decodes_as_the_same_file_reference_reader(checkpoint):
    directory, reference = checkpoint
    stored = read_weights(directory)
    codec = codecs.source_quantization(json.loads((directory / "config.json").read_text()))
    assert codec is not None and len(codec.names(stored)) == 7
    for name in codec.names(stored):
        np.testing.assert_array_equal(codec.read(stored, name), reference[name], err_msg=name)
