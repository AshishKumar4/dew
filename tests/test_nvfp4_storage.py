"""Weight-only compressed storage and the activation formats that it cannot compute."""

from pathlib import Path

import numpy as np
import pytest
from interop_support import stored_codec

FIXTURE = Path(__file__).parent / "fixtures" / "codecs" / "storage"


@pytest.fixture(scope="module", params=["fp8", "nvfp4"])
def checkpoint(request):
    directory = FIXTURE / request.param
    with np.load(directory / "reference.npz") as reference:
        return directory, dict(reference)


def test_weight_only_storage_decodes_as_the_same_file_reference_reader(checkpoint):
    directory, reference = checkpoint
    stored, codec = stored_codec(directory)
    for name in codec.names(stored):
        np.testing.assert_array_equal(codec.decode(stored, name), reference[name], err_msg=name)
