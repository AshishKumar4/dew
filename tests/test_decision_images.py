"""A decision request's images are read as the backbone's processor computes them
(`dew.decision.images.Images`): a processor that computes in PyTorch, as Llama 4's torchvision one does
with its bfloat16 pixels, is asked for its own tensors and read on the host as its model inputs are
(`Processor.host_media`)."""

from pathlib import Path

import numpy as np
from PIL import Image

from dew.decision.images import Images
from dew.interop import Pretrained

LLAMA4 = Path(__file__).parent / "fixtures" / "hf" / "llama4-native-tiny"


def test_a_torch_processors_images_read_as_its_model_inputs_do():
    """Llama 4's processor computes its pixels in PyTorch, in bfloat16, which NumPy cannot hold: a
    request's images read through it as its model inputs read them, each pixel field a host array
    equal to the processor's own inputs' field."""
    loaded = Pretrained.load(LLAMA4, dtype="float32", attention_impl="xla", max_seq_len=64)
    processor = loaded.processor
    assert processor is not None
    pictures = [Image.fromarray(array) for array in np.load(LLAMA4 / "raw_images.npy")[:1]]
    text = "<|image|>"
    media = Images.of(processor, pictures, text)
    direct = processor([text + "\n"], images=list(pictures))
    assert media.fields and all(isinstance(value, np.ndarray) for value in media.fields.values())
    for name, value in media.fields.items():
        # The model inputs hold each row's images under a row axis of their own.
        np.testing.assert_array_equal(value, np.asarray(direct.conditioning[name]).reshape(value.shape))
