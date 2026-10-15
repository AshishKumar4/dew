"""Real-data example caption geometry and byte MDLM training."""

import numpy as np
from tool_support import load


def test_caption_rows_keep_image_and_caption_targets_in_separate_spans():
    """The caption canvas is whole: the caption, then EOS to the canvas's end,
    every position attended and scored, as the sampler denoises it."""
    script = load("sft_diffusion_gemma_images", "examples")
    config = script.Config(image_size=16, prompt_tokens=24)
    pixels = np.arange(2 * 16 * 16 * 3, dtype=np.uint8).reshape(2, 16, 16, 3)
    batch = script.caption_batch({"image": pixels, "label": [0, 1]}, config, ["pink rose", "yellow tulip"])
    inputs = batch["text"]
    assert inputs.tokens.shape == (2, 88)
    assert np.all(inputs.token_fields["image_indices"][:, config.prompt_tokens:] == -1)
    for row, label in enumerate(["pink rose", "yellow tulip"]):
        response = list(("a photo of a " + label).encode())
        np.testing.assert_array_equal(inputs.tokens[row, 24:24+len(response)], response)
        assert np.all(inputs.tokens[row, 24+len(response):] == 256)
        assert inputs.token_fields["attention_mask"][row, 24:].all()
    np.testing.assert_allclose(inputs.conditioning["pixel_values"][:, 0],
                               pixels.transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1, atol=0, rtol=0)
