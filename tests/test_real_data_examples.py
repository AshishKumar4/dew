"""Real-data example caption geometry and byte MDLM training."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_caption_rows_keep_image_and_caption_targets_in_separate_spans():
    """The caption canvas is whole: the caption, then EOS to the canvas's end,
    every position attended and scored, as the sampler denoises it."""
    script = example("sft_diffusion_gemma_images")
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


def test_caption_example_trains_reloads_and_captions_the_held_out_split(tmp_path):
    """--smoke runs the whole example on parquet splits it writes: training on
    train + validation, then the saved run captions every test image."""
    import json

    script = example("sft_diffusion_gemma_images")
    out = tmp_path / "caption"
    run_dir = script.main(script.Config(smoke=True, out=out))
    report = json.loads((out / "result.json").read_text())
    assert report["steps"] == 4 and report["test_images"] == 8
    assert 0 <= report["caption_accuracy"] <= 1
    assert len((out / "captions.jsonl").read_text().splitlines()) == 8
    assert any(run_dir.iterdir())


def test_masked_lm_example_trains_scores_and_unmasks_from_the_saved_run(tmp_path):
    """--smoke runs the whole example over a local JSON text file: MDLM
    training with a held-out perplexity bound, then the saved run scores the
    whole held-out split and continues each prompt."""
    import json

    script = example("train_masked_lm")
    out = tmp_path / "run"
    script.main(script.Config(smoke=True, out=out))
    report = json.loads((out / "result.json").read_text())
    assert report["tokenizer"] == "byte" and report["steps"] == 4
    assert np.isfinite(report["held_out"]["val/perplexity"])
    assert [sample[:len(prompt)] for sample, prompt in zip(report["samples"], ("The cat", "A small boat"),
                                                           strict=True)] == ["The cat", "A small boat"]
