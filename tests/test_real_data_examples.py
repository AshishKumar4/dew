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
    script = example("sft_diffusion_gemma_images")
    config = script.Config(flowers="unused", image_size=16, prompt_tokens=24)
    pixels = np.arange(2 * 16 * 16 * 3, dtype=np.uint8).reshape(2, 16, 16, 3)
    batch = script.caption_batch({"image": pixels, "label": [0, 1]}, config, ["pink rose", "yellow tulip"])
    inputs = batch["text"]
    assert inputs.tokens.shape == (2, 88)
    assert np.all(inputs.token_fields["image_indices"][:, config.prompt_tokens:] == -1)
    for row, label in enumerate(["pink rose", "yellow tulip"]):
        response = [*("a photo of a " + label).encode(), 256]
        np.testing.assert_array_equal(inputs.tokens[row, 24:24+len(response)], response)
        assert inputs.token_fields["attention_mask"][row, 24:24+len(response)].all()
        assert not inputs.token_fields["attention_mask"][row, 24+len(response):].any()
    np.testing.assert_allclose(inputs.conditioning["pixel_values"][:, 0],
                               pixels.transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1, atol=0, rtol=0)


def test_caption_example_trains_and_executes_its_inference_call(tmp_path, monkeypatch):
    from dataclasses import replace
    from dew.data import Dataset, Loading

    script = example("sft_diffusion_gemma_images")
    config = script.Config(flowers="unused", image_size=16, prompt_tokens=24,
                           canvas_length=8, batch_size=8, steps=2, features=16, vision_features=16,
                           out=tmp_path / "caption")
    pixels = np.arange(8 * 16 * 16 * 3, dtype=np.uint8).reshape(8, 16, 16, 3)
    held = Dataset.from_records({"image": pixels, "label": np.arange(8) % 2}, batch=8,
                                loading=Loading(workers=0, threads=1, read_buffer=1))
    data = replace(held, train=script.mapped(held.train,
                   lambda batch: script.caption_batch(batch, config, ["pink rose", "yellow tulip"])))
    monkeypatch.setattr(script, "flowers_data", lambda selected: data)
    state = script.main(config)
    assert int(state.step) == 2
    assert (config.out / "result.json").exists()
    assert (config.out / "samples.txt").read_text().strip()


def test_masked_lm_example_trains_real_byte_windows_and_writes_a_sample(tmp_path):
    script = example("train_masked_lm")
    import json
    # A real WikiText line, repeated only to make the smoke corpus large
    # enough for fixed windows. The production command reads the full file.
    text = "Valkyria Chronicles III is a tactical role-playing video game developed by Sega.\n"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    np.frombuffer((text * 64).encode(), np.uint8).tofile(corpus / "train.bin")
    np.frombuffer((text * 8).encode(), np.uint8).tofile(corpus / "val.bin")
    (corpus / "meta.json").write_text(json.dumps({"tokenizer": "byte", "vocab_size": 256,
                                                "dtype": "uint8", "train_tokens": len(text) * 64}))
    state = script.main(script.Config(tokens=corpus, sequence_length=16, batch_size=8,
                        features=32, layers=1, heads=4, steps=2, sample_tokens=8,
                        sample_steps=4, out=tmp_path / "run"))
    assert int(state.updates) == 2
    report = json.loads((tmp_path / "run/result.json").read_text())
    assert np.isfinite(report["probe_nelbo_after"])
    assert report["probe_nelbo_before"] != report["probe_nelbo_after"]
    assert report["sample"].startswith("Once upon a time")
    assert (tmp_path / "run/checkpoints/2").is_dir()
