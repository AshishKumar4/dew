"""The model generator reads registrations and checks them without generated content."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def models_generator():
    path = Path(__file__).resolve().parents[1] / "site/scripts/gen_models.py"
    spec = importlib.util.spec_from_file_location("site_models_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_generator_reads_constructed_runtime_registry(models_generator, monkeypatch):
    from dew.interop import hf_decoders

    wrappers = dict(hf_decoders._WRAPPERS.items())
    wrappers.update(dict.fromkeys(("site_probe",), wrappers["gemma3"]))
    monkeypatch.setattr(hf_decoders, "_WRAPPERS", wrappers)

    actual = models_generator.wrappers()
    assert actual[:len(wrappers)] == list(wrappers)
    assert "site_probe" in actual


def test_models_check_needs_no_generated_content(models_generator, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(models_generator, "SITE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["gen_models.py", "--check"])

    models_generator.main()

    assert "decoder types" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_models_check_rejects_unlabelled_wrapper(models_generator, monkeypatch, tmp_path, capsys):
    from dew.interop import hf_decoders

    wrappers = {**hf_decoders._WRAPPERS, "site_probe": hf_decoders._WRAPPERS["gemma3"]}
    monkeypatch.setattr(hf_decoders, "_WRAPPERS", wrappers)
    monkeypatch.setattr(models_generator, "SITE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["gen_models.py", "--check"])

    with pytest.raises(SystemExit) as error:
        models_generator.main()

    assert error.value.code == 1
    assert "multimodal model_type 'site_probe' is registered" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_a_replay_is_cropped_to_its_aspect_around_the_picture():
    """A frame taller than the replay's 3:2 keeps its middle rows, so every
    replay on the page shares one shape."""
    import numpy as np
    from PIL import Image

    path = Path(__file__).resolve().parents[1] / "site/scripts/replay-atlas.py"
    spec = importlib.util.spec_from_file_location("site_replay_under_test", path)
    atlas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atlas)
    checkers = (np.indices((12, 12)).sum(0) % 2 * 255).astype(np.uint8)
    picture = Image.fromarray(np.repeat(checkers[..., None], 3, -1))
    assert atlas.content_box(picture, 1.5) == (0, 2, 12, 10)
