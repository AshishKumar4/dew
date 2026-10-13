"""Multimodal wrappers: config translation and weight maps with parity.

translate_wrapper_config turns a gemma3, llama4, gemma4 or qwen3_5 wrapper
into its decoder, tower and projector records; translate_wrapper_weights
routes the released model.* nesting into the three trees, whose prefixes the
translation tests write down per family. Tower and projector parity rides the
tiny wrapper fixtures, whose vision halves come from the shared systems in
tools/hf_reference.py at a fresh seed with these observed differences, fp32
on CPU:

- gemma3-tiny-mm tower: max |difference| 9.6e-07, projector 4.8e-07,
  tolerance 1e-4. Prefixes model.vision_tower.* and
  model.multi_modal_projector.*.
- llama4-tiny-mm tower: max |difference| 7.0e-06, projector 1.5e-06,
  tolerance 1e-4. Bare vision_model.* and multi_modal_projector.* with no
  model. prefix.
- gemma4-tiny-mm tower: max |difference| 2.5e-05, projector 9.6e-07,
  tolerance 1e-4. Prefixes model.vision_tower.* and model.embed_vision.*.
- qwen35-tiny-mm tower: max |difference| 7.4e-06, projector 3.9e-06,
  tolerance 1e-4. Prefixes model.visual.* with the merger under
  model.visual.merger.*.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from dew.interop.hf_decoders import translate_wrapper_config, translate_wrapper_weights
from dew.registry import projectors, towers

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def load_wrapper(name):
    directory = FIXTURES / name
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    variables = translate_wrapper_weights(
        load_file(str(directory / "model.safetensors")), record)
    return directory, record, variables


def wrapper_pixels(directory, record):
    """Fixture pixels as the tower reads them: the Gemma 4 fixture stores
    processor patches, which fold back into the image row-major."""
    pixels = np.load(directory / "pixels.npy")
    if record["tower"]["class"] != "gemma4":
        return pixels
    batch, count, _ = pixels.shape
    grid = int(count ** 0.5)
    patch = int(record["tower"]["fields"]["patch_size"])
    side = grid * patch
    return pixels.reshape(batch, grid, grid, patch, patch, 3).transpose(
        0, 5, 1, 3, 2, 4).reshape(batch, 3, side, side)


def test_wrapper_tower_and_projector_match_the_reference():
    """The vision halves of all four wrapper fixtures against their committed
    reference outputs."""
    for name, tolerance in (("gemma3-tiny-mm", 1e-4), ("llama4-tiny-mm", 1e-4),
                            ("gemma4-tiny-mm", 1e-4), ("qwen35-tiny-mm", 1e-4)):
        directory, record, variables = load_wrapper(name)
        pixels = wrapper_pixels(directory, record)
        tower_ref = np.load(directory / "tower_ref.npy")
        projector_ref = np.load(directory / "projector_ref.npy")
        tower = towers.from_record(record["tower"]).build()
        assert np.max(np.abs(np.asarray(
            tower.apply(variables["tower"], pixels)) - tower_ref)) < tolerance
        projector = projectors.from_record(record["projector"]).build()
        assert np.max(np.abs(np.asarray(
            projector.apply(variables["projector"], tower_ref))
            - projector_ref)) < tolerance
