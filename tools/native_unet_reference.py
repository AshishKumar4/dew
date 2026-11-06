"""Multilevel SD1/SD2/SDXL UNet geometries, Diffusers' own Flax UNet at each.

Run only in the isolated reference environment:
  python tools/native_unet_reference.py OUTPUT_DIRECTORY
These small models exercise down/up sampling, mixed skip widths, linear
projections, transformer depth, cross-only attention and nine-channel inputs.
Each case's weights and config are what tools/check_native_diffusion.py's
`model_site` runs Dew's UNet and Diffusers' own on, live over drawn inputs.
"""
import argparse
import json
from pathlib import Path

import jax
from diffusers import FlaxUNet2DConditionModel
from native_diffusion_reference_files import convert


CASES = {"sd1": (False, (1, 1, 1), 4, 8, False), "sd2": (True, (1, 1, 1), 4, 12, False),
         "sdxl-inpaint": (True, (1, 2, 2), 9, 16, True), "refiner": (True, (1, 2, 2), 4, 8, True)}
"""Each geometry: its linear projections, transformer depths, input channels, prompt-state width, and
whether it takes SDXL's added conditioning (`TIME_IDS`)."""
TIME_IDS = {"sdxl-inpaint": (32, 48, 0, 0, 32, 48), "refiner": (32, 48, 0, 0, 6)}
"""SDXL's micro-conditioning an added-conditioning geometry reads: the original size, the crop and the
target size, the refiner's aesthetic score in place of its target."""
POOLED = 8
"""The width of the pooled embedding the added conditioning reads beside the time ids'."""


def build(root):
    from reference_releases import require

    require("diffusers")
    for case, (linear, depths, channels, context_width, extra) in CASES.items():
        ids_count = len(TIME_IDS.get(case, ()))
        directory = Path(root) / case
        folder = directory / "unet"
        folder.mkdir(parents=True, exist_ok=True)
        cross = (False, True, True) if extra else (True, True, False)
        down = tuple("CrossAttnDownBlock2D" if flag else "DownBlock2D" for flag in cross)
        up = tuple("CrossAttnUpBlock2D" if flag else "UpBlock2D" for flag in cross[::-1])
        model = FlaxUNet2DConditionModel(
            sample_size=8, in_channels=channels, out_channels=4,
            down_block_types=down, up_block_types=up, block_out_channels=(32, 32, 64),
            layers_per_block=2, attention_head_dim=(4, 4, 8), cross_attention_dim=context_width,
            only_cross_attention=(False, case == "sd2", False), use_linear_projection=linear,
            transformer_layers_per_block=depths, addition_embed_type="text_time" if extra else None,
            addition_time_embed_dim=2 if extra else None,
            projection_class_embeddings_input_dim=POOLED + ids_count * 2 if extra else None)
        params = model.init_weights(jax.random.PRNGKey(31))
        model.save_pretrained(folder, params=params)
        (directory / "model_index.json").write_text(json.dumps({"unet": ["diffusers", "FlaxUNet2DConditionModel"]}))
        convert(directory)
        print(case, "parameter count", sum(value.size for value in jax.tree.leaves(params)))
        jax.clear_caches()


def bundle(root, destination):
    """Archive source-compatible checkpoint files."""
    from fixture_archive import write
    for case in CASES:
        convert(Path(root) / case)
    write(destination, ((Path(root) / case, case) for case in CASES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    build(parser.parse_args().directory)
