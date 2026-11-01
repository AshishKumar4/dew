"""Multilevel SD1/SD2/SDXL geometry oracles from Diffusers 0.34.0 Flax.

Run only in the isolated reference environment:
  python tools/native_unet_reference.py OUTPUT_DIRECTORY
These small models exercise down/up sampling, mixed skip widths, linear
projections, transformer depth, cross-only attention and nine-channel inputs.

  python tools/native_unet_reference.py --float64 ARCHIVE_DIRECTORY
writes tests/fixtures/native_unet_float64.npz: each case's prediction and
input VJP from the archive's own Diffusers weights and inputs, run in float64
(`jax_enable_x64`, the model's dtype and every parameter widened), the truth
the float64 rule holds Dew and the float32 reference to.
"""
import argparse
import json
from pathlib import Path

import diffusers
import jax
import jax.numpy as jnp
import numpy as np
from diffusers import FlaxUNet2DConditionModel
from native_diffusion_reference_files import convert


def build(root):
    from reference_releases import require

    require("diffusers")
    for case, linear, depths, channels, context_width, extra, ids_count in (
        ("sd1", False, (1, 1, 1), 4, 8, False, 0),
        ("sd2", True, (1, 1, 1), 4, 12, False, 0),
        ("sdxl-inpaint", True, (1, 2, 2), 9, 16, True, 6),
        ("refiner", True, (1, 2, 2), 4, 8, True, 5),
    ):
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
            projection_class_embeddings_input_dim=8 + ids_count * 2 if extra else None)
        params = model.init_weights(jax.random.PRNGKey(31))
        model.save_pretrained(folder, params=params)
        (directory / "model_index.json").write_text(json.dumps({"unet": ["diffusers", "FlaxUNet2DConditionModel"]}))
        convert(directory)
        noise = jax.random.normal(jax.random.PRNGKey(1), (1, channels, 8, 12))
        time = jnp.asarray([13], jnp.float32)
        context = jax.random.normal(jax.random.PRNGKey(2), (1, 8, context_width)) * 0.1
        pooled = jax.random.normal(jax.random.PRNGKey(3), (1, 8)) if extra else None
        time_ids = jnp.asarray([[32, 48, 0, 0, 32, 48]]) if ids_count == 6 else jnp.asarray([[32, 48, 0, 0, 6]])
        added = {"text_embeds": pooled, "time_ids": time_ids} if extra else None
        def predict(value):
            return model.apply({"params": params}, value, time, encoder_hidden_states=context, added_cond_kwargs=added).sample
        prediction = jax.jit(predict)(noise)
        probe = jax.random.normal(jax.random.PRNGKey(4), prediction.shape)
        tangent = jax.jit(jax.grad(lambda value: jnp.sum(predict(value) * probe)))(noise)
        arrays = {"input": np.asarray(noise).transpose(0, 2, 3, 1), "time": np.asarray(time),
                  "context": np.asarray(context), "output": np.asarray(prediction).transpose(0, 2, 3, 1),
                  "probe": np.asarray(probe).transpose(0, 2, 3, 1), "vjp": np.asarray(tangent).transpose(0, 2, 3, 1)}
        if extra:
            arrays.update(pooled=np.asarray(pooled), time_ids=np.asarray(time_ids))
        np.savez_compressed(directory / "reference.npz", **arrays)
        print(case, "parameter count", sum(value.size for value in jax.tree.leaves(params)))
        jax.clear_caches()


def widen(root):
    """Each case's float64 prediction and input VJP, from the archive at `root`."""
    from reference_releases import require

    require("diffusers")
    jax.config.update("jax_enable_x64", True)
    arrays = {}
    for case in ("sd1", "sd2", "sdxl-inpaint", "refiner"):
        directory = Path(root) / case
        reference = np.load(directory / "reference.npz")
        model, params = FlaxUNet2DConditionModel.from_pretrained(directory / "unet", dtype=jnp.float64)
        params = jax.tree.map(lambda value: jnp.asarray(value, jnp.float64), params)
        wide = {name: jnp.asarray(reference[name], jnp.float64) for name in reference.files}
        added = ({"text_embeds": wide["pooled"], "time_ids": wide["time_ids"]} if "pooled" in reference.files
                 else None)

        def predict(value, wide=wide, added=added, model=model, params=params):
            return model.apply({"params": params}, value, wide["time"], encoder_hidden_states=wide["context"],
                               added_cond_kwargs=added).sample

        noise = wide["input"].transpose(0, 3, 1, 2)
        prediction = jax.jit(predict)(noise)
        probe = wide["probe"].transpose(0, 3, 1, 2)
        tangent = jax.jit(jax.grad(lambda value: jnp.sum(predict(value) * probe)))(noise)
        if prediction.dtype != jnp.float64 or tangent.dtype != jnp.float64:
            raise TypeError(f"{case} ran in {prediction.dtype}, not float64")
        arrays[f"{case}/output"] = np.asarray(prediction).transpose(0, 2, 3, 1)
        arrays[f"{case}/vjp"] = np.asarray(tangent).transpose(0, 2, 3, 1)
    destination = Path(__file__).resolve().parents[1] / "tests/fixtures/native_unet_float64.npz"
    np.savez_compressed(destination, precision=np.asarray("ieee"), **arrays)
    print("wrote", destination)


def bundle(root, destination):
    """Archive source-compatible checkpoint files and numerical oracle arrays."""
    from fixture_archive import write
    cases = ("sd1", "sd2", "sdxl-inpaint", "refiner")
    for case in cases:
        convert(Path(root) / case)
    write(destination, ((Path(root) / case, case) for case in cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--float64", action="store_true", help="the archive's float64 truth, not new oracles")
    arguments = parser.parse_args()
    (widen if arguments.float64 else build)(arguments.directory)
