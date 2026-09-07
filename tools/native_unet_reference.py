"""Multilevel SD1/SD2/SDXL geometry oracles from Diffusers 0.34.0 Flax.

Run only in the isolated reference environment:
  python tools/native_unet_reference.py OUTPUT_DIRECTORY
These small models exercise down/up sampling, mixed skip widths, linear
projections, transformer depth, cross-only attention and nine-channel inputs.
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
    if diffusers.__version__ != "0.34.0":
        raise RuntimeError("Use Diffusers 0.34.0 for the reference")
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


def bundle(root, destination):
    """Archive only the native checkpoint files and oracle arrays."""
    import tarfile
    with tarfile.open(destination, "w:xz") as archive:
        for case in ("sd1", "sd2", "sdxl-inpaint", "refiner"):
            archive.add(Path(root) / case, arcname=case,
                        filter=lambda info: None if info.name.endswith(".msgpack") else info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    build(parser.parse_args().directory)
