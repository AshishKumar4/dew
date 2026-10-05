"""Diffusers reading a pipeline directory Dew wrote.

`assert_components_load` loads every component of an exported pipeline that
holds weights with the class its model_index.json declares (Diffusers' for
the denoiser and the autoencoder, transformers' for the text towers), each
with `output_loading_info`, and refuses any missing, unexpected or
mismatched tensor or loading error; a pipeline declared with a PyTorch
class also loads whole through `DiffusionPipeline.from_pretrained`.

A component declared with a Flax class (the Flax Stable Diffusion
pipelines) is read twice: from the msgpack Dew writes, with Diffusers' own
Flax class, whose parameter tree has to be the one the class initializes;
and from the safetensors file beside it, with its PyTorch twin.
transformers 5 ships no Flax CLIP, so a Flax text tower is read through its
PyTorch twin alone, and the Flax pipelines themselves cannot be built here.

`unet_prediction` runs the declared UNet (Flax or PyTorch) and
`denoiser_prediction` a declared PyTorch denoiser, in float32 or in float64
(`diffusers_wan_reference.float64` for PyTorch, x64 for Flax), for a
forward the caller holds Dew's trained one to by tests/reference_error.py's
rule.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

# Also restores the two transformers names Diffusers 0.34.0's pipeline
# modules import (tools/diffusers_wan_reference.py).
from tools.diffusers_wan_reference import float64

DENOISERS = ("unet", "transformer")


def weighted_components(export: Path) -> dict[str, tuple[str, str]]:
    """Each declared component with a weights file, by its (library, class)."""
    index = json.loads((Path(export) / "model_index.json").read_text())
    return {name: (entry[0], entry[1]) for name, entry in index.items()
            if isinstance(entry, list) and len(entry) == 2 and entry[0] is not None
            and any((Path(export) / name).glob("*.safetensors"))}


def library_module(library: str):
    """A model_index library: a package, or, as Diffusers resolves any other
    name (the safety checker's `stable_diffusion`), its pipeline module."""
    return importlib.import_module(library if library in ("diffusers", "transformers")
                                   else f"diffusers.pipelines.{library}")


# Components no class in this environment reads: Diffusers 0.34's
# StableDiffusionSafetyChecker is a transformers 4 model, which under
# transformers 5 never calls `post_init` and nests a CLIPVisionModel without
# the `vision_model.` level every published safety checker file has. Its
# reader is Diffusers 0.34 with transformers 4; `unread` names these for the
# caller, who holds their files to the source's.
UNREAD = ("StableDiffusionSafetyChecker",)


def unread(export: Path) -> list[str]:
    return [name for name, (_, declared) in weighted_components(export).items() if declared in UNREAD]


def load(export: Path, name: str, dtype: torch.dtype = torch.float32):
    """One component's PyTorch class (a Flax class's twin) at `dtype`,
    refused unless its loading report is clean."""
    library, declared = weighted_components(export)[name]
    cls = getattr(library_module(library), declared.removeprefix("Flax"))
    model, report = cls.from_pretrained(str(export), subfolder=name, torch_dtype=dtype,
                                        local_files_only=True, output_loading_info=True)
    problems = {key: value for key, value in report.items() if value}
    if problems:
        raise AssertionError(f"{export}/{name} ({cls.__name__}) loads with {problems}")
    return model.eval()


def load_flax(export: Path, name: str, *, wide: bool = False):
    """A Flax-declared Diffusers component from its msgpack: the module and
    its parameters, refused unless the tree is the one the class itself
    initializes, leaf for leaf and shape for shape. `wide` builds it in
    float64 with the parameters widened."""
    from flax.traverse_util import flatten_dict

    library, declared = weighted_components(export)[name]
    cls = getattr(library_module(library), declared)
    model, params = cls.from_pretrained(str(export), subfolder=name,
                                        dtype=jnp.float64 if wide else jnp.float32)
    expected = {key: value.shape for key, value in flatten_dict(
        jax.eval_shape(model.init_weights, jax.random.key(0))).items()}
    loaded = {key: np.shape(value) for key, value in flatten_dict(params).items()}
    if loaded != expected:
        reshaped = [key for key in expected.keys() & loaded.keys() if expected[key] != loaded[key]]
        raise AssertionError(f"{export}/{name} ({declared}) msgpack differs from the class's tree: "
                             f"missing {sorted(set(expected) - set(loaded))[:4]}, "
                             f"unexpected {sorted(set(loaded) - set(expected))[:4]}, reshaped {reshaped[:4]}")
    if wide:
        params = jax.tree.map(lambda value: np.asarray(value, np.float64), params)
    return model, params


def flax_declared(export: Path, name: str) -> bool:
    return weighted_components(export)[name][1].startswith("Flax")


def assert_components_load(export: Path) -> None:
    """Every weighted component loads cleanly, its Flax class too where
    Diffusers declares one, but those `unread` names; a PyTorch-declared
    pipeline loads whole, without them."""
    from diffusers import DiffusionPipeline

    skipped = unread(export)
    for name, (library, declared) in weighted_components(export).items():
        if name in skipped:
            continue
        load(export, name)
        if declared.startswith("Flax") and library == "diffusers":
            load_flax(export, name)
    declared = json.loads((Path(export) / "model_index.json").read_text())["_class_name"]
    if not declared.startswith("Flax"):
        DiffusionPipeline.from_pretrained(str(export), torch_dtype=torch.float32, local_files_only=True,
                                          **dict.fromkeys(skipped),
                                          **({"requires_safety_checker": False} if skipped else {}))


def precision(wide: bool):
    """The context a PyTorch float64 forward runs in: Diffusers pins some
    tables and casts to float32, which `float64` widens."""
    return float64() if wide else contextlib.nullcontext()


def unet_prediction(export: Path, sample: np.ndarray, timestep: float, context: np.ndarray,
                    pooled: np.ndarray | None = None, time_ids: np.ndarray | None = None, *,
                    wide: bool = False) -> np.ndarray:
    """The declared UNet's prediction for a channels-last `sample` (noise and
    any inpaint channels), in float32 or (`wide`) float64, channels last."""
    dtype = np.float64 if wide else np.float32
    sample = np.ascontiguousarray(np.moveaxis(np.asarray(sample, dtype), -1, 1))
    context = np.asarray(context, dtype)
    added = None if pooled is None else {"text_embeds": np.asarray(pooled, dtype),
                                         "time_ids": np.asarray(time_ids, dtype)}
    if flax_declared(export, "unet"):
        with jax.enable_x64(new_val=wide):
            model, params = load_flax(export, "unet", wide=wide)
            output = model.apply({"params": params}, jnp.asarray(sample), jnp.asarray([timestep]),
                                 jnp.asarray(context), added_cond_kwargs=added).sample
            return np.moveaxis(np.asarray(output), 1, -1)
    with precision(wide):
        unet = load(export, "unet", torch.float64 if wide else torch.float32)
        tensors = None if added is None else {key: torch.from_numpy(value) for key, value in added.items()}
        with torch.no_grad():
            output = unet(torch.from_numpy(sample), torch.tensor([timestep], dtype=unet.dtype),
                          encoder_hidden_states=torch.from_numpy(context), added_cond_kwargs=tensors).sample
    return np.moveaxis(output.numpy(), 1, -1)


def denoiser_prediction(export: Path, call, *, wide: bool = False) -> np.ndarray:
    """`call(model, tensor)` on the declared PyTorch denoiser, in float32 or
    (`wide`) float64: `tensor` turns an array into a tensor of the model's
    dtype, and `call` returns the prediction as an array."""
    [name] = [name for name in weighted_components(export) if name in DENOISERS]
    dtype = torch.float64 if wide else torch.float32
    with precision(wide):
        model = load(export, name, dtype)
        with torch.no_grad():
            return np.asarray(call(model, lambda value: torch.from_numpy(np.asarray(value)).to(dtype)))
