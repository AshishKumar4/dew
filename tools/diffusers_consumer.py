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

`flax_unet` is the declared Flax UNet as a JAX function, `unet_predictor` runs the
declared UNet (Flax or PyTorch) on the host, and `denoiser` a
declared PyTorch denoiser, in float32 or in float64 (`torch_precision.float64`
for PyTorch, x64 and `flax_precision.float64` for Flax), for a forward the
caller holds Dew's to by tests/reference_error.py's rule.
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

from tools import flax_precision
from tools.diffusers_reference_helpers import restore_transformers_names
from tools.torch_precision import float64

# Diffusers' own classes read what Dew wrote, through its pipeline modules.
restore_transformers_names()

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


def flax_unet(export: Path, *, wide: bool = False):
    """The declared Flax UNet (`load_flax`) as a JAX function of a channels-last `sample` (noise and any
    inpaint channels), the timesteps, the prompt states and, for SDXL, the pooled embedding and the time
    ids, its prediction channels last, in float32 or (`wide`) float64: a float64 run reads the float32 its
    modules name as float64 (`flax_precision.float64`) while it traces. A caller jits or differentiates it."""
    with jax.enable_x64(new_val=wide):
        model, params = load_flax(export, "unet", wide=wide)
    widening = flax_precision.float64 if wide else contextlib.nullcontext

    def predict(sample, times, context, pooled=None, time_ids=None):
        added = None if pooled is None else {"text_embeds": pooled, "time_ids": time_ids}
        with widening():
            output = model.apply({"params": params}, jnp.moveaxis(sample, -1, 1), times, context,
                                 added_cond_kwargs=added).sample
        return jnp.moveaxis(output, 1, -1)

    return predict


def unet_predictor(export: Path, *, wide: bool = False):
    """The declared UNet (Flax, `flax_unet`, or PyTorch), loaded once, as a host function of a channels-last
    `sample` (noise and any inpaint channels), the timestep, the prompt states and, for SDXL, the pooled
    embedding and the time ids, in float32 or (`wide`) float64, its prediction channels last."""
    dtype = np.float64 if wide else np.float32

    def given(sample, timestep, context, pooled, time_ids):
        """The call's arrays at the run's dtype, the timestep one per row."""
        sample = np.asarray(sample, dtype)
        return (sample, np.full((len(sample),), timestep, dtype), np.asarray(context, dtype),
                None if pooled is None else np.asarray(pooled, dtype),
                None if pooled is None else np.asarray(time_ids, dtype))

    if flax_declared(export, "unet"):
        arm = flax_unet(export, wide=wide)

        def predict(sample, timestep, context, pooled=None, time_ids=None):
            with jax.enable_x64(new_val=wide):
                return np.asarray(arm(*(None if value is None else jnp.asarray(value)
                                        for value in given(sample, timestep, context, pooled, time_ids))))

        return predict
    def call(unet, tensor, sample, timestep, context, pooled=None, time_ids=None):
        sample, times, context, pooled, time_ids = given(sample, timestep, context, pooled, time_ids)
        added = None if pooled is None else {"text_embeds": tensor(pooled), "time_ids": tensor(time_ids)}
        output = unet(tensor(np.ascontiguousarray(np.moveaxis(sample, -1, 1))), tensor(times),
                      encoder_hidden_states=tensor(context), added_cond_kwargs=added).sample
        return np.moveaxis(output.numpy(), 1, -1)

    return denoiser(export, call, wide=wide)


def denoiser(export: Path, call, *, wide: bool = False):
    """The declared PyTorch denoiser, loaded once in float32 or (`wide`)
    float64, as a function of a draw's inputs: `call(model, tensor, *inputs)`
    on it, `tensor` turning an array into a tensor of the model's dtype, and
    `call` returning the prediction as an array."""
    [name] = [name for name in weighted_components(export) if name in DENOISERS]
    dtype = torch.float64 if wide else torch.float32
    with precision(wide):
        model = load(export, name, dtype)

    def predict(*inputs):
        def tensor(value):
            return torch.from_numpy(np.asarray(value)).to(dtype)

        with precision(wide), torch.no_grad():
            return np.asarray(call(model, tensor, *inputs))
    return predict
