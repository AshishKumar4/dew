"""The saved tiny pipelines' own Diffusers pipelines run on drawn prompts, noise and pixels, in float32 and
float64: the reference arm of tools/venv_reference.py's `native-<task>` sites (its protocol
tools/reference_worker.py), run in the diffusers-pipelines cohort's environment (tools/reference_releases.py:
the Flax pipelines' CLIP towers are transformers 4's), on the CPU.

    python tools/native_diffusion_worker.py native-TASK-GROUP    `IN OUT` lines on stdin, each answered `done`
    python tools/native_diffusion_worker.py native-grid-NAME

TASK is a directory of tests/fixtures/tiny_diffusers.tar.xz, as tools/diffusers_pipeline_reference.py (sd,
xl, img2img, inpaint: Flax pipelines) and tools/diffusers_extended_reference.py (xl-img2img, xl-inpaint,
refiner, safety: PyTorch pipelines) saved it and called it, at its steps, guidance, mask and keys, computing
GROUP's leaves alone (`groups`: each group is its own site, recorded and sized on its own). IN holds
each case's `<case>.<input>`: the prompts as letter codes (`prompt`, `negative`, and an SDXL tower's second
`prompt_2`, `negative_2`), the `noise` channels last, the `pixels` an image task starts from, and the
safety checker's `checker_raw` images and `checker_input` tower input. OUT gets `<case>.reference.<leaf>`
and `<case>.truth.<leaf>`, the call in float32 and in float64 (the generators' float64 runs: a Flax
pipeline's modules read float32 as float64, a PyTorch pipeline built and run in `torch_precision.float64`),
every noise the call draws the drawn one:

- `hidden`, `pooled`: the prompt's encoder states and SDXL's pooled embedding;
- `encoded`, `decoded`: the autoencoder's posterior sample of the pixels and its decode of the noise;
- `start`: an SDXL image-to-image call's noised start;
- `final_latents`, `images`: the walk's last latents and its images, before any safety checker;
- `checker_pixels`, `checker_embeddings`: the checker's transform of `checker_raw` and its tower's
  projection of `checker_input`.

A `native-grid-NAME` site walks the saved SD pipeline's Flax UNet under Diffusers' scheduler NAME
(tools/diffusers_scheduler_pipeline_reference.py's `walk`) from the case's `noise`, its `latents`.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from reference_worker import serve

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/tiny_diffusers.tar.xz"
FLAX = ("sd", "xl", "img2img", "inpaint")
TASKS = (*FLAX, "xl-img2img", "xl-inpaint", "refiner", "safety")
GROUPS = {"prompt": ("hidden", "pooled"), "autoencoder": ("encoded", "decoded"),
          "walk": ("start", "final_latents", "images"), "checker": ("checker_pixels", "checker_embeddings")}
"""A task's leaves judged together, each group by what computes it: the prompt's states, the autoencoder's
encode and decode, the walk itself, and the safety checker's transform and tower."""


def leaves(task: str) -> tuple[str, ...]:
    """The leaves a task's call is held by: the prompt's states (SDXL's pooled embedding too), the
    autoencoder's encode and decode where the saved reference recorded them (the Flax calls), an SDXL image
    call's noised start and the PyTorch calls' last latents, the images, and the safety checker's
    transform and tower."""
    if task == "safety":
        return "images", "checker_pixels", "checker_embeddings"
    pooled = ("pooled",) if task.startswith("xl") or task == "refiner" else ()
    if task in FLAX:
        return "hidden", *pooled, "encoded", "decoded", "images"
    return "hidden", *pooled, *(("start",) if task == "xl-img2img" else ()), "final_latents", "images"


def groups(task: str) -> dict[str, tuple[str, ...]]:
    """`task`'s groups of its leaves, those it has."""
    held = leaves(task)
    return {group: kept for group, named in GROUPS.items() if (kept := tuple(leaf for leaf in named
                                                                             if leaf in held))}
KEY = 17
"""The posterior and pipeline key every saved call took (`jax.random.PRNGKey(17)`)."""


def text(codes: np.ndarray) -> str:
    """A drawn prompt, its letter codes as a string."""
    return bytes(np.asarray(codes, np.uint8)).decode()


@functools.cache
def saved() -> Path:
    """The saved pipelines, extracted once."""
    root = Path(tempfile.mkdtemp(prefix="dew-native-worker-"))
    with tarfile.open(FIXTURE) as archive:
        archive.extractall(root, filter="data")
    return root


@contextlib.contextmanager
def flax_float64():
    """The Flax pipelines' float64 run (`diffusers_pipeline_reference.truth`): JAX's x64 on, their modules' float32
    read as float64 (`flax_precision.float64` over the generator's WIDENED) and each posterior draw the float32
    one widened."""
    import flax_precision
    import jax
    import jax.numpy as jnp
    from diffusers.models.vae_flax import FlaxDiagonalGaussianDistribution
    from diffusers_pipeline_reference import WIDENED

    def sample(self, key):
        return self.mean + self.std * jax.random.normal(key, self.mean.shape, jnp.float32).astype(self.mean.dtype)

    jax.config.update("jax_enable_x64", True)
    try:
        with flax_precision.float64(WIDENED), patch.object(FlaxDiagonalGaussianDistribution, "sample", sample):
            yield
    finally:
        jax.config.update("jax_enable_x64", False)


@functools.cache
def flax_pipeline(task: str, wide: bool):
    """The saved Flax pipeline of `task` and its parameters, built at float64 where `wide`."""
    import jax
    import jax.numpy as jnp
    from diffusers import (
        FlaxDDIMScheduler,
        FlaxStableDiffusionImg2ImgPipeline,
        FlaxStableDiffusionInpaintPipeline,
        FlaxStableDiffusionPipeline,
        FlaxStableDiffusionXLPipeline,
    )

    directory = saved() / task
    cls = {"sd": FlaxStableDiffusionPipeline, "xl": FlaxStableDiffusionXLPipeline,
           "img2img": FlaxStableDiffusionImg2ImgPipeline, "inpaint": FlaxStableDiffusionInpaintPipeline}[task]
    options = {} if task == "xl" else {"safety_checker": None, "feature_extractor": None}
    if not wide:
        return cls.from_pretrained(str(directory), local_files_only=True, **options)
    pipeline, params = cls.from_pretrained(str(directory), local_files_only=True, dtype=jnp.float64, **options)
    for name in ("text_encoder", "text_encoder_2") if task == "xl" else ("text_encoder",):
        setattr(pipeline, name, type(getattr(pipeline, name)).from_pretrained(str(directory / name),
                                                                              dtype=jnp.float64))
    pipeline.scheduler = FlaxDDIMScheduler.from_pretrained(str(directory), subfolder="scheduler",
                                                           dtype=jnp.float64)[0]
    params = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64)
                          if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating) else x, params)
    params["scheduler"] = pipeline.scheduler.create_state()
    return pipeline, params


def flax_call(task: str, case: dict[str, np.ndarray], wide: bool, wanted) -> dict[str, np.ndarray]:
    """`task`'s Flax pipeline called as tools/diffusers_pipeline_reference.py's `build` calls it, on the
    case's prompts, noise and pixels: the `wanted` leaves."""
    import jax
    import jax.numpy as jnp
    from PIL import Image

    dtype = jnp.float64 if wide else jnp.float32
    pipeline, params = flax_pipeline(task, wide)
    key = jax.random.PRNGKey(KEY)
    prompt, negative = text(case["prompt"]), text(case["negative"])
    image = Image.fromarray(case["pixels"])
    mask = np.zeros((32, 32), np.uint8)
    mask[:, 16:] = 255
    out = {}
    if task == "xl":
        tokens, neg = pipeline.prepare_inputs([prompt]), pipeline.prepare_inputs([negative])
        if "hidden" in wanted:
            out["hidden"], out["pooled"] = pipeline.get_embeddings(tokens, params)
    else:
        tokens, neg = (pipeline.tokenizer([value], padding="max_length", max_length=8, truncation=True,
                                          return_tensors="np").input_ids for value in (prompt, negative))
        if "hidden" in wanted:
            out["hidden"] = pipeline.text_encoder(tokens, params=params["text_encoder"])[0]
    noise = jnp.asarray(case["noise"].transpose(0, 3, 1, 2), dtype)
    vae, scale = {"params": params["vae"]}, pipeline.vae.config.scaling_factor
    if "encoded" in wanted:
        raw = jnp.asarray(case["pixels"][None], dtype).transpose(0, 3, 1, 2) / 127.5 - 1
        out["encoded"] = pipeline.vae.apply(vae, raw, method=pipeline.vae.encode).latent_dist.sample(key) * scale
        out["decoded"] = pipeline.vae.apply(vae, noise / scale, method=pipeline.vae.decode).sample.transpose(0, 2, 3,
                                                                                                         1)
    if "images" not in wanted:
        return out
    common = dict(params=params, prng_seed=key, num_inference_steps=2, height=32, width=32,
                  guidance_scale=3.0, neg_prompt_ids=neg)
    if task == "img2img":
        ids, processed = pipeline.prepare_inputs([prompt], image)
        output = pipeline(ids, image=jnp.asarray(processed, dtype), strength=1.0, noise=noise, **common)
    elif task == "inpaint":
        ids, processed, masks = pipeline.prepare_inputs([prompt], image, Image.fromarray(mask))
        output = pipeline(ids, mask=jnp.asarray(masks, dtype), masked_image=jnp.asarray(processed, dtype),
                          latents=noise, **common)
    else:
        output = pipeline(tokens, latents=noise, **common)
    out["images"] = output.images
    return out


@contextlib.contextmanager
def torch_drawing(noise, wide: bool):
    """A PyTorch pipeline's calls at float32, or at float64 (`diffusers_extended_reference.truth`: built
    and run in `torch_precision.float64`, DDIM's tables its module's own widened), every noise its call
    draws the case's drawn `noise`."""
    import torch
    from diffusers_wan_reference import float64_scheduler
    from torch_precision import float64

    def drawn(shape, generator=None, device=None, dtype=None, layout=None):
        if tuple(shape) != noise.shape:
            raise ValueError(f"a draw of {tuple(shape)} where the case drew {noise.shape}")
        return torch.from_numpy(noise.copy()).to(dtype)

    with contextlib.ExitStack() as scope:
        for name in ("diffusers.utils.torch_utils",
                     "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img",
                     "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint",
                     "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion",
                     "diffusers.schedulers.scheduling_ddim"):
            scope.enter_context(patch.object(importlib.import_module(name), "randn_tensor", drawn))
        if wide:
            scope.enter_context(float64())
            scope.enter_context(float64_scheduler(importlib.import_module("diffusers.schedulers.scheduling_ddim")))
        scope.enter_context(torch.no_grad())
        yield


@functools.cache
def torch_pipeline(task: str, wide: bool):
    """The saved PyTorch pipeline of `task`, at float64 where `wide`."""
    import torch
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
    )
    from torch_precision import float64

    cls = {"xl-inpaint": StableDiffusionXLInpaintPipeline,
           "safety": StableDiffusionPipeline}.get(task, StableDiffusionXLImg2ImgPipeline)
    with float64() if wide else contextlib.nullcontext():
        pipe = cls.from_pretrained(saved() / task, local_files_only=True)
        if wide:
            pipe = pipe.to(torch.float64)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def torch_call(task: str, case: dict[str, np.ndarray], wide: bool, wanted) -> dict[str, np.ndarray]:
    """`task`'s PyTorch pipeline called as tools/diffusers_extended_reference.py's `build` calls it, on the
    case's prompts, noise and pixels: the `wanted` leaves."""
    import torch
    from PIL import Image

    pipe = torch_pipeline(task, wide)
    meta = json.loads((saved() / task / "reference.json").read_text())
    dtype = torch.float64 if wide else torch.float32
    noise = case["noise"].transpose(0, 3, 1, 2)
    out = {}
    with torch_drawing(noise, wide):
        if task == "safety" and "images" in wanted:
            # The images before the checker's selection, which is not a value to hold on par.
            with patch.object(pipe, "run_safety_checker", lambda image, device, dtype: (image, None)):
                out["images"] = pipe(text(case["prompt"]), num_inference_steps=2, height=32, width=32,
                                     guidance_scale=3.0, latents=torch.from_numpy(noise.copy()).to(dtype),
                                     negative_prompt=text(case["negative"]), output_type="np").images
        if task == "safety" and "checker_pixels" in wanted:
            checker, extractor = pipe.safety_checker, pipe.feature_extractor
            if wide:
                # The processor's resize and crop, then its rescale and normalize in float64.
                resized = extractor(list(case["checker_raw"]), do_rescale=False, do_normalize=False,
                                    return_tensors="np").pixel_values.astype(np.float64)
                shape = (1, -1, 1, 1)
                out["checker_pixels"] = ((resized * extractor.rescale_factor - np.reshape(extractor.image_mean, shape))
                                         / np.reshape(extractor.image_std, shape))
            else:
                out["checker_pixels"] = extractor(list(case["checker_raw"]), return_tensors="pt").pixel_values
            tower = torch.from_numpy(case["checker_input"]).to(dtype)
            out["checker_embeddings"] = checker.visual_projection(checker.vision_model(tower)[1])
        if task == "safety":
            return {name: np.asarray(value) for name, value in out.items()}
        prompts = dict(prompt=text(case["prompt"]), prompt_2=text(case["prompt_2"]),
                       negative_prompt=text(case["negative"]), negative_prompt_2=text(case["negative_2"]))
        if "hidden" in wanted:
            hidden, _, pooled, _ = pipe.encode_prompt(device=torch.device("cpu"), **prompts)
            return {"hidden": hidden.numpy(), "pooled": pooled.numpy()}
        scale = pipe.vae.config.scaling_factor
        processed = pipe.image_processor.preprocess(Image.fromarray(case["pixels"]), height=32, width=32).to(dtype)
        kwargs = dict(**prompts, num_inference_steps=meta["steps"], guidance_scale=3.0, height=32, width=32,
                      generator=torch.Generator().manual_seed(KEY), output_type="latent")
        if task == "xl-inpaint":
            mask = np.zeros((32, 32), np.uint8)
            mask[:, 16:] = 255
            masked = processed * torch.tensor(mask[None, None] < 128)
            kwargs.update(image=Image.fromarray(case["pixels"]), mask_image=Image.fromarray(mask), strength=1.0,
                          latents=torch.from_numpy(noise.copy()).to(dtype),
                          masked_image_latents=pipe.vae.encode(masked).latent_dist.mode() * scale)
        else:
            kwargs.update(image=pipe.vae.encode(processed).latent_dist.mode() * scale, strength=meta["strength"])
            if "denoising_start" in meta:
                kwargs["denoising_start"] = meta["denoising_start"]
        prepared = pipe.prepare_latents

        def prepare_latents(*arguments, **options):
            out["start"] = prepared(*arguments, **options).permute(0, 2, 3, 1)
            return out["start"].permute(0, 3, 1, 2)

        with patch.object(pipe, "prepare_latents", prepare_latents) if task == "xl-img2img" else contextlib.nullcontext():
            final = pipe(**kwargs).images
        decoded = pipe.vae.decode(final / scale, return_dict=False)[0]
    out.update(final_latents=final.permute(0, 2, 3, 1), images=(decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1))
    return {name: np.asarray(value) for name, value in out.items()}


def grid_walk(name: str, case: dict[str, np.ndarray], wide: bool) -> dict[str, np.ndarray]:
    """Grid `name`'s walk of the saved SD pipeline from the case's noise."""
    from diffusers_scheduler_pipeline_reference import walk

    return {"latents": walk(name, saved() / "sd", flax_pipeline("sd", wide), case["noise"], wide)}


def run(site: str, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every case's outputs in float32 (`reference`) and float64 (`truth`), for `site` (`grid-NAME`, or
    `TASK-GROUP`, the group's leaves)."""
    task, group = (site, None) if site.startswith("grid-") else site.rsplit("-", 1)
    wanted = () if group is None else groups(task)[group]
    names = sorted({key.split(".", 1)[0] for key in cases})
    out = {}
    for kind, wide in (("reference", False), ("truth", True)):
        for name in names:
            case = {key.removeprefix(f"{name}."): value for key, value in cases.items()
                    if key.startswith(f"{name}.")}
            if task in FLAX or task.startswith("grid-"):
                with flax_float64() if wide else contextlib.nullcontext():
                    answered = (grid_walk(task.removeprefix("grid-"), case, wide) if group is None
                                else flax_call(task, case, wide, wanted))
            else:
                answered = torch_call(task, case, wide, wanted)
            for leaf in answered if group is None else wanted:
                out[f"{name}.{kind}.{leaf}"] = np.asarray(answered[leaf], np.float64 if wide else np.float32)
    return out


if __name__ == "__main__":
    import torch
    from reference_releases import require

    require("diffusers-pipelines")
    torch.set_num_threads(2)
    serve(functools.partial(run, sys.argv[1].removeprefix("native-")))
