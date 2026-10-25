"""Tiny SDXL image/refiner and checker-bearing references, without downloads.

Run with the diffusers-pipelines cohort's releases (tools/reference_releases.py) in the
isolated reference env:
  python tools/diffusers_extended_reference.py BASE_XL OUTPUT TASK
TASK: xl-img2img, xl-inpaint, refiner, safety. BASE_XL comes from
 diffusers_pipeline_reference.py (use its SD output for safety).

`truth ROOT OUTPUT` reruns each saved task under ROOT in float64, the truth its float32 reference is
measured from (`torch_precision.float64`), every noise the call draws the float32 draw widened
so the draws are the reference's own, and adds them as `{task}/{name}` to the npz OUTPUT
(tests/fixtures/tiny_diffusers_f64.npz, which `diffusers_pipeline_reference.py truth` writes first).
"""
import argparse
import json
from pathlib import Path
import tempfile

import jax
import numpy as np
import torch
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from PIL import Image
from diffusers import (FlaxStableDiffusionPipeline, FlaxStableDiffusionXLPipeline,
                       FlaxUNet2DConditionModel, UNet2DConditionModel, AutoencoderKL,
                       DDIMScheduler, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline)
from diffusers.models.modeling_pytorch_flax_utils import load_flax_checkpoint_in_pytorch_model
from transformers import (CLIPConfig, CLIPVisionConfig, CLIPImageProcessor,
                          CLIPTextModel, CLIPTextModelWithProjection)
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from reference_releases import require


def convert(model_class, config, parameters, *, vae=False):
    model = model_class.from_config(dict(config))
    if vae:
        names = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out_0"}
        parameters = unflatten_dict({tuple(names.get(k, k) for k in keys): value
                                     for keys, value in flatten_dict(parameters).items()})
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as checkpoint:
        checkpoint.write(serialization.to_bytes(parameters))
        checkpoint.flush()
        return load_flax_checkpoint_in_pytorch_model(model, checkpoint.name)


def build(base, directory, task):
    require("diffusers-pipelines")
    base, directory = Path(base), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if task == "safety":
        source, params = FlaxStableDiffusionPipeline.from_pretrained(
            base, safety_checker=None, feature_extractor=None, local_files_only=True)
        vision = CLIPVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                                  num_attention_heads=2, image_size=8, patch_size=4, projection_dim=8)
        config = CLIPConfig.from_text_vision_configs(source.text_encoder.config, vision, projection_dim=8)
        torch.manual_seed(23)
        checker = StableDiffusionSafetyChecker(config)
        processor = CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8})
        text = CLIPTextModel.from_pretrained(base / "text_encoder", from_flax=True)
        pipe = StableDiffusionPipeline(
            vae=convert(AutoencoderKL, source.vae.config, params["vae"], vae=True), text_encoder=text,
            tokenizer=source.tokenizer, unet=convert(UNet2DConditionModel, source.unet.config, params["unet"]),
            scheduler=DDIMScheduler.from_config(dict(source.scheduler.config)), safety_checker=checker, feature_extractor=processor)
        pipe.save_pretrained(directory, safe_serialization=True)
        pipe.set_progress_bar_config(disable=True)
        noise = np.asarray(jax.random.normal(jax.random.PRNGKey(17), (1, 4, 8, 8)))
        with torch.no_grad():
            output = pipe("cat", num_inference_steps=2, height=32, width=32, guidance_scale=3.0,
                          latents=torch.from_numpy(noise.copy()), negative_prompt="dog", output_type="np")
            images = output.images
            pixels = np.rint(images * 255).astype(np.uint8)
            features = processor(list(pixels), return_tensors="pt").pixel_values
            embeddings = checker.visual_projection(checker.vision_model(features)[1])
        np.savez_compressed(directory / "reference.npz", images=images, noise=noise.transpose(0, 2, 3, 1),
                            checker_pixels=features.numpy(), checker_embeddings=embeddings.numpy(),
                            checker_flags=np.asarray(output.nsfw_content_detected))
        (directory / "reference.json").write_text(json.dumps({"task": task, "from_pt": True}))
        print(task, "saved checker-bearing PyTorch pipeline")
        return
    flax_pipe, params = FlaxStableDiffusionXLPipeline.from_pretrained(base, local_files_only=True)
    refiner = task == "refiner"
    inpaint = task == "xl-inpaint"
    unet_config = dict(flax_pipe.unet.config)
    unet_config.update(in_channels=9 if inpaint else 4, cross_attention_dim=8 if refiner else 16,
                       projection_class_embeddings_input_dim=18 if refiner else 20)
    flax_unet = FlaxUNet2DConditionModel.from_config(unet_config)
    torch.manual_seed(31)
    unet = UNet2DConditionModel.from_config(dict(flax_unet.config))
    vae = convert(AutoencoderKL, flax_pipe.vae.config, params["vae"], vae=True)
    text = None if refiner else CLIPTextModel.from_pretrained(base / "text_encoder", from_flax=True)
    text2 = CLIPTextModelWithProjection.from_pretrained(base / "text_encoder_2", from_flax=True)
    scheduler = DDIMScheduler.from_config(dict(flax_pipe.scheduler.config))
    cls = StableDiffusionXLInpaintPipeline if inpaint else StableDiffusionXLImg2ImgPipeline
    pipe = cls(vae=vae, unet=unet, text_encoder=text, text_encoder_2=text2,
               tokenizer=None if refiner else flax_pipe.tokenizer, tokenizer_2=flax_pipe.tokenizer_2,
               scheduler=scheduler, requires_aesthetics_score=refiner, add_watermarker=False)
    pipe.save_pretrained(directory, safe_serialization=True)
    pipe.set_progress_bar_config(disable=True)
    pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    image = Image.fromarray(pixels)
    mask = np.zeros((32, 32), np.uint8)
    mask[:, 16:] = 255
    processed = pipe.image_processor.preprocess(image, height=32, width=32)
    noise = torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(17))
    with torch.no_grad():
        image_latents = pipe.vae.encode(processed).latent_dist.mode() * pipe.vae.config.scaling_factor
        prompt, negative, pooled, neg_pooled = pipe.encode_prompt(
            "cat", prompt_2="fox", device=torch.device("cpu"), negative_prompt="dog", negative_prompt_2="owl")
        kwargs = dict(prompt="cat", prompt_2="fox", negative_prompt="dog", negative_prompt_2="owl",
                      num_inference_steps=4, guidance_scale=3.0, height=32, width=32,
                      generator=torch.Generator().manual_seed(17), output_type="latent")
        meta = {"task": task, "from_pt": True, "steps": 4, "strength": 0.5}
        arrays = dict(pixels=pixels, processed=processed.numpy(), noise=noise.permute(0, 2, 3, 1).numpy(),
                      image_latents=image_latents.permute(0, 2, 3, 1).numpy(), hidden=prompt.numpy(), pooled=pooled.numpy(),
                      negative_hidden=negative.numpy(), negative_pooled=neg_pooled.numpy())
        if inpaint:
            masked = processed * torch.tensor(mask[None, None] < 128)
            masked_latents = pipe.vae.encode(masked).latent_dist.mode() * pipe.vae.config.scaling_factor
            kwargs.update(image=image, mask_image=Image.fromarray(mask), strength=1.0, latents=noise,
                          masked_image_latents=masked_latents)
            meta["strength"] = 1.0
            arrays.update(mask=mask, masked_latents=masked_latents.permute(0, 2, 3, 1).numpy())
        else:
            kwargs.update(image=image_latents, strength=0.5)
            if refiner:
                kwargs["denoising_start"] = 0.5
                meta["denoising_start"] = 0.5
        final = pipe(**kwargs).images
        decoded = pipe.vae.decode(final / pipe.vae.config.scaling_factor, return_dict=False)[0]
        arrays.update(final_latents=final.permute(0, 2, 3, 1).numpy(),
                      images=(decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy())
    np.savez_compressed(directory / "reference.npz", **arrays)
    (directory / "reference.json").write_text(json.dumps(meta))
    print(task, "saved official PyTorch pipeline; latent sum", float(final.sum()))


def truth(root, destination):
    """Each saved PyTorch task rerun in float64 (see the module docstring)."""
    import importlib
    from contextlib import ExitStack
    from unittest.mock import patch

    from diffusers.utils import torch_utils
    from diffusers_wan_reference import float64_scheduler
    from torch_precision import float64

    require("diffusers-pipelines")
    torch.set_num_threads(2)
    wide = torch.float64
    draw = torch_utils.randn_tensor

    def narrow_draw(shape, generator=None, device=None, dtype=None, layout=None):
        return draw(shape, generator=generator, device=device, dtype=torch.float32, layout=layout).to(dtype)

    with np.load(destination) as stored:
        arrays = dict(stored)
    with ExitStack() as scope:
        for name in ("diffusers.utils.torch_utils",
                     "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img",
                     "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint",
                     "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion",
                     "diffusers.schedulers.scheduling_ddim"):
            scope.enter_context(patch.object(importlib.import_module(name), "randn_tensor", narrow_draw))
        scope.enter_context(float64())
        # The DDIM scheduler's tables are its module's own, built and set at every call inside it.
        scope.enter_context(float64_scheduler(importlib.import_module("diffusers.schedulers.scheduling_ddim")))
        for task in ("xl-img2img", "xl-inpaint", "refiner", "safety"):
            directory = Path(root) / task
            reference = np.load(directory / "reference.npz")
            meta = json.loads((directory / "reference.json").read_text())
            got = {}
            if task == "safety":
                pipe = StableDiffusionPipeline.from_pretrained(directory, local_files_only=True).to(wide)
                pipe.set_progress_bar_config(disable=True)
                noise = torch.from_numpy(reference["noise"].transpose(0, 3, 1, 2).copy()).to(wide)
                with torch.no_grad():
                    got["images"] = pipe("cat", num_inference_steps=2, height=32, width=32, guidance_scale=3.0,
                                         latents=noise, negative_prompt="dog", output_type="np").images
                    pixels = np.rint(reference["images"] * 255).astype(np.uint8)
                    # The processor's resize and crop, then its rescale and normalize in float64.
                    extractor = pipe.feature_extractor
                    resized = extractor(list(pixels), do_rescale=False, do_normalize=False,
                                        return_tensors="np").pixel_values.astype(np.float64)
                    shape = (1, -1, 1, 1)
                    features = torch.from_numpy(
                        (resized * extractor.rescale_factor - np.reshape(extractor.image_mean, shape))
                        / np.reshape(extractor.image_std, shape))
                    checker = pipe.safety_checker
                    got["checker_pixels"] = features.numpy()
                    got["checker_embeddings"] = checker.visual_projection(checker.vision_model(features)[1]).numpy()
            else:
                cls = StableDiffusionXLInpaintPipeline if task == "xl-inpaint" else StableDiffusionXLImg2ImgPipeline
                pipe = cls.from_pretrained(directory, local_files_only=True).to(wide)
                pipe.set_progress_bar_config(disable=True)
                processed = torch.from_numpy(reference["processed"]).to(wide)
                noise = torch.from_numpy(reference["noise"].transpose(0, 3, 1, 2).copy()).to(wide)
                with torch.no_grad():
                    scale = pipe.vae.config.scaling_factor
                    image_latents = pipe.vae.encode(processed).latent_dist.mode() * scale
                    prompt, _, pooled, _ = pipe.encode_prompt(
                        "cat", prompt_2="fox", device=torch.device("cpu"), negative_prompt="dog",
                        negative_prompt_2="owl")
                    got.update(hidden=prompt.numpy(), pooled=pooled.numpy(),
                               image_latents=image_latents.permute(0, 2, 3, 1).numpy())
                    kwargs = dict(prompt="cat", prompt_2="fox", negative_prompt="dog", negative_prompt_2="owl",
                                  num_inference_steps=meta["steps"], guidance_scale=3.0, height=32, width=32,
                                  generator=torch.Generator().manual_seed(17), output_type="latent")
                    if task == "xl-inpaint":
                        mask = reference["mask"]
                        masked = processed * torch.tensor(mask[None, None] < 128)
                        masked_latents = pipe.vae.encode(masked).latent_dist.mode() * scale
                        kwargs.update(image=Image.fromarray(reference["pixels"]), mask_image=Image.fromarray(mask),
                                      strength=1.0, latents=noise, masked_image_latents=masked_latents)
                        got["masked_latents"] = masked_latents.permute(0, 2, 3, 1).numpy()
                    else:
                        kwargs.update(image=image_latents, strength=meta["strength"])
                        if "denoising_start" in meta:
                            kwargs["denoising_start"] = meta["denoising_start"]
                    final = pipe(**kwargs).images
                    decoded = pipe.vae.decode(final / scale, return_dict=False)[0]
                    got.update(final_latents=final.permute(0, 2, 3, 1).numpy(),
                               images=(decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy())
            gaps = {name: float(np.max(np.abs(np.asarray(value, np.float64) - reference[name]))) for name, value in got.items()}
            print(json.dumps({"task": task, "float32_from_float64": gaps,
                              "dtypes": sorted({str(np.asarray(value).dtype) for value in got.values()})}))
            arrays.update({f"{task}/{name}": np.asarray(value) for name, value in got.items()})
    np.savez_compressed(destination, **arrays)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "truth":
        truth(sys.argv[2], sys.argv[3])
        raise SystemExit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("directory")
    parser.add_argument("task", choices=("safety", "xl-img2img", "xl-inpaint", "refiner"))
    args = parser.parse_args()
    build(args.base, args.directory, args.task)
