"""Tiny SDXL image/refiner and checker-bearing references, without downloads.

Run with Diffusers 0.34.0 / Transformers 4.49.0 in the isolated reference env:
  python tools/diffusers_extended_reference.py BASE_XL OUTPUT TASK
TASK: xl-img2img, xl-inpaint, refiner, safety. BASE_XL comes from
 diffusers_pipeline_reference.py (use its SD output for safety).
"""
import argparse
import json
from pathlib import Path
import tempfile

import diffusers
import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
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
    if (diffusers.__version__, transformers.__version__) != ("0.34.0", "4.49.0"):
        raise RuntimeError("Use the pinned isolated reference environment")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("directory")
    parser.add_argument("task", choices=("safety", "xl-img2img", "xl-inpaint", "refiner"))
    args = parser.parse_args()
    build(args.base, args.directory, args.task)
