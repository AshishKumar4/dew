"""Generate saved tiny Flax pipelines and reference arrays, without downloads.

Reference: diffusers==0.34.0, transformers==4.49.0, float32, CPU.
Run in an isolated reference environment:
  JAX_PLATFORMS=cpu USE_TF=0 python tools/diffusers_pipeline_reference.py DIR sd
Repeat for xl, img2img, inpaint. DIR contains actual save_pretrained components,
an authentic byte-level CLIP tokenizer and deterministic random parameters.
"""
import argparse
import json
from pathlib import Path

import diffusers
import jax
import jax.numpy as jnp
import numpy as np
import transformers
from diffusers import (
    FlaxAutoencoderKL, FlaxDDIMScheduler, FlaxUNet2DConditionModel,
    FlaxStableDiffusionPipeline, FlaxStableDiffusionXLPipeline,
    FlaxStableDiffusionImg2ImgPipeline, FlaxStableDiffusionInpaintPipeline,
)
from PIL import Image
from transformers import CLIPTextConfig, CLIPTokenizer, FlaxCLIPTextModel, FlaxCLIPTextModelWithProjection
from transformers.models.clip.tokenization_clip import bytes_to_unicode


def build(directory, task):
    if (diffusers.__version__, transformers.__version__) != ("0.34.0", "4.49.0"):
        raise RuntimeError("Use the pinned isolated reference environment")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vocabulary = list(bytes_to_unicode().values())
    vocabulary += [token + "</w>" for token in vocabulary]
    vocabulary += ["<|startoftext|>", "<|endoftext|>"]
    vocab = directory / "vocab.json"
    merges = directory / "merges.txt"
    vocab.write_text(json.dumps(dict(zip(vocabulary, range(len(vocabulary))))))
    merges.write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(vocab), str(merges), model_max_length=8)
    config = CLIPTextConfig(vocab_size=len(vocabulary), hidden_size=8, intermediate_size=16,
                            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
                            projection_dim=8, bos_token_id=512, eos_token_id=513, pad_token_id=513)
    text = FlaxCLIPTextModel(config, seed=0)
    vae = FlaxAutoencoderKL(block_out_channels=(32, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 3, up_block_types=("UpDecoderBlock2D",) * 3,
        layers_per_block=1, latent_channels=4, sample_size=32)
    xl = task == "xl"
    unet = FlaxUNet2DConditionModel(sample_size=8, in_channels=9 if task == "inpaint" else 4,
        out_channels=4, down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",), block_out_channels=(32,), layers_per_block=1,
        attention_head_dim=4, cross_attention_dim=16 if xl else 8,
        addition_embed_type="text_time" if xl else None,
        addition_time_embed_dim=2 if xl else None,
        projection_class_embeddings_input_dim=20 if xl else None)
    scheduler = FlaxDDIMScheduler(num_train_timesteps=20, beta_start=0.00085, beta_end=0.012,
                                 beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    params = {"vae": vae.init_weights(jax.random.PRNGKey(1)),
              "unet": unet.init_weights(jax.random.PRNGKey(2)),
              "text_encoder": text.params, "scheduler": scheduler.create_state()}
    components = dict(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
    if xl:
        config2 = CLIPTextConfig(**{**config.to_dict(), "hidden_act": "gelu"})
        text2 = FlaxCLIPTextModelWithProjection(config2, seed=3)
        params["text_encoder_2"] = text2.params
        pipeline = FlaxStableDiffusionXLPipeline(**components, text_encoder_2=text2, tokenizer_2=tokenizer)
    else:
        cls = {"sd": FlaxStableDiffusionPipeline, "img2img": FlaxStableDiffusionImg2ImgPipeline,
               "inpaint": FlaxStableDiffusionInpaintPipeline}[task]
        pipeline = cls(**components, safety_checker=None, feature_extractor=None)
    pipeline.save_pretrained(str(directory), params=params)
    # Prove references also consume the saved component layout.
    pipeline, params = type(pipeline).from_pretrained(
        str(directory), local_files_only=True,
        **({} if xl else {"safety_checker": None, "feature_extractor": None}))
    key = jax.random.PRNGKey(17)
    pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    mask = np.zeros((32, 32), np.uint8)
    mask[:, 16:] = 255
    image = Image.fromarray(pixels)
    mask_image = Image.fromarray(mask)
    prompts, negative = ["cat"], ["dog"]
    tokens = tokenizer(prompts, padding="max_length", max_length=8, truncation=True, return_tensors="np").input_ids
    neg = tokenizer(negative, padding="max_length", max_length=8, truncation=True, return_tensors="np").input_ids
    arrays = dict(pixels=pixels, mask=mask, tokens=tokens)
    if xl:
        tokens, neg = pipeline.prepare_inputs(prompts), pipeline.prepare_inputs(negative)
        hidden, pooled = pipeline.get_embeddings(tokens, params)
        arrays.update(tokens=tokens, hidden=hidden, pooled=pooled)
    else:
        arrays["hidden"] = pipeline.text_encoder(tokens, params=params["text_encoder"])[0]
    common = dict(params=params, prng_seed=key, num_inference_steps=2, height=32, width=32,
                  guidance_scale=3.0, neg_prompt_ids=neg)
    noise = jax.random.normal(key, (1, 4, 8, 8))
    if task == "img2img":
        ids, processed = pipeline.prepare_inputs(prompts, image)
        arrays["processed"] = processed
        output = pipeline(ids, image=processed, strength=1.0, noise=noise, **common)
    elif task == "inpaint":
        ids, processed, masks = pipeline.prepare_inputs(prompts, image, mask_image)
        arrays.update(processed=processed, processed_mask=masks)
        output = pipeline(ids, mask=masks, masked_image=processed, latents=noise, **common)
    else:
        output = pipeline(tokens, latents=noise, **common)
    arrays["images"] = output.images
    raw = jnp.asarray(pixels[None]).transpose(0, 3, 1, 2) / 127.5 - 1
    posterior = pipeline.vae.apply({"params": params["vae"]}, raw, method=pipeline.vae.encode).latent_dist
    arrays["encoded"] = posterior.sample(key) * pipeline.vae.config.scaling_factor
    arrays["decoded"] = pipeline.vae.apply({"params": params["vae"]}, noise / pipeline.vae.config.scaling_factor,
                                            method=pipeline.vae.decode).sample.transpose(0, 2, 3, 1)
    arrays["noise"] = noise.transpose(0, 2, 3, 1)
    if task in ("sd", "xl"):
        added = None if not xl else {"text_embeds": arrays["pooled"],
                                     "time_ids": jnp.asarray([[32, 32, 0, 0, 32, 32]])}
        arrays["prediction"] = unet.apply({"params": params["unet"]}, noise, jnp.asarray([10]),
            encoder_hidden_states=arrays["hidden"], added_cond_kwargs=added).sample.transpose(0, 2, 3, 1)
    np.savez_compressed(directory / "reference.npz", **arrays)
    print(json.dumps({"task": task, "directory": str(directory), "reference": diffusers.__version__,
                      "transformers": transformers.__version__, "backend": jax.default_backend(),
                      "images_sum": float(np.asarray(output.images).sum())}))


def bundle(root, destination):
    """Bundle saved task directories; xz shares their repeated weights.

    Generate sd/xl/img2img/inpaint with this tool and the four additional
    tasks with diffusers_extended_reference.py, then call bundle(ROOT,
    "tests/fixtures/tiny_diffusers.tar.xz"). No model downloads.
    """
    import tarfile
    with tarfile.open(destination, "w:xz") as archive:
        for task in ("sd", "xl", "img2img", "inpaint", "xl-img2img", "xl-inpaint", "refiner", "safety"):
            archive.add((Path(root) / task).resolve(), arcname=task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("task", choices=("sd", "xl", "img2img", "inpaint"))
    args = parser.parse_args()
    build(args.directory, args.task)
