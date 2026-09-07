"""SD / SDXL with official Diffusers Flax UNet, VAE and schedulers.

Diffusers 0.34.0 full Flax pipelines require removed Transformers Flax towers.
The runtime instead uses Dew's CLIP primitives and an explicit weight-layout
bridge. Orchestration follows those pipelines; reference tools run the full
pipelines in an isolated Transformers 4.49.0 environment. No runtime downgrade.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import NamedTuple, Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from diffusers.models.vae_flax import FlaxAutoencoderKL, FlaxAutoencoderKLOutput, FlaxDecoderOutput
from diffusers.models.unets.unet_2d_condition_flax import FlaxUNet2DConditionModel
from diffusers.schedulers.scheduling_ddim_flax import FlaxDDIMScheduler, DDIMSchedulerState
from diffusers.schedulers.scheduling_pndm_flax import FlaxPNDMScheduler, PNDMSchedulerState
from diffusers.schedulers.scheduling_lms_discrete_flax import FlaxLMSDiscreteScheduler, LMSDiscreteSchedulerState
from diffusers.schedulers.scheduling_dpmsolver_multistep_flax import FlaxDPMSolverMultistepScheduler, DPMSolverMultistepSchedulerState
from diffusers.schedulers.scheduling_euler_discrete_flax import FlaxEulerDiscreteScheduler, EulerDiscreteSchedulerState
from flax import linen as nn, serialization
from PIL import Image
from transformers import CLIPTokenizer

from dew.diffusion.process import Process
from dew.diffusion.schedules.discrete import DiscreteNoiseScheduler
from dew.diffusion.transforms import EpsilonPredictionTransform, VPredictionTransform
from dew.inputs import Condition, ConditionEncoder, Field, InputSpec, unit_range
from dew.nn.autoencoders import AutoEncoder
from dew.nn.text_encoders import CLIPTextTransformer, translate_clip_weights, translate_config
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.base import Variables
from dew.sampling.guidance import CFG
from dew.sampling.pipelines import DenoisingInputs, TextToImage
from dew.interop.diffusers_safety import SafetyChecker
from dew.interop.diffusers_lms import LMSState, SourceLMS

SchedulerState = DDIMSchedulerState | PNDMSchedulerState | LMSDiscreteSchedulerState | DPMSolverMultistepSchedulerState | EulerDiscreteSchedulerState | LMSState


def _load_model(model_class, directory, *, dtype, from_pt):
    safe = directory / "diffusion_pytorch_model.safetensors"
    if not from_pt or not safe.is_file():
        loaded = model_class.from_pretrained(directory, dtype=dtype, from_pt=from_pt)
        return loaded[0], loaded[1]
    from safetensors.numpy import load_file
    from flax.traverse_util import flatten_dict, unflatten_dict
    from diffusers.models.modeling_flax_pytorch_utils import rename_key, rename_key_and_reshape_tensor
    model = model_class.from_config(model_class.load_config(directory), dtype=dtype)
    expected = flatten_dict(jax.eval_shape(model.init_weights, jax.random.PRNGKey(0)))
    converted = {}
    for name, tensor in load_file(safe).items():
        key, value = rename_key_and_reshape_tensor(tuple(rename_key(name).split(".")), tensor, expected)
        spec = expected.get(key)
        if not isinstance(spec, jax.ShapeDtypeStruct) or value.shape != spec.shape:
            raise ValueError(f"Unexpected tensor or shape in {directory}: {name}")
        converted[key] = jnp.asarray(value)
    if converted.keys() != expected.keys():
        raise ValueError(f"Missing component parameters in {directory}: {expected.keys() - converted.keys()}")
    return model, unflatten_dict(converted)



def _load_scheduler(scheduler_class, directory):
    loaded = scheduler_class.from_pretrained(directory, return_unused_kwargs=False)
    return loaded[0], loaded[1]


class _Scheduler(Protocol):
    """The shared callable surface of the official, state-specific schedulers."""
    def set_timesteps(self, state, num_inference_steps: int, shape: tuple) -> SchedulerState: ...
    def scale_model_input(self, state, sample, timestep) -> jax.Array: ...
    def add_noise(self, state, original_samples, noise, timesteps) -> jax.Array: ...
    def step(self, state, model_output, timestep, sample, *, return_dict: bool) -> tuple[jax.Array, SchedulerState]: ...
    def save_pretrained(self, save_directory) -> None: ...


class _CLIPFeatures(NamedTuple):
    last: jax.Array
    penultimate: jax.Array
    pooled: jax.Array


class Denoiser(nn.Module):
    """NHWC and Dew keyword conventions around the official Flax UNet."""
    unet: FlaxUNet2DConditionModel

    @nn.compact
    def __call__(self, x, t, *, conditioning, train=False):
        if "mask" in conditioning:
            x = jnp.concatenate([x, conditioning["mask"], conditioning["masked_image"]], axis=-1)
        output = self.unet(
            x.transpose(0, 3, 1, 2), t,
            encoder_hidden_states=conditioning["encoder_hidden_states"],
            added_cond_kwargs=conditioning.get("added_cond_kwargs"), train=train, return_dict=False)
        return output[0].transpose(0, 2, 3, 1)


@dataclass(eq=False)
class VAE(AutoEncoder):
    model: FlaxAutoencoderKL
    params: Variables
    latent_scale: float

    @property
    def downscale_factor(self):
        return 2 ** (len(self.model.config["block_out_channels"]) - 1)

    @property
    def latent_channels(self):
        return self.model.config["latent_channels"]

    def encode_batch(self, params, x, key=None):
        output = self.model.apply(
            {"params": params}, x.transpose(0, 3, 1, 2), method=self.model.encode)
        assert isinstance(output, FlaxAutoencoderKLOutput)
        posterior = output.latent_dist
        # The official posterior is NHWC, while encode accepts NCHW.
        return posterior.mode() if key is None else posterior.sample(key)

    def decode_batch(self, params, z):
        output = self.model.apply(
            {"params": params}, z.transpose(0, 3, 1, 2), method=self.model.decode)
        assert isinstance(output, FlaxDecoderOutput)
        return output.sample.transpose(0, 2, 3, 1)


def _clip_features(tower, ids):
    """Select SDXL's pre-final-layer hidden state using existing CLIP layers."""
    hidden = tower.token_embedding(ids) + tower.position_embedding(jnp.arange(ids.shape[1]))
    penultimate = hidden
    for layer in tower.layers:
        penultimate = hidden
        hidden = layer(hidden)
    hidden = tower.final_layer_norm(hidden)
    index = jnp.argmax(ids if tower.eos_token_id == 2 else ids == tower.eos_token_id, axis=-1)
    return _CLIPFeatures(hidden, penultimate, hidden[jnp.arange(ids.shape[0]), index])


def _native_text(params):
    text = params["text_model"]
    return {**text["embeddings"], "final_layer_norm": text["final_layer_norm"],
            **{f"layers_{i}": layer for i, layer in text["encoder"]["layers"].items()}}


def _official_text(params):
    native = params["text_model"]
    return {"text_model": {
        "embeddings": {key: native[key] for key in ("token_embedding", "position_embedding")},
        "final_layer_norm": native["final_layer_norm"],
        "encoder": {"layers": {key.removeprefix("layers_"): value for key, value in native.items()
                               if key.startswith("layers_")}},
    }, **({"text_projection": params["text_projection"]} if "text_projection" in params else {})}

@partial(jax.jit, static_argnums=(0,))
def _predict(model, variables, sample, time, conditions):
    output = model.apply(variables, sample, time, **conditions)
    assert isinstance(output, jax.Array)
    return output



@dataclass(eq=False)
class Prompts(ConditionEncoder[str]):
    towers: tuple[CLIPTextTransformer, ...]
    tokenizers: tuple[CLIPTokenizer, ...]
    configs: tuple[dict, ...]
    params: Variables
    checkpoint: str
    height: int
    width: int
    names: tuple[str, ...]
    xl: bool
    aesthetics: bool = False

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs):
        return load_diffusers_pipeline(checkpoint, **kwargs).encoder

    def tokenize(self, data: Sequence[str], second=None):
        ids = []
        for name, tokenizer in zip(self.names, self.tokenizers):
            rows = second if len(self.towers) == 2 and name == "text_encoder_2" and second is not None else data
            ids.append(tokenizer(list(rows), padding="max_length", max_length=tokenizer.model_max_length,
                                 truncation=True, return_tensors="np").input_ids)
        return {"input_ids": np.stack(ids, axis=1) if self.xl else ids[0]}

    def time_ids(self, count, dtype, *, original_size=None, crops_coords_top_left=(0, 0),
                 target_size=None, aesthetic_score=6.0):
        size = (self.height, self.width)
        values = (*(original_size or size), *crops_coords_top_left,
                  *((aesthetic_score,) if self.aesthetics else (target_size or size)))
        return jnp.broadcast_to(jnp.asarray(values, dtype), (count, len(values)))

    def encode(self, params, tokens):
        ids = tokens["input_ids"]
        outputs = []
        for i, (name, tower) in enumerate(zip(self.names, self.towers)):
            output = tower.apply({"params": params[name]["text_model"]},
                                 ids[:, i] if self.xl else ids, method=_clip_features)
            assert isinstance(output, _CLIPFeatures)
            outputs.append(output)
        if not self.xl:
            return {"encoder_hidden_states": outputs[0].last}
        hidden = jnp.concatenate([output.penultimate for output in outputs], axis=-1)
        pooled = outputs[-1].pooled @ params[self.names[-1]]["text_projection"]["kernel"]
        return {"encoder_hidden_states": hidden, "added_cond_kwargs": {
            "text_embeds": pooled, "time_ids": tokens.get("time_ids", self.time_ids(ids.shape[0], hidden.dtype))}}

    def captions(self, tokens):
        ids = tokens["input_ids"][:, 0] if self.xl else tokens["input_ids"]
        return tuple(self.tokenizers[0].batch_decode(np.asarray(ids), skip_special_tokens=True))

    def to_json(self):
        return {"checkpoint": self.checkpoint, "height": self.height, "width": self.width}


class PretrainedObjective(DiffusionObjective):
    """Loaded denoiser training, including nine-channel masked-image conditions.

    Inpainting batches add binary NHWC mask (white pixels are repainted).
    Caption dropout never drops the spatial mask or masked-image latents.
    """
    def __init__(self, pipe, **kwargs):
        self.pipe = pipe
        self.loaded = pipe.params
        super().__init__(pipe.model, pipe.process, pipe.inputs, autoencoder=pipe.autoencoder, **kwargs)
        self.unconditional = pipe.conditions([""])[1]
        self._unfiltered_sample = self._sample
        if pipe.safety_checker is not None:
            self._sample = self._checked_sample

    def _checked_sample(self, params, batch, key, *, count):
        images = self._unfiltered_sample(params, batch, key, count=count)
        return self.pipe.bind(params)._finish(images)

    def _conditions(self, params, batch, key, *, dropout):
        given, null = super()._conditions(params, batch, key, dropout=dropout)
        if self.pipe.task != "inpainting" or self.pipe.model.unet.config["in_channels"] != 9:
            return given, null
        pixels = unit_range(batch[self.inputs.sample.key])
        mask = jnp.asarray(batch["mask"], jnp.float32)
        if mask.shape != (*pixels.shape[:3], 1):
            raise ValueError("Inpainting training mask must be binary NHWC with one channel")
        masked = self.pipe.vae.encode(params["autoencoder"], pixels * (mask < 0.5), jax.random.fold_in(key, 1))
        mask = jax.image.resize(mask, (*masked.shape[:3], 1), method="nearest")
        spatial = {"mask": mask, "masked_image": masked}
        return ({"conditioning": {**given["conditioning"], **spatial}},
                {"conditioning": {**null["conditioning"], **spatial}})

    def _sampling_batch(self, batch):
        selected = super()._sampling_batch(batch)
        return {**selected, "mask": batch["mask"]} if self.pipe.task == "inpainting" else selected

    def text_to_image(self, variables):
        return self.pipe.bind(variables)

    def init(self, key):
        return self.loaded



@dataclass(frozen=True, eq=False, kw_only=True)
class DiffusersTextToImage(TextToImage):
    model: Denoiser
    encoder: Prompts
    scheduler: _Scheduler
    scheduler_state: SchedulerState
    config: dict
    task: str
    safety_checker: SafetyChecker | None = None

    @property
    def vae(self) -> VAE:
        assert isinstance(self.autoencoder, VAE)
        return self.autoencoder

    def bind(self, variables):
        return replace(self, params=variables)

    def objective(self, **kwargs) -> DiffusionObjective:
        """Train the UNet; batch uint8 NHWC images with inputs.tokenize(captions).

        init(key) returns loaded variables, not fresh random weights. VAE/text
        towers remain frozen. Bind trained variables before sampling/saving.
        """
        return PretrainedObjective(self, **kwargs)

    def conditions(self, prompts, negative_prompts=None, *, prompt_2=None, negative_prompt_2=None,
                   original_size=None, crops_coords_top_left=(0, 0), target_size=None,
                   negative_original_size=None, negative_crops_coords_top_left=(0, 0), negative_target_size=None,
                   aesthetic_score=6.0, negative_aesthetic_score=2.5):
        rows = [prompts] if isinstance(prompts, str) else list(prompts)
        def paired(value, default):
            values = default if value is None else [value] * len(rows) if isinstance(value, str) else list(value)
            if len(values) != len(rows):
                raise ValueError("Conditioning prompts must have one row per prompt")
            return values
        params = self.params["encoders"]["conditioning"]
        given = self.encoder.encode(params, self.encoder.tokenize(rows, paired(prompt_2, rows)))
        if negative_prompts is None and negative_prompt_2 is None and self.encoder.xl and self.config.get("force_zeros_for_empty_prompt", True):
            null = {"encoder_hidden_states": jnp.zeros_like(given["encoder_hidden_states"]),
                    "added_cond_kwargs": {"text_embeds": jnp.zeros_like(given["added_cond_kwargs"]["text_embeds"]),
                                          "time_ids": given["added_cond_kwargs"]["time_ids"]}}
        else:
            negatives = paired(negative_prompts, [""] * len(rows))
            null = self.encoder.encode(params, self.encoder.tokenize(negatives, paired(negative_prompt_2, negatives)))
        if self.encoder.xl:
            dtype = given["encoder_hidden_states"].dtype
            given["added_cond_kwargs"]["time_ids"] = self.encoder.time_ids(
                len(rows), dtype, original_size=original_size, crops_coords_top_left=crops_coords_top_left,
                target_size=target_size, aesthetic_score=aesthetic_score)
            null["added_cond_kwargs"]["time_ids"] = self.encoder.time_ids(
                len(rows), dtype, original_size=negative_original_size or original_size,
                crops_coords_top_left=negative_crops_coords_top_left,
                target_size=negative_target_size or target_size, aesthetic_score=negative_aesthetic_score)
        return {"conditioning": given}, {"conditioning": null}

    def prepare(self, prompts, *, key, negative_prompts=None, **conditioning_options):
        if self.task != "text-to-image":
            raise ValueError("Image tasks require their image/mask call contract")
        given, null = self.conditions(prompts, negative_prompts, **conditioning_options)
        count = given["conditioning"]["encoder_hidden_states"].shape[0]
        height, width, channels = self.latent_shape
        noise = jax.random.normal(key, (count, channels, height, width)).transpose(0, 2, 3, 1)
        return DenoisingInputs(noise, given, null)

    def _finish(self, images):
        images = jnp.clip(images, -1, 1)
        if self.safety_checker is not None:
            return self.safety_checker.filter(self.params["encoders"]["safety_checker"], images)
        return images


    def save_pretrained(self, directory: str):
        """Write current bound parameters in the official Flax directory layout."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model_index.json").write_text(json.dumps(self.config, indent=2))
        self.model.unet.save_pretrained(destination / "unet", params=self.params["params"]["unet"])
        self.vae.model.save_pretrained(destination / "vae", params=self.params["autoencoder"])
        self.scheduler.save_pretrained(destination / "scheduler")
        if self.safety_checker is not None:
            self.safety_checker.save(destination, self.params["encoders"]["safety_checker"])
        for name, config, tokenizer in zip(self.encoder.names, self.encoder.configs, self.encoder.tokenizers):
            suffix = name.removeprefix("text_encoder")
            folder = destination / ("text_encoder" + suffix)
            folder.mkdir(exist_ok=True)
            (folder / "config.json").write_text(json.dumps(config, indent=2))
            params = self.params["encoders"]["conditioning"]["text_encoder" + suffix]
            (folder / "flax_model.msgpack").write_bytes(serialization.to_bytes(_official_text(params)))
            tokenizer.save_pretrained(destination / ("tokenizer" + suffix))

    def prepare_image(self, image, mask=None):
        """PIL images become NHWC [-1,1]; array images already use that range.

        SD follows the Flax multiple-of-32 resize. SDXL uses the configured
        resolution. A white binary mask marks the region to repaint.
        """
        if isinstance(image, (np.ndarray, jax.Array)):
            pixels = jnp.asarray(image)
            if pixels.ndim == 3:
                pixels = pixels[None]
        else:
            images = [image] if isinstance(image, Image.Image) else list(image)
            pixels = []
            for item in images:
                width, height = ((self.encoder.width, self.encoder.height) if self.encoder.xl else
                                 tuple(size - size % 32 for size in item.size))
                if not width or not height:
                    raise ValueError("Image dimensions must be at least 32")
                resized = item.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                value = (np.asarray(resized, np.float32) / 255 if self.encoder.xl
                         else jnp.asarray(resized, jnp.float32) / 255)
                pixels.append(jnp.asarray(value) * 2 - 1)
            pixels = jnp.stack(pixels)
        if mask is None:
            return pixels, None
        if isinstance(mask, (np.ndarray, jax.Array)):
            masks = jnp.asarray(mask, jnp.float32)
            if masks.ndim == 2:
                masks = masks[None, ..., None]
            elif masks.ndim == 3:
                masks = masks[..., None]
        else:
            masks = [mask] if isinstance(mask, Image.Image) else list(mask)
            values = []
            for item in masks:
                width, height = ((self.encoder.width, self.encoder.height) if self.encoder.xl else
                                 tuple(size - size % 32 for size in item.size))
                resized = item.resize((width, height), Image.Resampling.NEAREST) if self.encoder.xl else item.resize((width, height))
                values.append(jnp.asarray(resized.convert("L"), jnp.float32)[..., None] / 255)
            masks = jnp.stack(values)
        masks = (masks >= 0.5).astype(jnp.float32)
        if masks.shape != (*pixels.shape[:3], 1):
            raise ValueError("Image and mask batches must match after preprocessing")
        return pixels * (masks < 0.5), masks

    def __call__(self, prompts, *, steps=50, guidance: CFG | float | None = 7.5, sampler=None, key,
                 negative_prompts=None, image=None, mask=None, strength=None, latents=None,
                 masked_image_latents=None, denoising_start=None, denoising_end=None,
                 output_type="image", **conditioning_options):
        """Sample images or NHWC latents with the checkpoint scheduler.

        img2img/refiner accepts PIL pixels or NHWC latents as image. Inpainting
        accepts image and binary mask; optional masked_image_latents bypasses
        its VAE draw. denoising_end/start connect a base and refiner without
        decoding between them. Explicit Dew solvers are text-to-image only.
        """
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if output_type not in ("image", "latent"):
            raise ValueError("output_type must be image or latent")
        for fraction in (denoising_start, denoising_end):
            if fraction is not None and not 0 < fraction < 1:
                raise ValueError("denoising fractions must lie strictly between zero and one")
        if denoising_start is not None and denoising_end is not None and denoising_start >= denoising_end:
            raise ValueError("denoising_start must precede denoising_end")
        given: Variables
        null: Variables
        if isinstance(prompts, DenoisingInputs):
            if self.task != "text-to-image" or any(value is not None for value in (negative_prompts, image, mask, latents)) or conditioning_options:
                raise ValueError("Prepared inputs already carry noise and conditions; do not mix raw inputs")
            given, null, noise = prompts.conditions, prompts.unconditional, prompts.noise
        else:
            given, null = self.conditions(prompts, negative_prompts, **conditioning_options)
            count = given["conditioning"]["encoder_hidden_states"].shape[0]
            height, width, channels = self.latent_shape
            noise = (jax.random.normal(key, (count, channels, height, width)).transpose(0, 2, 3, 1)
                     if latents is None else jnp.asarray(latents))
        count = noise.shape[0]
        shape = (count, *self.latent_shape)
        if noise.shape != shape:
            raise ValueError(f"Expected NHWC noise shape {shape}, got {noise.shape}")
        if sampler is not None:
            if self.task != "text-to-image" or image is not None or mask is not None or denoising_end is not None:
                raise ValueError("Explicit Dew solvers accept complete text-to-image trajectories only")
            from dew.sampling.sample import sample
            denoise = self.process.denoiser(self.model, self.params, given, unconditional=null)
            cfg = CFG(guidance) if isinstance(guidance, (int, float)) else guidance
            result = sample(denoise, noise, steps=steps, solver=sampler, guidance=cfg, key=key)
            return result if output_type == "latent" else self._finish(self.vae.decode(self.params["autoencoder"], result))
        if isinstance(guidance, CFG):
            raise ValueError("Use a float scale with the checkpoint scheduler, or supply a Dew solver")
        scale = jnp.asarray(1.0 if guidance is None else guidance, jnp.float32)
        scheduler = self.scheduler
        nchw = noise.transpose(0, 3, 1, 2)
        state = scheduler.set_timesteps(self.scheduler_state, num_inference_steps=steps, shape=nchw.shape)
        timetable = np.asarray(state.timesteps)
        start, end = 0, len(timetable)
        if denoising_end is not None:
            cutoff = round(self.process.schedule.T * (1 - denoising_end))
            end = int(np.count_nonzero(timetable >= cutoff))
        clean = masks = None
        if self.task != "text-to-image":
            strength = (1.0 if self.task == "inpainting" else 0.8) if strength is None else strength
            if image is None or (self.task == "inpainting") != (mask is not None):
                raise ValueError("img2img requires image only; inpainting requires image and mask")
            image_latents = isinstance(image, (np.ndarray, jax.Array)) and image.shape[-1] == self.vae.latent_channels
            if image_latents:
                if jnp.shape(image) != shape:
                    raise ValueError("Input image latents must match the requested latent geometry")
                clean = jnp.asarray(image).transpose(0, 3, 1, 2)
                if mask is not None:
                    raise ValueError("Inpainting image must contain pixels; pass masked_image_latents separately")
            else:
                pixels, _ = self.prepare_image(image)
                if pixels.shape != (count, *self.inputs.sample.shape):
                    raise ValueError("Image batch must match prompts and configured resolution")
                if (self.task == "image-to-image" or self.model.unet.config["in_channels"] == 4
                        or strength < 1 or denoising_start is not None):
                    clean = self.vae.encode(self.params["autoencoder"], pixels, key).transpose(0, 3, 1, 2)
            if denoising_start is not None:
                cutoff = round(self.process.schedule.T * (1 - denoising_start))
                start = int(np.count_nonzero(timetable >= cutoff))
                if clean is None:
                    raise ValueError("denoising_start requires image latents")
                nchw = clean
            elif self.task == "image-to-image" or self.encoder.xl:
                if not 0 < strength <= 1 or int(steps * strength) == 0:
                    raise ValueError("strength must select at least one step and be <= 1")
                start = steps - int(steps * strength)
                if self.task == "image-to-image" or strength < 1:
                    noise_times = (jnp.full((count,), start) if isinstance(scheduler, FlaxEulerDiscreteScheduler)
                                   else jnp.asarray(timetable[start:start + 1]).repeat(count))
                    nchw = scheduler.add_noise(state, clean, nchw, noise_times)
            if self.task == "inpainting":
                masked, masks = self.prepare_image(image, mask)
                masks = jax.image.resize(masks, (count, shape[1], shape[2], 1), method="nearest")
                if self.model.unet.config["in_channels"] == 9:
                    _, mask_key = jax.random.split(key)
                    encoded = (self.vae.encode(self.params["autoencoder"], masked, mask_key)
                               if masked_image_latents is None else jnp.asarray(masked_image_latents))
                    if encoded.shape != shape:
                        raise ValueError("masked_image_latents must match the requested latent geometry")
                    given["conditioning"].update(mask=masks, masked_image=encoded)
                    null["conditioning"].update(mask=masks, masked_image=encoded)
        elif image is not None or mask is not None or denoising_start is not None:
            raise ValueError("Load an image task for image inputs or denoising_start")
        if start >= end:
            raise ValueError("The selected denoising interval contains no steps")
        if denoising_start is None and (self.task == "text-to-image" or not self.encoder.xl or
                                       (self.task == "inpainting" and strength == 1)):
            nchw = nchw * state.init_noise_sigma
        conditions = jax.tree.map(lambda n, p: jnp.concatenate([n, p]), null, given)
        for index in range(start, end):
            t = state.timesteps[index]
            model_input = scheduler.scale_model_input(state, jnp.concatenate([nchw, nchw]), t)
            prediction = _predict(self.model, self.params, model_input.transpose(0, 2, 3, 1),
                                  jnp.broadcast_to(t, (2 * count,)), conditions).transpose(0, 3, 1, 2)
            negative, positive = jnp.split(prediction, 2)
            prediction = negative + scale * (positive - negative)
            nchw, state = scheduler.step(state, prediction, t, nchw, return_dict=False)
            if self.task == "inpainting" and self.model.unet.config["in_channels"] == 4:
                retained = clean
                if index + 1 < end:
                    next_times = (jnp.full((count,), index + 1) if isinstance(scheduler, FlaxEulerDiscreteScheduler)
                                  else jnp.broadcast_to(state.timesteps[index + 1], (count,)))
                    retained = scheduler.add_noise(state, clean, noise.transpose(0, 3, 1, 2), next_times)
                assert masks is not None
                blend = masks.transpose(0, 3, 1, 2)
                nchw = retained * (1 - blend) + nchw * blend
        result = nchw.transpose(0, 2, 3, 1)
        return result if output_type == "latent" else self._finish(self.vae.decode(self.params["autoencoder"], result))



def load_diffusers_pipeline(directory: str, *, revision: str | None = None, dtype=jnp.float32,
                  height: int | None = None, width: int | None = None,
                  local_files_only: bool = False, from_pt: bool = False,
                  task: str | None = None) -> DiffusersTextToImage:
    """Load supported SD/SDXL directories or a pinned Hub snapshot.

    The active CLIP towers, scheduler, VAE and optional checker stay attached.
    A refiner uses its single projected text tower and aesthetic time IDs.
    from_pt selects PyTorch component weights; saves use Flax parameter files.
    """
    path = Path(directory)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(directory, revision=revision, local_files_only=local_files_only,
                                      allow_patterns=["model_index.json", "*/config.json", "*/scheduler_config.json", "*/preprocessor_config.json",
                                                      "*/flax_model.msgpack", "*/diffusion_flax_model.msgpack",
                                                      "*/tokenizer*", "*/vocab.json", "*/merges.txt", "*/special_tokens_map.json",
                                                      *(["*/*.safetensors", "*/diffusion_pytorch_model.bin"] if from_pt else [])]))
    config = json.loads((path / "model_index.json").read_text())
    name = config["_class_name"].removeprefix("Flax")
    supported = {"StableDiffusionPipeline": "text-to-image", "StableDiffusionXLPipeline": "text-to-image",
                 "StableDiffusionImg2ImgPipeline": "image-to-image", "StableDiffusionInpaintPipeline": "inpainting",
                 "StableDiffusionXLImg2ImgPipeline": "image-to-image", "StableDiffusionXLInpaintPipeline": "inpainting"}
    if name not in supported:
        raise ValueError(f"Unsupported Diffusers pipeline: {name}")
    task = supported[name] if task is None else task
    xl = "StableDiffusionXL" in name
    if task not in supported.values():
        raise ValueError(f"Unsupported pipeline task: {name} / {task}")
    checker = (SafetyChecker.load(path, from_pt=from_pt, dtype=dtype)
               if config.get("safety_checker", [None, None])[0] is not None else None)
    names = tuple(key for key in (("text_encoder", "text_encoder_2") if xl else ("text_encoder",))
                  if config.get(key, [None, None])[0] is not None)
    if not names or (xl and "text_encoder_2" not in names):
        raise ValueError("SD needs its text encoder; SDXL needs the projected second text encoder")
    unet, unet_params = _load_model(FlaxUNet2DConditionModel, path / "unet", dtype=dtype, from_pt=from_pt)
    assert isinstance(unet, FlaxUNet2DConditionModel)
    unet = FlaxUNet2DConditionModel.from_config(
        {name: tuple(value) if isinstance(value, list) else value for name, value in dict(unet.config).items()}, dtype=dtype)
    assert isinstance(unet, FlaxUNet2DConditionModel)
    vae_model, vae_params = _load_model(FlaxAutoencoderKL, path / "vae", dtype=dtype, from_pt=from_pt)
    assert isinstance(vae_model, FlaxAutoencoderKL)
    expected = (4, 9) if task == "inpainting" else (vae_model.config["latent_channels"],)
    if unet.config["in_channels"] not in expected:
        raise ValueError(f"{task} requires UNet input channels in {expected}")
    vae = VAE(vae_model, vae_params, vae_model.config["scaling_factor"])
    size = unet.config["sample_size"] * vae.downscale_factor
    height, width = int(height or config.get("dew_height", size)), int(width or config.get("dew_width", size))
    if height % 8 or width % 8:
        raise ValueError("height and width must be divisible by 8")
    scheduler_types = {cls.__name__.removeprefix("Flax"): cls for cls in (
        FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler, FlaxEulerDiscreteScheduler)}
    scheduler_name = config["scheduler"][1].removeprefix("Flax")
    if scheduler_name not in scheduler_types:
        raise ValueError(f"Unsupported Flax scheduler: {scheduler_name}")
    scheduler, state = _load_scheduler(scheduler_types[scheduler_name], path / "scheduler")
    transforms = {"epsilon": EpsilonPredictionTransform, "v_prediction": VPredictionTransform}
    prediction = scheduler.config.prediction_type
    if isinstance(scheduler, FlaxLMSDiscreteScheduler):
        scheduler = SourceLMS(json.loads((path / "scheduler" / "scheduler_config.json").read_text()))
    if prediction not in transforms:
        raise ValueError(f"Unsupported pretrained prediction type: {prediction}")
    schedule = DiscreteNoiseScheduler(np.asarray(state.common.betas), p2_loss_weight_gamma=0)
    towers, tokenizers, configs, text_params = [], [], [], {}
    for tower_name in names:
        suffix = tower_name.removeprefix("text_encoder")
        folder = path / tower_name
        text_config = json.loads((folder / "config.json").read_text())
        tower = CLIPTextTransformer(**translate_config(text_config), dtype=dtype)
        if from_pt:
            from safetensors.numpy import load_file
            native = translate_clip_weights(load_file(folder / "model.safetensors"))
        else:
            official = serialization.msgpack_restore((folder / "flax_model.msgpack").read_bytes())
            native = {"text_model": _native_text(official),
                      **({"text_projection": official["text_projection"]} if "text_projection" in official else {})}
        text_params["text_encoder" + suffix] = native
        towers.append(tower)
        tokenizers.append(CLIPTokenizer.from_pretrained(path / ("tokenizer" + suffix), local_files_only=True))
        configs.append(text_config)
    encoder = Prompts(tuple(towers), tuple(tokenizers), tuple(configs), text_params, str(path), height, width,
                      names, xl, config.get("requires_aesthetics_score", False))
    variables = {"params": {"unet": unet_params}, "autoencoder": vae_params,
                 "encoders": {"conditioning": text_params}}
    if checker is not None:
        variables["encoders"]["safety_checker"] = checker.params
    output_name = next(k for k, v in supported.items() if v == task and ("StableDiffusionXL" in k) == xl)
    prefix = "" if xl and task != "text-to-image" else "Flax"
    config = {**config, "_class_name": prefix + output_name, "dew_height": height, "dew_width": width}
    return DiffusersTextToImage(model=Denoiser(unet), process=Process(schedule, transforms[prediction]()),
        inputs=InputSpec(Field("image", (height, width, 3)), {"conditioning": Condition(encoder)}),
        params=variables, autoencoder=vae, encoder=encoder, scheduler=scheduler, scheduler_state=state,
        config=config, task=task, safety_checker=checker)
