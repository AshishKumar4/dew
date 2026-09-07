"""SD / SDXL with official Diffusers Flax UNet, VAE and schedulers.

Diffusers 0.34.0 full Flax pipelines require removed Transformers Flax towers.
The runtime instead uses Dew's CLIP primitives and an explicit weight-layout
bridge. Orchestration follows those pipelines; reference tools run the full
pipelines in an isolated Transformers 4.49.0 environment. No runtime downgrade.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
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
from flax import linen as nn, serialization
from PIL import Image
from transformers import CLIPTokenizer

from dew.diffusion.process import Process
from dew.diffusion.schedules.discrete import DiscreteNoiseScheduler
from dew.diffusion.transforms import EpsilonPredictionTransform, VPredictionTransform
from dew.inputs import Condition, ConditionEncoder, Field, InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.nn.text_encoders import CLIPTextTransformer, translate_clip_weights, translate_config
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.base import Variables
from dew.sampling.guidance import CFG
from dew.sampling.pipelines import TextToImage


SchedulerState = DDIMSchedulerState | PNDMSchedulerState | LMSDiscreteSchedulerState | DPMSolverMultistepSchedulerState


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


@dataclass(eq=False)
class Prompts(ConditionEncoder[str]):
    towers: tuple[CLIPTextTransformer, ...]
    tokenizers: tuple[CLIPTokenizer, ...]
    configs: tuple[dict, ...]
    params: Variables
    checkpoint: str
    height: int
    width: int

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs):
        return load_pipeline(checkpoint, **kwargs).encoder

    @property
    def xl(self):
        return len(self.towers) == 2

    def tokenize(self, data: Sequence[str]):
        ids = [tokenizer(list(data), padding="max_length", max_length=tokenizer.model_max_length,
                         truncation=True, return_tensors="np").input_ids for tokenizer in self.tokenizers]
        return {"input_ids": np.stack(ids, axis=1) if self.xl else ids[0]}

    def encode(self, params, tokens):
        ids = tokens["input_ids"]
        outputs = []
        for i, tower in enumerate(self.towers):
            name = "text_encoder" if i == 0 else "text_encoder_2"
            output = tower.apply({"params": params[name]["text_model"]},
                                 ids[:, i] if self.xl else ids, method=_clip_features)
            assert isinstance(output, _CLIPFeatures)
            outputs.append(output)
        if not self.xl:
            return {"encoder_hidden_states": outputs[0].last}
        hidden = jnp.concatenate([output.penultimate for output in outputs], axis=-1)
        pooled = outputs[1].pooled @ params["text_encoder_2"]["text_projection"]["kernel"]
        return {"encoder_hidden_states": hidden, "added_cond_kwargs": {
            "text_embeds": pooled,
            "time_ids": jnp.broadcast_to(jnp.asarray(
                [self.height, self.width, 0, 0, self.height, self.width], hidden.dtype), (ids.shape[0], 6))}}

    def captions(self, tokens):
        ids = tokens["input_ids"][:, 0] if self.xl else tokens["input_ids"]
        return tuple(self.tokenizers[0].batch_decode(np.asarray(ids), skip_special_tokens=True))

    def to_json(self):
        return {"checkpoint": self.checkpoint, "height": self.height, "width": self.width}


class PretrainedObjective(DiffusionObjective):
    """Diffusion training initialized with the loaded denoiser variables."""
    def __init__(self, pipe, **kwargs):
        if pipe.task == "inpainting":
            raise ValueError("Inpainting Objective training requires a masked-image training input specification")
        self.loaded = pipe.params
        super().__init__(pipe.model, pipe.process, pipe.inputs, autoencoder=pipe.autoencoder, **kwargs)
        self.unconditional = pipe.conditions([""])[1]

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

    def conditions(self, prompts, negative_prompts=None):
        rows = [prompts] if isinstance(prompts, str) else list(prompts)
        params = self.params["encoders"]["conditioning"]
        given = self.encoder.encode(params, self.encoder.tokenize(rows))
        if negative_prompts is None and self.encoder.xl:
            null = {"encoder_hidden_states": jnp.zeros_like(given["encoder_hidden_states"]),
                    "added_cond_kwargs": {"text_embeds": jnp.zeros_like(given["added_cond_kwargs"]["text_embeds"]),
                                          "time_ids": given["added_cond_kwargs"]["time_ids"]}}
        else:
            negatives = ([""] * len(rows) if negative_prompts is None else
                         [negative_prompts] * len(rows) if isinstance(negative_prompts, str) else list(negative_prompts))
            if len(negatives) != len(rows):
                raise ValueError("negative_prompts must have one row per prompt")
            null = self.encoder.encode(params, self.encoder.tokenize(negatives))
        return {"conditioning": given}, {"conditioning": null}

    def save_pretrained(self, directory: str):
        """Write current bound parameters in the official Flax directory layout."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model_index.json").write_text(json.dumps(self.config, indent=2))
        self.model.unet.save_pretrained(destination / "unet", params=self.params["params"]["unet"])
        self.vae.model.save_pretrained(destination / "vae", params=self.params["autoencoder"])
        self.scheduler.save_pretrained(destination / "scheduler")
        for i, (config, tokenizer) in enumerate(zip(self.encoder.configs, self.encoder.tokenizers)):
            suffix = "" if i == 0 else "_2"
            folder = destination / ("text_encoder" + suffix)
            folder.mkdir(exist_ok=True)
            (folder / "config.json").write_text(json.dumps(config, indent=2))
            params = self.params["encoders"]["conditioning"]["text_encoder" + suffix]
            (folder / "flax_model.msgpack").write_bytes(serialization.to_bytes(_official_text(params)))
            tokenizer.save_pretrained(destination / ("tokenizer" + suffix))

    def prepare_image(self, image, mask=None):
        """Official SD PIL preprocessing, returned as NHWC [-1,1] pixels.

        Sizes round down to multiples of 32. Inpainting additionally returns a
        binary NHWC mask and zeros white-mask pixels in the conditioning image.
        This is the nine-channel SD contract, not a universal image mask.
        """
        images = [image] if isinstance(image, Image.Image) else list(image)
        pixels = []
        for item in images:
            width, height = (size - size % 32 for size in item.size)
            if not width or not height:
                raise ValueError("Image dimensions must be at least 32")
            resized = item.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            pixels.append(jnp.asarray(resized, jnp.float32) / 255 * 2 - 1)
        pixels = jnp.stack(pixels)
        if mask is None:
            return pixels, None
        masks = [mask] if isinstance(mask, Image.Image) else list(mask)
        if len(masks) != len(images):
            raise ValueError("One mask is required per image")
        values = []
        for item in masks:
            width, height = (size - size % 32 for size in item.size)
            values.append(jnp.asarray(item.resize((width, height)).convert("L"), jnp.float32)[..., None] / 255)
        masks = (jnp.stack(values) >= 0.5).astype(jnp.float32)
        if masks.shape[:3] != pixels.shape[:3]:
            raise ValueError("Image and mask sizes must match after preprocessing")
        return pixels * (masks < 0.5), masks

    def __call__(self, prompts, *, steps=50, guidance: CFG | float | None = 7.5, sampler=None, key,
                 negative_prompts=None, image=None, mask=None, strength=0.8, latents=None):
        """Sample NHWC [-1,1] images with the checkpoint scheduler by default.

        image/mask are PIL inputs for loaded SD img2img/inpainting tasks only.
        latents is NHWC initial noise. Explicit Dew solvers are text-only.
        SDXL uses both tokenizers, penultimate states, projected pooled states,
        resolution/crop time IDs and zero negatives when negatives are omitted.
        """
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        given, null = self.conditions(prompts, negative_prompts)
        count = given["conditioning"]["encoder_hidden_states"].shape[0]
        shape = (count, *self.latent_shape)
        # Draw in reference NCHW order before crossing Dew's NHWC seam.
        noise = jax.random.normal(key, (count, shape[-1], *shape[1:3])).transpose(0, 2, 3, 1) if latents is None else latents
        if noise.shape != shape:
            raise ValueError(f"Expected NHWC noise shape {shape}, got {noise.shape}")
        if sampler is not None:
            if self.task != "text-to-image" or image is not None or mask is not None:
                raise ValueError("Dew solvers accept text-to-image only")
            from dew.sampling.sample import sample
            denoise = self.process.denoiser(self.model, self.params, given, unconditional=null)
            cfg = CFG(guidance) if isinstance(guidance, (int, float)) else guidance
            result = sample(denoise, noise, steps=steps, solver=sampler, guidance=cfg, key=key)
            return jnp.clip(self.vae.decode(self.params["autoencoder"], result), -1, 1)
        if isinstance(guidance, CFG):
            raise ValueError("Use a float scale with the checkpoint scheduler, or supply a Dew solver")
        scale = 1.0 if guidance is None else float(guidance)
        scheduler = self.scheduler
        nchw = noise.transpose(0, 3, 1, 2)
        state = scheduler.set_timesteps(self.scheduler_state, num_inference_steps=steps, shape=nchw.shape)
        start = 0
        if self.task != "text-to-image":
            if image is None or (self.task == "inpainting") != (mask is not None):
                raise ValueError("img2img requires image only; inpainting requires image and mask")
            pixels, masks = self.prepare_image(image, mask)
            if pixels.shape != (count, *self.inputs.sample.shape):
                raise ValueError("Preprocessed image batch must match prompts and configured resolution")
            if self.task == "image-to-image":
                if not 0 < strength <= 1 or int(steps * strength) == 0:
                    raise ValueError("strength must select at least one step and be <= 1")
                start = steps - int(steps * strength)
                clean = self.vae.encode(self.params["autoencoder"], pixels, key).transpose(0, 3, 1, 2)
                time = state.timesteps[start:start + 1].repeat(count)
                nchw = scheduler.add_noise(self.scheduler_state, clean, nchw, time)
            else:
                _, mask_key = jax.random.split(key)
                encoded = self.vae.encode(self.params["autoencoder"], pixels, mask_key)
                masks = jax.image.resize(masks, (*encoded.shape[:3], 1), method="nearest")
                given["conditioning"].update(mask=masks, masked_image=encoded)
                null["conditioning"].update(mask=masks, masked_image=encoded)
        elif image is not None or mask is not None:
            raise ValueError("Load the image-to-image or inpainting task for image inputs")
        nchw = nchw * state.init_noise_sigma
        conditions = jax.tree.map(lambda n, p: jnp.concatenate([n, p]), null, given)

        def step(index, carry):
            x, current = carry
            t = current.timesteps[index].astype(jnp.int32)
            model_input = scheduler.scale_model_input(current, jnp.concatenate([x, x]), t)
            output = self.model.apply(self.params, model_input.transpose(0, 2, 3, 1),
                                      jnp.broadcast_to(t, (2 * count,)), **conditions)
            assert isinstance(output, jax.Array)
            prediction = output.transpose(0, 3, 1, 2)
            negative, positive = jnp.split(prediction, 2)
            prediction = negative + scale * (positive - negative)
            return scheduler.step(current, prediction, t, x, return_dict=False)

        result, _ = jax.lax.fori_loop(start, steps, step, (nchw, state))
        return jnp.clip(self.vae.decode(self.params["autoencoder"], result.transpose(0, 2, 3, 1)), -1, 1)


def load_pipeline(directory: str, *, revision: str | None = None, dtype=jnp.float32,
                  height: int | None = None, width: int | None = None,
                  local_files_only: bool = False, from_pt: bool = False,
                  task: str | None = None) -> DiffusersTextToImage:
    """Load supported SD/SDXL directories or a pinned Hub snapshot.

    Flax msgpack weights are the default. With from_pt=True, Diffusers 0.34
    loads UNet/VAE diffusion_pytorch_model.bin files and Dew reads CLIP
    model.safetensors. SDXL refiner/image tasks and pipelines carrying a safety
    checker require additional native bridges and are rejected, never stripped.
    """
    path = Path(directory)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(directory, revision=revision, local_files_only=local_files_only,
                                      allow_patterns=["model_index.json", "*/config.json", "*/scheduler_config.json",
                                                      "*/flax_model.msgpack", "*/diffusion_flax_model.msgpack",
                                                      "*/tokenizer*", "*/vocab.json", "*/merges.txt", "*/special_tokens_map.json",
                                                      *(["*/*.safetensors", "*/diffusion_pytorch_model.bin"] if from_pt else [])]))
    config = json.loads((path / "model_index.json").read_text())
    name = config["_class_name"].removeprefix("Flax")
    supported = {"StableDiffusionPipeline": "text-to-image", "StableDiffusionXLPipeline": "text-to-image",
                 "StableDiffusionImg2ImgPipeline": "image-to-image", "StableDiffusionInpaintPipeline": "inpainting"}
    if name not in supported:
        raise ValueError(f"Unsupported Diffusers pipeline: {name}")
    task = supported[name] if task is None else task
    xl = name == "StableDiffusionXLPipeline"
    if task not in supported.values() or (xl and task != "text-to-image"):
        raise ValueError(f"Unsupported pipeline task: {name} / {task}")
    if config.get("safety_checker", [None, None])[0] is not None:
        raise ValueError("This checkpoint requires a native safety-checker bridge; its checker cannot be silently removed")
    if xl and (config.get("text_encoder", [None, None])[0] is None or
               config.get("text_encoder_2", [None, None])[0] is None):
        raise ValueError("SDXL base requires both text encoders; refiner is not supported")
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(path / "unet", dtype=dtype, from_pt=from_pt)
    vae_model, vae_params = FlaxAutoencoderKL.from_pretrained(path / "vae", dtype=dtype, from_pt=from_pt)
    expected = 9 if task == "inpainting" else vae_model.config.latent_channels
    if unet.config.in_channels != expected:
        raise ValueError(f"{task} requires a {expected}-channel UNet checkpoint")
    vae = VAE(vae_model, vae_params, vae_model.config.scaling_factor)
    size = unet.config.sample_size * vae.downscale_factor
    height, width = height or size, width or size
    if height % 8 or width % 8:
        raise ValueError("height and width must be divisible by 8")
    scheduler_types = {cls.__name__.removeprefix("Flax"): cls for cls in (
        FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler)}
    scheduler_name = config["scheduler"][1].removeprefix("Flax")
    if scheduler_name not in scheduler_types:
        raise ValueError(f"Unsupported Flax scheduler: {scheduler_name}")
    scheduler, state = _load_scheduler(scheduler_types[scheduler_name], path / "scheduler")
    transforms = {"epsilon": EpsilonPredictionTransform, "v_prediction": VPredictionTransform}
    prediction = scheduler.config.prediction_type
    if prediction not in transforms:
        raise ValueError(f"Unsupported pretrained prediction type: {prediction}")
    schedule = DiscreteNoiseScheduler(np.asarray(state.common.betas), p2_loss_weight_gamma=0)
    towers, tokenizers, configs, text_params = [], [], [], {}
    for suffix in (("", "_2") if xl else ("",)):
        folder = path / ("text_encoder" + suffix)
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
    encoder = Prompts(tuple(towers), tuple(tokenizers), tuple(configs), text_params, str(path), height, width)
    variables = {"params": {"unet": unet_params}, "autoencoder": vae_params,
                 "encoders": {"conditioning": text_params}}
    output_name = "StableDiffusionXLPipeline" if xl else next(k for k, v in supported.items() if v == task)
    config = {**config, "_class_name": "Flax" + output_name}
    return DiffusersTextToImage(model=Denoiser(unet), process=Process(schedule, transforms[prediction]()),
        inputs=InputSpec(Field("image", (height, width, 3)), {"conditioning": Condition(encoder)}),
        params=variables, autoencoder=vae, encoder=encoder, scheduler=scheduler, scheduler_state=state, config=config, task=task)
