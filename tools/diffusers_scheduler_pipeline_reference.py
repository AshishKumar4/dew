"""Full learned-model trajectories with the diffusers-pipelines cohort's source schedulers.

The official Flax UNet/text towers supply identical predictions to both sides;
the official Torch schedulers supply the reference updates. This isolates solver
semantics from known Flax-versus-Torch UNet arithmetic differences. Reference
host transfers occur only in this tool, never in Dew's denoising loop.

Run in the isolated reference environment with that cohort's releases (tools/reference_releases.py):
  python tools/diffusers_scheduler_pipeline_reference.py BASE_SD OUTPUT.npz
The source timetable includes PNDM's PRK/PLMS warmup and LMS's sigma-zero interval.

`--truth BASE_SD OUTPUT.npz` adds each walk's float64 latents (`{name}.latents.fp64`) to OUTPUT, the
truth its float32 walk is measured from: the Flax pipeline at float64 as
`diffusers_pipeline_reference.py truth` builds it, each scheduler with its module's float32 read as
float64 (`diffusers_wan_reference.float64_scheduler`), from the recorded noise.
"""
import argparse
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch
from diffusers import FlaxStableDiffusionPipeline, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler

from reference_releases import require


SCHEDULERS = (
    ("pndm-prk", PNDMScheduler, {"skip_prk_steps": False}),
    ("pndm-plms", PNDMScheduler, {"skip_prk_steps": True}),
    ("lms", LMSDiscreteScheduler, {}),
    ("lms-v", LMSDiscreteScheduler, {"prediction_type": "v_prediction"}),
    ("lms-karras", LMSDiscreteScheduler, {"use_karras_sigmas": True}),
    ("euler", EulerDiscreteScheduler, {}),
    ("dpm", DPMSolverMultistepScheduler, {}),
)


def guided(pipe, params):
    """The pipeline's UNet over the prompts' states, "dog" unconditioned and "cat" at guidance 3,
    compiled."""
    context = pipe.text_encoder(pipe.prepare_inputs(["dog", "cat"]), params=params["text_encoder"])[0]

    @jax.jit
    def predict(sample, time):
        output = pipe.unet.apply({"params": params["unet"]}, sample, jnp.broadcast_to(time, (2,)),
                                encoder_hidden_states=context).sample
        negative, positive = jnp.split(output, 2)
        return negative + 3.0 * (positive - negative)

    return predict


def build(base, destination):
    require("diffusers-pipelines")
    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
        base, local_files_only=True, safety_checker=None, feature_extractor=None)
    predict = guided(pipe, params)
    noise = np.asarray(jax.random.normal(jax.random.PRNGKey(17), (1, 4, 8, 8)))
    arrays = {"noise": noise.transpose(0, 2, 3, 1)}
    for name, cls, options in SCHEDULERS:
        scheduler = cls.from_config(dict(pipe.scheduler.config), **options)
        scheduler.set_timesteps(4)
        value = torch.from_numpy(noise.copy()) * scheduler.init_noise_sigma
        trajectory = []
        for time in scheduler.timesteps:
            model_input = scheduler.scale_model_input(torch.cat([value, value]), time)
            prediction = predict(jnp.asarray(model_input.numpy()), jnp.asarray(time.numpy()))
            value = scheduler.step(torch.from_numpy(np.array(prediction)), time, value).prev_sample
            trajectory.append(value.numpy().copy())
        arrays[name + ".latents"] = value.permute(0, 2, 3, 1).numpy()
        arrays[name + ".trajectory"] = np.stack(trajectory)
        arrays[name + ".timesteps"] = scheduler.timesteps.numpy()
        arrays[name + ".config"] = np.asarray(json.dumps(dict(scheduler.config)))
        print(name, "model calls", len(trajectory), "latent sum", float(value.sum()))
    np.savez_compressed(destination, **arrays)


def truth(base, destination):
    """Each walk's float64 latents (see the module docstring)."""
    from unittest.mock import patch

    from diffusers import FlaxDDIMScheduler
    from diffusers_pipeline_reference import Float64Numpy, widened_modules
    from diffusers_wan_reference import float64_scheduler

    require("diffusers-pipelines")
    jax.config.update("jax_enable_x64", True)
    torch.set_num_threads(2)
    with np.load(destination) as stored:
        arrays = dict(stored)
    patches = [patch.object(module, "jnp", Float64Numpy()) for module in widened_modules()]
    for item in patches:
        item.start()
    try:
        pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
            base, local_files_only=True, safety_checker=None, feature_extractor=None, dtype=jnp.float64)
        pipe.text_encoder = type(pipe.text_encoder).from_pretrained(f"{base}/text_encoder", dtype=jnp.float64)
        pipe.scheduler = FlaxDDIMScheduler.from_pretrained(base, subfolder="scheduler", dtype=jnp.float64)[0]
        params = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64)
                              if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating) else x, params)
        predict = guided(pipe, params)
        noise = arrays["noise"].transpose(0, 3, 1, 2).astype(np.float64)
        for name, cls, options in SCHEDULERS:
            with float64_scheduler(sys.modules[cls.__module__]):
                scheduler = cls.from_config(dict(pipe.scheduler.config), **options)
                scheduler.set_timesteps(4)
                value = torch.from_numpy(noise.copy()) * scheduler.init_noise_sigma
                for time in scheduler.timesteps:
                    model_input = scheduler.scale_model_input(torch.cat([value, value]), time)
                    prediction = predict(jnp.asarray(model_input.numpy()), jnp.asarray(time.numpy(), jnp.float64))
                    value = scheduler.step(torch.from_numpy(np.array(prediction)), time, value).prev_sample
            arrays[name + ".latents.fp64"] = value.permute(0, 2, 3, 1).numpy()
            gap = float(np.max(np.abs(arrays[name + ".latents.fp64"] - arrays[name + ".latents"])))
            print(json.dumps({"scheduler": name, "dtype": str(value.dtype), "float32_from_float64": gap}))
    finally:
        for item in patches:
            item.stop()
    np.savez_compressed(destination, **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("destination")
    parser.add_argument("--truth", action="store_true", help="add each walk's float64 latents to DESTINATION")
    args = parser.parse_args()
    (truth if args.truth else build)(args.base, args.destination)
