"""Full learned-model trajectories with Diffusers 0.34.0 source schedulers.

The official Flax UNet/text towers supply identical predictions to both sides;
the official Torch schedulers supply the reference updates. This isolates solver
semantics from known Flax-versus-Torch UNet arithmetic differences. Reference
host transfers occur only in this tool, never in Dew's denoising loop.

Run in the isolated Transformers 4.49.0 reference environment:
  python tools/diffusers_scheduler_pipeline_reference.py BASE_SD OUTPUT.npz
The source timetable includes PNDM's PRK/PLMS warmup and LMS's sigma-zero interval.
"""
import argparse
import json

import diffusers
import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
from diffusers import FlaxStableDiffusionPipeline, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler


def build(base, destination):
    if (diffusers.__version__, transformers.__version__) != ("0.34.0", "4.49.0"):
        raise RuntimeError("Use the pinned isolated reference environment")
    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
        base, local_files_only=True, safety_checker=None, feature_extractor=None)
    ids = pipe.prepare_inputs(["dog", "cat"])
    context = pipe.text_encoder(ids, params=params["text_encoder"])[0]

    @jax.jit
    def predict(sample, time):
        output = pipe.unet.apply({"params": params["unet"]}, sample, jnp.broadcast_to(time, (2,)),
                                encoder_hidden_states=context).sample
        negative, positive = jnp.split(output, 2)
        return negative + 3.0 * (positive - negative)

    noise = np.asarray(jax.random.normal(jax.random.PRNGKey(17), (1, 4, 8, 8)))
    arrays = {"noise": noise.transpose(0, 2, 3, 1)}
    for name, cls, options in (
        ("pndm-prk", PNDMScheduler, {"skip_prk_steps": False}),
        ("pndm-plms", PNDMScheduler, {"skip_prk_steps": True}),
        ("lms", LMSDiscreteScheduler, {}),
        ("lms-v", LMSDiscreteScheduler, {"prediction_type": "v_prediction"}),
        ("lms-karras", LMSDiscreteScheduler, {"use_karras_sigmas": True}),
        ("euler", EulerDiscreteScheduler, {}),
    ):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("destination")
    args = parser.parse_args()
    build(args.base, args.destination)
