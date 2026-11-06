"""Full learned-model trajectories with the diffusers-pipelines cohort's source schedulers: the saved SD
pipeline's own Flax UNet and text tower supply the predictions, "dog" unconditioned and "cat" at guidance 3,
and Diffusers' PyTorch schedulers the updates, four steps from a given noise (`walk`), in float32 or in
float64: the Flax pipeline at float64 as tools/native_diffusion_worker.py builds it (`flax_precision.float64`
over diffusers_pipeline_reference's WIDENED), each scheduler with its module's float32 read as float64
(`diffusers_wan_reference.float64_scheduler`). The timetable includes PNDM's PRK/PLMS warmup and LMS's
sigma-zero interval. tools/native_diffusion_worker.py's `native-grid-<name>` sites run it in that cohort's
environment (tools/reference_releases.py); tools/check_native_diffusion.py's `check_grids` holds Dew's walk
over each scheduler's own grid to it.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np

SCHEDULERS = {
    "pndm-prk": ("PNDMScheduler", {"skip_prk_steps": False}),
    "pndm-plms": ("PNDMScheduler", {"skip_prk_steps": True}),
    "lms": ("LMSDiscreteScheduler", {}),
    "lms-v": ("LMSDiscreteScheduler", {"prediction_type": "v_prediction"}),
    "lms-karras": ("LMSDiscreteScheduler", {"use_karras_sigmas": True}),
    "euler": ("EulerDiscreteScheduler", {}),
    "dpm": ("DPMSolverMultistepScheduler", {}),
}
"""Each grid's Diffusers scheduler class and the options it is made with from the SD pipeline's file."""
STEPS = 4


def scheduler(name: str, base: Path):
    """Grid `name`'s Diffusers scheduler, made from `base`'s scheduler file with its options."""
    import diffusers

    kind, options = SCHEDULERS[name]
    config = json.loads((Path(base) / "scheduler" / "scheduler_config.json").read_text())
    return getattr(diffusers, kind).from_config(config, **options)


def guided(pipe, params):
    """The pipeline's UNet over the prompts' states, "dog" unconditioned and "cat" at guidance 3,
    compiled."""
    import jax
    import jax.numpy as jnp

    context = pipe.text_encoder(pipe.prepare_inputs(["dog", "cat"]), params=params["text_encoder"])[0]

    @jax.jit
    def predict(sample, time):
        output = pipe.unet.apply({"params": params["unet"]}, sample, jnp.broadcast_to(time, (2,)),
                                encoder_hidden_states=context).sample
        negative, positive = jnp.split(output, 2)
        return negative + 3.0 * (positive - negative)

    return predict


def walk(name: str, base: Path, pipeline, noise: np.ndarray, wide: bool) -> np.ndarray:
    """Grid `name`'s walk of `base`'s SD pipeline (`pipeline`: its Flax pipeline and parameters, at float64
    where `wide`) from `noise` (channels last), its last latents channels last."""
    import diffusers
    import jax.numpy as jnp
    import torch

    module = sys.modules[getattr(diffusers, SCHEDULERS[name][0]).__module__]
    if wide:
        from diffusers_wan_reference import float64_scheduler

        scope = float64_scheduler(module)
    else:
        scope = contextlib.nullcontext()
    predict = guided(*pipeline)
    dtype = torch.float64 if wide else torch.float32
    with scope:
        walked = scheduler(name, base)
        walked.set_timesteps(STEPS)
        value = torch.from_numpy(np.ascontiguousarray(noise.transpose(0, 3, 1, 2))).to(dtype)
        value = value * walked.init_noise_sigma
        for time in walked.timesteps:
            model_input = walked.scale_model_input(torch.cat([value, value]), time)
            prediction = predict(jnp.asarray(model_input.numpy()), jnp.asarray(time.numpy()))
            value = walked.step(torch.from_numpy(np.array(prediction)), time, value).prev_sample
    return value.permute(0, 2, 3, 1).numpy()


__all__ = ["SCHEDULERS", "STEPS", "guided", "scheduler", "walk"]
