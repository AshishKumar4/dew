"""Native edge checks against diffusers, live: the source's timesteps exactly and its noise tables as
near float64 as the source's float32 (tools/native_diffusion_edges_reference.py records both), and
every walk, and the tiny UNet's output and its gradients, on par with diffusers' float32 run over
drawn inputs, both measured from diffusers' run in float64 (check 1, tests/reference_error.py)."""
import argparse
import copy
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.diffusion.process import DenoisingCondition
from dew.diffusion.schedules.source import SourceSchedule
from dew.interop.diffusion import component_tensors, translate_unet_weights, unet_fields
from dew.nn.backbones.unet_condition import UNet2DCondition
from dew.sampling.sample import sample

ROWS = 256
"""Each walk's independent rows, so its distance from float64 is an RMS over many roundings."""
STEPS = 4
TIME = 13.0
"""The UNet's timestep, as the recorded reference took it."""
WIDTHS = {"norm": 12, "odd": 10}
"""Each UNet case's latent width: 12 the group norm's whole groups, 10 an odd width."""
SEEDS = {"ddim-linspace": 8, "pndm-linspace": 8, "ddim-clip": 8, "euler-zero-snr": 8, "dpm-karras": 8,
         "norm": 8, "odd": 8}
"""Each case's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on armada."""


class Oracle(nn.Module):
    def __call__(self, value, time):
        return jnp.sin(value) * 0.07 + time[:, None] * 0.001


def oracle_walk(scheduler, initial, dtype):
    """diffusers' walk of `initial` [ROWS, 3] under the oracle model, at `dtype`."""
    import torch

    scheduler.set_timesteps(STEPS)
    x = torch.as_tensor(np.asarray(initial)).to(dtype) * scheduler.init_noise_sigma
    for time in scheduler.timesteps:
        model_input = scheduler.scale_model_input(x, time)
        prediction = torch.sin(model_input) * 0.07 + time.to(dtype) * 0.001
        x = scheduler.step(prediction, time, x).prev_sample
    return x.numpy()


def check_schedule(ref, case):
    import diffusers

    from diffusers_wan_reference import float64_scheduler

    config = json.loads(str(ref[case + ".config"]))
    policy = SourceSchedule.from_config(config)
    # The table is a function of the config alone, with no draw to vary: one comparison holds it.
    assert_as_exact_as_the_reference(policy.betas, ref[case + ".betas"], ref[case + ".betas.fp64"], f"{case} betas")
    process, times = policy.sampling(STEPS)
    model_times = np.asarray(process.sampler_schedule.model_time(times[:-1]))
    expected_times = ref[case + ".times"]
    if case == "pndm-linspace":
        expected_times = np.concatenate([expected_times[:1], expected_times[2:]])
    np.testing.assert_array_equal(model_times, expected_times)
    denoise = process.denoiser(Oracle(), {}, {})
    scale = process.sampler_schedule.prior_scale()
    source = getattr(diffusers, config["_class_name"])

    def draw(seed):
        rng = np.random.default_rng([0xED6E, seed])
        return ((rng.standard_normal((ROWS, 3)) * 8).astype(np.float32),)

    def dew(initial):
        return np.asarray(sample(denoise, jnp.asarray(initial) * scale, solver=policy.solver, times=times,
                                 key=jax.random.PRNGKey(0), final_denoise=False))

    def reference(initial):
        import torch

        return oracle_walk(source.from_config(config), initial, torch.float32)

    def truth(initial):
        import torch

        with float64_scheduler(sys.modules[source.__module__]):
            return oracle_walk(source.from_config(config), initial, torch.float64)

    assert_as_exact(dew, reference, draw, f"{case} walk", seeds=SEEDS[case], truth_fn=truth)


def check_unet(directory, case):
    import torch
    from diffusers import UNet2DConditionModel

    from diffusers_wan_reference import float64

    model = UNet2DCondition(**unet_fields(json.loads((directory / "unet/config.json").read_text()),
                                          attention_impl="xla"))
    params, _ = translate_unet_weights(component_tensors(directory, "unet"), model)
    source = UNet2DConditionModel.from_pretrained(directory / "unet").eval()
    wide = copy.deepcopy(source).double()
    width = WIDTHS[case]

    def draw(seed):
        """The latents [1, 8, width, 4] (Dew's channels-last), the prompt states and the cotangent."""
        rng = np.random.default_rng([0x0E1, seed])
        return tuple(rng.standard_normal(shape).astype(np.float32)
                     for shape in ((1, 8, width, 4), (1, 8, 8), (1, 8, width, 4)))

    @jax.jit
    def predict(value, context):
        return model.apply({"params": params}, value, jnp.full((1,), TIME, value.dtype),
                           conditioning=DenoisingCondition(context))

    def torch_arm(unet):
        def run(value, context):
            time = torch.full((1,), TIME, dtype=value.dtype)
            output = unet(value.permute(0, 3, 1, 2), time, encoder_hidden_states=context).sample
            return output.permute(0, 2, 3, 1)
        return torch_with_gradients(run)

    reference = torch_arm(source)

    def truth(*inputs):
        with float64():
            return torch_arm(wide)(*inputs)

    assert_as_exact(with_gradients(predict), reference, draw, f"{case} output and gradients",
                    seeds=SEEDS[case], truth_fn=truth)


def check(directory, case):
    directory = Path(directory)
    if case in WIDTHS:
        check_unet(directory, case)
    else:
        check_schedule(np.load(directory / "schedulers.npz"), case)
    print(json.dumps({"case": case, "held": True}))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from reference_error import (
        assert_as_exact,
        assert_as_exact_as_the_reference,
        torch_with_gradients,
        with_gradients,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("case")
    args = parser.parse_args()
    check(args.directory, args.case)
