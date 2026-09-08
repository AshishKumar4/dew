"""Native edge checks against tools/native_diffusion_edges_reference.py oracles."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.diffusion.schedules.source import SourceSchedule
from dew.interop.diffusion import component_tensors, translate_unet_weights, unet_fields
from dew.nn.backbones.unet_condition import DenoisingCondition, UNet2DCondition
from dew.sampling.sample import sample


class Oracle(nn.Module):
    def __call__(self, value, time):
        return jnp.sin(value) * 0.07 + time[:, None] * 0.001


def check(directory, case):
    directory = Path(directory)
    errors = {}
    if case in ("norm", "odd"):
        ref = np.load(directory / "unet_reference.npz")
        model = UNet2DCondition(**unet_fields(json.loads((directory / "unet/config.json").read_text()), attention_impl="xla"))
        params, _ = translate_unet_weights(component_tensors(directory, "unet"), model)
        given = DenoisingCondition(jnp.asarray(ref[case + ".context"]))
        def predict(value):
            return model.apply({"params": params}, value, jnp.asarray(ref[case + ".time"]), conditioning=given)
        value = jnp.asarray(ref[case + ".input"])
        output = jax.jit(predict)(value)
        gradient = jax.jit(jax.grad(lambda x: jnp.sum(predict(x) * ref[case + ".probe"])))(value)
        for name, actual, expected in (("prediction", output, ref[case + ".output"]),
                                       ("vjp", gradient, ref[case + ".vjp"])):
            errors[name] = float(np.max(np.abs(actual - expected)))
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    else:
        ref = np.load(directory / "schedulers.npz")
        policy = SourceSchedule.from_config(json.loads(str(ref[case + ".config"])))
        errors["betas"] = float(np.max(np.abs(policy.betas - ref[case + ".betas"])))
        np.testing.assert_allclose(policy.betas, ref[case + ".betas"], atol=2e-6, rtol=2e-6)
        process, times = policy.sampling(4)
        model_times = np.asarray(process.sampler_schedule.model_time(times[:-1]))
        expected_times = ref[case + ".times"]
        if case == "pndm-linspace":
            expected_times = np.concatenate([expected_times[:1], expected_times[2:]])
        errors["times"] = float(np.max(np.abs(model_times - expected_times)))
        np.testing.assert_allclose(model_times, expected_times, atol=1e-5, rtol=1e-5)
        initial = jnp.asarray([[10.0, -8.0, 0.3]]) * process.sampler_schedule.prior_scale()
        denoise = process.denoiser(Oracle(), {}, {})
        result = sample(denoise, initial, solver=policy.solver(), times=times,
                        key=jax.random.PRNGKey(0), final_denoise=False)
        errors["trajectory"] = float(np.max(np.abs(result - ref[case + ".final"])))
        np.testing.assert_allclose(result, ref[case + ".final"], atol=1e-4, rtol=1e-4)
    print(json.dumps({"case": case, "errors": errors}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("case")
    args = parser.parse_args()
    check(args.directory, args.case)
