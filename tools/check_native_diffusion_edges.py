"""Native edge checks against tools/native_diffusion_edges_reference.py oracles: the source's
tables and timesteps exactly, and every walk as near float64 as the source's float32
(tests/reference_error.py)."""
import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.diffusion.schedules.source import SourceSchedule
from dew.interop.diffusion import component_tensors, translate_unet_weights, unet_fields
from dew.diffusion.process import DenoisingCondition
from dew.nn.backbones.unet_condition import UNet2DCondition
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
        for name, actual in (("output", output), ("vjp", gradient)):
            errors[name] = float(np.max(np.abs(actual - ref[f"{case}.{name}"])))
            assert_as_exact_as_the_reference(actual, ref[f"{case}.{name}"], ref[f"{case}.{name}.fp64"],
                                             f"{case} {name}")
    else:
        ref = np.load(directory / "schedulers.npz")
        policy = SourceSchedule.from_config(json.loads(str(ref[case + ".config"])))
        np.testing.assert_array_equal(policy.betas, ref[case + ".betas"])
        process, times = policy.sampling(4)
        model_times = np.asarray(process.sampler_schedule.model_time(times[:-1]))
        expected_times = ref[case + ".times"]
        if case == "pndm-linspace":
            expected_times = np.concatenate([expected_times[:1], expected_times[2:]])
        np.testing.assert_array_equal(model_times, expected_times)
        initial = jnp.asarray([[10.0, -8.0, 0.3]]) * process.sampler_schedule.prior_scale()
        denoise = process.denoiser(Oracle(), {}, {})
        result = sample(denoise, initial, solver=policy.solver, times=times,
                        key=jax.random.PRNGKey(0), final_denoise=False)
        errors["trajectory"] = float(np.max(np.abs(result - ref[case + ".final"])))
        assert_as_exact_as_the_reference(result, ref[case + ".final"], ref[case + ".final.fp64"], f"{case} walk")
    print(json.dumps({"case": case, "errors": errors}))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from reference_error import assert_as_exact_as_the_reference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("case")
    args = parser.parse_args()
    check(args.directory, args.case)
