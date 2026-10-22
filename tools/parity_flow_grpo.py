#!/usr/bin/env python3
"""Record the Flow-GRPO transition of the pinned author code over check 1's seeds.

Run with torch and diffusers installed. Downloads one Python source file from
GitHub; no model, tokenizer, or weights are loaded. The KL oracle uses
torch.distributions on the actual conditional transition variance. The author
code's float32 run's evidence from its float64 run, every float32 pin widened,
is kept with that run's sketch (tools/recorded_references.py), each seed's
inputs drawn again by tests/test_flow_grpo.py (`transition_inputs`):

    PYTHONPATH=src:tools python tools/parity_flow_grpo.py
"""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

COMMIT = "879042cf5707f8b90daa98d147d7deac2317c5da"
SOURCE = (f"https://raw.githubusercontent.com/yifan123/flow_grpo/{COMMIT}/"
          "flow_grpo/diffusers_patch/sd3_sde_with_logprob.py")
RECORDED = Path(__file__).resolve().parents[1] / "tests/fixtures/rl/recorded_flow_transition.npz"
SIGMAS = (1.0, 0.7, 0.4, 0.1)
"""The fixture's schedule: each of its rows takes one of these points' transitions to the next."""
REPEATS = 16
"""The recorded draws' rows per point: a draw holds each transition this many times."""
NOISE_LEVEL = 0.7
REFERENCE_SHIFT = (0.1, -0.2)
"""The frozen reference's mean, the policy's moved by this, for the conditional KL."""


def transition_inputs(seed: int) -> dict[str, np.ndarray]:
    """A seed's states, velocities and actions at the fixture's laws, REPEATS rows a point of SIGMAS,
    the same on every run: check 1's draw for the recorded reference (tests/test_flow_grpo.py draws
    them again)."""
    rng = np.random.default_rng([112, seed])
    rows = REPEATS * (len(SIGMAS))
    x = rng.normal(0, 0.2, (rows, 2)).astype(np.float32)
    velocity = rng.normal(0, 0.2, (rows, 2)).astype(np.float32)
    action = (x - 0.15 * velocity + rng.normal(0, 0.1, (rows, 2))).astype(np.float32)
    return {"x": x, "velocity": velocity, "action": action}


def fetched_reference():
    """The pinned author module, fetched and run as published."""
    with tempfile.TemporaryDirectory(prefix="dew-flow-reference-") as directory:
        path = Path(directory) / "reference.py"
        with urlopen(SOURCE, timeout=30) as response:
            path.write_bytes(response.read())
        spec = importlib.util.spec_from_file_location("flow_reference", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the pinned Flow-GRPO reference")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    return reference


def author_transition(reference, inputs: dict, dtype) -> dict[str, np.ndarray]:
    """The author code's transition on `inputs` in the torch `dtype`: its mean, variance, mean log
    density and that density's velocity gradient, and its KL to the reference shifted by
    REFERENCE_SHIFT, by the names tests/test_flow_grpo.py reads."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    scheduler.set_timesteps(sigmas=list(SIGMAS))
    # The schedule as Dew's transition takes it, SIGMAS at float32 and the final zero, in `dtype`: the
    # scheduler's own round trip through its timesteps moves a sigma by an ulp, a different input. From
    # NumPy, since the float64 run's `torch.tensor` is widened and would hold SIGMAS unrounded.
    scheduler.sigmas = torch.from_numpy(np.asarray((*SIGMAS, 0.0), np.float32)).to(dtype)
    timesteps = scheduler.timesteps.repeat(REPEATS)
    v = torch.tensor(inputs["velocity"], dtype=dtype, requires_grad=True)
    _, log_prob, mean, coefficient = reference.sde_step_with_logprob(
        scheduler, v, timesteps, torch.tensor(inputs["x"], dtype=dtype), noise_level=NOISE_LEVEL,
        prev_sample=torch.tensor(inputs["action"], dtype=dtype))
    log_prob.sum().backward()
    t = scheduler.sigmas[:-1].repeat(REPEATS)
    s = scheduler.sigmas[1:].repeat(REPEATS)
    variance = coefficient[:, 0].square() * (t - s)
    std = variance.sqrt()[:, None]
    shifted = mean.detach() + torch.tensor([REFERENCE_SHIFT], dtype=dtype)
    policy = torch.distributions.Independent(torch.distributions.Normal(mean.detach(), std), 1)
    frozen = torch.distributions.Independent(torch.distributions.Normal(shifted, std), 1)
    kl = torch.distributions.kl_divergence(policy, frozen)
    return {"mean": mean.detach().numpy(), "variance": variance.detach().numpy(),
            "mean_log_prob": log_prob.detach().numpy(), "velocity_grad": v.grad.numpy(),
            "conditional_kl": kl.detach().numpy()}


def dew_transition(inputs: dict) -> dict:
    """Dew's transition on a draw (`transition_inputs`), by `author_transition`'s names: the count the
    recording plans on, and tests/test_flow_grpo.py's arm and, in float64, its truth."""
    import jax.numpy as jnp
    from reference_error import with_gradients

    from dew.sampling.flow import flow_transition

    dtype = jnp.asarray(inputs["x"]).dtype
    t = jnp.asarray(np.tile(np.asarray(SIGMAS, np.float32), REPEATS), dtype)
    s = jnp.asarray(np.tile(np.asarray((*SIGMAS[1:], 0.0), np.float32), REPEATS), dtype)
    dimensions = inputs["x"].shape[1]

    def density(velocity):
        transition = flow_transition(inputs["x"], velocity, t, s, noise_level=NOISE_LEVEL)
        return transition.log_prob(inputs["action"]) / dimensions

    transition = flow_transition(inputs["x"], inputs["velocity"], t, s, noise_level=NOISE_LEVEL)
    # Each row's density pulls back one, exact at every precision, through the gate's pullback.
    ones = jnp.ones(inputs["velocity"].shape[0], inputs["velocity"].dtype)
    _, gradient = with_gradients(density)(inputs["velocity"], ones)
    shifted = transition.mean + jnp.asarray(REFERENCE_SHIFT, dtype)
    return {"mean": transition.mean, "variance": transition.variance,
            "mean_log_prob": density(inputs["velocity"]), "velocity_grad": gradient,
            "conditional_kl": transition.kl(shifted)}


def write_recording(command: str) -> None:
    """RECORDED: the author code's transition recorded over check 1's seeds (`record_case`'s summary),
    its float32 run's evidence from its float64 run, the count planned on Dew's (`dew_transition`)."""
    import jax
    import torch
    from diffusers_wan_reference import float64
    from fetched_fixtures import save
    from recorded_references import provenance, record_case

    reference = fetched_reference()

    def evaluate(inputs):
        found = author_transition(reference, inputs, torch.float32)
        with float64():
            truth = author_transition(reference, inputs, torch.float64)
        return found, truth

    with jax.enable_x64(new_val=True):
        recorded = record_case("transition", transition_inputs, evaluate, dew_transition,
                               truth_proxy=dew_transition, store_inputs=False, summary=True)
    recorded["transition.provenance"] = provenance("Flow-GRPO sde_step_with_logprob", COMMIT, "cpu", command,
                                                   torch=torch.__version__)
    written = save(RECORDED, recorded, reference=f"flow_grpo {COMMIT}", license="MIT")
    print(f"{written}: {written.stat().st_size} bytes", flush=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    write_recording("PYTHONPATH=src:tools python tools/parity_flow_grpo.py")


if __name__ == "__main__":
    main()
