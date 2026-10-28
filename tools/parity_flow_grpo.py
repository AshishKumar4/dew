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
from pathlib import Path

import numpy as np

REPO, COMMIT = "yifan123/flow_grpo", "879042cf5707f8b90daa98d147d7deac2317c5da"
SOURCE = "flow_grpo/diffusers_patch/sd3_sde_with_logprob.py"
SHA256 = "4ba34220d7f856b4f6aae8f27f7af3ccd3591df597dd53b5dcb5237066e25075"
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
    # Each row's mean log density pulls back one, the cotangent both arms take from the draw.
    return {"x": x, "velocity": velocity, "action": action, "density_cotangent": np.ones(rows, np.float32)}


def fetched_reference():
    """The pinned author module, fetched once, checked and run as published."""
    from reference_source import github, loaded

    return loaded("flow_reference", github(REPO, COMMIT, SOURCE, SHA256))


def author_transition(reference, inputs: dict, dtype) -> dict[str, np.ndarray]:
    """The author code's transition on `inputs` in the torch `dtype`: its mean, variance, mean log
    density and that density's velocity gradient, and its KL to the reference shifted by
    REFERENCE_SHIFT, by the names tests/test_flow_grpo.py reads."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from reference_error import torch_with_gradients

    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    scheduler.set_timesteps(sigmas=list(SIGMAS))
    # The schedule as Dew's transition takes it, SIGMAS at float32 and the final zero, in `dtype`: the
    # scheduler's own round trip through its timesteps moves a sigma by an ulp, a different input. From
    # NumPy, since the float64 run's `torch.tensor` is widened and would hold SIGMAS unrounded.
    scheduler.sigmas = torch.from_numpy(np.asarray((*SIGMAS, 0.0), np.float32)).to(dtype)
    timesteps = scheduler.timesteps.repeat(REPEATS)
    host = torch.empty(0, dtype=dtype).numpy().dtype

    def stepped(v, x, action):
        return reference.sde_step_with_logprob(scheduler, v, timesteps, x, noise_level=NOISE_LEVEL,
                                               prev_sample=action)

    operands = tuple(np.asarray(inputs[name], host) for name in ("velocity", "x", "action"))
    _, log_prob, mean, coefficient = stepped(*(torch.as_tensor(operand) for operand in operands))
    # The density's velocity gradient, the draw's cotangent pulled back through the gate's torch twin.
    _, velocity_grad, _, _ = torch_with_gradients(lambda v, x, action: stepped(v, x, action)[1])(
        *operands, inputs["density_cotangent"])
    t = scheduler.sigmas[:-1].repeat(REPEATS)
    s = scheduler.sigmas[1:].repeat(REPEATS)
    variance = coefficient[:, 0].square() * (t - s)
    std = variance.sqrt()[:, None]
    shifted = mean.detach() + torch.tensor([REFERENCE_SHIFT], dtype=dtype)
    policy = torch.distributions.Independent(torch.distributions.Normal(mean.detach(), std), 1)
    frozen = torch.distributions.Independent(torch.distributions.Normal(shifted, std), 1)
    kl = torch.distributions.kl_divergence(policy, frozen)
    return {"mean": mean.detach().numpy(), "variance": variance.detach().numpy(),
            "mean_log_prob": log_prob.detach().numpy(), "velocity_grad": velocity_grad,
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
    # The draw's cotangent on each row's density, through the gate's pullback.
    cotangent = jnp.asarray(inputs["density_cotangent"], inputs["velocity"].dtype)
    _, gradient = with_gradients(density)(inputs["velocity"], cotangent)
    shifted = transition.mean + jnp.asarray(REFERENCE_SHIFT, dtype)
    return {"mean": transition.mean, "variance": transition.variance,
            "mean_log_prob": density(inputs["velocity"]), "velocity_grad": gradient,
            "conditional_kl": transition.kl(shifted)}


def write_recording(command: str) -> None:
    """RECORDED: the author code's transition recorded over check 1's seeds (`record_case`'s summary),
    its float32 run's evidence from its float64 run, the count planned on Dew's (`dew_transition`)."""
    import jax
    import torch
    from fetched_fixtures import save
    from recorded_references import provenance, record_case
    from torch_precision import float64

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
