#!/usr/bin/env python3
"""Regenerate the Flow-GRPO transition fixture from the pinned author code.

Run with torch and diffusers installed. Downloads one Python source file from
GitHub; no model, tokenizer, or weights are loaded. The KL oracle uses
torch.distributions on the actual conditional transition variance.

`recording` records the same transition over check 1's seeds instead
(tools/recorded_references.py): the author code's float32 run's evidence from
its float64 run, every float32 pin widened, with that run's sketch, each seed's
inputs drawn again by tests/test_flow_grpo.py (`transition_inputs`):

    PYTHONPATH=src:tools python tools/parity_flow_grpo.py recording
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
FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/rl/flow_transition.npz"
RECORDED = FIXTURE.with_name("recorded_flow_transition.npz")
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
    scheduler.sigmas = scheduler.sigmas.to(dtype)
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
    import jax
    import jax.numpy as jnp

    from dew.sampling.flow import flow_transition

    dtype = jnp.asarray(inputs["x"]).dtype
    t = jnp.asarray(np.tile(np.asarray(SIGMAS, np.float32), REPEATS), dtype)
    s = jnp.asarray(np.tile(np.asarray((*SIGMAS[1:], 0.0), np.float32), REPEATS), dtype)
    dimensions = inputs["x"].shape[1]

    def density(velocity):
        transition = flow_transition(inputs["x"], velocity, t, s, noise_level=NOISE_LEVEL)
        return transition.log_prob(inputs["action"]) / dimensions

    transition = flow_transition(inputs["x"], inputs["velocity"], t, s, noise_level=NOISE_LEVEL)
    gradient = jax.grad(lambda velocity: density(velocity).sum())(inputs["velocity"])
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
    import diffusers
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("recording",))
    parser.add_argument("--out", type=Path, default=FIXTURE)
    parsed = parser.parse_args()
    if parsed.mode == "recording":
        write_recording("PYTHONPATH=src:tools python tools/parity_flow_grpo.py recording")
        return
    out = parsed.out
    reference = fetched_reference()

    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    scheduler.set_timesteps(sigmas=[1.0, 0.7, 0.4, 0.1])
    rng = np.random.default_rng(112)
    x = rng.normal(0, 0.2, (4, 2)).astype(np.float32)
    velocity = rng.normal(0, 0.2, (4, 2)).astype(np.float32)
    action = (x - 0.15 * velocity + rng.normal(0, 0.1, (4, 2))).astype(np.float32)
    v = torch.tensor(velocity, requires_grad=True)
    _, log_prob, mean, coefficient = reference.sde_step_with_logprob(
        scheduler, v, scheduler.timesteps, torch.tensor(x), noise_level=0.7,
        prev_sample=torch.tensor(action))
    log_prob.sum().backward()
    if v.grad is None:
        raise RuntimeError("reference density did not differentiate velocity")
    t, s = scheduler.sigmas[:-1], scheduler.sigmas[1:]
    variance = coefficient[:, 0].square() * (t - s)
    reference_mean = mean.detach() + torch.tensor([[0.1, -0.2]])
    std = variance.sqrt()[:, None]
    policy = torch.distributions.Independent(torch.distributions.Normal(mean.detach(), std), 1)
    frozen = torch.distributions.Independent(torch.distributions.Normal(reference_mean, std), 1)
    kl = torch.distributions.kl_divergence(policy, frozen)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=x, velocity=velocity, action=action, t=t.numpy(), t_next=s.numpy(),
             mean=mean.detach().numpy(), variance=variance.numpy(),
             mean_log_prob=log_prob.detach().numpy(), velocity_grad=v.grad.numpy(),
             reference_mean=reference_mean.numpy(), conditional_kl=kl.numpy(),
             noise_level=np.float32(0.7), source=np.asarray(SOURCE),
             torch_version=np.asarray(torch.__version__), diffusers_version=np.asarray(diffusers.__version__))
    print(f"{out}: {out.stat().st_size} bytes; torch {torch.__version__}, diffusers {diffusers.__version__}")


if __name__ == "__main__":
    main()
