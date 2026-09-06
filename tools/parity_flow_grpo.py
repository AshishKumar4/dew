#!/usr/bin/env python3
"""Regenerate the Flow-GRPO transition fixture from the pinned author code.

Run with torch and diffusers==0.34.0 installed. Downloads one Python source
file from GitHub; no model, tokenizer, or weights are loaded. The KL oracle
uses torch.distributions on the actual conditional transition variance.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tempfile
from urllib.request import urlopen

import numpy as np

COMMIT = "879042cf5707f8b90daa98d147d7deac2317c5da"
SOURCE = (f"https://raw.githubusercontent.com/yifan123/flow_grpo/{COMMIT}/"
          "flow_grpo/diffusers_patch/sd3_sde_with_logprob.py")
FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/rl/flow_transition.npz"


def main() -> None:
    import diffusers
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE)
    out = parser.parse_args().out
    with tempfile.TemporaryDirectory(prefix="dew-flow-reference-") as directory:
        path = Path(directory) / "reference.py"
        with urlopen(SOURCE, timeout=30) as response:
            path.write_bytes(response.read())
        spec = importlib.util.spec_from_file_location("flow_reference", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the pinned Flow-GRPO reference")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)

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
