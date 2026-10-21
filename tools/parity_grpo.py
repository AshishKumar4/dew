#!/usr/bin/env python3
"""Record verl's GRPO loss and its gradient over seeds, to tests/fixtures/rl/grpo.npz.

verl's own functions compute it (`verl/trainer/ppo/core_algos.py`, verl
0.9.0): `compute_policy_loss_vanilla` under an `ActorConfig` with its GRPO
defaults (clip_ratio 0.2 both sides, clip_ratio_c 3.0, `token-mean`), plus
`beta` times `agg_loss` (`token-mean`) of `kl_penalty(..., "k3")` over the
same mask. Each seed draws a rollout (two prompts, two completions each):
old, current and reference log-probabilities, advantages with both signs,
and a mask with short tails, a few entries set past the clip points so each
term binds. The same calls run once in fp32, the values Dew's composition is
compared with, and once in float64, the exact values both are measured from,
over the seeds check 1 draws (`tools/recorded_references.py`,
tests/reference_error.py).

Run with verl 0.9.0, Torch 2.14.0 CPU and JAX (the recorder plans its seeds
on a JAX transcription):
  uv venv venvs/rlref --python 3.12
  uv pip install --python venvs/rlref/bin/python torch==2.14.0 \\
      --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python venvs/rlref/bin/python verl==0.9.0 trl==1.12.0 numpy jax==0.11.2
  venvs/rlref/bin/python tools/parity_grpo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
VERL = "0.9.0"

ROWS = 4
RESPONSE = 4
BETA = 0.01


def rollout(seed: int = 0) -> dict[str, np.ndarray]:
    """Seed `seed`'s rollout: log-probabilities, advantages and a ragged mask."""
    rng = np.random.default_rng(11 + seed)
    old = rng.normal(-0.5, 0.5, (ROWS, RESPONSE)).astype(np.float32)
    current = (old + rng.normal(0, 0.1, (ROWS, RESPONSE))).astype(np.float32)
    ref = (old + rng.normal(0, 0.2, (ROWS, RESPONSE))).astype(np.float32)
    advantages = rng.normal(0, 1, (ROWS, RESPONSE)).astype(np.float32)
    mask = np.ones((ROWS, RESPONSE), np.float32)
    mask[1, 2:] = 0
    mask[3, 3:] = 0
    # Past the clip points, so removing any term moves the loss: a ratio
    # above the high clip, one below the low clip with a negative advantage,
    # and a dual-clip case with a negative advantage.
    current[0, 0] = old[0, 0] + 0.5
    current[0, 1] = old[0, 1] - 0.5
    advantages[0, 0] = 1.5
    advantages[0, 1] = -1.5
    current[2, 0] = old[2, 0] + 1.5
    advantages[2, 0] = -2.0
    return {"old_log_probs": old, "current_log_probs": current, "ref_log_probs": ref,
            "advantages": advantages, "response_mask": mask}


def verl_loss(arrays: dict[str, np.ndarray], dtype) -> tuple[float, np.ndarray]:
    """verl's GRPO total and its gradient in the current log-probabilities."""
    import torch
    from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla, kl_penalty
    from verl.workers.config.actor import ActorConfig

    tensor = {name: torch.tensor(value, dtype=dtype) for name, value in arrays.items()}
    current = tensor["current_log_probs"].requires_grad_()
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    policy, _ = compute_policy_loss_vanilla(tensor["old_log_probs"], current, tensor["advantages"],
                                            tensor["response_mask"], "token-mean", config, None)
    kl = agg_loss(kl_penalty(current, tensor["ref_log_probs"], "k3"), tensor["response_mask"], "token-mean")
    loss = policy + BETA * kl
    loss.backward()
    assert current.grad is not None
    return loss.item(), current.grad.numpy()


def evaluated(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """verl's loss and gradient on a rollout in fp32, and in float64 on it widened."""
    import torch

    loss, gradient = verl_loss(arrays, torch.float32)
    exact_loss, exact_gradient = verl_loss({name: value.astype(np.float64) for name, value in arrays.items()},
                                           torch.float64)
    return ({"loss": np.asarray(loss, np.float32), "grad": gradient},
            {"loss": np.asarray(exact_loss, np.float64), "grad": exact_gradient})


def proxy(arrays):
    """The same composition in JAX, which plans the seeds the recorder confirms on verl."""
    import jax
    import jax.numpy as jnp

    mask = arrays["response_mask"]

    def loss(current):
        ratio = jnp.exp(current - arrays["old_log_probs"])
        advantages = arrays["advantages"]
        clipped = jnp.maximum(-advantages * ratio, -advantages * jnp.clip(ratio, 0.8, 1.2))
        policy = jnp.where(advantages < 0, jnp.minimum(clipped, -advantages * 3.0), clipped)
        delta = arrays["ref_log_probs"] - current
        kl = jnp.exp(delta) - delta - 1
        return (jnp.sum(policy * mask) + BETA * jnp.sum(kl * mask)) / jnp.sum(mask)

    value, gradient = jax.value_and_grad(loss)(arrays["current_log_probs"])
    return {"loss": value, "grad": gradient}


if __name__ == "__main__":
    from recorded_references import record_reference

    record_reference("grpo", rollout, evaluated, proxy, package="verl", repo="volcengine/verl",
                     tool="tools/parity_grpo.py", out=FIXTURES / "rl" / "grpo.npz", description=__doc__,
                     beta=BETA)
