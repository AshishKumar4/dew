#!/usr/bin/env python3
"""Write verl's GRPO loss to tests/fixtures/rl/grpo.npz.

The reference is verl 0.9's PPO path with the GRPO settings
(`verl/trainer/ppo/core_algos.py`): `compute_policy_loss_vanilla` with
`clip_ratio` 0.2 both sides and `clip_ratio_c` 3.0, the log-ratio clamped to
+-20 before the exponential, aggregated with `token-mean`, plus `beta` times
the `token-mean` of `kl_penalty_forward` with `k3` over the same mask. The
tensors are one fixed rollout (two prompts, two completions each): old,
current and reference log-probabilities, advantages with both signs, and a
mask with short tails. A few entries are set by hand past the clip points so
each term binds. Dew's GRPO composition must match the total. Runs in an
environment with torch installed; Dew never imports it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

ROWS = 4
RESPONSE = 4
BETA = 0.01
EPS_LOW = 0.2
EPS_HIGH = 0.2
DUAL_CLIP = 3.0


def main(argv: list[str] | None = None) -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES / "rl" / "grpo.npz")
    out = parser.parse_args(argv).out

    rng = np.random.default_rng(11)
    old = rng.normal(-0.5, 0.5, (ROWS, RESPONSE)).astype(np.float32)
    current = (old + rng.normal(0, 0.1, (ROWS, RESPONSE))).astype(np.float32)
    ref = (old + rng.normal(0, 0.2, (ROWS, RESPONSE))).astype(np.float32)
    advantages = rng.normal(0, 1, (ROWS, RESPONSE)).astype(np.float32)
    mask = np.ones((ROWS, RESPONSE), np.float32)
    mask[1, 2:] = 0
    mask[3, 3:] = 0
    # Entries past the clip points, so removing any term moves the loss: a
    # ratio above the high clip, one below the low clip with a negative
    # advantage, and a dual-clip case with a negative advantage.
    current[0, 0] = old[0, 0] + 0.5
    current[0, 1] = old[0, 1] - 0.5
    advantages[0, 0] = 1.5
    advantages[0, 1] = -1.5
    current[2, 0] = old[2, 0] + 1.5
    advantages[2, 0] = -2.0

    old_t = torch.tensor(old)
    current_t = torch.tensor(current, requires_grad=True)
    ref_t = torch.tensor(ref)
    advantages_t = torch.tensor(advantages)
    mask_t = torch.tensor(mask)

    log_ratio = torch.clamp(current_t - old_t, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    pg1 = -advantages_t * ratio
    pg2 = -advantages_t * torch.clamp(ratio, 1 - EPS_LOW, 1 + EPS_HIGH)
    worse = torch.maximum(pg1, pg2)
    capped = -advantages_t * DUAL_CLIP
    per_token = torch.where(advantages_t < 0, torch.minimum(capped, worse), worse)
    pg_loss = (per_token * mask_t).sum() / mask_t.sum()

    kl = torch.clamp(ref_t - current_t, min=-20, max=20)
    k3 = torch.clamp(torch.exp(kl) - kl - 1, min=-10, max=10)
    kl_loss = (k3 * mask_t).sum() / mask_t.sum()
    loss = pg_loss + BETA * kl_loss
    loss.backward()
    assert current_t.grad is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        old_log_probs=old,
        current_log_probs=current,
        ref_log_probs=ref,
        advantages=advantages,
        response_mask=mask,
        beta=np.asarray(BETA),
        verl_loss=np.asarray(loss.item(), dtype=np.float32),
        verl_current_grad=current_t.grad.numpy(),
        verl_version=np.array("0.9.0"),
        torch_version=np.array(torch.__version__),
    )
    print(f"{out}: {out.stat().st_size / 1e3:.1f} kB")
    print(f"  loss {loss.item():.6f} (pg {pg_loss.item():.6f}, "
          f"kl {kl_loss.item():.6f}), torch {torch.__version__}")


if __name__ == "__main__":
    main()
