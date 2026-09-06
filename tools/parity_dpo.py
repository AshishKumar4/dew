#!/usr/bin/env python3
"""Write TRL's DPO loss and its gradients to tests/fixtures/rl/dpo.npz.

The reference is TRL 1.12's DPO path with the defaults (`sigmoid` loss,
`reverse_kl`): per-token log-probabilities zeroed where
`completion_mask[:, 1:]` is 0 and summed over the sequence, the `[chosen,
rejected]` chunking, and `mean(-logsigmoid(beta * delta))`
(`trl/trainer/dpo_trainer.py`, the forward branch and `dpo_loss`). The
tensors are fixed random log-probabilities, so the fixture pins the math
without a model; the gradients come from autograd over the same four
tensors. Dew's `preference_logsigmoid` must match both. Runs in an
environment with torch and TRL installed; Dew never imports either.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

PAIRS = 3
SEQUENCE = 6
BETA = 0.1


def main(argv: list[str] | None = None) -> None:
    import torch
    import torch.nn.functional as F
    import trl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES / "rl" / "dpo.npz")
    out = parser.parse_args(argv).out

    rng = np.random.default_rng(7)
    rows = 2 * PAIRS
    policy = torch.tensor(rng.normal(size=(rows, SEQUENCE - 1)).astype(np.float32),
                          requires_grad=True)
    ref = torch.tensor(rng.normal(size=(rows, SEQUENCE - 1)).astype(np.float32),
                       requires_grad=True)
    # Full-length masks in TRL's layout: a prompt prefix, a completion span,
    # and one short row standing in for padding.
    mask = torch.ones(rows, SEQUENCE)
    mask[:, :2] = 0
    mask[1, 4:] = 0
    mask[rows - 1, 5:] = 0

    shift = mask[:, 1:]
    scored = policy * shift
    ref_scored = ref * shift
    logps = scored.sum(dim=1)
    ref_logps = ref_scored.sum(dim=1)
    chosen_logps, rejected_logps = logps.chunk(2, dim=0)
    ref_chosen, ref_rejected = ref_logps.chunk(2, dim=0)
    delta = (chosen_logps - ref_chosen) - (rejected_logps - ref_rejected)
    loss = (-F.logsigmoid(BETA * delta)).mean()
    loss.backward()

    assert policy.grad is not None and ref.grad is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        policy_logps=policy.detach().numpy(),
        ref_logps=ref.detach().numpy(),
        completion_mask=mask.numpy(),
        beta=np.asarray(BETA),
        trl_loss=np.asarray(loss.item(), dtype=np.float32),
        trl_policy_grad=policy.grad.numpy(),
        trl_ref_grad=ref.grad.numpy(),
        trl_version=np.array(trl.__version__),
        torch_version=np.array(torch.__version__),
    )
    print(f"{out}: {out.stat().st_size / 1e3:.1f} kB")
    print(f"  loss {loss.item():.6f}, trl {trl.__version__}, torch {torch.__version__}")


if __name__ == "__main__":
    main()
