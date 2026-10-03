#!/usr/bin/env python3
"""Write TRL's DPO loss, rewards and gradient to tests/fixtures/rl/dpo.npz.

TRL's own `DPOTrainer._compute_loss` (trl 1.12.0) computes them, with the
DPOConfig defaults it reads off the config class (the `sigmoid` loss,
`reverse_kl`, no length weighting) and `beta` 0.1. Everything from logits to
loss is TRL's code: the shift, `selective_log_softmax`, zeroing outside
`completion_mask[:, 1:]`, the per-sequence sums, the `[chosen, rejected]`
chunking, the reference forward and `mean(-logsigmoid(beta * delta))`. The
trainer around it is a stand-in holding those settings, and the policy and
reference models are stand-ins returning fixed logits, so the fixture pins
the loss without a model. Autograd gives the gradient in the policy logits.
The same call runs once in fp32, the values Dew is compared with, and once
in float64, the exact values both are measured from
(tests/reference_error.py).

Run with trl 1.12.0 and Torch 2.14.0 CPU (see tools/parity_grpo.py for the
environment):
  /tmp/rlref/bin/python tools/parity_dpo.py
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
TRL = "1.12.0"

PAIRS = 3
SEQUENCE = 6
VOCAB = 8
BETA = 0.1


def batch() -> dict[str, np.ndarray]:
    """Fixed logits, ids and TRL-layout masks for [chosen; rejected] rows: a
    prompt prefix, a completion span and two short rows standing in for
    padding."""
    rng = np.random.default_rng(7)
    rows = 2 * PAIRS
    mask = np.ones((rows, SEQUENCE), np.float32)
    mask[:, :2] = 0
    mask[1, 4:] = 0
    mask[rows - 1, 5:] = 0
    return {"policy_logits": rng.normal(size=(rows, SEQUENCE, VOCAB)).astype(np.float32),
            "ref_logits": rng.normal(size=(rows, SEQUENCE, VOCAB)).astype(np.float32),
            "input_ids": rng.integers(0, VOCAB, (rows, SEQUENCE)).astype(np.int64),
            "completion_mask": mask}


def trl_loss(arrays: dict[str, np.ndarray], dtype) -> dict[str, np.ndarray]:
    """TRL's DPO loss over `arrays`, its gradient in the policy logits and
    the chosen and rejected rewards it logs."""
    import torch
    from trl import DPOConfig, DPOTrainer

    defaults = {field.name: field.default if field.default is not dataclasses.MISSING
                else field.default_factory() if field.default_factory is not dataclasses.MISSING else None
                for field in dataclasses.fields(DPOConfig)}
    policy = torch.tensor(arrays["policy_logits"], dtype=dtype, requires_grad=True)
    reference = torch.tensor(arrays["ref_logits"], dtype=dtype)

    class Fixed(torch.nn.Module):
        """A model whose forward returns the same logits whatever it is fed."""

        def __init__(self, logits):
            super().__init__()
            self.fixed = logits

        def forward(self, **_):
            return SimpleNamespace(logits=self.fixed)

    identity = SimpleNamespace(device=torch.device("cpu"), gather=lambda x: x, gather_for_metrics=lambda x: x,
                               unwrap_model=lambda model: model)
    trainer = SimpleNamespace(
        model=SimpleNamespace(training=True, is_gradient_checkpointing=False),
        ref_model=Fixed(reference),
        accelerator=identity, args=SimpleNamespace(gradient_checkpointing_kwargs=None),
        aux_loss_enabled=False, ld_alpha=defaults["ld_alpha"], precompute_ref_logps=False,
        f_divergence_type=defaults["f_divergence_type"], loss_types=list(defaults["loss_type"]),
        loss_weights=[1.0], beta=BETA, use_weighting=defaults["use_weighting"],
        _metrics=defaultdict(lambda: defaultdict(list)), _total_train_tokens=0)
    inputs = {"input_ids": torch.tensor(arrays["input_ids"]),
              "attention_mask": torch.ones(arrays["input_ids"].shape, dtype=torch.long),
              "completion_mask": torch.tensor(arrays["completion_mask"], dtype=torch.long)}
    loss = DPOTrainer._compute_loss(trainer, Fixed(policy), inputs, return_outputs=False)
    loss.backward()
    assert policy.grad is not None
    metrics = trainer._metrics["train"]
    return {"loss": np.asarray(loss.item()), "policy_logits_grad": policy.grad.numpy(),
            "rewards": np.asarray([metrics["rewards/chosen"][0], metrics["rewards/rejected"][0]])}


def main(argv: list[str] | None = None) -> None:
    import torch
    import trl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES / "rl" / "dpo.npz")
    out = parser.parse_args(argv).out
    if trl.__version__ != TRL:
        raise RuntimeError(f"run with trl {TRL}, not {trl.__version__}")
    torch.set_num_threads(1)
    arrays = batch()
    fp32 = trl_loss(arrays, torch.float32)
    exact = trl_loss(arrays, torch.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays, beta=np.asarray(BETA),
             **{f"trl_{name}": np.asarray(value, np.float32) for name, value in fp32.items()},
             **{f"trl_{name}_f64": value for name, value in exact.items()},
             trl_version=np.array(trl.__version__), torch_version=np.array(torch.__version__))
    print(f"{out}: loss {fp32['loss']:.6f}, trl {trl.__version__}, torch {torch.__version__}")


if __name__ == "__main__":
    main()
