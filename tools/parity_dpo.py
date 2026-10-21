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


def batch(seed: int = 0) -> dict[str, np.ndarray]:
    """Seed `seed`'s logits, ids and TRL-layout masks for [chosen; rejected]
    rows: a prompt prefix, a completion span and two short rows standing in
    for padding."""
    rng = np.random.default_rng(7 + seed)
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


def evaluated(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """TRL's loss, gradient and rewards on a batch in fp32, and in float64."""
    import torch

    fp32, exact = trl_loss(arrays, torch.float32), trl_loss(arrays, torch.float64)
    return ({name: np.asarray(value, np.float32) for name, value in fp32.items()},
            {name: np.asarray(value, np.float64) for name, value in exact.items()})


def proxy(arrays):
    """Dew's own composition (`dew.rl.preference_logsigmoid_terms`) on a batch, which plans the seeds
    the recorder confirms on TRL."""
    import jax
    import jax.numpy as jnp

    from dew.rl import preference_logsigmoid_terms

    ids = jnp.asarray(arrays["input_ids"], jnp.int32)[:, 1:, None]
    mask = arrays["completion_mask"][:, 1:]

    def token_log_probs(logits):
        return jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1]), ids, axis=-1)[..., 0]

    def loss(policy_logits):
        policy, ref = token_log_probs(policy_logits), token_log_probs(arrays["ref_logits"])
        half = policy.shape[0] // 2
        terms, (chosen, rejected) = preference_logsigmoid_terms(
            policy[:half], policy[half:], ref[:half], ref[half:], mask[:half], mask[half:], BETA)
        return jnp.mean(terms), jnp.stack([jnp.mean(chosen), jnp.mean(rejected)])

    (value, rewards), gradient = jax.value_and_grad(loss, has_aux=True)(arrays["policy_logits"])
    return {"loss": value, "policy_logits_grad": gradient, "rewards": rewards}


if __name__ == "__main__":
    from recorded_references import record_reference

    record_reference("dpo", batch, evaluated, proxy, package="trl", repo="huggingface/trl",
                     tool="tools/parity_dpo.py", out=FIXTURES / "rl" / "dpo.npz", description=__doc__,
                     beta=BETA)
