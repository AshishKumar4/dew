#!/usr/bin/env python3
"""Record verl-omni's Flow-GRPO loss (`FlowGRPOLoss.compute_loss`) over check 1's seeds.

Diffusion RL's interchange with verl-omni starts from one shared algorithm's loss on fixed tensors
(docs/design/post-training.md §13): a transition's samples' mean log densities under the policy and the
rollout's and their advantages, each seed's drawn again by tests/test_flow_grpo.py (`loss_inputs`). Its
loss and that loss's gradient in the policy's log densities, its float32 run's evidence from its float64
run, are kept with that run's sketch (tools/recorded_references.py). The pinned module is fetched and its
loss run as published, its body unchanged (`reference_source.defined_in`): the module's imports reach
verl's and verl-omni's trainer, serving and Ray stacks, none of which the loss reads, so the loss alone is
defined, over torch. Run with torch installed:

    PYTHONPATH=src:tools:tests python tools/parity_verl_omni_flow_loss.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO, COMMIT = "verl-project/verl-omni", "beab6b764b0b666c6de8468bf84b2e7762f4349c"
SOURCE = "verl_omni/trainer/diffusion/diffusion_algos.py"
SHA256 = "0cd321e17f50fa624ff682048d9c7ef25ea9ad8d097acc6cf1d6cb938e3c0bee"
RECORDED = Path(__file__).resolve().parents[1] / "tests/fixtures/rl/recorded_verl_omni_flow_loss.npz"
ROWS = 1024
"""A transition's samples a draw holds: enough that each seed's gradient, exactly zero wherever the clip holds
a sample's ratio and one or two roundings elsewhere, resolves its error."""
CLIP_RANGE = 1e-4
"""The ratio's clip, Flow-GRPO's (`FlowGRPOObjective.clip_range`'s default)."""
ADV_CLIP_MAX = 5.0
"""The advantages' clip (`FlowGRPOObjective.adv_clip_max`'s default)."""


def loss_inputs(seed: int) -> dict[str, np.ndarray]:
    """A seed's transition, the same on every run: each sample's mean log density under the rollout's
    policy (`previous`) and under the policy (`current`), the two a log ratio apart on the clip's own
    scale, so some ratios the clip moves and some it leaves, and advantages some of which the advantage
    clip moves; and the loss's cotangent, which both arms pull back."""
    rng = np.random.default_rng([6122, seed])
    previous = rng.normal(-1.0, 0.5, ROWS).astype(np.float32)
    current = (previous + rng.normal(0.0, 2 * CLIP_RANGE, ROWS)).astype(np.float32)
    advantage = rng.normal(0.0, 3.0, ROWS).astype(np.float32)
    return {"current": current, "previous": previous, "advantage": advantage,
            "loss_cotangent": np.ones((), np.float32)}


def fetched_reference():
    """`FlowGRPOLoss.compute_loss` at `COMMIT`, fetched, checked and its body unchanged, as a function of the
    class it ignores and its keyword arguments. Its annotations name verl-omni's `DiffusionActorConfig`, of
    which it reads `diffusion_loss.clip_ratio` and `diffusion_loss.adv_clip_max` alone: a namespace."""
    from typing import Any, Optional

    import torch
    from reference_source import defined_in, github

    local = github(REPO, COMMIT, SOURCE, SHA256)
    scope = {"torch": torch, "Any": Any, "Optional": Optional, "DiffusionActorConfig": SimpleNamespace}
    return defined_in(local.read_text(), {"compute_loss"}, scope, within="FlowGRPOLoss",
                      filename=str(local))["compute_loss"].__func__


def verl_omni_loss(compute_loss, inputs: dict, dtype) -> dict[str, np.ndarray]:
    """verl-omni's Flow-GRPO loss of `inputs` in the torch `dtype`, and its gradient in the policy's log
    densities, by the names tests/test_flow_grpo.py reads."""
    import torch

    config = SimpleNamespace(diffusion_loss=SimpleNamespace(clip_ratio=CLIP_RANGE, adv_clip_max=ADV_CLIP_MAX))
    host = torch.empty(0, dtype=dtype).numpy().dtype
    current = torch.tensor(np.asarray(inputs["current"], host), requires_grad=True)
    loss, _ = compute_loss(
        None, old_log_prob=torch.tensor(np.asarray(inputs["previous"], host)), log_prob=current,
        advantages=torch.tensor(np.asarray(inputs["advantage"], host)), config=config)
    loss.backward(torch.tensor(np.asarray(inputs["loss_cotangent"], host)))
    assert current.grad is not None, "the loss reads the policy's log densities"
    return {"loss": loss.detach().numpy(), "current_grad": current.grad.numpy()}


def dew_loss(inputs: dict) -> dict:
    """Dew's Flow-GRPO loss of a draw (`loss_inputs`), by `verl_omni_loss`'s names: the transition's terms
    (`flow_grpo_terms`), every sample kept, their surrogates' mean: tests/test_flow_grpo.py's arm and, in
    float64, its truth."""
    import jax.numpy as jnp
    from reference_error import with_gradients

    from dew.objectives.rl.flow import flow_grpo_terms

    previous, advantage = jnp.asarray(inputs["previous"]), jnp.asarray(inputs["advantage"])
    keep = jnp.ones(previous.shape, bool)

    def loss(current):
        pg, mass, _ = flow_grpo_terms(current, previous, advantage, keep, clip_range=CLIP_RANGE,
                                      adv_clip_max=ADV_CLIP_MAX)
        return pg / mass

    value, gradient = with_gradients(loss)(jnp.asarray(inputs["current"]),
                                           jnp.asarray(inputs["loss_cotangent"]))
    return {"loss": value, "current_grad": gradient}


def write_recording(command: str) -> None:
    """RECORDED: verl-omni's loss recorded over check 1's seeds (`record_case`'s summary), its float32
    run's evidence from its float64 run."""
    import jax
    import torch
    from fetched_fixtures import save
    from recorded_references import provenance, record_case

    from _dew_torch.precision import float64

    reference = fetched_reference()

    def evaluate(inputs):
        found = verl_omni_loss(reference, inputs, torch.float32)
        with float64():
            truth = verl_omni_loss(reference, inputs, torch.float64)
        return found, truth

    with jax.enable_x64(new_val=True):
        recorded = record_case("loss", loss_inputs, evaluate, store_inputs=False, summary=True)
    recorded["loss.provenance"] = provenance("verl-omni FlowGRPOLoss.compute_loss", COMMIT, "cpu", command,
                                             torch=torch.__version__)
    written = save(RECORDED, recorded, reference=f"verl-omni {COMMIT}", license="Apache-2.0")
    print(f"{written}: {written.stat().st_size} bytes", flush=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    write_recording("PYTHONPATH=src:tools:tests python tools/parity_verl_omni_flow_loss.py")


if __name__ == "__main__":
    main()
