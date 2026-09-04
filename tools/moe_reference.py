#!/usr/bin/env python3
"""Write the mixture-of-experts fixtures tests/test_moe.py checks against.

Everything here runs under torch and transformers, which Dew does not depend
on, so this is the only place the reference routers are executed. The fixtures
it writes are what the suite compares against.

Set up the venv and run it:

    uv venv /tmp/moeref --python 3.12
    uv pip install --python /tmp/moeref/bin/python torch \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/moeref/bin/python transformers==5.16.1 numpy
    /tmp/moeref/bin/python tools/moe_reference.py

What lands in tests/fixtures/moe:

- config.json: the fields of each reference config Dew's modules are built
  from, so the test repeats no numbers of its own.
- mixtral.npz: the hidden states, the weights of a `MixtralSparseMoeBlock` in
  the per-expert names a checkpoint carries, the router's top-k weights and
  indices, and the block's output.
- deepseek.npz: the same for a `DeepseekV3MoE`, with a nonzero
  `e_score_correction_bias` so the selection bias is exercised, its shared
  expert's weights under `mlp.shared_experts.*`, and the block output, which
  is the routed sum plus that shared branch.
- deepseek_v4.npz: the router and experts of a DeepSeek V4 sparse layer,
  `DeepseekV4TopKRouter` scoring sqrt(softplus) with a nonzero selection
  bias and `DeepseekV4Experts` clamping the gate and up projections at
  `swiglu_limit`, with the experts' output on the router's choice. The limit
  is set low enough that the clamp bites on these weights, so an expert
  block without it disagrees.

The expert weights are written under `mlp.experts.N.gate_proj.weight` and its
siblings, which is what the checkpoints hold: transformers 5.16.1 merges those
into one `mlp.experts.gate_up_proj` tensor while loading
(`transformers/core_model_loading.py:1545`), gate rows first, and this undoes
that merge so the fixture is in the layout a translation into Dew reads.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4TopKRouter,
)
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "moe"

BATCH, LENGTH, HIDDEN = 2, 5, 16
EXPERT_HIDDEN = 24

MIXTRAL = dict(
    hidden_size=HIDDEN, intermediate_size=EXPERT_HIDDEN, num_local_experts=8,
    num_experts_per_tok=2, hidden_act="silu", router_jitter_noise=0.0)

# n_group of 4 over 8 experts leaves two per group, and topk_group of 2 makes
# four of the eight reachable, which is exactly the top_k: a wrong group mask
# changes which experts a token gets rather than only their order.
DEEPSEEK = dict(
    hidden_size=HIDDEN, moe_intermediate_size=EXPERT_HIDDEN, n_routed_experts=8,
    num_experts_per_tok=4, n_group=4, topk_group=2, norm_topk_prob=True,
    routed_scaling_factor=2.5, n_shared_experts=1, hidden_act="silu")

# V4 sizes its experts by intermediate_size. A limit of 1.0 against weights
# of scale 0.5 over 16 inputs clamps most gate and up values, which is what
# makes the clamp observable at this size.
DEEPSEEK_V4 = dict(
    hidden_size=HIDDEN, intermediate_size=EXPERT_HIDDEN, num_local_experts=8,
    num_experts_per_tok=4, scoring_func="sqrtsoftplus", swiglu_limit=1.0,
    routed_scaling_factor=2.5, hidden_act="silu")


def hidden_states() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20)
    return torch.randn((BATCH, LENGTH, HIDDEN), generator=generator)


def scatter_weights(module: torch.nn.Module, seed: int) -> None:
    """Random weights with something in every tensor.

    A router whose gate is left at its initial zeros routes every token to
    expert 0 and would pass a parity test that had the selection backwards.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in module.parameters():
            tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.5)


def expert_tensors(experts: torch.nn.Module) -> dict:
    """The fused expert parameters back in their per-expert checkpoint names."""
    gate_up = experts.gate_up_proj.detach().to(torch.float32).numpy()
    down = experts.down_proj.detach().to(torch.float32).numpy()
    width = gate_up.shape[1] // 2
    tensors = {}
    for index in range(gate_up.shape[0]):
        tensors[f"mlp.experts.{index}.gate_proj.weight"] = gate_up[index, :width]
        tensors[f"mlp.experts.{index}.up_proj.weight"] = gate_up[index, width:]
        tensors[f"mlp.experts.{index}.down_proj.weight"] = down[index]
    return tensors


def write_mixtral(directory: Path) -> None:
    block = MixtralSparseMoeBlock(MixtralConfig(**MIXTRAL))
    scatter_weights(block, seed=21)
    block.eval()
    states = hidden_states()
    with torch.no_grad():
        _, weights, indices = block.gate(states.reshape(-1, HIDDEN))
        output = block(states)
    arrays = {
        "hidden": states.to(torch.float32).numpy(),
        "mlp.gate.weight": block.gate.weight.detach().to(torch.float32).numpy(),
        "router_weights": weights.to(torch.float32).numpy(),
        "router_indices": indices.to(torch.int32).numpy(),
        "block_output": output.to(torch.float32).numpy(),
        **expert_tensors(block.experts),
    }
    np.savez(directory / "mixtral.npz", **arrays)
    print(f"mixtral.npz: {len(arrays)} arrays, indices[:4]="
          f"{arrays['router_indices'][:4].tolist()}")


def write_deepseek(directory: Path) -> None:
    block = DeepseekV3MoE(DeepseekV3Config(**DEEPSEEK))
    scatter_weights(block, seed=22)
    block.eval()
    # The bias is a buffer a training step moves, so a fixture written at its
    # zeros would not tell a router that reads it from one that ignores it.
    bias = torch.linspace(-0.4, 0.4, DEEPSEEK["n_routed_experts"])
    with torch.no_grad():
        block.gate.e_score_correction_bias.copy_(bias)
    states = hidden_states()
    with torch.no_grad():
        _, weights, indices = block.gate(states)
        output = block(states)
    shared = {f"mlp.shared_experts.{name}": tensor.detach().to(torch.float32).numpy()
              for name, tensor in block.shared_experts.named_parameters()}
    arrays = {
        "hidden": states.to(torch.float32).numpy(),
        "mlp.gate.weight": block.gate.weight.detach().to(torch.float32).numpy(),
        "mlp.gate.e_score_correction_bias": bias.to(torch.float32).numpy(),
        "router_weights": weights.to(torch.float32).numpy(),
        "router_indices": indices.to(torch.int32).numpy(),
        "block_output": output.to(torch.float32).numpy(),
        **expert_tensors(block.experts),
        **shared,
    }
    np.savez(directory / "deepseek.npz", **arrays)
    print(f"deepseek.npz: {len(arrays)} arrays, indices[:4]="
          f"{arrays['router_indices'][:4].tolist()}")


def write_deepseek_v4(directory: Path) -> None:
    config = DeepseekV4Config(**DEEPSEEK_V4)
    router = DeepseekV4TopKRouter(config)
    experts = DeepseekV4Experts(config)
    scatter_weights(router, seed=23)
    scatter_weights(experts, seed=24)
    router.eval()
    experts.eval()
    bias = torch.linspace(-0.4, 0.4, DEEPSEEK_V4["num_local_experts"])
    with torch.no_grad():
        router.e_score_correction_bias.copy_(bias)
    states = hidden_states()
    with torch.no_grad():
        _, weights, indices = router(states)
        output = experts(states.reshape(-1, HIDDEN), indices, weights)
        # The clamp has to be doing something for the fixture to test it.
        gate_up = torch.nn.functional.linear(
            states.reshape(-1, HIDDEN), experts.gate_up_proj[0])
        clipped = float((gate_up.abs() > DEEPSEEK_V4["swiglu_limit"]).float().mean())
    arrays = {
        "hidden": states.to(torch.float32).numpy(),
        "mlp.gate.weight": router.weight.detach().to(torch.float32).numpy(),
        "mlp.gate.e_score_correction_bias": bias.to(torch.float32).numpy(),
        "router_weights": weights.to(torch.float32).numpy(),
        "router_indices": indices.to(torch.int32).numpy(),
        "experts_output": output.to(torch.float32).numpy(),
        **expert_tensors(experts),
    }
    np.savez(directory / "deepseek_v4.npz", **arrays)
    print(f"deepseek_v4.npz: {len(arrays)} arrays, indices[:4]="
          f"{arrays['router_indices'][:4].tolist()}, expert 0 clamps "
          f"{clipped:.0%} of its gate and up values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    (out / "config.json").write_text(
        json.dumps({"mixtral": MIXTRAL, "deepseek": DEEPSEEK,
                    "deepseek_v4": DEEPSEEK_V4}, indent=2) + "\n")
    write_mixtral(out)
    write_deepseek(out)
    write_deepseek_v4(out)
    size = sum(path.stat().st_size for path in out.iterdir())
    print(f"{out}: {size / 1e3:.0f} kB, {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
