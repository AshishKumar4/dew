"""Gemma 4's routed branch against transformers 5.16.1.

The block fixture comes from tools/hf_reference_b.py: the feed-forward half
of one `Gemma4TextDecoderLayer` with `enable_moe_block` on random weights,
run the way the layer runs it. Everything runs at fp32 on CPU, and each
parity test states its tolerance and the largest difference observed.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from interop_support import fixture_arrays

from dew.nn.gemma4_moe import Gemma4Experts, Gemma4TextRouter

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gemma4"
HIDDEN, EXPERTS, TOP_K, WIDTH = 32, 4, 2, 16


def fixture() -> dict:
    return fixture_arrays(FIXTURES / "moe.npz")


def router_params(tensors: dict) -> dict:
    return {"proj": {"kernel": jnp.asarray(tensors["router.proj.weight"].T)},
            "scale": jnp.asarray(tensors["router.scale"]),
            "per_expert_scale": jnp.asarray(tensors["router.per_expert_scale"])}


def branch_variables(tensors: dict) -> dict:
    gate_up = tensors["experts.gate_up_proj"]
    return {"params": {
        "router": router_params(tensors),
        "experts": {"gate_proj": {"kernel": jnp.asarray(np.swapaxes(gate_up[:, :WIDTH], 1, 2))},
                    "up_proj": {"kernel": jnp.asarray(np.swapaxes(gate_up[:, WIDTH:], 1, 2))},
                    "down_proj": {"kernel": jnp.asarray(np.swapaxes(tensors["experts.down_proj"], 1, 2))}},
        "experts_input_norm": {"scale": jnp.asarray(tensors["pre_feedforward_layernorm_2.weight"])},
        "mlp_branch_norm": {"scale": jnp.asarray(tensors["post_feedforward_layernorm_1.weight"])},
        "experts_output_norm": {"scale": jnp.asarray(tensors["post_feedforward_layernorm_2.weight"])},
    }}


def branch() -> Gemma4Experts:
    return Gemma4Experts(num_experts=EXPERTS, top_k=TOP_K, hidden_features=WIDTH,
                         out_features=HIDDEN, activation="geglu")


def test_the_routed_branch_matches_the_reference_block():
    """The dense MLP's output and the routed experts' output, each behind
    its own norm, summed: what the layer's post_feedforward_layernorm then
    reads. The router reads the raw residual and the experts its normed
    copy. Tolerance 1e-5; observed 7.2e-07."""
    tensors = fixture()
    output = branch().apply(branch_variables(tensors),
                            jnp.asarray(tensors["hidden"]), jnp.asarray(tensors["mlp_out"]))
    assert float(np.max(np.abs(np.asarray(output) - tensors["output"]))) < 1e-5

    hidden = jnp.asarray(tensors["hidden"]).reshape(-1, HIDDEN)
    weights, indices = Gemma4TextRouter(num_experts=EXPERTS, top_k=TOP_K).apply(
        {"params": router_params(tensors)}, hidden)
    assert np.array_equal(np.asarray(indices), tensors["router_indices"])
    assert float(np.max(np.abs(np.asarray(weights) - tensors["router_weights"]))) < 1e-6
