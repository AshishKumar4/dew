"""The MoE router against vLLM's own grouped_topk.

tools/vllm_router_reference.py runs `grouped_topk` from vLLM v0.30.0 (the
router vLLM serves Kimi Linear and the DeepSeek-V3 family with) on gate
logits and a balancing bias, and records the experts it chooses and their
weights in float32 and in float64 (tests/fixtures/router/vllm.npz). Dew's
`Router` over the same logits (an identity gate) chooses the same experts
in the same order, and its weights are held to tests/reference_error.py's
rule: no further from float64 than twice vLLM's float32 weights.

Dew's router divides by the chosen scores' sum plus 1e-20, as the Kimi
Linear and DeepSeek releases do, where vLLM divides by the sum alone; the
fixture's sums are far above where that shows in float32.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.nn.moe import Router

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "router" / "vllm.npz"
CASES = ("kimi", "grouped", "softmax", "unbiased")


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE) as loaded:
        arrays = dict(loaded)
    return arrays, json.loads(arrays.pop("meta").tobytes())


@pytest.mark.parametrize("case", CASES)
def test_the_router_chooses_and_weighs_as_vllm(reference, case):
    arrays, meta = reference
    config = meta["cases"][case]

    def part(name):
        return arrays[f"{case}/{name}"]

    experts = config["experts"]
    router = Router(num_experts=experts, in_features=experts, top_k=config["topk"],
                    score_function=config["scoring_func"], normalize_weights=config["renormalize"],
                    routed_scaling_factor=config["routed_scaling_factor"],
                    expert_groups=config["num_expert_group"], groups_per_token=config["topk_group"],
                    group_score="top2" if config["bias"] else "max", expert_bias=config["bias"],
                    precision=jax.lax.Precision.HIGHEST)
    variables = {"params": {"kernel": jnp.eye(experts, dtype=jnp.float32)}}
    if config["bias"]:
        variables["moe"] = {"e_score_correction_bias": jnp.asarray(part("bias"))}
    weights, ids = jax.jit(router.apply)(variables, jnp.asarray(part("logits")))
    np.testing.assert_array_equal(np.asarray(ids), part("fp32.ids"))
    assert_as_exact_as_the_reference(np.asarray(weights), part("fp32.weights"), part("fp64.weights"),
                                     f"{case} weights")
