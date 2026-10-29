"""The MoE router against vLLM's own grouped_topk.

tools/vllm_router_reference.py runs `grouped_topk` from vLLM v0.30.0 (the
router vLLM serves Kimi Linear and the DeepSeek-V3 family with) on gate
logits and a balancing bias drawn so float32 rounding cannot change a choice
(each choice's float64 margin at least MARGIN): the experts it chooses on the
fixture's rows (tests/fixtures/router/vllm.npz), which Dew's `Router` over the
same logits (an identity gate) chooses in the same order, and its weights in
float32 and float64 over check 1's seeds of rows (recorded_vllm.npz), which
Dew's are on par with.

Dew's router divides by the chosen scores' sum plus 1e-20, as the Kimi
Linear and DeepSeek releases do, where vLLM divides by the sum alone; the
drawn sums are far above where that shows in float32.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_with_meta
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording

from dew.nn.moe import Router

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "router" / "vllm.npz"
RECORDING = FIXTURE.with_name("recorded_vllm.npz")
CASES = ("kimi", "grouped", "softmax", "unbiased")


@pytest.fixture(scope="module")
def reference():
    return fixture_with_meta(FIXTURE)


def routed(case: str):
    """Dew's `Router` for `case` (tools/vllm_router_reference.py `CASES`) as a function of a draw's gate
    logits and bias: its weights and its chosen experts, compiled."""
    from tools.vllm_router_reference import CASES as SETTINGS

    config = SETTINGS[case]
    experts = config["experts"]
    router = Router(num_experts=experts, in_features=experts, top_k=config["topk"],
                    score_function=config["scoring_func"], normalize_weights=config["renormalize"],
                    routed_scaling_factor=config["routed_scaling_factor"],
                    expert_groups=config["num_expert_group"], groups_per_token=config["topk_group"],
                    group_score="top2" if config["bias"] else "max", expert_bias=config["bias"],
                    precision=jax.lax.Precision.HIGHEST)

    @jax.jit
    def route(inputs):
        variables = {"params": {"kernel": jnp.eye(experts, dtype=jnp.float32)}}
        if config["bias"]:
            variables["moe"] = {"e_score_correction_bias": inputs["bias"]}
        weights, ids = router.apply(variables, inputs["logits"])
        return {"weights": weights, "ids": ids}
    return route


@pytest.mark.parametrize("case", CASES)
def test_the_router_chooses_and_weighs_as_vllm(reference, case):
    """The experts vLLM chooses on the fixture's rows, in its order; and the weights on par with
    vLLM's from its float64 run over the recording's seeds (check 1)."""
    from tools.vllm_router_reference import drawn

    arrays, _ = reference
    route = routed(case)
    chosen = route({name: jnp.asarray(arrays[f"{case}/{name}"]) for name in ("logits", "bias")})
    np.testing.assert_array_equal(np.asarray(chosen["ids"]), arrays[f"{case}/fp32.ids"])
    seeds = declared_seeds(RECORDING)[case]
    draw, theirs, truth = indexed_recording(recording(RECORDING), case, seeds,
                                            inputs_of=lambda seed: drawn(case, seed))
    assert_as_exact(lambda index, inputs: {"weights": route(inputs)["weights"]}, theirs, draw,
                    f"{case} weights", seeds=seeds, truth_fn=truth)
