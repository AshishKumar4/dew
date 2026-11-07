"""The MoE router against vLLM's own grouped_topk.

tools/vllm_router_worker.py runs `grouped_topk` from vLLM v0.30.0 (the router vLLM serves Kimi Linear and
the DeepSeek-V3 family with) on drawn gate logits and a balancing bias. Over those draws, Dew's `Router`
over the same logits (an identity gate) gives each token's weight of every expert, its chosen experts'
weights scattered by their ids, as exact as vLLM's float32 run against its float64 run (check 1), each
router its own recording (tools/venv_reference.py's `router-<case>` sites), Dew's float64 run held to the
recorded sketch. A token's chosen experts are its weights' support, so an expert Dew chose that vLLM did
not is an error of a whole weight. Every drawn token's k-th and (k+1)-th selection scores, and its groups'
scores where it chooses among groups, are at least MARGIN apart in float64 (`margins`), so float32
rounding cannot change a choice.

Dew's router divides by the chosen scores' sum plus 1e-20, as the Kimi Linear and DeepSeek releases do,
where vLLM divides by the sum alone; the drawn sums are far above where that shows in float32.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_runs import assert_arms_as_recorded

from dew.nn.moe import Router

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "router"
CASES = {
    "kimi": {"experts": 256, "topk": 8, "renormalize": True, "num_expert_group": 1, "topk_group": 1,
             "scoring_func": "sigmoid", "routed_scaling_factor": 2.446, "bias": True},
    "grouped": {"experts": 64, "topk": 6, "renormalize": True, "num_expert_group": 8, "topk_group": 3,
                "scoring_func": "sigmoid", "routed_scaling_factor": 2.5, "bias": True},
    "softmax": {"experts": 32, "topk": 4, "renormalize": False, "num_expert_group": 1, "topk_group": 1,
                "scoring_func": "softmax", "routed_scaling_factor": 1.0, "bias": True},
    "unbiased": {"experts": 32, "topk": 4, "renormalize": True, "num_expert_group": 4, "topk_group": 2,
                 "scoring_func": "softmax", "routed_scaling_factor": 1.0, "bias": False},
}
"""Kimi Linear's published router (256 experts, 8 chosen, sigmoid, renormalized, scaled by 2.446, one group,
a bias that changes choices), then the other paths: 64 experts in 8 groups, 3 groups per token by their two
best biased scores; softmax, not renormalized, with a bias; softmax over 4 groups, 2 per token by their best
score, no bias. Each is drawn to its worker as `grouped_topk`'s arguments, its scoring function as
whether it is the sigmoid."""
ROUTER_SEEDS = {"kimi": 8, "grouped": 8, "softmax": 16, "unbiased": 8}
"""Each router's declared seeds, sized by `seeds_for` paired as its recording confirmed them."""
OUTPUTS = ("weights",)
"""The leaf `dew` gives, each site's recorded leaf."""
TOKENS = 64
MARGIN = 1e-5


def margins(logits: np.ndarray, bias: np.ndarray, config: dict) -> np.ndarray:
    """Each token's smallest float64 gap between a chosen and an unchosen score: the k-th and (k+1)-th
    expert among its groups, and its last chosen and first unchosen group."""
    wide = logits.astype(np.float64)
    if config["scoring_func"] == "softmax":
        scores = np.exp(wide - wide.max(-1, keepdims=True))
        scores /= scores.sum(-1, keepdims=True)
    else:
        scores = 1 / (1 + np.exp(-wide))
    selection = scores + bias.astype(np.float64)
    groups = selection.reshape(len(selection), config["num_expert_group"], -1)
    grouped = np.sort(groups, -1)[..., -2:].sum(-1) if config["bias"] else groups.max(-1)
    ranked = -np.sort(-grouped, -1)
    gaps = [np.full(len(selection), np.inf)]
    if config["topk_group"] < config["num_expert_group"]:
        gaps.append(ranked[:, config["topk_group"] - 1] - ranked[:, config["topk_group"]])
    kept = np.argsort(-grouped, -1)[:, :config["topk_group"]]
    within = np.take_along_axis(groups, kept[..., None], 1).reshape(len(selection), -1)
    ordered = -np.sort(-within, -1)
    gaps.append(ordered[:, config["topk"] - 1] - ordered[:, config["topk"]])
    return np.min(gaps, 0)


def drawing(case: str):
    """`draw(seed)` of `case`: a bias of the experts' scale (zero for an unbiased router) and TOKENS rows of
    gate logits, each drawn until its choices clear MARGIN (`margins`)."""
    config = CASES[case]
    experts, sigmoid = config["experts"], config["scoring_func"] == "sigmoid"

    def draw(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng([56, seed])
        bias = rng.standard_normal(experts) * 0.05 if config["bias"] else np.zeros(experts)
        bias = bias.astype(np.float32)
        rows = []
        while len(rows) < TOKENS:
            row = (rng.standard_normal((1, experts)) * 2).astype(np.float32)
            if margins(row, bias, config)[0] >= MARGIN:
                rows.append(row[0])
        numbers = {key: value for key, value in config.items() if key not in ("experts", "scoring_func")}
        return {"logits": np.stack(rows), "bias": bias, "config.sigmoid": np.asarray(sigmoid),
                **{f"config.{key}": np.asarray(value) for key, value in numbers.items()}}
    return draw


def computing(case: str):
    """`dew(dtype, inputs)` of `case`: Dew's router's weight of every expert at `dtype`, its chosen
    experts' weights scattered by their ids."""
    config = CASES[case]
    experts = config["experts"]

    def dew(dtype, inputs) -> dict:
        router = Router(num_experts=experts, in_features=experts, top_k=config["topk"],
                        score_function=config["scoring_func"], normalize_weights=config["renormalize"],
                        routed_scaling_factor=config["routed_scaling_factor"],
                        expert_groups=config["num_expert_group"], groups_per_token=config["topk_group"],
                        group_score="top2" if config["bias"] else "max", expert_bias=config["bias"],
                        precision=jax.lax.Precision.HIGHEST)
        variables = {"params": {"kernel": jnp.eye(experts, dtype=dtype)}}
        if config["bias"]:
            variables["moe"] = {"e_score_correction_bias": jnp.asarray(inputs["bias"], dtype)}
        weights, ids = router.apply(variables, jnp.asarray(inputs["logits"], dtype))
        rows = jnp.arange(weights.shape[0])[:, None]
        return {"weights": jnp.zeros((weights.shape[0], experts), weights.dtype).at[rows, ids].set(weights)}
    return dew


def pipeline_arms(case: str):
    """tools/venv_reference.py's `router-<case>` site: its recording, draw, Dew run and leaf."""
    return FIXTURES / f"{case}.npz", drawing(case), computing(case), OUTPUTS


@pytest.mark.parametrize("case", list(CASES))
def test_the_router_chooses_and_weighs_as_vllm(case):
    """Dew's weight of every expert as exact as vLLM's float32 against float64 (check 1)."""
    assert_arms_as_recorded(pipeline_arms(case), f"router-{case}", ROUTER_SEEDS[case], f"{case} router")
