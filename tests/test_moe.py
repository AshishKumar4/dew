"""Mixture of experts: router parity, the grouped matmul and the balancing
bias. The decoder that grows experts and the expert mesh axis are in
tests/test_moe_decoder.py.

The parity fixtures come from transformers 5.16.1 through
tools/moe_reference.py: `MixtralSparseMoeBlock` for softmax routing and the
block output, `DeepseekV3MoE` for sigmoid scores, the selection bias, the
group limit and the shared expert, and DeepSeek V4's router and experts for
sqrt(softplus) scores and the swiglu limit. Everything runs at fp32 on CPU,
and each parity test states its tolerance and the largest difference
observed.

Slot order inside a token's top-k carries no meaning: the reference calls
`torch.topk(sorted=False)` and both implementations sum over the k slots, so
the comparisons sort each token's slots by expert id first.
"""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from interop_support import fixture_arrays
from moe_support import by_expert, router_variables

from dew.nn.mlp import GatedMLP
from dew.nn.moe import ExpertMLP, Router, SparseMLP, load_balance_update

pytestmark = pytest.mark.kernel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "moe"
CONFIG = json.loads((FIXTURES / "config.json").read_text())
ROUTER_TOLERANCE = 1e-6
"""A router's gate values against the reference's, at fp32 on CPU: the largest observed
difference is 1.79e-07, from the gate matmul reading a transposed kernel."""
OUTPUT_TOLERANCE = 2e-5
"""An expert block's output against the reference's, at fp32 on CPU: the largest observed
difference is 7.63e-06 on outputs that reach 34.9, the summation order's spread."""


def fixture(name: str) -> dict:
    return fixture_arrays(FIXTURES / f"{name}.npz")


def sparse_variables(tensors, num_experts):
    """A checkpoint's per-expert tensors as a `SparseMLP` parameter tree.

    This is the whole translation a Qwen3.5-MoE or DeepSeek V4 checkpoint
    needs for its feed-forward: stack the experts of one projection onto a
    leading dimension, transpose each expert's matrix.
    """
    def stack(projection):
        return jnp.asarray(np.stack([
            tensors[f"mlp.experts.{expert}.{projection}.weight"].T
            for expert in range(num_experts)]))

    return {"params": {
        "gate": {"kernel": jnp.asarray(tensors["mlp.gate.weight"].T)},
        "experts": {projection: {"kernel": stack(projection)}
                    for projection in ("gate_proj", "up_proj", "down_proj")},
    }}


def mixtral_router(**overrides) -> Router:
    config = CONFIG["mixtral"]
    return Router(num_experts=config["num_local_experts"],
                  in_features=config["hidden_size"],
                  top_k=config["num_experts_per_tok"], **overrides)


def deepseek_router(**overrides) -> Router:
    config = CONFIG["deepseek"]
    settings = {"score_function": 'sigmoid',
                    "normalize_weights": config["norm_topk_prob"],
                    "routed_scaling_factor": config["routed_scaling_factor"],
                    "expert_groups": config["n_group"],
                    "groups_per_token": config["topk_group"],
                    "expert_bias": True}
    settings.update(overrides)
    return Router(num_experts=config["n_routed_experts"],
                  in_features=config["hidden_size"],
                  top_k=config["num_experts_per_tok"], **settings)


# --------------------------------------------------------------------------
# Router parity against transformers 5.16.1
# --------------------------------------------------------------------------

def test_router_reproduces_the_mixtral_choice_and_gate_values():
    """MixtralSparseMoeBlock: softmax over the experts, top-k, renormalise."""
    tensors = fixture("mixtral")
    hidden = jnp.asarray(tensors["hidden"]).reshape(-1, CONFIG["mixtral"]["hidden_size"])
    weights, indices = mixtral_router().apply(router_variables(tensors), hidden)

    # Largest observed difference 1.79e-07 at fp32 on CPU, from the gate
    # matmul reading a transposed kernel.
    assert_routes_as_recorded(tensors, weights, indices)


def assert_routes_as_recorded(tensors, weights, indices) -> None:
    """The router chose each token's recorded experts and gated them as recorded, expert by expert."""
    theirs_indices, theirs_weights = by_expert(tensors["router_indices"], tensors["router_weights"])
    ours_indices, ours_weights = by_expert(
        indices.reshape(-1, indices.shape[-1]), weights.reshape(-1, weights.shape[-1]))
    assert np.array_equal(ours_indices, theirs_indices)
    assert np.max(np.abs(ours_weights - theirs_weights)) < ROUTER_TOLERANCE


def test_the_selection_bias_never_reaches_the_gate_values():
    """DeepSeek's balancing bias decides which experts a token gets and has no
    say in what they contribute, so the weights are the unbiased scores."""
    tensors = fixture("deepseek")
    config = CONFIG["deepseek"]
    hidden = jnp.asarray(tensors["hidden"])
    router = deepseek_router()
    variables = router_variables(tensors, bias=True)
    (weights, indices), sown = router.apply(variables, hidden, mutable=["router"])

    scores = sown["router"]["scores"][0]
    biased = scores + jnp.asarray(tensors["mlp.gate.e_score_correction_bias"])
    scale = config["routed_scaling_factor"]

    def gathered(source):
        picked = jnp.take_along_axis(source, indices, axis=-1)
        return picked / jnp.sum(picked, axis=-1, keepdims=True) * scale

    assert np.max(np.abs(np.asarray(weights) - np.asarray(gathered(scores)))) < 1e-6
    # And the same gather off the biased scores is the bug this rules out.
    assert np.max(np.abs(np.asarray(weights) - np.asarray(gathered(biased)))) > 0.01


def test_the_expert_block_reproduces_the_mixtral_block_output():
    """The whole sparse feed-forward, which is the router plus the grouped
    matmul plus the weighted sum, against the reference block."""
    tensors = fixture("mixtral")
    config = CONFIG["mixtral"]
    block = SparseMLP(num_experts=config["num_local_experts"],
                      top_k=config["num_experts_per_tok"],
                      hidden_features=config["intermediate_size"],
                      out_features=config["hidden_size"])
    output = block.apply(sparse_variables(tensors, config["num_local_experts"]),
                         jnp.asarray(tensors["hidden"]))

    # Largest observed difference 7.63e-06 at fp32 on CPU, on outputs that
    # reach 24.7, so 3.1e-07 of the scale. The reference sums each expert's
    # contribution into a zeroed buffer while the ragged path sums a token's
    # k slots, so the two differ in summation order alone.
    assert np.max(np.abs(np.asarray(output) - tensors["block_output"])) < OUTPUT_TOLERANCE


def deepseek_block(shared: bool = True) -> SparseMLP:
    """`DeepseekV3MoE` as a SparseMLP: its router, experts and shared branch."""
    config = CONFIG["deepseek"]
    width = config["moe_intermediate_size"]
    return SparseMLP(
        num_experts=config["n_routed_experts"],
        top_k=config["num_experts_per_tok"],
        hidden_features=width, out_features=config["hidden_size"],
        score_function='sigmoid',
        routed_scaling_factor=config["routed_scaling_factor"],
        expert_groups=config["n_group"], groups_per_token=config["topk_group"],
        expert_bias=True,
        shared=None if not shared else functools.partial(
            GatedMLP, hidden_features=width * config["n_shared_experts"],
            out_features=config["hidden_size"]))


def deepseek_variables(tensors, shared: bool = True) -> dict:
    config = CONFIG["deepseek"]
    variables = sparse_variables(tensors, config["n_routed_experts"])
    variables["moe"] = {"gate": {"e_score_correction_bias": jnp.asarray(
        tensors["mlp.gate.e_score_correction_bias"])}}
    if shared:
        variables["params"]["shared_experts"] = {
            projection: {"kernel": jnp.asarray(
                tensors[f"mlp.shared_experts.{projection}.weight"].T)}
            for projection in ("gate_proj", "up_proj", "down_proj")}
    return variables


def test_the_expert_block_reproduces_the_deepseek_block_output():
    """`DeepseekV3MoE` end to end: the group-limited router, the routed
    experts scaled by 2.5, and the shared expert every token takes, summed.

    Tolerance 2e-5; largest observed difference 7.63e-06 at fp32 on CPU, on
    outputs that reach 34.9, the same summation-order spread as Mixtral's.
    """
    tensors = fixture("deepseek")
    output = deepseek_block().apply(deepseek_variables(tensors),
                                    jnp.asarray(tensors["hidden"]))
    assert np.max(np.abs(np.asarray(output) - tensors["block_output"])) < OUTPUT_TOLERANCE


def v4_router(**overrides) -> Router:
    config = CONFIG["deepseek_v4"]
    settings = {"score_function": 'sqrtsoftplus',
                    "routed_scaling_factor": config["routed_scaling_factor"],
                    "expert_bias": True}
    settings.update(overrides)
    return Router(num_experts=config["num_local_experts"],
                  in_features=config["hidden_size"],
                  top_k=config["num_experts_per_tok"], **settings)


@pytest.mark.parametrize(("name", "router"), [
    pytest.param("deepseek", deepseek_router, id="v3-group-limited"),
    pytest.param("deepseek_v4", v4_router, id="v4-sqrt-softplus")])
def test_router_reproduces_the_deepseek_choice(name, router):
    """DeepseekV3MoE's router (sigmoid scores, a per-expert selection bias, the
    node limit over expert groups) and `DeepseekV4TopKRouter`'s (sqrt(softplus)
    scores, the selection bias), each renormalising and scaling: largest
    observed weight differences 2.38e-07 and 1.19e-07 at fp32 on CPU, on
    weights the routed scaling factor has already multiplied."""
    tensors = fixture(name)
    weights, indices = router().apply(router_variables(tensors, bias=True), jnp.asarray(tensors["hidden"]))
    assert_routes_as_recorded(tensors, weights, indices)


def test_a_sqrt_softplus_routers_gradient_is_finite_where_softplus_underflows():
    """DeepSeek V4's router scores `sqrt(softplus(logits))`: where an expert's logit takes softplus below
    float32's least normal, the plain formula's derivative, `sqrt`'s at zero, is infinite, and the
    router's gradient came out NaN from an unselected expert's zero cotangent (row 3919); its score
    there is the formula's tail (`test_the_sqrt_softplus_score_is_the_formula_and_its_tail`), and the
    gradient finite."""
    router = v4_router(expert_bias=False)
    hidden = jax.random.normal(jax.random.key(0), (4, CONFIG["deepseek_v4"]["hidden_size"]))
    variables = router.init(jax.random.key(1), hidden)
    # Logits far down for every expert but a few, softplus underflowing for most.
    pushed = jax.tree.map(lambda leaf: leaf * 4000.0, variables)

    def routed(variables):
        weights, _ = router.apply(variables, hidden)
        return jnp.sum(weights)

    gradient = jax.grad(routed)(pushed)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradient))


def test_the_sqrt_softplus_score_is_the_formula_and_its_tail():
    """`moe.sqrt_softplus` is the plain `sqrt(softplus(logits))` bit for bit wherever `exp(logits)` is
    above float32's epsilon, and below it the formula's tail `exp(logits / 2)`, the formula's to within
    that epsilon: an expert down the tail keeps a positive score where the plain formula's is zero."""
    from dew.nn import moe

    logits = jnp.asarray([-170.0, -110.0, -40.0, -16.0, -15.0, 0.0, 30.0], jnp.float32)
    scores, plain = np.asarray(moe.sqrt_softplus(logits)), np.asarray(jnp.sqrt(jax.nn.softplus(logits)))
    above = np.asarray(logits) >= np.log(np.finfo(np.float32).eps)
    np.testing.assert_array_equal(scores[above], plain[above])
    assert np.all(scores > 0) and not np.all(plain > 0)


@pytest.mark.parametrize("fill", [0.0, None], ids=["axk2-zero", "deepseek-inf"])
def test_a_router_chooses_an_expert_outside_its_best_groups_by_the_score_its_reference_fills_in(fill):
    """A grouped router whose best group's experts all choose below zero: A.X-K2's fills every expert
    outside that group at 0.0 (modeling_axk2.py:340), so those outrank the group's own, and DeepSeek V3's
    at -inf, so they never do (`Router.excluded_group_score`). Of eight experts in four groups, two
    groups are a token's best and the four experts outside them are its choice, or never are."""
    import torch
    from transformers import AutoConfig
    from transformers.models.axk2.modeling_axk2 import AXK2TopkRouter
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    fields = {"hidden_size": 16, "n_routed_experts": 8, "num_experts_per_tok": 4, "n_group": 4,
              "topk_group": 2, "routed_scaling_factor": 1.0, "norm_topk_prob": True}
    reference = (AXK2TopkRouter(AutoConfig.for_model("axk2", **fields)) if fill is not None
                 else DeepseekV3TopkRouter(AutoConfig.for_model("deepseek_v3", **fields)))
    generator = np.random.default_rng(0x3080)
    weight = generator.standard_normal((8, 16)).astype(np.float32)
    # Every expert's score for choice falls below zero, the excluded experts' fill above it or not.
    bias = np.full((8,), -5.0, np.float32)
    hidden = generator.standard_normal((6, 16)).astype(np.float32)
    reference.weight.data = torch.from_numpy(weight)
    reference.e_score_correction_bias.data = torch.from_numpy(bias)
    with torch.no_grad():
        _, _, theirs = reference(torch.from_numpy(hidden))
    router = Router(num_experts=8, in_features=16, top_k=4, score_function='sigmoid',
                    normalize_weights=True, expert_groups=4, groups_per_token=2, expert_bias=True,
                    excluded_group_score=fill)
    variables = {"params": {"kernel": jnp.asarray(weight.T)},
                 "moe": {"e_score_correction_bias": jnp.asarray(bias)}}
    _, indices = router.apply(variables, jnp.asarray(hidden))
    np.testing.assert_array_equal(np.sort(np.asarray(indices), -1), np.sort(theirs.numpy(), -1))


def v4_experts(swiglu_limit) -> ExpertMLP:
    config = CONFIG["deepseek_v4"]
    return ExpertMLP(num_experts=config["num_local_experts"],
                     hidden_features=config["intermediate_size"],
                     out_features=config["hidden_size"], swiglu_limit=swiglu_limit)


def test_the_experts_reproduce_the_deepseek_v4_clamped_output():
    """`DeepseekV4Experts` on the router's choice: the gate capped at
    swiglu_limit from above and the up projection on both sides, then silu.

    Tolerance 2e-5; largest observed difference 4.77e-07 at fp32 on CPU, on
    outputs that reach 3.1; without the clamp the same weights are off by 25.
    """
    tensors = fixture("deepseek_v4")
    config = CONFIG["deepseek_v4"]
    variables = {"params": sparse_variables(
        tensors, config["num_local_experts"])["params"]["experts"]}
    hidden = jnp.asarray(tensors["hidden"]).reshape(-1, config["hidden_size"])
    output = v4_experts(config["swiglu_limit"]).apply(
        variables, hidden, jnp.asarray(tensors["router_weights"]),
        jnp.asarray(tensors["router_indices"]))
    assert np.max(np.abs(np.asarray(output) - tensors["experts_output"])) < OUTPUT_TOLERANCE


# --------------------------------------------------------------------------
# The aux-loss-free balancing bias
# --------------------------------------------------------------------------

def test_the_load_balance_update_pushes_against_the_busy_experts():
    """Lifted from maxtext layers/moe.py:238, checked on counts by hand: over
    four experts, three tokens on expert 0 and one on expert 1 averages one
    per expert, so expert 0 loses bias, expert 1 holds and the idle two gain."""
    indices = jnp.asarray([[[0, 0], [0, 1]]])
    counts = jnp.sum(jax.nn.one_hot(indices.ravel(), 4, dtype=jnp.int32), axis=0)
    update = load_balance_update(counts, rate=0.001)

    assert update.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(update), np.array([-0.001, 0.0, 0.001, 0.001], np.float32))


# --------------------------------------------------------------------------
# The grouped matmul
# --------------------------------------------------------------------------

def routed_experts(num_experts=8, top_k=1, tokens=6, width=8, hidden=12,
                   implementation='xla'):
    """An ExpertMLP with routing that leaves some experts idle."""
    experts = ExpertMLP(num_experts=num_experts, hidden_features=hidden,
                        out_features=width, implementation=implementation)
    x = jax.random.normal(jax.random.key(0), (tokens, width))
    indices = jnp.asarray(
        np.arange(tokens * top_k).reshape(tokens, top_k) % 3, jnp.int32)
    weights = jnp.full((tokens, top_k), 1.0 / top_k, jnp.float32)
    variables = nn.unbox(experts.init(jax.random.key(1), x, weights, indices))
    return experts, variables, x, weights, indices


def test_an_expert_no_token_reached_cannot_change_the_output():
    """Group sizes and the sort have to agree: if a token's rows land in the
    wrong expert's group, an idle expert's weights start showing up."""
    experts, variables, x, weights, indices = routed_experts()
    baseline = experts.apply(variables, x, weights, indices)
    used = set(np.asarray(indices).ravel().tolist())

    for expert in range(8):
        zeroed = {"params": {name: {"kernel": value["kernel"].at[expert].set(0.0)}
                             for name, value in variables["params"].items()}}
        output = experts.apply(zeroed, x, weights, indices)
        changed = not np.array_equal(np.asarray(output), np.asarray(baseline))
        assert changed == (expert in used), expert


def test_the_grouped_matmul_matches_a_per_expert_loop():
    """jax.lax.ragged_dot over sorted tokens against the same contractions
    written out one expert at a time."""
    experts, variables, x, weights, indices = routed_experts(top_k=2)
    output = experts.apply(variables, x, weights, indices)

    kernels = {name: np.asarray(value["kernel"])
               for name, value in variables["params"].items()}
    reference = np.zeros(x.shape, np.float32)
    for token in range(x.shape[0]):
        row = np.asarray(x[token])
        for slot in range(indices.shape[-1]):
            expert = int(indices[token, slot])
            gate = jax.nn.silu(row @ kernels["gate_proj"][expert])
            hidden = np.asarray(gate) * (row @ kernels["up_proj"][expert])
            reference[token] += float(weights[token, slot]) * (
                hidden @ kernels["down_proj"][expert])

    # Largest observed difference 1.19e-07 at fp32 on CPU.
    assert np.max(np.abs(np.asarray(output) - reference)) < 1e-6


def test_float32_expert_routing_with_x64_keeps_its_output_and_gradients():
    """Enabling a float64 oracle must not prevent the float32 model running.

    bincount defaults to int64 under x64; TPU ragged-dot cannot lower those
    group sizes. Exercise the routed experts, not just a hand-typed count.
    """
    with jax.enable_x64(new_val=False):
        experts, variables, x, weights, indices = routed_experts(top_k=2)

    def step(variables, x):
        output = experts.apply(variables, x, weights, indices)
        return jnp.square(output).sum(), output

    results = []
    for enabled in (False, True):
        with jax.enable_x64(enabled):
            results.append(jax.device_get(jax.jit(
                jax.value_and_grad(step, argnums=(0, 1), has_aux=True))(variables, x)))
    for expected, actual in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
        assert actual.dtype == expected.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
