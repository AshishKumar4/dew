"""DeepSeek V2's router and balance loss against their references.

The router fixture comes from tools/moe_reference.py (`DeepseekV2TopkRouter`
under group_limited_greedy with no renormalisation); the balance loss is
checked against the released `MoEGate` equations by hand and then through
`LMObjective`, where a training run adds it.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_arrays
from moe_support import by_expert, router_variables

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.moe import Router, global_router_loss
from dew.objectives.base import Step
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.objectives.routing import ExpertBalance, router_moments

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "moe"
CONFIG = json.loads((FIXTURES / "config.json").read_text())["deepseek_v2"]


def fixture() -> dict:
    return fixture_arrays(FIXTURES / "deepseek_v2.npz")


def v2_router(normalize_weights: bool = CONFIG["norm_topk_prob"],
              group_score: str = "max") -> Router:
    return Router(num_experts=CONFIG["n_routed_experts"], in_features=CONFIG["hidden_size"],
                  top_k=CONFIG["num_experts_per_tok"], score_function="softmax",
                  normalize_weights=normalize_weights,
                  routed_scaling_factor=CONFIG["routed_scaling_factor"],
                  expert_groups=CONFIG["n_group"], groups_per_token=CONFIG["topk_group"],
                  group_score=group_score)


def test_router_reproduces_the_deepseek_v2_group_limited_choice():
    """Softmax scores, the best-expert group score, top-k without
    renormalisation, scaled by 2.5. Tolerance 1e-6, observed 6.0e-07."""
    tensors = fixture()
    hidden = jnp.asarray(tensors["hidden"]).reshape(-1, CONFIG["hidden_size"])
    weights, indices = v2_router().apply(router_variables(tensors), hidden)
    theirs_indices, theirs_weights = by_expert(tensors["router_indices"], tensors["router_weights"])
    ours_indices, ours_weights = by_expert(indices, weights)
    assert np.array_equal(ours_indices, theirs_indices)
    assert np.max(np.abs(ours_weights - theirs_weights)) < 1e-6

    normalized, _ = v2_router(normalize_weights=True).apply(router_variables(tensors), hidden)
    assert np.allclose(np.sum(np.asarray(normalized), axis=-1), 2.5, atol=1e-5)
    assert not np.allclose(np.sum(tensors["router_weights"], axis=-1), 2.5, atol=1e-3)
    _, grouped = v2_router(group_score="top2").apply(router_variables(tensors), hidden)
    assert not np.array_equal(np.sort(np.asarray(grouped), axis=-1),
                              np.sort(tensors["router_indices"], axis=-1))


def test_the_objective_adds_every_sparse_layers_balance_loss():
    """Through LMObjective: the loss with aux_loss_alpha exceeds the loss
    without it by the balance loss of both sparse layers, computed from
    what their routers sowed, and the term moves the router kernels."""
    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=2, num_heads=2,
                              mlp_features=32, max_seq_len=8, qk_norm=False,
                              mixture=Mixture(experts=4, top_k=2, norm_topk_prob=False,
                                              score_function="softmax"))
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 9), 0, 32)
    plain = LMObjective(model, 8)
    balanced = LMObjective(model, 8, balance=ExpertBalance(aux_loss_alpha=0.05, seq_aux=False))
    params = plain.init(jax.random.PRNGKey(0))
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)
    base, _ = plain.scalar_loss(params, {TEXT_KEY: tokens}, step)
    loss, aux = balanced.scalar_loss(params, {TEXT_KEY: tokens}, step)

    _, sown = model.apply(params, tokens[:, :-1], mutable=["router"],
                          method=type(model).hidden_states)
    expected = sum(
        global_router_loss(router_moments(sown["router"][layer]["mlp"]["gate"]["scores"][0],
                                          sown["router"][layer]["mlp"]["gate"]["indices"][0], {}), 0.05)
        for layer in ("layers_0", "layers_1"))
    assert float(loss - base) == pytest.approx(float(expected), rel=1e-5)
    assert float(aux.metrics["aux_loss"]) == pytest.approx(float(expected), rel=1e-5)
    grads = jax.grad(
        lambda p: balanced.scalar_loss(p, {TEXT_KEY: tokens}, step)[0]
        - plain.scalar_loss(p, {TEXT_KEY: tokens}, step)[0]
    )(params)
    assert float(jnp.abs(grads["params"]["layers_0"]["mlp"]["gate"]["kernel"]).max()) > 0
