"""DeepSeek V2's router and balance loss against their references.

The router fixture comes from tools/moe_reference.py (`DeepseekV2TopkRouter`
under group_limited_greedy with no renormalisation); the balance loss is
checked against the released `MoEGate` equations by hand and then through
`LMObjective`, which is where a training run adds it.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.nn.moe import Router, deepseek_v2_aux_loss
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective, TEXT_KEY

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "moe"
CONFIG = json.loads((FIXTURES / "config.json").read_text())["deepseek_v2"]


def fixture() -> dict:
    with np.load(FIXTURES / "deepseek_v2.npz") as data:
        return {key: np.asarray(value) for key, value in data.items()}


def by_expert(indices, weights):
    order = np.argsort(np.asarray(indices), axis=-1)
    return (np.take_along_axis(np.asarray(indices), order, axis=-1),
            np.take_along_axis(np.asarray(weights), order, axis=-1))


def v2_router(normalize_weights: bool = CONFIG["norm_topk_prob"],
              group_score: str = "max") -> Router:
    return Router(num_experts=CONFIG["n_routed_experts"], in_features=CONFIG["hidden_size"],
                  top_k=CONFIG["num_experts_per_tok"], score_function="softmax",
                  normalize_weights=normalize_weights,
                  routed_scaling_factor=CONFIG["routed_scaling_factor"],
                  expert_groups=CONFIG["n_group"], groups_per_token=CONFIG["topk_group"],
                  group_score=group_score)


def router_variables(tensors) -> dict:
    return {"params": {"kernel": jnp.asarray(tensors["mlp.gate.weight"].T)}}


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


def test_v2_parity_needs_the_unnormalised_weights_and_the_max_group_score():
    """The V3 rules on the same weights disagree: renormalising sums each
    token's gate values to 2.5, and scoring a group by its two best experts
    changes which groups win on some rows."""
    tensors = fixture()
    hidden = jnp.asarray(tensors["hidden"]).reshape(-1, CONFIG["hidden_size"])
    weights, _ = v2_router(normalize_weights=True).apply(router_variables(tensors), hidden)
    assert np.allclose(np.sum(np.asarray(weights), axis=-1), 2.5, atol=1e-5)
    assert not np.allclose(np.sum(tensors["router_weights"], axis=-1), 2.5, atol=1e-3)
    _, indices = v2_router(group_score="top2").apply(router_variables(tensors), hidden)
    theirs = np.sort(tensors["router_indices"], axis=-1)
    assert not np.array_equal(np.sort(np.asarray(indices), axis=-1), theirs)


def test_the_balance_loss_matches_the_released_gate_equations():
    """One sequence of three tokens over four experts with top-1 routing:
    f = (2, 1, 0, 0) slots / (3 * 1 / 4), P the mean score, loss alpha *
    sum(f * P). The batch form averages the sequences; the flat form pools
    the tokens, and the two disagree once the sequences differ."""
    scores = jnp.asarray([[[0.7, 0.1, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]],
                          [[0.1, 0.1, 0.1, 0.7], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]])
    indices = jnp.asarray([[[0], [0], [1]], [[3], [2], [3]]])
    first = np.array([2, 1, 0, 0]) / (3 / 4) * np.mean(np.asarray(scores[0]), axis=0)
    second = np.array([0, 0, 1, 2]) / (3 / 4) * np.mean(np.asarray(scores[1]), axis=0)
    expected = 0.5 * (first.sum() + second.sum()) * 0.01
    assert float(deepseek_v2_aux_loss(scores, indices, 0.01, seq_aux=True)) == pytest.approx(expected)
    pooled = np.array([2, 1, 1, 2]) / 6 * 4 * np.mean(np.asarray(scores).reshape(-1, 4), axis=0)
    assert float(deepseek_v2_aux_loss(scores, indices, 0.01, seq_aux=False)) == pytest.approx(
        pooled.sum() * 0.01)
    assert float(deepseek_v2_aux_loss(scores, indices, 0.01, seq_aux=False)) != pytest.approx(expected)


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
    balanced = LMObjective(model, 8, aux_loss_alpha=0.05, seq_aux=False)
    params = plain.init(jax.random.PRNGKey(0))
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)
    base, _ = plain.loss(params, {TEXT_KEY: tokens}, step)
    loss, aux = balanced.loss(params, {TEXT_KEY: tokens}, step)

    _, sown = model.apply(params, tokens[:, :-1], mutable=["router"],
                          method=type(model).hidden_states)
    expected = sum(
        deepseek_v2_aux_loss(sown["router"][layer]["mlp"]["gate"]["scores"][0],
                             sown["router"][layer]["mlp"]["gate"]["indices"][0], 0.05, False)
        for layer in ("layers_0", "layers_1"))
    assert float(loss - base) == pytest.approx(float(expected), rel=1e-5)
    assert float(aux.metrics["aux_loss"]) == pytest.approx(float(expected), rel=1e-5)
    grads = jax.grad(lambda p: balanced.loss(p, {TEXT_KEY: tokens}, step)[0] - plain.loss(
        p, {TEXT_KEY: tokens}, step)[0])(params)
    assert float(jnp.abs(grads["params"]["layers_0"]["mlp"]["gate"]["kernel"]).max()) > 0
