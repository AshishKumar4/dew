"""Router losses and bias effects over the real queries of packed documents."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import unfreeze
from model_support import decoder

from dew import Evaluation
from dew.batch import VALID_ROWS, coordinates
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.objectives.routing import ExpertBalance, RoutedStatistics


@pytest.fixture(scope="module", params=[0, 1], ids=["trunk", "mtp"])
def routed(request):
    model = decoder().clone(mixture=Mixture(experts=2, top_k=1, bias=True), qk_norm=False,
                            num_nextn_predict_layers=request.param)
    variables = unfreeze(dict(LMObjective(model, 8).init(jax.random.key(0))))
    params = variables["params"]
    embedding = jnp.zeros_like(params["embed_tokens"]["embedding"])
    params["embed_tokens"]["embedding"] = (embedding.at[:, 1].set(1)
                                           .at[1, 0].set(1).at[2, 0].set(-1))
    layer = params["layers_0"]
    layer["self_attn"] = jax.tree.map(jnp.zeros_like, layer["self_attn"])
    kernel = jnp.zeros_like(layer["mlp"]["gate"]["kernel"])
    layer["mlp"]["gate"]["kernel"] = kernel.at[0].set(jnp.asarray([0.5, -0.5]))
    return model, variables, {"mtp_weight": 0.3} if request.param else {}


def step(*, training=True):
    return Step(step=jnp.asarray(0), key=jax.random.key(3), ema=None, training=training)


def padded(kind, extra, *, empty=False):
    fill = 0 if kind == "pad_id" else 3
    ids = jnp.asarray([[fill] * (5 + extra) if empty else [1, 2, 2, 2] + [fill] * (1 + extra)])
    valid = jnp.zeros_like(ids, bool) if empty else jnp.arange(ids.shape[1])[None, :] < 4
    batch: dict = {TEXT_KEY: ids}
    if kind == "attention_mask":
        batch[TEXT_KEY] = ModelInputs(ids, {"attention_mask": valid})
    elif kind == "segments":
        batch["text_coordinates"] = coordinates(segments=valid.astype(jnp.int32), max_segments=1,
                                                 right_padded=True)
    elif kind == "coordinates.valid":
        batch["text_coordinates"] = coordinates(valid=valid, right_padded=True)
    return batch, 0 if kind == "pad_id" else None


@pytest.mark.parametrize("seq_aux", [False, True])
@pytest.mark.parametrize("kind", ["pad_id", "attention_mask", "segments", "coordinates.valid"])
def test_router_terms_and_bias_updates_ignore_appended_padding(routed, seq_aux, kind):
    model, variables, depths = routed
    outcomes = []
    for extra in (0, 4):
        batch, pad_id = padded(kind, extra)
        balance = ExpertBalance(aux_loss_alpha=0.2, seq_aux=seq_aux, router_z_loss=0.1, balance_rate=0.05)
        objective = LMObjective(model, 4 + extra, balance=balance, pad_id=pad_id, **depths)
        _, aux = objective.scalar_loss(variables, batch, step())
        assert aux.effects is not None
        outcomes.append((aux, objective.apply_effects(variables, aux.effects)))
    first, last = outcomes
    for name in ("aux_loss", "router_z_loss", "moe/max_load", "moe/min_load"):
        np.testing.assert_allclose(first[0].metrics[name], last[0].metrics[name], rtol=1e-5, atol=1e-6)
    for before, after in zip(jax.tree.leaves(first[0].effects), jax.tree.leaves(last[0].effects),
                             strict=True):
        assert jnp.issubdtype(before.dtype, jnp.integer)
        np.testing.assert_array_equal(before, after)
    for before, after in zip(jax.tree.leaves(first[1]), jax.tree.leaves(last[1]), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("seq_aux", [False, True])
@pytest.mark.parametrize("kind", ["pad_id", "attention_mask", "segments"])
def test_all_padding_has_no_router_support_or_bias_effect(routed, seq_aux, kind):
    model, variables, depths = routed
    batch, pad_id = padded(kind, 0, empty=True)
    balance = ExpertBalance(aux_loss_alpha=0.2, seq_aux=seq_aux, router_z_loss=0.1, balance_rate=0.05)
    objective = LMObjective(model, 4, balance=balance, pad_id=pad_id, **depths)
    statistics, aux = objective.loss(variables, batch, step())
    value, supported = objective.reduce_loss(statistics)
    assert value.shape == supported.shape == ()
    assert not bool(supported)
    assert float(value) == 0
    assert all(np.isfinite(float(value)) for value in aux.metrics.values())
    assert aux.effects is not None
    assert all(not np.any(count) for count in jax.tree.leaves(aux.effects))
    moved = objective.apply_effects(variables, aux.effects)
    for before, after in zip(jax.tree.leaves(variables["moe"]), jax.tree.leaves(moved["moe"]),
                             strict=True):
        np.testing.assert_array_equal(before, after)


def test_global_router_validation_is_scalar_and_ignores_repeated_rows(routed):
    model, variables, depths = routed
    objective = LMObjective(model, 8, balance=ExpertBalance(aux_loss_alpha=0.2, seq_aux=False), **depths)
    batch = {TEXT_KEY: jnp.asarray([[1, 1, 2, 2, 2, 3, 3, 3, 3],
                                   [2, 2, 1, 1, 1, 3, 3, 3, 3],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1]]),
             "text_coordinates": coordinates(segments=jnp.asarray([[1, 1, 2, 2, 2, 0, 0, 0, 0],
                                                                   [1, 1, 2, 2, 2, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                                               max_segments=2, right_padded=True)}
    real = jax.tree.map(lambda value: value[:2], batch)
    expected, _ = objective.scalar_loss(variables, real, step(training=False))
    batch = jax.tree.map(lambda value: value[jnp.asarray([0, 1, 2, 2, 2, 2, 2, 2])], batch)
    batch[VALID_ROWS] = jnp.arange(8) < 2
    result = Evaluation.run(objective, variables, lambda partition: iter([batch]), key=3, loss=True)
    assert result.records == 2
    assert result.scores["val/loss"] == pytest.approx(float(expected), rel=1e-5)
    statistics, _ = objective.loss(variables, batch, step(training=False))
    assert isinstance(statistics, RoutedStatistics)
    value, supported = objective.reduce_loss(statistics)
    assert value.shape == supported.shape == ()
    assert all(term.positions.shape == () for term in statistics.global_routers)
    assert float(statistics.global_routers[0].positions) == 10


def test_sequence_balance_normalizes_each_packed_document(routed):
    model, variables, depths = routed
    packed = {TEXT_KEY: jnp.asarray([[1, 1, 1, 2, 2, 2, 2, 2, 0]]),
              "text_coordinates": coordinates(segments=jnp.asarray([[1, 1, 1, 2, 2, 2, 2, 2, 0]]),
                                                max_segments=2, right_padded=True)}
    separate = {TEXT_KEY: jnp.asarray([[1, 1, 1, 0, 0, 0], [2, 2, 2, 2, 2, 0]]),
                "text_coordinates": coordinates(segments=jnp.asarray([[1, 1, 1, 0, 0, 0],
                                                                       [1, 1, 1, 1, 1, 0]]),
                                                  max_segments=1, right_padded=True)}
    readings = []
    for batch in (packed, separate):
        objective = LMObjective(model, batch[TEXT_KEY].shape[1] - 1, pad_id=0, **depths,
                                balance=ExpertBalance(aux_loss_alpha=0.2, seq_aux=True))

        def balance(params, objective=objective, batch=batch):
            return objective.scalar_loss(params, batch, step())[1].metrics["aux_loss"]

        readings.append(jax.value_and_grad(balance)(variables))
        statistics, _ = objective.loss(variables, batch, step())
        assert isinstance(statistics, RoutedStatistics)
        assert all(float(term.mass) == 2 for term in statistics.sequence)
    assert float(readings[0][0]) > 0.35
    np.testing.assert_allclose(readings[0][0], readings[1][0], rtol=1e-5, atol=1e-6)
    for packed_grad, separate_grad in zip(jax.tree.leaves(readings[0][1]),
                                           jax.tree.leaves(readings[1][1]), strict=True):
        np.testing.assert_allclose(packed_grad, separate_grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("term", [
    {"balance_rate": 0.0}, {"balance_rate": -0.1}, {"balance_rate": float("inf")},
    {"balance_rate": float("nan")}, {"aux_loss_alpha": 0.0}, {"aux_loss_alpha": -0.2},
    {"aux_loss_alpha": float("nan")}, {"router_z_loss": -0.1}, {"router_z_loss": float("inf")},
    {"router_z_loss": float("nan")},
])
def test_expert_balance_refuses_a_term_it_cannot_weight(term):
    """A balance rate and a balance loss weight are positive and finite, and the z-loss weight finite and
    nonnegative, so a term that cannot carry its weight is refused at construction. The valid setup
    differs from the refused one only in that term."""
    valid = {"balance_rate": 0.01, "aux_loss_alpha": 0.1, "router_z_loss": 0.0}
    ExpertBalance(**valid)
    with pytest.raises(ValueError):
        ExpertBalance(**{**valid, **term})
