"""Quantized training through Qwix: the value, the wrapping, the training.

Qwix-dependent tests skip without the package; the value's refusals run
everywhere, since construction never imports it.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew import models  # noqa: F401  registers the models
from dew.config import OptimConfig, _rebuild
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.training.distributed import Layout, MeshSpec, build_mesh, shard_batch
from dew.training.optim import build_optimizer
from dew.training.quantization import Quantization, apply

VOCAB = 64
SEQ_LEN = 8
BATCH = 4
TINY = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
            num_kv_heads=2, mlp_features=64, max_seq_len=16)


def tiny(**overrides):
    return models.build("causal_transformer", **{**TINY, **overrides})


def token_batch():
    rng = np.random.default_rng(0)
    return {"text": rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}


def test_fp8_full_is_refused_for_want_of_a_calibration_pass():
    with pytest.raises(ValueError, match="calibration pass"):
        Quantization(dtype="fp8_full")


def test_nanoo_fp8_is_refused_as_amd_only():
    with pytest.raises(ValueError, match="AMD"):
        Quantization(dtype="nanoo_fp8")


def test_a_value_that_says_nothing_is_refused():
    with pytest.raises(ValueError, match="no patterns"):
        Quantization(patterns=())
    with pytest.raises(ValueError, match="does not compile"):
        Quantization(patterns=(".*(unclosed",))
    with pytest.raises(ValueError, match="calibration"):
        Quantization(calibration="percentile,99")
    with pytest.raises(ValueError, match="tile_size"):
        Quantization(tile_size=0)
    with pytest.raises(ValueError, match="bwd_qtype"):
        Quantization(bwd_qtype="int4")
    with pytest.raises(ValueError, match="bwd_stochastic_rounding"):
        Quantization(bwd_stochastic_rounding="gaussian")
    with pytest.raises(ValueError, match="int8 or fp8"):
        Quantization(dtype="int4")


def test_apply_refuses_a_non_value():
    with pytest.raises(ValueError, match="a Quantization value"):
        apply(tiny(), {"dtype": "int8"})


def test_the_value_round_trips_through_json():
    spec = Quantization(dtype="fp8", patterns=(".*mlp.*", ".*self_attn.*"),
                        calibration="absmax,0.8", tile_size=32,
                        bwd_qtype="int8", bwd_stochastic_rounding="uniform")
    record = json.loads(json.dumps(dataclasses.asdict(spec)))
    assert _rebuild(Quantization, record) == spec


def quantized_forward(spec, **overrides):
    model = tiny(**overrides)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    qmodel = apply(model, spec)
    ids = jnp.asarray(token_batch()["text"][:, :SEQ_LEN])
    return model, qmodel, variables, ids


def test_the_wrapped_forward_matches_qwixs_own_call():
    """Dew's wrapping against the provider built by hand: identical logits,
    and both away from the fp32 forward. The gap is the assertion: rules
    that never reached a matmul would leave the fp32 numerics bitwise.
    Observed on CPU: wrapped and hand-built bitwise equal, 1.2e-01 from
    fp32 on logits of order 3."""
    qwix = pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(Quantization())
    rules = [qwix.QtRule(module_path=".*", weight_qtype=jnp.int8,
                         act_qtype=jnp.int8)]
    reference = qwix.quantize_model(model, qwix.QtProvider(rules))
    plain = model.apply(variables, ids)
    wrapped = qmodel.apply(variables, ids)
    manual = reference.apply(variables, ids)
    assert float(jnp.max(jnp.abs(wrapped - manual))) == 0.0
    assert float(jnp.max(jnp.abs(wrapped - plain))) > 1e-2


def test_a_pattern_that_matches_nothing_quantizes_nothing():
    """A narrowed pattern is selective: a regex no module path matches runs
    the fp32 numerics bitwise. A pattern that matched everything would move
    the logits here, so the equality fails it. Observed on CPU: bitwise
    equal."""
    pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(
        Quantization(patterns=(".*does_not_exist.*",)))
    assert float(jnp.max(jnp.abs(
        qmodel.apply(variables, ids) - model.apply(variables, ids)))) == 0.0


def test_an_int8_trunk_trains_with_finite_loss():
    """Five adamw steps through the int8 trunk: every loss finite and the
    parameters moved. Observed on CPU: 5.17 down to 4.07."""
    pytest.importorskip("qwix")
    model, qmodel, variables, _ = quantized_forward(Quantization())
    objective = LMObjective(qmodel, SEQ_LEN)
    batch = token_batch()
    solver = build_optimizer(OptimConfig(learning_rate=1e-3), 5)
    params, opt_state = variables["params"], solver.init(variables["params"])

    @jax.jit
    def train(params, opt_state, key):
        (loss, _), grads = jax.value_and_grad(
            lambda p: objective.loss(
                {**variables, "params": p}, batch,
                Step(step=jnp.zeros((), jnp.int32), key=key, ema=None)),
            has_aux=True)(params)
        updates, opt_state = solver.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = []
    key = jax.random.key(0)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = train(params, opt_state, subkey)
        losses.append(float(loss))
    assert all(np.isfinite(losses)), losses
    assert losses[0] != losses[-1]


def test_a_scanned_quantized_stack_scores_as_the_plain_one():
    """Quantization composes with the scan. Both stacks compiled as a
    training step compiles them, the wrapped scan agrees with the wrapped
    plain loop bitwise, while staying away from its fp32 twin; the distance
    is the assertion that the rules reached under the scan. Observed on
    CPU: 0.0 and 1.2e-01 on logits of order 3. (Eager the two differ by
    3.6e-02, fusion order in the uncompiled matmuls, so both sides compile
    here.)"""
    pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(
        Quantization(), scan_layers=True)
    plain_wrapped = apply(tiny(), Quantization())
    scanned = jax.jit(qmodel.apply)(variables, ids)
    plain = jax.jit(plain_wrapped.apply)(variables, ids)
    reference = jax.jit(model.apply)(variables, ids)
    assert float(jnp.max(jnp.abs(scanned - reference))) > 1e-2
    assert float(jnp.max(jnp.abs(scanned - plain))) == 0.0


@pytest.mark.mesh
def test_a_quantized_pipeline_has_finite_loss_and_gradients():
    """The int8 trunk over two stages on the eight simulated devices: finite
    loss and gradients, and the loss within int8 rounding of the plain
    wrapped loop's, the way the bf16 pipeline test holds its policy's
    rounding. Observed on CPU: 1.6e-03 on a loss of order 5."""
    pytest.importorskip("qwix")
    model = tiny(num_layers=4)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    qmodel = apply(model, Quantization())
    rows = np.random.default_rng(0).integers(0, VOCAB, size=(8, SEQ_LEN + 1)).astype(np.int32)
    batch = {"text": rows}

    def run(spec):
        objective = LMObjective(qmodel, SEQ_LEN)
        mesh = build_mesh(spec)
        placed = jax.device_put(
            variables, Layout(min_shard=256).shardings(mesh, variables))
        placed_batch = shard_batch(mesh, batch)
        info = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3),
                    ema=None)

        def loss(params):
            return objective.loss({**variables, "params": params},
                                  placed_batch, info)

        with jax.set_mesh(mesh), pipeline_microbatches(spec.microbatches):
            (value, _), grads = jax.jit(
                jax.value_and_grad(loss, has_aux=True))(placed["params"])
        return float(value), jax.tree.map(np.asarray, grads)

    loss, grads = run(MeshSpec(fsdp=8))
    piped, piped_grads = run(MeshSpec(fsdp=4, stage=2, microbatches=2))
    assert np.isfinite(loss) and np.isfinite(piped)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(piped_grads))
    assert abs(loss - piped) < 1e-2 * max(1.0, abs(loss)), (loss, piped)


def test_stochastic_rounding_draws_from_its_own_stream():
    """The rounding option runs when the apply hands Qwix its stream, with
    finite gradients. Observed on CPU: finite."""
    pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(
        Quantization(bwd_qtype="int8", bwd_stochastic_rounding="uniform"))
    rngs = {"stochastic_rounding": jax.random.key(0)}

    def loss(params):
        hidden = qmodel.apply({**variables, "params": params}, ids, rngs=rngs,
                              method=type(qmodel).hidden_states)
        return jnp.mean(hidden ** 2)

    value, grads = jax.jit(jax.value_and_grad(loss))(variables["params"])
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))
