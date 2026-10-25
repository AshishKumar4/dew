"""Block recomputation preserves training, stateful collections and checkpoints.

JAX 0.11.1 CPU: with both models stepping from the same variables and
optimizer state, the two momentum-SGD steps observed maximum differences of
4.8e-7 in loss, 1.4e-5 in gradients and optimizer state, 6.0e-8 in updated
variables, 3.1e-5 in reported metrics and 7.6e-6 in per-head QK maxima,
across the eight shape/scan cases. Momentum SGD keeps update differences on
the gradient scale; Adam can amplify roundoff-size gradients near AltUp
zero-initialized scales. The same tests also run on JAX 0.10.2 and on one
RTX 4080 (without the four simulated-mesh cases). Cached decoding is bitwise
identical.

The named policies recompute the same block from more or fewer saved
residuals: against `full`, on the dense shape with MTP and dropout, the
loss is bitwise identical on CPU and differs by one float32 ulp (4.8e-7)
on an RTX 4080, and the gradients differ by at most 8.4e-7 over the
eighteen policy/scan cases (CPU, JAX 0.11.1), from the saved values
entering fusions the recomputed ones do not. The offloaded policies run on
the CPU backend, whose lowering of a host transfer is the identity, so
their residuals are typed in host memory space and compile to the same
program as their saved counterparts; the transfer itself needs a GPU or TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from model_support import largest_difference

from dew.checkpoints import Checkpoints
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.objectives.routing import ExpertBalance
from dew.training.distributed import Layout, MeshSpec, shard_batch
from dew.training.state import TrainState
from dew.training.transaction import write_back

SHAPES = {
    "dense": {"num_nextn_predict_layers": 1},
    "sparse": {"mixture": {"experts": 4, "top_k": 2, "bias": True},
               "num_nextn_predict_layers": 1},
    "shared": {"kv_shared_layers": (2, 3), "per_layer_input_dim": 8,
               "use_double_wide_mlp": True, "sandwich_norms": True},
    "altup": {"altup": {"num_inputs": 2}, "kv_shared_layers": (2, 3),
              "per_layer_input_dim": 8, "laurel_rank": 4},
    "mla": {"mixer": {"class": "mla", "fields": {"q_lora_rank": 8, "kv_lora_rank": 8,
                      "qk_nope_head_dim": 4, "qk_rope_head_dim": 4, "v_head_dim": 4}}},
}


def model_for(shape, **overrides):
    return CausalTransformer(
        vocab_size=32, emb_features=16, num_layers=4, num_heads=2,
        num_kv_heads=1, mlp_features=32, max_seq_len=16,
        **{"dropout_rate": 0.15, **SHAPES[shape], **overrides})


def batch():
    return {"text": jnp.asarray(np.random.default_rng(5).integers(1, 32, (8, 9)),
                                jnp.int32)}


def objective_for(model):
    sparse = model.mixture is not None
    return LMObjective(model, 8,
                       balance=ExpertBalance(balance_rate=0.01, aux_loss_alpha=0.001) if sparse else None,
                       mtp_weight=0.2 if model.num_nextn_predict_layers else None)


def training_step(model):
    """One committed step as the trainer takes it: the optimizer update, then
    sequential collection writes and the objective's deferred effects."""
    objective = objective_for(model)
    optimizer = optax.sgd(1e-3, momentum=0.9)

    def step(variables, state, tokens, key):
        info = Step(jnp.zeros((), jnp.int32), key, None, qk_stats=True)
        (loss, aux), grads = jax.value_and_grad(
            lambda params: objective.scalar_loss({**variables, "params": params},
                                       tokens, info),
            has_aux=True)(variables["params"])
        updates, state = optimizer.update(grads, state, variables["params"])
        moved = write_back({**variables, "params": optax.apply_updates(variables["params"], updates)},
                           aux.variables)
        if aux.effects is not None:
            moved = write_back(moved, objective.apply_effects(moved, aux.effects))
        return moved, state, loss, grads, aux

    return jax.jit(step), optimizer


def train_state(step, variables, opt_state, key):
    """A single-microbatch state after `step` committed updates."""
    count = jnp.asarray(step, jnp.int32)
    return TrainState(step=count, microstep=count, updates=count, variables=variables,
                      opt_state=opt_state, ema=None, key=key, scale=None,
                      window_size=jnp.asarray(1, jnp.int32))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("scan", [False, True])
def test_recomputed_blocks_train_the_same_model(shape, scan):
    """Dropout, MTP, MoE balancing and shared K/V retain their derivatives.

    Each step hands both models the same variables and optimizer state, the
    plain model's from the previous step, with a fresh key, and compares the
    committed step including the deferred router-bias effects. Independent
    chains cannot be compared past one step: recomputation changes rounding
    in the backward pass (3e-8 in the updated variables here), and one
    token's second and third router scores in layer 1 of the sparse model
    were 1.6e-7 apart at the second step, so that step routed the token to
    a different expert and moved the loss by 6e-4.
    """
    plain = model_for(shape, scan_layers=scan)
    remat = plain.clone(remat='full')
    tokens = batch()
    ids = tokens["text"][:, :-1]
    variables = plain.init(jax.random.key(0), ids)
    recomputed = remat.init(jax.random.key(0), ids)
    assert jax.tree.structure(variables) == jax.tree.structure(recomputed)
    assert largest_difference(variables, recomputed) == 0
    assert largest_difference(jax.jit(plain.apply)(variables, ids),
                      jax.jit(remat.apply)(variables, ids)) < 2e-5
    run, solver = training_step(plain)
    rerun, _ = training_step(remat)
    state = solver.init(variables["params"])
    original = variables
    for index in range(2):
        key = jax.random.fold_in(jax.random.key(4), index)
        moved, next_state, loss, grads, aux = run(variables, state, tokens, key)
        recomputed, other_state, other_loss, other_grads, other_aux = rerun(
            variables, state, tokens, key)
        assert abs(float(loss - other_loss)) < 2e-5
        assert largest_difference(grads, other_grads) < 3e-5
        assert largest_difference(moved, recomputed) < 2e-5
        assert largest_difference(next_state, other_state) < 2e-5
        assert largest_difference(aux.metrics, other_aux.metrics) < 1e-4
        assert largest_difference(aux.qk_stats, other_aux.qk_stats) < 1e-4
        if shape == "sparse":
            # Router slot counts are integers; recomputation routes identically.
            assert largest_difference(aux.effects, other_aux.effects) == 0
        variables, state = moved, next_state
    assert largest_difference(original["params"], variables["params"]) > 1e-4
    if shape == "sparse":
        assert largest_difference(original["moe"], variables["moe"]) > 0


@pytest.mark.parametrize("shape", ["dense", "shared", "altup"])
@pytest.mark.parametrize("scan", [False, True])
def test_remat_prefills_and_appends_the_same_cache(shape, scan):
    plain = model_for(shape, scan_layers=scan)
    remat = plain.clone(remat='full')
    ids = batch()["text"][:2]
    plain_vars = plain.init(jax.random.key(0), ids[:, :2], decode=True)
    remat_vars = remat.init(jax.random.key(0), ids[:, :2], decode=True)
    for start, end in [(0, 5), (5, 6), (6, 8)]:
        expected, cache = plain.apply(plain_vars, ids[:, start:end],
                                       decode=True, mutable=["cache"])
        actual, recomputed_cache = remat.apply(remat_vars, ids[:, start:end],
                                               decode=True, mutable=["cache"])
        assert largest_difference(actual, expected) == 0

        plain_vars = {**plain_vars, **cache}
        remat_vars = {**remat_vars, **recomputed_cache}


@pytest.mark.mesh
@pytest.mark.parametrize("shape", ["dense", "sparse"])
@pytest.mark.parametrize("scan", [False, True])
def test_recomputed_pipeline_preserves_updates_and_sown_values(shape, scan):
    plain = model_for(shape, scan_layers=scan, num_nextn_predict_layers=0)
    remat = plain.clone(remat='full')
    # Eight rows split four ways over data x fsdp hold two a device, which
    # two microbatches divide.
    mesh_spec = MeshSpec(fsdp=2, stage=2)
    mesh = mesh_spec.build()
    tokens = shard_batch(mesh, batch())
    variables = plain.init(jax.random.key(0), batch()["text"][:, :-1])
    variables = jax.device_put(variables, Layout(min_shard=16).shardings(mesh, variables))
    run, solver = training_step(plain)
    rerun, _ = training_step(remat)
    state = solver.init(variables["params"])
    with jax.set_mesh(mesh), pipeline_microbatches(2):
        left = run(variables, state, tokens, jax.random.key(7))
        right = rerun(variables, state, tokens, jax.random.key(7))
    assert abs(float(left[2] - right[2])) < 2e-5
    assert largest_difference(left[0], right[0]) < 2e-5
    assert largest_difference(left[3], right[3]) < 3e-5
    assert largest_difference(left[4].qk_stats, right[4].qk_stats) < 1e-4
    assert largest_difference(left[4].metrics, right[4].metrics) < 1e-4


@pytest.mark.parametrize("saved_remat", [None, 'full'])
def test_checkpoint_resumes_with_remat_switched(tmp_path, saved_remat):
    model = model_for("shared", scan_layers=True, remat=saved_remat)
    variables = model.init(jax.random.key(0), batch()["text"][:, :-1])
    run, optimizer = training_step(model)
    moved, opt_state, _, _, _ = run(
        variables, optimizer.init(variables["params"]), batch(), jax.random.key(3))
    state = train_state(1, moved, opt_state, jax.random.key(4))
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(1, state, None)
    checkpoints.wait()
    resumed_model = model.clone(remat='full' if saved_remat is None else None)
    template_vars = resumed_model.init(jax.random.key(8), batch()["text"][:, :-1])
    template = train_state(0, template_vars, optimizer.init(template_vars["params"]),
                           jax.random.key(0))
    template = jax.tree.map(lambda value: jax.ShapeDtypeStruct(
        value.shape, value.dtype, sharding=value.sharding), template)
    restored, position = checkpoints.restore(template)
    assert position is None
    assert int(restored.step) == 1 and int(restored.updates) == 1
    expected = run(state.variables, state.opt_state, batch(), state.key)
    resume, _ = training_step(resumed_model)
    actual = resume(restored.variables, restored.opt_state, batch(), restored.key)
    assert abs(float(actual[2] - expected[2])) < 2e-5
    assert largest_difference(actual[0], expected[0]) < 2e-5


NAMED = [name for name in REMAT_POLICIES if name != 'full']


def gradient_step(model, variables, key):
    objective = objective_for(model)

    def loss(params):
        info = Step(jnp.zeros((), jnp.int32), key, None, qk_stats=True)
        return objective.scalar_loss({**variables, "params": params}, batch(), info)

    return jax.jit(jax.value_and_grad(loss, has_aux=True))(variables["params"])


@pytest.mark.parametrize("policy", NAMED)
@pytest.mark.parametrize("scan", [False, True])
def test_named_policies_train_the_same_model_as_full(policy, scan):
    """Saved and offloaded residuals feed the backward pass the values the
    recomputation would have produced, dropout and MTP included."""
    full = model_for("dense", scan_layers=scan, remat='full')
    variables = full.init(jax.random.key(0), batch()["text"][:, :-1])
    key = jax.random.key(4)
    (loss, aux), grads = gradient_step(full, variables, key)
    (other_loss, other_aux), other_grads = gradient_step(
        full.clone(remat=policy), variables, key)
    assert abs(float(loss - other_loss)) < 2e-6
    assert largest_difference(grads, other_grads) < 2e-6
    for name, value in aux.metrics.items():
        other = other_aux.metrics[name]
        if name == "perplexity":
            # Compare in cross-entropy units; exp magnifies the same rounding.
            np.testing.assert_allclose(np.log(value), np.log(other), atol=2e-6, rtol=0)
        else:
            np.testing.assert_allclose(value, other, atol=1e-5, rtol=0)
