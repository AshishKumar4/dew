"""Block recomputation preserves training, stateful collections and checkpoints.

JAX 0.11.1 CPU: the two-step momentum-SGD comparisons observed maximum
differences of 4.8e-7 in loss, 2.4e-5 in gradients, 1.2e-7 in updated
variables and 7.7e-6 in per-head QK maxima, across the eight shape/scan
cases. Momentum SGD keeps update differences on the gradient scale; Adam
can amplify roundoff-size gradients near AltUp zero-initialized scales.
The same tests also run on JAX 0.10.2 and on one RTX 4080 (without the
four simulated-mesh cases). Cached decoding is bitwise identical.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.checkpoints import Checkpoints
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.sharding import pipeline_microbatches
from dew.objectives import scalar_loss
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.training.distributed import Layout, MeshSpec, build_mesh, shard_batch
from dew.training.state import TrainState


SHAPES = {
    "dense": {"num_nextn_predict_layers": 1},
    "sparse": {"mixture": {"experts": 4, "top_k": 2, "bias": True},
               "num_nextn_predict_layers": 1},
    "shared": {"num_kv_shared_layers": 2, "per_layer_input_dim": 8,
               "use_double_wide_mlp": True, "sandwich_norms": True},
    "altup": {"altup": {"num_inputs": 2}, "num_kv_shared_layers": 2,
              "per_layer_input_dim": 8, "laurel_rank": 4},
}


def model_for(shape, **overrides):
    return CausalTransformer(
        vocab_size=32, emb_features=16, num_layers=4, num_heads=2,
        num_kv_heads=1, mlp_features=32, max_seq_len=16,
        dropout_rate=0.15, **{**SHAPES[shape], **overrides})


def batch():
    return {"text": jnp.asarray(np.random.default_rng(5).integers(1, 32, (8, 9)),
                                jnp.int32)}


def difference(left, right):
    return max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def objective_for(model):
    sparse = model.mixture is not None
    return LMObjective(model, 8, qk_stats=True,
                       balance_rate=0.01 if sparse else None,
                       aux_loss_alpha=0.001 if sparse else None,
                       mtp_weight=0.2 if model.num_nextn_predict_layers else None)


def training_step(model):
    objective = objective_for(model)
    optimizer = optax.sgd(1e-3, momentum=0.9)

    def step(variables, state, tokens, key):
        info = Step(jnp.zeros((), jnp.int32), key, None)
        (loss, aux), grads = jax.value_and_grad(
            lambda params: scalar_loss(objective, {**variables, "params": params},
                                       tokens, info),
            has_aux=True)(variables["params"])
        updates, state = optimizer.update(grads, state, variables["params"])
        moved = {**variables, "params": optax.apply_updates(variables["params"], updates),
                 **(aux.variables or {})}
        return moved, state, loss, grads, aux

    return jax.jit(step), optimizer


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("scan", [False, True])
def test_recomputed_blocks_train_the_same_model(shape, scan):
    """Dropout, MTP, MoE balancing and shared K/V retain their derivatives.

    Both optimizer steps use their own updated variables and fresh RNG keys;
    matching only the first step would miss lost mutable collection writes.
    """
    plain = model_for(shape, scan_layers=scan)
    remat = plain.clone(remat=True)
    tokens = batch()
    ids = tokens["text"][:, :-1]
    variables = plain.init(jax.random.key(0), ids)
    recomputed = remat.init(jax.random.key(0), ids)
    assert jax.tree.structure(variables) == jax.tree.structure(recomputed)
    assert difference(variables, recomputed) == 0
    assert difference(jax.jit(plain.apply)(variables, ids),
                      jax.jit(remat.apply)(variables, ids)) < 2e-5
    run, solver = training_step(plain)
    rerun, _ = training_step(remat)
    state, other_state = solver.init(variables["params"]), solver.init(recomputed["params"])
    original = variables
    for index in range(2):
        key = jax.random.fold_in(jax.random.key(4), index)
        variables, state, loss, grads, aux = run(variables, state, tokens, key)
        recomputed, other_state, other_loss, other_grads, other_aux = rerun(
            recomputed, other_state, tokens, key)
        assert abs(float(loss - other_loss)) < 2e-5
        assert difference(grads, other_grads) < 3e-5
        assert difference(variables, recomputed) < 2e-5
        assert difference(aux.metrics, other_aux.metrics) < 1e-4
        assert difference(aux.qk_stats, other_aux.qk_stats) < 1e-4
    assert difference(original["params"], variables["params"]) > 1e-4


@pytest.mark.parametrize("shape", ["dense", "shared", "altup"])
@pytest.mark.parametrize("scan", [False, True])
def test_remat_prefills_and_appends_the_same_cache(shape, scan):
    plain = model_for(shape, scan_layers=scan)
    remat = plain.clone(remat=True)
    ids = batch()["text"][:2]
    plain_vars = plain.init(jax.random.key(0), ids[:, :2], decode=True)
    remat_vars = remat.init(jax.random.key(0), ids[:, :2], decode=True)
    for start, end in [(0, 5), (5, 6), (6, 8)]:
        expected, cache = plain.apply(plain_vars, ids[:, start:end],
                                       decode=True, mutable=["cache"])
        actual, recomputed_cache = remat.apply(remat_vars, ids[:, start:end],
                                               decode=True, mutable=["cache"])
        assert difference(actual, expected) == 0
        assert difference(cache, recomputed_cache) == 0
        plain_vars = {**plain_vars, **cache}
        remat_vars = {**remat_vars, **recomputed_cache}


@pytest.mark.mesh
@pytest.mark.parametrize("shape", ["dense", "sparse"])
@pytest.mark.parametrize("scan", [False, True])
def test_recomputed_pipeline_preserves_updates_and_sown_values(shape, scan):
    plain = model_for(shape, scan_layers=scan, num_nextn_predict_layers=0)
    remat = plain.clone(remat=True)
    mesh_spec = MeshSpec(fsdp=2, stage=2, microbatches=4)
    mesh = build_mesh(mesh_spec)
    tokens = shard_batch(mesh, batch())
    variables = plain.init(jax.random.key(0), batch()["text"][:, :-1])
    variables = jax.device_put(variables, Layout(min_shard=16).shardings(mesh, variables))
    run, solver = training_step(plain)
    rerun, _ = training_step(remat)
    state = solver.init(variables["params"])
    with jax.set_mesh(mesh), pipeline_microbatches(mesh_spec.microbatches):
        left = run(variables, state, tokens, jax.random.key(7))
        right = rerun(variables, state, tokens, jax.random.key(7))
    assert abs(float(left[2] - right[2])) < 2e-5
    assert difference(left[0], right[0]) < 2e-5
    assert difference(left[3], right[3]) < 3e-5
    assert difference(left[4].qk_stats, right[4].qk_stats) < 1e-4
    assert difference(left[4].metrics, right[4].metrics) < 1e-4


@pytest.mark.parametrize("saved_remat", [False, True])
def test_checkpoint_resumes_with_remat_switched(tmp_path, saved_remat):
    model = model_for("shared", scan_layers=True, remat=saved_remat)
    variables = model.init(jax.random.key(0), batch()["text"][:, :-1])
    run, optimizer = training_step(model)
    moved, opt_state, _, _, _ = run(
        variables, optimizer.init(variables["params"]), batch(), jax.random.key(3))
    state = TrainState(jnp.asarray(1), moved, opt_state, None, jax.random.key(4))
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(1, state, None)
    checkpoints.wait()
    resumed_model = model.clone(remat=not saved_remat)
    template_vars = resumed_model.init(jax.random.key(8), batch()["text"][:, :-1])
    template = TrainState(jnp.asarray(0), template_vars,
                          optimizer.init(template_vars["params"]), None, jax.random.key(0))
    template = jax.tree.map(lambda value: jax.ShapeDtypeStruct(
        value.shape, value.dtype, sharding=value.sharding), template)
    restored, position = checkpoints.restore(template)
    assert position is None
    expected = run(state.params, state.opt_state, batch(), state.key)
    resume, _ = training_step(resumed_model)
    actual = resume(restored.params, restored.opt_state, batch(), restored.key)
    assert abs(float(actual[2] - expected[2])) < 2e-5
    assert difference(actual[0], expected[0]) < 2e-5
