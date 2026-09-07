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

from dew.checkpoints import Checkpoints
from dew.nn.backbones.causal_transformer import (
    REMAT_POLICIES, RESIDUALS, CausalTransformer, RematPolicy,
)
from dew.nn.sharding import pipeline_microbatches
from dew.objectives import scalar_loss
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.training.distributed import Layout, MeshSpec, build_mesh, shard_batch
from dew.training.state import TrainState
from dew.training.trainer import write_back


SHAPES = {
    "dense": {"num_nextn_predict_layers": 1},
    "sparse": {"mixture": {"experts": 4, "top_k": 2, "bias": True},
               "num_nextn_predict_layers": 1},
    "shared": {"num_kv_shared_layers": 2, "per_layer_input_dim": 8,
               "use_double_wide_mlp": True, "sandwich_norms": True},
    "altup": {"altup": {"num_inputs": 2}, "num_kv_shared_layers": 2,
              "per_layer_input_dim": 8, "laurel_rank": 4},
    "mla": {"mixer": {"kind": "mla", "q_lora_rank": 8, "kv_lora_rank": 8,
                      "qk_nope_head_dim": 4, "qk_rope_head_dim": 4, "v_head_dim": 4}},
}


def model_for(shape, **overrides):
    return CausalTransformer(
        vocab_size=32, emb_features=16, num_layers=4, num_heads=2,
        num_kv_heads=1, mlp_features=32, max_seq_len=16,
        **{"dropout_rate": 0.15, **SHAPES[shape], **overrides})


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
    """One committed step as the trainer takes it: the optimizer update, then
    sequential collection writes and the objective's deferred effects."""
    objective = objective_for(model)
    optimizer = optax.sgd(1e-3, momentum=0.9)

    def step(variables, state, tokens, key):
        info = Step(jnp.zeros((), jnp.int32), key, None)
        (loss, aux), grads = jax.value_and_grad(
            lambda params: scalar_loss(objective, {**variables, "params": params},
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
    return TrainState(step=count, microstep=count, updates=count, params=variables,
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
    assert difference(variables, recomputed) == 0
    assert difference(jax.jit(plain.apply)(variables, ids),
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
        assert difference(grads, other_grads) < 3e-5
        assert difference(moved, recomputed) < 2e-5
        assert difference(next_state, other_state) < 2e-5
        assert difference(aux.metrics, other_aux.metrics) < 1e-4
        assert difference(aux.qk_stats, other_aux.qk_stats) < 1e-4
        if shape == "sparse":
            # Router slot counts are integers; recomputation routes identically.
            assert difference(aux.effects, other_aux.effects) == 0
        variables, state = moved, next_state
    assert difference(original["params"], variables["params"]) > 1e-4
    if shape == "sparse":
        assert difference(original["moe"], variables["moe"]) > 0


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
        assert difference(actual, expected) == 0

        plain_vars = {**plain_vars, **cache}
        remat_vars = {**remat_vars, **recomputed_cache}


@pytest.mark.mesh
@pytest.mark.parametrize("shape", ["dense", "sparse"])
@pytest.mark.parametrize("scan", [False, True])
def test_recomputed_pipeline_preserves_updates_and_sown_values(shape, scan):
    plain = model_for(shape, scan_layers=scan, num_nextn_predict_layers=0)
    remat = plain.clone(remat='full')
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
    expected = run(state.params, state.opt_state, batch(), state.key)
    resume, _ = training_step(resumed_model)
    actual = resume(restored.params, restored.opt_state, batch(), restored.key)
    assert abs(float(actual[2] - expected[2])) < 2e-5
    assert difference(actual[0], expected[0]) < 2e-5


NAMED = [name for name in REMAT_POLICIES if name != 'full']


def gradient_step(model, variables, key):
    objective = objective_for(model)

    def loss(params):
        info = Step(jnp.zeros((), jnp.int32), key, None)
        return scalar_loss(objective, {**variables, "params": params}, batch(), info)

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
    assert difference(grads, other_grads) < 2e-6
    for name, value in aux.metrics.items():
        other = other_aux.metrics[name]
        if name == "perplexity":
            # Compare in cross-entropy units; exp magnifies the same rounding.
            np.testing.assert_allclose(np.log(value), np.log(other), atol=2e-6, rtol=0)
        else:
            np.testing.assert_allclose(value, other, atol=1e-5, rtol=0)


def residuals(model, variables, capsys):
    """What the backward pass keeps, one description per residual, as
    `jax.ad_checkpoint.print_saved_residuals` reports it."""
    objective = objective_for(model)

    def loss(params):
        info = Step(jnp.zeros((), jnp.int32), jax.random.key(1), None)
        return scalar_loss(objective, {**variables, "params": params}, batch(), info)[0]

    capsys.readouterr()
    jax.ad_checkpoint.print_saved_residuals(loss, variables["params"])
    return capsys.readouterr().out.splitlines()


def named(lines):
    """The residual names kept, with how many layers kept each."""
    counts = {}
    for line in lines:
        if " named '" in line:
            name = line.split(" named '")[1].split("'")[0]
            counts[name] = counts.get(name, 0) + 1
    return counts


KEPT = {
    # What each recipe keeps in a block whose backward pass can use it: the
    # projection outputs the norms and the attention read back, never
    # `down_proj`, whose output only enters the residual sum.
    'full': (),
    'minimal': ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj'),
    'minimal_with_context': ('q_proj', 'k_proj', 'v_proj', 'context', 'o_proj', 'gate_proj', 'up_proj'),
    'save_dot_except_mlp': ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
    'save_dot_with_context_except_mlp': ('q_proj', 'k_proj', 'v_proj', 'context', 'o_proj'),
    'save_dot_except_mlpwi': ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
    'save_qkv_proj': ('q_proj', 'k_proj', 'v_proj'),
    'save_out_proj': ('o_proj',),
}


@pytest.mark.parametrize("policy", sorted(KEPT))
def test_a_policy_keeps_the_residuals_it_names_in_every_layer(policy, capsys):
    model = model_for("dense", dropout_rate=0.0, num_nextn_predict_layers=0, remat=policy)
    variables = model.init(jax.random.key(0), batch()["text"][:, :-1])
    assert named(residuals(model, variables, capsys)) == {
        name: model.num_layers for name in KEPT[policy]}


@pytest.mark.parametrize("policy, kept", [
    ('minimal_offloaded', 6), ('qkv_proj_offloaded', 3)])
@pytest.mark.parametrize("scan", [False, True])
def test_an_offloaded_policy_types_its_residuals_in_host_memory(policy, kept, scan, capsys):
    """The residuals the policy offloads leave the forward pass in host
    memory space, one per kept name per layer, stacked over a scanned run."""
    model = model_for("dense", dropout_rate=0.0, num_nextn_predict_layers=0,
                      scan_layers=scan, remat=policy)
    variables = model.init(jax.random.key(0), batch()["text"][:, :-1])
    lines = residuals(model, variables, capsys)
    hosted = [line for line in lines if line.startswith("f32<host>")]
    assert len(hosted) == (kept if scan else kept * model.num_layers)
    assert not named(lines)


def test_latent_attention_keeps_its_fused_kv_projection(capsys):
    model = model_for("mla", dropout_rate=0.0, remat='save_qkv_proj')
    variables = model.init(jax.random.key(0), batch()["text"][:, :-1])
    assert named(residuals(model, variables, capsys)) == {
        'q_proj': model.num_layers, 'kv_proj': model.num_layers}


def test_a_scanned_run_keeps_more_as_the_policy_saves_more(capsys):
    counts = {}
    for policy in ('full', 'save_qkv_proj', 'minimal'):
        model = model_for("dense", dropout_rate=0.0, num_nextn_predict_layers=0,
                          scan_layers=True, remat=policy)
        variables = model.init(jax.random.key(0), batch()["text"][:, :-1])
        counts[policy] = len(residuals(model, variables, capsys))
    assert counts['full'] < counts['save_qkv_proj'] < counts['minimal']


def test_a_policy_record_builds_the_named_recipe():
    model = model_for("dense", remat={"save": ["q_proj", "k_proj", "v_proj", "kv_proj"]})
    assert model.remat == REMAT_POLICIES['save_qkv_proj']
    assert model_for("dense", remat='full').remat == RematPolicy()
    assert model_for("dense").remat is None


@pytest.mark.parametrize("remat, message", [
    ('minimal_flash', "remat names one of"),
    (True, "remat is a RematPolicy"),
    ({"save": ["query_proj"]}, "names residuals from"),
    ({"save": ["q_proj"], "offload": ["q_proj"]}, "not both"),
    ({"saved": ["q_proj"]}, "holds 'save' and 'offload'"),
])
def test_a_policy_off_the_list_is_refused(remat, message):
    with pytest.raises(ValueError, match=message):
        model_for("dense", remat=remat)
