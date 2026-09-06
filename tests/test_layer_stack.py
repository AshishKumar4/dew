"""The decoder's layer stack under flax's scan and over the stage axis.

`scan_layers` groups consecutive layers that share a parameter shape and a
computation and runs each group as iterations of one body; a stage axis on
the mesh in context runs the stack as a GPipe pipeline. Both read the
plain loop's variables tree through `StackView`, so the tests here hold the
tree, the logits, the decode path, the loss and the gradients to the plain
loop's, with the largest observed difference written beside each bound.
"""

import json
from pathlib import Path

import jax
from dew.objectives.base import scalar_loss
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.interop.hf_decoders import load_pretrained_decoder, translate_config

from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models, with_precision
from dew.training import Layout, MeshSpec, build_mesh
from dew.training.distributed import shard_batch

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
VOCAB = 64
SEQ_LEN = 15
BATCH = 8
TINY_SHARD = 256

FIXTURE_NAMES = ("qwen3-tiny", "deepseek-v3-tiny", "gemma4-e2b", "gemma3n-tiny")


def fixture_pair(name, **overrides):
    """The fixture's model and its scanned twin, with the fixture's weights."""
    directory = FIXTURES / name
    model, variables, built = load_pretrained_decoder(
        str(directory), dtype="float32", attention_impl="reference", **overrides)
    scanned = models.build("causal_transformer", **{**built, "scan_layers": True})
    return model, scanned, variables, directory


def paths(tree):
    return [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(tree)]


@pytest.mark.parametrize("name", sorted(FIXTURE_NAMES))
def test_a_scanned_fixture_scores_as_the_plain_loop(name):
    """The fixture's weights through the scanned stack: the runs the geometry
    implies, the same logits as the plain loop and the reference parity of
    tests/test_hf_decoders.py. Largest observed |logit difference| against
    the plain loop on CPU: qwen3-tiny 3.3e-06, and 0.0 for the three whose
    layers all differ from their neighbours."""
    model, scanned, variables, directory = fixture_pair(name)
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    reference = np.load(directory / "logits.npy")

    
    plain = np.asarray(model.apply(variables, ids))
    logits = np.asarray(scanned.apply(variables, ids))

    difference = float(np.max(np.abs(logits - plain)))
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(plain, axis=-1))
    assert float(np.max(np.abs(logits - reference))) < 1e-4


@pytest.mark.parametrize("name", sorted(FIXTURE_NAMES))
def test_a_scanned_fixture_decodes_through_the_cache(name):
    """A prefill and single-token steps through the scanned stack's cache
    against the whole sequence at once. Largest observed difference on
    CPU: 3.4e-06 (qwen3-tiny)."""
    model, scanned, variables, directory = fixture_pair(name, max_seq_len=16)
    ids = jnp.asarray(np.load(directory / "input_ids.npy")[:1, :8], jnp.int32)
    whole = scanned.apply(variables, ids)

    state = scanned.init(jax.random.key(0), ids[:, :1], decode=True)
    assert paths(state["cache"]) == paths(model.init(jax.random.key(0), ids[:, :1],
                                                     decode=True)["cache"])
    steps = []
    for position in range(ids.shape[1]):
        step, state = scanned.apply(
            {**variables, "cache": state["cache"]}, ids[:, position:position + 1],
            decode=True, mutable=["cache"])
        steps.append(step)

    difference = float(jnp.max(jnp.abs(jnp.concatenate(steps, axis=1) - whole)))
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"


def gemma4_shaped(**overrides):
    """A gemma4-e2b shaped decoder twice the fixture's depth: runs of sliding
    layers, a full layer between them, four sharing layers at the end."""
    config = translate_config(json.loads((FIXTURES / "gemma4-e2b" / "config.json").read_text()))
    config.update(
        num_layers=12,
        layer_types=("sliding_attention",) * 5 + ("full_attention",)
        + ("sliding_attention",) * 5 + ("full_attention",),
        num_kv_shared_layers=4, max_seq_len=16)
    config.update(overrides)
    return models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="reference"))


def gemma3n_shaped(**overrides):
    """A gemma3n-tiny shaped decoder of ten layers: sparsity on the first
    five, one width, the last two sharing keys and values."""
    config = translate_config(json.loads((FIXTURES / "gemma3n-tiny" / "config.json").read_text()))
    config.update(
        num_layers=10,
        layer_types=("sliding_attention",) * 4 + ("full_attention",)
        + ("sliding_attention",) * 4 + ("full_attention",),
        mlp_features=(48,) * 10, activation_sparsity_pattern=(0.95,) * 5 + (0.0,) * 5,
        num_kv_shared_layers=2, max_seq_len=16)
    config.update(overrides)
    return models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="reference"))


def deepseek_shaped(**overrides):
    """A deepseek-v3-tiny shaped decoder of six layers, one dense then five routed."""
    config = translate_config(json.loads((FIXTURES / "deepseek-v3-tiny" / "config.json").read_text()))
    config.update(num_layers=6, layer_types=("full_attention",) * 6, max_seq_len=16)
    config["mixture"] = {**config["mixture"], "layers": (1, 2, 3, 4, 5)}
    config.update(overrides)
    return models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="reference"))


# Bounds on scanned logits and gradients against the plain loop.
# Gemma 3n keeps the initialized forward check but checks gradients
# at reference-scale projections with a live per-layer-input residual; its
# default zero correction scales put that residual exactly at RMSNorm's
# epsilon floor. The resulting high-gain Jacobian is sensitive even to
# one-ULP weight perturbations without any sparse-ReLU branch changing.
# CE gradients are of order one. At fp32, Transformers 5.16.1 on identical
# weights differs by at most 6.92e-6; scan/plain by 7.53e-7 on CPU and
# 8.11e-6 on RTX 4080 (JAX 0.10/0.11). The 1e-5 bound also rejects
# wrong-layer and missing-residual mutations.
SHAPES = {
    "gemma4": (gemma4_shaped, 1e-4, 1e-4),
    "gemma3n": (gemma3n_shaped, 5e-4, 1e-5),
    "deepseek": (deepseek_shaped, 1e-6, 1e-6),
}


def same_metrics(left: dict, right: dict, relative: float = 1e-5) -> bool:
    """Every metric within `relative` of the other run's; perplexity is exp
    of the cross entropy, so an absolute bound would say nothing about it."""
    return left.keys() == right.keys() and all(
        abs(left[name] - right[name]) <= relative * max(1.0, abs(left[name])) for name in left)


def scanned_pair(build):
    plain, scanned = build(), build(scan_layers=True)
    ids = jax.random.randint(jax.random.key(1), (2, 12), 0, VOCAB)
    variables = plain.init(jax.random.key(0), ids)
    return plain, scanned, variables, ids


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_runs_of_like_layers_scan_and_the_rest_unroll(shape):
    """The grouping read off the geometry, and the scanned logits and
    gradients against the plain loop's on a stack deep enough to scan, to
    the bounds SHAPES states."""
    build, logit_bound, gradient_bound = SHAPES[shape]
    plain, scanned, variables, ids = scanned_pair(build)
    

    logits = jax.jit(scanned.apply)(variables, ids)
    difference = float(jnp.max(jnp.abs(logits - jax.jit(plain.apply)(variables, ids))))
    assert difference < logit_bound, f"max |logit difference| {difference:.3e}"

    if shape == "gemma3n":
        variables = gemma3n_training_variables(variables)

    def gradients(model):
        def loss(params):
            current = {**variables, "params": params}
            if shape == "gemma3n":
                return next_token_loss(model, current, ids)
            return jnp.mean(model.apply(current, ids) ** 2)
        return jax.jit(jax.value_and_grad(loss))(variables["params"])

    loss, grads = gradients(plain)
    scanned_loss, scanned_grads = gradients(scanned)
    assert abs(float(loss - scanned_loss)) < logit_bound
    difference = largest_difference(grads, scanned_grads)
    assert difference < gradient_bound, f"max |gradient difference| {difference:.3e}"


def gemma3n_training_variables(variables):
    """Reference initializer scale, with the per-layer input branch active.

    LeCun kernels have variance 1/fan_in; Gemma 3n's reference config uses
    initializer_range=0.02. Only this gradient case rescales them. Init
    identity and fixture/initialized logits still use their original weights.
    """
    def rescale(path, value):
        if path[-1].key == "kernel":
            return value * np.float32(np.sqrt(value.shape[0]) * 0.02)
        if path[-1].key == "correct_output_scale":
            return jnp.ones_like(value)
        return value
    return jax.tree_util.tree_map_with_path(rescale, variables)


def next_token_loss(model, variables, ids):
    logits = model.apply(variables, ids)[:, :-1]
    return -jnp.take_along_axis(
        jax.nn.log_softmax(logits), ids[:, 1:, None], axis=-1).mean()


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_scanned_init_is_the_plain_init(shape):
    """`init` runs the plain loop whatever `scan_layers` says, so the tree
    holds the same leaves with the same values, and the scanned model reads
    a checkpoint of the plain one leaf for leaf."""
    build = SHAPES[shape][0]
    plain, scanned = build(), build(scan_layers=True)
    ids = jnp.ones((1, 4), jnp.int32)
    first, second = plain.init(jax.random.key(0), ids), scanned.init(jax.random.key(0), ids)

    assert paths(first) == paths(second)
    assert all(jax.tree.leaves(jax.tree.map(
        lambda a, b: bool(jnp.array_equal(a, b)), first, second)))


def test_a_scanned_moe_stack_sows_and_balances_like_the_plain_loop():
    """The routers sow through the scanned view, so the balance loss and the
    aux-loss-free bias move as they do under the plain loop. Observed loss
    difference on CPU 0.0, bias difference 0.0."""
    plain, scanned, variables, ids = scanned_pair(deepseek_shaped)
    batch = {"text": jax.random.randint(jax.random.key(2), (BATCH, 12), 0, VOCAB)}
    step = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3), ema=None)

    outcomes = []
    for model in (plain, scanned):
        objective = LMObjective(model, 11, balance_rate=0.01, aux_loss_alpha=0.1)
        loss, aux = scalar_loss(objective, variables, batch, step)
        assert aux.effects is not None
        outcomes.append((float(loss), {name: float(value) for name, value in aux.metrics.items()},
                         objective.apply_effects(variables, aux.effects)["moe"]))
    (loss, metrics, moe), (scanned_loss, scanned_metrics, scanned_moe) = outcomes

    assert abs(loss - scanned_loss) < 1e-5, (loss, scanned_loss)
    assert same_metrics(metrics, scanned_metrics), (metrics, scanned_metrics)
    assert paths(moe) == paths(scanned_moe)
    assert max(jax.tree.leaves(jax.tree.map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))), moe, scanned_moe))) < 1e-6


TINY = dict(vocab_size=VOCAB, emb_features=32, num_layers=4, num_heads=4,
            num_kv_heads=2, mlp_features=64, max_seq_len=SEQ_LEN)


def tiny(**overrides):
    return models.build("causal_transformer", **{**TINY, **overrides})


def bf16(**overrides):
    return models.build("causal_transformer", **with_precision(
        "causal_transformer", {**TINY, **overrides}, dtype="bfloat16", attention_impl="xla"))


def test_a_scanned_stack_under_a_bf16_policy_scores_as_the_plain_loop_does():
    """Under a bf16 compute policy the plain loop's first layer returns the
    residual stream in fp32 and the scan takes it in fp32 from the start,
    so the first residual sum is rounded once less. Against the same
    weights at fp32 the scanned logits sit as far off as the plain loop's:
    observed on CPU 5.2e-02 against 5.0e-02 on logits of order 4, and
    3.1e-02 between the two. The fp32 tests above hold the two paths to
    fp32 tolerance."""
    plain, scanned = bf16(), bf16(scan_layers=True)
    exact = tiny(attention_impl="xla")
    ids = jax.random.randint(jax.random.key(1), (2, 12), 0, VOCAB)
    variables = plain.init(jax.random.key(0), ids)
    logits, scanned_logits = plain.apply(variables, ids), scanned.apply(variables, ids)
    reference = exact.apply(variables, ids)

    plain_distance = float(jnp.max(jnp.abs(logits - reference)))
    scanned_distance = float(jnp.max(jnp.abs(scanned_logits - reference)))
    assert scanned_distance < 1e-1 and scanned_distance < 1.25 * plain_distance, (
        plain_distance, scanned_distance)
    assert float(jnp.max(jnp.abs(logits - scanned_logits))) < 1e-1





# --------------------------------------------------------------------------
# The pipeline over the stage axis: the eight simulated devices
# --------------------------------------------------------------------------

mesh_lane = pytest.mark.mesh


def routed(**overrides):
    return tiny(mixture={"experts": 4, "top_k": 2, "bias": True}, **overrides)


def token_batch():
    rng = np.random.default_rng(0)
    return {"text": rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}


def loss_and_grads(objective, spec, variables, batch):
    """The objective's loss, metrics and gradients on `spec`'s mesh, with the
    pipeline's schedule in context the way the trainer's compiled step
    puts it there."""
    mesh = build_mesh(spec)
    layout = Layout(min_shard=TINY_SHARD)
    placed = jax.device_put(variables, layout.shardings(mesh, variables))
    batch = shard_batch(mesh, batch)
    step = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3), ema=None)

    def loss(params):
        return scalar_loss(objective, {**variables, "params": params}, batch, step)

    with jax.set_mesh(mesh), pipeline_microbatches(spec.microbatches):
        (value, aux), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(placed["params"])
    return (float(value), {name: float(number) for name, number in aux.metrics.items()},
            jax.tree.map(np.asarray, grads))


def largest_difference(left, right) -> float:
    return max(jax.tree.leaves(jax.tree.map(
        lambda a, b: float(np.max(np.abs(a - b))), left, right)))


@mesh_lane
@pytest.mark.parametrize("build, balance", [(tiny, {}), (routed, {"balance_rate": 0.01})],
                         ids=["dense", "moe"])
def test_a_pipeline_over_two_stages_has_the_loss_and_gradient_of_one(build, balance):
    """Four layers over two stages fed four microbatches, against the same
    step on the fsdp mesh: the loss, every metric and every gradient leaf.
    Largest observed differences on CPU: loss 0.0 (dense) and 4.8e-07
    (moe), gradients 1.6e-07 (dense) and 3.4e-07 (moe) on leaves of order
    0.2."""
    model = build()
    objective = LMObjective(model, SEQ_LEN, **balance)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    batch = token_batch()

    loss, metrics, grads = loss_and_grads(objective, MeshSpec(fsdp=4), variables, batch)
    piped, piped_metrics, piped_grads = loss_and_grads(
        objective, MeshSpec(fsdp=2, stage=2, microbatches=4), variables, batch)

    assert abs(loss - piped) < 1e-5, (loss, piped)
    assert same_metrics(metrics, piped_metrics), (metrics, piped_metrics)
    difference = largest_difference(grads, piped_grads)
    assert difference < 1e-5, f"max |gradient difference| {difference:.3e}"


@mesh_lane
def test_a_scanned_pipeline_has_the_loss_and_gradient_of_the_plain_loop():
    """Stages of two like layers scan inside the pipeline; the loss and the
    gradients hold to the plain loop's. Largest observed differences on
    CPU: loss 0.0, gradients 1.6e-07 on leaves of order 0.2."""
    variables = tiny().init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    batch = token_batch()

    loss, _, grads = loss_and_grads(LMObjective(tiny(), SEQ_LEN), MeshSpec(fsdp=4), variables, batch)
    piped, _, piped_grads = loss_and_grads(
        LMObjective(tiny(scan_layers=True), SEQ_LEN),
        MeshSpec(fsdp=2, stage=2, microbatches=2), variables, batch)

    assert abs(loss - piped) < 1e-5, (loss, piped)
    difference = largest_difference(grads, piped_grads)
    assert difference < 1e-5, f"max |gradient difference| {difference:.3e}"


@mesh_lane
def test_a_pipeline_under_a_bf16_policy_trains_the_loss_of_the_plain_loop():
    """The stages hand the residual stream on in the dtype it settles in, so
    a bf16 policy pipelines as it scans: the loss within bf16 rounding of
    the plain loop's and every gradient finite. Observed loss difference on
    CPU: 3.4e-04 on a loss of order 5."""
    model = bf16()
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    batch = token_batch()

    loss, _, _ = loss_and_grads(LMObjective(model, SEQ_LEN), MeshSpec(fsdp=4), variables, batch)
    piped, _, piped_grads = loss_and_grads(
        LMObjective(model, SEQ_LEN), MeshSpec(fsdp=2, stage=2, microbatches=4), variables, batch)

    assert abs(loss - piped) < 1e-2 * max(1.0, abs(loss)), (loss, piped)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(piped_grads))




@mesh_lane
def test_a_pipeline_refuses_a_stack_it_cannot_split_evenly():
    """Three layers over two stages, a routed layer where the first stage
    has a dense one, and sharing layers at the end of the stack are each
    refused with the layers that differ and what differs."""
    spec = MeshSpec(fsdp=4, stage=2)
    batch = token_batch()

    def run(model):
        variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
        loss_and_grads(LMObjective(model, SEQ_LEN), spec, variables, batch)

    with pytest.raises(ValueError, match="3 layers do not split into 2 stages"):
        run(tiny(num_layers=3))
    with pytest.raises(ValueError, match="layer 2 differs from layer 0 in routed"):
        run(tiny(mixture={"experts": 4, "top_k": 2, "layers": (2, 3)}))
    with pytest.raises(ValueError, match="layer 2 differs from layer 0 in provider"):
        run(tiny(num_kv_shared_layers=1))


@mesh_lane
def test_a_pipeline_refuses_a_schedule_that_does_not_fit_the_batch():
    """Eight rows do not make sixteen microbatches, and microbatches are a
    multiple of the stages; both are refused before anything runs."""
    model = tiny()
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    with pytest.raises(ValueError, match="microbatch count that divides the rows"):
        loss_and_grads(LMObjective(model, SEQ_LEN), MeshSpec(fsdp=4, stage=2, microbatches=16),
                       variables, token_batch())
    with pytest.raises(ValueError, match="multiple of stage"):
        MeshSpec(stage=2, microbatches=3)
    with pytest.raises(ValueError, match="stage is 1"):
        MeshSpec(microbatches=4)


@mesh_lane
def test_decoding_under_a_stage_axis_is_refused():
    model = tiny()
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    with jax.set_mesh(build_mesh(MeshSpec(fsdp=4, stage=2))):
        with pytest.raises(ValueError, match="decode outside jax.set_mesh"):
            model.apply(variables, jnp.ones((1, 1), jnp.int32), decode=True, mutable=["cache"])


@mesh_lane
def test_a_layout_rule_onto_the_stage_axis_is_refused():
    with pytest.raises(ValueError, match="stage axis holds the pipeline"):
        Layout(rules={"mlp": "stage"})


def test_scanned_dropout_uses_the_supplied_rng():
    model = tiny(num_layers=3, dropout_rate=0.2, scan_layers=True)
    ids = jnp.asarray(token_batch()["text"][:2, :5])
    variables = model.init(jax.random.key(0), ids)

    @jax.jit
    def loss_and_grad(params, key):
        def loss(weights):
            logits = model.apply({"params": weights}, ids, train=True,
                                 rngs={"dropout": key})
            return jnp.mean(logits ** 2)
        return jax.value_and_grad(loss)(params)

    first = loss_and_grad(variables["params"], jax.random.key(1))
    repeated = loss_and_grad(variables["params"], jax.random.key(1))
    changed = loss_and_grad(variables["params"], jax.random.key(2))

    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(repeated)):
        np.testing.assert_array_equal(left, right)
        assert np.isfinite(left).all()
    assert float(first[0]) != float(changed[0])
    assert largest_difference(first[1], changed[1]) > 1e-5
