"""A checkpoint's input quantization in every view of its projection sites.

Each quantized Linear holds its input scale in its own `QUANTIZATION`
variables, so the views that rename or restack its scope carry the scale
with its kernel: the scanned stack, the pipeline's stages, the packed query,
key and value of the serving step, and every entry method of the model. Each
view's logits are held to the same float64 rule as the plain loop's, against
the compressed-tensors author's reference (tests/test_nvfp4_qdq.py), which
the model without its input QDQ fails.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from interop_support import NVFP4_QDQ_CASES, nvfp4_qdq_case
from reference_error import FACTOR, assert_as_exact_as_the_reference, distance

import dew
from dew.inference.projections import inference_projections, projection_groups
from dew.interop import Pretrained
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.sharding import pipeline_microbatches
from dew.registry import wrapper_layers
from dew.training import Layout, MeshSpec
from dew.training.quantization import INPUT_SCALES, QUANTIZATION, InputQuantization


@pytest.fixture(scope="module", params=NVFP4_QDQ_CASES)
def case(request):
    directory, reference = nvfp4_qdq_case(request.param)
    return directory, Pretrained.load(directory, dtype="float32", attention_impl="reference"), reference


def assert_quantized(logits, reference, label):
    assert_as_exact_as_the_reference(np.asarray(logits), reference["logits"], reference["logits_f64"], label)


def test_without_its_scales_the_model_fails_the_rule_every_view_is_held_to(case):
    """The rule detects a dropped QDQ: the same weights with no input scale
    lie far outside it."""
    _, loaded, reference = case
    ids = reference["ids"].astype(np.int32)
    assert_quantized(loaded.model.apply(loaded.variables, ids), reference, "plain loop")
    bare = {name: tree for name, tree in loaded.variables.items() if name != QUANTIZATION}
    dropped = np.asarray(loaded.model.apply(bare, ids))
    assert distance(dropped, reference["logits_f64"]) > FACTOR * distance(reference["logits"],
                                                                        reference["logits_f64"])


def test_the_scanned_stack_quantizes_each_layer_with_its_own_scales(case):
    """A scanned run stacks every collection on the layer axis, the scales
    with the kernels, so each layer reads its own."""
    _, loaded, reference = case
    scanned = loaded.model.clone(scan_layers=True)
    assert_quantized(scanned.apply(loaded.variables, reference["ids"].astype(np.int32)), reference, "scanned")


def test_layers_quantized_apart_refuse_one_scanned_run_by_name(case):
    """A checkpoint that quantizes two otherwise like layers differently, here
    one with a projection left unquantized, gives them different scale trees,
    which one scanned body cannot read; the scan says so."""
    _, loaded, reference = case
    variables = jax.tree.map(lambda leaf: leaf, dict(loaded.variables))
    layer = variables[QUANTIZATION]["layers_1"]
    mixer = next(name for name in layer if "q_proj" in layer[name] or "qkv" in layer[name])
    layer[mixer] = {name: leaves for name, leaves in layer[mixer].items() if name != "q_proj"}
    with pytest.raises(ValueError, match="'quantization' differs between layers_0 and layers_1"):
        loaded.model.clone(scan_layers=True).apply(variables, reference["ids"].astype(np.int32))


@pytest.mark.mesh(devices=2)
def test_the_pipeline_stages_quantize_each_layer_with_its_own_scales(case):
    """Two stages of one layer each, two microbatches of one row: the stage
    axis stacks the scales as it stacks the kernels."""
    _, loaded, reference = case
    mesh = MeshSpec(stage=2, microbatches=2).build(jax.devices()[:2])
    placed = jax.device_put(loaded.variables, Layout().shardings(mesh, loaded.variables))
    with jax.set_mesh(mesh), pipeline_microbatches(2):
        logits = jax.jit(loaded.model.apply)(placed, jnp.asarray(reference["ids"], jnp.int32))
    assert_quantized(logits, reference, "two stages")


def test_packed_projections_carry_their_members_shared_scale(case):
    """The query, key and value read one input, and the checkpoint scales it
    once for all three, so their packed product quantizes it once."""
    _, loaded, reference = case
    groups = projection_groups(loaded.model, loaded.variables)
    assert groups
    packed = inference_projections(loaded.model, loaded.variables)
    for group in groups:
        node = packed[QUANTIZATION]
        for part in group.path:
            node = node[part]
        assert group.packed in node and not set(group.members) & set(node)
    assert_quantized(loaded.model.apply(packed, reference["ids"].astype(np.int32)), reference, "packed")


def test_members_scaled_apart_stay_unpacked(case):
    """Members whose input scales differ cannot share one quantized input, so
    their group stays unpacked and each keeps its own."""
    _, loaded, _ = case
    groups = projection_groups(loaded.model, loaded.variables)
    group = groups[0]
    variables = jax.tree.map(lambda leaf: leaf, dict(loaded.variables))
    node = variables[QUANTIZATION]
    for part in group.path:
        node = node[part]
    leaf = next(iter(node[group.members[1]]))
    node[group.members[1]] = {leaf: np.float32(2) * node[group.members[1]][leaf]}
    assert group not in projection_groups(loaded.model, variables)


def test_the_served_pipeline_quantizes_as_the_loaded_model(case):
    """`dew.pipeline` packs the projections while it places them; its task
    computes the loaded model's quantized logits."""
    directory, _, reference = case
    task = dew.pipeline(str(directory), dtype=jnp.float32)
    logits = task.model.apply(task.variables, reference["ids"].astype(np.int32))
    assert_quantized(logits, reference, "pipeline")


def test_a_server_of_the_packed_task_reloads_the_loaded_models_own_tree(case):
    """`dew.pipeline` serves the query, key and value packed under their one
    input scale. A reload takes the loaded model's own tree, each member's
    scale at its own scope, and packs it as placement did, so the server
    holds the weights it held before. Members scaled apart cannot share the
    packed product's one scale, and their reload is refused."""
    from dew.inference.serving import Server

    directory, loaded, _ = case
    server = Server.from_task(dew.pipeline(str(directory), dtype=jnp.float32), slots=2, capacity=64)
    served = jax.tree.map(np.asarray, server.variables)
    server.reload(loaded.variables)
    for again, before in zip(jax.tree.leaves(server.variables), jax.tree.leaves(served), strict=True):
        np.testing.assert_array_equal(np.asarray(again), before)

    group = projection_groups(loaded.model, loaded.variables)[0]
    variables = jax.tree.map(lambda leaf: leaf, dict(loaded.variables))
    node = variables[QUANTIZATION]
    for part in group.path:
        node = node[part]
    member = node[group.members[1]]
    leaf = next(iter(member))
    node[group.members[1]] = {**member, leaf: np.float32(2) * member[leaf]}
    with pytest.raises(ValueError, match="quantize their one input apart"):
        server.reload(variables)


def test_every_entry_method_quantizes_the_sites_it_reaches(case):
    """The quantizers run under the model's `apply`, so an entry method a
    wrapper never named, as DiffusionGemma's `encode` was, quantizes too:
    encoding a prompt appends what the quantized decoder's prefill does."""
    _, loaded, reference = case
    ids = jnp.asarray(reference["ids"], jnp.int32)
    text = loaded.model
    cache = text.apply(loaded.variables, ids.shape[0], method="init_cache", mutable=["cache"])[1]
    prefill = text.apply({**loaded.variables, **cache}, ids, decode=True, mutable=["cache"])[0]
    base, (layer,) = wrapper_layers(type(text))
    plain = base(**{field.name: getattr(text, field.name) for field in dataclasses.fields(text)
                    if field.init and field.name not in ("parent", "name")})
    block = layer._dew_wrapper().apply(DiffusionGemma(text=plain, canvas_length=4))
    variables = {name: {"text": tree} for name, tree in loaded.variables.items()}
    cache = block.apply(variables, ids.shape[0], method="init_cache", mutable=["cache"])[1]
    encoded = block.apply({**variables, **cache}, ids, method="encode", mutable=["cache"])[0]
    np.testing.assert_array_equal(np.asarray(encoded), np.asarray(prefill))


def test_a_linear_given_its_input_by_name_quantizes_it_too():
    """Flax passes a Linear's input positionally or as `inputs=`; the site
    quantizes it either way."""
    x = jnp.linspace(-3, 3, 64, dtype=jnp.float32).reshape(2, 32)
    model = InputQuantization().apply(nn.Dense(8))
    params = model.init(jax.random.key(0), x)["params"]
    variables = {"params": params, QUANTIZATION: {INPUT_SCALES["fp8"]: np.float32(0.004)}}
    positional = model.apply(variables, x)
    np.testing.assert_array_equal(np.asarray(model.apply(variables, inputs=x)), np.asarray(positional))
    assert not np.array_equal(np.asarray(nn.Dense(8).apply({"params": params}, x)), np.asarray(positional))


def test_a_scheme_its_arithmetic_ignores_is_refused_by_name():
    """ModelOpt's NVFP4 always rounds its local scales to E4M3, so a record
    that says otherwise is refused where it is made, not computed apart."""
    with pytest.raises(ValueError, match="always rounds its local scales to E4M3"):
        InputQuantization(e4m3_scale=False, format="modelopt")

