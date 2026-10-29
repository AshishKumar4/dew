"""A checkpoint's input quantization in every view of its projection sites.

Each quantized Linear holds its input scale in its own `QUANTIZATION`
variables, so the views that rename or restack its scope carry the scale
with its kernel: the scanned stack, the pipeline's stages, the packed query,
key and value of the serving step, and every entry method of the model. Each
view's logits are on par with the compressed-tensors author's reference
(tests/interop_support.py `compressed_tensors_arms`, run in the test) from its float64
twin over drawn ids (check 1), from which the model without its input QDQ
moves (`assert_moves`).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from interop_support import NVFP4_QDQ, NVFP4_QDQ_CASES, compressed_tensors_arms, fixture_ids
from reference_error import assert_as_exact, assert_moves, dew_programs, widened

import dew
from dew.inference.projections import inference_projections, pending, projection_groups
from dew.interop import Pretrained
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.sharding import pipeline_microbatches
from dew.quantization import INPUT_SCALES, QUANTIZATION, InputQuantization
from dew.registry import wrapper_layers
from dew.training import Layout, MeshSpec

SEEDS = 0
"""Each view's declared draws of ids against the compressed-tensors reference (check 1), sized by
`seeds_for` paired with the view's logits on the CPU."""

APART_SEEDS = 0
"""The scanned runs' declared draws of ids against the plain loop of the same weights (check 1), sized by
`seeds_for` paired with the scanned logits on the CPU."""


@pytest.fixture(scope="module", params=NVFP4_QDQ_CASES)
def case(request):
    """The checkpoint's directory, loaded, check 1's draw of its ids (`fixture_ids`) and the
    compressed-tensors reference and truth run in the test (`compressed_tensors_arms`)."""
    directory = NVFP4_QDQ / request.param
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    return directory, loaded, fixture_ids(directory), compressed_tensors_arms(directory, input_qdq=True)


def assert_quantized(view, case, label):
    """`view`, a function of a draw's ids, on par with the reference (check 1)."""
    _, _, draw, (reference, truth) = case
    assert_as_exact(view, reference, draw, label, seeds=SEEDS, truth_fn=truth)


def test_without_its_scales_the_model_fails_the_rule_every_view_is_held_to(case):
    """The rule detects a dropped QDQ: the plain loop is on par with the reference, and the same weights
    with no input scale move past its rounding (`assert_moves`)."""
    _, loaded, draw, (reference, truth) = case
    assert_quantized(lambda ids: loaded.model.apply(loaded.variables, ids), case, "plain loop")
    bare = {name: tree for name, tree in loaded.variables.items() if name != QUANTIZATION}
    assert_moves(reference, lambda ids: loaded.model.apply(bare, ids), draw, "without its scales",
                 seeds=SEEDS, truth_fn=truth)


def test_the_scanned_stack_quantizes_each_layer_with_its_own_scales(case):
    """A scanned run stacks every collection on the layer axis, the scales
    with the kernels, so each layer reads its own."""
    _, loaded, _, _ = case
    scanned = loaded.model.clone(scan_layers=True)
    assert_quantized(lambda ids: scanned.apply(loaded.variables, ids), case, "scanned")


def quantized_apart(case):
    """The case's loaded model, its scanned clone, its first draw's ids, and a copy of its variables whose
    second layer's query projection is left unquantized: layers quantized apart, which one scanned body
    cannot read alike."""
    _, loaded, draw, _ = case
    variables = jax.tree.map(lambda leaf: leaf, dict(loaded.variables))
    layer = variables[QUANTIZATION]["layers_1"]
    mixer = next(name for name in layer if "q_proj" in layer[name] or "qkv" in layer[name])
    layer[mixer] = {name: leaves for name, leaves in layer[mixer].items() if name != "q_proj"}
    return loaded, loaded.model.clone(scan_layers=True), draw(0)[0], variables


def test_layers_quantized_apart_run_as_scanned_runs_of_their_own(case):
    """A checkpoint that quantizes two otherwise like layers differently, here
    one with a projection left unquantized, gives them different scale trees,
    which one scanned body cannot read: the scan runs them as runs of their
    own, each layer with its own scales, and its logits are held to the rule
    the plain loop of the same weights is, against their own float64
    forward, over drawn ids (check 1). The stored tree stays one subtree per
    layer, so the scanned and the plain model read and init the same tree."""
    loaded, scanned, ids, variables = quantized_apart(case)
    wide = loaded.model.clone(dtype=jnp.float64)
    assert_as_exact(lambda ids: scanned.apply(variables, ids), lambda ids: loaded.model.apply(variables, ids),
                    case[2], "layers quantized apart, scanned", seeds=APART_SEEDS,
                    truth_fn=lambda ids: wide.apply(widened(variables), ids))
    initialized = [jax.eval_shape(model.init, jax.random.key(0), ids) for model in (scanned, loaded.model)]
    assert jax.tree.structure(initialized[0]) == jax.tree.structure(initialized[1])


def test_layers_quantized_apart_that_one_bank_holds_are_refused(case):
    """A store that banks a run of like layers holds one array per leaf for all of them, which
    layers quantized apart cannot share: the run stays the one the store holds, and its unlike
    rows are refused by name, rather than read as runs of their own the store does not hold."""
    from dew.nn.backbones.layer_plan import group_name

    loaded, scanned, ids, variables = quantized_apart(case)
    params = dict(variables["params"])
    for first, count in scanned.bind(loaded.variables).groups:
        if count > 1:
            rows = [params.pop(f"layers_{index}") for index in range(first, first + count)]
            params[group_name(first, count)] = jax.tree.map(lambda *leaves: jnp.stack(leaves), *rows)
    with pytest.raises(ValueError):
        scanned.apply({**variables, "params": params}, ids)


def test_a_store_that_banks_layers_quantized_apart_reads_as_the_runs_it_banks(case):
    """Four like layers, the last two quantized apart from the first two, are two runs, which a
    store places as one bank each (`HeldBanks.place`): the model reading that store runs the same
    two runs, and its logits are on par with the plain loop of the same weights, from their own float64
    forward, over drawn ids (check 1)."""
    from dew.inference.banks import HeldBanks

    _, loaded, draw, _ = case
    model = loaded.model
    four = model.clone(num_layers=4, layer_types=None if model.layer_types is None else model.layer_types * 2)
    variables = jax.tree.map(lambda leaf: leaf, dict(loaded.variables))
    for tree in variables.values():
        for index in range(2):
            if f"layers_{index}" in tree:
                tree[f"layers_{index + 2}"] = jax.tree.map(lambda leaf: leaf, tree[f"layers_{index}"])
    for index in (2, 3):
        layer = variables[QUANTIZATION][f"layers_{index}"]
        mixer = next(name for name in layer if "q_proj" in layer[name] or "qkv" in layer[name])
        layer[mixer] = {name: leaves for name, leaves in layer[mixer].items() if name != "q_proj"}
    scanned = four.clone(scan_layers=True)
    bound = scanned.bind(variables)
    assert bound.groups == ((0, 2), (2, 2))
    store = HeldBanks(variables).place(bound, layout=Layout(min_shard=1, tolerance=1.0))
    wide = four.clone(dtype=jnp.float64)
    assert_as_exact(lambda ids: scanned.apply(store, ids), lambda ids: four.apply(variables, ids), draw,
                    "banked runs quantized apart", seeds=APART_SEEDS,
                    truth_fn=lambda ids: wide.apply(widened(variables), ids))


@pytest.mark.mesh(devices=2)
def test_the_pipeline_stages_quantize_each_layer_with_its_own_scales(case):
    """Two stages of one layer each, two microbatches of one row: the stage
    axis stacks the scales as it stacks the kernels."""
    _, loaded, _, _ = case
    mesh = MeshSpec(stage=2).build(jax.devices()[:2])
    placed = jax.device_put(loaded.variables, Layout().shardings(mesh, loaded.variables))

    @dew_programs
    def staged(ids):
        with jax.set_mesh(mesh), pipeline_microbatches(2):
            return jax.jit(loaded.model.apply)(placed, jnp.asarray(ids, jnp.int32))

    assert_quantized(staged, case, "two stages")


def test_packed_projections_carry_their_members_shared_scale(case):
    """The query, key and value read one input, and the checkpoint scales it
    once for all three, so their packed product quantizes it once."""
    _, loaded, _, _ = case
    groups = projection_groups(loaded.model, loaded.variables)
    assert groups
    packed = inference_projections(loaded.model, loaded.variables)
    for group in groups:
        node = packed[QUANTIZATION]
        for part in group.path:
            node = node[part]
        assert group.packed in node and not set(group.members) & set(node)
    assert_quantized(lambda ids: loaded.model.apply(packed, ids), case, "packed")


def test_members_scaled_apart_stay_unpacked(case):
    """Members whose input scales differ cannot share one quantized input, so
    their group stays unpacked and each keeps its own."""
    _, loaded, _, _ = case
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
    """`dew.pipeline` packs the projections once it has placed them; its task
    computes the loaded model's quantized logits."""
    directory, _, _, _ = case
    task = dew.pipeline(str(directory), dtype=jnp.float32)
    assert_quantized(lambda ids: task.model.apply(task.variables, ids), case, "pipeline")


def test_a_generation_over_the_loaded_tree_packs_under_its_one_scale_in_the_call(case):
    """A task over the loaded model's own tree, its query, key and value apart, packs them inside
    its compiled generation, under the one input scale `pending` finds on the loaded scales before
    the call: it draws, bit for bit, what a task over the tree packed beforehand draws. The
    pipeline's task, packed at its load, generates too."""
    directory, loaded, draw, _ = case
    ids = draw(0)[0][:1, :4]
    borrowed = loaded.text_generation()
    assert pending(loaded.model, loaded.variables)
    prepacked = borrowed.bind(inference_projections(loaded.model, loaded.variables))
    drawn, expected = (task(ids, 3, key=0).host() for task in (borrowed, prepacked))
    np.testing.assert_array_equal(drawn.tokens, expected.tokens)
    np.testing.assert_array_equal(drawn.raw_log_probs, expected.raw_log_probs)
    assert drawn.lengths[0] == 3 and np.isfinite(drawn.raw_log_probs).all()
    generated = dew.pipeline(str(directory), dtype=jnp.float32)(ids, 3, key=0).host()
    assert generated.lengths[0] == 3 and np.isfinite(generated.raw_log_probs).all()


def test_a_server_of_the_packed_task_reloads_the_loaded_models_own_tree(case):
    """`dew.pipeline` serves the query, key and value packed under their one
    input scale. A reload takes the loaded model's own tree, each member's
    scale at its own scope, and packs it as the load did, so the server
    holds the weights it held before. Members scaled apart cannot share the
    packed product's one scale, and their reload is refused."""
    from dew.inference.serving import Server

    directory, loaded, _, _ = case
    server = Server.from_task(dew.pipeline(str(directory), dtype=jnp.float32), slots=8, capacity=64)
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
    with pytest.raises(ValueError):
        server.reload(variables)


def test_every_entry_method_quantizes_the_sites_it_reaches(case):
    """The quantizers run under the model's `apply`, so an entry method a
    wrapper never named, as DiffusionGemma's `encode` was, quantizes too:
    encoding a prompt appends what the quantized decoder's prefill does."""
    _, loaded, draw, _ = case
    ids = jnp.asarray(draw(0)[0])
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


def test_a_scheme_its_arithmetic_ignores_is_refused():
    """ModelOpt's NVFP4 always rounds its local scales to E4M3, so a record
    that says otherwise is refused where it is made, not computed apart."""
    InputQuantization(e4m3_scale=True, format="modelopt")
    InputQuantization(e4m3_scale=False)
    with pytest.raises(ValueError):
        InputQuantization(e4m3_scale=False, format="modelopt")

