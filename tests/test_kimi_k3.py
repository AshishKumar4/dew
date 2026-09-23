"""Kimi K3's text decoder against the released remote code.

kimi-k3-source holds the released config and the name, dtype and shape of
every tensor of moonshotai/Kimi-K3 at f831ab6 (read off the 96 shard
headers). kimi-k3-tiny is written by tools/kimi_k3_reference.py from the
pinned modeling_kimi_linear.py with fla-core 0.5.2's kernels in fp32 on
CUDA; its routed experts are compressed-tensors MXFP4 as in the release.
No model weights are downloaded at test time.
"""

import json
import lzma
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.interop import load_pretrained
from dew.interop.codecs import decode_e2m1, packed_mxfp4_tensor_names, quantize_packed_mxfp4
from dew.interop.hf_decoders import _FAMILIES, _flatten, translate_config
from dew.nn.inputs import ModelInputs
from dew.nn.moe import Situ
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling import Sampling, generate

ROOT = Path(__file__).parent / "fixtures" / "hf"
SOURCE = ROOT / "kimi-k3-source"
TINY = ROOT / "kimi-k3-tiny"


@pytest.fixture(scope="module")
def source():
    loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
    reference = np.load(TINY / "reference.npz")
    inputs = ModelInputs(jnp.asarray(reference["input_ids"], jnp.int32),
                         {"attention_mask": jnp.asarray(reference["attention_mask"], bool)})
    return loaded, inputs, reference


def test_released_config_builds_the_published_geometry():
    fields = translate_config(json.loads((SOURCE / "config.json").read_text()))
    model = models.build("causal_transformer", fields)
    assert model.num_layers == 93 and model.emb_features == 7168
    assert model.per_layer_types.count("linear_attention") == 69
    assert model.per_layer_types.count("full_attention") == 24
    assert model.per_layer_types[-1] == "full_attention"
    assert model.mixture.experts == 896 and model.mixture.top_k == 16
    assert model.sparse_layers == tuple(range(1, 93))
    assert (model.mixture.expert_features, model.mixture.shared_features) == (3072, 6144)
    assert model.mixture.latent_features == 3584 and model.mixture.latent_norm
    assert model.attention_residuals.block_size == 12
    assert model.attention_residuals.blocks(93) == 8
    assert model.mlp == Situ(4.0, 25.0)


def released_tensors() -> dict[str, tuple[str, list[int]]]:
    return json.loads(lzma.decompress((SOURCE / "tensors.json.xz").read_bytes()))


def test_every_released_tensor_lands_on_one_leaf_of_the_released_tree():
    """497,220 source tensors: 494,592 MXFP4 halves of 247,296 expert
    weights, 2,460 dense text tensors and 168 tower tensors. Decoded and
    prepared, the 249,756 text tensors land on distinct leaves of the tree
    the released config builds, each at the leaf's shape (A_log trimmed from
    its zero-padded 128 to the 96 heads), and cover all 2,736 of its leaves,
    before 1.5 TB of weights would be read."""
    config = json.loads((SOURCE / "config.json").read_text())
    fields, family = translate_config(config), _FAMILIES["kimi_k3"]
    model = models.build("causal_transformer", fields)
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))
    tree = {tuple(name.split(".")): leaf.shape for name, leaf in _flatten(dict(shapes)).items()}
    tensors = released_tensors()
    decoded = {}
    for name, (dtype, shape) in tensors.items():
        if name.endswith(".weight_packed"):
            assert dtype == "U8" and tensors[name.removesuffix("packed") + "scale"] == ["U8", [shape[0], shape[1] // 16]]
            decoded[name.removesuffix("_packed")] = (shape[0], 2 * shape[1])
        elif not name.endswith(".weight_scale"):
            decoded[name] = tuple(shape)
    assert set(packed_mxfp4_tensor_names(dict.fromkeys(tensors))) == set(decoded)
    prepared = family.prepare_weights({name: np.broadcast_to(np.float32(0), shape) for name, shape in decoded.items()})
    placed, experts, towers = {}, {}, 0
    for name, value in prepared.items():
        path = family.weight_path(name, fields)
        if path is None:
            assert name.startswith(("vision_tower.", "mm_projector."))
            towers += 1
            continue
        shape = value.shape[::-1] if path[-1] == "kernel" and value.ndim == 2 else value.shape
        if len(path) >= 4 and path[-4] == "experts" and path[-3].isdigit():
            leaf = (*path[:-3], path[-2], path[-1])
            assert tree[leaf][1:] == shape, name
            experts.setdefault(leaf, set()).add(int(path[-3]))
            continue
        assert path not in placed, (name, placed.get(path))
        assert tree[path] == shape, (name, tree[path], shape)
        placed[path] = name
    assert all(indices == set(range(tree[leaf][0])) for leaf, indices in experts.items())
    assert set(placed) | set(experts) == set(tree)
    assert (len(tensors), len(prepared) - towers, towers, len(tree)) == (497220, 249756, 168, 2736)


def test_situ_matches_the_released_activation():
    """SituAndMul at the released betas (4, 25) over gate and up in
    [-80, 80]: the fp32 product is reproduced to 1 ulp-scale rounding."""
    reference = np.load(TINY / "reference.npz")
    situ = Situ(4.0, 25.0)
    actual = situ(jnp.asarray(reference["situ_gate"]), jnp.asarray(reference["situ_up"]))
    np.testing.assert_allclose(np.asarray(actual), reference["situ"], rtol=2e-6, atol=1e-6)


def test_forward_matches_the_reference_over_left_padding(source):
    """fp32 logits over 70 tokens (past one KDA chunk), one row left-padded
    by 9, against fla 0.5.2 on CUDA with its intra-chunk solve in IEEE fp32
    (tools/kimi_k3_reference.py). The tolerance adds the two sides' fp32
    rounding, each measured against a float64 forward of these weights with
    the exact KDA recurrence: Dew's logits miss it by at most 2.2e-5 on CPU
    (1.6e-5 on the 4080 at matmul precision highest) and the fixture's by
    2.3e-5. Tolerance 5e-5; largest difference 4.4e-5 on CPU and 3.5e-5 on
    the 4080, argmax exact. At fla's default TF32 solve the fixture itself
    missed the float64 logits by up to 1.3e-4, which this tolerance refuses."""
    loaded, inputs, reference = source
    valid = reference["attention_mask"].astype(bool)
    logits = jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    np.testing.assert_allclose(np.asarray(logits)[valid], reference["logits"][valid], atol=5e-5, rtol=0)
    np.testing.assert_array_equal(np.asarray(logits)[valid].argmax(-1), reference["logits"][valid].argmax(-1))


def test_update_exports_the_trained_model_back_in_the_source_layout(source, tmp_path):
    """One all-parameter SGD step at the reference's learning rate (1e-2,
    which moves the logits by up to 4.2): loss within 1e-5, and updated
    logits within the sum of the two sides' distances from a float64 step.
    Those distances are the fp32 rounding of the gradient, most of it in
    layer 0's A_log, whose gradient sums 2,096 token and key-dimension terms
    to 1/70 of their absolute sum. Against the float64 step the updated
    logits miss by at most 5.1e-5 for Dew on CPU, 1.25e-4 for Dew on the
    4080 and 2.0e-4 for the fixture. Tolerance 3.5e-4; largest difference
    2.1e-4 on CPU and 2.3e-4 on the 4080. The export writes every source
    name back: towers byte-exact, A_log zero-padded to its stored length,
    trained experts as MXFP4 pairs, everything else as trained; reloading
    reproduces that."""
    loaded, inputs, reference = source
    objective = LMObjective(loaded.model, inputs.tokens.shape[1] - 1, pretrained=loaded.variables,
                            ema_decay=None, pad_id=0)
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    def loss(params):
        statistics, _ = objective.loss({**loaded.variables, "params": params}, {"text": inputs}, step)
        return objective.reduce_loss(statistics)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(loaded.variables["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    variables = {**loaded.variables, "params": jax.tree.map(
        lambda weight, grad: weight - reference["learning_rate"] * grad, loaded.variables["params"], gradient)}
    valid = reference["attention_mask"].astype(bool)
    updated = loaded.model.apply(variables, inputs.tokens, **inputs.kwargs())
    np.testing.assert_allclose(np.asarray(updated)[valid], reference["updated_logits"][valid], atol=3.5e-4, rtol=0)

    loaded.save(tmp_path, variables=variables)
    written, shipped = load_file(str(tmp_path / "model.safetensors")), load_file(str(TINY / "model.safetensors"))
    assert set(written) == set(shipped)
    for name in shipped:
        if name.startswith(("vision_tower.", "mm_projector.")):
            np.testing.assert_array_equal(written[name], shipped[name])
        if name.endswith("A_log"):
            assert written[name].shape == shipped[name].shape and not written[name][2:].any()
    assert written["language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"].dtype == np.uint8
    assert json.loads((tmp_path / "config.json").read_text()) == json.loads((TINY / "config.json").read_text())
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    trained = _flatten(variables)
    for name, after in _flatten(restored.variables).items():
        before = np.asarray(trained[name])
        if ".experts." in name:
            # Stacked [E, in, out]; compressed-tensors groups each [out, in] weight along its input.
            before = decode_e2m1(*quantize_packed_mxfp4(before.swapaxes(-1, -2))).swapaxes(-1, -2)
        np.testing.assert_array_equal(np.asarray(after), before, err_msg=name)


def test_greedy_generation_and_decode_steps_match_the_reference(source):
    """Four greedy tokens from both rows: the ids are exact, and each chosen
    token's log-probability under the cached KDA recurrence and MLA cache
    matches the reference's cached step logits."""
    loaded, inputs, reference = source
    generated = generate(loaded.model, loaded.variables, inputs, 4, key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens)[:, -4:], reference["generated"][:, -4:])
    expected = np.asarray(jax.nn.log_softmax(jnp.asarray(reference["step_logits"]), axis=-1))
    chosen = np.take_along_axis(expected, reference["generated"][:, -4:, None], -1)[..., 0]
    np.testing.assert_allclose(np.asarray(generated.raw_log_probs)[:, :4], chosen, atol=1e-4, rtol=0)


def test_prefill_and_token_steps_match_the_parallel_forward(source):
    loaded, inputs, _ = source
    model, variables = loaded.model, loaded.variables
    ids = inputs.tokens[:1]
    full = np.asarray(model.apply(variables, ids))
    state = model.apply(variables, 1, method="init_cache", mutable=["cache"])[1]
    out, state = model.apply({**variables, **state}, ids[:, :40], decode=True, mutable=["cache"])
    pieces = [np.asarray(out)]
    for index in range(40, ids.shape[1]):
        out, state = model.apply({**variables, **state}, ids[:, index:index + 1], decode=True, mutable=["cache"])
        pieces.append(np.asarray(out))
    np.testing.assert_allclose(np.concatenate(pieces, axis=1), full, atol=1e-4, rtol=0)


@pytest.mark.parametrize("field,value", [
    ("mla_use_nope", False), ("num_nextn_predict_layers", 1), ("hidden_act", "relu"),
    ("moe_router_activation_func", "tanh"), ("sliding_window", 128),
])
def test_text_config_outside_the_released_computation_is_refused(field, value):
    config = json.loads((SOURCE / "config.json").read_text())
    config["text_config"] = {**config["text_config"], field: value}
    with pytest.raises(ValueError, match=field):
        translate_config(config)


def test_a_quantization_scheme_other_than_mxfp4_is_refused(tmp_path):
    from shutil import copytree

    directory = copytree(TINY, tmp_path / "source")
    config = json.loads((directory / "config.json").read_text())
    config["text_config"]["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = 16
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="group_size"):
        load_pretrained(directory, dtype="float32", attention_impl="reference")


def test_a_nonzero_a_log_pad_is_refused():
    family = _FAMILIES["kimi_k3"]
    stem = "language_model.model.layers.0.self_attn."
    tensors = {stem + "A_log": np.array([0.5, 0.25, 0.0, 1.0], np.float32),
               stem + "dt_bias": np.zeros(8, np.float32), stem + "f_a_proj.weight": np.zeros((4, 3), np.float32)}
    with pytest.raises(ValueError, match="nonzero"):
        family.prepare_weights(tensors)
    tensors[stem + "A_log"][3] = 0
    np.testing.assert_array_equal(family.prepare_weights(tensors)[stem + "A_log"], [0.5, 0.25])
