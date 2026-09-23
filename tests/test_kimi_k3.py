"""Kimi K3's text decoder against the released remote code.

kimi-k3-source holds the released config and the name, dtype and shape of
every tensor of moonshotai/Kimi-K3 at f831ab6 (read off the 96 shard
headers). kimi-k3-tiny is written by tools/kimi_k3_reference.py from the
pinned modeling_kimi_linear.py with fla-core 0.5.2's kernels in fp32 on
CUDA; its routed experts are compressed-tensors MXFP4 as in the release.
Its numerics.npz is the same model evaluated in float64
(tools/kimi_float64_reference.py), which Dew and the reference are both
measured from (tests/reference_error.py). No model weights are downloaded
at test time.
"""

import json
import lzma
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import FACTOR, assert_as_exact_as_the_reference
from safetensors.numpy import load_file
from scipy.special import log_softmax

from dew.interop import load_pretrained
from dew.interop.codecs import PACKED_MXFP4, decode_e2m1, quantize_packed_mxfp4
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
    with np.load(TINY / "reference.npz") as stored, np.load(TINY / "numerics.npz") as exact:
        reference = {name: stored[name] for name in stored.files} | {name: exact[name] for name in exact.files}
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
    assert set(PACKED_MXFP4.tensor_names(dict.fromkeys(tensors))) == set(decoded)
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
    (tools/kimi_k3_reference.py): Dew's RMS distance from the float64
    logits at most twice the reference's, and argmax exact. Observed RMS
    from float64: Dew 1.67e-6, the reference 1.97e-6 (ratio 0.85 on CPU,
    0.87 on an RTX 4080, 0.89 on an RTX 3090)."""
    loaded, inputs, reference = source
    valid = reference["attention_mask"].astype(bool)
    logits = jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    assert_as_exact_as_the_reference(np.asarray(logits)[valid], reference["logits"][valid],
                                     reference["logits_f64"][valid], "logits")
    np.testing.assert_array_equal(np.asarray(logits)[valid].argmax(-1), reference["logits"][valid].argmax(-1))


def test_update_exports_the_trained_model_back_in_the_source_layout(source, tmp_path):
    """One all-parameter SGD step at the reference's learning rate (1e-2,
    which moves the logits by up to 4.2). The loss is within 1e-5 of the
    reference's (9.5e-7 apart), and the updated logits are held to the
    reference's own distance from a float64 step as the forward is.
    Observed RMS from float64: Dew 5.57e-6, the reference 8.15e-6 (ratio
    0.68 on CPU, 0.91 on an RTX 4080, 1.26 on an RTX 3090).

    The export writes every source name back: towers byte-exact, A_log
    zero-padded to its stored length, trained experts as MXFP4 pairs,
    everything else as trained; reloading reproduces that."""
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
    assert_as_exact_as_the_reference(np.asarray(updated)[valid], reference["updated_logits"][valid],
                                     reference["updated_logits_f64"][valid], "updated logits")

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
    """Four greedy tokens from both rows through the cached KDA recurrence and
    MLA cache, held as tests/test_generate_reference.py holds greedy decoding.
    E is the reference's largest decode logit error from the float64 logits
    teacher-forced over the same path (2.28e-5). Every step's float64 top-2
    margin exceeds 2 FACTOR E (smallest 6.2e-4 against 9.1e-5), so the ids
    must be exact, and each chosen token's log probability is within 2
    FACTOR E of float64's (observed 3.1e-6)."""
    loaded, inputs, reference = source
    error = float(np.max(np.abs(reference["step_logits"] - reference["step_logits_f64"])))
    ranked = np.sort(reference["step_logits_f64"], -1)
    assert np.min(ranked[..., -1] - ranked[..., -2]) > 2 * FACTOR * error
    generated = generate(loaded.model, loaded.variables, inputs, 4, key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens)[:, -4:], reference["generated"][:, -4:])
    exact = np.take_along_axis(log_softmax(reference["step_logits_f64"], -1),
                               reference["generated"][:, -4:, None], -1)[..., 0]
    assert np.max(np.abs(np.asarray(generated.raw_log_probs, np.float64)[:, :4] - exact)) <= 2 * FACTOR * error


def test_cached_prefill_and_token_steps_are_as_exact_as_the_reference(source):
    """Row 0 prefilled for 40 tokens, then decoded one token at a time through
    the KDA state and the MLA cache: its logits are held to the reference
    forward's distance from float64, as the parallel forward is. Observed
    RMS from float64: Dew 1.39e-6, the reference 2.09e-6 (ratio 0.66 on CPU)."""
    loaded, inputs, reference = source
    model, variables = loaded.model, loaded.variables
    ids = inputs.tokens[:1]
    state = model.apply(variables, 1, method="init_cache", mutable=["cache"])[1]
    out, state = model.apply({**variables, **state}, ids[:, :40], decode=True, mutable=["cache"])
    pieces = [np.asarray(out)]
    for index in range(40, ids.shape[1]):
        out, state = model.apply({**variables, **state}, ids[:, index:index + 1], decode=True, mutable=["cache"])
        pieces.append(np.asarray(out))
    assert_as_exact_as_the_reference(np.concatenate(pieces, axis=1)[0], reference["logits"][0],
                                     reference["logits_f64"][0], "cached decode")


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
