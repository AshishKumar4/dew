"""Kimi Linear against its released remote code.

kimi-linear-source holds the released config of
moonshotai/Kimi-Linear-48B-A3B-Instruct at e1df551 and the name, dtype and
shape of every tensor (read off the 20 shard headers). kimi-linear-tiny is
written by tools/kimi_linear_reference.py from the pinned modeling_kimi.py
with fla-core 0.4.0's kernels in fp32 on CUDA, its gate weighing the
released selection by the unbiased scores, as K3's revision of the file and
vLLM do. No model weights are downloaded at test time.
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
from dew.interop.hf_decoders import _FAMILIES, _flatten, translate_config
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling import Sampling, generate

ROOT = Path(__file__).parent / "fixtures" / "hf"
SOURCE = ROOT / "kimi-linear-source"
TINY = ROOT / "kimi-linear-tiny"


@pytest.fixture(scope="module")
def source():
    loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
    reference = np.load(TINY / "reference.npz")
    inputs = ModelInputs(jnp.asarray(reference["input_ids"], jnp.int32),
                         {"attention_mask": jnp.asarray(reference["attention_mask"], bool)})
    return loaded, inputs, reference


def released_config() -> dict:
    return json.loads((SOURCE / "config.json").read_text())


def test_released_config_builds_the_published_geometry():
    model = models.build("causal_transformer", translate_config(released_config()))
    assert model.num_layers == 27 and model.emb_features == 2304
    assert model.per_layer_types.count("full_attention") == 7 and model.per_layer_types[-1] == "full_attention"
    kda, mla = model.kinds["linear_attention"].mixer, model.kinds["full_attention"].mixer
    assert kda.linear_lower_bound is None and not kda.use_full_rank_gate
    assert mla.q_lora_rank is None and mla.mla_use_nope and not mla.mla_use_output_gate
    assert model.mixture.experts == 256 and model.mixture.top_k == 8 and model.mixture.bias
    assert model.sparse_layers == tuple(range(1, 27)) and model.mixture.groups == 1
    assert (model.mixture.expert_features, model.mixture.shared_features, model.mixture.scaling) == (1024, 1024, 2.446)
    assert model.mlp == "swiglu" and model.attention_residuals is None


def test_every_released_tensor_lands_on_one_leaf_of_the_released_tree():
    """All 20,493 source tensors land on distinct leaves of the tree the
    released config builds, each at the leaf's shape (A_log read from its
    stored [1, 1, 32, 1] as the 32 heads), and cover all 603 of its leaves,
    before any weight would be read."""
    fields, family = translate_config(released_config()), _FAMILIES["kimi_linear"]
    model = models.build("causal_transformer", fields)
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))
    tree = {tuple(name.split(".")): leaf.shape for name, leaf in _flatten(dict(shapes)).items()}
    tensors = json.loads(lzma.decompress((SOURCE / "tensors.json.xz").read_bytes()))
    prepared = family.prepare_weights({name: np.broadcast_to(np.float32(0), shape)
                                       for name, (_, shape) in tensors.items()})
    placed, experts = {}, {}
    for name, value in prepared.items():
        path = family.weight_path(name, fields)
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
    assert (len(tensors), len(tree)) == (20493, 603)


def test_forward_matches_the_reference_over_left_padding(source):
    """fp32 logits over 70 tokens (past one KDA chunk), one row left-padded
    by 9, against fla 0.4.0 on CUDA in IEEE fp32
    (tools/kimi_linear_reference.py). The tolerance adds the two sides' fp32
    rounding, each measured against a float64 forward of these weights with
    the exact KDA recurrence: Dew's logits miss it by at most 1.3e-5 on CPU
    (1.0e-5 on the 4080 at matmul precision highest) and the fixture's by
    1.2e-5. Tolerance 3e-5; largest difference 1.5e-5 on CPU and 1.4e-5 on
    the 4080, argmax exact."""
    loaded, inputs, reference = source
    logits = loaded.model.apply(loaded.variables, inputs.tokens, **inputs.kwargs())
    valid = reference["attention_mask"].astype(bool)
    np.testing.assert_allclose(np.asarray(logits)[valid], reference["logits"][valid], atol=3e-5, rtol=0)
    np.testing.assert_array_equal(np.asarray(logits)[valid].argmax(-1), reference["logits"][valid].argmax(-1))


def test_update_exports_the_trained_model_back_in_the_source_layout(source, tmp_path):
    """One all-parameter SGD step at the reference's learning rate (1e-2,
    which moves the logits by up to 3.8): loss within 1e-5, and updated
    logits within the sum of the two sides' distances from a float64 step,
    which are the fp32 rounding of the gradient: 1.5e-4 for Dew on CPU,
    1.4e-5 for Dew on the 4080 and 1.5e-4 for the fixture, the two larger
    at the same token, where the float64 step's own code run in float32
    misses by 8.3e-5. Tolerance 3.5e-4; largest difference 4.3e-5 on CPU and 1.5e-4
    on the 4080. The export writes every source name back at its stored
    shape, A_log as [1, 1, heads, 1], and reloading restores the trained
    weights exactly."""
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
    assert {name: value.shape for name, value in written.items()} == {name: value.shape for name, value in shipped.items()}
    assert written["model.layers.0.self_attn.A_log"].shape == (1, 1, 2, 1)
    assert json.loads((tmp_path / "config.json").read_text()) == json.loads((TINY / "config.json").read_text())
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    trained = _flatten(variables)
    for name, after in _flatten(restored.variables).items():
        np.testing.assert_array_equal(np.asarray(after), np.asarray(trained[name]), err_msg=name)


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


def test_group_limited_routing_is_refused():
    """The released gate fills the groups it does not select with 0.0, not
    -inf (modeling_kimi.py:684-685), so their experts still compete, where
    Dew's grouped router masks them. The release routes in one group,
    num_expert_group 1 of topk_group 1, which never reaches that."""
    config = released_config()
    assert (config["num_expert_group"], config["topk_group"]) == (1, 1)
    config.update(num_expert_group=8, topk_group=4)
    with pytest.raises(ValueError, match="num_expert_group 8 over topk_group 4"):
        translate_config(config)


@pytest.mark.parametrize("field,value", [
    ("q_lora_rank", 32), ("hidden_act", "situ"), ("attn_res_block_size", 4),
    ("mla_use_output_gate", True), ("routed_expert_hidden_size", 512),
])
def test_config_outside_the_released_computation_is_refused(field, value):
    """A low-rank query, SiTU and K3's additions are what the released
    modeling_kimi.py does not compute."""
    config = released_config()
    config[field] = value
    with pytest.raises(ValueError, match=field):
        translate_config(config)
