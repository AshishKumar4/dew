"""Source-matched Qwen3.8 checkpoints using their actual Qwen3.5 model types.

The source revisions, configs, templates and weight indexes live in
qwen38-source. The tiny fixtures come from Transformers 5.16.1 through
qwen_qualification_reference.py; no model weights are downloaded at test time.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.hf_decoders import translate_config, translate_wrapper_config
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling import Sampling, generate

ROOT = Path(__file__).parent / "fixtures" / "hf"


@pytest.fixture(scope="module", params=["dense", "moe"])
def source(request):
    if request.param == "dense":
        pytest.importorskip("torchvision")
    directory = ROOT / f"qwen38-{request.param}-tiny"
    loaded = load_pretrained(directory, dtype="float32", attention_impl="reference")
    assert loaded.processor is not None
    prompts = json.loads((directory / "prompts.json").read_text())
    if request.param == "dense":
        inputs = loaded.processor(prompts, images=[[image] for image in np.load(directory / "images.npy")])
    else:
        inputs = loaded.processor(prompts)
    return loaded, inputs, np.load(directory / "reference.npz")


def test_source_geometry_is_configurable_without_allocating_released_weights():
    dense = translate_wrapper_config(json.loads((ROOT / "qwen38-source/dense/config.json").read_text()))
    moe = translate_config(json.loads((ROOT / "qwen38-source/moe/config.json").read_text()))
    dense_model = models.build("causal_transformer", **dense["text"])
    moe_model = models.build("causal_transformer", **moe)
    assert dense_model.num_layers == 64 and dense_model.mixture is None
    assert moe_model.num_layers == 92
    assert moe_model.mixture.experts == 512 and moe_model.mixture.top_k == 10
    assert moe_model.mixture.shared_gate


def test_source_processor_and_forward_match_reference(source):
    """fp32 max error 7.04e-6 dense / 2.19e-6 MoE; tolerance 1e-4."""
    loaded, inputs, reference = source
    np.testing.assert_array_equal(inputs.tokens, reference["input_ids"])
    valid = np.asarray(inputs.token_fields["attention_mask"])
    result = jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    np.testing.assert_allclose(np.asarray(result)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(result)[valid].argmax(-1), reference["logits"][valid].argmax(-1))


def test_source_update_exports_and_decodes_as_reference(source, tmp_path):
    """All-parameter SGD error 7.90e-6 dense / 3.46e-6 MoE; greedy ids exact.

    This catches reversed fused-expert transposes, incorrect shared gating
    gradients and source layouts that drop trained parameters on export.
    """
    loaded, inputs, reference = source
    objective = LMObjective(loaded.model, inputs.tokens.shape[1] - 1,
                            pretrained=loaded.variables, ema_decay=None,
                            pad_id=int(loaded.generation_config["pad_token_id"]))
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    def loss(params):
        statistics, _ = objective.loss({**loaded.variables, "params": params}, {"text": inputs}, step)
        return objective.reduce_loss(statistics)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(loaded.variables["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    variables = {**loaded.variables, "params": jax.tree.map(
        lambda weight, grad: weight - reference["learning_rate"] * grad,
        loaded.variables["params"], gradient)}
    loaded.save(tmp_path, variables=variables)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    for before, after in zip(jax.tree.leaves(variables), jax.tree.leaves(restored.variables), strict=True):
        np.testing.assert_array_equal(before, after)
    output = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(np.asarray(output)[valid], reference["updated_logits"][valid], atol=1e-4, rtol=0)
    generated = generate(loaded.model, loaded.variables, inputs, 3, key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], reference["generated"][:, -3:])


def test_moe_shared_gate_cannot_be_replaced_by_an_ungated_branch():
    loaded = load_pretrained(ROOT / "qwen38-moe-tiny", dtype="float32", attention_impl="reference")
    reference = np.load(ROOT / "qwen38-moe-tiny/reference.npz")
    wrong = loaded.model.clone(mixture=dataclasses.replace(loaded.model.mixture, shared_gate=False))
    output = wrong.apply(loaded.variables, reference["input_ids"], attention_mask=reference["attention_mask"].astype(bool))
    valid = reference["attention_mask"].astype(bool)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(output)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
