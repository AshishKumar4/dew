"""Source-matched Qwen3.8 checkpoints using their actual Qwen3.5 model types.

The source revisions, configs, templates and weight indexes live in
qwen38-source. The tiny fixtures come from Transformers 5.16.1 through
qwen_qualification_reference.py; no model weights are downloaded at test time.
"""

import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import parameter_loss

from dew.interop import Pretrained
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

ROOT = Path(__file__).parent / "fixtures" / "hf"



def stepped_and_reloaded(loaded, inputs, expected_loss, learning_rate: float, directory, **options):
    """One SGD step of the LM objective (`options`, its MTP weight) over `inputs`, the source's pad
    id masking its padding: its loss within 1e-5 of `expected_loss`, then the stepped weights saved
    to `directory` and loaded back. Returns the stepped variables and the reloaded source."""
    objective = LMObjective(loaded, inputs.tokens.shape[1] - 1, ema_decay=None,
                            pad_id=int(loaded.generation_config["pad_token_id"]), **options)
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)
    loss = parameter_loss(objective, loaded.variables, {"text": inputs}, step)
    value, gradient = jax.jit(jax.value_and_grad(loss))(loaded.variables["params"])
    np.testing.assert_allclose(value, expected_loss, atol=1e-5, rtol=0)
    variables = {**loaded.variables, "params": jax.tree.map(
        lambda weight, grad: weight - learning_rate * grad, loaded.variables["params"], gradient)}
    loaded.save(directory, variables=variables)
    return variables, Pretrained.load(directory, dtype="float32", attention_impl="reference")

def test_a_media_source_trains_and_saves_text_without_loading_media(monkeypatch, tmp_path):
    import optax
    from transformers import AutoProcessor

    from dew.data.dataset import Loader
    from dew.data.text import load_tokenizer
    from dew.training import Layout, MeshSpec, Trainer

    directory = ROOT / "qwen38-dense-tiny"
    tokenizer = load_tokenizer(str(directory), local_files_only=True)

    def load_media(*args, **kwargs):
        pytest.fail("text-only preprocessing constructed a media processor")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", load_media)
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="xla")
    processor = loaded.processor
    from dew.interop.sources import SourceReference

    assert processor is not None and processor.source == SourceReference(str(directory))
    texts = ["token4 token5", "token4 token5 token6"]
    expected = tokenizer(text=texts, padding=True, truncation=False, return_tensors="np")
    inputs = processor(texts)
    np.testing.assert_array_equal(inputs.tokens, expected["input_ids"])
    assert processor.decode(inputs.tokens) == tokenizer.batch_decode(
        expected["input_ids"], skip_special_tokens=True)
    np.testing.assert_array_equal(processor(texts[0]).tokens[0], tokenizer.encode(texts[0]))
    for content in ("token4", [{"type": "text", "text": "token4"}]):
        messages = [{"role": "user", "content": content}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert processor.reference.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True) == rendered
        expected_chat = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="np", add_generation_prompt=True)
        with monkeypatch.context() as torchless:
            # A text-only chat needs no Torch, as a text prompt needs none.
            torchless.setitem(sys.modules, "torch", None)
            chat = processor.chat(messages)
        np.testing.assert_array_equal(chat.tokens, expected_chat["input_ids"])

    objective = LMObjective(loaded, inputs.tokens.shape[1] - 1, ema_decay=None,
                            pad_id=tokenizer.pad_token_id)
    rows = 2 * jax.device_count()
    batch = inputs.take_rows(jnp.arange(rows) % 2)
    data = Loader(train=lambda partition: iter([{"text": batch}]), val=None, records=rows, batch=rows)
    state = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(0), mesh=MeshSpec(),
                    layout=Layout(min_shard=2**30)).fit(data, steps=1, log_every=1)
    assert any(np.any(np.asarray(before) != np.asarray(after)) for before, after in zip(
        jax.tree.leaves(loaded.variables), jax.tree.leaves(state.variables), strict=True))
    exported = tmp_path / "exported"
    loaded.save(exported, variables=state.variables)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        assert (exported / name).read_bytes() == (directory / name).read_bytes()
    restored = Pretrained.load(exported, dtype="float32", attention_impl="xla")
    assert restored.processor is not None
    np.testing.assert_array_equal(restored.processor(texts).tokens, inputs.tokens)
    assert restored.processor.decode(inputs.tokens) == processor.decode(inputs.tokens)


def test_media_requests_reuse_the_source_processor(monkeypatch):
    from transformers import AutoProcessor

    directory = ROOT / "qwen38-dense-tiny"
    references = []
    load = AutoProcessor.from_pretrained

    def load_media(*args, **kwargs):
        reference = load(*args, **kwargs)
        references.append(reference)
        return reference

    monkeypatch.setattr(AutoProcessor, "from_pretrained", load_media)
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    processor = loaded.processor
    assert processor is not None
    processor("token4")
    assert not references
    image = np.load(directory / "images.npy")[0]
    prompt = json.loads((directory / "prompts.json").read_text())[0]
    first = processor(prompt, images=[image])
    second = processor(prompt, images=[image])
    assert len(references) == 1
    np.testing.assert_array_equal(first.tokens, second.tokens)
    np.testing.assert_array_equal(first.conditioning["pixel_values"], second.conditioning["pixel_values"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": "token4"}]}]
    actual = processor.chat(messages)
    expected = references[0].apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
    np.testing.assert_array_equal(actual.tokens, expected["input_ids"].numpy())
    assert actual.conditioning
    assert len(references) == 1


@pytest.fixture(scope="module", params=["dense", "moe"])
def source(request):
    if request.param == "dense":
        pytest.importorskip("torchvision")
    directory = ROOT / f"qwen38-{request.param}-tiny"
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    assert loaded.processor is not None
    prompts = json.loads((directory / "prompts.json").read_text())
    if request.param == "dense":
        inputs = loaded.processor(prompts, images=[[image] for image in np.load(directory / "images.npy")])
    else:
        inputs = loaded.processor(prompts)
    return loaded, inputs, np.load(directory / "reference.npz")


def test_source_processor_and_forward_match_reference(source):
    """fp32 max error 7.04e-6 dense / 2.19e-6 MoE; tolerance 1e-4."""
    loaded, inputs, reference = source
    np.testing.assert_array_equal(inputs.tokens, reference["input_ids"])
    valid = np.asarray(inputs.token_fields["attention_mask"])
    result = jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(
        loaded.variables
    )
    np.testing.assert_allclose(np.asarray(result)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(result)[valid].argmax(-1), reference["logits"][valid].argmax(-1))


def test_source_update_exports_and_decodes_as_reference(source, tmp_path):
    """All-parameter SGD error 7.90e-6 dense / 3.46e-6 MoE; greedy ids exact.

    This catches reversed fused-expert transposes, incorrect shared gating
    gradients and source layouts that drop trained parameters on export.
    """
    loaded, inputs, reference = source
    variables, restored = stepped_and_reloaded(loaded, inputs, reference["loss"], reference["learning_rate"],
                                               tmp_path)
    for before, after in zip(jax.tree.leaves(variables), jax.tree.leaves(restored.variables), strict=True):
        np.testing.assert_array_equal(before, after)
    output = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(
        np.asarray(output)[valid], reference["updated_logits"][valid], atol=1e-4, rtol=0
    )
    generated = generate(loaded.model, loaded.variables, inputs, 3, key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], reference["generated"][:, -3:])


def test_shipped_prediction_weights_match_published_composition(source):
    """MTP error 4.06e-6 dense / 1.32e-6 MoE with the real HF decoder layer.

    Source glue is pinned in qwen_mtp_reference.py to vLLM's forward. The
    embedding/hidden norms cannot be swapped even when their shapes match.
    """
    loaded, inputs, _ = source
    reference = np.load(loaded.source / "mtp_reference.npz")
    hidden = loaded.model.apply(loaded.variables, inputs.tokens,
                                 method=loaded.model.hidden_states, **inputs.kwargs())
    predictions = loaded.model.apply(loaded.variables, hidden, inputs.tokens,
                                      method=loaded.model.mtp_logits, **inputs.kwargs())[0]
    valid = reference["valid"]
    np.testing.assert_allclose(np.asarray(predictions)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(
        np.asarray(predictions)[valid].argmax(-1), reference["logits"][valid].argmax(-1)
    )
    changed = jax.tree.map(lambda value: value, loaded.variables)
    text = changed["params"].get("language_model", changed["params"])
    depth = text["mtp_0"]
    depth["enorm"], depth["hnorm"] = depth["hnorm"], depth["enorm"]
    wrong = loaded.model.apply(changed, hidden, inputs.tokens,
                                method=loaded.model.mtp_logits, **inputs.kwargs())[0]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(wrong)[valid], reference["logits"][valid], atol=1e-4, rtol=0)


def test_prediction_cache_matches_reference_candidate_steps(source):
    """Independent cached MTP max errors 4.77e-7 dense / 4.18e-7 MoE."""
    loaded, inputs, _ = source
    reference = np.load(loaded.source / "mtp_reference.npz")
    positions = reference["positions"][..., 1:]
    rotary = np.moveaxis(positions, 0, -1) if positions.ndim == 3 else None
    state = loaded.model.apply(loaded.variables, inputs.tokens.shape[0],
                                method=loaded.model.init_mtp_cache, mutable=["cache"])[1]
    chunks = []
    for index in range(inputs.tokens.shape[1] - 1):
        options = {"attention_mask": jnp.asarray(reference["valid"][:, index:index + 1]),
                   "input_embeddings": jnp.asarray(reference["embeddings"][:, index + 1:index + 2])}
        if rotary is None:
            options["positions"] = jnp.asarray(positions[:, index:index + 1])
        else:
            options["rotary_positions"] = jnp.asarray(rotary[:, index:index + 1])
        (logits, _), state = loaded.model.apply(
            {**loaded.variables, **state}, jnp.asarray(reference["hidden"][:, index:index + 1]),
            inputs.tokens[:, index + 1:index + 2], decode=True,
            method=loaded.model.mtp_step, mutable=["cache"], **options)
        chunks.append(np.asarray(logits))
    result = np.concatenate(chunks, axis=1)
    valid = reference["valid"]
    np.testing.assert_allclose(result[valid], reference["logits"][valid], atol=1e-4, rtol=0)


def test_prediction_loss_respects_padding_and_exports_trained_depth(source, tmp_path):
    """Main+0.2*MTP CE loss error below 1e-6; post-update logits below 4.3e-6.

    The reference masks every consumed token of a depth's shifted target.
    Padded hidden states must not contribute auxiliary supervision.
    """
    loaded, inputs, _ = source
    reference = np.load(loaded.source / "mtp_reference.npz")
    _, restored = stepped_and_reloaded(loaded, inputs, reference["loss"], 1e-4, tmp_path, mtp_weight=0.2)
    hidden = restored.model.apply(restored.variables, inputs.tokens,
                                   method=restored.model.hidden_states, **inputs.kwargs())
    logits = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    predictions = restored.model.apply(restored.variables, hidden, inputs.tokens,
                                        method=restored.model.mtp_logits, **inputs.kwargs())[0]
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(
        np.asarray(logits)[valid], reference["updated_logits"][valid], atol=1e-4, rtol=0
    )
    np.testing.assert_allclose(np.asarray(predictions)[reference["valid"]],
                               reference["updated_mtp_logits"][reference["valid"]], atol=1e-4, rtol=0)



@pytest.fixture(scope="module")
def video_source():
    pytest.importorskip("torchvision")
    path = ROOT / "qwen38-dense-tiny"
    loaded = Pretrained.load(path, dtype="float32", attention_impl="reference")
    assert loaded.processor is not None
    raw = json.loads((path / "video_inputs.json").read_text())
    inputs = loaded.processor(raw["prompts"], images=[[image] for image in np.load(path / "images.npy")],
                              videos=[np.load(path / f"video_{index}.npy") for index in range(2)],
                              video_metadata=raw["metadata"])
    return loaded, inputs, np.load(path / "video_reference.npz")


def test_timestamped_videos_and_images_share_the_reference_frame_order(video_source):
    """Five- and three-frame clips pad their final temporal patch; grids differ.

    Interleaving image and video occurrences matches HF: logits 5.98e-6,
    MTP logits 3.97e-6, greedy tokens exact. A frame-order reversal fails.
    """
    loaded, inputs, reference = video_source
    np.testing.assert_array_equal(inputs.tokens, reference["input_ids"])
    valid = np.asarray(inputs.token_fields["attention_mask"])
    logits = loaded.model.apply(loaded.variables, inputs.tokens, **inputs.kwargs())
    hidden = loaded.model.apply(loaded.variables, inputs.tokens,
                                 method=loaded.model.hidden_states, **inputs.kwargs())
    mtp = loaded.model.apply(loaded.variables, hidden, inputs.tokens,
                              method=loaded.model.mtp_logits, **inputs.kwargs())[0]
    np.testing.assert_allclose(np.asarray(logits)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    admitted = valid[:, :-1] & valid[:, 1:]
    np.testing.assert_allclose(
        np.asarray(mtp)[admitted], reference["mtp_logits"][admitted], atol=1e-4, rtol=0
    )
    wrong = dataclasses.replace(inputs, conditioning={
        **inputs.conditioning, "pixel_values": inputs.conditioning["pixel_values"][:, ::-1]})
    moved = loaded.model.apply(loaded.variables, wrong.tokens, **wrong.kwargs())
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(moved)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    result = generate(loaded.model, loaded.variables, inputs, 3, key=jax.random.key(1),
                       sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(result.tokens[:, -3:], reference["generated"][:, -3:])


def test_video_prediction_training_exports_the_reference_update(video_source, tmp_path):
    """Main+MTP video update error 8.80e-6 with unchanged 1e-4 tolerance.

    These 78 tokens cross a DeltaNet chunk. The former post-exp-only mask
    gave finite forward values but non-finite gradients and exported weights.
    """
    loaded, inputs, reference = video_source
    _, restored = stepped_and_reloaded(loaded, inputs, reference["loss"], 1e-4, tmp_path, mtp_weight=0.2)
    logits = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(
        np.asarray(logits)[valid], reference["updated_logits"][valid], atol=1e-4, rtol=0
    )


def test_actual_checkpoint_default_nucleus_policy_reaches_the_native_draw(source):
    """Both published generation configs specify top-k 20 and top-p 0.95.

    Validate the selected action's filtered likelihood against Transformers,
    rather than merely inspecting the parsed fields.
    """
    import torch
    from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

    loaded, inputs, _ = source
    task = loaded.text_generation()
    actual = task(inputs, 1, key=jax.random.key(91))
    logits = loaded.model.apply(loaded.variables, inputs.tokens, **inputs.kwargs())[:, -1]
    scores = torch.FloatTensor(np.asarray(logits).copy())
    ids = torch.LongTensor(np.asarray(inputs.tokens).copy())
    scores = TopKLogitsWarper(20)(ids, scores)
    scores = TopPLogitsWarper(0.95)(ids, scores)
    token = np.asarray(actual.tokens[:, -1])
    expected = scores.log_softmax(-1).numpy()[np.arange(token.shape[0]), token]
    np.testing.assert_allclose(actual.behavior_log_probs[:, 0], expected, atol=1e-5, rtol=0)
    repeated = task(inputs, 1, key=jax.random.key(91))
    np.testing.assert_array_equal(actual.tokens, repeated.tokens)
