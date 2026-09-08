"""Waveforms through native Gemma audio into text logits and greedy decoding.

tests/fixtures/hf/gemma-3n-audio-tiny and gemma-4-audio-tiny come from
tools/audio_wrapper_reference.py. The reference is the real conditional
Transformers 5.16.1 model. Observed fp32 CPU logit errors at valid
positions: Gemma 3n 2.4e-7, Gemma 4 2.1e-7; the bound is 1e-4.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.data.audio import AudioProcessor
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.nn import vision as V
from dew.nn.audio import Gemma3nAudio, audio_config, audio_weights
from dew.registry import models, with_precision


FIXTURES = Path(__file__).parent / "fixtures" / "hf"
HIGHEST = jax.lax.Precision.HIGHEST


def _component(tensors, prefix):
    return {name.removeprefix(prefix): value for name, value in tensors.items() if name.startswith(prefix)}


class _Wrapper:
    """The reference composition for one family, on top of the native parts."""

    def __init__(self, name):
        self.path = FIXTURES / name
        self.config = json.loads((self.path / "config.json").read_text())
        tensors = load_file(str(self.path / "model.safetensors"))
        text_config = translate_config(self.config["text_config"])
        text = {"model." + key: value for key, value in _component(tensors, "model.language_model.").items()}
        text["lm_head.weight"] = tensors["lm_head.weight"]
        self.decoder = models.build("causal_transformer", **with_precision(
            "causal_transformer", text_config, dtype="float32", attention_impl="reference")).clone(precision=HIGHEST)
        self.text_variables = translate_weights(text, text_config)
        self.audio = audio_config(self.config["audio_config"])
        self.encoder = self.audio.build().clone(precision=HIGHEST)
        self.encoder_variables = audio_weights(_component(tensors, "model.audio_tower."), self.audio)
        embed_audio = _component(tensors, "model.embed_audio.")
        if isinstance(self.audio, Gemma3nAudio):
            record = self.config["audio_config"]
            self.projector = V.Gemma3nProjectorModule(
                self.audio.hidden_size, text_config["emb_features"], vocab_size=record["vocab_size"],
                vocab_offset=record["vocab_offset"], norm_eps=self.audio.rms_norm_eps, precision=HIGHEST)
            self.projector_variables = {"params": V.translate_gemma3n_projector_weights(embed_audio)}
            self.vision = V.projector_from_record(V.translate_gemma3n_projector_config(
                self.config, text_config["emb_features"])).build().clone(precision=HIGHEST)
            self.vision_variables = {"params": V.translate_gemma3n_projector_weights(
                _component(tensors, "model.embed_vision."))}
        else:
            self.projector = V.Gemma4ProjectorModule(text_width=text_config["emb_features"],
                                                     norm_eps=self.audio.rms_norm_eps, precision=HIGHEST)
            self.projector_variables = {"params": V.translate_gemma4_projector_weights(embed_audio)}

    def variables(self):
        return {"encoder": self.encoder_variables["params"], "projector": self.projector_variables["params"]}

    def logits(self, params, tokens, valid, features, feature_mask):
        """Reference order: hard vocabularies, then soft audio, then the decoder."""
        encoded = self.encoder.apply({**self.encoder_variables, "params": params["encoder"]}, features, feature_mask)
        projector_variables = {"params": params["projector"]}
        audio_id = self.config["audio_token_id"]
        rows = jnp.arange(tokens.shape[0])[:, None]
        if isinstance(self.audio, Gemma3nAudio):
            soft = jnp.asarray(self.projector.apply(projector_variables, encoded.features, method=self.projector.soft_embeddings))
            padding = jnp.asarray(self.projector.apply(
                projector_variables, jnp.array([[self.config["text_config"]["vocab_size"] - 1]], jnp.int32),
                method=self.projector.embed_hard))
            count = self.config["audio_soft_tokens_per_image"]
            soft = jnp.where(encoded.mask[..., None], soft, padding)
            soft = jnp.concatenate([soft, jnp.broadcast_to(padding, (soft.shape[0], count - soft.shape[1], soft.shape[2]))], 1)
            safe = jnp.where((tokens >= 0) & (tokens < self.decoder.per_layer_input_vocab), tokens, 0)
            embeddings = jnp.asarray(self.decoder.apply(self.text_variables, tokens, method=lambda m, t: m.embed_tokens(t)))
            embeddings = embeddings * jnp.sqrt(jnp.float32(self.decoder.emb_features))
            embeddings = jnp.asarray(self.vision.apply(self.vision_variables, embeddings, tokens, method=self.vision.merge_hard_embeddings))
            embeddings = jnp.asarray(self.projector.apply(projector_variables, embeddings, tokens, method=self.projector.merge_hard_embeddings))
            slots = jnp.argsort(jnp.where(tokens == audio_id, jnp.arange(tokens.shape[1]), tokens.shape[1]), axis=1)[:, :count]
            embeddings = embeddings.at[rows, slots].set(soft)
        else:
            soft = jnp.asarray(self.projector.apply(projector_variables, encoded.features))
            placeholders = (tokens == audio_id) | (tokens == self.config["image_token_id"]) | (tokens == self.config["video_token_id"])
            safe = jnp.where(placeholders, self.config["text_config"]["pad_token_id"], tokens)
            embeddings = jnp.asarray(self.decoder.apply(self.text_variables, safe, method=lambda m, t: m.embed_tokens(t)))
            embeddings = embeddings * jnp.sqrt(jnp.float32(self.decoder.emb_features))
            # Valid frames form a prefix; each row's slots take its own features in order.
            order = jnp.cumsum(tokens == audio_id, axis=1) - 1
            picked = soft[rows, jnp.maximum(order, 0)]
            embeddings = jnp.where((tokens == audio_id)[..., None], picked, embeddings)
        positions = jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0).astype(jnp.int32)
        return self.decoder.apply(self.text_variables, safe, input_embeddings=embeddings,
                                  embedding_positions=jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape),
                                  positions=positions, segment_ids=valid.astype(jnp.int32))


@pytest.fixture(params=("gemma-3n-audio-tiny", "gemma-4-audio-tiny"), scope="module")
def wrapper(request):
    return _Wrapper(request.param)


def _inputs(wrapper):
    reference = np.load(wrapper.path / "reference.npz")
    meta = json.loads((wrapper.path / "meta.json").read_text())
    processor = AudioProcessor(wrapper.config["audio_config"]["model_type"],
                               json.loads((wrapper.path / "processor_config.json").read_text())["feature_extractor"])
    features = processor([np.load(wrapper.path / f"waveform_{index}.npy") for index in range(2)],
                         sampling_rate=meta["sampling_rate"])
    np.testing.assert_array_equal(features["input_features_mask"], reference["input_features_mask"])
    np.testing.assert_allclose(features["input_features"], reference["input_features"], rtol=0, atol=1e-6)
    return reference, jnp.asarray(reference["input_ids"]), jnp.asarray(reference["attention_mask"].astype(bool)), \
        jnp.asarray(features["input_features"]), jnp.asarray(features["input_features_mask"])


def test_waveforms_condition_text_logits_like_the_reference_model(wrapper):
    reference, tokens, valid, features, feature_mask = _inputs(wrapper)
    error, logits = jax.jit(checkify.checkify(wrapper.logits))(
        wrapper.variables(), tokens, valid, features, feature_mask)
    error.throw()
    mask = np.asarray(valid)
    np.testing.assert_allclose(np.asarray(logits)[mask], reference["logits"][mask], rtol=0, atol=1e-4)
    np.testing.assert_array_equal(np.asarray(logits)[mask].argmax(-1), reference["logits"][mask].argmax(-1))


def test_greedy_continuation_follows_the_reference(wrapper):
    reference, tokens, valid, features, feature_mask = _inputs(wrapper)
    generated = reference["generated"]
    for step in range(generated.shape[1] - tokens.shape[1]):
        error, logits = jax.jit(checkify.checkify(wrapper.logits))(
            wrapper.variables(), tokens, valid, features, feature_mask)
        error.throw()
        next_token = jnp.argmax(logits[:, -1], axis=-1).astype(tokens.dtype)
        np.testing.assert_array_equal(np.asarray(next_token), generated[:, tokens.shape[1]])
        tokens = jnp.concatenate([tokens, next_token[:, None]], axis=1)
        valid = jnp.concatenate([valid, jnp.ones((tokens.shape[0], 1), jnp.bool_)], axis=1)


def test_text_loss_reaches_audio_encoder_and_projector_parameters(wrapper):
    reference, tokens, valid, features, feature_mask = _inputs(wrapper)
    targets = jnp.asarray(reference["generated"][:, tokens.shape[1]])

    def loss(params):
        logits = wrapper.logits(params, tokens, valid, features, feature_mask)
        return -jnp.mean(jax.nn.log_softmax(logits[:, -1])[jnp.arange(tokens.shape[0]), targets])

    error, grads = jax.jit(checkify.checkify(jax.grad(loss)))(wrapper.variables())
    error.throw()
    for component in ("encoder", "projector"):
        leaves = jax.tree.leaves(grads[component])
        assert all(bool(jnp.isfinite(leaf).all()) for leaf in leaves)
        assert max(float(jnp.abs(leaf).max()) for leaf in leaves) > 1e-6
