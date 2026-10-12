"""Frozen-base and trainable-factor work, including the actual gradient matrix shapes."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from recording import RecordingTracker
from test_model_flops import decoder, forward_products, packed_batch

from dew.data.text import ByteTokenizer
from dew.decision.head import JointSchemaHead
from dew.decision.layout import Specials, StateFirstLayout
from dew.decision.objective import DecisionObjective
from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.interop import Pretrained
from dew.lora import LoRA
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.flops import MissingFlopsCount, counted_flops, dense_training_flops, projection_flops
from dew.nn.inputs import ModelInputs
from dew.nn.protocols import CountedTraining
from dew.objectives.base import FROZEN
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.training import Trainer
from dew.training.model_flops import FORWARD_FLOPS, TOKENS, TRAINING_FLOPS, prepare_counts


class DenseCount(nn.Module):
    inputs: int
    outputs: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.outputs, use_bias=False, name="layer")(x)

    @nn.nowrap
    def model_flops(self, lengths):
        return projection_flops(lengths, self.inputs, self.outputs)

    @nn.nowrap
    def parameter_flops(self, lengths):
        return self.model_flops(lengths)

    @nn.nowrap
    def matrix_usage(self, path, lengths, *, logits=False):
        return np.zeros(np.shape(lengths), np.int64) if logits else np.asarray(lengths)


def test_frozen_dense_rank_factors_equal_the_actual_forward_and_gradient_dots():
    model = DenseCount(8, 16)
    x = jnp.ones((2, 3, 8))
    variables = model.init(jax.random.key(401), x)
    adapted = LoRA(rank=2, alpha=2, modules=("layer",)).apply(model, variables, key=501)
    lengths = np.array([3, 3])
    expected = dense_training_flops(lengths, 8, 16, trainable=False, adapter_rank=2)
    assert adapted.model.training_flops(lengths) == expected

    def forward_backward(params, frozen, x):
        def forward(params, x):
            return adapted.model.apply({"params": params, FROZEN: frozen}, x)
        y, pullback = jax.vjp(forward, params, x)
        return y, *pullback(jnp.ones_like(y))

    assert expected == forward_products(forward_backward, adapted.variables["params"],
                                        adapted.variables[FROZEN], x)


@pytest.mark.parametrize("loop", [None, Loop(3, layers=(0, 1))])
def test_decoder_frozen_matrices_and_repeated_adapter_scopes_are_separate(loop):
    model = decoder(loop=loop)
    tokens = jnp.ones((2, 6), jnp.int32)
    variables = model.init(jax.random.key(402), tokens)
    adapted = LoRA(rank=2, alpha=4, modules=("q_proj", "down_proj")).apply(model, variables, key=502)
    lengths = np.array([3, 5])
    layer_passes = 5 if loop is not None else 3
    extra = 2 * 2 * ((8 + 8) + (16 + 8)) * 8 * layer_passes
    forward = model.model_flops(lengths)
    learned = model.parameter_flops(lengths)
    assert adapted.model.model_flops(lengths) == forward + extra
    assert adapted.model.training_flops(lengths) == 3 * forward - learned + 3 * extra
    assert adapted.model.logits_training_flops(lengths) == 2 * model.logits_flops(lengths)


def test_adapted_prediction_depths_count_their_maps_and_the_reused_head():
    model = decoder(num_nextn_predict_layers=2, attention_impl="reference", tie_embeddings=False)
    tokens = jnp.ones((1, 6), jnp.int32)
    variables = model.init(jax.random.key(409), tokens)
    adapted = LoRA(rank=2, alpha=4, modules=("q_proj", "eh_proj", "lm_head")).apply(
        model, variables, key=509)
    lengths = np.array([6], np.int64)
    # The input projection exists once per depth and the head is shared across both.
    predicted = 5 + 4
    factors = 2 * 2 * predicted * ((8 + 8) + (16 + 8) + (8 + model.vocab_size))
    native = model.prediction_flops(lengths)
    learned = model.prediction_parameter_flops(lengths)
    assert adapted.model.prediction_flops(lengths) == native + factors
    assert adapted.model.prediction_training_flops(lengths) == 3 * native - learned + 3 * factors

    def forward(variables, tokens):
        hidden = adapted.model.apply(variables, tokens, method="hidden_states")
        logits = adapted.model.apply(variables, hidden, method="logits_from_hidden")
        return logits, adapted.model.apply(variables, hidden, tokens, method="mtp_logits")

    objective = LMObjective(adapted.model, variables=adapted.variables, seq_len=6, mtp_weight=1.0)
    batch = {TEXT_KEY: np.ones((1, 7), np.int32)}
    upper = 2 * model.num_heads * model.head_dim * (model.num_layers * 6**2 + 5**2 + 4**2)
    assert objective.model_flops(None, batch) == forward_products(forward, adapted.variables, tokens) - upper
    assert objective.training_flops(None, batch) == (adapted.model.training_flops(lengths)
        + adapted.model.logits_training_flops(lengths) + adapted.model.prediction_training_flops(lengths))


def test_adapted_image_geometry_refuses_inherited_native_work_without_matrix_usage():
    model = SimpleDiT(emb_features=8, num_layers=1, num_heads=2, patch_size=2)
    image, time = jnp.ones((1, 4, 4, 3)), jnp.ones((1,))
    variables = model.init(jax.random.key(410), image, time)
    adapted = LoRA(rank=2, alpha=4, modules=("final_proj",)).apply(model, variables, key=510)
    objective = DiffusionObjective(adapted.model, presets.Flow(), InputSpec(Field("images", (4, 4, 3))),
                                   variables=adapted.variables, autoencoder=None, ema_decay=None)
    for count in (objective.model_flops, objective.training_flops):
        with pytest.raises(MissingFlopsCount, match="adapter matrix usage"):
            count(None, {"images": np.asarray(image)})


def test_objective_assembles_peft_training_work_without_three_times_forward():
    model = decoder()
    variables = model.init(jax.random.key(403), jnp.ones((2, 6), jnp.int32))
    adapted = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(model, variables, key=503)
    objective = LMObjective(adapted.model, variables=adapted.variables, seq_len=6, pad_id=0)
    batch = packed_batch()
    expected = (adapted.model.training_flops(np.array([2, 2, 3]))
                + adapted.model.logits_training_flops(np.array([2, 2])))
    assert objective.training_flops(None, batch) == expected
    counted = prepare_counts(objective, batch, count_model=True)
    assert counted[TRAINING_FLOPS].sum() == expected
    assert counted[FORWARD_FLOPS].sum() == objective.model_flops(None, batch)
    assert expected != 3 * counted[FORWARD_FLOPS].sum()
    assert counted[TOKENS].sum() == 7


def test_clef_head_remains_trainable_while_backbone_is_adapted():
    model = decoder()
    variables = model.init(jax.random.key(404), jnp.ones((2, 6), jnp.int32))
    adapted = LoRA(rank=2, modules=("q_proj",)).apply(model, variables, key=504)
    head = JointSchemaHead(8, width=8, heads=2, routing_layers=1, layers=1, feedforward=16)
    objective = DecisionObjective(adapted, head=head, tokenizer=ByteTokenizer(),
                                  specials=Specials(begin=None, separator=10, marker=0,
                                                    marker_text="\x00", pad=255),
                                  layout=StateFirstLayout(max_len=16))
    batch = {"valid": np.array([[True] * 3 + [False], [True] * 4]),
             "questions": np.ones((2, 1), bool), "options": np.ones((2, 1, 2), bool)}
    expected = (adapted.model.training_flops(np.array([3, 4]))
                + 3 * head.decision_flops(np.array([3, 4]), np.array([1, 1]), np.array([2, 2])))
    assert objective.training_flops(None, batch) == expected


def test_unknown_adapted_family_claims_no_work_and_an_explicit_count_is_refused():
    class Unknown(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(8, name="layer")(x)
    model = Unknown()
    variables = model.init(jax.random.key(405), jnp.ones((2, 8)))
    adapted = LoRA(rank=2, modules=("layer",)).apply(model, variables, key=505)
    assert not isinstance(adapted.model, CountedTraining)
    with pytest.raises(MissingFlopsCount, match="Unknown"):
        counted_flops(adapted.model, np.array([1, 1]))


def test_native_multimodal_qwen_adapter_counts_text_without_its_unused_image_tower():
    """The same wrapper/adapter shape c57 uses, without downloading a 9B model."""
    path = Path(__file__).parent / "fixtures" / "hf" / "qwen35-tiny-mm"
    source = Pretrained.load(str(path), dtype=jnp.float32)
    adapted = source.adapt(LoRA(rank=2, modules=("q_proj", "down_proj")), key=507)
    lengths = np.array([3, 5])
    original = source.model
    forward = original.model_flops(lengths)
    frozen_maps = original.parameter_flops(lengths)
    adapter_forward = adapted.model.model_flops(lengths) - forward
    assert adapter_forward > 0
    assert adapted.model.training_flops(lengths) == 3 * forward - frozen_maps + 3 * adapter_forward
    assert original.model_flops(lengths) == original.language_model.model_flops(lengths)


def test_unknown_media_work_can_count_tokens_without_claiming_a_model_flop_count():
    objective = LMObjective(decoder(), seq_len=6)
    batch = {"text": ModelInputs(np.ones((2, 7), np.int32),
                                conditioning={"pixels": np.ones((2, 1), np.float32)})}
    measured = prepare_counts(objective, batch, count_model=False)
    assert measured[TOKENS].sum() == 12 and TRAINING_FLOPS not in measured
    with pytest.raises(MissingFlopsCount, match="conditioning tower"):
        prepare_counts(objective, batch, count_model=True)


def test_trainer_reads_training_work_once_from_the_existing_interval_record(monkeypatch):
    from dew.nn.kernels import TARGETS, GpuTarget

    kind = jax.devices()[0].device_kind
    monkeypatch.setitem(TARGETS, kind, GpuTarget(kind, kind, dense_peaks={kind: {'bf16': 1e12}}))
    model = decoder().clone(num_layers=1)
    variables = model.init(jax.random.key(406), jnp.ones((8, 6), jnp.int32))
    adapted = LoRA(rank=2, modules=("q_proj",)).apply(model, variables, key=506)
    objective = LMObjective(adapted.model, variables=adapted.variables, seq_len=6, pad_id=0)
    tracker = RecordingTracker()

    class Batches:
        batch, records, val, steps_per_epoch = 8, 16, None, 2
        def train(self, partition):
            for length in (3, 5):
                tokens = np.broadcast_to((np.arange(7) < length).astype(np.int32), (8, 7))
                yield {"text": tokens}

    trainer = Trainer(objective, optax.sgd(0.01), key=0, accumulation=2, tracker=tracker)
    trainer.fit(Batches(), steps=2, log_every=2)
    ticks = [scalars for _, scalars in tracker.scalars if "train/mfu" in scalars]
    assert len(ticks) == 1 and ticks[0]["train/mfu"] > 0
    assert ticks[0]["train/tokens"] * 2 == 8 * (3 + 5)
