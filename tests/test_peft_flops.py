"""Frozen-base and trainable-factor work, including the actual gradient matrix shapes."""

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
from dew.lora import LoRA
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.flops import MissingFlopsCount, dense_training_flops, projection_flops
from dew.objectives.lm import LMObjective
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
            return adapted.model.apply({"params": params, "FROZEN": frozen}, x)
        y, pullback = jax.vjp(forward, params, x)
        return y, *pullback(jnp.ones_like(y))

    assert expected == forward_products(forward_backward, adapted.variables["params"],
                                        adapted.variables["FROZEN"], x)


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


def test_objective_assembles_peft_training_work_without_three_times_forward():
    model = decoder()
    variables = model.init(jax.random.key(403), jnp.ones((2, 6), jnp.int32))
    adapted = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(model, variables, key=503)
    objective = LMObjective(adapted, seq_len=6, pad_id=0)
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


def test_unknown_adapted_family_is_refused_instead_of_being_counted_as_full_ft():
    class Unknown(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(8, name="layer")(x)
    model = Unknown()
    variables = model.init(jax.random.key(405), jnp.ones((2, 8)))
    adapted = LoRA(rank=2, modules=("layer",)).apply(model, variables, key=505)
    with pytest.raises(MissingFlopsCount, match="Unknown"):
        adapted.model.training_flops(np.array([1, 1]))


def test_trainer_reads_training_work_once_from_the_existing_interval_record(monkeypatch):
    import dew.training.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "peak_flops", lambda device: 1e12)
    model = decoder().clone(num_layers=1)
    variables = model.init(jax.random.key(406), jnp.ones((8, 6), jnp.int32))
    adapted = LoRA(rank=2, modules=("q_proj",)).apply(model, variables, key=506)
    objective = LMObjective(adapted, seq_len=6, pad_id=0)
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
