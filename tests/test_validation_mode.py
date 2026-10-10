"""Validation turns off model dropout while keeping the objective's random draws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffusion_stubs import StubText
from flax import linen as nn

from dew.data import ByteTokenizer
from dew.decision import Choice, DecisionObjective, Example, Specials, StateFirstLayout
from dew.decision.objective import Encoding
from dew.diffusion import presets
from dew.diffusion.discrete import MDLM
from dew.diffusion.process import DenoisingCondition
from dew.inputs import Condition, Field, InputSpec
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.jepa import JepaEncoder, JepaPredictor
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.objectives import DistillationObjective
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionObjective, GuidanceDistillationObjective
from dew.objectives.diffusion.block import BlockDiffusionObjective
from dew.objectives.diffusion.few_step import MeanFlowObjective, ShortcutObjective
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.jepa import JepaObjective, MultiBlockMask
from dew.objectives.lm import LMObjective
from dew.objectives.supervised import Supervised


def decoder(dropout, **fields):
    return CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2,
                             mlp_features=16, max_seq_len=16, dropout_rate=dropout,
                             attention_impl="reference", **fields)


def validation(objective, variables, batch, step):
    return objective.reduce_loss(objective._validation_loss(variables, batch, step))[0]


def assert_key_independent_validation(objective, variables, batch):
    steps = [Step(jnp.asarray(0), jax.random.key(key), None) for key in (1, 2)]
    first, second = [validation(objective, variables, batch, step) for step in steps]
    np.testing.assert_array_equal(first, second)
    training = [objective.scalar_loss(variables, batch, step)[0] for step in steps]
    assert training[0] != training[1]


@pytest.mark.parametrize("distilled", [False, True], ids=["lm", "distillation"])
def test_token_validation_disables_dropout(distilled):
    objective = LMObjective(decoder(0.5), seq_len=4, ema_decay=None)
    if distilled:
        objective = DistillationObjective(objective, LMObjective(decoder(0.5), seq_len=4, ema_decay=None))
    variables = objective.init(jax.random.key(0))
    batch = {"text": jnp.arange(20, dtype=jnp.int32).reshape(4, 5) % 16}
    assert_key_independent_validation(objective, variables, batch)


class Regressor(nn.Module):
    @nn.compact
    def __call__(self, x, train=False):
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.Dense(2)(nn.Dropout(0.5)(x, deterministic=not train))


def test_supervised_validation_disables_dropout_and_keeps_running_statistics():
    objective = Supervised(Regressor(), lambda output, batch: (output - batch["target"]) ** 2,
                           inputs=InputSpec(Field("x", (3,))), mode="train")
    variables = objective.init(jax.random.key(0))
    batch = {"x": jnp.arange(24, dtype=jnp.float32).reshape(8, 3), "target": jnp.ones((8, 2))}
    before = jax.tree.map(np.asarray, variables)
    assert_key_independent_validation(objective, variables, batch)
    _, aux = objective.validation_loss(variables, batch, Step(jnp.asarray(0), jax.random.key(1), None))
    assert aux.variables is None
    for got, expected in zip(jax.tree.leaves(variables), jax.tree.leaves(before), strict=True):
        np.testing.assert_array_equal(got, expected)


def test_decision_validation_disables_dropout():
    tokenizer = ByteTokenizer()
    specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    layout = StateFirstLayout(max_len=32, head_max_len=16, option_tokens=2)
    model = decoder(0.5).clone(vocab_size=256, max_seq_len=32)
    objective = DecisionObjective(model, tokenizer=tokenizer, specials=specials, layout=layout)
    encode = Encoding(layout, tokenizer, specials, width=2)
    question = Choice("Pick", {"a": "one", "b": "two"})
    rows = [encode(Example(f"row {index}", {"q": question}, {"q": "a"}), ("q",), None)
            for index in range(4)]
    batch = {name: jnp.asarray(np.stack([row[name] for row in rows])) for name in rows[0]}
    assert_key_independent_validation(objective, objective.init(jax.random.key(0)), batch)


def assert_dropout_disabled(dropped, plain, variables, batch):
    """The same step keeps corruption and masks identical across the two models."""
    step = Step(jnp.asarray(0), jax.random.key(1), variables)
    expected = validation(plain, variables, batch, step)
    np.testing.assert_array_equal(validation(dropped, variables, batch, step), expected)
    assert dropped.scalar_loss(variables, batch, step)[0] != plain.scalar_loss(variables, batch, step)[0]


@pytest.mark.parametrize("part", ["encoder", "predictor"])
def test_jepa_validation_disables_dropout(part):
    mask = MultiBlockMask.for_grid((4, 4), num_targets=2, scale=(0.2, 0.3))

    def objective(dropout):
        encoder = JepaEncoder(patch_size=2, emb_features=8, num_layers=1, num_heads=2,
                              dropout_rate=dropout if part == "encoder" else 0)
        predictor = JepaPredictor(grid=(4, 4), emb_features=8, predictor_features=8,
                                  num_layers=1, num_heads=2,
                                  dropout_rate=dropout if part == "predictor" else 0)
        return JepaObjective(encoder, predictor, mask, sample=Field("image", (8, 8, 3)))

    dropped, plain = objective(0.5), objective(0)
    batch = {"image": jnp.arange(4 * 8 * 8 * 3, dtype=jnp.float32).reshape(4, 8, 8, 3) % 256}
    assert_dropout_disabled(dropped, plain, dropped.init(jax.random.key(0)), batch)


@pytest.mark.parametrize("family", ["masked", "block"])
def test_discrete_diffusion_validation_disables_dropout(family):
    def objective(dropout):
        if family == "masked":
            return MaskedDiffusionObjective(decoder(dropout, causal=False), MDLM(mask_id=15)(),
                                            seq_len=6, ema_decay=None)
        return BlockDiffusionObjective(DiffusionGemma(decoder(dropout, layer_scalar="trainable"), 2),
                                        prompt_length=2, num_canvases=2)

    dropped, plain = objective(0.5), objective(0)
    batch = {"text": jnp.arange(24, dtype=jnp.int32).reshape(4, 6) % 14 + 1}
    assert_dropout_disabled(dropped, plain, dropped.init(jax.random.key(0)), batch)


class PixelModel(nn.Module):
    """A pixel projection with time, interval and guidance inputs."""

    dropout_rate: float
    interval = True

    @nn.compact
    def __call__(self, x, time, conditioning=None, duration=None, train=False):
        if not self.is_initializing() and not train:
            assert not self.has_rng("dropout")
            assert not self.has_rng("stochastic_rounding")
        h = nn.Dense(8)(x) + time[:, None, None, None] / 1000
        if duration is not None:
            h = h + duration[:, None, None, None] / 1000
        if conditioning is not None:
            h = h + jnp.mean(conditioning.context, axis=(1, 2))[:, None, None, None]
            h = h + conditioning.guidance[:, None, None, None]
        h = nn.Dropout(self.dropout_rate)(nn.tanh(h), deterministic=not train)
        return nn.Dense(x.shape[-1])(h)


class GuidanceText(StubText):
    def encode(self, params, tokens):
        text = super().encode(params, tokens)
        return DenoisingCondition(context=text.hidden, mask=text.mask,
                                  guidance=jnp.ones((text.hidden.shape[0],)))


@pytest.mark.parametrize("family", ["pixel", "mean_flow", "shortcut", "guidance"])
def test_pixel_diffusion_validation_disables_dropout(family):
    inputs = InputSpec(Field("image", (2, 2, 3)))
    batch = {"image": jnp.arange(48, dtype=jnp.float32).reshape(4, 2, 2, 3)}
    if family == "guidance":
        inputs = InputSpec(inputs.sample, {"conditioning": Condition(GuidanceText.from_pretrained("stub"))})
        batch.update(inputs.tokenize(["a", "b", "c", "d"]))

    def objective(dropout):
        model = PixelModel(dropout)
        if family == "mean_flow":
            return MeanFlowObjective(model, presets.MeanFlow(), inputs, norm_p=0)
        if family == "shortcut":
            return ShortcutObjective(model, presets.Shortcut(), inputs, sections=4, bootstrap_every=2)
        if family == "guidance":
            teacher = DiffusionObjective(PixelModel(0.5), presets.Flow(), inputs, guidance=None)
            return GuidanceDistillationObjective(model, presets.Flow(), inputs,
                                                 variables=teacher.init(jax.random.key(3)))
        return DiffusionObjective(model, presets.Flow(), inputs, guidance=None)

    dropped, plain = objective(0.5), objective(0)
    variables = dropped.init(jax.random.key(0))
    assert_dropout_disabled(dropped, plain, variables, batch)


def test_diffusion_validation_keeps_the_evaluation_noise():
    objective = DiffusionObjective(PixelModel(0), presets.Flow(), InputSpec(Field("image", (2, 2, 3))),
                                   guidance=None)
    variables = objective.init(jax.random.key(0))
    batch = {"image": jnp.arange(48, dtype=jnp.float32).reshape(4, 2, 2, 3)}
    values = [validation(objective, variables, batch, Step(jnp.asarray(0), jax.random.key(key), None))
              for key in (1, 2)]
    assert values[0] != values[1]
