"""Every registered objective as its recipe or example builds it, with the
first batch the data source that recipe trains on writes (`cases`), never
one drawn from its declaration (`Objective.inputs`). The trainer checks
that batch against the declaration before compiling, so a suite that runs
each case (test_whole_validation) fails on a declaration that drifts from
its data rather than at a user's first step.
"""

from __future__ import annotations

import json

import jax
from decision_support import decision_rows
from objective_data import IMAGES, SEQ_LEN, first
from rl_cases import packed

from dew.config import ObjectiveConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.config.jepa import JepaRunConfig
from dew.data import TFDS, Memory, images as image_steps, presets
from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import Choice, DecisionObjective, Example, Specials, StateFirstLayout
from dew.diffusion.discrete import MDLM
from dew.diffusion.presets import Flow, MeanFlow, Shortcut
from dew.inputs import Field, InputSpec, TextCondition
from dew.nn.backbones import CausalTransformer, SimpleDiT
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.objectives import DistillationObjective
from dew.objectives.base import TEACHER
from dew.objectives.diffusion import (
    AdversarialDistillationObjective,
    ConsistencyDistillationObjective,
    GuidanceDistillationObjective,
)
from dew.objectives.diffusion.block import BlockDiffusionObjective
from dew.objectives.diffusion.few_step import SMOOTH_TIME_SCALE
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.jepa import JepaEncoder, JepaObjective, JepaPredictor, MultiBlockMask
from dew.objectives.lm import LMObjective
from dew.objectives.rl import DPOObjective, GRPOObjective, PPOObjective, ValueHead
from dew.objectives.supervised import CrossEntropy, Supervised
from dew.registry import ModelConfig
from dew.sampling.solvers import Euler


def decoder(**fields):
    return CausalTransformer(vocab_size=260, emb_features=16, num_layers=1, num_heads=2,
                             mlp_features=32, max_seq_len=32, **fields)


def diffusion(preset=None, objective="diffusion", **fields) -> DiffusionRunConfig:
    """The diffusion recipe's config: a captioned image dataset and `objective`
    with `fields`, sampling in two Euler steps."""
    return DiffusionRunConfig(
        model=ModelConfig.from_model(SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2)),
        data=TFDS("dew_images", path=str(IMAGES), split="all").pipe(presets.images, 4, batch=2,
                                                                      caption=image_steps.caption()),
        preset=Flow() if preset is None else preset, val_metrics=(),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig(objective, {"solver": Euler(), "steps": 2, **fields}))


def with_teacher(kind):
    """A distilled diffusion kind over the recipe's model, inputs and data,
    its teacher the recipe's own objective at initialization."""
    config = diffusion(guidance=None, ema_decay=None)
    base = config.build()
    drawn = base.init(jax.random.key(0))
    teacher = base.model_variables(drawn)
    shared = (base.model, base.process, base.inputs)
    built = {
        "ladd": lambda: AdversarialDistillationObjective(
            *shared, feature_layers=("dit_block_0",), cmap_dim=4, kernel_size=(1, 1),
            variables={TEACHER: teacher}, steps=2),
        # sCM differentiates the student in time, which needs smooth time features.
        "rcm": lambda: ConsistencyDistillationObjective(
            base.model.clone(time_scale=SMOOTH_TIME_SCALE), *shared[1:], variables={TEACHER: teacher},
            steps=2),
        "guidance_distillation": lambda: GuidanceDistillationObjective(
            *shared, variables={TEACHER: drawn}, steps=2),
    }[kind]()
    return built, first(config.data, built)


def preference_pairs():
    """The chain recipe's DPO stage: `Preferences` at `seq_len`, DPO one below."""
    pair = {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 5],
            "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1]}
    pairs = Memory([json.loads(json.dumps(pair))] * 4).pipe(presets.preferences, seq_len=6, batch=2)
    return DPOObjective(decoder(), 6 - 1), first(pairs)


def jepa():
    """The JEPA recipe's sample field over its image dataset, the grid it patches."""
    config = JepaRunConfig(data=TFDS("dew_images", path=str(IMAGES), split="all").pipe(
        presets.images, 16, batch=2), probe_classes=2)
    sample = config.sample()
    encoder = JepaEncoder(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=2)
    grid = (sample.shape[-2] // encoder.patch_size,) * 2
    predictor = JepaPredictor(grid=grid, emb_features=16, predictor_features=8, num_layers=1,
                              num_heads=2, mlp_ratio=2)
    objective = JepaObjective(encoder, predictor, MultiBlockMask.for_grid(grid, num_targets=2),
                              sample=sample)
    return objective, first(config.data)


def decision():
    """A decision head over a byte decoder, on a two-option question, and the
    first batch its dataset lays out."""
    team = Choice("Which team?", ["billing", "technical"])
    examples = [Example(text, {"team": team}, {"team": label})
                for text, label in (("charged twice", 0), ("site down", 1))] * 2
    objective = DecisionObjective(decoder(), tokenizer=ByteTokenizer(),
                                  specials=Specials(None, 10, 0, "\x00", 255),
                                  layout=StateFirstLayout(max_len=32, head_max_len=24, option_tokens=6))
    return objective, next(decision_rows(examples, 2, objective=objective.data()).train(DataPartition()))


def cases(windows):
    """Per registered name, its objective and the first batch of the data its
    recipe trains on."""
    def built(config):
        objective = config.build()
        return objective, first(config.data, objective)

    def masked():
        mask = 259
        return (MaskedDiffusionObjective(decoder(causal=False), MDLM(mask_id=mask)(), SEQ_LEN + 1),
                first(windows))

    def block():
        canvas, prompt = 4, 8
        model = DiffusionGemma(text=decoder(layer_scalar="frozen"), canvas_length=canvas)
        response = SEQ_LEN + 1 - prompt
        return (BlockDiffusionObjective(model, prompt_length=prompt, num_canvases=response // canvas,
                                        canvas_size=canvas), first(windows))

    return {
        "lm": lambda: (LMObjective(decoder(), SEQ_LEN), first(windows)),
        "masked_diffusion": masked,
        "block_diffusion": block,
        "distillation": lambda: (DistillationObjective(LMObjective(decoder(), SEQ_LEN),
                                                       LMObjective(decoder(), SEQ_LEN)),
                                 first(windows)),
        "dpo": preference_pairs,
        "grpo": lambda: packed(lambda seq_len: GRPOObjective(decoder(), seq_len)),
        "ppo": lambda: packed(lambda seq_len: PPOObjective(decoder(), seq_len,
                                                           critic=ValueHead(decoder()))),
        "diffusion": lambda: built(diffusion(guidance=None, ema_decay=None)),
        "mean_flow": lambda: built(diffusion(MeanFlow(), "mean_flow", ema_decay=None)),
        "shortcut": lambda: built(diffusion(Shortcut(), "shortcut", sections=4, bootstrap_every=2,
                                            ema_decay=None)),
        "flow_grpo": lambda: built(diffusion(objective="flow_grpo", guidance=None)),
        "ladd": lambda: with_teacher("ladd"),
        "rcm": lambda: with_teacher("rcm"),
        "guidance_distillation": lambda: with_teacher("guidance_distillation"),
        "jepa": jepa,
        "decision": decision,
        "supervised": lambda: (Supervised(decoder(), CrossEntropy(labels="text"),
                                          inputs=InputSpec(Field("text", (SEQ_LEN + 1,)))),
                               first(windows)),
    }
