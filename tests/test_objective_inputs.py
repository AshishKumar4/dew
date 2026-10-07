"""Every registered objective declares the batch its own data path writes.

`Objective.inputs` names the sample field the loss reads, its per-example
shape and the condition fields, and the trainer checks the first batch
against it before compiling. Each case builds one objective the way its
recipe or example does and draws one batch from the data source that recipe
trains on, never from the declaration, so a declaration that drifts from its
data fails here rather than at a user's first step.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from dew.config import ModelConfig
from dew.data import DataPartition, PreferencePairs, TFDSImages, TokenCorpus, TokenWindows
from dew.data.text import ByteTokenizer
from dew.decision import Choice, DecisionObjective, Example, Specials, StateFirstLayout
from dew.diffusion.discrete import MDLM
from dew.diffusion.presets import Flow, MeanFlow, Shortcut
from dew.inputs import Field, InputSpec
from dew.nn.backbones import CausalTransformer, SimpleDiT
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.objectives import DistillationObjective
from dew.objectives.diffusion import (
    AdversarialDistillation,
    AdversarialDistillationObjective,
    ConsistencyDistillation,
    ConsistencyDistillationObjective,
    GuidanceDistillationObjective,
    MeanFlowTraining,
    ShortcutTraining,
)
from dew.objectives.diffusion.block import BlockDiffusionObjective
from dew.objectives.diffusion.config import DiffusionRunConfig, FlowGRPO, TextCondition
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.jepa import JepaEncoder, JepaObjective, JepaPredictor, MultiBlockMask
from dew.objectives.lm import LMObjective
from dew.objectives.rl import DPOObjective, GRPOObjective, PPOObjective, ValueHead
from dew.objectives.rl.sessions import Call, Session, Status, pack
from dew.objectives.supervised import CrossEntropy, Supervised
from dew.registry import objectives
from dew.sampling.solvers import Euler
from recipes.jepa.train import JepaRunConfig, sample_field

IMAGES = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"
SEQ_LEN = 15


def first(spec, objective=None):
    """The first training batch `spec` loads, as its recipe loads it."""
    tokenize = None if objective is None else objective.inputs.tokenize
    return next(spec.load(batch=2, tokenize=tokenize).train(DataPartition()))


def decoder(**fields):
    return CausalTransformer(vocab_size=260, emb_features=16, num_layers=1, num_heads=2,
                             mlp_features=32, max_seq_len=32, **fields)


def corpus_windows(corpus: Path) -> TokenWindows:
    """TokenWindows over a byte corpus written to `corpus`, what the LM recipe trains on."""
    TokenCorpus.write(["the quick brown fox jumps over the lazy dog " * 8] * 4, corpus, val_fraction=0.25)
    return TokenWindows(path=str(corpus), seq_len=SEQ_LEN, val_batches=None)


@pytest.fixture(scope="module")
def windows(tmp_path_factory):
    return corpus_windows(tmp_path_factory.mktemp("corpus"))


def diffusion(preset=None, **training) -> DiffusionRunConfig:
    """The diffusion recipe's config: a captioned image dataset, one kind of training."""
    return DiffusionRunConfig(
        model=ModelConfig.from_model(SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2)),
        data=TFDSImages(path=str(IMAGES), image_size=4, augmentation="none", val_batches=None),
        preset=Flow() if preset is None else preset, solver=Euler(), guidance=None, sampling_steps=2,
        ema_decay=None, val_metrics=(), text=TextCondition(encoder="char_table", checkpoint="char_table"),
        **training)


def with_teacher(kind):
    """A distilled diffusion kind over the recipe's model, inputs and data,
    its teacher the recipe's own objective at initialization."""
    config = diffusion()
    base = config.build()
    drawn = base.init(jax.random.key(0))
    teacher = base.model_variables(drawn)
    shared = (base.model, base.process, base.inputs)
    built = {
        "ladd": lambda: AdversarialDistillationObjective(
            *shared, AdversarialDistillation(feature_layers=("dit_block_0",), cmap_dim=4, kernel_size=(1, 1)),
            teacher=base.model, teacher_variables=teacher, guidance=None, steps=2),
        "rcm": lambda: ConsistencyDistillationObjective(
            *shared, ConsistencyDistillation(), teacher=base.model, teacher_variables=teacher, guidance=None,
            steps=2),
        "guidance_distillation": lambda: GuidanceDistillationObjective(
            *shared, teacher=base, teacher_variables=drawn, steps=2),
    }[kind]()
    return built, first(config.data, built)


def packed(objective_for):
    """Rows `sessions.pack` writes, what every GRPO rollout and the RLVR
    example's scheduler train on, at the width the objective names."""
    width = SEQ_LEN + 1
    sessions = [Session("t", str(sample // 2), sample, 0,
                        (Call((1, 2, 3), (4, 5), (-0.1, -0.2), "stop", 0),),
                        Status.COMPLETED, float(sample % 2)) for sample in range(4)]
    return objective_for(width - 1), pack(sessions, width)


def preference_pairs():
    """The chain recipe's DPO stage: `PreferencePairs` at `seq_len`, DPO one below."""
    pair = {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 5],
            "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1]}
    spec = PreferencePairs(records=(json.dumps(pair),) * 4, seq_len=6, val_batches=None)
    return DPOObjective(decoder(), spec.seq_len - 1), first(spec)


def jepa():
    """The JEPA recipe's sample field over its image dataset, the grid it patches."""
    config = JepaRunConfig(data=TFDSImages(path=str(IMAGES), image_size=16, augmentation="none",
                                           val_batches=None), probe_classes=2)
    sample = sample_field(config)
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
    return objective, next(objective.dataset(examples, batch=2).train(DataPartition()))


def cases(windows):
    """Per registered name, its objective and the first batch of the data its
    recipe trains on."""
    def built(config):
        objective = config.build()
        return objective, first(config.data, objective)

    def masked():
        mask = 259
        return (MaskedDiffusionObjective(decoder(causal=False), MDLM(mask_id=mask)(), windows.seq_len + 1),
                first(windows))

    def block():
        canvas, prompt = 4, 8
        model = DiffusionGemma(text=decoder(layer_scalar="frozen"), canvas_length=canvas)
        response = windows.seq_len + 1 - prompt
        return (BlockDiffusionObjective(model, prompt_length=prompt, num_canvases=response // canvas,
                                        canvas_size=canvas), first(windows))

    return {
        "lm": lambda: (LMObjective(decoder(), windows.seq_len), first(windows)),
        "masked_diffusion": masked,
        "block_diffusion": block,
        "distillation": lambda: (DistillationObjective(LMObjective(decoder(), windows.seq_len),
                                                       LMObjective(decoder(), windows.seq_len)),
                                 first(windows)),
        "dpo": preference_pairs,
        "grpo": lambda: packed(lambda seq_len: GRPOObjective(decoder(), seq_len)),
        "ppo": lambda: packed(lambda seq_len: PPOObjective(decoder(), seq_len,
                                                           critic=ValueHead(decoder()))),
        "diffusion": lambda: built(diffusion()),
        "mean_flow": lambda: built(diffusion(MeanFlow(), mode=MeanFlowTraining())),
        "shortcut": lambda: built(diffusion(Shortcut(), mode=ShortcutTraining(sections=4,
                                                                                  bootstrap_every=2))),
        "flow_grpo": lambda: built(diffusion(mode=FlowGRPO())),
        "ladd": lambda: with_teacher("ladd"),
        "rcm": lambda: with_teacher("rcm"),
        "guidance_distillation": lambda: with_teacher("guidance_distillation"),
        "jepa": jepa,
        "decision": decision,
        "supervised": lambda: (Supervised(decoder(), CrossEntropy(labels="text"),
                                          inputs=InputSpec(Field("text", (windows.seq_len + 1,)))),
                               first(windows)),
    }


def test_every_registered_objective_has_a_recipe_case(windows):
    assert set(cases(windows)) == set(objectives), "an aliased objective needs a case here"


@pytest.mark.parametrize("name", sorted(objectives))
def test_the_declared_inputs_match_the_recipe_batch(name, windows):
    objective, batch = cases(windows)[name]()
    assert type(objective) is objectives[name]
    objective.inputs.check(batch)


def test_a_drifted_declaration_is_refused_before_compiling(windows):
    """What the cases guard: GRPO's packed `input_ids` are not the LM's
    `text` windows, so an LM declaration over them is refused."""
    _, batch = packed(lambda seq_len: GRPOObjective(decoder(), seq_len))
    with pytest.raises(ValueError, match="needs field 'text'"):
        LMObjective(decoder(), SEQ_LEN).inputs.check(batch)
    with pytest.raises(ValueError, match="declares shape"):
        GRPOObjective(decoder(), SEQ_LEN + 1).inputs.check(batch)
    assert np.shape(batch["input_ids"])[1] == SEQ_LEN + 1
