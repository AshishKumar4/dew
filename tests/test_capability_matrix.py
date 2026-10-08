"""Every model family against every method its capabilities and its math admit.

A consumer reads a model through what it can do (`dew.nn.protocols`), never
through which class it is, so a method that trains or samples one family
trains or samples every family with the same capabilities. This file proves
that over tiny fixtures on the CPU: every family Dew registers
(`dew.registry.models`), at the variants whose mixers, heads and media
differ, and plain `flax.linen.Module`s a user writes
(tests/capability_modules.py), which nothing registers.

A family declares facts: the capabilities it has and the math it computes
(whether its states read only the past, whether its denoiser takes an
interval). A method declares the facts it reads and the facts it needs. Both
are written down here, before anything runs, so the cells are fixed by the
declarations, never by trying an operation and skipping what raises:

- a supported cell, a family with every fact its method needs, trains: the
  first batch is checked, the trainer's step logs a finite loss and moves
  the trained leaves, the run checkpoints, evaluation runs, and the saved
  run reloads to the same numbers;
- a refused cell, a family the method reads but whose math it cannot run,
  fails with an error that names the missing fact.

A family Dew registers without a fixture here fails
`test_every_registered_family_has_a_fixture`, and an objective without a row
fails `test_every_registered_objective_has_a_row`, so a new family or
method extends this matrix before it lands.
"""

from __future__ import annotations

import atexit
import dataclasses
import functools
import importlib
import pkgutil
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from capability_modules import Denoiser, FamilyText, NNXDenoiser, NNXLanguageModel, NNXTokens, TokenModel
from flax.nnx import bridge
from recording import RecordingTracker
from test_precision_policy import PER_ARCH, TINY

import dew
from dew.checkpoints import Checkpoints
from dew.data import DataPartition
from dew.data.preferences import IDS_KEY as PAIR_IDS_KEY, MASK_KEY as PAIR_MASK_KEY
from dew.data.prompts import LENGTH_KEY, PROMPT_KEY
from dew.data.text import ByteTokenizer
from dew.decision import Choice, DecisionObjective, Example, MarkerLayout, Specials, StateFirstLayout
from dew.diffusion import presets
from dew.diffusion.discrete import MDLM
from dew.inference.tasks import TextGeneration
from dew.inputs import Condition, Field, InputSpec
from dew.inputs.encoders import CharTable
from dew.interop import Pretrained
from dew.lora import Adapter, LoRA
from dew.objectives.base import TEACHER, Step
from dew.objectives.diffusion import (
    AdversarialDistillationObjective,
    ConsistencyDistillationObjective,
    DiffusionObjective,
    GuidanceDistillationObjective,
    MeanFlowObjective,
    ShortcutObjective,
)
from dew.objectives.diffusion.block import BlockDiffusionObjective
from dew.objectives.diffusion.few_step import SMOOTH_TIME_SCALE
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.distillation import DistillationObjective
from dew.objectives.jepa import JepaObjective, MultiBlockMask
from dew.objectives.lm import LMObjective
from dew.objectives.rl import PPOObjective, ValueHead
from dew.objectives.rl.flow import FlowGRPOObjective, FlowRollout
from dew.objectives.rl.grpo import GRPOObjective
from dew.objectives.rl.ppo import OLD_VALUES_KEY, RETURNS_KEY
from dew.objectives.rl.preference import DPOObjective
from dew.objectives.rl.sessions import (
    ADVANTAGES_KEY,
    BEHAVIOR_LOG_PROBS_KEY,
    IDS_KEY,
    OLD_LOG_PROBS_KEY,
    POSITIONS_KEY,
    RESPONSE_MASK_KEY,
    SEGMENT_IDS_KEY,
)
from dew.objectives.supervised import Accuracy, CrossEntropy, Supervised
from dew.registry import models, objectives
from dew.sampling import Euler, Sampling
from dew.training import Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

ROWS = 8
"""One row per simulated device (tests/lane_environment.py)."""
SEQ = 8
TOKENS = (3, 31)
"""The ids a token batch draws from: inside every fixture's vocabulary (the
smallest holds 32) and clear of the ids they reserve for images, padding
and masks."""
STEP = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(7), ema=None)

# --- facts -----------------------------------------------------------------
# What a family can do and what its math is. A method reads some of them and
# needs others; the cells follow from the two declarations alone.

LOGITS = "logits"
"""Full-sequence vocabulary logits (`dew.nn.protocols.Logits`)."""
STATES = "hidden_states"
"""Full-sequence final states (`dew.nn.protocols.HiddenStates`)."""
CAUSAL = "causal"
"""Each position reads itself and those before it (`TokenModel.causal`)."""
BIDIRECTIONAL = "bidirectional"
"""Each position reads the whole sequence (`TokenModel.causal` False)."""
MASK_TOKEN = "mask_token"
"""A vocabulary id masked diffusion corrupts tokens to (`TokenModel.mask_token_id`)."""
PACKING = "packing"
"""Reads several documents in one row apart, by their segment ids and positions."""
CACHE = "cache"
"""Decodes token by token over a key/value or recurrent cache."""
TEXT = "text"
"""Its vocabulary holds a tokenizer a run records: bytes, or the source's own."""

DENOISES = "denoises"
"""Maps a noisy sample and its time to the prediction a process steps (`DenoisingModel`)."""
INTERVAL = "interval"
"""Reads an interval's duration beside its end time, set (`IntervalModel`)."""
GUIDED = "guided"
"""Reads the guidance scale a guidance-distilled student is told (`DenoisingCondition.guidance`)."""
FEATURES = "features"
"""Names blocks whose outputs are square token grids a discriminator head reads (LADD)."""
ENCODES = "encodes"
"""Encodes a sample's patches into token states a JEPA predictor reads (`HiddenStates`-like)."""
CANVAS = "canvas"
"""Refines a canvas of tokens against a cache of committed clean ones (`BlockDenoiser`)."""

REASONS: Mapping[str, str] = {
    LOGITS: r"logits|sample alone",
    CAUSAL: r"causal",
    BIDIRECTIONAL: r"bidirectional|reads the whole|future",
    PACKING: r"pack|segment",
    CACHE: r"cache",
    INTERVAL: r"interval",
    GUIDED: r"guidance",
    FEATURES: r"feature",
    CANVAS: r"canvas|BlockDenoiser|encode",
}
"""What the error of a refused cell names, by the fact the family lacks."""


# --- families --------------------------------------------------------------

@dataclass(frozen=True)
class Loaded:
    """A model as a user hands it over: bare, a loaded source, or an adapter.

    `given` is that object; `model` and `variables` are what it holds, None
    for a bare model whose tree a method draws fresh, and `processor` the
    source's own text processor, if it has one.
    """

    given: object
    model: object
    variables: Mapping | None = None
    processor: object = None


@dataclass(frozen=True)
class Family:
    """One fixture: a registered family at one variant, or a user's module."""

    name: str
    registered: str | None
    """The `dew.registry.models` name this is a variant of; None for a model nothing registers."""
    facts: frozenset[str]
    vocab: int
    load: Callable[[], Loaded] = field(compare=False)
    inputs: Callable[[], InputSpec] | None = field(default=None, compare=False)
    """A denoiser's sample field and conditions; None for a token model."""

    def __str__(self) -> str:
        return self.name


def built(name: str, **config) -> Callable[[], Loaded]:
    """A registered family at a tiny hand-written config, with a fresh tree."""
    def load() -> Loaded:
        model = models.build(name, **config)
        return Loaded(model, model)
    return load


def released(directory: str) -> Callable[[], Loaded]:
    """A committed tiny checkpoint in a released family's own format, loaded offline."""
    def load() -> Loaded:
        source = Pretrained.load(FIXTURES / directory, dtype="float32", attention_impl="reference")
        return Loaded(source, source.model, source.variables, source.text_processor)
    return load


def adapted(load: Callable[[], Loaded], *modules: str) -> Callable[[], Loaded]:
    """`load`'s model under a rank-2 LoRA on `modules`, its base frozen."""
    def adapt() -> Loaded:
        held = load()
        variables = held.variables or held.model.init(jax.random.key(0), jnp.zeros((1, SEQ), jnp.int32))
        adapter = LoRA(rank=2, modules=modules).apply(held.model, variables, key=1)
        return Loaded(adapter, adapter.model, adapter.variables)
    return adapt


def user(module, **fields) -> Callable[[], Loaded]:
    def load() -> Loaded:
        model = module(**fields)
        return Loaded(model, model)
    return load


def torch_fallback() -> Loaded:
    """A random tiny GPT-NeoX, which no Dew family reads, through the torchax fallback."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchax")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPTNeoXConfig(num_hidden_layers=1, hidden_size=16, num_attention_heads=2,
                                        intermediate_size=32, vocab_size=256, max_position_embeddings=64)
    directory = tempfile.mkdtemp(prefix="dew-matrix-neox-")
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    transformers.GPTNeoXForCausalLM(config).save_pretrained(directory)
    with pytest.warns(UserWarning, match="tier 3"):
        source = Pretrained.load(directory, fallback="torchax", dtype="float32")
    return Loaded(source, source.model, source.variables)


DECODER = {"vocab_size": 256, "emb_features": 16, "num_layers": 2, "num_heads": 2, "num_kv_heads": 1,
           "mlp_features": 32, "max_seq_len": 64, "attention_impl": "reference"}
"""A two-layer causal transformer 16 wide over the byte vocabulary."""

CAUSAL_LM = frozenset({LOGITS, STATES, CAUSAL})
DECODER_LM = CAUSAL_LM | {PACKING, CACHE}
BIDIRECTIONAL_LM = frozenset({LOGITS, STATES, BIDIRECTIONAL})

PIXELS, FRAMES = 16, 2


def described(shape: tuple[int, ...], keyword: str = "textcontext", **published) -> Callable[[], InputSpec]:
    """A denoiser's sample, `[*shape]` pixels or latents (a video's leading
    axis is its frames), with a prompt's states under `keyword`: a character
    table's (`TextContext`), or `FamilyText`'s `DenoisingCondition` at a
    published family's widths."""
    def inputs() -> InputSpec:
        encoder = FamilyText.from_pretrained(**published) if published else CharTable.from_pretrained()
        return InputSpec(Field("video" if len(shape) == 4 else "image", shape), {keyword: Condition(encoder)})
    return inputs


IMAGE = described((PIXELS, PIXELS, 3))
VIDEO = described((FRAMES, PIXELS, PIXELS, 3))
PIXEL_DENOISER = frozenset({DENOISES})
DENOISER_CONFIGS: Mapping[str, tuple[Mapping, Callable[[], InputSpec], frozenset[str]]] = {
    "simple_dit": ({**TINY, **PER_ARCH["simple_dit"]}, IMAGE, PIXEL_DENOISER | {INTERVAL, FEATURES}),
    "hybrid_dit": ({**TINY, **PER_ARCH["hybrid_dit"]}, IMAGE, PIXEL_DENOISER | {INTERVAL, FEATURES}),
    "simple_mmdit": ({**TINY, **PER_ARCH["simple_mmdit"]}, IMAGE, PIXEL_DENOISER),
    "hierarchical_mmdit": (PER_ARCH["hierarchical_mmdit"], IMAGE, PIXEL_DENOISER),
    "simple_udit": ({**TINY, **PER_ARCH["simple_udit"]}, IMAGE, PIXEL_DENOISER),
    "uvit": ({**TINY, **PER_ARCH["uvit"], "image_size": PIXELS}, IMAGE, PIXEL_DENOISER),
    "unet": ({**TINY, **PER_ARCH["unet"]}, IMAGE, PIXEL_DENOISER),
    "edm2_unet": (PER_ARCH["edm2_unet"], IMAGE, PIXEL_DENOISER),
    "video_dit": ({**TINY, **PER_ARCH["video_dit"]}, VIDEO, PIXEL_DENOISER),
    "unet_3d": ({**TINY, **PER_ARCH["unet_3d"]}, VIDEO, PIXEL_DENOISER),
    "unet_2d_condition": (PER_ARCH["unet_2d_condition"], described((PIXELS, PIXELS, 4), "conditioning",
                                                                    width=16), PIXEL_DENOISER),
    "sd3_transformer": (PER_ARCH["sd3_transformer"],
                        described((8, 8, 4), "conditioning", width=12, pooled=10), PIXEL_DENOISER),
    "flux_transformer": (PER_ARCH["flux_transformer"],
                         described((8, 8, 4), "conditioning", width=16, pooled=10, guidance=3.5),
                         PIXEL_DENOISER | {GUIDED}),
    "flux2_transformer": (PER_ARCH["flux2_transformer"],
                          described((4, 4, 16), "conditioning", width=16, guidance=3.5), PIXEL_DENOISER),
    "qwen_image_transformer": (PER_ARCH["qwen_image_transformer"],
                               described((4, 4, 4), "conditioning", width=16, masked=True), PIXEL_DENOISER),
    "z_image_transformer": (PER_ARCH["z_image_transformer"],
                            described((4, 4, 4), "conditioning", width=16, masked=True), PIXEL_DENOISER),
    "wan_transformer": (PER_ARCH["wan_transformer"], described((FRAMES, 8, 8, 4), "conditioning", width=16),
                        PIXEL_DENOISER),
}
"""Every registered denoiser at a tiny config (tests/test_precision_policy.py's),
with the sample and the conditioning it reads."""

JEPA_GRID = (PIXELS // 4, PIXELS // 4)
JEPA_ENCODER = {"patch_size": 4, "emb_features": 32, "num_layers": 1, "num_heads": 2, "mlp_ratio": 2}
JEPA_PREDICTOR = {"grid": JEPA_GRID, "emb_features": 32, "predictor_features": 16, "num_layers": 1,
                  "num_heads": 2, "mlp_ratio": 2}

FAMILIES: Mapping[str, tuple[Family, ...]] = {
    "causal_transformer": (
        Family("causal_transformer", "causal_transformer", DECODER_LM | {TEXT}, 256,
               built("causal_transformer", **DECODER)),
        # The head as its own biased matrix, then a nonlinear head under a
        # softcap: the first is a table with a bias, the second no table.
        Family("causal_transformer+untied_biased_head", "causal_transformer", DECODER_LM, 256,
               built("causal_transformer", **DECODER, tie_embeddings=False, head_bias=True)),
        Family("causal_transformer+nonlinear_head", "causal_transformer", DECODER_LM, 256,
               built("causal_transformer", **DECODER, head_transform="gelu", final_logit_softcap=30.0)),
        Family("causal_transformer+lora_with_head", "causal_transformer", DECODER_LM | {TEXT}, 256,
               adapted(built("causal_transformer", **DECODER, tie_embeddings=False),
                       "q_proj", "v_proj", "lm_head")),
        Family("causal_transformer+mla_moe", "causal_transformer", DECODER_LM, 256,
               released("deepseek-v3-tiny")),
        Family("causal_transformer+gated_delta_hybrid", "causal_transformer", DECODER_LM, 256,
               released("qwen35-tiny")),
        Family("causal_transformer+mamba2", "causal_transformer", DECODER_LM, 32, released("mamba2-tiny")),
        # Kimi K2.5's language model; Dew reads no Kimi vision tower.
        Family("causal_transformer+kimi_k25", "causal_transformer", DECODER_LM, 256,
               released("kimi-k25-tiny")),
        Family("causal_transformer+bidirectional", "causal_transformer", BIDIRECTIONAL_LM | {TEXT}, 256,
               built("causal_transformer", **DECODER, causal=False)),
        Family("causal_transformer+modernbert", "causal_transformer", BIDIRECTIONAL_LM, 64,
               released("modernbert-tiny")),
        Family("causal_transformer+llada", "causal_transformer", BIDIRECTIONAL_LM | {MASK_TOKEN}, 128,
               released("llada-tiny")),
        Family("causal_transformer+dream", "causal_transformer", BIDIRECTIONAL_LM | {MASK_TOKEN}, 128,
               released("dream-tiny")),
    ),
    "multimodal_transformer": (
        Family("multimodal_transformer+gemma3", "multimodal_transformer", DECODER_LM, 256,
               released("gemma3-tiny-mm")),
        Family("multimodal_transformer+qwen35", "multimodal_transformer", DECODER_LM | {TEXT}, 256,
               released("qwen35-tiny-mm")),
        Family("multimodal_transformer+qwen38", "multimodal_transformer", DECODER_LM | {TEXT}, 289,
               released("qwen38-dense-tiny")),
        Family("multimodal_transformer+gemma3n", "multimodal_transformer", DECODER_LM, 64,
               released("gemma3n-vision-tiny")),
        Family("multimodal_transformer+gemma4", "multimodal_transformer", DECODER_LM, 64,
               released("gemma4-tiny-mm")),
        Family("multimodal_transformer+llama4", "multimodal_transformer", DECODER_LM, 96,
               released("llama4-tiny-mm")),
    ),
    "diffusion_gemma": (
        # The clean encoder reads the committed prefix causally and packs as
        # its text model does; its cache commits a canvas at a time, so it
        # decodes no token after token.
        Family("diffusion_gemma", "diffusion_gemma", CAUSAL_LM | {PACKING, TEXT, CANVAS}, 64,
               released("diffusion-gemma-workflow")),
    ),
    **{name: (Family(name, name, facts, 0, built(name, **config), inputs),)
       for name, (config, inputs, facts) in DENOISER_CONFIGS.items()},
    # An encoder pairs with a predictor over its own patch grid, a video
    # encoder's tubelets with the factorized one.
    "jepa_encoder": (Family("jepa_encoder", "jepa_encoder", frozenset({ENCODES}), 0,
                            built("jepa_encoder", **JEPA_ENCODER), described((PIXELS, PIXELS, 3))),),
    "jepa_video_encoder": (Family("jepa_video_encoder", "jepa_video_encoder", frozenset({ENCODES}), 0,
                                  built("jepa_video_encoder", **JEPA_ENCODER),
                                  described((FRAMES, PIXELS, PIXELS, 3))),),
    "jepa_predictor": (Family("jepa_predictor", "jepa_predictor", frozenset({ENCODES}), 0,
                              built("jepa_predictor", **JEPA_PREDICTOR), described((PIXELS, PIXELS, 3))),),
}
"""Every registered family's fixtures, keyed by its registry name."""

UNREGISTERED: tuple[Family, ...] = (
    Family("user+causal", None, CAUSAL_LM | {TEXT}, 256, user(TokenModel, vocab_size=256)),
    Family("user+bidirectional", None, BIDIRECTIONAL_LM | {MASK_TOKEN, TEXT}, 256,
           user(TokenModel, vocab_size=256, causal=False, mask_token_id=255)),
    # Ordinary, unpacked language modelling only: no packing, no cache.
    Family("torch+gpt_neox", None, CAUSAL_LM, 256, torch_fallback),
    # Flax NNX through Flax's own bridge, which runs a module's `__call__`
    # and the methods a Linen wrapper names (capability_modules.NNXTokens).
    Family("nnx+causal", None, CAUSAL_LM, 256, user(NNXTokens, nnx_class=NNXLanguageModel, args=(256,))),
    Family("user+denoiser", None, PIXEL_DENOISER | {INTERVAL}, 0, user(Denoiser), IMAGE),
    Family("nnx+denoiser", None, PIXEL_DENOISER, 0, user(bridge.ToLinen, nnx_class=NNXDenoiser, args=(3,)),
           IMAGE),
)

ALL: tuple[Family, ...] = (*(family for variants in FAMILIES.values() for family in variants), *UNREGISTERED)
BY_NAME: Mapping[str, Family] = {family.name: family for family in ALL}
KINDS: frozenset[str] = frozenset({variants[0].name for variants in FAMILIES.values()}
                                  | {family.name for family in UNREGISTERED})
"""One fixture of each registered family and every model nothing registers.
A row whose loss reads a family only through the same token scoring as
`lm` covers these; `lm` and `generation` cover every mixer and head."""


@functools.cache
def loaded(family: Family) -> Loaded:
    """Each fixture once per process, its tree on the host: the trainer
    donates the device buffers it is given, and a host tree hands it copies."""
    held = family.load()
    if held.variables is None:
        return held
    variables = jax.device_get(held.variables)
    given = held.given
    if isinstance(given, Adapter):
        given = dataclasses.replace(given, variables=variables)
    return dataclasses.replace(held, given=given, variables=variables)


def registered_families() -> dict[str, type]:
    """Every model family Dew's own modules register, by name.

    A table fills as its members' modules import, so every module of Dew is
    imported first; a family a plugin registers is the plugin's to cover.
    """
    for module in pkgutil.walk_packages(dew.__path__, "dew."):
        importlib.import_module(module.name)
    return {name: member for name, member in models.items() if member.__module__.startswith("dew.")}


def test_every_registered_family_has_a_fixture():
    """A family registered without one has nothing proving its methods work on it."""
    families = registered_families()
    assert sorted(set(families) - set(FAMILIES)) == []
    assert sorted(set(FAMILIES) - set(families)) == []


@pytest.mark.parametrize("family", [family for family in ALL if family.registered is not None], ids=str)
def test_a_fixture_is_the_family_it_is_listed_under(family):
    """The coverage above counts a family only by fixtures that build it: the
    nearest registered class its model is, an adapter's being a subclass of
    the family it adapts and a family's its own over a family it extends."""
    nearest = next(name for kind in type(loaded(family).model).__mro__
                   for name, member in models.items() if member is kind)
    assert nearest == family.registered


# --- running a method ------------------------------------------------------

class Batches:
    """The `Dataset` a trainer reads: `batch`, over and over."""

    def __init__(self, batch: Mapping):
        self._batch = batch
        self.batch = ROWS
        self.records = None
        self.val = None
        self.steps_per_epoch = None

    def train(self, partition) -> Iterator[Mapping]:
        while True:
            yield self._batch


def finite(tree) -> bool:
    return all(bool(np.all(np.isfinite(leaf))) for leaf in jax.tree.leaves(tree)
               if np.issubdtype(np.asarray(leaf).dtype, np.inexact))


@dataclass(frozen=True)
class Trained:
    """A supported cell's run: its directory, its state, and its trained tree on the host."""

    directory: Path
    state: object
    variables: Mapping

    @property
    def step(self) -> Step:
        """The step an evaluation of the run reads: its key, and its average when it keeps one."""
        averaged = None if self.state.ema is None else jax.device_get(self.state.averaged)
        return Step(step=jnp.asarray(1), key=jax.random.key(3), ema=averaged)


def assert_trains(objective, batch: Mapping, directory: Path, *, evaluated: Mapping | None = None,
                  data=None, sampled: bool = True, rollout=None, scores: bool = True) -> Trained:
    """The proof every supported cell gives, on `batch`.

    The first batch passes the objective's input check and a cut one does
    not. The trainer builds the tree, and its step on the batch logs a finite
    loss and leaves every trained leaf finite, some of them moved: Adam's
    first update moves a leaf exactly where its gradient is nonzero, and a
    gradient that is not finite would leave the leaf so. The run checkpoints
    its step, and evaluation scores `evaluated` (the batch by default) to
    finite artifacts, or, for an objective that scores no tokens (`scores`
    False), to none. `data` replaces the batch stream with the method's own;
    a method that trains on what its `rollout` samples from the batch, not
    on the batch (`sampled` False), has no sample field to check.
    """
    if objective.inputs is not None and sampled:
        objective.inputs.check(batch)
        cut = {key: value[:, :-1] if key == objective.inputs.sample.key else value
               for key, value in batch.items()}
        with pytest.raises(ValueError, match="shape"):
            objective.inputs.check(cut)
    tracker = RecordingTracker()
    checkpoints = Checkpoints(str(directory))
    trainer = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=checkpoints, tracker=tracker,
                      rollout=rollout)
    initial = trainer.initial_state()
    started = jax.device_get(initial.variables["params"])
    state = trainer.fit(Batches(batch) if data is None else data, steps=1, log_every=1, state=initial)
    checkpoints.wait()
    losses = [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]
    assert len(losses) == 1 and np.isfinite(losses[0]), losses
    variables = jax.device_get(state.variables)
    assert finite(variables), "a trained leaf is not finite"
    moved = [not np.array_equal(before, after) for before, after in
             zip(jax.tree.leaves(started), jax.tree.leaves(variables["params"]), strict=True)]
    assert any(moved), "the step moved no trained leaf: no gradient reached them"
    assert Checkpoints(str(directory)).latest == 1
    trained = Trained(directory, state, variables)
    artifact = objective.evaluate(variables, batch if evaluated is None else evaluated, trained.step)
    # An objective that scores no tokens evaluates to nothing (`Objective.evaluate`).
    assert (artifact is not None and finite(artifact)) if scores else artifact is None, artifact
    return trained


def token_batch(width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(*TOKENS, size=(ROWS, width)).astype(np.int32)


@functools.cache
def _loss_program(objective):
    return jax.jit(lambda tree, batch: objective.scalar_loss(tree, batch, STEP)[0])


def scored(objective, variables, batch) -> np.ndarray:
    """The objective's loss over `batch`, compiled once per objective and run
    on host copies, so two trees of one shape run one program."""
    return np.asarray(_loss_program(objective)(jax.device_get(variables), batch))


def assert_reloads(objective, trained: Trained, family: Family, rebuilt, batch) -> None:
    """The saved run scores the batch to the same bits as the state that wrote it.

    A registered family's run rebuilds its model from the run's record
    (`saved_task.from_run`), and `rebuilt` makes the objective over the
    task. A model nothing registers keeps its live module and restores the
    checkpoint's tree: registration is what rebuilding a model from a run
    needs, and nothing else does.
    """
    expected = scored(objective, trained.variables, batch)
    if family.registered is None:
        restored = Checkpoints(str(trained.directory)).variables()
        np.testing.assert_array_equal(scored(objective, restored, batch), expected)
        return
    again = rebuilt(objective.saved_task.from_run(str(trained.directory)))
    np.testing.assert_array_equal(scored(again, again.initializer(jax.random.key(0)), batch), expected)


# --- token methods -----------------------------------------------------------

def lm(family: Family, directory: Path) -> None:
    """Next-token likelihood through a saved run's reload, its head scored whole and in tiles alike.

    A tiled head contracts the output table tile by tile instead of holding
    the logits, so the two agree to the chunked head's 1e-5 relative bound.
    """
    held = loaded(family)
    batch = {"text": token_batch(SEQ + 1)}
    objective = LMObjective(held.model, SEQ, variables=held.variables, head_tile="whole")
    trained = assert_trains(objective, batch, directory)
    tiled = LMObjective(held.model, SEQ, variables=held.variables, head_tile="tiled", head_chunks=2)
    np.testing.assert_allclose(scored(tiled, trained.variables, batch),
                               scored(objective, trained.variables, batch), rtol=1e-5)
    assert_reloads(objective, trained, family,
                   lambda task: LMObjective(task.model, SEQ, variables=task.variables, head_tile="whole"),
                   batch)


def preference_batch() -> dict:
    """Chosen and rejected completions of one prompt per row."""
    ids = np.stack([token_batch(SEQ + 1, seed=1), token_batch(SEQ + 1, seed=2)], axis=1)
    ids[:, 1, :SEQ // 2] = ids[:, 0, :SEQ // 2]
    mask = np.zeros(ids.shape, np.int32)
    mask[:, :, SEQ // 2:] = 1
    return {PAIR_IDS_KEY: ids, PAIR_MASK_KEY: mask}


def dpo(family: Family, directory: Path) -> None:
    """Preference pairs against the frozen starting policy."""
    held = loaded(family)
    objective = DPOObjective(held.model, SEQ, variables=held.variables)
    trained = assert_trains(objective, preference_batch(), directory)
    assert trained.state.ema is not None, "the reference policy is the frozen average"


def rollouts(prompt: int = SEQ // 2) -> dict:
    """One sampled response per row after a prompt, with its advantage and old
    log-likelihoods, packed as Dew's sessions packer writes them: one chain a row."""
    width = SEQ + 1
    rng = np.random.default_rng(4)
    mask = np.zeros((ROWS, width), np.float32)
    mask[:, prompt:] = 1
    likelihoods = np.where(mask != 0, rng.normal(-3.0, 0.5, mask.shape), 0).astype(np.float32)
    return {IDS_KEY: token_batch(width, seed=3), RESPONSE_MASK_KEY: mask,
            OLD_LOG_PROBS_KEY: likelihoods, BEHAVIOR_LOG_PROBS_KEY: likelihoods,
            ADVANTAGES_KEY: np.where(mask != 0, rng.normal(0, 1, mask.shape), 0).astype(np.float32),
            PROMPT_KEY: token_batch(prompt, seed=5), LENGTH_KEY: np.full((ROWS,), prompt, np.int32),
            SEGMENT_IDS_KEY: np.ones((ROWS, width), np.int32),
            POSITIONS_KEY: np.tile(np.arange(width, dtype=np.int32), (ROWS, 1))}


def grpo(family: Family, directory: Path) -> None:
    """The clipped surrogate over rolled-out responses, with the KL to a frozen reference."""
    held = loaded(family)
    objective = GRPOObjective(held.model, SEQ, beta=0.01, variables=held.variables)
    assert_trains(objective, rollouts(), directory)


def ppo(family: Family, directory: Path) -> None:
    """The actor's surrogate and a critic over the same family's states."""
    held = loaded(family)
    objective = PPOObjective(held.model, SEQ, critic=ValueHead(held.model), beta=0.01,
                             variables=held.variables)
    batch = rollouts()
    rng = np.random.default_rng(6)
    batch[OLD_VALUES_KEY] = (batch[RESPONSE_MASK_KEY] * rng.normal(0, 1, (ROWS, SEQ + 1))).astype(np.float32)
    batch[RETURNS_KEY] = (batch[RESPONSE_MASK_KEY] * rng.normal(0, 1, (ROWS, SEQ + 1))).astype(np.float32)
    assert_trains(objective, batch, directory)


def distillation(family: Family, directory: Path) -> None:
    """The student's own loss and a frozen teacher of another family over the
    same vocabulary: each pair `TEACHERS` names."""
    held = loaded(family)
    teacher = loaded(BY_NAME[TEACHERS[family.name]])
    student = LMObjective(held.model, SEQ, variables=held.variables)
    objective = DistillationObjective(
        student, LMObjective(teacher.model, SEQ, variables=teacher.variables), alpha=0.5, temperature=2.0)
    trained = assert_trains(objective, {"text": token_batch(SEQ + 1)}, directory)
    teachers = [leaf for path, leaf in jax.tree_util.tree_leaves_with_path(trained.variables)
                if "teacher" in jax.tree_util.keystr(path)]
    assert teachers, "the teacher's tree is part of the run"


TEACHERS: Mapping[str, str] = {
    "causal_transformer": "user+causal",
    "user+causal": "causal_transformer+mla_moe",
    "causal_transformer+mla_moe": "multimodal_transformer+gemma3",
    "multimodal_transformer+gemma3": "torch+gpt_neox",
    "torch+gpt_neox": "causal_transformer",
    "multimodal_transformer+gemma4": "diffusion_gemma",
    "diffusion_gemma": "multimodal_transformer+gemma4",
}
"""Each distillation cell's student and its teacher: two rings through the
families, one over the byte-sized vocabulary and one over 64 ids, every kind
a student once and a teacher once."""


def decision(family: Family, directory: Path) -> None:
    """Questions with known answers over the family's states, in the layout its
    order reads: the state first for a causal backbone, the markers first for a
    bidirectional one. The saved run answers as a `Decide` task."""
    held = loaded(family)
    tokenizer, specials = vocabulary(held)
    layout = (StateFirstLayout if CAUSAL in family.facts else MarkerLayout)(
        max_len=48, head_max_len=32, option_tokens=4)
    objective = DecisionObjective(held.given, tokenizer=tokenizer, specials=specials, layout=layout)
    data = objective.dataset(EXAMPLES, batch=ROWS, validation=EXAMPLES)
    batch = next(iter(data.train(DataPartition())))
    trained = assert_trains(objective, batch, directory, data=data)
    live = objective.pipeline(trained.state)
    asked = {"team": INTENT}
    if family.registered is None:
        return
    reloaded = objective.saved_task.from_run(str(directory))
    for answer, again in zip(live("charged twice", asked).values(), reloaded("charged twice", asked).values(),
                             strict=True):
        np.testing.assert_array_equal(answer.probabilities, again.probabilities)


INTENT = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
EXAMPLES = [Example(state, {"team": INTENT}, {"team": label})
            for state, label in (("charged twice", "billing"), ("site is down", "technical"),
                                 ("how much is pro", "sales"), ("refund please", "billing"),
                                 ("error 500", "technical"), ("upgrade cost", "sales"),
                                 ("double charge", "billing"), ("cannot log in", "technical"))]


def vocabulary(held: Loaded):
    """The tokenizer a source's own processor encodes with, else bytes, and its special tokens."""
    if held.processor is not None:
        return None, None
    return ByteTokenizer(), Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)


def mdlm(family: Family, directory: Path) -> None:
    """Masked diffusion over the whole sequence, through a saved run's reload.

    The process carries the mask id training corrupts to, so a model that
    names none trains; its saved run, which samples by unmasking that id,
    reloads only from a model that names it, and otherwise says so.
    """
    held = loaded(family)
    batch = {"text": token_batch(SEQ)}
    objective = MaskedDiffusionObjective(held.model, MDLM(mask_id=mask_id(family))(), SEQ,
                                         variables=held.variables, ema_decay=None, steps=2, samples=2)
    trained = assert_trains(objective, batch, directory)
    if MASK_TOKEN in family.facts or family.registered is None:
        assert_reloads(objective, trained, family,
                       lambda task: MaskedDiffusionObjective(task.model, task.process, SEQ,
                                                             variables=task.variables, ema_decay=None),
                       batch)
        return
    with pytest.raises(ValueError, match="mask_token_id"):
        objective.saved_task.from_run(str(directory))


def mask_id(family: Family) -> int:
    """The id a family trained by masked diffusion masks with, and the last
    id every fixture's vocabulary holds for one that was not."""
    model = loaded(family).model
    return model.mask_token_id if MASK_TOKEN in family.facts else TOKENS[1]


def generation(family: Family, directory: Path) -> None:
    """Cached greedy decoding continues a prompt with the tokens the full
    forward pass ranks first at every position it decoded."""
    held = loaded(family)
    variables = held.variables or held.model.init(jax.random.key(0), jnp.zeros((1, SEQ), jnp.int32))
    task = TextGeneration(held.model, variables, sampling=Sampling(temperature=0.0))
    prompt = token_batch(SEQ // 2, seed=8)[:2]
    generated = task(prompt.tolist(), SEQ // 2, key=0)
    tokens = np.asarray(generated.tokens)
    assert tokens.shape == (2, SEQ) and (tokens[:, :SEQ // 2] == prompt).all()
    logits = np.asarray(held.model.apply(variables, jnp.asarray(tokens)))
    np.testing.assert_array_equal(logits[:, SEQ // 2 - 1:-1].argmax(-1), tokens[:, SEQ // 2:])


# --- denoising methods -------------------------------------------------------

CAPTIONS = ["a red square", "two dots", "", "a line", "noise", "x", "blue", "grid"]


def denoised_batch(inputs: InputSpec) -> dict:
    """`ROWS` samples of the sample field, as uint8 pixels or latents, and their captions' tokens."""
    batch = {inputs.sample.key: np.random.default_rng(9).integers(0, 256, (ROWS, *inputs.sample.shape),
                                                                  dtype=np.uint8)}
    for condition in inputs.conditions.values():
        batch[condition.field] = condition.encoder.tokenize(CAPTIONS)
    return batch


def denoising(task, **policy) -> DiffusionObjective:
    """Rectified flow over `task`: a denoiser with its inputs, or a saved run's loaded pipeline."""
    return DiffusionObjective(task, **policy, ema_decay=None, guidance=None, steps=2, solver=Euler())


def diffusion(family: Family, directory: Path) -> None:
    """Rectified flow on the family's own sample and conditioning, through a saved run's reload."""
    held = loaded(family)
    objective = denoising(held.model, process=presets.Flow(), inputs=family.inputs(),
                          variables=held.variables)
    batch = denoised_batch(objective.inputs)
    trained = assert_trains(objective, batch, directory)
    assert_reloads(objective, trained, family,
                   lambda task: denoising(task.model, process=task.process, inputs=task.inputs,
                                          variables=task.variables), batch)


def smooth(model):
    """`model` with its time a unit a loss can differentiate in (`TimeScaled`)."""
    return model.clone(time_scale=SMOOTH_TIME_SCALE) if hasattr(model, "time_scale") else model


def interval_model(family: Family):
    """The family's denoiser of an interval, smooth in time."""
    return smooth(loaded(family).model.clone(interval=True))


def mean_flow(family: Family, directory: Path) -> None:
    """MeanFlow's average velocity over an interval, through a saved run's reload."""
    objective = MeanFlowObjective(interval_model(family), presets.MeanFlow()(), family.inputs(),
                                  ema_decay=None)
    batch = denoised_batch(objective.inputs)
    trained = assert_trains(objective, batch, directory)
    assert_reloads(objective, trained, family, lambda task: MeanFlowObjective(
        task.model, task.process, task.inputs, variables=task.variables, ema_decay=None),
                   batch)


def refuse_mean_flow(family: Family) -> None:
    starting(MeanFlowObjective(loaded(family).model, presets.MeanFlow()(), family.inputs(), ema_decay=None),
             denoised_batch(family.inputs()))


def shortcut(family: Family, directory: Path) -> None:
    """A shortcut model's steps of two sizes, a self-consistency target among them."""
    objective = ShortcutObjective(interval_model(family), presets.Shortcut()(), family.inputs(),
                                  sections=4, bootstrap_every=2, ema_decay=None)
    assert_trains(objective, denoised_batch(objective.inputs), directory)


def refuse_shortcut(family: Family) -> None:
    starting(ShortcutObjective(loaded(family).model, presets.Shortcut()(), family.inputs(),
                               sections=4, bootstrap_every=2, ema_decay=None),
             denoised_batch(family.inputs()))


def teacher_of(family: Family, model, inputs: InputSpec):
    """A teacher of the family's own kind, its variables drawn apart from the student's."""
    objective = denoising(model, process=presets.Flow(), inputs=inputs)
    return objective.model_variables(jax.device_get(objective.initializer(jax.random.key(11))))


def rcm(family: Family, directory: Path) -> None:
    """rCM's consistency and score distillation from a frozen teacher of the same family."""
    student = smooth(loaded(family).model)
    inputs = family.inputs()
    objective = ConsistencyDistillationObjective(
        student, presets.Flow()(), inputs, student_update_freq=1, max_simulation_steps=2,
        variables={TEACHER: teacher_of(family, student, inputs)}, ema_decay=None)
    assert_trains(objective, denoised_batch(inputs), directory)


def ladd(family: Family, directory: Path) -> None:
    """LADD's adversarial distillation on a frozen teacher's block features."""
    student = loaded(family).model
    inputs = family.inputs()
    objective = AdversarialDistillationObjective(
        student, presets.Flow()(), inputs, feature_layers=FEATURE_LAYERS[family.name], cmap_dim=8,
        kernel_size=(3, 3), variables={TEACHER: teacher_of(family, student, inputs)}, ema_decay=None)
    assert_trains(objective, denoised_batch(inputs), directory)


FEATURE_LAYERS: Mapping[str, tuple[str, ...]] = {"simple_dit": ("dit_block_0",),
                                                  "hybrid_dit": ("ssm_block_0",)}
"""The blocks a LADD run names for its discriminator heads, as its configuration does."""


def refuse_ladd(family: Family) -> None:
    student = loaded(family).model
    inputs = family.inputs()
    starting(AdversarialDistillationObjective(
        student, presets.Flow()(), inputs, feature_layers=("dit_block_0",), cmap_dim=8, kernel_size=(3, 3),
        variables={TEACHER: teacher_of(family, student, inputs)}, ema_decay=None),
        denoised_batch(inputs))


def guidance_distillation(family: Family, directory: Path) -> None:
    """A student told the guidance scale regresses onto a guided teacher of its family."""
    student = loaded(family).model
    inputs = family.inputs()
    # The teacher's own towers, as the checkpoint it comes from holds them.
    teacher = denoising(student, process=presets.Flow(), inputs=family.inputs())
    objective = GuidanceDistillationObjective(
        student, presets.Flow()(), inputs, variables={TEACHER: teacher.initializer(jax.random.key(11))},
        scales=(1.0, 4.0), ema_decay=None)
    assert_trains(objective, denoised_batch(inputs), directory)


def refuse_guidance_distillation(family: Family) -> None:
    student = loaded(family).model
    inputs = family.inputs()
    teacher = denoising(student, process=presets.Flow(), inputs=inputs)
    starting(GuidanceDistillationObjective(
        student, presets.Flow()(), inputs, variables={TEACHER: teacher.initializer(jax.random.key(11))},
        scales=(1.0, 4.0), ema_decay=None),
        denoised_batch(inputs))


def flow_grpo(family: Family, directory: Path) -> None:
    """Flow-GRPO's clipped surrogate over the transitions its rollout samples
    from the policy's SDE, each scored against its group by a reward."""
    held = loaded(family)
    inputs = family.inputs()
    objective = FlowGRPOObjective(held.model, presets.Flow(), inputs, guidance=None, steps=2,
                                  variables=held.variables)
    rollout = FlowRollout(objective, reward=lambda samples, batch: np.mean(samples, axis=tuple(
        range(1, np.ndim(samples)))), groups=2, steps=3)
    assert_trains(objective, denoised_batch(inputs), directory, sampled=False, rollout=rollout)


def jepa(family: Family, directory: Path) -> None:
    """I-JEPA's or V-JEPA's prediction of masked blocks' target states, its target encoder averaged."""
    video = len(family.inputs().sample.shape) == 4
    encoder = loaded(family).model if family.name != "jepa_predictor" else models.build(
        "jepa_encoder", **JEPA_ENCODER)
    predictor = loaded(family).model if family.name == "jepa_predictor" else models.build(
        "jepa_predictor", **JEPA_PREDICTOR, factorized=video)
    objective = JepaObjective(encoder, predictor, MultiBlockMask.for_grid(JEPA_GRID, num_targets=2,
                                                                         scale=(0.2, 0.3)),
                              sample=family.inputs().sample)
    batch = {**denoised_batch(InputSpec(objective.sample, {})), "label": np.arange(ROWS) % 2}
    assert_trains(objective, batch, directory)


def block_diffusion(family: Family, directory: Path) -> None:
    """Block-diffusion SFT: canvases denoised against the clean prefix the model's cache holds."""
    held = loaded(family)
    objective = BlockDiffusionObjective(held.given, prompt_length=4, num_canvases=2, canvas_size=2)
    tokens = token_batch(8, seed=13) % family.vocab
    # Response tokens after the prompt are canvases; every clean token after the first is a target.
    batch = {"text": tokens, "canvas_mask": np.ones((ROWS, tokens.shape[1] - 4), bool),
             "encoder_target_mask": np.ones(tokens.shape, np.float32)}
    trained = assert_trains(objective, batch, directory)
    assert_reloads(objective, trained, family, lambda task: BlockDiffusionObjective(
        task.model, prompt_length=4, num_canvases=2, canvas_size=2, variables=task.variables), batch)


def refuse_block_diffusion(family: Family) -> None:
    BlockDiffusionObjective(loaded(family).model, prompt_length=4, num_canvases=2, canvas_size=2)


def supervised_over(family: Family) -> Supervised:
    """Supervised's cross entropy over what the model's call returns, each
    position's token its label, reported beside its accuracy."""
    return Supervised(loaded(family).given, CrossEntropy(labels="text"), (Accuracy(labels="text"),),
                      inputs=InputSpec(Field("text", (SEQ,))))


def supervised(family: Family, directory: Path) -> None:
    """Supervised's loss trains and checkpoints, and the checkpoint scores the
    batch to the same bits. It saves no task (`saved_task` None) and scores
    no tokens, so evaluation is nothing and the checkpoint is the reload."""
    objective = supervised_over(family)
    batch = {"text": token_batch(SEQ) % family.vocab}
    trained = assert_trains(objective, batch, directory, scores=False)
    restored = Checkpoints(str(trained.directory)).variables()
    expected = scored(objective, trained.variables, batch)
    np.testing.assert_array_equal(scored(objective, restored, batch), expected)


def refuse_supervised(family: Family) -> None:
    supervised_over(family)


CALLED = frozenset({"causal_transformer", "causal_transformer+modernbert", "multimodal_transformer+gemma3",
                    "user+causal", "torch+gpt_neox", "nnx+causal", "simple_dit", "user+denoiser"})
"""The families `supervised` covers: models whose call maps tokens to their
logits, as a decoder, an encoder, a media wrapper, a user's module, a torch
model and an NNX one, and denoisers, whose call also needs a time it cannot
give; its loss is the user's own, so `lm` covers the rest."""

DISTILLED = frozenset({"simple_dit", "unet", "flux_transformer", "user+denoiser"})
"""The denoisers the few-step and distillation rows cover: a transformer whose
time is scaled, a convolutional one, a published family's DenoisingCondition,
and a module nothing registers; `diffusion` covers every family."""


# --- methods ---------------------------------------------------------------

@dataclass(frozen=True)
class Method:
    """One row: the facts it reads, the facts its math needs, and how a cell runs."""

    name: str
    reads: frozenset[str]
    """The facts that make a family one this method reads at all."""
    needs: frozenset[str]
    """The facts its math needs past `reads`; a family without one is refused."""
    run: Callable[[Family, Path], None]
    refuse: Callable[[Family], None] | None = None
    """Builds and starts the method on a refused family; it has to raise."""
    trains: tuple[str, ...] = ()
    """The registered objectives this row trains."""
    within: Callable[[Family], bool] = lambda family: True
    """Which families of those it reads a cell covers, for a row whose cells are pairs."""

    def __str__(self) -> str:
        return self.name


def starting(objective, batch) -> None:
    objective.scalar_loss(objective.initializer(jax.random.key(0)), batch, STEP)


def refuse_lm(family: Family) -> None:
    held = loaded(family)
    starting(LMObjective(held.model, SEQ, variables=held.variables), {"text": token_batch(SEQ + 1)})


def refuse_mdlm(family: Family) -> None:
    held = loaded(family)
    starting(MaskedDiffusionObjective(held.model, MDLM(mask_id=mask_id(family))(), SEQ,
                                      variables=held.variables, ema_decay=None),
             {"text": token_batch(SEQ)})


def refuse_grpo(family: Family) -> None:
    held = loaded(family)
    starting(GRPOObjective(held.model, SEQ, beta=0.01, variables=held.variables), rollouts())


def refuse_ppo(family: Family) -> None:
    held = loaded(family)
    batch = rollouts()
    batch[OLD_VALUES_KEY] = batch[RETURNS_KEY] = np.zeros((ROWS, SEQ + 1), np.float32)
    starting(PPOObjective(held.model, SEQ, critic=ValueHead(held.model), variables=held.variables), batch)


def refuse_generation(family: Family) -> None:
    held = loaded(family)
    variables = held.variables or held.model.init(jax.random.key(0), jnp.zeros((1, SEQ), jnp.int32))
    TextGeneration(held.model, variables)(token_batch(SEQ // 2)[:1].tolist(), 2, key=0)


METHODS: tuple[Method, ...] = (
    Method("lm", frozenset({LOGITS}), frozenset({CAUSAL}), lm, refuse_lm, trains=("lm",)),
    Method("dpo", frozenset({LOGITS}), frozenset({CAUSAL}), dpo, trains=("dpo",),
           within=lambda family: family.name in KINDS),
    # Both train on the packed rollout rows Dew's sessions packer writes.
    Method("grpo", frozenset({LOGITS}), frozenset({CAUSAL, PACKING}), grpo, refuse_grpo, trains=("grpo",),
           within=lambda family: family.name in KINDS),
    Method("ppo", frozenset({LOGITS, STATES}), frozenset({CAUSAL, PACKING}), ppo, refuse_ppo,
           trains=("ppo",), within=lambda family: family.name in KINDS),
    Method("distillation", frozenset({LOGITS}), frozenset({CAUSAL}), distillation, trains=("distillation",),
           within=lambda family: family.name in TEACHERS),
    Method("decision", frozenset({STATES, TEXT}), frozenset(), decision, trains=("decision",)),
    Method("mdlm", frozenset({LOGITS}), frozenset({BIDIRECTIONAL}), mdlm, refuse_mdlm,
           trains=("masked_diffusion",)),
    Method("generation", frozenset({LOGITS, CAUSAL}), frozenset({CACHE}), generation, refuse_generation),
    Method("diffusion", frozenset({DENOISES}), frozenset(), diffusion, trains=("diffusion",)),
    Method("mean_flow", frozenset({DENOISES}), frozenset({INTERVAL}), mean_flow, refuse_mean_flow,
           trains=("mean_flow",),
           within=lambda family: family.name in DISTILLED | {"hybrid_dit", "nnx+denoiser"}),
    Method("shortcut", frozenset({DENOISES}), frozenset({INTERVAL}), shortcut, refuse_shortcut,
           trains=("shortcut",), within=lambda family: family.name in DISTILLED),
    Method("rcm", frozenset({DENOISES}), frozenset(), rcm, trains=("rcm",),
           within=lambda family: family.name in DISTILLED),
    Method("flow_grpo", frozenset({DENOISES}), frozenset(), flow_grpo, trains=("flow_grpo",),
           within=lambda family: family.name in DISTILLED),
    Method("ladd", frozenset({DENOISES}), frozenset({FEATURES}), ladd, refuse_ladd, trains=("ladd",),
           within=lambda family: family.name in DISTILLED | {"hybrid_dit"}),
    Method("guidance_distillation", frozenset({DENOISES}), frozenset({GUIDED}), guidance_distillation,
           refuse_guidance_distillation, trains=("guidance_distillation",),
           within=lambda family: family.name in DISTILLED),
    Method("jepa", frozenset({ENCODES}), frozenset(), jepa, trains=("jepa",)),
    Method("supervised", frozenset({LOGITS, DENOISES}), frozenset({LOGITS}), supervised, refuse_supervised,
           trains=("supervised",), within=lambda family: family.name in CALLED),
    Method("block_diffusion", frozenset({LOGITS, CAUSAL}), frozenset({CANVAS}), block_diffusion,
           refuse_block_diffusion, trains=("block_diffusion",),
           within=lambda family: family.name in KINDS),
)


def cells(supported: bool) -> list:
    return [pytest.param(method, family, id=f"{method}-{family}")
            for method in METHODS for family in ALL
            if method.reads <= family.facts and method.within(family)
            and (method.needs <= family.facts) is supported and (supported or method.refuse is not None)]


@pytest.mark.parametrize("method, family", cells(supported=True))
def test_a_supported_cell_trains_evaluates_and_reloads(method, family, tmp_path):
    method.run(family, tmp_path / "run")


CRASHES = re.compile(r"unexpected keyword argument|missing \d+ required|has no attribute")
"""Python's own signature and attribute errors: a crash on the way, not a refusal."""


@pytest.mark.parametrize("method, family", cells(supported=False))
def test_a_refused_cell_names_what_the_family_lacks(method, family):
    reason = "|".join(REASONS[fact] for fact in sorted(method.needs - family.facts))
    with pytest.raises((TypeError, ValueError), match=re.compile(reason, re.IGNORECASE)) as refused:
        method.refuse(family)
    assert not CRASHES.search(str(refused.value)), refused.value


def test_every_registered_objective_has_a_row():
    """An objective no row trains has no proof that it reads every family it should."""
    registered_families()
    dew_objectives = {name for name, member in objectives.items() if member.__module__.startswith("dew.")}
    rows = {name for method in METHODS for name in method.trains}
    assert sorted(dew_objectives - rows) == []
