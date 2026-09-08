#!/usr/bin/env python3
"""Time the real training step, one architecture at a time.

tools/benchmark_data.py measures the loader. This measures what a step of the
Trainer costs for a given architecture, batch size and fsdp width. The step
is the one the trainer compiles for a real run (same objective, same
sharding, same donated state), so a number from this tool is a number from
training.

FLOPs are read off the compiled executable's optimized HLO
(dew.telemetry.instrumentation), and the utilisation is the figure the
trainer logs as train/mfu. Each case is timed twice over the same number of
steps: once with the asynchronous dispatch a real run uses, which gives
ms/step, and once waiting on every step, which gives the p10/p50/p90 spread.

Two registry architectures are composites: `multimodal_transformer` wraps a
decoder in an image tower and projector, and `diffusion_gemma` reads one
decoder both ways for the official block-diffusion loss. A case builds those
from its trunk `config` plus the `media` or `canvas` record the wrapper
needs, so their rows measure the tower and the canvas passes, not the plain
decoder step.

Usage:
    python tools/benchmark_step.py --preset cpu-smoke
    python tools/benchmark_step.py --preset small --json-out /tmp/bench.json
    python tools/benchmark_step.py --preset small --architectures simple_dit unet
    python tools/benchmark_step.py --architectures causal_transformer \\
        --attention-impl cudnn
    python tools/benchmark_step.py --architectures unet \\
        --xla-flags=--xla_gpu_triton_gemm_any=true
    python tools/benchmark_step.py --architectures simple_dit \\
        --profile-dir /tmp/dew-trace --profile-steps 5
    python tools/benchmark_step.py --cases '[{"architecture": "simple_dit",
        "config": {"patch_size": 2, "emb_features": 512, "num_layers": 12,
        "num_heads": 8}, "batch_size": 32, "image_size": 32, "fsdp_size": 2}]'
    python tools/benchmark_step.py --preset small \\
        --architectures multimodal_transformer diffusion_gemma
"""

import contextlib
import dataclasses
import glob
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Iterator, Literal, Mapping, Sequence

import jax
import numpy as np
import optax
import tyro

from dew.diffusion import presets
from dew.inputs import CharTable, Condition, Field, InputSpec
from dew.inputs.encoders import ConditionEncoder
from dew.nn.backbones.unet_condition import DenoisingCondition
from dew.objectives.base import Variables
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import ModelInputs
from dew.nn.multimodal import MultimodalTransformer, VisionConditioner
from dew.nn.vision import ProjectorBase, TowerBase, projector_from_record, tower_from_record
from dew.objectives.diffusion import BlockDiffusionObjective, DiffusionObjective
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.objectives.lm import LMObjective
from dew import models  # naming a registry fills it
from dew.registry import resolve_dtype, with_precision
from dew.telemetry.devices import apply_xla_flags
from dew.telemetry.instrumentation import model_flops_utilization
from dew.training import Layout, MeshSpec, Trainer
from dew.training.distributed import DevicePrefetchIterator

# The CLIP-L/14 context's shape, from the library's table encoder: a benchmark
# of the model should not spend its first minute downloading a text tower, and
# the step cost depends only on the context's shape.
TEXT_TOKENS = 77
TEXT_FEATURES = 768

Batch = dict[str, np.ndarray | Mapping[str, np.ndarray] | ModelInputs]
Row = dict[str, object]


class _UNetTextTable(ConditionEncoder[str]):
    """The benchmark table in the native conditional UNet input format."""

    def __init__(self, table: CharTable):
        self.table = table
        self.params = table.params

    @classmethod
    def from_pretrained(cls, checkpoint: str = "char_table", *, tokens: int = TEXT_TOKENS,
                        features: int = TEXT_FEATURES, vocab: int = 130, seed: int = 0, dtype=None):
        return cls(CharTable.from_pretrained(checkpoint, tokens=tokens, features=features,
                                            vocab=vocab, seed=seed, dtype=dtype))

    def tokenize(self, data: Sequence[str]) -> Mapping[str, np.ndarray]:
        return self.table.tokenize(data)

    def encode(self, params: Variables, tokens) -> DenoisingCondition:
        return DenoisingCondition(self.table.encode(params, tokens).hidden)

    def to_json(self) -> dict:
        return self.table.to_json()


def text_condition(architecture: str = "") -> Condition:
    table = _UNetTextTable if architecture == "unet_2d_condition" else CharTable
    return Condition(table.from_pretrained(tokens=TEXT_TOKENS, features=TEXT_FEATURES))


@dataclass(frozen=True)
class Case:
    """One measurement: what to build, how much to feed it, how to shard it."""

    architecture: str
    config: dict[str, object] = field(default_factory=dict)
    dtype: str = 'float32'
    """Compute dtype, written into the model config by the precision policy."""
    batch_size: int = 8
    fsdp_size: int = 1
    expert_size: int = 1
    """Devices the expert dimension of an MoE layer is split over, as
    --trainer.expert-size; 1 replicates every expert."""
    image_size: int = 32
    channels: int = 3
    """Input channels, including four-channel latent diffusion inputs."""
    frames: int = 0
    """Video models take (frames, H, W, C) samples; 0 means images."""
    predictor: dict[str, object] | None = None
    """Set for JEPA: the architecture is an encoder and this builds its predictor."""
    canvas: dict[str, object] | None = None
    """Set for a block-diffusion decoder: `prompt_length` and `canvas_size`.
    The architecture is the shared encoder/decoder trunk `config` builds, and
    the row of `seq_len + 1` tokens splits into a clean prompt and whole
    canvases the way recipes/lm/train.py's block objective splits it."""
    media: dict[str, object] | None = None
    """Set for a vision-conditioned decoder: the `tower` and `projector`
    records, the `family` and `image_token_id` the wrapper carries, `images`
    per row and the `pixels` shape the checkpoint's processor emits. The
    architecture is the decoder `config` builds and the images ride the batch
    beside the tokens, as they do on the loaded multimodal path."""
    seq_len: int = 0
    """Set for language models: batches are token windows of this length, not images."""
    packed_documents: int = 0
    """Set for language models: documents packed into every row, with the
    segment ids and positions the packed loader emits; 0 feeds a fixed
    window, the form a stream of tokens gives."""
    head_chunks: int | None = None
    """For language models, the vocabulary slices the loss scores a batch in;
    None is the objective's own default."""
    fsdp_min_param_size: int = 2 ** 16

    @property
    def is_jepa(self) -> bool:
        return self.predictor is not None

    @property
    def is_lm(self) -> bool:
        return self.seq_len > 0

    @property
    def packs_documents(self) -> bool:
        """Whether --packed-documents means anything for this case: a plain
        token window packs, a canvas row is the objective's own fixed
        geometry, and a media row is what a processor emitted."""
        return self.is_lm and self.canvas is None and self.media is None

    @property
    def sample_shape(self) -> tuple[int, ...]:
        square = (self.image_size, self.image_size, self.channels)
        return square if self.frames == 0 else (self.frames, *square)

    @property
    def label(self) -> str:
        mixture = self.config.get("mixture")
        experts = f" x{mixture['experts']}experts" if isinstance(mixture, dict) else ""
        canvases = f" x{canvas_split(self)[2]}canvas" if self.canvas else ""
        images = f" x{images_per_row(self)}img" if self.media else ""
        return (f"{self.architecture}{experts}{canvases}{images} b{self.batch_size} "
                f"fsdp{self.fsdp_size} expert{self.expert_size}")


def _count(record: Mapping[str, object], key: str, owner: str) -> int:
    value = record.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"{owner} needs a positive integer {key}, got {value!r}")
    return value


def canvas_split(case: Case) -> tuple[int, int, int]:
    """A block-diffusion row as (prompt_length, canvas_size, num_canvases).

    The row holds `seq_len + 1` tokens, the width the token-windows loader
    emits, and splits the way recipes/lm/train.py splits it for the official
    fine-tuning objective: a clean prompt, then whole canvases.
    """
    canvas = case.canvas or {}
    prompt = _count(canvas, "prompt_length", "a canvas case")
    width = _count(canvas, "canvas_size", "a canvas case")
    response = case.seq_len + 1 - prompt
    if response < width or response % width:
        raise ValueError(
            f"{case.architecture}'s row of {case.seq_len + 1} tokens is not "
            f"{prompt} prompt tokens followed by whole canvases of {width}")
    return prompt, width, response // width


def images_per_row(case: Case) -> int:
    return _count(case.media or {}, "images", "a media case")


def media_pixels(case: Case) -> tuple[int, ...]:
    """One processed image's (channels, height, width), as a processor emits it."""
    shape = (case.media or {}).get("pixels")
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(type(size) is not int or size < 1 for size in shape)):
        raise ValueError(
            f"a media case's pixels is [channels, height, width], got {shape!r}")
    return tuple(shape)


def media_values(case: Case) -> tuple[str, TowerBase, ProjectorBase, int]:
    """The wrapper's media fields as built values: the checkpoint family, the
    tower and projector its records name, and the placeholder token id."""
    media = case.media or {}
    family = media.get("family")
    if not isinstance(family, str) or not family:
        raise ValueError(f"a media case names its checkpoint family, got {family!r}")
    for name in ("tower", "projector"):
        if not isinstance(media.get(name), Mapping):
            raise ValueError(f"a media case's {name} is a record, got {media.get(name)!r}")
    token = media.get("image_token_id")
    if type(token) is not int or token < 0:
        raise ValueError(f"a media case's image_token_id is an id, got {token!r}")
    return (family, tower_from_record(media["tower"]),
            projector_from_record(media["projector"]), token)


def image_tokens(case: Case) -> int:
    """Text slots one image fills: the soft tokens this case's own projector
    emits for it.

    Read off the tower and projector instead of declared beside them, so a
    row cannot mark a width the modules do not produce. The trace is
    abstract: no parameter is allocated and no kernel compiles.
    """
    family, tower, projector, _ = media_values(case)
    conditioner = VisionConditioner(family, tower, projector)
    features = jax.eval_shape(
        lambda pixels: conditioner.init_with_output(
            jax.random.key(0), {"pixel_values": pixels})[0],
        jax.ShapeDtypeStruct((1, 1, *media_pixels(case)), np.float32))
    return features.shape[1]


def cases_from_json(text: str) -> list[Case]:
    """`--cases`: a JSON list of objects whose keys are Case fields."""
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("--cases is a JSON list of objects, one per case")
    fields = {f.name for f in dataclasses.fields(Case)}
    cases = []
    for spec in parsed:
        if not isinstance(spec, dict):
            raise ValueError(f"--cases entry {spec!r} is not a JSON object")
        unknown = sorted(set(spec) - fields)
        if unknown:
            raise ValueError(
                f"--cases entry has no such field {unknown}; valid fields are {sorted(fields)}")
        cases.append(Case(**spec))
    return cases


JsonCases = Annotated[
    list[Case],
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda args: cases_from_json(args[0]),
        is_instance=lambda value: isinstance(value, list),
        str_from_instance=lambda cases: [json.dumps([dataclasses.asdict(c) for c in cases])],
    ),
]
"""A list of cases, written as one JSON string on the command line."""


def cpu_smoke_cases() -> list[Case]:
    """Tiny enough to run anywhere, real enough to compile the same step.

    The last two are the composites at a size a CPU compiles in seconds: the
    same wrappers, the same objectives and the same media and canvas work as
    --preset small, on a two-layer decoder.
    """
    tiny_decoder: dict[str, object] = {"vocab_size": 256, "emb_features": 32,
                                       "num_layers": 2, "num_heads": 2,
                                       "mlp_features": 64, "max_seq_len": 16}
    return [
        Case("simple_dit", {"patch_size": 4, "emb_features": 64, "num_layers": 2,
                            "num_heads": 2, "mlp_ratio": 2},
             batch_size=8, image_size=16, fsdp_min_param_size=256),
        Case("unet_2d_condition", {"stages": [{"features": 32, "heads": 2}, {"features": 64, "heads": 4}],
                                    "blocks_per_level": 1, "in_channels": 4, "out_channels": 4},
             batch_size=8, image_size=16, channels=4, fsdp_min_param_size=256),
        Case("jepa_encoder", {"patch_size": 4, "emb_features": 32, "num_layers": 2,
                              "num_heads": 2, "mlp_ratio": 2},
             predictor={"grid": (4, 4), "emb_features": 32, "predictor_features": 16,
                        "num_layers": 1, "num_heads": 2, "mlp_ratio": 2},
             batch_size=8, image_size=16, fsdp_min_param_size=256),
        Case("causal_transformer", tiny_decoder,
             batch_size=8, seq_len=16, fsdp_min_param_size=256),
        Case("multimodal_transformer", tiny_decoder,
             media={"family": "gemma3", "image_token_id": 255, "images": 1,
                    "pixels": [3, 16, 16],
                    "tower": {"kind": "siglip", "hidden_size": 32, "intermediate_size": 64,
                              "num_layers": 1, "num_heads": 2, "image_size": 16,
                              "patch_size": 8},
                    "projector": {"kind": "gemma", "vision_width": 32, "text_width": 32,
                                  "patches_per_side": 2, "tokens_per_side": 2}},
             batch_size=8, seq_len=15, fsdp_min_param_size=256),
        Case("diffusion_gemma", {**tiny_decoder, "layer_scalar": "frozen"},
             canvas={"prompt_length": 8, "canvas_size": 4},
             batch_size=8, seq_len=15, fsdp_min_param_size=256),
    ]


def small_cases(dtype: str) -> list[Case]:
    """Every registry architecture at a size that fits one 16 GB card in bf16.

    Sized so the whole sweep takes minutes: real token counts (256 image
    tokens at 64px/patch 4) and real widths, but few layers.
    """
    dit: dict[str, object] = {"patch_size": 4, "emb_features": 384, "num_layers": 6,
                              "num_heads": 6, "mlp_ratio": 4}
    unet: dict[str, object] = {"emb_features": 256, "feature_depths": [64, 128, 256],
                               "attention_configs": [None, {"heads": 4}, {"heads": 4}],
                               "num_res_blocks": 2, "num_middle_res_blocks": 1}
    encoder: dict[str, object] = {"patch_size": 4, "emb_features": 384, "num_layers": 6,
                                  "num_heads": 6, "mlp_ratio": 4}
    predictor: dict[str, object] = {"grid": (16, 16), "emb_features": 384,
                                    "predictor_features": 192, "num_layers": 3,
                                    "num_heads": 6, "mlp_ratio": 4}
    # GPT-2 small's width and heads at a quarter of its depth, on 512-token
    # rows. The two composites wrap this same decoder, so their rows are the
    # plain decoder's rows plus the work their wrapper adds.
    decoder: dict[str, object] = {"vocab_size": 50304, "emb_features": 768, "num_layers": 3,
                                  "num_heads": 12, "mlp_features": 3072, "max_seq_len": 512}

    cases = [
        Case("unet", unet, batch_size=16, image_size=64),
        Case("unet_2d_condition", {"stages": [{"features": 64, "heads": 4}, {"features": 128, "heads": 4},
                                              {"features": 256, "heads": 8, "cross_attention": False}],
                                    "blocks_per_level": 1, "in_channels": 4, "out_channels": 4},
             batch_size=4, image_size=32, channels=4),
        Case("uvit", {**dit, "num_layers": 6}, batch_size=16, image_size=64),
        Case("simple_udit", {**dit, "num_layers": 6}, batch_size=16, image_size=64),
        Case("simple_dit", dit, batch_size=16, image_size=64),
        Case("simple_mmdit", dit, batch_size=16, image_size=64),
        Case("hierarchical_mmdit",
             {"base_patch_size": 2, "emb_features": (192, 384, 576),
              "num_layers": (2, 2, 2), "num_heads": (3, 6, 9), "mlp_ratio": 4},
             batch_size=16, image_size=64),
        Case("hybrid_dit", {**dit, "ssm_state_dim": 64, "ssm_attention_ratio": "3:1"},
             batch_size=16, image_size=64),
        Case("video_dit", {**dit, "num_layers": 4}, batch_size=4, image_size=64, frames=8),
        Case("unet_3d", {**unet, "temporal_heads": 4},
             batch_size=4, image_size=64, frames=8),
        Case("jepa_encoder", encoder, predictor=predictor, batch_size=16, image_size=64),
        Case("jepa_video_encoder", {**encoder, "num_layers": 4},
             predictor={**predictor, "num_layers": 2, "factorized": True},
             batch_size=4, image_size=64, frames=8),
        Case("causal_transformer", decoder, batch_size=16, seq_len=512),
        # The same decoder with an 8-expert, top-2 feed-forward on every second
        # layer, which is the sparse shape the 4.7 acceptance run trains
        Case("causal_transformer",
             {**decoder, "mixture": {"experts": 8, "top_k": 2, "every": 2}},
             batch_size=16, seq_len=512),
        # The same decoder as a vision-conditioned one: SigLIP-so400m's widths
        # at four layers over a 448px crop, pooled to the 256 soft tokens
        # Gemma 3 gives an image, so half of every 512-token row is media. One
        # image a row keeps the tower's cost the per-row cost a caption batch
        # pays, and the batch is halved because each row carries one. The trunk
        # is the plain decoder, so the row is the tower, the projector, the
        # fusion and a causal decoder; Gemma 3's bidirectional-over-image mask
        # is a per-family attention setting this case does not turn on.
        Case("multimodal_transformer", decoder,
             media={"family": "gemma3", "image_token_id": 50303, "images": 1,
                    "pixels": [3, 448, 448],
                    "tower": {"kind": "siglip", "hidden_size": 1152,
                              "intermediate_size": 4304, "num_layers": 4,
                              "num_heads": 16, "image_size": 448, "patch_size": 14},
                    "projector": {"kind": "gemma", "vision_width": 1152, "text_width": 768,
                                  "patches_per_side": 32, "tokens_per_side": 16}},
             batch_size=8, seq_len=512),
        # The same decoder read both ways by the official DiffusionGemma
        # fine-tuning loss: one encoder pass over the whole row into the
        # cache, then two decoder passes over the response for the
        # self-conditioning branch. `frozen` is the published scalar policy
        # the objective migrates to a trained one. The published vocabulary is
        # 262144 and this loss holds whole `[B, S, vocab]` logit tensors, so
        # the case keeps the other rows' vocabulary and a quarter of their
        # batch.
        Case("diffusion_gemma", {**decoder, "layer_scalar": "frozen"},
             canvas={"prompt_length": 256, "canvas_size": 128},
             batch_size=4, seq_len=511),
    ]
    cases = [dataclasses.replace(case, dtype=dtype) for case in cases]
    # jepa_predictor has no step of its own: it is built through the registry
    # inside the two JEPA cases above.
    covered = {case.architecture for case in cases} | {"jepa_predictor"}
    missing = set(models) - covered
    if missing:
        raise ValueError(
            f"--preset small does not cover {sorted(missing)}; add a case for every "
            "architecture in dew.registry.models")
    return cases


@dataclass(frozen=True)
class BenchmarkConfig:
    """Which cases to time, and how."""

    preset: Literal['small', 'cpu-smoke'] = 'small'
    cases: JsonCases = field(default_factory=list)
    """Explicit cases as a JSON list of Case fields; replaces the preset."""
    architectures: list[str] | None = None
    """Keep only these cases from the preset."""
    warmup: int = 2
    steps: int = 100
    """Measured steps per case, timed twice: once dispatched asynchronously for
    ms/step, once waiting per step for the p10/p50/p90 spread."""
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16'
    """Model compute dtype for --preset small; losses stay fp32 either way."""
    attention_impl: Literal['auto', 'reference', 'xla', 'cudnn', 'tpu'] = 'auto'
    """Attention kernel, through the same precision policy a recipe uses."""
    xla_flags: str | None = None
    """Appended to XLA_FLAGS before the first JAX call, as TrainerConfig.xla_flags
    is. A flag only takes effect in a process that has not opened a backend
    yet, so a sweep runs one configuration per process."""
    batch_size: int | None = None
    """Override every case's batch size."""
    fsdp_size: int | None = None
    image_size: int | None = None
    frames: int | None = None
    """Frame count for the video cases; image cases are left alone."""
    packed_documents: int | None = None
    """Documents per row for the language-model cases; the others are left
    alone. This is the packed loader's batch, which reroutes attention off the
    fused kernel."""
    profile_dir: str | None = None
    """Trace `profile_steps` more steps into this directory with jax.profiler
    after the timed windows, and read the device timeline back into the row:
    the fraction of the window the device was busy, kernels per step and
    kernel milliseconds per step by category. The trace itself stays on disk
    for TensorBoard or Perfetto."""
    profile_steps: int = 5
    json_out: str | None = None
    quiet: bool = True
    """Silence the trainer's own prints, which are per-run noise here."""


def build_cases(config: BenchmarkConfig) -> list[Case]:
    if config.cases:
        cases = config.cases
    elif config.preset == 'cpu-smoke':
        cases = cpu_smoke_cases()
    else:
        cases = small_cases(config.dtype)

    if config.architectures:
        wanted = set(config.architectures)
        unknown = wanted - {case.architecture for case in cases}
        if unknown:
            raise ValueError(f"--architectures {sorted(unknown)} not in preset {config.preset}")
        cases = [case for case in cases if case.architecture in wanted]

    overrides: dict[str, int] = {}
    for name in ('batch_size', 'fsdp_size', 'image_size'):
        value = getattr(config, name)
        if value is not None:
            overrides[name] = value

    def apply(case: Case) -> Case:
        # An image model handed a (T, H, W, C) sample is a rank error, so
        # --frames only resizes the video cases, and packing is a plain token
        # window's batch.
        frames = {} if config.frames is None or case.frames == 0 else {'frames': config.frames}
        packed = ({'packed_documents': config.packed_documents}
                  if config.packed_documents is not None and case.packs_documents else {})
        return dataclasses.replace(case, **overrides, **frames, **packed)

    return [apply(case) for case in cases]


def lm_objective(case: Case, model) -> LMObjective:
    """Next-token cross entropy over the case's rows, with its own head
    chunking where it names one."""
    if case.head_chunks is None:
        return LMObjective(model, case.seq_len)
    return LMObjective(model, case.seq_len, head_chunks=case.head_chunks)


def build_trainer(case: Case, attention_impl: str = 'auto') -> Trainer:
    """The trainer a recipe would build for this case, minus the tracker and the
    checkpoints.

    The model goes through the same precision function the recipes use, so the
    dtype and the attention kernel land in the nested unet attention configs
    too, and a row of this table is a row a real run would produce.

    A composite takes built values rather than a flat record, so its trunk
    goes through the policy and the wrapper takes it, the way the pretrained
    loader assembles the same two models (dew.interop.pretrained and
    dew.interop.diffusion_gemma.build).
    """
    def built(architecture: str, config: Mapping[str, object]):
        return models.build(architecture, **with_precision(
            architecture, config, dtype=case.dtype, attention_impl=attention_impl))

    sample_key = "video" if case.frames else "image"

    if case.canvas is not None:
        prompt, width, count = canvas_split(case)
        model = DiffusionGemma(text=built("causal_transformer", case.config),
                               canvas_length=width)
        objective = BlockDiffusionObjective(
            model, prompt_length=prompt, canvas_size=width, num_canvases=count)
    elif case.media is not None:
        family, tower, projector, token = media_values(case)
        model = MultimodalTransformer(
            built("causal_transformer", case.config), tower, projector, family, token,
            dtype=resolve_dtype(case.dtype),
            attention_impl=None if attention_impl == 'reference' else attention_impl)
        objective = lm_objective(case, model)
    elif case.is_lm:
        objective = lm_objective(case, built(case.architecture, case.config))
    elif case.predictor is not None:
        model = built(case.architecture, case.config)
        patch = case.config.get("patch_size", 16)
        if not isinstance(patch, int):
            raise ValueError(f"{case.architecture}'s patch_size is {patch!r}, not an int")
        grid = (case.image_size // patch, case.image_size // patch)
        objective = JepaObjective(
            model, built("jepa_predictor", {**case.predictor, "grid": grid}),
            multi_block_mask(grid, num_targets=2, scale=(0.2, 0.3)),
            sample=Field(sample_key, case.sample_shape))
    else:
        model = built(case.architecture, case.config)
        keyword = "conditioning" if case.architecture == "unet_2d_condition" else "textcontext"
        inputs = InputSpec(Field(sample_key, case.sample_shape),
                           {keyword: text_condition(case.architecture)})
        objective = DiffusionObjective(model, presets.EDM()(), inputs)

    return Trainer(
        objective, optax.adam(1e-4), key=jax.random.key(0),
        mesh=MeshSpec(fsdp=case.fsdp_size, expert=case.expert_size),
        layout=Layout(min_shard=case.fsdp_min_param_size),
        checkpoints=None, tracker=None)


def media_row(case: Case, tokens: np.ndarray, rng: np.random.Generator) -> ModelInputs:
    """One media batch in the numeric form a processor hands the model: the
    placeholder run the images fill, the feature each of those slots reads,
    and the pixels themselves.

    Every row marks `image_tokens` slots for each of its images, so every
    feature the tower computes is read by a slot. A shorter run would pay for
    features the decoder never sees, and a longer one would read a feature
    twice.
    """
    _, _, _, token = media_values(case)
    images = images_per_row(case)
    vocab = case.config.get("vocab_size")
    if not isinstance(vocab, int) or token >= vocab:
        raise ValueError(
            f"a media row marks its slots with token {token}, which is not in "
            f"{case.architecture}'s vocabulary of {vocab!r}")
    slots = images * image_tokens(case)
    if slots > tokens.shape[1]:
        raise ValueError(
            f"{images} images fill {slots} slots of {case.architecture}'s "
            f"{tokens.shape[1]}-token row, which has no room for them")
    tokens = tokens.copy()
    tokens[:, :slots] = token
    indices = np.full(tokens.shape, -1, np.int32)
    indices[:, :slots] = np.arange(slots)
    pixels = rng.normal(size=(case.batch_size, images, *media_pixels(case)))
    return ModelInputs(tokens, {"image_indices": indices},
                       {"pixel_values": pixels.astype(np.float32)})


def batches(case: Case) -> Iterator[Batch]:
    """One host batch, reused: the loader is benchmarked by benchmark_data.py."""
    rng = np.random.default_rng(0)
    batch: Batch = {}
    if case.is_lm:
        vocab = case.config["vocab_size"]
        if not isinstance(vocab, int):
            raise ValueError(f"{case.architecture}'s vocab_size is {vocab!r}, not an int")
        width = case.seq_len + 1
        # A canvas row's target masks are read off the pad id, so a drawn zero
        # would move the objective's own target support with the seed. Every
        # other row takes it as an ordinary token.
        lowest = 1 if case.canvas is not None else 0
        tokens = rng.integers(lowest, vocab, size=(case.batch_size, width)).astype(np.int32)
        batch["text"] = tokens if case.media is None else media_row(case, tokens, rng)
        if case.packed_documents:
            # Equal documents tiling the row. A packed row from the loader is
            # ragged and can end in padding; what the mask and the kernel cost
            # see is how many segments the row carries, and equal ones make
            # the case reproducible.
            per_document = -(-width // case.packed_documents)
            document = np.repeat(np.arange(case.packed_documents), per_document)[:width]
            rows = (case.batch_size, 1)
            batch["text_segment_ids"] = np.tile(document + 1, rows).astype(np.int32)
            batch["text_positions"] = np.tile(
                np.arange(width) - document * per_document, rows).astype(np.int32)
    else:
        sample_key = "video" if case.frames else "image"
        batch[sample_key] = rng.integers(
            0, 256, size=(case.batch_size, *case.sample_shape)).astype(np.float32)
        if not case.is_jepa:
            batch["text"] = text_condition(case.architecture).encoder.tokenize(["a flower"] * case.batch_size)
    while True:
        yield batch


def device_peak_bytes() -> int | None:
    """The allocator's high-water mark, where the backend reports one (not CPU).

    Monotonic for the life of the process and with no reset hook, so in a sweep
    it is this case's own peak only for the first case; every later case gets
    an upper bound plus its own delta.
    """
    stats = jax.local_devices()[0].memory_stats()
    if not stats:
        return None
    return stats.get('peak_bytes_in_use') or stats.get('bytes_in_use')


def parameter_count(params) -> int:
    return int(sum(np.prod(leaf.shape, dtype=np.int64) for leaf in jax.tree.leaves(params)))


# A kernel's category from the tokens of its name, first match wins: XLA
# names its fusions after the ops they hold (`loop_convert_fusion`,
# `input_add_reduce_fusion`, `gemm_fusion_dot`), cuDNN and cuBLAS after the
# kernel family. Whole tokens, not substrings, so `convert` is not `conv`.
KERNEL_CATEGORIES = (
    ("attention", ("sdpa", "fmha", "flash")),
    ("conv", ("conv", "fprop", "dgrad", "wgrad", "implicit")),
    ("gemm", ("gemm", "cublas", "cutlass", "nvjet", "xmma", "matmul", "dot")),
    ("reduce", ("reduce",)),
    ("convert", ("convert",)),
    ("copy", ("memcpy", "memset", "copy", "transpose", "concatenate", "gather",
              "scatter", "slice", "pad", "broadcast", "select", "dynamic")),
    ("elementwise", ("fusion",)),
)


def kernel_category(name: str) -> str:
    lowered = name.lower()
    if "cudnn::fusion" in lowered:
        # cuDNN's helpers around its flash kernel (dO.O, dQ rearrangement).
        return "attention"
    tokens = set(re.split(r"[^a-z0-9]+", lowered))
    for category, needles in KERNEL_CATEGORIES:
        if tokens & set(needles):
            return category
    return "other"


def device_timeline(directory: str, steps: int) -> dict[str, Any]:
    """What the device did during the traced `steps`, from the newest trace
    under `directory`.

    Busy is the union of kernel intervals across the traced devices and
    streams, divided by the earliest-start to latest-end window. For a
    multi-device trace this measures time when any device ran a kernel,
    not average utilization across devices. Category timings sum events
    and can overlap; they are not an additional wall-clock measurement.
    """
    from jax.profiler import ProfileData

    traces = sorted(glob.glob(os.path.join(directory, "**", "*.xplane.pb"), recursive=True),
                    key=os.path.getmtime)
    kernels = []
    for plane in ProfileData.from_file(traces[-1]).planes:
        if not plane.name.startswith("/device:"):
            continue
        for line in plane.lines:
            for event in line.events:
                kernels.append((event.name, event.start_ns, event.end_ns))
    if not kernels:
        raise ValueError(
            f"the trace under {directory} holds no device kernels: the profiler "
            "saw no accelerator, and a CPU run has no device timeline to read")
    kernels.sort(key=lambda kernel: kernel[1])
    busy, current_start, current_end = 0, kernels[0][1], kernels[0][2]
    for _, start, end in kernels[1:]:
        if start > current_end:
            busy += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    busy += current_end - current_start
    window = current_end - kernels[0][1]
    by_category: dict[str, float] = {}
    by_name: dict[str, float] = {}
    for name, start, end in kernels:
        by_category[kernel_category(name)] = by_category.get(kernel_category(name), 0.0) + end - start
        by_name[name] = by_name.get(name, 0.0) + end - start
    per_step = 1e-6 / steps
    return {
        "profiled_steps": steps,
        "kernels_per_step": len(kernels) / steps,
        "device_busy_percent": 100.0 * busy / window,
        "device_busy_ms_per_step": busy * per_step,
        "device_window_ms_per_step": window * per_step,
        "kernel_ms_by_category": {
            category: ms * per_step
            for category, ms in sorted(by_category.items(), key=lambda item: -item[1])},
        "top_kernels": [(name[:100], ms * per_step)
                        for name, ms in sorted(by_name.items(), key=lambda item: -item[1])[:12]],
    }


def measure(case: Case, config: BenchmarkConfig) -> Row:
    """Warm up, then time the compiled step over a fixed number of steps."""
    if config.steps < 1:
        raise ValueError(f"--steps must be at least 1, got {config.steps}")
    peak_before = device_peak_bytes()
    trainer = build_trainer(case, config.attention_impl)

    with DevicePrefetchIterator(batches(case), trainer.device_mesh) as source:
        abstract = jax.eval_shape(trainer.initial_state)
        state = jax.jit(trainer.initial_state, out_shardings=trainer.shardings(abstract))()
        

        initial_batch = next(source)
        jax.block_until_ready((state, initial_batch))
        compile_start = time.perf_counter()
        compiled = trainer.compile(state, initial_batch)
        compile_seconds = time.perf_counter() - compile_start

        def step(state):
            state, loss, _, finite, _ = compiled(state, next(source))
            return state, loss, finite

        # At least one warm step, so the first dispatch of the executable is
        # outside the timed window.
        state, loss, is_finite = step(state)
        for _ in range(config.warmup - 1):
            state, loss, is_finite = step(state)
        loss.block_until_ready()

        start = time.perf_counter()
        for _ in range(config.steps):
            state, loss, is_finite = step(state)
        loss.block_until_ready()
        elapsed = time.perf_counter() - start

        # A second window of the same length, waiting on every step, for the
        # spread. The loop above dispatches asynchronously on purpose, as a run
        # does, so timing its individual iterations would time the dispatch and
        # not the step. These per-step numbers are a different quantity from
        # ms_per_step above, and each carries one synchronisation.
        synced = []
        for _ in range(config.steps):
            step_start = time.perf_counter()
            state, loss, is_finite = step(state)
            loss.block_until_ready()
            synced.append((time.perf_counter() - step_start) * 1e3)
        p10, p50, p90 = np.percentile(synced, [10, 50, 90])

        timeline = {}
        if config.profile_dir:
            # After the timed windows, so the trace's own overhead is not in them.
            directory = os.path.join(config.profile_dir, case.label.replace(" ", "_"))
            jax.profiler.start_trace(directory)
            try:
                for _ in range(config.profile_steps):
                    state, loss, is_finite = step(state)
                loss.block_until_ready()
            finally:
                primary = sys.exception()
                try:
                    jax.profiler.stop_trace()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(f"Profiler stop failed: {error!r}")
            timeline = device_timeline(directory, config.profile_steps)
        flops = trainer.flops_per_step
        step_time = elapsed / config.steps
        utilization = model_flops_utilization(flops, step_time)
        peak = device_peak_bytes()
        row: Row = {
            "architecture": case.architecture,
            "batch_size": case.batch_size,
            "fsdp_size": case.fsdp_size,
            "expert_size": case.expert_size,
            "sample_shape": [case.seq_len] if case.is_lm else list(case.sample_shape),
            "packed_documents": case.packed_documents,
            # A row's own extra work, so a number is readable without its
            # case: the images each row carries and their processed shape,
            # and the canvases the block loss splits the response into.
            "images_per_row": 0 if case.media is None else images_per_row(case),
            "image_pixels": None if case.media is None else list(media_pixels(case)),
            "canvases_per_row": 0 if case.canvas is None else canvas_split(case)[2],
            "dtype": case.dtype,
            "attention_impl": config.attention_impl,
            "xla_flags": config.xla_flags,
            "devices": trainer.device_mesh.devices.size,
            "device_kind": jax.devices()[0].device_kind,
            "params": parameter_count(state.params),
            "measured_steps": config.steps,
            "compile_seconds": round(compile_seconds, 2),
            "ms_per_step": round(step_time * 1e3, 3),
            "p10_ms": round(float(p10), 3),
            "p50_ms": round(float(p50), 3),
            "p90_ms": round(float(p90), 3),
            "samples_per_sec": round(case.batch_size / step_time, 2),
            "flops_per_step": flops,
            "utilization": utilization,
            "peak_device_bytes": peak,
            "case_peak_delta_bytes": (
                None if peak is None or peak_before is None else max(0, peak - peak_before)),
            "loss": float(loss),
            "finite": bool(is_finite),
            **timeline,
        }
        return row


TABLE_COLUMNS = (
    # Wide enough for the longest registry name, multimodal_transformer.
    ("architecture", "architecture", 22, "{}"),
    ("batch_size", "batch", 6, "{}"),
    ("fsdp_size", "fsdp", 5, "{}"),
    ("expert_size", "expert", 7, "{}"),
    ("params", "params", 12, "{:,}"),
    ("ms_per_step", "ms/step", 9, "{:.1f}"),
    ("p10_ms", "p10", 7, "{:.1f}"),
    ("p50_ms", "p50", 7, "{:.1f}"),
    ("p90_ms", "p90", 7, "{:.1f}"),
    ("samples_per_sec", "samples/s", 10, "{:.1f}"),
    ("flops_per_step", "GFLOP/step", 11, "{:.1f}"),
    ("utilization", "util %", 7, "{:.1f}"),
    ("peak_device_bytes", "peak GiB", 9, "{:.2f}"),
)
# Units the table shows a column in: FLOPs as GFLOP, a fraction as a
# percentage, bytes as GiB.
TABLE_SCALE = {"flops_per_step": 1e-9, "utilization": 100.0, "peak_device_bytes": 2 ** -30}


def format_table(rows: list[Row]) -> str:
    header = " ".join(title.rjust(width) if key != "architecture" else title.ljust(width)
                      for key, title, width, _ in TABLE_COLUMNS)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for key, _, width, fmt in TABLE_COLUMNS:
            value = row.get(key)
            if value is None:
                text = "n/a"
            elif isinstance(value, (int, float)):
                text = fmt.format(value * TABLE_SCALE.get(key, 1))
            else:
                text = fmt.format(value)
            cells.append(text.ljust(width) if key == "architecture" else text.rjust(width))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def run(config: BenchmarkConfig) -> list[Row]:
    rows: list[Row] = []
    for case in build_cases(config):
        # The trainer narrates state generation and input shapes per case,
        # which buries the numbers this tool exists to print.
        sink = (contextlib.redirect_stdout(io.StringIO()) if config.quiet
                else contextlib.nullcontext())
        with sink:
            row = measure(case, config)
        rows.append(row)
        print(f"{case.label}: {row['ms_per_step']} ms/step, "
              f"{row['samples_per_sec']} samples/s")
        categories = row.get("kernel_ms_by_category")
        if isinstance(categories, dict):
            categories = ", ".join(f"{category} {ms:.2f}" for category, ms in categories.items())
            print(f"  device busy {row['device_busy_percent']:.1f}% of "
                  f"{row['device_window_ms_per_step']:.2f} ms/step, "
                  f"{row['kernels_per_step']:.0f} kernels/step; ms/step by category: "
                  f"{categories}")
        if config.json_out:
            # A GPU sweep is minutes of compilation per case; rewriting the
            # file as each case lands means an interrupted sweep still keeps
            # the cases it did measure.
            write_json(rows, config.json_out)
    return rows


def write_json(rows: list[Row], path: str) -> None:
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=2)


def main(config: BenchmarkConfig) -> list[Row]:
    apply_xla_flags(config.xla_flags)
    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print(f"dtype {config.dtype}, attention_impl {config.attention_impl}, "
          f"XLA_FLAGS {os.environ.get('XLA_FLAGS', '')!r}")
    rows = run(config)
    print()
    print(format_table(rows))
    if config.json_out:
        print(f"\nWrote {config.json_out}")
    else:
        print()
        print(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
