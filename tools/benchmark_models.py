"""Build Dew objectives, trainers and fixed host batches for benchmark cases."""

import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from benchmark_cases import (
    TEXT_FEATURES,
    TEXT_TOKENS,
    Case,
    _count,
    canvas_split,
    images_per_row,
    media_pixels,
)
from jax.sharding import Mesh

import dew.nn.backbones
import dew.nn.backbones.jepa  # noqa: F401  (registers the kind)
from dew.data import DataPartition, preferences
from dew.data.chat import ROLES_KEY, Role
from dew.diffusion import presets
from dew.diffusion.discrete import MDLM
from dew.diffusion.process import DenoisingCondition
from dew.inputs import CharTable, Condition, Field, InputSpec
from dew.inputs.encoders import ConditionEncoder
from dew.nn.backbones.flux import FluxTransformer
from dew.nn.backbones.flux2 import Flux2Transformer
from dew.nn.backbones.qwen_image import QwenImageTransformer
from dew.nn.backbones.sd3 import SD3Transformer
from dew.nn.backbones.unet_condition import UNet2DCondition
from dew.nn.backbones.wan import WanTransformer
from dew.nn.backbones.z_image import ZImageTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import ModelInputs
from dew.nn.multimodal import MultimodalTransformer, VisionConditioner
from dew.nn.vision import ProjectorBase, TowerBase
from dew.objectives.base import Objective, Variables
from dew.objectives.diffusion import BlockDiffusionObjective, DiffusionObjective
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.jepa import JepaObjective, MultiBlockMask
from dew.objectives.lm import LMObjective
from dew.objectives.rl import DPOObjective, GRPOObjective, sessions
from dew.registry import float64_twin, models, projectors, resolve_dtype, towers, with_precision
from dew.training import Layout, MeshSpec, Trainer

Batch = dict[str, np.ndarray | Mapping[str, np.ndarray] | ModelInputs]


class _DenoisingTextTable(ConditionEncoder[str]):
    """Synthetic token and pooled features for native diffusion models."""

    def __init__(self, table: CharTable, features: int, pooled_features: int | None,
                 guidance: float | None, masked: bool = False):
        self.table = table
        self.params = table.params
        self.features = features
        self.pooled_features = pooled_features
        self.guidance = guidance
        self.masked = masked
        """Whether the condition marks the real tokens, as Z-Image reads them."""

    @classmethod
    def from_pretrained(cls, checkpoint: str = "char_table", *, tokens: int = TEXT_TOKENS,
                        features: int = TEXT_FEATURES, pooled_features: int | None = None,
                        guidance: float | None = None, masked: bool = False, vocab: int = 130, seed: int = 0,
                        dtype=None):
        return cls(CharTable.from_pretrained(checkpoint, tokens=tokens,
                                            features=max(features, pooled_features or 0),
                                            vocab=vocab, seed=seed, dtype=dtype),
                   features, pooled_features, guidance, masked)

    def tokenize(self, data: Sequence[str]) -> Mapping[str, np.ndarray]:
        return self.table.tokenize(data)

    def encode(self, params: Variables, tokens) -> DenoisingCondition:
        text = self.table.encode(params, tokens)
        hidden = text.hidden
        pooled = None if self.pooled_features is None else hidden[:, 0, :self.pooled_features]
        guidance = None if self.guidance is None else jnp.full((hidden.shape[0],), self.guidance)
        return DenoisingCondition(hidden[..., :self.features], pooled, guidance=guidance,
                                  mask=jnp.asarray(text.mask, bool) if self.masked else None)

    def to_json(self) -> dict:
        return {**self.table.to_json(), "features": self.features, "pooled_features": self.pooled_features,
                "guidance": self.guidance, "masked": self.masked}


def mesh_spec(mesh: Mapping[str, int]) -> MeshSpec:
    """The `MeshSpec` a case's mesh record names, refusing a field it lacks."""
    fields = {f.name for f in dataclasses.fields(MeshSpec)}
    unknown = sorted(set(mesh) - fields)
    if unknown:
        raise ValueError(f"mesh has no field {unknown}; MeshSpec's fields are {sorted(fields)}")
    return MeshSpec(**mesh)


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
    return (family, towers.from_record(media["tower"]),
            projectors.from_record(media["projector"]), token)


def image_tokens(case: Case) -> int:
    """Text slots one image fills: the soft tokens this case's own projector
    emits for it.

    Read off the tower and projector instead of declared beside them, so a
    row cannot mark a width the modules do not produce. The trace is
    abstract: no parameter is allocated and no kernel compiles.
    """
    _, tower, projector, _ = media_values(case)
    conditioner = VisionConditioner(tower, projector)
    features = jax.eval_shape(
        lambda pixels: conditioner.init_with_output(
            jax.random.key(0), {"pixel_values": pixels})[0],
        jax.ShapeDtypeStruct((1, 1, *media_pixels(case)), np.float32))
    return features.shape[1]


def decoder_objective(
    case: Case, model,
) -> LMObjective | DPOObjective | GRPOObjective | MaskedDiffusionObjective:
    """What the case's decoder trains over its rows (`Case.decoder_objective`),
    with its own head chunking where it names one and its further objective
    keywords. Masked diffusion corrupts to the vocabulary's last id, which
    `global_batch` never draws."""
    chunks: dict[str, Any] = {} if case.head_chunks is None else {"head_chunks": case.head_chunks}
    keywords = {**chunks, **case.objective}
    match case.decoder_objective:
        case "lm":
            return LMObjective(model, case.seq_len, **keywords)
        case "sft":
            return LMObjective(model, case.seq_len, loss_role=Role.ASSISTANT, **keywords)
        case "dpo":
            return DPOObjective(model, case.seq_len, **keywords)
        case "grpo":
            return GRPOObjective(model, case.seq_len, **keywords)
        case "mdlm":
            vocab = _count(case.config, "vocab_size", case.architecture)
            return MaskedDiffusionObjective(model, MDLM(mask_id=vocab - 1)(), case.seq_len, **keywords)


def build_objective(case: Case, attention_impl: str = 'auto', *, widened: bool = False) -> Objective:
    """The objective a recipe would train for this case.

    The model goes through the same precision function the recipes use, so the
    dtype and the attention kernel land in the nested unet attention configs
    too, and a row of this table is a row a real run would produce. With
    `widened` every model is the float32 configuration's float64 twin
    (`dew.registry.float64_twin`), nested stages included, which computes in
    float64 throughout under x64: layout_parity's fp64 step.

    A composite takes built values rather than a flat record, so its trunk
    goes through the policy and the wrapper takes it, the way the pretrained
    loader assembles the same two models (dew.interop.pretrained and
    dew.interop.diffusion_gemma.build).
    """
    if widened:
        dtype = "float32"
    elif case.dtype is None:
        raise ValueError(f"{case.label} names no dtype; build_cases gives it the run's --dtype")
    else:
        dtype = case.dtype

    def built(architecture: str, config: Mapping[str, object]):
        fields = with_precision(architecture, config, dtype=dtype, attention_impl=attention_impl,
                                matmul_precision=case.matmul_precision)
        return models.build(architecture, **(float64_twin(fields) if widened else fields))

    sample_key = "video" if case.frames else "image"

    if case.canvas is not None:
        prompt, width, count = canvas_split(case)
        model = DiffusionGemma(text=built("causal_transformer", case.config),
                               canvas_length=width)
        objective = BlockDiffusionObjective(
            model, prompt_length=prompt, canvas_size=width, num_canvases=count)
    elif case.media is not None:
        family, tower, projector, token = media_values(case)
        # The decoder carries the attention kernel; the wrapper reads none.
        model = MultimodalTransformer(
            built("causal_transformer", case.config), tower, projector, family, token,
            dtype=jnp.float64 if widened else resolve_dtype(dtype))
        objective = decoder_objective(case, model)
    elif case.is_lm:
        objective = decoder_objective(case, built(case.architecture, case.config))
    elif case.predictor is not None:
        model = built(case.architecture, case.config)
        patch = case.config.get("patch_size", 16)
        if not isinstance(patch, int):
            raise ValueError(f"{case.architecture}'s patch_size is {patch!r}, not an int")
        grid = (case.image_size // patch, case.image_size // patch)
        objective = JepaObjective(
            model, built("jepa_predictor", {**case.predictor, "grid": grid}),
            MultiBlockMask.for_grid(grid, num_targets=2, scale=(0.2, 0.3)),
            sample=Field(sample_key, case.sample_shape))
    else:
        model = built(case.architecture, case.config)
        preset = presets.EDM(regime="pixel")
        if isinstance(model, QwenImageTransformer):
            keyword = "conditioning"
            encoder = _DenoisingTextTable.from_pretrained()
            preset = presets.Flow()
        elif isinstance(model, WanTransformer):
            keyword = "conditioning"
            encoder = _DenoisingTextTable.from_pretrained(features=model.text_dim)
            preset = presets.Flow()
        elif isinstance(model, ZImageTransformer):
            keyword = "conditioning"
            encoder = _DenoisingTextTable.from_pretrained(features=model.cap_feat_dim, masked=True)
            preset = presets.Flow()
        elif isinstance(model, Flux2Transformer):
            keyword = "conditioning"
            encoder = _DenoisingTextTable.from_pretrained(
                features=model.joint_attention_dim, guidance=3.5 if model.guidance_embeds else None)
            preset = presets.Flow()
        elif isinstance(model, (SD3Transformer, FluxTransformer)):
            keyword = "conditioning"
            encoder = _DenoisingTextTable.from_pretrained(
                features=model.joint_attention_dim, pooled_features=model.pooled_projection_dim,
                guidance=3.5 if isinstance(model, FluxTransformer) and model.guidance_embeds else None)
            preset = presets.Flow()
        elif isinstance(model, UNet2DCondition):
            keyword = "conditioning"
            encoder = _DenoisingTextTable.from_pretrained()
        else:
            keyword = "textcontext"
            encoder = CharTable.from_pretrained(tokens=TEXT_TOKENS, features=TEXT_FEATURES)
        inputs = InputSpec(Field(sample_key, case.sample_shape),
                           {keyword: Condition(encoder)})
        objective = DiffusionObjective(model, preset, inputs)
    return objective


def build_trainer(case: Case, attention_impl: str = 'auto',
                  optimizer: optax.GradientTransformation | None = None) -> Trainer:
    """The trainer a recipe would build for this case, minus the tracker and the
    checkpoints, on the case's device order."""
    trainer = Trainer(
        build_objective(case, attention_impl), optimizer or optax.adam(1e-4), key=jax.random.key(0),
        mesh=mesh_spec(case.mesh), layout=Layout(min_shard=case.fsdp_min_param_size),
        accumulation=case.accumulation, checkpoints=None, tracker=None)
    if case.device_order is not None:
        by_id = {device.id: device for device in jax.devices()}
        trainer.device_mesh = trainer.mesh.build([by_id[index] for index in case.device_order])
    return trainer


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


def global_batch(case: Case) -> Batch:
    """The case's whole batch, drawn the same on every process."""
    rng = np.random.default_rng(0)
    batch: Batch = {}
    if case.is_lm:
        vocab = case.config["vocab_size"]
        if not isinstance(vocab, int):
            raise ValueError(f"{case.architecture}'s vocab_size is {vocab!r}, not an int")
        width = case.seq_len + 1
        prompt = width // 2
        match case.decoder_objective:
            case "mdlm":
                # Rows of seq_len tokens, none of them the mask id, the
                # vocabulary's last.
                return {"text": rng.integers(0, vocab - 1, size=(case.batch_size, case.seq_len))
                        .astype(np.int32)}
            case "dpo":
                # Pairs that share their prompt; the completions after it count.
                pairs = rng.integers(0, vocab, size=(case.batch_size, 2, width)).astype(np.int32)
                pairs[:, 1, :prompt] = pairs[:, 0, :prompt]
                completion = np.zeros(pairs.shape, np.int32)
                completion[:, :, prompt:] = 1
                return {preferences.IDS_KEY: pairs, preferences.MASK_KEY: completion}
            case "grpo":
                # One rollout a row, a prompt and the response the mask counts:
                # each row's advantage on its response tokens, and the
                # sampler's likelihood near a fresh model's, so the ratios stay
                # inside the clip and every token moves the gradient.
                response = np.zeros((case.batch_size, width), np.float32)
                response[:, prompt:] = 1
                behavior = (rng.normal(-np.log(vocab), 0.05, size=response.shape) * response).astype(
                    np.float32)
                return {
                    sessions.IDS_KEY: rng.integers(0, vocab, size=response.shape).astype(np.int32),
                    sessions.SEGMENT_IDS_KEY: np.ones(response.shape, np.int32),
                    sessions.POSITIONS_KEY: np.tile(np.arange(width, dtype=np.int32), (case.batch_size, 1)),
                    sessions.RESPONSE_MASK_KEY: response,
                    sessions.ADVANTAGES_KEY: (rng.normal(size=(case.batch_size, 1)) * response).astype(
                        np.float32),
                    sessions.BEHAVIOR_LOG_PROBS_KEY: behavior,
                }
        # A canvas row's target masks are read off the pad id, so a drawn zero
        # would move the objective's own target support with the seed. Every
        # other row takes it as an ordinary token.
        lowest = 1 if case.canvas is not None else 0
        tokens = rng.integers(lowest, vocab, size=(case.batch_size, width)).astype(np.int32)
        batch["text"] = tokens if case.media is None else media_row(case, tokens, rng)
        if case.decoder_objective == "sft":
            # A user's turn, then the assistant's, which alone the loss counts.
            roles = np.full((case.batch_size, width), Role.USER, np.int8)
            roles[:, prompt:] = Role.ASSISTANT
            batch[ROLES_KEY] = roles
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
            batch["text"] = CharTable.from_pretrained(tokens=TEXT_TOKENS).tokenize(
                ["a flower"] * case.batch_size)
    return batch


def batches(case: Case, mesh: Mesh) -> Iterator[Batch]:
    """This process's share of one host batch, reused: the loader is
    benchmarked by benchmark_data.py. Every process draws the same global
    batch and keeps the rows of the share `DataPartition.of` names, as many
    as a loader would read."""
    partition = DataPartition.of(mesh)
    rows = partition.rows(case.batch_size)
    start = partition.index * rows
    mine = jax.tree.map(lambda leaf: leaf[start:start + rows], global_batch(case))
    while True:
        yield mine


