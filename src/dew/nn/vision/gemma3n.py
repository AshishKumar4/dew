"""Gemma 3n's MobileNet-v5 vision tower and its hard/soft vision embedder,
with their checkpoint maps.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.unread import Stored
from dew.interop.weights import in_params, placed
from dew.nn.blocks import TokenEmbedding
from dew.nn.dense import Dense
from dew.nn.norms import RMSNorm
from dew.nn.precision import float32_dot_general
from dew.registry import Record, from_record

from ..mobilenet import _ARCHITECTURE, MobileNetV5, MobileNetV5Encoder
from .common import ProjectorBase, ProjectorLayout, TowerBase, TowerGeometry, TowerLayout, _vision_section

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


@dataclasses.dataclass(frozen=True)
class Gemma3nVision(MobileNetV5, TowerBase):
    """Gemma 3n's vision tower: MobileNet-v5's encoder, by its construction
    fields (`MobileNetV5`)."""

    def build(self) -> nn.Module:
        return MobileNetV5Encoder(MobileNetV5(**dataclasses.asdict(self)))

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(channels=self.in_chans)


class Gemma3nProjector(ProjectorBase):
    """Gemma3nMultimodalEmbedder's vision hard and soft token paths."""

    vision_width: int
    text_width: int
    vocab_size: int = 128
    vocab_offset: int = 262144
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if min(self.vision_width, self.text_width, self.vocab_size) < 1 or self.vocab_offset < 0:
            raise ValueError(
                "vision/text widths and vocab_size must be positive; vocab_offset is nonnegative"
            )
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        self.embedding = TokenEmbedding(self.vocab_size, self.vision_width, dtype=self.dtype,
                                  name="embedding")
        norm = functools.partial(RMSNorm, epsilon=self.norm_eps, dtype=self.dtype)
        self.hard_embedding_norm = norm(name="hard_embedding_norm")
        self.soft_embedding_norm = norm(name="soft_embedding_norm")
        self.embedding_projection = Dense(self.text_width, use_bias=False,
                                              dtype=self.dtype, precision=self.precision,
                                              name="embedding_projection", dot_general=float32_dot_general)
        self.embedding_post_projection_norm = norm(with_scale=False,
                                                   name="embedding_post_projection_norm")

    def __call__(self, features):
        features = jnp.asarray(features)
        return self.soft_embeddings(features * jnp.asarray(self.vision_width ** 0.5, features.dtype))

    def soft_embeddings(self, features):
        """The reference embedder over soft features, without vision-only scaling."""
        features = jnp.asarray(features)
        if features.ndim != 3 or features.shape[-1] != self.vision_width:
            raise ValueError(f"soft features must be [B, N, {self.vision_width}], got {features.shape}")
        if not jnp.issubdtype(features.dtype, jnp.floating):
            raise ValueError("soft features must be floating point")
        if self.is_initializing():
            # Both paths belong to one checkpoint even when init starts with
            # image features. Linen creates an embedding only when called.
            self.hard_embedding_norm(self.embedding(jnp.zeros((1, 1), jnp.int32)))
        return self.embedding_post_projection_norm(
            self.embedding_projection(self.soft_embedding_norm(features)))

    def embed_hard(self, ids):
        """Embed vocabulary ids in [vocab_offset, vocab_offset + vocab_size).

        Pure: the host processor validates ids before device work.
        """
        embedded = self.embedding(ids - self.vocab_offset)
        if self.is_initializing():
            self.soft_embedding_norm(jnp.zeros_like(embedded))
        return self.embedding_post_projection_norm(
            self.embedding_projection(self.hard_embedding_norm(embedded)))

    def merge_hard_embeddings(self, token_embeddings, ids):
        """Replace this vocabulary range's slots with hard embeddings, through the
        reference's dummy id for every other slot."""
        mask = (ids >= self.vocab_offset) & (ids < self.vocab_offset + self.vocab_size)
        chosen = jnp.where(mask, ids, self.vocab_offset + self.vocab_size - 1)
        hard = self.embed_hard(chosen).astype(token_embeddings.dtype)
        return jnp.where(mask[..., None], hard, token_embeddings)


def gemma3n_vision_path(hf_name: str) -> tuple[str, ...]:
    """A timm MobileNet-v5 weight into the corresponding Linen module."""
    bare = hf_name.removeprefix("timm_model.")
    parts = tuple(bare.split("."))
    prefix: tuple[str, ...] = ()
    tails: set[tuple[str, ...]]
    if parts[:1] == ("blocks",) and len(parts) >= 5 and parts[1].isdigit() and parts[2].isdigit():
        stage, index = int(parts[1]), int(parts[2])
        if stage >= len(_ARCHITECTURE) or index >= len(_ARCHITECTURE[stage]):
            raise ValueError(f"unknown tensor name {hf_name!r}")
        spec = _ARCHITECTURE[stage][index]
        prefix = (f"stages_{stage}", f"blocks_{index}")
        parts = parts[3:]
        if spec.kind == "edge":
            tails = {(name, "weight") for name in ("conv_exp", "conv_pwl", "bn1", "bn2")}
        elif spec.kind == "inverted":
            modules = ["pw_exp", "pw_proj"]
            if spec.start_kernel:
                modules.append("dw_start")
            if spec.middle_kernel:
                modules.append("dw_mid")
            tails = {(name, child, "weight") for name in modules for child in ("conv", "bn")}
            tails.add(("layer_scale", "gamma"))
        else:
            tails = {("norm", "weight"), ("layer_scale", "gamma")}
            tails.update(("attn", name, "proj", "weight") for name in ("query", "key", "value", "output"))
            if spec.kv_stride > 1:
                tails.update(("attn", name, child, "weight") for name in ("key", "value")
                             for child in ("down_conv", "norm"))
    elif parts[:1] == ("conv_stem",):
        tails = {("conv", "weight"), ("conv", "bias"), ("bn", "weight")}
        prefix, parts = ("conv_stem",), parts[1:]
    elif parts[:1] == ("msfa",):
        tails = {("norm", "weight")}
        tails.update(("ffn", name, child, "weight") for name in ("pw_exp", "pw_proj")
                     for child in ("conv", "bn"))
        prefix, parts = ("msfa",), parts[1:]
    else:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    if len(parts) < 2 or parts not in tails:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    if parts[-1] != "weight":
        return prefix + parts
    norm = parts[-2] in ("bn", "bn1", "bn2", "norm")
    return prefix + parts[:-1] + ("scale" if norm else "kernel",)


GEMMA3N_PROJECTOR_TENSORS = {
    "embedding.weight": ("embedding", "embedding"),
    "hard_embedding_norm.weight": ("hard_embedding_norm", "scale"),
    "soft_embedding_norm.weight": ("soft_embedding_norm", "scale"),
    "embedding_projection.weight": ("embedding_projection", "kernel"),
}
"""The Gemma 3n embedder's tensors past `embed_vision.` (or its audio's), each to its leaf."""


def translate_gemma3n_projector_weights(
    hf_tensors: Mapping[str, Stored], *, param_dtype: str = "float32"
) -> tuple[Mapping[str, object], tuple[WeightLayout, ...]]:
    """A Gemma 3n vision embedder's parameter tree, with the layouts that write it back."""
    paths = GEMMA3N_PROJECTOR_TENSORS
    if set(hf_tensors) != set(paths):
        raise ValueError(f"vision embedder tensors differ: missing {sorted(set(paths) - set(hf_tensors))}, "
                         f"unknown {sorted(set(hf_tensors) - set(paths))}")
    return placed(hf_tensors, paths.__getitem__, param_dtype)


def _gemma3n_vision_record(
        hf_config: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """The vision embedder's fields and the encoder's model_args, the whole
    record validated before either component consumes it."""
    vision = _vision_section(hf_config)
    if vision.get("model_type", "gemma3n_vision") != "gemma3n_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not gemma3n_vision")
    used = {"model_type", "architecture", "hidden_size", "do_pooling", "model_args",
            "vocab_size", "vocab_offset", "rms_norm_eps"}
    # These are serialized HF metadata. Vocabulary and RMS fields above feed
    # the vision embedder; construction fields feed the MobileNet encoder.
    metadata = _records.HF_METADATA | {"initializer_range", "label_names", "num_classes"}
    unknown = (set(vision) - used - metadata
               - {key for key in vision if str(key).startswith("_")})
    if unknown:
        raise ValueError(f"vision_config fields {sorted(unknown)} have no counterpart")
    if vision.get("architecture", "mobilenetv5_300m_enc") != "mobilenetv5_300m_enc":
        raise ValueError(f"architecture {vision.get('architecture')!r} is not the MobileNet-v5 encoder")
    embedder = {
        "vision_width": _records.integer(vision.get("hidden_size", 2048), "hidden_size"),
        "vocab_size": _records.integer(vision.get("vocab_size", 128), "vocab_size"),
        "vocab_offset": _records.integer(vision.get("vocab_offset", 262144), "vocab_offset"),
        "norm_eps": _records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
    }
    if embedder["vision_width"] != 2048:
        raise ValueError("hidden_size must be 2048; timm's MobileNet-v5 encoder fixes its adapter width")
    if vision.get("do_pooling", False):
        raise ValueError("do_pooling=True requests a classifier head the encoder does not have")
    options = vision.get("model_args")
    options = _records.record({} if options is None else options, "model_args")
    allowed = {field.name for field in dataclasses.fields(Gemma3nVision)}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"MobileNet-v5 model_args {sorted(unknown)} are not supported")
    return embedder, options


def translate_gemma3n_vision_config(hf_config: Mapping[str, object]) -> Record:
    _, options = _gemma3n_vision_record(hf_config)
    value: Gemma3nVision = from_record(Gemma3nVision, options)
    return {"class": "gemma3n", "fields": {**dataclasses.asdict(value)}}


def translate_gemma3n_projector_config(hf_config: Mapping[str, object],
                                       text_width: int) -> Record:
    embedder, _ = _gemma3n_vision_record(hf_config)
    return {"class": "gemma3n", "fields": NativeFields(Gemma3nProjector,
                                                       {**embedder, "text_width": text_width})}


Gemma3nVision.checkpoint = TowerLayout("vision_tower.", in_params(gemma3n_vision_path))
Gemma3nProjector.checkpoint = ProjectorLayout("embed_vision.", GEMMA3N_PROJECTOR_TENSORS,
                                              read=translate_gemma3n_projector_weights)
