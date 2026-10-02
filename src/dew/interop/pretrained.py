"""Load native models and their host processors from a Hugging Face source.

`load_pretrained` is the front door: it reads a source directory or repo,
translates its config and weights through `dew.interop.hf_decoders`, and
returns a `Pretrained` holding the model, its variables and its processor.
`Pretrained` also carries the source's own decoding controls and the layouts
that write every tensor back, so `Pretrained.save` restores what it read.
"""

from __future__ import annotations

import functools
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, TypedDict, TypeGuard, Unpack

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import linen as nn

from dew import records
from dew._model_types import _QWEN35_TEXT_TYPES, _QWEN35_TYPES
from dew.artifacts import agreed
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import Origin, SourceSchedule
from dew.inference import BlockGeneration, MaskedGeneration, TextGeneration
from dew.inference.pipeline import place
from dew.inputs import Condition, Field, InputSpec
from dew.inputs.diffusion import (
    Composition,
    DiffusionConditioner,
    HiddenStatesConditioner,
    QwenImageConditioner,
    T5Segment,
)
from dew.interop import gguf, hf_decoders as decoders, mamba2, sources, verify
from dew.interop.codecs import SourceQuantization, source_quantization
from dew.interop.generation_config import (
    audit_masked,
    eos_ids,
    generation_limit,
    pad_id,
    return_sequences,
    source_decoding,
)
from dew.interop.safetensors_io import MAX_SHARD_SIZE, LazyTensors
from dew.interop.streaming import SourceLeaf
from dew.nn import audio as audio_nn
from dew.nn.autoencoders import AutoEncoder
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import Media, ModelInputs, pad_token_rows
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.text_encoders import ParamTree
from dew.objectives.base import Variables
from dew.records import JSON
from dew.registry import (
    dtype_name,
    models,
    precision_fields,
    projectors,
    resolve_dtype,
    towers,
    with_precision,
)
from dew.sampling.guidance import CFG
from dew.sampling.pipelines import TextToImage
from dew.sampling.text import Sampling

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from dew.lora import LoRA
    from dew.objectives.lm import LMObjective
    from dew.training.distributed import Layout, MeshSpec


class ProcessorCall(TypedDict, total=False):
    """Names every keyword dew hands a host processor beside `images`.

    A source processor takes far more than these; these are the ones dew
    passes, so the bag names them rather than standing for any keyword at
    all. `tests/test_interop.py` reads a real processor through this call.
    """

    text: str | list[str]
    padding: bool
    truncation: bool
    return_tensors: str | None
    audio: Media
    videos: Media
    video_metadata: Sequence[Mapping[str, object]]


class HostProcessor(Protocol):
    """Declares the HF processor operations kept outside compiled model computation."""

    def __call__(self, *, images: Media | None = None,
                 **kwargs: Unpack[ProcessorCall]) -> Mapping[str, object]: ...
    # The files it wrote are its own bookkeeping; dew calls this for the effect.
    def save_pretrained(self, save_directory: str) -> None: ...
    def apply_chat_template(self, conversation: Sequence[Mapping[str, object]],
                            **kwargs: JSON) -> str | Sequence[int] | Mapping[str, object]: ...
    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]: ...


def _hosts(reference: PreTrainedTokenizerBase) -> TypeGuard[HostProcessor]:
    """Whether `reference` offers every HostProcessor operation, checked by
    name because a tokenizer's own annotations are narrower than what dew
    passes (see `data.chat._token_ids`)."""
    return all(callable(getattr(reference, name, None))
               for name in ("__call__", "save_pretrained", "apply_chat_template", "batch_decode"))


def _patch_streams(values: Mapping[str, object], image_id: int, video_id: int | None, *,
                   kernel: int, table_size: int, patch_size: int
                   ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return each modality's patch frames, their coordinates and their pooled
    lengths, by the placeholder id that stands in the prompt for them.

    The source flattens its video and frame axes, not its prompt rows
    (modeling_gemma4.py:2476-2488), so the two streams stay separate here and
    the caller's placeholder runs establish row order over them. A stream
    arrives whole: its pixels with its positions, its coordinates inside the
    position table, and its patches in complete pooling blocks.
    """
    streams: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pixel_name, position_name, token_id, ndim in (
        ("pixel_values", "image_position_ids", image_id, 3),
        ("pixel_values_videos", "video_position_ids", video_id, 4),
    ):
        if pixel_name not in values and position_name not in values:
            continue
        if pixel_name not in values or position_name not in values:
            raise ValueError(f"{pixel_name} and {position_name} arrive together")
        if token_id is None:
            raise ValueError(f"{pixel_name} requires an integer placeholder id")
        if token_id in streams:
            raise ValueError("image_token_id and video_token_id must identify separate payload streams")
        frames, coordinates = np.asarray(values[pixel_name]), np.asarray(values[position_name])
        if (frames.ndim != ndim or not np.issubdtype(frames.dtype, np.floating)
                or min(frames.shape[1:]) < 1):
            raise ValueError(f"{pixel_name} must contain floating patch frames of rank {ndim}")
        if (coordinates.shape != (*frames.shape[:-1], 2)
                or not np.issubdtype(coordinates.dtype, np.integer)):
            raise ValueError(f"{position_name} must contain aligned integer patch coordinates")
        if frames.shape[-1] != 3 * patch_size ** 2:
            raise ValueError(f"{pixel_name} patch width disagrees with vision_config.patch_size")
        frames = frames.reshape(-1, *frames.shape[-2:])
        coordinates = coordinates.reshape(-1, *coordinates.shape[-2:])
        valid_patches = (coordinates >= 0).all(axis=-1)
        padding = (coordinates == -1).all(axis=-1)
        if not np.all(valid_patches | padding):
            raise ValueError(f"{position_name} uses nonnegative coordinates or paired (-1, -1) padding")
        if np.any(coordinates >= table_size):
            raise ValueError(f"{position_name} exceeds position_embedding_size")
        counts = valid_patches.sum(axis=1)
        if frames.shape[1] % kernel ** 2 or np.any(counts % kernel ** 2):
            raise ValueError("patch counts must divide into complete pooling blocks")
        streams[token_id] = (frames, coordinates, counts // kernel ** 2)
    return streams


def _row_padding(reference: HostProcessor) -> tuple[int, Literal["left", "right"]]:
    """Return the id and the side to pad text rows with, at the boundary a host
    processor draws: it delegates text to the tokenizer it wraps, a tokenizer
    is its own, and either may state neither field, so both names are read off
    the object here and narrowed once for the rows.
    """
    tokenizer = getattr(reference, "tokenizer", reference)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    side = getattr(tokenizer, "padding_side", "right")
    if side not in ("left", "right"):
        raise ValueError("the tokenizer padding_side must be left or right")
    return (0 if pad_id is None else records.integer(pad_id, "pad_token_id"),
            "left" if side == "left" else "right")


def _row_start(reference: HostProcessor) -> int | None:
    """Return the id the tokenizer starts a sequence with, or None, read at the
    boundary `_row_padding` reads its padding at: off the tokenizer a
    processor wraps, or the tokenizer itself."""
    tokenizer = getattr(reference, "tokenizer", reference)
    bos = getattr(tokenizer, "bos_token_id", None)
    return None if bos is None else records.integer(bos, "bos_token_id")


@dataclass(frozen=True)
class Processor:
    """Runs host text and image preprocessing, then normalizes the numeric layout.

    The checkpoint processor owns resizing, normalization and special-token
    expansion. Dew organizes its outputs into row-aligned arrays; it does
    not reproduce the checkpoint's image preprocessing algorithms.
    """

    reference: HostProcessor
    config: Mapping[str, object]
    record: Mapping[str, object]
    vocab_size: int

    def __call__(self, text: str | Sequence[str], *, images: Media | None = None,
                 audio: Media | None = None, videos: Media | None = None,
                 video_metadata: Sequence[Mapping[str, object]] | None = None) -> ModelInputs:
        if video_metadata is not None and videos is None:
            raise ValueError("video_metadata requires videos")
        if images is None and audio is None and videos is None:
            rows = [text] if isinstance(text, str) else list(text)
            values = self.reference(text=rows, padding=False, truncation=False, return_tensors=None)
            pad_id, side = _row_padding(self.reference)
            ids = values["input_ids"]
            ids = ids if isinstance(ids, Sequence) else np.asarray(ids)
            fields = {name: value if isinstance(value, Sequence) else np.asarray(value)
                      for name, value in values.items() if name != "input_ids"}
            tokens, fields = pad_token_rows(ids, pad_id=pad_id, padding_side=side, fields=fields)
            return self.from_hf({"input_ids": tokens, **fields})
        # truncation is off for text anyway; reloaded Gemma processors forward
        # the tokenizer's unset max_length into audio kwargs otherwise.
        arguments: ProcessorCall = {
            "text": text if isinstance(text, str) else list(text),
            "padding": not isinstance(text, str) and len(text) > 1, "truncation": False, "return_tensors": "pt"}
        if audio is not None:
            arguments["audio"] = audio
        if videos is not None:
            arguments["videos"] = videos
        if video_metadata is not None:
            arguments["video_metadata"] = video_metadata
        # An audio-only processor takes no images keyword at all, so the
        # absent one is left out of the call rather than passed as None.
        if images is None:
            return self._from_tensors(self.reference(**arguments))
        return self._from_tensors(self.reference(images=images, **arguments))

    def chat(self, messages: Sequence[Mapping[str, object]], *, add_generation_prompt: bool = True,
             **template_options: JSON) -> ModelInputs:
        """Run the source's actual chat template and processor into numeric inputs.

        Template controls such as reasoning_effort and preserve_thinking are
        interpreted by the checkpoint template. Media-bearing content uses
        the same reference processor and numeric normalization as plain text.
        """
        values = self.reference.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=add_generation_prompt, **template_options)
        if not isinstance(values, Mapping):
            raise TypeError("the source chat processor must return named numeric inputs")
        return self._from_tensors(values)

    def _from_tensors(self, values: Mapping[str, object]) -> ModelInputs:
        import torch

        arrays: dict[str, np.ndarray] = {}
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"processor field {name!r} is not a tensor")
            # Llama 4's processor emits bf16; float32 widening is exact.
            arrays[name] = (value.float() if value.dtype == torch.bfloat16 else value).numpy()
        return self.from_hf(arrays)

    def from_hf(self, values: Mapping[str, object]) -> ModelInputs:
        """Validate and normalize actual processor outputs before device use."""
        known = {"input_ids", "attention_mask", "pixel_values", "token_type_ids", "mm_token_type_ids",
                 "image_position_ids", "image_grid_thw", "input_features", "input_features_mask",
                 "pixel_values_videos", "video_grid_thw", "video_position_ids"}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"processor fields {sorted(unknown)} have no native model input")
        tokens = np.asarray(values["input_ids"])
        if tokens.ndim != 2 or not np.issubdtype(tokens.dtype, np.integer) or min(tokens.shape) < 1:
            raise ValueError("input_ids must be nonempty integer [B, S] rows")
        valid = np.asarray(values.get("attention_mask", np.ones(tokens.shape, bool)), dtype=bool)
        if valid.shape != tokens.shape:
            raise ValueError("attention_mask must align with input_ids")
        positions = np.maximum(np.cumsum(valid, axis=1) - 1, 0).astype(np.int32)
        # An unpadded batch carries no validity field. The host knows the rows
        # are whole here; the model would have to take an all-true mask on
        # trust and build one anyway.
        token_fields = {"positions": jnp.asarray(positions)}
        if not valid.all():
            token_fields["attention_mask"] = jnp.asarray(valid)
        gemma = self.config.get("model_type") == "gemma4"
        if gemma:
            for pixels, coordinates in (("pixel_values", "image_position_ids"),
                                         ("pixel_values_videos", "video_position_ids")):
                if (pixels in values) != (coordinates in values):
                    raise ValueError(f"{pixels} and {coordinates} arrive together")
            if "mm_token_type_ids" in values:
                types = np.asarray(values["mm_token_type_ids"])
                if types.shape != tokens.shape or not np.issubdtype(types.dtype, np.integer):
                    raise ValueError("mm_token_type_ids must be integer values aligned with input_ids")
                # Image-to-video adjacency stays one vision run; hard text
                # breaks it (modeling_gemma4.py:2148-2157).
                vision = (types == 1) | (types == 2)
                previous = np.concatenate([np.zeros((tokens.shape[0], 1), bool), vision[:, :-1]], axis=1)
                groups = np.cumsum(vision & ~previous, axis=1, dtype=np.int32) - 1
                token_fields["image_groups"] = jnp.asarray(np.where(vision, groups, -1))
        elif "video_position_ids" in values:
            raise ValueError("video_position_ids require the Gemma4 visual tower")
        conditioning: dict[str, jax.Array] = {}
        if "pixel_values" in values or "pixel_values_videos" in values:
            if ("pixel_values_videos" in values
                    and self.config.get("model_type") not in (*_QWEN35_TYPES, "gemma4")):
                raise ValueError("video patch inputs require a Qwen3.5 or Gemma4 visual tower")
            image_fields, conditioning = self._images(values, tokens)
            token_fields.update(image_fields)
            if self.config.get("model_type") in _QWEN35_TYPES:
                token_fields["rotary_positions"] = self._image_rotary_positions(
                    tokens, valid, image_fields["image_groups"], conditioning["image_grid_thw"])
        if ("input_features" in values) != ("input_features_mask" in values):
            raise ValueError("input_features and input_features_mask arrive together")
        if "input_features" in values:
            audio_fields, audio_conditioning = self._audio(values, tokens)
            token_fields.update(audio_fields)
            conditioning = {**conditioning, **audio_conditioning}
        if np.any(tokens < 0) or np.any(tokens >= self.vocab_size):
            raise ValueError("input_ids must lie in the text vocabulary, including any hard media ranges")
        prepared = ModelInputs(jnp.asarray(tokens, jnp.int32), token_fields, conditioning)
        prepared.validate()
        return prepared

    def _images(self, values: Mapping[str, object], tokens: np.ndarray
                ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """Row-align the processor's image and video features to their placeholders.

        It runs in two halves. The first reads the processor's output into
        `chunks`, one feature block per image, with `lengths` its placeholder
        count, `shape` the padded block shape and `capacity` the slots one
        image occupies; `grid` and `patch_positions` carry the per-family
        extras. The second copies each chunk into `padded` at its row and image
        slot and writes `indices`, the slot each placeholder token reads.

        The three branches are the three source layouts: Qwen's packed patch
        runs, a per-patch position stream, and a fixed tokens-per-image grid.
        """
        image_id = self.record.get("image_token_id", self.config.get("image_token_id"))
        if type(image_id) is not int:
            raise ValueError("image_token_id must be an integer")
        qwen = self.config.get("model_type") in _QWEN35_TYPES
        gemma = self.config.get("model_type") == "gemma4"
        video_id = (self.config.get("video_token_id", 258884) if gemma else
                    self.config.get("video_token_id") if qwen else None)
        if video_id is not None and type(video_id) is not int:
            raise ValueError("video_token_id must be an integer")
        pixels = np.asarray(values.get("pixel_values", values.get("pixel_values_videos")))
        if pixels.ndim not in (2, 3, 4) or not np.issubdtype(pixels.dtype, np.floating):
            raise ValueError("pixel_values must contain floating image tensors or patch vectors")
        pixel_dtype = pixels.dtype
        runs = []
        for row in tokens:
            locations = np.flatnonzero((row == image_id) | (row == video_id if video_id is not None else False))
            boundaries = np.flatnonzero((np.diff(locations) != 1) | (row[locations[1:]] != row[locations[:-1]])) + 1
            runs.append([] if not len(locations) else np.split(locations, boundaries))
        image_counts = np.array([len(row) for row in runs], np.int32)
        patch_positions = None
        grid = None
        merge = 1
        if qwen:
            expected_types = np.where(tokens == image_id, 1, np.where(tokens == video_id, 2, 0))
            supplied_types = np.asarray(values.get("mm_token_type_ids"))
            if not np.array_equal(supplied_types, expected_types):
                raise ValueError("mm_token_type_ids must identify each image and video placeholder")
            chunks, grid, merge = self._qwen_frames(values, tokens, runs)
            shape = (max(chunk.shape[0] for chunk in chunks), chunks[0].shape[-1])
            lengths = np.prod(grid, axis=1) // merge ** 2
            capacity = shape[0] // merge ** 2
        elif "image_position_ids" in values or "video_position_ids" in values:
            chunks, patch_positions, lengths, kernel = self._positioned_chunks(values, tokens, runs,
                                                                               image_id, video_id)
            shape = (max(chunk.shape[0] for chunk in chunks), chunks[0].shape[-1])
            pixel_dtype = np.result_type(*(chunk.dtype for chunk in chunks))
            capacity = shape[0] // kernel ** 2
        else:
            count = self.record.get("tokens_per_image")
            if type(count) is not int or count < 1 or pixels.ndim != 4:
                raise ValueError("fixed-resolution pixel_values require a positive tokens_per_image")
            lengths = np.full(pixels.shape[0], count, np.int32)
            capacity = count
            chunks, shape = list(pixels), pixels.shape[1:]
        if len(chunks) != int(image_counts.sum()):
            raise ValueError("pixel_values and image placeholder counts disagree")
        width = int(image_counts.max())
        if width < 1:
            raise ValueError("pixels require image placeholders")
        padded = np.zeros((tokens.shape[0], width, *shape), pixel_dtype)
        padded_grid = None if grid is None else np.zeros((tokens.shape[0], width, 3), np.int32)
        if padded_grid is not None:
            padded_grid[..., 1:] = merge
        padded_positions = (None if patch_positions is None else np.full(
            (tokens.shape[0], width, shape[0], 2), -1, np.int32))
        indices = np.full(tokens.shape, -1, np.int32)
        groups = None if gemma else np.full(tokens.shape, -1, np.int32)
        token_lengths = np.zeros((tokens.shape[0], width), np.int32)
        offset = 0
        for row, blocks in enumerate(runs):
            for image, slots in enumerate(blocks):
                length = int(lengths[offset])
                if len(slots) != length:
                    raise ValueError("image placeholder counts do not match the projector's feature count")
                chunk = chunks[offset]
                padded[row, image, :chunk.shape[0]] = chunk
                if padded_grid is not None and grid is not None:
                    padded_grid[row, image] = grid[offset]
                if padded_positions is not None and patch_positions is not None:
                    positions = patch_positions[offset]
                    padded_positions[row, image, :positions.shape[0]] = positions
                token_lengths[row, image] = length
                indices[row, slots] = image * capacity + np.arange(length)
                if groups is not None:
                    groups[row, slots] = image
                offset += 1
        conditioning = {"pixel_values": jnp.asarray(padded), "image_lengths": jnp.asarray(image_counts),
                        "image_token_lengths": jnp.asarray(token_lengths)}
        if padded_positions is not None:
            conditioning["image_position_ids"] = jnp.asarray(padded_positions)
        if padded_grid is not None:
            conditioning["image_grid_thw"] = jnp.asarray(padded_grid)
        fields = {"image_indices": jnp.asarray(indices)}
        if groups is not None:
            fields["image_groups"] = jnp.asarray(groups)
        return fields, conditioning

    def _positioned_chunks(self, values: Mapping[str, object], tokens: np.ndarray, runs: list[list[np.ndarray]],
                           image_id: int, video_id: int | None
                           ) -> tuple[list[np.ndarray], list[np.ndarray], list[int], int]:
        """The per-patch position layout's feature blocks, in placeholder order:
        each image or video's patches, their positions and placeholder count,
        and the vision pooling kernel that sets how many slots one fills."""
        vision = self.config.get("vision_config")
        if not isinstance(vision, Mapping):
            raise ValueError("patch position ids require a vision config")
        kernel = vision.get("pooling_kernel_size")
        if type(kernel) is not int or kernel < 1:
            raise ValueError("pooling_kernel_size must be a positive integer")
        table_size, patch_size = vision.get("position_embedding_size"), vision.get("patch_size")
        if type(table_size) is not int or table_size < 1 or type(patch_size) is not int or patch_size < 1:
            raise ValueError("position_embedding_size and patch_size must be positive integers")
        streams = _patch_streams(values, image_id, video_id, kernel=kernel,
                                 table_size=table_size, patch_size=patch_size)
        offsets = dict.fromkeys(streams, 0)
        chunks, patch_positions, lengths = [], [], []
        for row, blocks in enumerate(runs):
            for slots in blocks:
                token_id = int(tokens[row, slots[0]])
                stream = streams.get(token_id)
                offset = offsets.get(token_id, 0)
                if stream is None or offset >= len(stream[0]):
                    raise ValueError("pixel_values and image/video placeholder counts disagree")
                chunks.append(stream[0][offset])
                patch_positions.append(stream[1][offset])
                lengths.append(int(stream[2][offset]))
                offsets[token_id] += 1
        if any(offsets[token_id] != len(stream[0]) for token_id, stream in streams.items()):
            raise ValueError("pixel_values and image/video placeholder counts disagree")
        if not chunks:
            raise ValueError("pixels require image or video placeholders")
        return chunks, patch_positions, lengths, kernel

    def _audio(self, values: Mapping[str, object], tokens: np.ndarray
               ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """Row-align the processor's clip features and index their placeholders.

        Each contiguous run of the audio placeholder is one clip, in the
        processor's clip order. Gemma 4 inserts one placeholder per encoded
        valid frame; Gemma 3n inserts its fixed slot count. Both are checked
        against the encoder's frame stride here, before any device work.
        """
        audio_id = self.record.get("audio_token_id")
        audio = self.record.get("audio")
        if type(audio_id) is not int or not isinstance(audio, Mapping):
            raise ValueError("this source has no audio tower")
        encoder = towers.from_record(audio)
        if not isinstance(encoder, (audio_nn.Gemma3nAudio, audio_nn.Gemma4Audio)):
            raise ValueError("audio conditioning requires a Gemma audio encoder")
        features = np.asarray(values["input_features"])
        mask = np.asarray(values["input_features_mask"])
        if features.ndim != 3 or not np.issubdtype(features.dtype, np.floating):
            raise ValueError("input_features must be floating [clips, frames, mel]")
        if mask.shape != features.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("input_features_mask must be bool [clips, frames], True for valid frames")
        encoded = mask[:, ::audio_nn.encoded_frame_stride(encoder)]
        slots = self.record.get("audio_soft_tokens")
        if slots is None:
            expected = encoded.sum(axis=1)
            capacity = encoded.shape[1]
        else:
            if type(slots) is not int or encoded.shape[1] > slots:
                raise ValueError(f"{encoded.shape[1]} encoded frames exceed the {slots} audio slots per clip")
            expected = np.full(features.shape[0], slots)
            capacity = slots
        runs = []
        for row in tokens:
            locations = np.flatnonzero(row == audio_id)
            runs.append([] if not len(locations) else np.split(locations, np.flatnonzero(np.diff(locations) != 1) + 1))
        counts = np.array([len(row) for row in runs], np.int32)
        if int(counts.sum()) != features.shape[0]:
            raise ValueError("input_features and audio placeholder runs disagree")
        width = int(counts.max())
        padded = np.zeros((tokens.shape[0], width, *features.shape[1:]), features.dtype)
        padded_mask = np.zeros((tokens.shape[0], width, features.shape[1]), bool)
        indices = np.full(tokens.shape, -1, np.int32)
        offset = 0
        for row, clips in enumerate(runs):
            for clip, run in enumerate(clips):
                if len(run) != int(expected[offset]):
                    raise ValueError("audio placeholder counts do not match the encoded frame counts")
                padded[row, clip] = features[offset]
                padded_mask[row, clip] = mask[offset]
                indices[row, run] = clip * capacity + np.arange(len(run))
                offset += 1
        conditioning = {"input_features": jnp.asarray(padded), "input_features_mask": jnp.asarray(padded_mask),
                        "audio_lengths": jnp.asarray(counts)}
        return {"audio_indices": jnp.asarray(indices)}, conditioning

    def _qwen_frames(self, values: Mapping[str, object], tokens: np.ndarray, runs):
        """Return views of packed image/video patches in text order, one item per frame.

        Qwen3.5 get_rope_index repeats video grids by their temporal count and
        resets each frame's temporal coordinate to zero. The processor places
        timestamps between frames. Attention never crosses a frame, so these
        views share the existing visual tower and merger without new weights.
        """
        vision = self.config.get("vision_config")
        if not isinstance(vision, Mapping):
            raise ValueError("packed visual inputs require a vision config")
        merge = vision.get("spatial_merge_size")
        if type(merge) is not int or merge < 1:
            raise ValueError("spatial_merge_size must be a positive integer")
        streams = {}
        for pixel_name, grid_name, token_name, split in (
                ("pixel_values", "image_grid_thw", "image_token_id", False),
                ("pixel_values_videos", "video_grid_thw", "video_token_id", True)):
            if (pixel_name in values) != (grid_name in values):
                raise ValueError(f"{pixel_name} and {grid_name} arrive together")
            if pixel_name not in values:
                continue
            pixels, grid = np.asarray(values[pixel_name]), np.asarray(values[grid_name])
            if (grid.ndim != 2 or grid.shape[1] != 3 or not np.issubdtype(grid.dtype, np.integer)
                    or np.any(grid <= 0) or np.any(grid[:, 1:] % merge)):
                raise ValueError(f"{grid_name} requires positive integer grids tiled by spatial_merge_size")
            if pixels.ndim != 2 or not np.issubdtype(pixels.dtype, np.floating):
                raise ValueError(f"{pixel_name} must be floating packed patch vectors")
            if int(np.prod(grid, axis=1).sum()) != pixels.shape[0]:
                raise ValueError(f"{pixel_name} does not match {grid_name}")
            pieces = []
            offset = 0
            for time, height, width in grid:
                count = int(height * width)
                for _ in range(int(time) if split else 1):
                    frames = 1 if split else int(time)
                    pieces.append((pixels[offset:offset + frames * count],
                                   (frames, int(height), int(width))))
                    offset += frames * count
            token_id = self.config[token_name]
            if type(token_id) is not int:
                raise ValueError(f"{token_name} must be an integer")
            streams[token_id] = pieces
        offsets = dict.fromkeys(streams, 0)
        ordered = []
        for row, spans in enumerate(runs):
            for span in spans:
                token = int(tokens[row, span[0]])
                if token not in streams or offsets[token] == len(streams[token]):
                    raise ValueError("visual placeholders exceed their image/video frame payloads")
                ordered.append(streams[token][offsets[token]])
                offsets[token] += 1
        if not ordered or any(offsets[token] != len(stream) for token, stream in streams.items()):
            raise ValueError("visual payloads and placeholder frame counts disagree")
        chunks, grids = zip(*ordered, strict=True)
        widths = {chunk.shape[-1] for chunk in chunks}
        if len(widths) != 1:
            raise ValueError("image and video patch widths must agree")
        return list(chunks), np.asarray(grids, np.int32), merge

    def _image_rotary_positions(self, tokens: np.ndarray, valid: np.ndarray,
                                groups: jax.Array, grids: jax.Array) -> jax.Array:
        """Compute Qwen3.5's get_rope_index on host-normalized image grids.

        Text advances one coordinate per token. An image occupies its merged
        temporal/height/width grid, and following text starts after its longest
        spatial side. Padding keeps coordinate zero, as in the reference.
        """
        vision = records.record(self.config["vision_config"], "vision_config")
        merge = records.integer(vision["spatial_merge_size"], "spatial_merge_size")
        image_groups, grid_values = np.asarray(groups), np.asarray(grids)
        prepared = np.zeros((*tokens.shape, 3), np.int32)
        for row in range(tokens.shape[0]):
            slots = np.flatnonzero(valid[row])
            start = cursor = 0
            while start < len(slots):
                group = int(image_groups[row, slots[start]])
                stop = start + 1
                while stop < len(slots) and image_groups[row, slots[stop]] == group:
                    stop += 1
                span = slots[start:stop]
                if group < 0:
                    prepared[row, span] = (cursor + np.arange(len(span)))[:, None]
                    cursor += len(span)
                else:
                    time, height, width = (int(value) for value in grid_values[row, group])
                    coordinates = np.indices((time, height // merge, width // merge)).reshape(3, -1).T
                    if len(span) != len(coordinates):
                        raise ValueError("image tokens do not match their multimodal rotary grid")
                    prepared[row, span] = coordinates + cursor
                    cursor += max(height, width) // merge
                start = stop
        return jnp.asarray(prepared)

    def decode(self, tokens: jax.typing.ArrayLike) -> list[str]:
        """Decode token rows with the tokenizer retained by the source processor."""
        array = np.asarray(tokens)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("decode expects integer [B, S] token rows")
        return self.reference.batch_decode(array.tolist(), skip_special_tokens=True)

    @property
    def bos_id(self) -> int | None:
        """The id the source's tokenizer starts a sequence with, or None."""
        return _row_start(self.reference)

    def save_pretrained(self, directory: str | Path) -> None:
        """Save the same processor and tokenizer used by this source."""
        self.reference.save_pretrained(str(directory))


@dataclass(frozen=True)
class WeightLayout:
    """Holds an existing source tensor's location and reversible storage layout.

    `expert_index` is the expert a per-expert source tensor holds. The
    loader stacks those tensors onto an expert dimension
    (`hf_decoders._stack_experts`), so one stacked leaf answers for every
    expert of a layer and the index says which slice this tensor is.

    `dtype` is the width the source stores this tensor in where that is
    not its leaf's: DeepSeek V4's token-to-expert table is int64 on disk
    and int32 in the collection, and the export writes back what the
    checkpoint held.

    `padded` is the length a 1-D source tensor stores past its leaf's, as
    zeros, for the names its family declares (`DecoderFamily.zero_padded`):
    Kimi K3 ships each KDA layer's `A_log` for 96 heads padded to 128
    entries. The family's prepare step checks and trims the tail, and export
    writes the zeros back.
    """

    name: str
    paths: tuple[tuple[str, ...], ...]
    shape: tuple[int, ...]
    transpose: tuple[int, ...] | None = None
    concatenate: int | None = None
    expert_index: int | None = None
    dtype: np.dtype | None = None
    padded: int | None = None

    def _leaf(self, variables: Mapping[str, object], path: tuple[str, ...],
              scalar_mode: str | None) -> np.ndarray | jax.Array:
        if path[-1] == "layer_scalar":
            if scalar_mode not in ("frozen", "trainable"):
                raise ValueError("layer_scalar export requires an explicit model mode")
            path = (("constants" if scalar_mode == "frozen" else "params"), *path[1:])
        node: object = variables
        for part in path:
            if not isinstance(node, Mapping):
                raise ValueError(f"parameter path {path} does not traverse a mapping")
            node = node[part]
        if not isinstance(node, (np.ndarray, jax.Array)):
            raise ValueError(f"{self.name} reads {path}, which holds {type(node).__name__} rather than an array")
        return node

    def stored_dtype(self, variables: Mapping[str, object], scalar_mode: str | None = None) -> np.dtype:
        """The dtype `export` writes, read from the leaf without copying it."""
        if self.dtype is not None:
            return np.dtype(self.dtype)
        return np.dtype(self._leaf(variables, self.paths[0], scalar_mode).dtype)

    def export(self, variables: Mapping[str, object], scalar_mode: str | None = None) -> np.ndarray:
        leaves = []
        for path in self.paths:
            node = self._leaf(variables, path, scalar_mode)
            if self.expert_index is not None:
                # Slice the expert where the leaf lives. One stacked leaf
                # answers for E source tensors, so copying it to the host
                # per tensor would move the whole stack E times.
                if node.ndim == 0 or not 0 <= self.expert_index < node.shape[0]:
                    raise ValueError(
                        f"{self.name} is expert {self.expert_index} of {path}, which "
                        f"holds {node.shape}")
                node = node[self.expert_index]
            leaves.append(np.asarray(node))
        value = leaves[0] if self.concatenate is None else np.concatenate(leaves, axis=self.concatenate)
        if self.transpose is not None:
            value = value.transpose(self.transpose)
        if self.padded is not None:
            value = np.pad(np.asarray(value), (0, self.padded - value.shape[0]))
        if value.size != math.prod(self.shape):
            raise ValueError(
                f"{self.name} assembles {value.shape} from {self.paths}, which does not "
                f"fill the source's {self.shape}")
        value = np.ascontiguousarray(value).reshape(self.shape)
        return value if self.dtype is None else value.astype(self.dtype)

    def restore(self, tensor: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Return the leaf of `shape` whose export is `tensor`.

        The inverse of `export` for a layout that binds one whole leaf; a
        tensor assembled from several leaves has no single leaf to restore.
        """
        if len(self.paths) != 1 or self.concatenate is not None or self.expert_index is not None:
            raise ValueError(f"{self.name} is assembled from several leaves, so no one leaf restores it")
        if tensor.shape != self.shape:
            raise ValueError(f"{self.name} stores {self.shape}, not {tensor.shape}")
        transpose = self.transpose or tuple(range(len(shape)))
        stored = tensor.reshape(tuple(shape[axis] for axis in transpose))
        return np.ascontiguousarray(stored.transpose(sorted(range(len(shape)), key=transpose.__getitem__)))


def _stacked_expert(path: tuple[str, ...]) -> tuple[tuple[str, ...], int | None]:
    """Map a per-expert leaf path to the stacked leaf the loaded tree holds.

    A checkpoint that names one tensor per expert maps through the family
    to `experts/K/projection/kernel`, a path `hf_decoders._stack_experts`
    consumed on the way in: the tree keeps one `experts/projection/kernel`
    stacked in expert order, so that leaf and K are where the tensor's
    values live.
    """
    if (len(path) >= 4 and path[-4] == "experts" and path[-3].isdigit()
            and path[-1] == "kernel"):
        return (*path[:-3], path[-2], path[-1]), int(path[-3])
    return path, None


def _leading_axes(variables: Mapping[str, object], path: tuple[str, ...],
                  expert_index: int | None) -> int:
    """Return how many axes a bound leaf carries ahead of its stored matrix.

    A kernel stores `[in, out]` where its source stores `[out, in]`, and a
    grouped projection's leaf keeps one such matrix per group: DeepSeek
    V4's `[groups, in, rank]` is stored `[groups * rank, in]`, the group
    axis folded into the rows (modeling_deepseek_v4.py:294-323). The
    transpose is the same swap of the trailing pair under those axes, so
    the leaf says how many there are. A per-expert source tensor is one
    slice of its stacked leaf, which is taken before the transpose.
    """
    node: object = variables
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            raise ValueError(f"the loaded tree holds no {path}, which {part!r} names")
        node = node[part]
    if not isinstance(node, np.ndarray | jax.Array | SourceLeaf):
        return 0
    rank = node.ndim - (0 if expert_index is None else 1)
    return max(rank - 2, 0)


def _language_layout(name: str, text_name: str, tensor: np.ndarray,
                     config, model_type: str, variables: Mapping[str, object],
                     component: str | None = None) -> WeightLayout | None:
    """Return the text family's leaf map plus its inverse storage operations."""
    family = decoders.families()[model_type]
    # A family whose checkpoint packs its experts as `[E, out, in]`
    # (`_gemma4_prepare` swaps them into dew's `[E, in, out]`) writes them
    # back swapped.
    from dew.interop.families.gemma import _gemma4_prepare

    packed = family.prepare_weights is _gemma4_prepare

    def nested(path: tuple[str, ...]) -> tuple[str, ...]:
        return path if component is None else (path[0], component, *path[1:])

    transpose = None
    concatenate = None
    expert_index = None
    head_name, embedding_name = family.tied_head_names
    if text_name == head_name and config["tie_embeddings"]:
        # The tied head has no leaf of its own, so its source name binds to
        # the embedding it copies, under whatever name that family stores.
        embedding = family.weight_path(embedding_name, config)
        if embedding is None:
            raise ValueError(f"{embedding_name!r} has no parameter path to tie {name!r} to")
        paths = (nested(embedding),)
    elif text_name.endswith(".experts.gate_up_proj") and (packed or model_type == "llama4_text"):
        names = [text_name.removesuffix("gate_up_proj") + projection for projection in ("gate_proj", "up_proj")]
        paths_list = []
        for key in names:
            path = family.weight_path(key, config)
            if path is None:
                raise ValueError(f"fused expert tensor {name!r} has no parameter path")
            paths_list.append(nested(path))
        paths = tuple(paths_list)
        concatenate = -1
        if packed:
            transpose = (0, 2, 1)
    else:
        path = family.weight_path(text_name, config)
        if path is None:
            return None
        path, expert_index = _stacked_expert(path)
        paths = (nested(path),)
        if path[-1] == "kernel" and tensor.ndim == 2:
            lead = _leading_axes(variables, paths[0], expert_index)
            transpose = (*range(lead), lead + 1, lead)
        elif text_name.endswith(".experts.down_proj") and packed:
            transpose = (0, 2, 1)
    # A weight is fp32 in the tree whatever the checkpoint stored it as, so
    # only an index table's own width has to be carried back.
    stored = None if np.issubdtype(tensor.dtype, np.floating) else tensor.dtype
    padded = tensor.shape[0] if text_name.endswith(family.zero_padded) else None
    return WeightLayout(name, paths, tensor.shape, transpose, concatenate,
                        expert_index, stored, padded)


def _wrapper_layouts(tensors, record, variables):
    """Return one `WeightLayout` per wrapper source tensor.

    Each layout keeps the source's own tensor name and points at the leaf path
    the loader built, so an export writes the names the source shipped.
    """
    from dew.nn import vision

    tower_kind = record["tower"]["kind"]
    audio_encoder = None if record["audio"] is None else towers.from_record(record["audio"])
    bindings = []
    retained = {}
    for name, tensor in tensors.items():
        group, local = decoders._wrapper_route(name, record)
        if group == "language_model":
            layout = _language_layout(name, local, tensor, record["text"], record["text_model_type"],
                                      variables, "language_model")
            if layout is None:
                retained[name] = tensor
            else:
                bindings.append(layout)
            continue
        paths: tuple[tuple[str, ...], ...] = ()
        transpose = None
        if group == "projector":
            path = vision.projector_weight_path(record["projector"]["kind"], local)
            paths = (("params", "projector", *path),)
            if path[-1] == "kernel" and local != "mm_input_projection_weight":
                transpose = (1, 0)
        elif group == "tower":
            path = vision.TOWER_PATHS[tower_kind](local)
            if path is not None:
                paths = ((path[0], "tower", *path[1:]),) if tower_kind == "gemma4" else (("params", "tower", *path),)
                if path[-1] == "kernel":
                    transpose = (1, 0) if tensor.ndim in (2, 5) else (3, 2, 0, 1)
        elif group == "audio_projector":
            path = vision.projector_weight_path(record["audio_projector"]["kind"], local)
            paths = (("params", "audio_projector", *path),)
            if path[-1] == "kernel":
                transpose = (1, 0)
        else:
            if not isinstance(audio_encoder, (audio_nn.Gemma3nAudio, audio_nn.Gemma4Audio)):
                raise ValueError("source export requires a Gemma audio encoder")
            path = audio_nn.audio_weight_path(local, audio_encoder)
            paths = ((path[0], "audio_tower", *path[1:]),)
            if path[-1] == "kernel":
                # Kernels store [*window, in, out]; the source keeps [out, in, *window].
                transpose = {2: (1, 0), 3: (2, 1, 0), 4: (3, 2, 0, 1)}[tensor.ndim]
        if paths:
            bindings.append(WeightLayout(name, paths, tensor.shape, transpose))
        else:
            # SigLIP's pooling head and reference-ignored auxiliary tensors
            # have no forward consumer; export preserves their source bytes.
            retained[name] = tensor
    return tuple(bindings), retained


def _share_quantized_aliases(tensors: dict[str, np.ndarray], aliases: tuple[tuple[str, str], ...],
                             quantized: tuple[str, ...]) -> None:
    """Share only aliases verified on original values before codec narrowing.

    A component may mix quantized and unquantized copies, or several MTP
    copies of a tied head. Fold the checked relationships transitively, then
    reuse the already decoded storage: no second cast or weight copy.
    """
    if not aliases:
        return
    links: dict[str, set[str]] = {}
    for left, right in aliases:
        links.setdefault(left, set()).add(right)
        links.setdefault(right, set()).add(left)
    remaining = set(links)
    while remaining:
        first = remaining.pop()
        group, pending = {first}, [first]
        while pending:
            fresh = links[pending.pop()] - group
            group.update(fresh)
            remaining.difference_update(fresh)
            pending.extend(fresh)
        representative = next((name for name in quantized if name in group), None)
        if representative is not None:
            value = tensors[representative]
            for name in group:
                tensors[name] = value


@dataclass(frozen=True)
class Pretrained:
    """Holds a native model, explicit variables and its checkpoint's host processor.

    `model_config` is the record the model was built from, in Dew's own
    vocabulary with the run's compute dtype and attention kernel, so a caller
    logs the model it ran.
    """

    model: nn.Module
    variables: Variables
    processor: Processor | None
    config: Mapping[str, object]
    source: Path
    model_config: Mapping[str, object]
    generation_config: Mapping[str, object] = field(default_factory=dict)
    weight_layouts: tuple[WeightLayout, ...] = ()
    retained_tensors: Mapping[str, np.ndarray] = field(default_factory=dict)
    export_adapter: Callable[[nn.Module, Mapping[str, object], Mapping[str, object]], Mapping[str, np.ndarray]] | None = field(default=None, repr=False)
    process: Process | None = None
    inputs: InputSpec | None = None
    autoencoder: AutoEncoder | None = None
    schedule: SourceSchedule | None = None
    task: SourceTask | None = None
    finish: Callable[[Mapping[str, object], jax.Array], jax.Array] | None = field(default=None, repr=False)
    quantized_tensors: tuple[str, ...] = ()
    quantized_scale_dtype: str | None = None
    """The dtype a quantized source stored its scales in, where its format
    leaves that to the checkpoint (DeepSeek-V4's `.scale`: float8_e8m0fnu,
    float32 in the Base releases), so `save` writes them back in it."""
    quantization_grid: Mapping[str, np.ndarray] = field(default_factory=dict, repr=False)
    """The scales and zeros an integer format (AWQ, GPTQ) encodes a saved
    weight against, as the source stored them."""
    revision: str | None = None
    """The Hub commit the source resolved to, whatever branch or tag was
    asked for; None for a local directory."""
    adapter: LoRA | None = None
    """The low-rank adapter `lora` put on the model, whose factors the
    variables hold; None for the source as published."""

    @property
    def layouts(self) -> Mapping[str, WeightLayout]:
        """Map each source module name to the layout of its weight: `model.<module>`
        for a decoder, `unet.<module>` for a pipeline component.

        These are the names a published adapter file writes, so this is what
        `dew.lora` binds a low-rank delta through.
        """
        return {layout.name.removesuffix(".weight").replace("/", "."): layout
                for layout in self.weight_layouts if layout.name.endswith(".weight")}

    def lora(self, *, rank: int, modules: Sequence[str], key: jax.Array, alpha: float | None = None,
             rslora: bool = False, dropout: float = 0.0) -> Pretrained:
        """Return this source with a fresh low-rank adapter on the projections `modules` name.

        The bundle that comes back holds the adapted model, the variables
        with the factors in them (B zero, so it computes what the source
        does) and the adapter, and nothing of a run: `lm_objective` trains
        the factors alone, `adapter.save` writes PEFT's directory and `save`
        the source's layout with the factors merged in. `dew.lora.LoRA.fresh`
        describes the arguments.
        """
        from dew.lora import LoRA

        if self.adapter is not None:
            raise ValueError("this bundle already carries an adapter; adapt the source it was made from")
        if self.schedule is not None:
            raise ValueError("a pipeline's adapter spans its components, each adapted where it runs; "
                             "build one with dew.lora.LoRA.fresh(source.model, source.variables, source.layouts, ...)")
        adapter, variables = LoRA.fresh(self.model, self.variables, self.layouts, rank=rank, modules=modules,
                                        key=key, alpha=alpha, rslora=rslora, dropout=dropout)
        return replace(self, model=adapter.adapt(self.model), variables=variables, adapter=adapter)

    def lm_objective(self, seq_len: int, **options) -> LMObjective:
        """Build next-token training from this source's model and variables.

        `options` are `LMObjective`'s training and evaluation controls. This
        bundle supplies `pretrained` itself and its processor unless one is
        passed, and an adapted bundle its adapter's filter as `trainable`,
        so the run moves the factors alone.
        """
        from dew.objectives.lm import LMObjective

        if "pretrained" in options:
            raise ValueError("a Pretrained bundle already supplies the initial variables; omit pretrained=")
        if self.adapter is not None:
            if "trainable" in options:
                raise ValueError("the adapter already selects what trains, its own factors; omit trainable=")
            options["trainable"] = self.adapter.trainable
        return LMObjective(self.model, seq_len, pretrained=self.variables,
                           **{"processor": self.processor, **options})

    def text_generation(self, *, sampling: Sampling | None = None) -> TextGeneration | MaskedGeneration:
        """Build the text generation task this source describes.

        Masked generation refines a full response with Unmask, not the source
        family's custom generation recipe. AR sampling overrides are refused.

        Without an override the task runs the source's policy (`task.sampling`
        holds every common control its config sets), any chain the rarer
        controls need, and the strategy its config names. An explicit
        `sampling` replaces the policy and clears that chain with it, because
        the chain was built around the policy the caller just replaced; the
        source's EOS and pad ids fill the ones it leaves None, and
        `num_return_sequences` still comes from the source.
        """
        if self.process is not None:
            raise TypeError("a latent diffusion source generates through text_to_image")
        if isinstance(self.model, DiffusionGemma):
            raise TypeError("a DiffusionGemma source generates through block_generation")
        if not isinstance(self.model, CausalTransformer | MultimodalTransformer):
            raise TypeError(
                f"text_generation decodes through a native Dew decoder's KV cache, and this "
                f"source loaded as {type(self.model).__name__}; generate with transformers' "
                f"AutoModelForCausalLM.from_pretrained({str(self.source)!r}).generate, or load a "
                f"registered family without fallback")
        decoder = self.model if isinstance(self.model, CausalTransformer) else None
        mask_id = None if decoder is None else decoder.mask_token_id
        if decoder is not None and not decoder.causal and mask_id is not None:
            from dew.diffusion.discrete import MDLM
            if sampling is not None:
                raise TypeError("native MDLM accepts denoising steps, not autoregressive sampling controls")
            audit_masked(self.config, self.generation_config)
            return MaskedGeneration(self.model, self.variables, MDLM(mask_id=mask_id)(), self.processor,
                eos_token_ids=eos_ids(self.config, self.generation_config),
                pad_token_id=pad_id(self.config, self.generation_config),
                max_new_tokens=generation_limit(self.config, self.generation_config, "max_new_tokens"),
                max_length=generation_limit(self.config, self.generation_config, "max_length"),
                n=return_sequences(self.config, self.generation_config))
        rows = return_sequences(self.config, self.generation_config)
        policy, logits, strategy = source_decoding(
            self.config, self.generation_config, self.model, rows, sampling)
        return TextGeneration(self.model, self.variables, self.processor, policy,
                              max_new_tokens=generation_limit(self.config, self.generation_config, "max_new_tokens"),
                              max_length=generation_limit(self.config, self.generation_config, "max_length"),
                              n=rows, logits=logits, strategy=strategy)

    def block_generation(self) -> BlockGeneration:
        """Build the DiffusionGemma as a canvas task, defaulting to the source's sampler config."""
        from dew.interop import diffusion_gemma
        if not isinstance(self.model, DiffusionGemma):
            raise TypeError("block generation needs a DiffusionGemma source")
        return BlockGeneration(self.model, self.variables,
                               diffusion_gemma.generation_process(self.config, self.generation_config),
                               self.processor, eos_ids(self.config, self.generation_config),
                               pad_id(self.config, self.generation_config),
                               max_new_tokens=generation_limit(self.config, self.generation_config, "max_new_tokens"),
                               max_length=generation_limit(self.config, self.generation_config, "max_length"),
                               n=return_sequences(self.config, self.generation_config))

    def text_to_image(self) -> TextToImage:
        """Build the latent diffusion source as an image task with its published policy."""
        if self.process is None or self.inputs is None or self.schedule is None:
            raise TypeError("text_to_image needs a latent diffusion source")
        if self.task is None:
            raise TypeError("text_to_image needs the source's own call policy")
        return TextToImage(self.model, self.process, self.inputs, self.variables, self.autoencoder,
                           grid=self.task.grid, final_denoise=False, sampler=self.schedule.solver(),
                           steps=self.task.steps, guidance=self.task.guidance, finish=self.finish)

    def export(self, variables: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
        """The tensors `save` writes, by their source names; `dew.inference.NCCLPush` sends these.

        A diffusion source writes one set per component, so it has none. An
        adapted bundle (`lora`) writes the source's own tensors with the
        factors merged into their kernels, PEFT's `merge_and_unload`, from
        its variables or a trainer's split of them; `adapter.save` writes
        the factors alone.
        """
        values = self.variables if variables is None else variables
        if self.adapter is not None:
            values = self.adapter.merge(values)
        quantization = self._quantization()
        if self.schedule is not None:
            raise ValueError("a diffusion source writes one tensor set per component; save it instead")
        family = self.config.get("model_type")
        if self.export_adapter is not None:
            tensors = self.export_adapter(self.model, values, self.config)
        elif (isinstance(self.model, CausalTransformer) and isinstance(family, str)
              and not decoders.families().get(family, decoders.families()[verify.CONVENTION]).preserve_source_layout
              and quantization is None):
            # The decoder export's own encoder, so this and `save_pretrained_decoder`
            # leave the same weights. A quantized source keeps its packed format
            # by going back over its source names, below.
            return decoders.export_decoder_weights(self.model, values, decoders._export_config(self.model))
        elif self.weight_layouts:
            # Source names and geometry first; the packed format goes back over them.
            text = self.model.language_model if isinstance(self.model, MultimodalTransformer) else self.model
            scalar_mode = text.layer_scalar if isinstance(text, CausalTransformer) else None
            layouts = {layout.name: layout for layout in self.weight_layouts}
            if quantization is None:
                # Each tensor is assembled when its shard is written (`save_sharded`).
                specs = {**{name: jax.ShapeDtypeStruct(np.shape(value), np.asarray(value).dtype)
                            for name, value in self.retained_tensors.items()},
                         **{name: jax.ShapeDtypeStruct(layout.shape, layout.stored_dtype(values, scalar_mode))
                            for name, layout in layouts.items()}}
                return LazyTensors(specs, lambda name: (layouts[name].export(values, scalar_mode) if name in layouts
                                                        else self.retained_tensors[name]))
            tensors = {**self.retained_tensors,
                       **{name: layout.export(values, scalar_mode) for name, layout in layouts.items()}}
        else:
            raise ValueError("this source has no reversible weight layout")
        if quantization is not None:
            tensors = quantization.requantize(tensors, self.quantized_tensors)
        return tensors

    def _quantization(self) -> SourceQuantization | None:
        """The config's quantization format, refused when the loader recorded no tensors to write back in it."""
        quantization = source_quantization(self.config, scale_dtype=self.quantized_scale_dtype,
                                           grid=self.quantization_grid)
        if quantization is not None and not self.quantized_tensors:
            raise ValueError(
                "this source's config declares a quantization_config and the loader recorded "
                "no quantized tensors to write back in it")
        return quantization

    def save(self, directory: str | Path, *, variables: Mapping[str, object] | None = None,
             max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Write `export`'s tensors, in shards of at most `max_shard_size`, the
        source's own config.json, as published, and its tokenizer assets."""
        from dew.interop.safetensors_io import save_hf_layout
        values = self.variables if variables is None else variables
        destination = Path(directory)
        if self.schedule is not None:
            from dew.interop import diffusion
            self._quantization()
            diffusion.save_source(self, values, destination)
            return
        save_hf_layout(self.export(values), dict(self.config), destination, max_shard_size)
        decoders.save_export_assets(destination, tokenizer=self.processor,
                                    generation_config=dict(self.generation_config))


def _native_variables(parts: Mapping[str, Mapping[str, ParamTree]]) -> dict[str, dict[str, ParamTree]]:
    collections: dict[str, dict[str, ParamTree]] = {}
    for component, variables in parts.items():
        for collection, tree in variables.items():
            collections.setdefault(collection, {})[component] = tree
    return collections


class _Call(NamedTuple):
    """Holds one pinned pipeline's own `__call__` policy, read from Diffusers
    0.34.0 (Qwen-Image 2.1 from 6256aa76): the family it belongs to, the
    steps and guidance scale it defaults to, whether that scale guides two
    branches or is the value the model embeds, and the text sequence budget
    it pads its T5 tower to. Qwen-Image's pipeline pads to the longest prompt
    of a call, so its budget is the prompt window Dew pads each row to."""

    family: Literal["sd", "sdxl", "sd3", "flux", "flux2", "qwen_image", "z_image"]
    steps: int
    guidance: float
    guided: bool
    sequence: int = 0


# The class a file declares is the one whose defaults it gets, so no class is
# normalized into another: the XL inpainting pipeline keeps 7.5 where the
# other two XL ones lowered to 5.0, every Flax pipeline kept its own 7.5, and
# Flux's 3.5 is the guidance its transformer embeds while its own true
# classifier-free guidance is off at the pinned default.
_PIPELINE_POLICY: Mapping[str, _Call] = MappingProxyType({
    "StableDiffusionPipeline": _Call("sd", 50, 7.5, guided=True),
    "StableDiffusionImg2ImgPipeline": _Call("sd", 50, 7.5, guided=True),
    "StableDiffusionInpaintPipeline": _Call("sd", 50, 7.5, guided=True),
    "StableDiffusionXLPipeline": _Call("sdxl", 50, 5.0, guided=True),
    "StableDiffusionXLImg2ImgPipeline": _Call("sdxl", 50, 5.0, guided=True),
    "StableDiffusionXLInpaintPipeline": _Call("sdxl", 50, 7.5, guided=True),
    "StableDiffusion3Pipeline": _Call("sd3", 28, 7.0, guided=True, sequence=256),
    "FluxPipeline": _Call("flux", 28, 3.5, guided=False, sequence=512),
    # `true_cfg_scale` defaults to 1.0: the release samples unguided.
    "QwenImage21Pipeline": _Call("qwen_image", 40, 1.0, guided=True, sequence=512),
    # FLUX.2 [dev] embeds its 4.0; [klein] guides two branches at 4.0 unless
    # its index marks it step-distilled, which `_call_policy` reads.
    "Flux2Pipeline": _Call("flux2", 50, 4.0, guided=False, sequence=512),
    "Flux2KleinPipeline": _Call("flux2", 50, 4.0, guided=True, sequence=512),
    # Z-Image guides as `pos + 5.0 (pos - neg)`, which is Dew's
    # `neg + 6.0 (pos - neg)`.
    "ZImagePipeline": _Call("z_image", 50, 6.0, guided=True, sequence=512),
    "FlaxStableDiffusionPipeline": _Call("sd", 50, 7.5, guided=True),
    "FlaxStableDiffusionImg2ImgPipeline": _Call("sd", 50, 7.5, guided=True),
    "FlaxStableDiffusionInpaintPipeline": _Call("sd", 50, 7.5, guided=True),
    "FlaxStableDiffusionXLPipeline": _Call("sdxl", 50, 7.5, guided=True),
})


def _call_policy(index: Mapping[str, object], denoiser: _Denoiser) -> _Call:
    """Return the call policy this file's own pipeline carries.

    A directory that declares no pipeline - a bare component tree - takes its
    family's reference pipeline. A directory that declares one Dew does not
    implement is refused rather than run under another pipeline's defaults,
    and a declared pipeline of another family is refused too: a component
    this loader reads does not qualify a workflow it does not.
    """
    published = index.get("_class_name")
    expected = _PIPELINE_POLICY[denoiser.pipeline]
    if published is None:
        return expected
    found = _PIPELINE_POLICY.get(published) if isinstance(published, str) else None
    if found is None:
        raise ValueError(f"Native diffusion does not implement the published pipeline "
                         f"{published!r}")
    if found.family != expected.family:
        raise ValueError(f"The declared pipeline {published!r} is a {found.family} pipeline, and "
                         f"this directory's denoiser belongs to {denoiser.pipeline!r}")
    if published == "Flux2KleinPipeline" and records.boolean(index.get("is_distilled", False), "is_distilled"):
        # A step-distilled [klein] ignores its guidance scale.
        return found._replace(guided=False)
    return found


@dataclass(frozen=True)
class SourceTask:
    """Holds a published pipeline's own call policy.

    `steps` and `guidance` are the defaults its `__call__` signature carries,
    and `grid` prepares the sampling grid the way that pipeline prepares it,
    with the sigma origin it uses and the latent geometry it lays out already
    bound. A pipeline whose guidance is a model input rather than two branches
    carries `guidance=None`.
    """

    steps: int
    guidance: CFG | None
    grid: Callable[[int], tuple[Process, jax.Array]]


@dataclass(frozen=True)
class _TextTowers:
    """The CLIP towers a UNet, SD3 or Flux denoiser reads, the T5 tower where
    its family has one, and the composition `DiffusionConditioner` builds
    from them. `embeds_guidance` marks a transformer that takes the guidance
    scale as a model input rather than as two guided branches."""

    composition: Composition
    towers: tuple[str, ...]
    t5_tower: str | None = None
    embeds_guidance: bool = False

    def components(self, index: Mapping[str, object]) -> tuple[str, ...]:
        """The text components this directory holds, which a conditioner load fetches."""
        return tuple(name for name in (*self.towers, self.t5_tower)
                     if name is not None and _present(index, name))

    def build(self, directory: Path, index: Mapping[str, object], denoiser: _Denoiser,
              policy: _Call, compute, size: int, *, param_dtype: str,
              attention_impl: str = "auto", params: Variables | None = None
              ) -> tuple[DiffusionConditioner, tuple[WeightLayout, ...], dict[str, Mapping[str, object]]]:
        """Construct the published text composition from metadata and either weight source."""
        names = tuple(name for name in self.towers if _present(index, name))
        if not names:
            raise ValueError("A latent diffusion source needs at least one text encoder")
        towers, tokenizers, text_params, layouts = _clip_towers(
            directory, names, compute, param_dtype=param_dtype, params=params)
        components: dict[str, Mapping[str, object]] = {
            name: _component_config(directory, name) for name in names}
        t5 = None
        if self.t5_tower is not None and _present(index, self.t5_tower):
            t5, t5_params, t5_layouts, components[self.t5_tower] = _t5_tower(
                directory, compute, self.t5_tower, policy.sequence, param_dtype=param_dtype,
                params=None if params is None else params[self.t5_tower])
            if params is None:
                text_params = {**text_params, self.t5_tower: t5_params}
            layouts += t5_layouts
        height, width = index.get("dew_height", size), index.get("dew_width", size)
        if type(height) is not int or type(width) is not int or height < 1 or width < 1:
            raise ValueError("Image geometry must contain positive integer dimensions")
        encoder = DiffusionConditioner(
            towers, tokenizers, names, text_params, str(directory), height, width,
            denoiser.context_width, composition=self.composition, t5=t5,
            guidance=policy.guidance if self.embeds_guidance and not policy.guided else None,
            aesthetics=bool(index.get("requires_aesthetics_score", False)), param_dtype=param_dtype)
        return encoder, layouts, components

    def unconditional(self, index: Mapping[str, object]) -> dict:
        """The empty-prompt row a file's own pipeline guides against: the XL
        pipelines zero it where their index says so, and the SD3 pipeline
        encodes it with its towers, having no such control."""
        zero = self.composition == "clip_pooled" and bool(index.get("force_zeros_for_empty_prompt", True))
        return {"text": "", "negative": True, "zero": zero}


@dataclass(frozen=True)
class _QwenImageText:
    """Qwen-Image's Qwen3-VL text encoder, which `QwenImageConditioner` runs
    over its pipeline's chat template, padded to the call's token budget."""

    def build(self, directory: Path, index: Mapping[str, object], denoiser: _Denoiser,
              policy: _Call, compute, size: int, *, param_dtype: str,
              attention_impl: str = "auto", params: Variables | None = None
              ) -> tuple[QwenImageConditioner, tuple[WeightLayout, ...], dict[str, Mapping[str, object]]]:
        """Construct `QwenImageConditioner` at the pipeline's prompt budget."""
        return _qwen_image_conditioning(directory, index, compute, size, tokens=policy.sequence,
                                        param_dtype=param_dtype, attention_impl=attention_impl,
                                        params=params)

    def unconditional(self, index: Mapping[str, object]) -> dict:
        """The empty prompt, encoded through the same template."""
        return {"text": ""}


@dataclass(frozen=True)
class _HiddenStatesText:
    """The text encoder FLUX.2 (`pipeline="flux2"`: Mistral-3 for [dev],
    Qwen3 for [klein]) or Z-Image (`"z_image"`: Qwen3) reads hidden states
    from, which `HiddenStatesConditioner` runs, padded to the call's token
    budget. `embeds_guidance` marks FLUX.2 [dev]'s transformer, which reads
    the guidance scale as an input."""

    pipeline: Literal["flux2", "z_image"]
    embeds_guidance: bool = False

    def build(self, directory: Path, index: Mapping[str, object], denoiser: _Denoiser,
              policy: _Call, compute, size: int, *, param_dtype: str,
              attention_impl: str = "auto", params: Variables | None = None
              ) -> tuple[HiddenStatesConditioner, tuple[WeightLayout, ...], dict[str, Mapping[str, object]]]:
        """Construct `HiddenStatesConditioner` at the pipeline's prompt budget."""
        return _hidden_states_conditioning(
            directory, index, compute, size, pipeline=self.pipeline, tokens=policy.sequence,
            guidance=policy.guidance if self.embeds_guidance else None, param_dtype=param_dtype,
            attention_impl=attention_impl, params=params)

    def unconditional(self, index: Mapping[str, object]) -> dict:
        """The empty prompt a guided call encodes as its negative."""
        return {"text": ""}


@dataclass(frozen=True)
class _Denoiser:
    """Holds what one architecture contributes to a diffusion source.

    Model construction and conditioning conventions use metadata only.
    The weight reader is invoked only by a complete source load; a restored
    conditioner can reuse the same architecture metadata without reading
    denoiser or autoencoder weights. `text` is the family's text
    conditioning: which components it reads, how it builds its encoder and
    the unconditional row its pipeline guides against.
    """

    component: str
    model: nn.Module
    weights: Callable[[str], tuple[Variables, tuple[WeightLayout, ...]]]
    built: Mapping[str, object]
    config: Mapping[str, object]
    text: _TextTowers | _QwenImageText | _HiddenStatesText
    patch: int
    latent_input: int
    sample_size: int
    context_width: int
    pipeline: str
    origin: Origin = "scheduler"


def _load_diffusion_source(directory: Path, index: Mapping[str, object], *, dtype: str,
                           attention_impl: str, param_dtype: str = "float32",
                           variables: Variables | None = None) -> Pretrained:
    """Read a published latent diffusion directory into native modules and variables.

    Two denoiser families ship this layout: a UNet reading one or two CLIP
    towers through cross attention, and an MM-DiT transformer reading them
    jointly beside a T5 tower. The directory's own denoiser component selects
    the family, and everything the families share - the autoencoder, the text
    towers, the geometry, the conditioning, the safety head a file declares,
    the schedule and the call policy - is read once here.

    Supplied `variables` are a saved tree in this layout, bound as they are:
    every module is built from the directory's metadata and no weight file is
    read, so the directory needs only its configs and tokenizers.
    """
    compute = resolve_dtype(dtype)
    denoiser = _denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    if variables is None:
        denoiser_variables, denoiser_layouts = denoiser.weights(param_dtype)
    else:
        denoiser_variables = {name: value for name, value in variables.items()
                              if name not in ("encoders", "autoencoder")}
        denoiser_layouts = ()
    held = {} if variables is None else variables["encoders"]
    policy = _call_policy(index, denoiser)
    autoencoder, vae_params, vae_layouts, vae_config = _diffusion_vae(
        directory, compute, param_dtype=param_dtype,
        params=None if variables is None else variables["autoencoder"])
    encoder, text_layouts, components = denoiser.text.build(
        directory, index, denoiser, policy, compute,
        denoiser.sample_size * autoencoder.downscale_factor, param_dtype=param_dtype,
        attention_impl=attention_impl, params=held.get("conditioning"))
    components.update({denoiser.component: denoiser.config, "vae": vae_config})
    height, width = encoder.height, encoder.width
    inpaint = denoiser.latent_input == autoencoder.latent_channels * 2 + 1
    inputs = InputSpec(Field("image", (height, width, records.integer(vae_config["in_channels"], "in_channels"))),
                       {encoder.keyword: Condition(encoder, unconditional=denoiser.text.unconditional(index))},
                       mask=Field("mask", (height, width, 1)) if inpaint else None)
    encoders: dict[str, object] = {encoder.keyword: encoder.params}
    finish, safety_layouts = None, ()
    if _present(index, "safety_checker"):
        finish, encoders["safety"], safety_layouts, safety_configs = _image_safety(
            directory, compute, param_dtype=param_dtype, params=held.get("safety"))
        components.update(safety_configs)
    schedule = SourceSchedule.from_config(_component_config(directory, "scheduler"))
    components["scheduler"] = dict(schedule.config)
    patch = denoiser.patch * autoencoder.downscale_factor
    tokens = (height // patch) * (width // patch)
    task = SourceTask(min(policy.steps, schedule.train_steps),
                      CFG(policy.guidance) if policy.guided and policy.guidance > 1 else None,
                      functools.partial(schedule.sampling, origin=denoiser.origin, tokens=tokens))
    variables = {**denoiser_variables, "encoders": encoders, "autoencoder": vae_params}
    config = {"model_index": {**index, "dew_height": height, "dew_width": width}, **components}
    return Pretrained(denoiser.model, variables, None, config, directory, denoiser.built,
                      weight_layouts=denoiser_layouts + vae_layouts + text_layouts + safety_layouts,
                      process=schedule.training_process(tokens), inputs=inputs, autoencoder=autoencoder,
                      schedule=schedule, finish=finish, task=task)


def load_diffusion_source(checkpoint: str, *, dtype: str = "bfloat16", param_dtype: str = "float32",
                          revision: str | None = None, attention_impl: str = "auto",
                          size: tuple[int, int] | None = None,
                          variables: Variables | None = None) -> Pretrained:
    """A published diffusion pipeline, to train from its own weights.

    `size` is the (height, width) in pixels the pipeline runs at instead of
    its own: the geometry its conditioning, its training shift and its
    sampling grid are bound to. Supplied `variables` are a saved tree of the
    same pipeline, as a run that fine-tuned it wrote them: the modules are
    built from the directory's metadata and bind those variables, and no
    weight downloads.
    """
    directory = sources.snapshot(checkpoint, revision, weights=False)
    if not (directory / "model_index.json").is_file():
        raise ValueError(f"{checkpoint} is not a diffusion pipeline: it has no model_index.json")
    with open(directory / "model_index.json") as handle:
        index = json.load(handle)
    if variables is None:
        # Both fetches at the commit the metadata resolved to.
        directory = sources.snapshot(checkpoint, directory.name, weights=tuple(
            name for name in index if _present(index, name)))
    if size is not None:
        index = {**index, "dew_height": size[0], "dew_width": size[1]}
    loaded = _load_diffusion_source(directory, index, dtype=dtype, attention_impl=attention_impl,
                                    param_dtype=param_dtype, variables=variables)
    return replace(loaded, revision=None if os.path.isdir(checkpoint) else directory.name)


def load_diffusion_conditioner(checkpoint: str, *, dtype: str | None = "bfloat16",
                               param_dtype: str = "float32", revision: str | None = None,
                               attention_impl: str = "auto", params: Variables | None = None
                               ) -> DiffusionConditioner:
    """Load conditioning weights, or bind supplied parameters using metadata only."""
    from dew.nn.autoencoders import AutoencoderKL

    compute = resolve_dtype(dtype)
    resolve_dtype(param_dtype)
    directory = sources.snapshot(checkpoint, revision, weights=False)
    with open(directory / "model_index.json") as handle:
        index = json.load(handle)
    denoiser = _denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    if not isinstance(denoiser.text, _TextTowers):
        raise ValueError(f"{checkpoint} conditions through its Qwen3-VL encoder; "
                         "build it with QwenImageConditioner.from_pretrained")
    if params is None:
        # snapshot_download returns a commit directory. Keep both fetches on
        # that commit even when the requested Hub branch moves between them.
        directory = sources.snapshot(checkpoint, directory.name,
                                       weights=denoiser.text.components(index))
    vae = AutoencoderKL(channels=tuple(_component_config(directory, "vae")["block_out_channels"]))
    encoder, _, _ = denoiser.text.build(
        directory, index, denoiser, _call_policy(index, denoiser), compute,
        denoiser.sample_size * vae.downscale_factor, param_dtype=param_dtype, params=params)
    return encoder


def _unet_denoiser(directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build the published UNet: cross attention over one or two CLIP towers, whose
    pooled text conditioning is the one its added time features ask for."""
    from dew.interop import diffusion
    from dew.nn.backbones.unet_condition import UNet2DCondition

    config = _component_config(directory, "unet")
    fields = diffusion.unet_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = UNet2DCondition(**fields)

    def weights(param_dtype: str) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, layouts = diffusion.translate_unet_weights(
            diffusion.component_tensors(directory, "unet"), model, param_dtype=param_dtype)
        return {"params": params}, layouts

    pooled = model.additional_time_features > 0
    built = {"name": "unet_2d_condition",
             "fields": {**fields, "dtype": dtype,
                        "stages": [asdict(stage) for stage in model.stages]}}
    return _Denoiser(
        component="unet", model=model, weights=weights,
        built=built, config=config,
        text=_TextTowers("clip_pooled" if pooled else "clip", ("text_encoder", "text_encoder_2")), patch=1, latent_input=model.in_channels,
        sample_size=records.integer(config["sample_size"], "sample_size"),
        context_width=records.integer(config.get("cross_attention_dim", 1280), "cross_attention_dim"),
        pipeline="StableDiffusionXLPipeline" if pooled else "StableDiffusionPipeline")


def _denoiser(directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build the published denoiser this directory holds: its transformer, by the
    class it names, or else its UNet."""
    if not (directory / "transformer" / "config.json").is_file():
        return _unet_denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    config = _component_config(directory, "transformer")
    published = config.get("_class_name")
    if published == "SD3Transformer2DModel":
        return _sd3_denoiser(config, directory, dtype=dtype, attention_impl=attention_impl)
    if published == "FluxTransformer2DModel":
        return _flux_denoiser(config, directory, dtype=dtype, attention_impl=attention_impl)
    if published == "QwenImage21Transformer2DModel":
        return _qwen_image_denoiser(config, directory, dtype=dtype, attention_impl=attention_impl)
    if published == "Flux2Transformer2DModel":
        return _flux2_denoiser(config, directory, dtype=dtype, attention_impl=attention_impl)
    if published == "ZImageTransformer2DModel":
        return _z_image_denoiser(config, directory, dtype=dtype, attention_impl=attention_impl)
    raise ValueError(f"Native diffusion does not implement the published transformer "
                     f"{published!r}")


def _sd3_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build SD3's MM-DiT: both CLIP towers and the T5 tower read jointly, with the
    stored position buffer in its own frozen collection."""
    from dew.interop import diffusion
    from dew.nn.backbones.sd3 import SD3Transformer

    fields = diffusion.sd3_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = SD3Transformer(**fields)

    def weights(param_dtype: str) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, buffers, layouts = diffusion.translate_sd3_weights(
            diffusion.component_tensors(directory, "transformer"), param_dtype=param_dtype)
        return {"params": params, "buffers": buffers}, layouts

    built = {"name": "sd3_transformer",
             "fields": {**fields, "dtype": dtype,
                        "dual_attention_layers": list(fields["dual_attention_layers"])}}
    return _Denoiser(
        component="transformer", model=model, weights=weights,
        built=built, config=config,
        text=_TextTowers("sd3", ("text_encoder", "text_encoder_2"), t5_tower="text_encoder_3"),
        patch=fields["patch_size"],
        latent_input=fields["in_channels"], sample_size=records.integer(config["sample_size"], "sample_size"),
        context_width=fields["joint_attention_dim"], pipeline="StableDiffusion3Pipeline")


def _flux_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build Flux's transformer: one CLIP tower for the pooled vector, the T5 tower
    for the sequence, and a latent its pipeline packs in 2x2 patches.

    The class declares no sample size; its pipeline's `default_sample_size`
    is 128 latent positions, which a directory overrides with its own
    geometry. It starts from the sigmas its pipeline hands the scheduler.
    """
    from dew.interop import diffusion
    from dew.nn.backbones.flux import FluxTransformer

    fields = diffusion.flux_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = FluxTransformer(**fields)

    def weights(param_dtype: str) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, layouts = diffusion.translate_flux_weights(
            diffusion.component_tensors(directory, "transformer"), param_dtype=param_dtype)
        return {"params": params}, layouts

    built = {"name": "flux_transformer",
             "fields": {**fields, "dtype": dtype,
                        "axes_dims_rope": list(fields["axes_dims_rope"])}}
    return _Denoiser(
        component="transformer", model=model, weights=weights,
        built=built, config=config,
        text=_TextTowers("flux", ("text_encoder",), t5_tower="text_encoder_2",
                         embeds_guidance=fields["guidance_embeds"]), patch=2,
        latent_input=fields["in_channels"] // 4,
        sample_size=records.integer(config.get("sample_size", 128), "sample_size"),
        context_width=fields["joint_attention_dim"], pipeline="FluxPipeline",
        origin="linspace")


def _qwen_image_denoiser(config: dict, directory: Path, *, dtype: str | None,
                         attention_impl: str) -> _Denoiser:
    """Build Qwen-Image 2.1's transformer: one stream over the Qwen3-VL
    encoder's prompt states and the latent, one token per position.

    The class declares no sample size, and neither does the published
    config; its pipeline renders at `output_resolution` 1024 pixels, 64
    latent positions through the VAE's 16x. It starts from the sigmas its
    pipeline hands the scheduler.
    """
    from dew.interop import diffusion
    from dew.nn.backbones.qwen_image import QwenImageTransformer

    fields = diffusion.qwen_image_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = QwenImageTransformer(**fields)

    def weights(param_dtype: str) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, layouts = diffusion.translate_qwen_image_weights(
            diffusion.component_tensors(directory, "transformer"), param_dtype=param_dtype)
        return {"params": params}, layouts

    built = {"name": "qwen_image_transformer",
             "fields": {**fields, "dtype": dtype, "axes_dims_rope": list(fields["axes_dims_rope"])}}
    return _Denoiser(
        component="transformer", model=model, weights=weights, built=built, config=config,
        text=_QwenImageText(), patch=1,
        latent_input=fields["in_channels"], sample_size=64,
        context_width=fields["context_in_dim"], pipeline="QwenImage21Pipeline", origin="linspace")


def _flux2_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build FLUX.2's transformer over its VAE's folded latent, one token per
    position, conditioned by stacked text-encoder states.

    Its pipelines render at `default_sample_size` 128 through the VAE's 8x,
    which is 64 folded positions, and hand the scheduler `linspace(1, 1/N,
    N)` with their own empirical mu.
    """
    from dew.interop import diffusion
    from dew.nn.backbones.flux2 import Flux2Transformer

    fields = diffusion.flux2_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = Flux2Transformer(**fields)

    def weights(param_dtype: str) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, layouts = diffusion.translate_flux2_weights(
            diffusion.component_tensors(directory, "transformer"), param_dtype=param_dtype)
        return {"params": params}, layouts

    built = {"name": "flux2_transformer",
             "fields": {**fields, "dtype": dtype, "axes_dims_rope": list(fields["axes_dims_rope"])}}
    guided = fields["guidance_embeds"]
    return _Denoiser(
        component="transformer", model=model, weights=weights, built=built, config=config,
        text=_HiddenStatesText("flux2", embeds_guidance=guided), patch=1, latent_input=fields["in_channels"], sample_size=64,
        context_width=fields["joint_attention_dim"],
        pipeline="Flux2Pipeline" if guided else "Flux2KleinPipeline", origin="empirical")


def _z_image_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build Z-Image's single-stream transformer over the Flux VAE's latent,
    cut into 2x2 patches, conditioned by its Qwen3 encoder's second-to-last
    layer.

    Its pipeline renders at 1024 pixels by default, 128 latent positions
    through the VAE's 8x, and hands its statically shifting scheduler
    `linspace(1, 1/N, N)`.
    """
    from dew.interop import diffusion
    from dew.nn.backbones.z_image import ZImageTransformer

    fields = diffusion.z_image_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = ZImageTransformer(**fields)

    def weights(param_dtype: str) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, layouts = diffusion.translate_z_image_weights(
            diffusion.component_tensors(directory, "transformer"), param_dtype=param_dtype)
        return {"params": params}, layouts

    built = {"name": "z_image_transformer",
             "fields": {**fields, "dtype": dtype, "axes_dims": list(fields["axes_dims"]),
                        "axes_lens": list(fields["axes_lens"])}}
    return _Denoiser(
        component="transformer", model=model, weights=weights, built=built, config=config,
        text=_HiddenStatesText("z_image"), patch=2, latent_input=fields["in_channels"], sample_size=128,
        context_width=fields["cap_feat_dim"], pipeline="ZImagePipeline", origin="linspace")


def _component_config(directory: Path, name: str) -> dict:
    """Read one published component's own config file."""
    file = "scheduler_config.json" if name == "scheduler" else "config.json"
    with open(directory / name / file) as handle:
        return json.load(handle)


def _present(index: Mapping[str, object], name: str) -> bool:
    """Return whether the index declares a component rather than declaring it absent."""
    entry = index.get(name)
    return isinstance(entry, list) and entry[0] is not None


def _diffusion_vae(directory: Path, compute, *, param_dtype: str = "float32",
                   params: Variables | None = None
                   ) -> tuple[AutoEncoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build the published autoencoder, its parameters and their source layouts;
    supplied `params` are bound without a weight read."""
    from dew.interop import diffusion
    from dew.nn.autoencoders import AutoencoderKL, StableDiffusionVAE
    from dew.nn.autoencoders.vae import _vae_path

    config = _component_config(directory, "vae")
    if config.get("_class_name") == "AutoencoderKLQwenImage21":
        from dew.nn.autoencoders.qwen_image import load_qwen_image_vae
        return load_qwen_image_vae(directory, compute, param_dtype=param_dtype, params=params)
    if config.get("_class_name") == "AutoencoderKLFlux2":
        from dew.nn.autoencoders.flux2 import load_flux2_vae
        return load_flux2_vae(directory, compute, param_dtype=param_dtype, params=params)
    model = AutoencoderKL(
        channels=tuple(config["block_out_channels"]), latent_channels=config["latent_channels"],
        image_channels=config["in_channels"], blocks_per_level=config["layers_per_block"],
        norm_groups=config["norm_num_groups"], quantize=diffusion.flag(config, "use_quant_conv", default=True),
        post_quantize=diffusion.flag(config, "use_post_quant_conv", default=True), dtype=compute)
    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        tensors = diffusion.component_tensors(directory, "vae")
        params, layouts = diffusion.record_layouts(
            "vae", tensors, lambda name: _vae_path(name, np.ndim(tensors[name])), ("autoencoder",),
            param_dtype=param_dtype)
    autoencoder = StableDiffusionVAE(str(directory), dtype=compute, params=params, model=model,
                                     latent_shift=config.get("shift_factor") or 0.0,
                                     latent_scale=config.get("scaling_factor", 0.18215))
    return autoencoder, params, layouts, config


def _clip_towers(directory: Path, names: tuple[str, ...], compute, *, param_dtype: str = "float32",
                 params: Variables | None = None):
    """Build the published CLIP text towers, their tokenizers, their parameters and
    the layouts those parameters came from."""
    from transformers import CLIPTokenizer

    from dew.interop import diffusion
    from dew.nn.text_encoders import CLIPTextTransformer, translate_config

    towers, tokenizers, layouts = [], [], ()
    bound = {} if params is None else params
    for name in names:
        config = _component_config(directory, name)
        towers.append(CLIPTextTransformer(**translate_config(config), dtype=compute))
        if params is None:
            tower, recorded = diffusion.record_layouts(
                name, diffusion.component_tensors(directory, name), _text_head_path,
                ("encoders", "conditioning", name), param_dtype=param_dtype)
            bound = {**bound, name: tower}
            layouts += recorded
        tokenizers.append(CLIPTokenizer.from_pretrained(
            directory / ("tokenizer" + name.removeprefix("text_encoder"))))
    return tuple(towers), tuple(tokenizers), bound, layouts


def _t5_tower(directory: Path, compute, component: str, tokens: int, *, param_dtype: str = "float32",
              params: Variables | None = None):
    """Build the published T5 encoder as the conditioner's segment, with its
    parameters, their layouts and its config.

    `component` is where the family keeps it: an SD3 directory's third text
    encoder, a Flux directory's second one; `tokens` is the sequence budget
    the pipeline pads to.
    """
    from dew.data.text import load_tokenizer
    from dew.interop import diffusion
    from dew.nn.text_encoders import T5EncoderTransformer, _t5_path, t5_embedding, translate_t5_config

    config = _component_config(directory, component)
    tower = T5EncoderTransformer(**translate_t5_config(config), dtype=compute)
    layouts = ()
    if params is None:
        tensors = diffusion.component_tensors(directory, component)
        t5_embedding(tensors)
        params, layouts = diffusion.record_layouts(
            component, tensors, _t5_path, ("encoders", "conditioning", component), param_dtype=param_dtype)
    tokenizer = load_tokenizer(str(directory / ("tokenizer" + component.removeprefix("text_encoder"))))
    return T5Segment(tower, tokenizer, component, tokens), params, layouts, config


def _qwen_vl_text_config(config: Mapping[str, object]) -> dict:
    """The Qwen3-VL encoder's text_config as the Qwen3 decoder it computes for a prompt.

    Qwen-Image encodes its text-to-image prompt with no image, and a
    text-only row puts all three of Qwen3-VL's rotary axes at the token's
    position, so the sections of `mrope_section` - interleaved or not -
    rotate every channel pair at that one position: the plain rotary table.
    The rest of the text_config is Qwen3's, down to the per-head query and
    key norms, and the Qwen3 translator reads and checks it.
    """
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"Qwen-Image's text encoder is a qwen3_vl model, not {config.get('model_type')!r}")
    text = dict(records.record(config["text_config"], "text_config"))
    if text.get("model_type") != "qwen3_vl_text":
        raise ValueError(f"A qwen3_vl text_config is qwen3_vl_text, not {text.get('model_type')!r}")
    half = records.integer(text["head_dim"], "head_dim") // 2
    for key in ("rope_parameters", "rope_scaling"):
        entry = text.get(key)
        if isinstance(entry, Mapping):
            entry = dict(entry)
            section = records.integers(entry.pop("mrope_section"), "mrope_section")
            records.boolean(entry.pop("mrope_interleaved", False), "mrope_interleaved")
            if sum(section) != half:
                raise ValueError(f"mrope_section {section} must cover the {half} channel pairs")
            text[key] = entry
    return {**text, "model_type": "qwen3",
            "tie_word_embeddings": records.boolean(config.get("tie_word_embeddings", False),
                                                   "tie_word_embeddings")}


def _qwen_text_path(record: decoders.DecoderFields):
    """Map a Qwen3-VL checkpoint's tensors: its language model as the Qwen3
    decoder's, its head too, and its vision tower held as stored, which the
    text-to-image prompt never reads and an export writes back."""
    family = decoders.families()["qwen3"]

    def path(name: str) -> tuple[str, ...] | None:
        if name.startswith("model.language_model."):
            return family.weight_path("model." + name.removeprefix("model.language_model."), record)
        if name == "lm_head.weight":
            return family.weight_path(name, record)
        if name.startswith("model.visual."):
            # One flat leaf per stored tensor: these are carried, not run,
            # so no module path, and no sharding rule, reads them.
            return ("visual", name.removeprefix("model.visual."))
        raise ValueError(f"unknown tensor name {name!r}")
    return path


def _qwen_image_conditioning(directory: Path, index: Mapping[str, object], compute, size: int, *,
                             tokens: int, param_dtype: str, attention_impl: str,
                             params: Variables | None = None
                             ) -> tuple[QwenImageConditioner, tuple[WeightLayout, ...],
                                        dict[str, Mapping[str, object]]]:
    """Build Qwen-Image's conditioner: the Qwen3-VL language model, its
    processor's tokenizer and chat template, the parameters and their layouts."""
    from dew.data.text import load_tokenizer
    from dew.interop import diffusion

    config = _component_config(directory, "text_encoder")
    record = decoders.translate_config(_qwen_vl_text_config(config))
    named = dtype_name(compute)
    if named is None:
        raise ValueError("Qwen-Image's Qwen3-VL encoder computes in a named dtype; pass dtype")
    built = with_precision("causal_transformer", record, dtype=named, attention_impl=attention_impl)
    decoder = models.build("causal_transformer", built)
    if not isinstance(decoder, CausalTransformer):
        raise TypeError("causal_transformer registry entry must build CausalTransformer")
    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        tower, layouts = diffusion.record_layouts(
            "text_encoder", diffusion.component_tensors(directory, "text_encoder"),
            _qwen_text_path(record), ("encoders", "conditioning", "text_encoder"),
            param_dtype=param_dtype)
        params = {"text_encoder": tower}
    height, width = index.get("dew_height", size), index.get("dew_width", size)
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError("Image geometry must contain positive integer dimensions")
    encoder = QwenImageConditioner(
        decoder, load_tokenizer(str(directory / "processor")), params, str(directory),
        height, width, tokens=tokens, param_dtype=param_dtype)
    return encoder, layouts, {"text_encoder": config}


def load_qwen_image_conditioner(checkpoint: str, *, dtype: str | None = "bfloat16",
                                param_dtype: str = "float32", revision: str | None = None,
                                attention_impl: str = "auto", tokens: int = 512,
                                params: Variables | None = None):
    """Load Qwen-Image's text conditioning, or bind supplied parameters using metadata only."""
    compute = resolve_dtype(dtype)
    resolve_dtype(param_dtype)
    directory = sources.snapshot(checkpoint, revision, weights=False)
    with open(directory / "model_index.json") as handle:
        index = json.load(handle)
    denoiser = _denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    if not isinstance(denoiser.text, _QwenImageText):
        raise ValueError(f"{checkpoint} is not a Qwen-Image checkpoint")
    if params is None:
        directory = sources.snapshot(checkpoint, directory.name, weights=("text_encoder",))
    encoder, _, _ = _qwen_image_conditioning(
        directory, index, compute, denoiser.sample_size * 16, tokens=tokens,
        param_dtype=param_dtype, attention_impl=attention_impl, params=params)
    return encoder


_FLUX2_TEXT: Mapping[str, tuple[Literal["qwen3", "mistral3"], tuple[int, ...]]] = MappingProxyType({
    "qwen3": ("qwen3", (9, 18, 27)), "mistral3": ("mistral3", (10, 20, 30))})
"""Each FLUX.2 text encoder's `model_type`, the template its pipeline
formats a prompt with, and the `hidden_states` it stacks."""


def _hidden_states_path(record: decoders.DecoderFields, family: str, multimodal: bool):
    """Map a hidden-states text encoder's tensors: a Qwen3 language model's
    own names, or a Mistral-3's language model as the Mistral decoder's with
    its vision tower and projector held as stored, which a text prompt never
    reads and an export writes back. A Mistral-3 checkpoint names its
    language model `language_model.model.` and its head
    `language_model.lm_head` (transformers 4.50 and 5 write these), or
    `model.language_model.` and `lm_head` (4.52 to 4.57)."""
    decoder = decoders.families()[family]

    def path(name: str) -> tuple[str, ...] | None:
        if not multimodal or name == "lm_head.weight":
            return decoder.weight_path(name, record)
        if name == "language_model.lm_head.weight":
            return decoder.weight_path("lm_head.weight", record)
        for prefix in ("model.language_model.", "language_model.model."):
            if name.startswith(prefix):
                return decoder.weight_path("model." + name.removeprefix(prefix), record)
        if name.removeprefix("model.").startswith(("vision_tower.", "multi_modal_projector.")):
            return ("visual", name)
        raise ValueError(f"unknown tensor name {name!r}")
    return path


def _hidden_states_conditioning(directory: Path, index: Mapping[str, object], compute, size: int, *,
                                pipeline: Literal["flux2", "z_image"], tokens: int, guidance: float | None,
                                param_dtype: str, attention_impl: str, params: Variables | None = None
                                ) -> tuple[HiddenStatesConditioner, tuple[WeightLayout, ...],
                                           dict[str, Mapping[str, object]]]:
    """Build FLUX.2's or Z-Image's conditioner: the text encoder's language
    model, the tokenizer and chat template, the parameters and their layouts.
    Z-Image reads the output of the encoder's second-to-last layer with the
    template's thinking on."""
    from dew.data.text import load_tokenizer
    from dew.interop import diffusion

    config = _component_config(directory, "text_encoder")
    kind = records.text(config.get("model_type"), "model_type")
    known = _FLUX2_TEXT if pipeline == "flux2" else {"qwen3": _FLUX2_TEXT["qwen3"]}
    if kind not in known:
        raise ValueError(f"The {pipeline} text encoder is one of {sorted(known)}, not {kind!r}")
    template, layers = known[kind]
    multimodal = kind == "mistral3"
    text = dict(records.record(config["text_config"], "text_config")) if multimodal else config
    if multimodal:
        text["tie_word_embeddings"] = records.boolean(config.get("tie_word_embeddings", False),
                                                      "tie_word_embeddings")
    record = decoders.translate_config(text)
    named = dtype_name(compute)
    if named is None:
        raise ValueError(f"The {pipeline} text encoder computes in a named dtype; pass dtype")
    decoder = models.build("causal_transformer",
                           with_precision("causal_transformer", record, dtype=named, attention_impl=attention_impl))
    if not isinstance(decoder, CausalTransformer):
        raise TypeError("causal_transformer registry entry must build CausalTransformer")
    if pipeline == "z_image":
        layers = (decoder.num_layers - 1,)
    if max(layers) >= decoder.num_layers:
        # transformers' last hidden state is the final norm's output, which
        # none of these pipelines reads of its released encoder.
        raise ValueError(f"{pipeline} reads hidden state {max(layers)}, which a {decoder.num_layers}-layer "
                         "encoder does not have before its final norm")
    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        tower, layouts = diffusion.record_layouts(
            "text_encoder", diffusion.component_tensors(directory, "text_encoder"),
            _hidden_states_path(record, records.text(text["model_type"], "model_type"), multimodal),
            ("encoders", "conditioning", "text_encoder"), param_dtype=param_dtype)
        params = {"text_encoder": tower}
    height, width = index.get("dew_height", size), index.get("dew_width", size)
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError("Image geometry must contain positive integer dimensions")
    encoder = HiddenStatesConditioner(
        decoder, load_tokenizer(str(directory / "tokenizer")), params, str(directory), height, width,
        template=template, layers=layers, thinking=pipeline == "z_image", tokens=tokens, guidance=guidance,
        param_dtype=param_dtype)
    return encoder, layouts, {"text_encoder": config}


def load_hidden_states_conditioner(checkpoint: str, *, dtype: str | None = "bfloat16",
                                   param_dtype: str = "float32", revision: str | None = None,
                                   attention_impl: str = "auto", tokens: int = 512,
                                   params: Variables | None = None) -> HiddenStatesConditioner:
    """Load FLUX.2's or Z-Image's text conditioning, or bind supplied
    parameters using metadata only."""
    compute = resolve_dtype(dtype)
    resolve_dtype(param_dtype)
    directory = sources.snapshot(checkpoint, revision, weights=False)
    with open(directory / "model_index.json") as handle:
        index = json.load(handle)
    denoiser = _denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    text = denoiser.text
    if not isinstance(text, _HiddenStatesText):
        raise ValueError(f"{checkpoint} is neither a FLUX.2 nor a Z-Image checkpoint")
    if params is None:
        directory = sources.snapshot(checkpoint, directory.name, weights=("text_encoder",))
    policy = _call_policy(index, denoiser)
    encoder, _, _ = _hidden_states_conditioning(
        directory, index, compute, denoiser.sample_size * 16, pipeline=text.pipeline, tokens=tokens,
        guidance=policy.guidance if text.embeds_guidance else None, param_dtype=param_dtype,
        attention_impl=attention_impl, params=params)
    return encoder


def _image_safety(directory: Path, compute, *, param_dtype: str = "float32",
                  params: Variables | None = None):
    """Build the safety head a file declares: the finish, its parameters, their
    layouts and the two configs it ships; supplied `params` are bound without
    a weight read."""
    from dew.inputs.diffusion import CLIPImageTransform, CLIPSafetyHead, ImageSafety
    from dew.interop import diffusion
    from dew.nn.text_encoders import CLIPVisionTransformer, translate_vision_config

    config = _component_config(directory, "safety_checker")
    with open(directory / "feature_extractor" / "preprocessor_config.json") as handle:
        transform = json.load(handle)
    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        tensors = diffusion.component_tensors(directory, "safety_checker")
        # The root scoring vectors and thresholds are state, not tower/projection
        # weights. Preserve their FP32 contract without post-casting a whole tree.
        state = {name: value for name, value in tensors.items()
                 if (path := _safety_path(name)) is not None and len(path) == 1}
        weights = {name: value for name, value in tensors.items() if name not in state}
        params, layouts = diffusion.record_layouts(
            "safety_checker", weights, _safety_path, ("encoders", "safety"), param_dtype=param_dtype)
        scoring, state_layouts = diffusion.record_layouts(
            "safety_checker", state, _safety_path, ("encoders", "safety"), param_dtype="float32")
        params.update(scoring)
        layouts += state_layouts
    head = CLIPSafetyHead(CLIPVisionTransformer(**translate_vision_config(config), dtype=compute),
                          int(config["projection_dim"]), dtype=compute)
    return (ImageSafety(head, CLIPImageTransform.from_config(transform)), params, layouts,
            {"safety_checker": config, "feature_extractor": transform})


def _text_head_path(name: str):
    from dew.nn.text_encoders import _text_path
    if name == "text_projection.weight":
        return ("text_projection", "kernel")
    path = _text_path(name)
    return None if path is None else ("text_model", *path)


def _safety_path(name: str):
    from dew.nn.text_encoders import _clip_path
    if name.startswith(("concept_embeds", "special_care_embeds")):
        return (name,)
    return _clip_path(name.removeprefix("vision_model."))


def _wrapper_text_fields(config: Mapping[str, object], record: decoders.WrapperFields,
                         max_seq_len: int | None) -> decoders.DecoderFields:
    """Read the decoder fields a wrapper's text_config states, with the corrections
    its own family makes to them.

    Gemma 3 projects its logits without the causal-LM class's final tanh cap
    and attends its image spans both ways; Gemma 4 does the same on its
    sliding layers when the config asks for it; Qwen3.5 splits its rotary
    into the three mrope sections its wrapper positions rows with.
    """
    family = config.get("model_type")
    text_config = records.record(config["text_config"], "text_config")
    text_fields: decoders.DecoderFields = {**record["text"]}
    if max_seq_len is not None:
        text_fields["max_seq_len"] = max_seq_len
    if family == "gemma3":
        text_fields["final_logit_softcap"] = None
        text_fields["mixer"] = {"kind": "attention", "bidirectional_images": True}
    if family == "gemma4" and text_config.get("use_bidirectional_attention") == "vision":
        kinds = dict(text_fields.get("kinds") or {})
        sliding: decoders.KindFields = {
            **kinds.get("sliding_attention", {}),
            "mixer": {"kind": "attention", "bidirectional_images": True}}
        kinds["sliding_attention"] = sliding
        text_fields["kinds"] = kinds
    if family in _QWEN35_TYPES:
        rope = records.record(text_config.get("rope_parameters") or {}, "rope_parameters")
        sections = rope.get("mrope_section", [11, 11, 10])
        if (not isinstance(sections, (list, tuple)) or len(sections) != 3
                or any(type(value) is not int or value < 0 for value in sections)):
            raise ValueError("mrope_section must contain three nonnegative integer widths")
        kinds = dict(text_fields.get("kinds") or {})
        full: decoders.KindFields = {
            **kinds.get("full_attention", {}),
            "mixer": {"kind": "attention",
                      "mrope_section": [sections[0], sections[1], sections[2]]}}
        kinds["full_attention"] = full
        text_fields["kinds"] = kinds
    return text_fields


def _wrapper_model(config: Mapping[str, object], record: decoders.WrapperFields,
                   language_model: CausalTransformer, *, dtype: str) -> MultimodalTransformer:
    """Build the wrapper its record describes: the decoder above, the towers and
    projectors it names, and the placeholder ids its prompts carry."""
    family = records.text(config["model_type"], "model_type")
    text_config = records.record(config["text_config"], "text_config")
    audio_record = record["audio"]
    audio_projector = record["audio_projector"]
    return MultimodalTransformer(
        language_model, towers.from_record(record["tower"]),
        projectors.from_record(record["projector"]), family,
        record["image_token_id"], dtype=resolve_dtype(dtype),
        # A text_config that states a null pad id states none, and the
        # wrapper's own field defaults to 0 for exactly that.
        pad_token_id=records.integer(text_config.get("pad_token_id") or 0, "pad_token_id"),
        extra_placeholder_ids=(tuple(records.integer(config.get(name, default), name) for name, default in
            (("video_token_id", 258884), ("audio_token_id", 258881))) if family == "gemma4" else ()),
        audio=None if audio_record is None else towers.from_record(audio_record),
        audio_projection=(None if audio_projector is None
                          else projectors.from_record(audio_projector)),
        audio_soft_tokens=record["audio_soft_tokens"])


def _source_processor(directory: Path, config: Mapping[str, object], record: Mapping[str, object],
                      model: nn.Module, gguf_path: Path | None = None) -> Processor | None:
    """Build the host preprocessing a source ships, or None where it ships none.

    Only a model with towers reads images or audio, and only through the
    processor its repo ships; a text decoder takes its tokenizer whatever
    processor files sit beside it (DiffusionGemma publishes a Gemma 4
    processor config beside a text-only model), and a tiny multimodal
    fixture without processor files tokenizes text only.
    """
    processor_files = any((directory / name).exists()
                          for name in ("processor_config.json", "preprocessor_config.json"))
    if isinstance(model, MultimodalTransformer) and processor_files:
        from transformers import AutoProcessor
        options = {"backend": "pil"} if config.get("model_type") == "gemma3" else {}
        reference = AutoProcessor.from_pretrained(str(directory), local_files_only=True, **options)
        return Processor(reference, config, record, model.vocab_size)
    if (directory / "tokenizer_config.json").exists() or gguf_path is not None:
        from dew.data.text import load_tokenizer
        # A GGUF repo ships none; the file carries its tokenizer.
        tokenizer = (gguf.tokenizer(gguf_path) if gguf_path is not None
                     and not (directory / "tokenizer_config.json").exists()
                     else load_tokenizer(str(directory), local_files_only=True))
        if not _hosts(tokenizer):
            raise TypeError(f"the tokenizer in {directory} lacks a host processor operation")
        return Processor(tokenizer, config, record, model.vocab_size)
    return None


def split_revision(source: str) -> tuple[str, str | None]:
    """Split a `repo@revision` reference into the repo and the revision.

    Levanter's `RepoRef` spelling: a branch, tag or commit after the last
    '@'. A local directory, or a reference without '@', names no revision.
    Hub repo ids cannot contain '@'.
    """
    if os.path.isdir(source) or "@" not in source:
        return source, None
    name, revision = source.rsplit("@", 1)
    if not name or not revision:
        raise ValueError(f"{source!r} is not a repo@revision reference")
    return name, revision


AUTO = "auto"
"""The param_dtype that stores a checkpoint's parameters in its own dtype."""


def _checkpoint_dtype(config: Mapping[str, object], tensors: Mapping[str, np.ndarray]) -> str:
    """Return the storage dtype a checkpoint states, for param_dtype 'auto'.

    transformers' dtype='auto' rule (modeling_utils.py `_get_dtype`, 5.16.1):
    config.json's `dtype` (`torch_dtype` before 5.0), else the dtype of the
    first floating tensor. Packed FP8 or FP4 payloads are no storage dtype,
    so the first tensor stored in one is what a quantized checkpoint without
    a stated dtype resolves to. A diffusers pipeline states none, so its
    denoiser's tensors decide.
    """
    stated = config.get("dtype", config.get("torch_dtype"))
    if stated is not None:
        storage = dtype_name(resolve_dtype(records.text(stated, "dtype")))
        if storage is None:
            raise ValueError(f"dtype={stated!r} names no floating parameter storage")
        return storage
    storable = {np.dtype(np.float32): "float32", np.dtype(np.float16): "float16",
                np.dtype(ml_dtypes.bfloat16): "bfloat16"}
    for tensor in tensors.values():
        if tensor.dtype in storable:
            return storable[tensor.dtype]
    raise ValueError("param_dtype 'auto' found neither a stated dtype nor a float32, bfloat16 or "
                     "float16 tensor in the checkpoint")


def _pipeline_source(name_or_dir: str | Path, directory: Path, commit: str | None, single_file: str | None,
                     placed: Callable[[Variables], Variables], *, dtype: str, attention_impl: str,
                     param_dtype: str) -> Pretrained | None:
    """The source as a latent diffusion pipeline, or None when it is a decoder.

    A single file converts into the pipeline it describes; a directory with
    a model_index.json and no config.json of its own is one. A decoder that
    also ships a pipeline index for its sampler (DiffusionGemma) is loaded as
    the decoder its config names. `placed` puts the pipeline's variables on
    the mesh; a single file's conversion is kept only once that succeeds too.
    """

    def pipeline(directory: Path) -> Pretrained:
        with open(directory / "model_index.json") as handle:
            index = json.load(handle)
        storage = param_dtype
        if storage == AUTO:
            from dew.interop import diffusion
            denoiser = "transformer" if (directory / "transformer" / "config.json").is_file() else "unet"
            storage = _checkpoint_dtype({}, diffusion.component_tensors(directory, denoiser))
        loaded = _load_diffusion_source(directory, index, dtype=dtype, attention_impl=attention_impl,
                                        param_dtype=storage)
        return replace(loaded, variables=placed(loaded.variables), revision=commit)

    if single_file is not None:
        from dew.interop import single_file as original
        # A repo or directory that describes the pipeline (model_index.json
        # and its component configs) is the configs, as from_single_file(config=)
        # takes; a Hub repo's are its metadata snapshot, so the weights of a
        # component the file lacks come from the repo at that commit.
        configs = directory if (directory / "model_index.json").is_file() else None
        hub = None if configs is None or commit is None else (str(name_or_dir), commit)
        with original.unpacked(sources.repo_file(name_or_dir, directory, single_file), configs, hub) as (
                converted, published):
            return replace(pipeline(converted), source=published)
    if (directory / "model_index.json").is_file() and not (directory / "config.json").is_file():
        with open(directory / "model_index.json") as handle:
            index = json.load(handle)
        # The metadata fetch returns its commit directory, so the weights
        # come from that commit even if the requested branch moves.
        return pipeline(sources.snapshot(str(name_or_dir), directory.name, weights=tuple(
            name for name in index if _present(index, name))))
    return None


def _source_config(name_or_dir: str | Path, directory: Path, commit: str | None,
                   gguf_path: Path | None) -> tuple[Mapping[str, object], Mapping[str, np.ndarray] | None]:
    """The decoder's config, and its tensors where the config came with them
    (a GGUF file holds both); refused, naming what the source ships instead,
    before any weight downloads when there is no config.json."""
    if gguf_path is not None:
        return gguf.read(gguf_path)
    if not (directory / "config.json").is_file():
        # A GGUF repo often ships none, and says which argument reads it.
        files = (sources.repo_files(str(name_or_dir), directory) if commit is not None else
                 {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()})
        shipped = sources.missing_weights(str(name_or_dir), files)
        raise FileNotFoundError(f"{name_or_dir} has no config.json, which says what model its weights "
                                f"are; {shipped}")
    with open(directory / "config.json") as handle:
        config = records.record(json.load(handle), "config.json")
    # A GGUF file holds no kimi_k25 wrapper, so only a config.json can carry this.
    text_config = config.get("text_config")
    if (config.get("model_type") == "kimi_k25" and isinstance(text_config, Mapping)
            and text_config.get("quantization_config") is not None):
        raise ValueError(
            "text_config.quantization_config is not supported for the kimi_k25 text-only loader; "
            "provide dequantized text weights and remove that quantization descriptor")
    return config, None


def _decoded(tensors: Mapping[str, np.ndarray], config: Mapping[str, object], param_dtype: str
             ) -> tuple[Mapping[str, np.ndarray], tuple[str, ...], str | None, dict[str, np.ndarray]]:
    """The tensors with a quantized source's weights decoded, and what `save`
    needs to write them back: the quantized names, the scales' dtype and the
    integer formats' grid."""
    quantization = source_quantization(config)
    if quantization is None:
        return tensors, (), None, {}
    quantized_tensors, scale_dtype = quantization.names(tensors), quantization.scale_dtype(tensors)
    grid = {part: tensors[part] for name in quantized_tensors for part in quantization.grid(name)
            if part in tensors}
    aliases: tuple[tuple[str, str], ...] = ()
    if param_dtype != "float32":
        aliases = decoders.validate_source_aliases(
            quantization.tensor_names(tensors), partial(quantization.read, tensors), config)
    tensors = quantization.dequantize(tensors, param_dtype=param_dtype)
    _share_quantized_aliases(tensors, aliases, quantized_tensors)
    return tensors, quantized_tensors, scale_dtype, grid


def _decoder_layouts(tensors: Mapping[str, np.ndarray], record: decoders.DecoderFields, family: str,
                     variables: Variables) -> tuple[tuple[WeightLayout, ...], dict[str, np.ndarray]]:
    """Each source tensor's binding into the tree, and the tensors none binds."""
    bindings, retained = [], {}
    for name, tensor in tensors.items():
        binding = _language_layout(name, name, tensor, record, family, variables)
        if binding is None:
            retained[name] = tensor
        else:
            bindings.append(binding)
    return tuple(bindings), retained


def _generation_config(directory: Path) -> Mapping[str, object]:
    """The source's generation_config.json, or {} when it has none."""
    generation_path = directory / "generation_config.json"
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}

    def policy_read() -> None:
        """Refuse a generation_config.json that is not an object.

        Loading for training or export does not opt into the source sampler;
        active policy support is checked when the caller creates its task.
        """
        if not isinstance(generation_config, dict):
            raise ValueError("generation_config.json must contain an object")

    agreed("pretrained generation policy", policy_read)
    return generation_config


class _Built(NamedTuple):
    """A decoder source built and bound: the model, its variables, the
    record it was translated to, what it was built from, and the source
    tensors' bindings and the ones none binds."""

    model: nn.Module
    variables: Variables
    record: Mapping[str, object]
    built: Mapping[str, object]
    layouts: tuple[WeightLayout, ...]
    retained: dict[str, np.ndarray]


def _wrapper_source(config: Mapping[str, object], tensors: Mapping[str, np.ndarray], directory: Path, *,
                    dtype: str, attention_impl: str, max_seq_len: int | None, param_dtype: str,
                    lazy: bool) -> _Built:
    """A multimodal wrapper: the decoder under its text_config and the towers beside it."""
    record = decoders.translate_wrapper_config(config)
    text_fields = _wrapper_text_fields(config, record, max_seq_len)
    # The Transformers conditional classes ignore auxiliary prediction
    # layers. A released config advertises a depth even when its checkpoint
    # contains only the trunk; a source with mtp.* retains its actual depth.
    if (record['text_model_type'] in _QWEN35_TEXT_TYPES
            and not any(name.startswith(('mtp.', 'model.mtp.')) for name in tensors)):
        text_fields['num_nextn_predict_layers'] = 0
        record['text']['num_nextn_predict_layers'] = 0
    text: decoders.DecoderFields = {**text_fields, **precision_fields(
        "causal_transformer", text_fields, dtype=dtype, attention_impl=attention_impl)}
    wrapper: decoders.WrapperFields = {**record, "text": text}
    language_model = models.build("causal_transformer", wrapper["text"])
    if not isinstance(language_model, CausalTransformer):
        raise TypeError("causal_transformer registry entry must build CausalTransformer")
    model = _wrapper_model(config, record, language_model, dtype=dtype)
    parts = decoders.translate_wrapper_weights(tensors, record, param_dtype=param_dtype, lazy=lazy)
    variables = _native_variables({**parts, "language_model": decoders.with_constants(
        parts["language_model"], record["text"], directory)})
    layouts, retained = _wrapper_layouts(tensors, record, variables)
    return _Built(model, variables, record, wrapper, layouts, retained)


def _decoder_source(config: Mapping[str, object], tensors: Mapping[str, np.ndarray], directory: Path,
                    verified: verify.VerifiedMapping | None, *, dtype: str, attention_impl: str,
                    max_seq_len: int | None, param_dtype: str, lazy: bool) -> _Built:
    """A decoder of a registered family, or of one it was verified as (tier 2)."""
    if verified is None:
        record = decoders.translate_config(config)
        # translate_config refused every model_type but a registered family's name.
        family = records.text(config.get("model_type"), "model_type")
    else:
        record = verified.translate(config, tensors)
        family = verify.CONVENTION
    if max_seq_len is not None:
        record["max_seq_len"] = max_seq_len
    built = with_precision("causal_transformer", record, dtype=dtype, attention_impl=attention_impl)
    model = models.build("causal_transformer", built)
    variables = decoders.with_constants(decoders.translate_weights(
        tensors, record, family, param_dtype=param_dtype, lazy=lazy), record, directory)
    decoders._check_tree(variables, model)
    # The bindings are what an adapter loader resolves source names through
    # and what a quantized source is written back through, so a
    # derived-export family binds too; `save` picks its writer by
    # preserve_source_layout and quantization, not by whether bindings exist.
    # A family whose tensors are rewritten before the path map reads them
    # (Gemma 4's prepare) has no raw-name bindings.
    entry = decoders.families()[family]
    layouts, retained = ((), {})
    if entry.preserve_source_layout or entry.prepare_weights is decoders.DecoderFamily.prepare_weights:
        layouts, retained = _decoder_layouts(tensors, record, family, variables)
    return _Built(model, variables, record, built, layouts, retained)


def load_pretrained(name_or_dir: str | Path, *, dtype: str = "bfloat16", param_dtype: str = "float32",
                    attention_impl: str = "auto", max_seq_len: int | None = None,
                    revision: str | None = None, gguf_file: str | None = None,
                    single_file: str | None = None,
                    mesh: MeshSpec | None = None, layout: Layout | None = None,
                    fallback: str | None = None) -> Pretrained:
    """Load a source into a native Flax model with explicit parameter trees.

    ``name_or_dir`` is a local HF directory or a Hub model identifier. The
    decoder/tower/projector maps preserve their established internal paths;
    wrapper variables join under their existing component names. Processor
    artifacts are loaded only when the source contains them.
    dtype selects computation; param_dtype independently selects floating
    parameter storage and defaults to FP32 masters, or 'auto' stores the
    checkpoint's own dtype (`_checkpoint_dtype`). Frozen component weights
    (text encoders and VAE) follow it too; router, clipping, positional and
    safety state retain their own FP32/integer contracts.

    Without `mesh` or `layout` the variables are host arrays. With either,
    they are placed on that mesh (the default `MeshSpec()` when only
    `layout` is given) under that layout, one leaf at a time: a decoder's
    leaves are read from the mapped checkpoint one device shard at a time
    and cast and transposed there (`dew.interop.streaming`), so the host
    never holds the translated model. Towers, projectors and a quantized
    source's dequantized tensors are still built whole on the host first.

    ``gguf_file`` names a GGUF file in the repo or directory: its metadata is
    the config, its block-quantized tensors are dequantized to float32
    (`dew.interop.gguf`), and its tokenizer is the processor where the repo
    ships no tokenizer.

    ``single_file`` names an original-format diffusion checkpoint in the
    repo or directory: diffusers' own key maps convert it once into Dew's
    cache as the diffusers pipeline it describes (`dew.interop.single_file`),
    which then loads; the cache entry is published only once that load
    succeeds. The configs are the repo or directory's own when it has a
    model_index.json, and otherwise the diffusers repo diffusers infers from
    the checkpoint, at the commit fetched. A component the file lacks gets
    its weights from the same place, or is refused by name.

    `fallback="torchax"` opts into tier 3 for any causal LM transformers
    can build, registered or not: transformers' PyTorch forward lowered to
    JAX by torchax (`dew.interop.torchax_fallback`), with no Dew kernels,
    sharding rules or cached generation.
    """
    if fallback not in (None, "torchax"):
        raise ValueError(f"fallback={fallback!r} names no loader; the one fallback is 'torchax', "
                         "tier 3 through transformers' PyTorch forward")
    streaming = mesh is not None or layout is not None

    def placed(variables: Variables) -> Variables:
        return place(variables, mesh, layout) if streaming else variables

    if param_dtype != AUTO:
        storage = dtype_name(resolve_dtype(param_dtype))
        if storage is None:
            raise ValueError("param_dtype must select floating parameter storage")
        param_dtype = storage
    directory = sources.snapshot(str(name_or_dir), revision, weights=False)
    # A Hub snapshot directory is named by its commit.
    commit = None if os.path.isdir(name_or_dir) else directory.name
    if fallback is not None:
        from dew.interop import torchax_fallback
        loaded = torchax_fallback.load(name_or_dir, directory, commit, dtype=dtype, param_dtype=param_dtype,
                                       attention_impl=attention_impl, max_seq_len=max_seq_len)
        return replace(loaded, variables=placed(loaded.variables))
    pipeline = _pipeline_source(name_or_dir, directory, commit, single_file, placed, dtype=dtype,
                                attention_impl=attention_impl, param_dtype=param_dtype)
    if pipeline is not None:
        return pipeline
    gguf_path = None if gguf_file is None else sources.repo_file(name_or_dir, directory, gguf_file)
    config, tensors = _source_config(name_or_dir, directory, commit, gguf_path)
    mamba_ssm = mamba2.is_mamba_ssm(config)
    if mamba_ssm:
        config = mamba2.config_from_mamba_ssm(config)
    family = config.get("model_type")
    # Before any weight downloads: a format the codec cannot read is refused
    # on the config alone.
    source_quantization(config)
    # An unregistered decoder is checked against transformers on the config
    # alone, before its weights download (tier 2, dew.interop.verify).
    verified = (verify.verify_mapping(config) if isinstance(family, str) and family not in decoders.families()
                and family != "diffusion_gemma" and "text_config" not in config else None)
    if tensors is None:
        directory = sources.snapshot(str(name_or_dir), directory.name)
        tensors = sources.load_shards(directory)
    if mamba_ssm:
        tensors = mamba2.tensors_from_mamba_ssm(tensors)
    if param_dtype == AUTO:
        param_dtype = _checkpoint_dtype(config, tensors)
    tensors, quantized_tensors, scale_dtype, grid = _decoded(tensors, config, param_dtype)
    export_adapter = None
    if family == "diffusion_gemma":
        from dew.interop import diffusion_gemma
        model = diffusion_gemma.build(config, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len)
        variables = diffusion_gemma.translate_weights(tensors, config, param_dtype=param_dtype)
        record, layouts, retained = config, (), {}
        built: Mapping[str, object] = {**config, "dtype": dtype, "attention_impl": attention_impl}
        export_adapter = diffusion_gemma.export_weights
    elif "text_config" in config and (family not in decoders.families() or decoders._bundles(config)):
        # A wrapper repo carries its decoder under text_config. Where its
        # model_type is a registered decoder family, its towers have no
        # counterpart and the text half, read from the nested config, is the
        # model, unless the family reads its media bundle whole
        # (`DecoderFamily.wrapper`); `translate_config` refuses the rest.
        model, variables, record, built, layouts, retained = _wrapper_source(
            config, tensors, directory, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len,
            param_dtype=param_dtype, lazy=streaming)
    else:
        model, variables, record, built, layouts, retained = _decoder_source(
            config, tensors, directory, verified, dtype=dtype, attention_impl=attention_impl,
            max_seq_len=max_seq_len, param_dtype=param_dtype, lazy=streaming)
    processor = _source_processor(directory, config, record, model, gguf_path)
    generation_config = _generation_config(directory)
    return Pretrained(model, placed(variables), processor, config, directory, built, generation_config,
                      layouts, retained, export_adapter, quantized_tensors=quantized_tensors,
                      quantized_scale_dtype=scale_dtype, quantization_grid=grid,
                      # The weights' commit: a pickle repo's may be its conversion's.
                      revision=None if commit is None else directory.name)
