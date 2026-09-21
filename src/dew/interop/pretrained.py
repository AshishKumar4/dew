"""Load native models and their host processors from a Hugging Face source."""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NamedTuple, Protocol, TypedDict, Unpack

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew import records
from dew.artifacts import agree_process_phase
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import Origin, SourceSchedule
from dew.inference import BlockGeneration, MaskedGeneration, TextGeneration
from dew.inputs import Condition, Field, InputSpec
from dew.inputs.diffusion import Composition, DiffusionConditioner, T5Segment
from dew.interop import hf_decoders as decoders
from dew.interop.quantized import (
    dequantize_checkpoint,
    fp8_format,
    fp8_tensor_names,
    pack_fp8,
    read_fp8_tensor,
    scaled_names,
)
from dew.nn import audio as audio_nn
from dew.nn.autoencoders import AutoEncoder, StableDiffusionVAE
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.gpt_oss import mxfp4_stems, mxfp4_tensor_names, pack_mxfp4, read_mxfp4_tensor, unpack_mxfp4
from dew.nn.inputs import Media, ModelInputs, pad_token_rows
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.text_encoders import ParamTree
from dew.nn.vision import projector_from_record, tower_from_record
from dew.objectives.base import Variables
from dew.records import JSON
from dew.registry import dtype_name, models, precision_fields, resolve_dtype, with_precision
from dew.sampling import decoding
from dew.sampling.guidance import CFG
from dew.sampling.pipelines import TextToImage
from dew.sampling.strategies import Beam, Speculative, Strategy
from dew.sampling.text import Sampling


class ProcessorCall(TypedDict, total=False):
    """Every keyword dew hands a host processor beside `images`.

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
    """The HF processor operations kept outside compiled model computation."""

    def __call__(self, *, images: Media | None = None,
                 **kwargs: Unpack[ProcessorCall]) -> Mapping[str, object]: ...
    # The files it wrote are its own bookkeeping; dew calls this for the effect.
    def save_pretrained(self, save_directory: str) -> None: ...
    def apply_chat_template(self, conversation: Sequence[Mapping[str, object]],
                            **kwargs: JSON) -> str | Sequence[int] | Mapping[str, object]: ...
    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]: ...


def _row_padding(reference: HostProcessor) -> tuple[int, Literal["left", "right"]]:
    """The id and the side to pad text rows with, at the boundary a host
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


@dataclass(frozen=True)
class Processor:
    """Host text/image preprocessing followed by numeric layout normalization.

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
        if images is None and audio is None and videos is None:
            if video_metadata is not None:
                raise ValueError("video_metadata requires videos")
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
            if videos is None:
                raise ValueError("video_metadata requires videos")
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
            if "pixel_values_videos" in values and self.config.get("model_type") not in ("qwen3_5", "gemma4"):
                raise ValueError("video patch inputs require a Qwen3.5 or Gemma4 visual tower")
            image_fields, conditioning = self._images(values, tokens)
            token_fields.update(image_fields)
            if self.config.get("model_type") == "qwen3_5":
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
        image_id = self.record.get("image_token_id", self.config.get("image_token_id"))
        if type(image_id) is not int:
            raise ValueError("image_token_id must be an integer")
        qwen = self.config.get("model_type") == "qwen3_5"
        gemma = self.config.get("model_type") == "gemma4"
        video_id = (self.config.get("video_token_id", 258884) if gemma else
                    self.config.get("video_token_id") if qwen else None)
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
            vision = self.config.get("vision_config")
            if not isinstance(vision, Mapping):
                raise ValueError("patch position ids require a vision config")
            kernel = vision.get("pooling_kernel_size")
            if type(kernel) is not int or kernel < 1:
                raise ValueError("pooling_kernel_size must be a positive integer")
            table_size, patch_size = vision.get("position_embedding_size"), vision.get("patch_size")
            if type(table_size) is not int or table_size < 1 or type(patch_size) is not int or patch_size < 1:
                raise ValueError("position_embedding_size and patch_size must be positive integers")
            streams: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for pixel_name, position_name, token_id, ndim in (
                ("pixel_values", "image_position_ids", image_id, 3),
                ("pixel_values_videos", "video_position_ids", video_id, 4),
            ):
                if pixel_name not in values and position_name not in values:
                    continue
                if pixel_name not in values or position_name not in values:
                    raise ValueError(f"{pixel_name} and {position_name} arrive together")
                if type(token_id) is not int:
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
                # The source flattens video then frame axes, not prompt rows
                # (modeling_gemma4.py:2476-2488). Keep the two modality streams
                # separate until their placeholder runs establish row order.
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
        encoder = tower_from_record(audio)
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
        """Views of packed image/video patches in text order, one item per frame.

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
        """Qwen3.5's get_rope_index on host-normalized image grids.

        Text advances one coordinate per token. An image occupies its merged
        temporal/height/width grid, and following text starts after its longest
        spatial side. Padding keeps coordinate zero, as in the reference.
        """
        vision = self.config.get("vision_config")
        if not isinstance(vision, Mapping):
            raise ValueError("image rotary positions require a vision config")
        merge = vision.get("spatial_merge_size")
        if type(merge) is not int or merge < 1:
            raise ValueError("spatial_merge_size must be a positive integer")
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

    def save_pretrained(self, directory: str | Path) -> None:
        """Save the same processor and tokenizer used by this source."""
        self.reference.save_pretrained(str(directory))


@dataclass(frozen=True)
class WeightLayout:
    """An existing source tensor's location and reversible storage layout.

    `expert_index` is the expert a per-expert source tensor holds. The
    loader stacks those tensors onto an expert dimension
    (`hf_decoders._stack_experts`), so one stacked leaf answers for every
    expert of a layer and the index says which slice this tensor is.

    `dtype` is the width the source stores this tensor in where that is
    not its leaf's: DeepSeek V4's token-to-expert table is int64 on disk
    and int32 in the collection, and the export writes back what the
    checkpoint held.
    """

    name: str
    paths: tuple[tuple[str, ...], ...]
    shape: tuple[int, ...]
    transpose: tuple[int, ...] | None = None
    concatenate: int | None = None
    expert_index: int | None = None
    dtype: np.dtype | None = None

    def export(self, variables: Mapping[str, object], scalar_mode: str | None = None) -> np.ndarray:
        leaves = []
        for path in self.paths:
            if path[-1] == "layer_scalar":
                if scalar_mode not in ("frozen", "trainable"):
                    raise ValueError("layer_scalar export requires an explicit model mode")
                path = (("constants" if scalar_mode == "frozen" else "params"), *path[1:])
            node: object = variables
            for part in path:
                if not isinstance(node, Mapping):
                    raise ValueError(f"parameter path {path} does not traverse a mapping")
                node = node[part]
            if self.expert_index is not None:
                # Slice the expert where the leaf lives. One stacked leaf
                # answers for E source tensors, so copying it to the host
                # per tensor would move the whole stack E times.
                if not isinstance(node, (np.ndarray, jax.Array)):
                    raise ValueError(
                        f"{self.name} takes an expert of {path}, which holds "
                        f"{type(node).__name__} rather than an array")
                if node.ndim == 0 or not 0 <= self.expert_index < node.shape[0]:
                    raise ValueError(
                        f"{self.name} is expert {self.expert_index} of {path}, which "
                        f"holds {node.shape}")
                node = node[self.expert_index]
            leaves.append(np.asarray(node))
        value = leaves[0] if self.concatenate is None else np.concatenate(leaves, axis=self.concatenate)
        if self.transpose is not None:
            value = value.transpose(self.transpose)
        if value.size != math.prod(self.shape):
            raise ValueError(
                f"{self.name} assembles {value.shape} from {self.paths}, which does not "
                f"fill the source's {self.shape}")
        value = np.ascontiguousarray(value).reshape(self.shape)
        return value if self.dtype is None else value.astype(self.dtype)

    def restore(self, tensor: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """The leaf of `shape` whose export is `tensor`.

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
    """A per-expert leaf path as the stacked leaf the loaded tree holds.

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
    """How many axes a bound leaf carries ahead of its stored matrix.

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
    if not isinstance(node, np.ndarray | jax.Array):
        return 0
    rank = node.ndim - (0 if expert_index is None else 1)
    return max(rank - 2, 0)


def _language_layout(name: str, text_name: str, tensor: np.ndarray,
                     config, model_type: str, variables: Mapping[str, object],
                     component: str | None = None) -> WeightLayout | None:
    """The text family's existing leaf map plus its inverse storage operations."""
    family = decoders._FAMILIES[model_type]
    # A family whose checkpoint packs its experts as `[E, out, in]`
    # (`_gemma4_prepare` swaps them into dew's `[E, in, out]`) writes them
    # back swapped.
    packed = family.prepare_weights is decoders._gemma4_prepare

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
    return WeightLayout(name, paths, tensor.shape, transpose, concatenate,
                        expert_index, stored)



def _wrapper_layouts(tensors, record, variables):
    """Retain source names while borrowing the loader's internal leaf paths."""
    from dew.nn import vision

    tower_kind = record["tower"]["kind"]
    projector_kind = record["projector"]["kind"]
    tower_path: Callable[[str], tuple[str, ...] | None] = {"siglip": vision.siglip_vision_path, "llama4": vision.llama4_vision_path,
                  "gemma4": vision.gemma4_vision_path, "qwen3_5": vision.qwen35_vision_path,
                  "gemma3n": vision.gemma3n_vision_path}[tower_kind]
    tower_prefix = decoders._WRAPPER_TOWER_PREFIX[tower_kind]
    projector_prefix = decoders._WRAPPER_PROJECTOR_PREFIX[projector_kind]
    audio_encoder = None
    if record["audio"] is not None:
        audio_encoder = tower_from_record(record["audio"])
        if not isinstance(audio_encoder, (audio_nn.Gemma3nAudio, audio_nn.Gemma4Audio)):
            raise ValueError("source export requires a Gemma audio encoder")
    bindings = []
    retained = {}
    for name, tensor in tensors.items():
        bare = name.removeprefix("model.")
        paths: tuple[tuple[str, ...], ...] = ()
        transpose = None
        concatenate = None
        if bare.startswith(projector_prefix):
            tail = bare.removeprefix(projector_prefix)
            path = vision.projector_weight_path(projector_kind, tail)
            paths = (("params", "projector", *path),)
            if path[-1] == "kernel" and tail != "mm_input_projection_weight":
                transpose = (1, 0)
        elif bare.startswith(tower_prefix):
            path = tower_path(bare.removeprefix(tower_prefix))
            if path is not None:
                paths = ((path[0], "tower", *path[1:]),) if tower_kind == "gemma4" else (("params", "tower", *path),)
                if path[-1] == "kernel":
                    transpose = (1, 0) if tensor.ndim == 2 else (3, 2, 0, 1)
                    if tensor.ndim == 5:
                        transpose = (1, 0)
        elif audio_encoder is not None and bare.startswith(decoders._WRAPPER_AUDIO_PROJECTOR_PREFIX):
            tail = bare.removeprefix(decoders._WRAPPER_AUDIO_PROJECTOR_PREFIX)
            path = vision.projector_weight_path(record["audio_projector"]["kind"], tail)
            paths = (("params", "audio_projector", *path),)
            if path[-1] == "kernel":
                transpose = (1, 0)
        elif audio_encoder is not None and bare.startswith(decoders._WRAPPER_AUDIO_PREFIX):
            path = audio_nn.audio_weight_path(bare.removeprefix(decoders._WRAPPER_AUDIO_PREFIX), audio_encoder)
            paths = ((path[0], "audio_tower", *path[1:]),)
            if path[-1] == "kernel":
                # Kernels store [*window, in, out]; the source keeps [out, in, *window].
                transpose = {2: (1, 0), 3: (2, 1, 0), 4: (3, 2, 0, 1)}[tensor.ndim]
        elif bare.startswith(("language_model.", "mtp.")) or bare == "lm_head.weight":
            tail = bare.removeprefix("language_model.")
            text_name = tail if tail.startswith(("model.", "lm_head.", "mtp.")) else "model." + tail
            layout = _language_layout(name, text_name, tensor, record["text"],
                                      record["text_model_type"], variables,
                                      "language_model")
            if layout is None:
                retained[name] = tensor
            else:
                bindings.append(layout)
            continue
        else:
            raise ValueError(f"unknown source tensor {name!r}")
        if paths:
            bindings.append(WeightLayout(name, paths, tensor.shape, transpose, concatenate))
        else:
            # SigLIP's pooling head and reference-ignored auxiliary tensors
            # have no forward consumer; export preserves their source bytes.
            retained[name] = tensor
    return tuple(bindings), retained


@dataclass(frozen=True)
class _SourceQuantization:
    """A source format the loader undoes and `Pretrained.save` restores.

    `names` reads which tensors arrived quantized off the raw checkpoint,
    before dequantize replaces them with dense weights in requested storage;
    `requantize` writes those names back in the format.
    tensor_names and read expose original per-weight values for alias checks;
    read dequantizes on demand rather than retaining an FP32 model.
    """

    names: Callable[[Mapping[str, np.ndarray]], tuple[str, ...]]
    dequantize: Callable[[Mapping[str, np.ndarray]], dict[str, np.ndarray]]
    requantize: Callable[[Mapping[str, np.ndarray], tuple[str, ...]], dict[str, np.ndarray]]
    tensor_names: Callable[[Mapping[str, np.ndarray]], tuple[str, ...]]
    read: Callable[[Mapping[str, np.ndarray], str], np.ndarray]


def _source_quantization(config: Mapping[str, object], *, param_dtype: str = "float32") -> _SourceQuantization | None:
    """The format a config's `quantization_config` declares, or None for a dense source."""
    quantization = config.get("quantization_config")
    if quantization is None:
        return None
    if not isinstance(quantization, Mapping):
        raise ValueError(f"quantization_config must be an object, got {quantization!r}")
    method = quantization.get("quant_method")
    if method == "fp8":
        block, ue8m0 = fp8_format(quantization)
        return _SourceQuantization(
            scaled_names, partial(dequantize_checkpoint, block=block, param_dtype=param_dtype),
            partial(pack_fp8, block=block, ue8m0=ue8m0),
            fp8_tensor_names, partial(read_fp8_tensor, block=block))
    if method == "mxfp4":
        return _SourceQuantization(
            mxfp4_stems, partial(unpack_mxfp4, param_dtype=param_dtype), pack_mxfp4,
            mxfp4_tensor_names, read_mxfp4_tensor)
    raise ValueError(
        f"quantization_config names quant_method {method!r}; this loader reads DeepSeek's "
        f"fp8 blocks and GPT OSS's mxfp4 and nothing else")


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
    """A native model, explicit variables and its checkpoint's host processor.

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

    @property
    def layouts(self) -> Mapping[str, WeightLayout]:
        """Source module name to the layout of its weight: `model.<module>`
        for a decoder, `unet.<module>` for a pipeline component.

        These are the names a published adapter file writes, so this is what
        `dew.lora` binds a low-rank delta through.
        """
        return {layout.name.removesuffix(".weight").replace("/", "."): layout
                for layout in self.weight_layouts if layout.name.endswith(".weight")}

    def text_generation(self, *, sampling: Sampling | None = None) -> TextGeneration | MaskedGeneration:
        """Native MDLM for masked models, source decoding controls for causal models.

        Masked generation refines a full response with Unmask, not the source
        family's custom generation recipe. AR sampling overrides are refused.

        Without an override the task runs the source's whole chain, its
        criteria and the strategy its config names. An explicit `sampling`
        replaces the basic policy and clears that chain with it, because the
        chain was built around the policy the caller just replaced; the
        criteria and `num_return_sequences` still come from the source.
        """
        if self.process is not None:
            raise TypeError("a latent diffusion source generates through text_to_image")
        if isinstance(self.model, DiffusionGemma):
            raise TypeError("a DiffusionGemma source generates through block_generation")
        decoder = self.model if isinstance(self.model, CausalTransformer) else None
        mask_id = None if decoder is None else decoder.mask_token_id
        if decoder is not None and not decoder.causal and mask_id is not None:
            from dew.diffusion.discrete import MDLM
            if sampling is not None:
                raise TypeError("native MDLM accepts denoising steps, not autoregressive sampling controls")
            _audit_masked(self.config, self.generation_config)
            return MaskedGeneration(self.model, self.variables, MDLM(mask_id=mask_id)(), self.processor,
                eos_token_ids=_eos_ids(self.config, self.generation_config),
                pad_token_id=_pad_id(self.config, self.generation_config),
                max_new_tokens=_generation_limit(self.config, self.generation_config, "max_new_tokens"),
                max_length=_generation_limit(self.config, self.generation_config, "max_length"),
                n=_return_sequences(self.config, self.generation_config))
        rows = _return_sequences(self.config, self.generation_config)
        policy, logits, stopping, strategy = _source_decoding(
            self.config, self.generation_config, self.model, self.processor, rows,
            sampling)
        return TextGeneration(self.model, self.variables, self.processor, policy,
                              max_new_tokens=_generation_limit(self.config, self.generation_config, "max_new_tokens"),
                              max_length=_generation_limit(self.config, self.generation_config, "max_length"),
                              n=rows, logits=logits, stopping=stopping, strategy=strategy)

    def block_generation(self) -> BlockGeneration:
        """The DiffusionGemma as a canvas task, defaulting to the source's sampler config."""
        from dew.interop import diffusion_gemma
        if not isinstance(self.model, DiffusionGemma):
            raise TypeError("block generation needs a DiffusionGemma source")
        return BlockGeneration(self.model, self.variables,
                               diffusion_gemma.generation_process(self.config, self.generation_config),
                               self.processor, _eos_ids(self.config, self.generation_config),
                               _pad_id(self.config, self.generation_config),
                               max_new_tokens=_generation_limit(self.config, self.generation_config, "max_new_tokens"),
                               max_length=_generation_limit(self.config, self.generation_config, "max_length"),
                               n=_return_sequences(self.config, self.generation_config))

    def text_to_image(self) -> TextToImage:
        """The latent diffusion source as an image task with its published policy."""
        if self.process is None or self.inputs is None or self.schedule is None:
            raise TypeError("text_to_image needs a latent diffusion source")
        if self.task is None:
            raise TypeError("text_to_image needs the source's own call policy")
        return TextToImage(self.model, self.process, self.inputs, self.variables, self.autoencoder,
                           grid=self.task.grid, final_denoise=False, sampler=self.schedule.solver(),
                           steps=self.task.steps, guidance=self.task.guidance, finish=self.finish)

    def save(self, directory: str | Path, *, variables: Mapping[str, object] | None = None) -> None:
        """Write trained variables back to the source layout with its tokenizer assets."""
        from dew.interop.safetensors_io import save_hf_layout
        values = self.variables if variables is None else variables
        quantization = _source_quantization(self.config)
        if quantization is not None and not self.quantized_tensors:
            raise ValueError(
                "this source's config declares a quantization_config and the loader recorded "
                "no quantized tensors to write back in it")
        destination = Path(directory)
        generation_config = dict(self.generation_config)
        if self.schedule is not None:
            from dew.interop import diffusion
            diffusion.save_source(self, values, destination)
            return
        family = self.config.get("model_type")
        if self.export_adapter is not None:
            tensors = self.export_adapter(self.model, values, self.config)
        elif (isinstance(self.model, CausalTransformer) and isinstance(family, str)
              and not decoders._FAMILIES[family].preserve_source_layout and quantization is None):
            # A family that derives its export from the model is written by
            # the decoder export, which writes the whole directory: weights,
            # the config it derives, this processor's files and this
            # generation config. One export path, so a decoder saved here and
            # one saved directly leave the same files behind. A quantized
            # source is not derived: its packed format goes back over the
            # source names, so it takes the layout writer below.
            decoders.save_pretrained_decoder(self.model, values, destination,
                                             tokenizer=self.processor,
                                             generation_config=generation_config)
            return
        elif self.weight_layouts:
            # Source names and geometry first; the packed format goes back over them.
            text = self.model.language_model if isinstance(self.model, MultimodalTransformer) else self.model
            scalar_mode = text.layer_scalar if isinstance(text, CausalTransformer) else None
            tensors = {**self.retained_tensors,
                       **{layout.name: layout.export(values, scalar_mode) for layout in self.weight_layouts}}
        else:
            raise ValueError("this source has no reversible weight layout")
        if quantization is not None:
            tensors = quantization.requantize(tensors, self.quantized_tensors)
        save_hf_layout(tensors, dict(self.config), destination)
        decoders.save_export_assets(destination, tokenizer=self.processor,
                                    generation_config=generation_config)





def _native_variables(parts: Mapping[str, Mapping[str, ParamTree]]) -> dict[str, dict[str, ParamTree]]:
    collections: dict[str, dict[str, ParamTree]] = {}
    for component, variables in parts.items():
        for collection, tree in variables.items():
            collections.setdefault(collection, {})[component] = tree
    return collections


def _generation_value(config: Mapping[str, object], generation_config: Mapping[str, object],
                      name: str, default: JSON = None) -> JSON:
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise ValueError("text_config must be a mapping")
    return records.json_value(generation_config.get(name, config.get(name, text.get(name, default))), name)


def _eos_ids(config: Mapping[str, object], generation_config: Mapping[str, object]) -> tuple[int, ...]:
    value = _generation_value(config, generation_config, "eos_token_id")
    if value is None:
        return ()
    values = (value,) if type(value) is int else value
    if not isinstance(values, (tuple, list)):
        raise ValueError("eos_token_id must be an integer or a sequence of integers")
    ids: list[int] = []
    for entry in values:
        if type(entry) is not int or entry < 0:
            raise ValueError("eos_token_id must be an integer or a sequence of integers")
        ids.append(entry)
    return tuple(ids)


def _pad_id(config: Mapping[str, object], generation_config: Mapping[str, object]) -> int:
    value = _generation_value(config, generation_config, "pad_token_id", 0)
    if value is None:
        value = 0
    if type(value) is not int or value < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    return value


def _generation_limit(config: Mapping[str, object], generation_config: Mapping[str, object], name: str) -> int | None:
    """Read a nonnegative source generation limit."""
    value = _generation_value(config, generation_config, name)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _return_sequences(config: Mapping[str, object], generation_config: Mapping[str, object]) -> int:
    """The source's continuations per prompt; one when it declares none."""
    value = _generation_value(config, generation_config, "num_return_sequences")
    if value is None:
        return 1
    if type(value) is not int or value < 1:
        raise ValueError("num_return_sequences must be a positive integer")
    return value

def _probability_control(config: Mapping[str, object], generation_config: Mapping[str, object],
                         name: str, default: float) -> float:
    value = _generation_value(config, generation_config, name, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


@dataclass(frozen=True)
class _Control:
    """One source control's consumer, activation rule and unsupported case."""

    owner: Literal["policy", "task", "metadata", "inapplicable", "transform",
                   "criterion", "strategy", "capacity", "unsupported"]
    neutral: tuple[JSON, ...] = ()
    mode: Literal["always", "sampling", "beam"] = "always"
    refusal: str | None = None
    masked_neutral: tuple[JSON, ...] | None = None


# GenerationConfig is external data. Keep each control's disposition and
# neutral values together; the native component constructors own their defaults.
# Supplied-input causal tasks do not synthesize BOS/decoder-start tokens or
# switch their native result type through return_dict_in_generate.
_CONTROLS = {
    "_commit_hash": _Control("metadata"),
    "_from_model_config": _Control("metadata"),
    "assistant_confidence_threshold": _Control("strategy"),
    "assistant_early_exit": _Control("unsupported", refusal="early-exit proposal is not implemented"),
    "assistant_ensemble_weight": _Control(
        "unsupported",
        neutral=(1.0,),
        refusal="ensemble verification below one accepts a biased distribution"),
    "assistant_lookbehind": _Control(
        "unsupported",
        refusal="translating between two tokenizers' token spaces is not implemented"),
    "bad_words_ids": _Control("transform"),
    "begin_suppress_tokens": _Control("transform"),
    "bos_token_id": _Control("inapplicable"),
    "cache_config": _Control("unsupported", refusal="quantized and offloaded caches are not implemented"),
    "cache_implementation": _Control(
        "unsupported",
        neutral=('static',),
        refusal="the native cache is the fixed-capacity static one", masked_neutral=()),
    "compile_config": _Control("unsupported", refusal="the native decoder owns its compilation"),
    "constraints": _Control("unsupported", refusal="constrained beam search is not implemented"),
    "continuous_batching_config": _Control("unsupported", refusal="continuous batching is not implemented"),
    "decoder_start_token_id": _Control("inapplicable"),
    "disable_compile": _Control(
        "unsupported",
        neutral=(False,),
        refusal="the native decoder always runs compiled"),
    "diversity_penalty": _Control(
        "unsupported",
        neutral=(0.0,),
        mode="beam",
        refusal="diverse group beam search is not implemented"),
    "do_sample": _Control("policy", masked_neutral=(True,)),
    "dola_layers": _Control("unsupported", refusal="DoLa is a decoding strategy that is not implemented"),
    "early_stopping": _Control("strategy", neutral=(False,), mode="beam"),
    "encoder_no_repeat_ngram_size": _Control("transform", neutral=(0,)),
    "encoder_repetition_penalty": _Control("transform", neutral=(1.0,)),
    "eos_token_id": _Control("task"),
    "epsilon_cutoff": _Control("transform", neutral=(0.0,), mode="sampling"),
    "eta_cutoff": _Control("transform", neutral=(0.0,), mode="sampling"),
    "exponential_decay_length_penalty": _Control("transform"),
    "force_words_ids": _Control("unsupported", refusal="constrained beam search is not implemented"),
    "forced_bos_token_id": _Control("transform"),
    "forced_eos_token_id": _Control("transform"),
    "guidance_scale": _Control(
        "transform",
        neutral=(1.0,),
        refusal="classifier-free guidance evaluates the model a second time per step"),
    "is_assistant": _Control(
        "unsupported",
        neutral=(False,),
        refusal="a source loads as a target model, not as another model's assistant"),
    "length_penalty": _Control("strategy", neutral=(1.0,), mode="beam"),
    "low_memory": _Control(
        "unsupported",
        neutral=(False,),
        refusal="sequential beam evaluation is not implemented"),
    "max_cache_len": _Control("capacity"),
    "max_length": _Control("task"),
    "max_matching_ngram_size": _Control("unsupported", refusal="prompt lookup proposal is not implemented"),
    "max_new_tokens": _Control("task"),
    "max_time": _Control("unsupported", refusal="a host clock cannot stop a coordinated device loop"),
    "min_length": _Control("transform", neutral=(0,)),
    "min_new_tokens": _Control("transform", neutral=(0,)),
    "min_p": _Control("policy", mode="sampling", masked_neutral=(0.0,)),
    "no_repeat_ngram_size": _Control("transform", neutral=(0,)),
    "num_assistant_tokens": _Control("strategy"),
    "num_assistant_tokens_schedule": _Control(
        "unsupported",
        neutral=('constant',),
        refusal="only a constant proposal length fits a fixed device block"),
    "num_beam_groups": _Control(
        "unsupported",
        neutral=(1,),
        mode="beam",
        refusal="diverse group beam search is not implemented"),
    "num_beams": _Control("strategy", neutral=(1,)),
    "num_return_sequences": _Control("task"),
    "output_attentions": _Control(
        "unsupported",
        neutral=(False,),
        refusal="generation does not return attentions"),
    "output_hidden_states": _Control(
        "unsupported",
        neutral=(False,),
        refusal="generation does not return hidden states"),
    "output_logits": _Control(
        "unsupported",
        neutral=(False,),
        refusal="generation does not return per-step logits"),
    "output_scores": _Control(
        "unsupported",
        neutral=(False,),
        refusal="generation does not return per-step distributions"),
    "pad_token_id": _Control("task"),
    "penalty_alpha": _Control(
        "unsupported",
        neutral=(0.0,),
        refusal="contrastive search is a decoding strategy that is not implemented"),
    "prefill_chunk_size": _Control("unsupported", refusal="the native prefill evaluates a prompt in one call"),
    "prompt_lookup_num_tokens": _Control("unsupported", refusal="prompt lookup proposal is not implemented"),
    "remove_invalid_values": _Control("transform", neutral=(False,)),
    "renormalize_logits": _Control("transform", neutral=(False,)),
    "repetition_penalty": _Control("transform", neutral=(1.0,)),
    "return_dict_in_generate": _Control("inapplicable"),
    "sequence_bias": _Control("transform"),
    "speculation_type": _Control("strategy"),
    "stop_strings": _Control("criterion"),
    "suppress_tokens": _Control("transform"),
    "target_lookbehind": _Control(
        "unsupported",
        refusal="translating between two tokenizers' token spaces is not implemented"),
    "temperature": _Control("policy", masked_neutral=(1.0,)),
    "token_healing": _Control(
        "unsupported",
        neutral=(False,),
        refusal="retokenizing the prompt is prompt construction, not decoding"),
    "tokenizer_name": _Control("metadata"),
    "top_h": _Control("transform", mode="sampling"),
    "top_k": _Control("policy", masked_neutral=(0,)),
    "top_p": _Control("policy", mode="sampling", masked_neutral=(1.0,)),
    "transformers_version": _Control("metadata"),
    "typical_p": _Control("transform", neutral=(1.0,), mode="sampling"),
    "use_cache": _Control(
        "unsupported",
        neutral=(True,),
        refusal="native decoding always runs through its own cache", masked_neutral=(False,)),
    "use_mtp": _Control("strategy", neutral=(False,)),
    "watermarking_config": _Control("transform", refusal="no watermarking transform is implemented"),
}



def _neutral(value: JSON, neutral: tuple[JSON, ...]) -> bool:
    if value is None:
        return True
    # A config flag is not a config number, so 0 does not neutralize False.
    numeric = type(value) in (int, float)
    return any(value == entry and ((numeric and not isinstance(entry, bool)) or type(value) is type(entry))
               for entry in neutral)


def _active(config: Mapping[str, object], generation_config: Mapping[str, object],
            name: str, *, masked: bool = False) -> JSON:
    """The control's value when it is active, None when it changes nothing."""
    value = _generation_value(config, generation_config, name)
    rule = _CONTROLS.get(name)
    neutral = () if rule is None else rule.neutral
    if masked and rule is not None and rule.masked_neutral is not None:
        neutral = rule.masked_neutral
    return None if _neutral(value, neutral) else value


def _audit_masked(config: Mapping[str, object], generation_config: Mapping[str, object]) -> None:
    """Native MDLM has no AR policy chain or KV cache, but shares task controls."""
    refused = []
    for name in sorted(_CONTROLS.keys() | generation_config.keys()):
        rule = _CONTROLS.get(name)
        if rule is not None and rule.owner in ("task", "metadata", "inapplicable"):
            continue
        if _active(config, generation_config, name, masked=True) is not None:
            refused.append(name)
    if refused:
        raise ValueError(f"native MDLM cannot honor active source controls {refused}")


def _audit(config: Mapping[str, object], generation_config: Mapping[str, object],
           model: nn.Module, do_sample: bool, beams: JSON, overridden: bool) -> None:
    """Refuse active unsupported controls after applying the caller's override."""
    refused: list[str] = []
    for name in sorted(_CONTROLS.keys() | generation_config.keys()):
        rule = _CONTROLS.get(name)
        if rule is not None:
            if rule.owner in ("policy", "metadata", "inapplicable", "task"):
                continue
            if overridden and rule.owner == "transform":
                continue
            if rule.mode == "beam" and _neutral(beams, _CONTROLS["num_beams"].neutral):
                continue
            if rule.mode == "sampling" and not do_sample:
                continue
            if rule.owner == "capacity":
                _cache_capacity(config, generation_config, model)
                continue
        if _active(config, generation_config, name) is None:
            continue
        if rule is not None and rule.refusal is None:
            continue
        reason = rule.refusal if rule is not None else "the native decoder does not know this control"
        refused.append(f"{name} ({reason})")
    if refused:
        raise ValueError(
            f"native decoding cannot honor active source controls {refused}; "
            "text_generation(sampling=Sampling(...)) replaces the basic policy and the "
            "transform chain, and TextGeneration(model, variables, processor, logits=..., "
            "stopping=..., strategy=...) builds the task from components outright")



def _decoder(model: nn.Module) -> CausalTransformer | MultimodalTransformer | None:
    """The decoder a source built, or None for a model that is not one.

    `CausalTransformer` declares what native decoding reads off a model, and
    `MultimodalTransformer` forwards those four fields to the decoder it
    holds, so the two answer together for everything but the decoder's own.
    """
    return model if isinstance(model, CausalTransformer | MultimodalTransformer) else None


def _cache_capacity(config: Mapping[str, object], generation_config: Mapping[str, object],
                    model: nn.Module) -> None:
    """A declared cache length is real, and has to fit the model's own."""
    value = _generation_value(config, generation_config, "max_cache_len")
    if value is None:
        return
    decoder = _decoder(model)
    capacity = None if decoder is None else decoder.max_seq_len
    if type(value) is not int or value < 1:
        raise ValueError("max_cache_len must be a positive integer")
    if capacity is not None and value > capacity:
        raise ValueError(f"max_cache_len {value} exceeds the model's max_seq_len {capacity}")


def _source_sampling(config: Mapping[str, object], generation_config: Mapping[str, object],
                     do_sample: bool) -> Sampling:
    """The policy tail a source declares, whatever else it also declares."""
    temperature = _generation_value(config, generation_config, "temperature", Sampling.temperature)
    if temperature is None:
        temperature = Sampling.temperature
    if not isinstance(temperature, (float, int)) or isinstance(temperature, bool):
        raise ValueError("temperature must be numeric")
    top_k = _generation_value(config, generation_config, "top_k")
    if top_k is not None and type(top_k) is not int:
        raise ValueError("top_k must be an integer")
    return Sampling(
        temperature=float(temperature) if do_sample else 0.0,
        top_k=top_k if do_sample and top_k else None,
        eos_id=(_eos_ids(config, generation_config) or None),
        pad_id=_pad_id(config, generation_config),
        top_p=_probability_control(config, generation_config, "top_p", Sampling.top_p)
        if do_sample else Sampling.top_p,
        min_p=_probability_control(config, generation_config, "min_p", Sampling.min_p)
        if do_sample else Sampling.min_p)


def _token_list(value: object, name: str) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list of token ids")
    ids = []
    for token in value:
        if type(token) is not int or token < 0:
            raise ValueError(f"{name} must hold non-negative integer token ids")
        ids.append(token)
    return ids


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _as_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _as_decay(value: object) -> tuple[int, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("exponential_decay_length_penalty must be (start_index, factor)")
    return (_as_int("exponential_decay_length_penalty start", value[0]),
            _as_float("exponential_decay_length_penalty factor", value[1]))


def _as_bias(value: object) -> list[tuple[list[int], float]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("sequence_bias must be a non-empty list of token ids and bias pairs")
    entries = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("each sequence_bias entry is a token id list and a bias")
        entries.append((_token_list(entry[0], "sequence_bias"), _as_float("sequence_bias", entry[1])))
    return entries


def _as_words(value: object) -> list[list[int]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("bad_words_ids must be a non-empty list of token id lists")
    return [_token_list(word, "bad_words_ids") for word in value]


def _as_strings(value: object) -> tuple[str, ...]:
    strings = (value,) if isinstance(value, str) else value
    if not isinstance(strings, (list, tuple)) or not strings:
        raise ValueError("stop_strings must be a string or a non-empty list of strings")
    for entry in strings:
        if not isinstance(entry, str) or not entry:
            raise ValueError("stop_strings must hold non-empty strings")
    return tuple(entry for entry in strings if isinstance(entry, str))


def _source_transforms(config: Mapping[str, object], generation_config: Mapping[str, object],
                       sampling: Sampling, do_sample: bool,
                       searching: bool) -> tuple[decoding.LogitsTransform, ...]:
    """The source's whole transform chain, in `_get_logits_processor`'s order.

    This is the complete chain the task runs, so the policy's own tail is
    built here rather than appended afterwards and every warper lands where
    the reference puts it: temperature, top-h, top-k, top-p, min-p, typical,
    epsilon, eta, and `renormalize_logits` last of all. Without sampling the
    reference adds no warper at all and picks the argmax, which is the
    trailing `Greedy`. Beam search picks its own continuations, so it ends
    the chain after the processors.
    """
    eos = jnp.asarray(_eos_ids(config, generation_config) or (), jnp.int32)
    read = functools.partial(_active, config, generation_config)
    transforms: list[decoding.LogitsTransform] = []
    if (value := read("sequence_bias")) is not None:
        transforms.append(decoding.sequence_bias(_as_bias(value)))
    if (value := read("encoder_repetition_penalty")) is not None:
        transforms.append(decoding.PromptRepetitionPenalty(_as_float("encoder_repetition_penalty", value)))
    if (value := read("repetition_penalty")) is not None:
        transforms.append(decoding.RepetitionPenalty(_as_float("repetition_penalty", value)))
    if (value := read("no_repeat_ngram_size")) is not None:
        transforms.append(decoding.NoRepeatNGram(_as_int("no_repeat_ngram_size", value)))
    if (value := read("encoder_no_repeat_ngram_size")) is not None:
        transforms.append(decoding.PromptNoRepeatNGram(_as_int("encoder_no_repeat_ngram_size", value)))
    if (value := read("bad_words_ids")) is not None:
        transforms.append(decoding.bad_words(_as_words(value), sampling.eos_id))
    if (value := read("min_length")) is not None and eos.size:
        transforms.append(decoding.MinLength(_as_int("min_length", value), eos))
    if (value := read("min_new_tokens")) is not None and eos.size:
        transforms.append(decoding.MinNewTokens(_as_int("min_new_tokens", value), eos))
    if (value := read("forced_bos_token_id")) is not None:
        transforms.append(decoding.ForcedBOS(_as_int("forced_bos_token_id", value)))
    if (value := read("forced_eos_token_id")) is not None:
        # The reference forces at the effective end of the request, and a call
        # may set its own budget, so the control stays request relative.
        transforms.append(decoding.ForcedEOS(
            jnp.asarray(_token_list(value, "forced_eos_token_id"), jnp.int32)))
    if read("remove_invalid_values") is not None:
        transforms.append(decoding.RemoveInvalidValues())
    if (value := read("exponential_decay_length_penalty")) is not None:
        start, factor = _as_decay(value)
        transforms.append(decoding.ExponentialDecayLengthPenalty(start, factor, eos))
    if (value := read("suppress_tokens")) is not None:
        transforms.append(decoding.SuppressTokens(
            jnp.asarray(_token_list(value, "suppress_tokens"), jnp.int32)))
    if (value := read("begin_suppress_tokens")) is not None:
        transforms.append(decoding.BeginSuppressTokens(
            jnp.asarray(_token_list(value, "begin_suppress_tokens"), jnp.int32),
            read("forced_bos_token_id") is not None))
    if searching:
        if read("renormalize_logits") is not None:
            transforms.append(decoding.Renormalize())
        return tuple(transforms)
    if not do_sample:
        transforms.append(decoding.Greedy())
    else:
        if sampling.temperature != 1.0:
            transforms.append(decoding.Temperature(sampling.temperature))
        if (value := read("top_h")) is not None:
            transforms.append(decoding.TopH(_as_float("top_h", value)))
        if sampling.top_k is not None:
            transforms.append(decoding.TopK(sampling.top_k))
        if sampling.top_p < 1.0:
            transforms.append(decoding.TopP(sampling.top_p))
        if sampling.min_p > 0.0:
            transforms.append(decoding.MinP(sampling.min_p))
        if (value := read("typical_p")) is not None:
            transforms.append(decoding.Typical(_as_float("typical_p", value)))
        if (value := read("epsilon_cutoff")) is not None:
            transforms.append(decoding.EpsilonCutoff(_as_float("epsilon_cutoff", value)))
        if (value := read("eta_cutoff")) is not None:
            transforms.append(decoding.EtaCutoff(_as_float("eta_cutoff", value)))
    if read("renormalize_logits") is not None:
        transforms.append(decoding.Renormalize())
    return tuple(transforms)


def _source_stopping(config: Mapping[str, object], generation_config: Mapping[str, object],
                     processor: Processor | None, vocab_size: int | None
                     ) -> tuple[decoding.Stopping, ...]:
    """The source's active criteria beyond the policy's EOS ids."""
    value = _active(config, generation_config, "stop_strings")
    if value is None:
        return ()
    if processor is None:
        raise ValueError("stop_strings need the source's processor to compile its vocabulary")
    if vocab_size is None:
        raise ValueError("stop_strings need the model's vocab_size to compile its vocabulary")
    return (decoding.stop_strings(processor, _as_strings(value), vocab_size),)


def _source_strategy(config: Mapping[str, object], generation_config: Mapping[str, object],
                     model: nn.Module, do_sample: bool, rows: int) -> Strategy | None:
    """The device loop a source's config names, or None for plain sampling."""
    read = functools.partial(_active, config, generation_config)
    beams = read("num_beams")
    speculating = read("use_mtp") is not None or _mtp_mode(read("speculation_type"))
    if beams is not None and speculating:
        raise ValueError("a source cannot ask for beam search and speculative decoding at once")
    if beams is not None:
        if do_sample:
            raise ValueError("stochastic beam search is refused: a selected beam's marginal "
                             "probability is not the per-step candidate probability, so no honest "
                             "behaviour likelihood exists")
        width = _as_int("num_beams", beams)
        if rows > width:
            raise ValueError(f"num_return_sequences {rows} exceeds num_beams {width}")
        early = _generation_value(config, generation_config, "early_stopping")
        penalty = _generation_value(config, generation_config, "length_penalty")
        if early is None:
            early = Beam.early_stopping
        if early not in (True, False, "never"):
            raise ValueError("early_stopping is True, False or 'never'")
        return Beam(width=width,
                    length_penalty=Beam.length_penalty if penalty is None else _as_float("length_penalty", penalty),
                    early_stopping=early is True if isinstance(early, bool) else "never",
                    stop_ids=len(_eos_ids(config, generation_config)))
    if not speculating:
        return None
    decoder = _decoder(model)
    if decoder is None or not decoder.num_nextn_predict_layers:
        raise ValueError("the source asks for multi-token-prediction speculation, but this "
                         "checkpoint carries no prediction-depth weights")
    length = read("num_assistant_tokens")
    threshold = read("assistant_confidence_threshold")
    drafted = Speculative.block - 1 if length is None else _as_int("num_assistant_tokens", length)
    if drafted < 1:
        raise ValueError("num_assistant_tokens must draft at least one token")
    # The block includes the target draw the proposer chains from.
    return Speculative(block=drafted + 1,
                       confidence=Speculative.confidence if threshold is None else
                       _as_float("assistant_confidence_threshold", threshold))


def _mtp_mode(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, str) or value.lower() not in ("mtp", "multi_token_prediction"):
        raise ValueError(f"speculation_type {value!r} names no native proposer; only the model's "
                         "own prediction depths draft natively")
    return True


def _source_decoding(config: Mapping[str, object], generation_config: Mapping[str, object],
                     model: nn.Module, processor: Processor | None, rows: int,
                     override: Sampling | None
                     ) -> tuple[Sampling, tuple[decoding.LogitsTransform, ...] | None,
                                tuple[decoding.Stopping, ...], Strategy | None]:
    """The policy, chain, criteria and strategy a loaded source decodes with.

    An explicit policy replaces the first two, so they are not built and the
    controls behind them are not judged: a watermark the caller just replaced
    cannot block the call.
    """
    requested_mode = _generation_value(config, generation_config, "do_sample")
    if override is None and requested_mode is not None and type(requested_mode) is not bool:
        raise ValueError("do_sample must be a boolean")
    do_sample = requested_mode is True
    _audit(config, generation_config, model, do_sample,
           _generation_value(config, generation_config, "num_beams"), override is not None)
    strategy = _source_strategy(config, generation_config, model, do_sample, rows)
    decoder = _decoder(model)
    criteria = _source_stopping(config, generation_config, processor,
                                None if decoder is None else decoder.vocab_size)
    policy = override if override is not None else _source_sampling(config, generation_config, do_sample)
    transforms = (None if override is not None else
                  _source_transforms(config, generation_config, policy, do_sample, isinstance(strategy, Beam)))
    return policy, transforms, criteria, strategy


class _Call(NamedTuple):
    """One pinned pipeline's own `__call__` policy, read from Diffusers
    0.34.0: the family it belongs to, the steps and guidance scale it
    defaults to, whether that scale guides two branches or is the value the
    model embeds, and the text sequence budget it pads its T5 tower to."""

    family: Literal["sd", "sdxl", "sd3", "flux"]
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
    "FlaxStableDiffusionPipeline": _Call("sd", 50, 7.5, guided=True),
    "FlaxStableDiffusionImg2ImgPipeline": _Call("sd", 50, 7.5, guided=True),
    "FlaxStableDiffusionInpaintPipeline": _Call("sd", 50, 7.5, guided=True),
    "FlaxStableDiffusionXLPipeline": _Call("sdxl", 50, 7.5, guided=True),
})


def _call_policy(index: Mapping[str, object], denoiser: _Denoiser) -> _Call:
    """The call policy this file's own pipeline carries.

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
    return found


@dataclass(frozen=True)
class SourceTask:
    """A published pipeline's own call policy.

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
class _Denoiser:
    """What one architecture contributes to a diffusion source.

    Model construction and conditioning conventions use metadata only.
    The weight reader is invoked only by a complete source load; a restored
    conditioner can reuse the same architecture metadata without reading
    denoiser or autoencoder weights.

    """

    component: str
    model: nn.Module
    weights: Callable[[str], tuple[Variables, tuple[WeightLayout, ...]]]
    built: Mapping[str, object]
    config: Mapping[str, object]
    composition: Composition
    towers: tuple[str, ...]
    patch: int
    latent_input: int
    sample_size: int
    context_width: int
    pipeline: str
    origin: Origin = "scheduler"
    embeds_guidance: bool = False
    t5_tower: str | None = None


def _load_diffusion_source(directory: Path, index: Mapping[str, object], *, dtype: str,
                           attention_impl: str, param_dtype: str = "float32") -> Pretrained:
    """A published latent diffusion directory as native modules and variables.

    Two denoiser families ship this layout: a UNet reading one or two CLIP
    towers through cross attention, and an MM-DiT transformer reading them
    jointly beside a T5 tower. The directory's own denoiser component selects
    the family, and everything the families share - the autoencoder, the text
    towers, the geometry, the conditioning, the safety head a file declares,
    the schedule and the call policy - is read once here.
    """
    compute = resolve_dtype(dtype)
    denoiser = (_transformer_denoiser if (directory / "transformer" / "config.json").is_file()
                else _unet_denoiser)(directory, dtype=dtype, attention_impl=attention_impl)
    denoiser_variables, denoiser_layouts = denoiser.weights(param_dtype)
    policy = _call_policy(index, denoiser)
    autoencoder, vae_params, vae_layouts, vae_config = _diffusion_vae(directory, compute, param_dtype=param_dtype)
    encoder, text_layouts, components = _conditioning(
        directory, index, denoiser, policy, compute,
        denoiser.sample_size * autoencoder.downscale_factor, param_dtype=param_dtype)
    components.update({denoiser.component: denoiser.config, "vae": vae_config})
    height, width = encoder.height, encoder.width
    inpaint = denoiser.latent_input == autoencoder.latent_channels * 2 + 1
    inputs = InputSpec(Field("image", (height, width, 3)),
                       {"conditioning": Condition(encoder, unconditional=_unconditional(
                           denoiser.composition, index))},
                       mask=Field("mask", (height, width, 1)) if inpaint else None)
    encoders: dict[str, object] = {"conditioning": encoder.params}
    finish, safety_layouts = None, ()
    if _present(index, "safety_checker"):
        finish, encoders["safety"], safety_layouts, safety_configs = _image_safety(
            directory, compute, param_dtype=param_dtype)
        components.update(safety_configs)
    schedule = SourceSchedule.from_config(_component_config(directory, "scheduler"))
    components["scheduler"] = dict(schedule.config)
    patch = denoiser.patch * autoencoder.downscale_factor
    task = SourceTask(min(policy.steps, schedule.train_steps),
                      CFG(policy.guidance) if policy.guided and policy.guidance > 1 else None,
                      functools.partial(schedule.sampling, origin=denoiser.origin,
                                        tokens=(height // patch) * (width // patch)))
    variables = {**denoiser_variables, "encoders": encoders, "autoencoder": vae_params}
    config = {"model_index": {**index, "dew_height": height, "dew_width": width}, **components}
    return Pretrained(denoiser.model, variables, None, config, directory, denoiser.built,
                      weight_layouts=denoiser_layouts + vae_layouts + text_layouts + safety_layouts,
                      process=schedule.training_process(), inputs=inputs, autoencoder=autoencoder,
                      schedule=schedule, finish=finish, task=task)


def _conditioning(directory: Path, index: Mapping[str, object], denoiser: _Denoiser,
                  policy: _Call, compute, size: int, *, param_dtype: str,
                  params: Variables | None = None):
    """Construct the published text composition from metadata and either weight source."""
    names = tuple(name for name in denoiser.towers if _present(index, name))
    if not names:
        raise ValueError("A latent diffusion source needs at least one text encoder")
    towers, tokenizers, text_params, layouts = _clip_towers(
        directory, names, compute, param_dtype=param_dtype, params=params)
    components: dict[str, Mapping[str, object]] = {name: _component_config(directory, name) for name in names}
    t5 = None
    if denoiser.t5_tower is not None and _present(index, denoiser.t5_tower):
        t5, t5_params, t5_layouts, components[denoiser.t5_tower] = _t5_tower(
            directory, compute, denoiser.t5_tower, policy.sequence, param_dtype=param_dtype,
            params=None if params is None else params[denoiser.t5_tower])
        if params is None:
            text_params = {**text_params, denoiser.t5_tower: t5_params}
        layouts += t5_layouts
    height, width = index.get("dew_height", size), index.get("dew_width", size)
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError("Image geometry must contain positive integer dimensions")
    encoder = DiffusionConditioner(
        towers, tokenizers, names, text_params, str(directory), height, width,
        denoiser.context_width, composition=denoiser.composition, t5=t5,
        guidance=policy.guidance if denoiser.embeds_guidance and not policy.guided else None,
        aesthetics=bool(index.get("requires_aesthetics_score", False)), param_dtype=param_dtype)
    return encoder, layouts, components


def load_diffusion_conditioner(checkpoint: str, *, dtype: str | None = "bfloat16",
                               param_dtype: str = "float32", revision: str | None = None,
                               attention_impl: str = "auto", params: Variables | None = None
                               ) -> DiffusionConditioner:
    """Load conditioning weights, or bind supplied parameters using metadata only."""
    from dew.nn.autoencoders import AutoencoderKL

    compute = resolve_dtype(dtype)
    resolve_dtype(param_dtype)
    directory = decoders._snapshot(checkpoint, revision, weights=False)
    with open(directory / "model_index.json") as handle:
        index = json.load(handle)
    denoiser = (_transformer_denoiser if (directory / "transformer" / "config.json").is_file()
                else _unet_denoiser)(directory, dtype=dtype, attention_impl=attention_impl)
    if params is None:
        names = tuple(name for name in (*denoiser.towers, denoiser.t5_tower)
                      if name is not None and _present(index, name))
        # snapshot_download returns a commit directory. Keep both fetches on
        # that commit even when the requested Hub branch moves between them.
        directory = decoders._snapshot(checkpoint, directory.name, weights=names)
    vae = AutoencoderKL(channels=tuple(_component_config(directory, "vae")["block_out_channels"]))
    encoder, _, _ = _conditioning(
        directory, index, denoiser, _call_policy(index, denoiser), compute,
        denoiser.sample_size * vae.downscale_factor, param_dtype=param_dtype, params=params)
    return encoder


def _unet_denoiser(directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """The published UNet: cross attention over one or two CLIP towers, whose
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
        built=built, config=config, composition="clip_pooled" if pooled else "clip",
        towers=("text_encoder", "text_encoder_2"), patch=1, latent_input=model.in_channels,
        sample_size=records.integer(config["sample_size"], "sample_size"),
        context_width=records.integer(config.get("cross_attention_dim", 1280), "cross_attention_dim"),
        pipeline="StableDiffusionXLPipeline" if pooled else "StableDiffusionPipeline")


def _transformer_denoiser(directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """The published transformer this directory holds, by the class it names."""
    config = _component_config(directory, "transformer")
    published = config.get("_class_name")
    if published == "SD3Transformer2DModel":
        return _sd3_denoiser(config, directory, dtype=dtype, attention_impl=attention_impl)
    if published == "FluxTransformer2DModel":
        return _flux_denoiser(config, directory, dtype=dtype, attention_impl=attention_impl)
    raise ValueError(f"Native diffusion does not implement the published transformer "
                     f"{published!r}")


def _sd3_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """SD3's MM-DiT: both CLIP towers and the T5 tower read jointly, with the
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
        built=built, config=config, composition="sd3",
        towers=("text_encoder", "text_encoder_2"), patch=fields["patch_size"],
        latent_input=fields["in_channels"], sample_size=records.integer(config["sample_size"], "sample_size"),
        context_width=fields["joint_attention_dim"], pipeline="StableDiffusion3Pipeline",
        t5_tower="text_encoder_3")


def _flux_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Flux's transformer: one CLIP tower for the pooled vector, the T5 tower
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
        built=built, config=config, composition="flux", towers=("text_encoder",), patch=2,
        latent_input=fields["in_channels"] // 4,
        sample_size=records.integer(config.get("sample_size", 128), "sample_size"),
        context_width=fields["joint_attention_dim"], pipeline="FluxPipeline",
        origin="linspace", embeds_guidance=fields["guidance_embeds"], t5_tower="text_encoder_2")


def _component_config(directory: Path, name: str) -> dict:
    """One published component's own config file."""
    file = "scheduler_config.json" if name == "scheduler" else "config.json"
    with open(directory / name / file) as handle:
        return json.load(handle)


def _present(index: Mapping[str, object], name: str) -> bool:
    """Whether the index declares a component rather than declaring it absent."""
    entry = index.get(name)
    return isinstance(entry, list) and entry[0] is not None


def _diffusion_vae(directory: Path, compute, *, param_dtype: str = "float32"
                   ) -> tuple[StableDiffusionVAE, Variables, tuple[WeightLayout, ...], dict]:
    """The published autoencoder, its parameters and their source layouts."""
    from dew.interop import diffusion
    from dew.nn.autoencoders import AutoencoderKL, StableDiffusionVAE
    from dew.nn.autoencoders.vae import _vae_path

    config = _component_config(directory, "vae")
    model = AutoencoderKL(
        channels=tuple(config["block_out_channels"]), latent_channels=config["latent_channels"],
        image_channels=config["in_channels"], blocks_per_level=config["layers_per_block"],
        norm_groups=config["norm_num_groups"], quantize=config.get("use_quant_conv", True),
        post_quantize=config.get("use_post_quant_conv", True), dtype=compute)
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
    """The published CLIP text towers, their tokenizers, their parameters and
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
    """The published T5 encoder as the conditioner's segment, with its
    parameters, their layouts and its config.

    `component` is where the family keeps it: an SD3 directory's third text
    encoder, a Flux directory's second one; `tokens` is the sequence budget
    the pipeline pads to.
    """
    from transformers import AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(
        directory / ("tokenizer" + component.removeprefix("text_encoder")))
    return T5Segment(tower, tokenizer, component, tokens), params, layouts, config


def _unconditional(composition: str, index: Mapping[str, object]) -> dict:
    """The empty-prompt row a file's own pipeline guides against: the XL
    pipelines zero it where their index says so, and the SD3 pipeline encodes
    it with its towers, having no such control."""
    zero = composition == "clip_pooled" and bool(index.get("force_zeros_for_empty_prompt", True))
    return {"text": "", "negative": True, "zero": zero}


def _image_safety(directory: Path, compute, *, param_dtype: str = "float32"):
    """The safety head a file declares: the finish, its parameters, their
    layouts and the two configs it ships."""
    from dew.inputs.diffusion import CLIPImageTransform, CLIPSafetyHead, ImageSafety
    from dew.interop import diffusion
    from dew.nn.text_encoders import CLIPVisionTransformer, translate_vision_config

    config = _component_config(directory, "safety_checker")
    with open(directory / "feature_extractor" / "preprocessor_config.json") as handle:
        transform = json.load(handle)
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


def load_pretrained(name_or_dir: str | Path, *, dtype: str = "bfloat16", param_dtype: str = "float32",
                    attention_impl: str = "auto", max_seq_len: int | None = None,
                    revision: str | None = None) -> Pretrained:
    """Load a source into a native Flax model with explicit parameter trees.

    ``name_or_dir`` is a local HF directory or a Hub model identifier. The
    decoder/tower/projector maps preserve their established internal paths;
    wrapper variables join under their existing component names. Processor
    artifacts are loaded only when the source contains them.
    dtype selects computation; param_dtype independently selects floating
    parameter storage and defaults to FP32 masters. Frozen component weights
    (text encoders and VAE) follow it too; router, clipping, positional and
    safety state retain their own FP32/integer contracts.
    """
    storage = dtype_name(resolve_dtype(param_dtype))
    if storage is None:
        raise ValueError("param_dtype must select floating parameter storage")
    param_dtype = storage
    directory = decoders._snapshot(str(name_or_dir), revision)
    if (directory / "model_index.json").is_file() and not (directory / "config.json").is_file():
        # A latent diffusion pipeline is a directory of components with no
        # model of its own; a decoder that also ships a pipeline index for its
        # sampler (DiffusionGemma) is loaded as the decoder its config names.
        with open(directory / "model_index.json") as handle:
            return _load_diffusion_source(directory, json.load(handle), dtype=dtype,
                                          attention_impl=attention_impl, param_dtype=param_dtype)
    with open(directory / "config.json") as handle:
        config = json.load(handle)
    text_config = config.get("text_config")
    if (config.get("model_type") == "kimi_k25" and isinstance(text_config, Mapping)
            and text_config.get("quantization_config") is not None):
        raise ValueError(
            "text_config.quantization_config is not supported for the kimi_k25 text-only loader; "
            "provide dequantized text weights and remove that quantization descriptor")
    tensors = decoders._load_shards(directory)
    family = config.get("model_type")
    quantization = _source_quantization(config, param_dtype=param_dtype)
    quantized_tensors = () if quantization is None else quantization.names(tensors)
    if quantization is not None:
        aliases: tuple[tuple[str, str], ...] = ()
        if param_dtype != "float32":
            aliases = decoders.validate_source_aliases(
                quantization.tensor_names(tensors), partial(quantization.read, tensors), config)
        tensors = quantization.dequantize(tensors)
        _share_quantized_aliases(tensors, aliases, quantized_tensors)
    layouts: tuple[WeightLayout, ...] = ()
    retained: dict[str, np.ndarray] = {}
    export_adapter = None
    if family == "diffusion_gemma":
        from dew.interop import diffusion_gemma
        model = diffusion_gemma.build(config, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len)
        variables = diffusion_gemma.translate_weights(tensors, config, param_dtype=param_dtype)
        record = config
        built: Mapping[str, object] = {**config, "dtype": dtype, "attention_impl": attention_impl}
        export_adapter = diffusion_gemma.export_weights
    elif "text_config" in config and family not in decoders._FAMILIES:
        # A wrapper repo carries its decoder under text_config. Where the
        # wrapper's own model_type is a registered decoder family, its
        # towers have no counterpart and its text half is the model, so it
        # takes the decoder branch below and its translator reads the
        # nested config; `translate_config` refuses the rest by the same
        # rule.
        record = decoders.translate_wrapper_config(config)
        text_fields: decoders.DecoderFields = {**record["text"]}
        if max_seq_len is not None:
            text_fields["max_seq_len"] = max_seq_len
        if family == "gemma3":
            # Gemma3ForConditionalGeneration projects logits without the
            # causal-LM class's optional final tanh cap.
            text_fields["final_logit_softcap"] = None
            text_fields["mixer"] = {"kind": "attention", "bidirectional_images": True}
        if family == "gemma4" and config["text_config"].get("use_bidirectional_attention") == "vision":
            kinds = dict(text_fields.get("kinds") or {})
            sliding: decoders.KindFields = {
                **kinds.get("sliding_attention", {}),
                "mixer": {"kind": "attention", "bidirectional_images": True}}
            kinds["sliding_attention"] = sliding
            text_fields["kinds"] = kinds
        if family == "qwen3_5":
            rope = config["text_config"].get("rope_parameters") or {}
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

        text: decoders.DecoderFields = {**text_fields, **precision_fields(
            "causal_transformer", text_fields, dtype=dtype, attention_impl=attention_impl)}
        wrapper: decoders.WrapperFields = {**record, "text": text}
        built = wrapper
        language_model = models.build("causal_transformer", wrapper["text"])
        if not isinstance(language_model, CausalTransformer):
            raise TypeError("causal_transformer registry entry must build CausalTransformer")
        audio_record = record["audio"]
        audio_projector = record["audio_projector"]
        model = MultimodalTransformer(
            language_model, tower_from_record(record["tower"]),
            projector_from_record(record["projector"]), family,
            record["image_token_id"], dtype=resolve_dtype(dtype),
            pad_token_id=config["text_config"].get("pad_token_id", 0),
            extra_placeholder_ids=(tuple(config.get(name, default) for name, default in
                (("video_token_id", 258884), ("audio_token_id", 258881))) if family == "gemma4" else ()),
            audio=None if audio_record is None else tower_from_record(audio_record),
            audio_projection=(None if audio_projector is None
                              else projector_from_record(audio_projector)),
            audio_soft_tokens=record["audio_soft_tokens"],
            attention_impl=attention_impl)
        variables = _native_variables(decoders.translate_wrapper_weights(tensors, record, param_dtype=param_dtype))
        layouts, retained = _wrapper_layouts(tensors, record, variables)
    else:
        record = decoders.translate_config(config)
        if max_seq_len is not None:
            record["max_seq_len"] = max_seq_len
        built = with_precision("causal_transformer", record, dtype=dtype, attention_impl=attention_impl)
        model = models.build("causal_transformer", built)
        variables = decoders.translate_weights(tensors, record, family, param_dtype=param_dtype)
        decoders._check_tree(variables, model)
        # The bindings are what an adapter loader resolves source names
        # through and what a quantized source is written back through, so a
        # derived-export family binds too; `save` picks its writer by
        # preserve_source_layout and quantization, not by whether bindings
        # exist. A family whose tensors are rewritten before the path map
        # reads them (Gemma 4's prepare) has no raw-name bindings.
        entry = decoders._FAMILIES[family]
        if entry.preserve_source_layout or entry.prepare_weights is dict:
            bindings = []
            for name, tensor in tensors.items():
                layout = _language_layout(name, name, tensor, record, family, variables)
                if layout is None:
                    retained[name] = tensor
                else:
                    bindings.append(layout)
            layouts = tuple(bindings)
    processor = None
    processor_files = any((directory / name).exists()
                          for name in ("processor_config.json", "preprocessor_config.json"))
    if isinstance(model, MultimodalTransformer) and processor_files:
        # Only a model with towers reads images or audio, and only through
        # the processor its repo ships; a text decoder takes its tokenizer
        # whatever processor files sit beside it (DiffusionGemma publishes a
        # Gemma 4 processor config beside a text-only model), and a tiny
        # multimodal fixture without processor files tokenizes text only.
        from transformers import AutoProcessor
        options = {"backend": "pil"} if family == "gemma3" else {}
        reference = AutoProcessor.from_pretrained(str(directory), local_files_only=True, **options)
        processor = Processor(reference, config, record, model.vocab_size)
    elif (directory / "tokenizer_config.json").exists():
        from transformers import AutoTokenizer
        reference = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        processor = Processor(reference, config, record, model.vocab_size)
    generation_path = directory / "generation_config.json"
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    error = None
    try:
        # Loading for training/export does not opt into the source sampler.
        # Active policy support is checked when the caller creates its task.
        if not isinstance(generation_config, dict):
            raise ValueError("generation_config.json must contain an object")
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase="pretrained generation policy")
    return Pretrained(model, variables, processor, config, directory, built, generation_config,
                      layouts, retained, export_adapter, quantized_tensors=quantized_tensors)
