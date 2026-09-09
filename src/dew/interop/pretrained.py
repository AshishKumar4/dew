"""Load native models and their host processors from a Hugging Face source."""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.artifacts import agree_process_phase
from dew.interop import hf_decoders as decoders
from dew.interop.quantized import dequantize_checkpoint, fp8_format, pack_fp8, scaled_names
from dew.inference import BlockGeneration, TextGeneration
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.gpt_oss import mxfp4_stems, pack_mxfp4, unpack_mxfp4
from dew.sampling import decoding
from dew.sampling.strategies import Beam, Speculative, Strategy
from dew.sampling.text import Sampling
from dew.nn import audio as audio_nn
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs, pad_token_rows
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.mixers.attention import AttentionMixer
from dew.nn.vision import projector_from_record, tower_from_record
from dew.objectives.base import Variables
from dew.registry import models, resolve_dtype, with_precision
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import Origin, SourceSchedule
from dew.inputs import Condition, Field, InputSpec
from dew.inputs.diffusion import Composition, DiffusionConditioner, T5Segment
from dew.nn.autoencoders import AutoEncoder, StableDiffusionVAE
from dew.objectives.diffusion import DiffusionObjective
from dew.sampling.guidance import CFG
from dew.sampling.pipelines import TextToImage


class HostProcessor(Protocol):
    """The HF processor operations kept outside compiled model computation."""

    def __call__(self, **kwargs: object) -> Mapping[str, object]: ...
    def save_pretrained(self, save_directory: str) -> object: ...
    def apply_chat_template(self, conversation: Sequence[Mapping[str, object]], **kwargs: object) -> object: ...
    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]: ...


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

    def __call__(self, text: str | Sequence[str], *, images: object | None = None,
                 audio: object | None = None, videos: object | None = None,
                 video_metadata: object | None = None) -> ModelInputs:
        if images is None and audio is None and videos is None:
            if video_metadata is not None:
                raise ValueError("video_metadata requires videos")
            rows = [text] if isinstance(text, str) else list(text)
            values = self.reference(text=rows, padding=False, truncation=False, return_tensors=None)
            tokenizer = getattr(self.reference, "tokenizer", self.reference)
            pad_id = getattr(tokenizer, "pad_token_id", None)
            side = getattr(tokenizer, "padding_side", "right")
            if side not in ("left", "right"):
                raise ValueError("the tokenizer padding_side must be left or right")
            ids = values["input_ids"]
            ids = ids if isinstance(ids, Sequence) else np.asarray(ids)
            fields = {name: value if isinstance(value, Sequence) else np.asarray(value)
                      for name, value in values.items() if name != "input_ids"}
            tokens, fields = pad_token_rows(ids, pad_id=0 if pad_id is None else pad_id,
                                           padding_side=side, fields=fields)
            return self.from_hf({"input_ids": tokens, **fields})
        # truncation is off for text anyway; reloaded Gemma processors forward
        # the tokenizer's unset max_length into audio kwargs otherwise.
        arguments: dict[str, object] = {
            "text": text if isinstance(text, str) else list(text),
            "padding": not isinstance(text, str) and len(text) > 1, "truncation": False, "return_tensors": "pt"}
        if images is not None:
            arguments["images"] = images
        if audio is not None:
            arguments["audio"] = audio
        if videos is not None:
            arguments["videos"] = videos
        if video_metadata is not None:
            if videos is None:
                raise ValueError("video_metadata requires videos")
            arguments["video_metadata"] = video_metadata
        return self._from_tensors(self.reference(**arguments))

    def chat(self, messages: Sequence[Mapping[str, object]], *, add_generation_prompt: bool = True,
             **template_options: object) -> ModelInputs:
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
                 "pixel_values_videos", "video_grid_thw"}
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
        conditioning: dict[str, jax.Array] = {}
        if "pixel_values" in values or "pixel_values_videos" in values:
            if "pixel_values_videos" in values and self.config.get("model_type") != "qwen3_5":
                raise ValueError("video patch inputs require the Qwen3.5 visual tower")
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
        result = ModelInputs(jnp.asarray(tokens, jnp.int32), token_fields, conditioning)
        result.validate()
        return result

    def _images(self, values: Mapping[str, object], tokens: np.ndarray
                ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        image_id = self.record.get("image_token_id", self.config.get("image_token_id"))
        if type(image_id) is not int:
            raise ValueError("image_token_id must be an integer")
        qwen = self.config.get("model_type") == "qwen3_5"
        video_id = self.config.get("video_token_id") if qwen else None
        pixels = np.asarray(values.get("pixel_values", values.get("pixel_values_videos")))
        if pixels.ndim not in (2, 3, 4) or not np.issubdtype(pixels.dtype, np.floating):
            raise ValueError("pixel_values must contain floating image tensors or patch vectors")
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
        elif "image_position_ids" in values:
            vision = self.config.get("vision_config")
            if not isinstance(vision, Mapping):
                raise ValueError("image_position_ids require a vision config")
            kernel = vision.get("pooling_kernel_size")
            if type(kernel) is not int or kernel < 1:
                raise ValueError("pooling_kernel_size must be a positive integer")
            patch_positions = np.asarray(values["image_position_ids"])
            if (pixels.ndim != 3 or patch_positions.shape != pixels.shape[:2] + (2,)
                    or not np.issubdtype(patch_positions.dtype, np.integer)):
                raise ValueError("patch pixels require aligned integer image_position_ids")
            valid_patches = (patch_positions >= 0).all(axis=-1)
            padding = (patch_positions == -1).all(axis=-1)
            if not np.all(valid_patches | padding):
                raise ValueError("image_position_ids use nonnegative coordinates or paired (-1, -1) padding")
            table_size = vision.get("position_embedding_size")
            if type(table_size) is not int or np.any(patch_positions >= table_size):
                raise ValueError("image_position_ids exceed position_embedding_size")
            counts = valid_patches.sum(axis=1)
            if pixels.shape[1] % kernel ** 2 or np.any(counts % kernel ** 2):
                raise ValueError("patch counts must divide into complete pooling blocks")
            lengths = counts // kernel ** 2
            capacity = pixels.shape[1] // kernel ** 2
            chunks, shape = list(pixels), pixels.shape[1:]
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
        padded = np.zeros((tokens.shape[0], width, *shape), pixels.dtype)
        padded_grid = None if grid is None else np.zeros((tokens.shape[0], width, 3), np.int32)
        if padded_grid is not None:
            padded_grid[..., 1:] = merge
        padded_positions = (None if patch_positions is None else np.full(
            (tokens.shape[0], width, *patch_positions.shape[1:]), -1, np.int32))
        indices = np.full(tokens.shape, -1, np.int32)
        groups = np.full(tokens.shape, -1, np.int32)
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
                    padded_positions[row, image] = patch_positions[offset]
                token_lengths[row, image] = length
                indices[row, slots] = image * capacity + np.arange(length)
                groups[row, slots] = image
                offset += 1
        conditioning = {"pixel_values": jnp.asarray(padded), "image_lengths": jnp.asarray(image_counts),
                        "image_token_lengths": jnp.asarray(token_lengths)}
        if padded_positions is not None:
            conditioning["image_position_ids"] = jnp.asarray(padded_positions)
        if padded_grid is not None:
            conditioning["image_grid_thw"] = jnp.asarray(padded_grid)
        return {"image_indices": jnp.asarray(indices), "image_groups": jnp.asarray(groups)}, conditioning

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
            items = []
            offset = 0
            for time, height, width in grid:
                count = int(height * width)
                for _ in range(int(time) if split else 1):
                    frames = 1 if split else int(time)
                    items.append((pixels[offset:offset + frames * count], (frames, int(height), int(width))))
                    offset += frames * count
            token_id = self.config[token_name]
            if type(token_id) is not int:
                raise ValueError(f"{token_name} must be an integer")
            streams[token_id] = items
        offsets = dict.fromkeys(streams, 0)
        ordered = []
        for row, spans in enumerate(runs):
            for span in spans:
                token = int(tokens[row, span[0]])
                if token not in streams or offsets[token] == len(streams[token]):
                    raise ValueError("visual placeholders exceed their image/video frame payloads")
                ordered.append(streams[token][offsets[token]])
                offsets[token] += 1
        if not ordered or any(offsets[token] != len(items) for token, items in streams.items()):
            raise ValueError("visual payloads and placeholder frame counts disagree")
        chunks, grids = zip(*ordered)
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
        result = np.zeros((*tokens.shape, 3), np.int32)
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
                    result[row, span] = (cursor + np.arange(len(span)))[:, None]
                    cursor += len(span)
                else:
                    time, height, width = (int(value) for value in grid_values[row, group])
                    coordinates = np.indices((time, height // merge, width // merge)).reshape(3, -1).T
                    if len(span) != len(coordinates):
                        raise ValueError("image tokens do not match their multimodal rotary grid")
                    result[row, span] = coordinates + cursor
                    cursor += max(height, width) // merge
                start = stop
        return jnp.asarray(result)


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
    """

    name: str
    paths: tuple[tuple[str, ...], ...]
    shape: tuple[int, ...]
    transpose: tuple[int, ...] | None = None
    concatenate: int | None = None
    expert_index: int | None = None

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
        return np.ascontiguousarray(value).reshape(self.shape)


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


_SPLIT_GATE_UP = ("llama4_text", "gemma4_text", "qwen3_5_moe_text")
"""Families whose fused `experts.gate_up_proj` loads as two stacked kernels;
GPT OSS keeps the reference's fused leaf and maps the name itself."""

_TRANSPOSED_EXPERTS = ("gemma4_text", "qwen3_5_moe_text")
"""Families whose expert matrices are stored [E, out, in]."""


def _language_layout(name: str, text_name: str, tensor: np.ndarray,
                     config, model_type: str, component: str | None = None) -> WeightLayout | None:
    """The text family's existing leaf map plus its inverse storage operations."""
    family = decoders._FAMILIES[model_type]

    def nested(path: tuple[str, ...]) -> tuple[str, ...]:
        return path if component is None else (path[0], component, *path[1:])

    transpose = None
    concatenate = None
    expert_index = None
    if text_name == "lm_head.weight" and config["tie_embeddings"]:
        paths = (nested(("params", "embed_tokens", "embedding")),)
    elif text_name.endswith(".experts.gate_up_proj") and model_type in _SPLIT_GATE_UP:
        names = [text_name.removesuffix("gate_up_proj") + projection for projection in ("gate_proj", "up_proj")]
        paths_list = []
        for key in names:
            path = family.weight_path(key, config)
            if path is None:
                raise ValueError(f"fused expert tensor {name!r} has no parameter path")
            paths_list.append(nested(path))
        paths = tuple(paths_list)
        concatenate = -1
        if model_type in _TRANSPOSED_EXPERTS:
            transpose = (0, 2, 1)
    else:
        path = family.weight_path(text_name, config)
        if path is None:
            return None
        path, expert_index = _stacked_expert(path)
        paths = (nested(path),)
        if path[-1] == "kernel" and tensor.ndim == 2:
            transpose = (1, 0)
        elif text_name.endswith(".experts.down_proj") and model_type in _TRANSPOSED_EXPERTS:
            transpose = (0, 2, 1)
    return WeightLayout(name, paths, tensor.shape, transpose, concatenate, expert_index)



def _wrapper_layouts(tensors, record):
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
                                      record["text_model_type"], "language_model")
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
    before `dequantize` replaces them with dense float32 weights;
    `requantize` writes those names back in the format.
    """

    names: Callable[[Mapping[str, np.ndarray]], tuple[str, ...]]
    dequantize: Callable[[Mapping[str, np.ndarray]], dict[str, np.ndarray]]
    requantize: Callable[[Mapping[str, np.ndarray], tuple[str, ...]], dict[str, np.ndarray]]


def _source_quantization(config: Mapping[str, object]) -> _SourceQuantization | None:
    """The format a config's `quantization_config` declares, or None for a dense source."""
    quantization = config.get("quantization_config")
    if quantization is None:
        return None
    if not isinstance(quantization, Mapping):
        raise ValueError(f"quantization_config must be an object, got {quantization!r}")
    method = quantization.get("quant_method")
    if method == "fp8":
        block, ue8m0 = fp8_format(quantization)
        return _SourceQuantization(scaled_names, partial(dequantize_checkpoint, block=block),
                                   partial(pack_fp8, block=block, ue8m0=ue8m0))
    if method == "mxfp4":
        return _SourceQuantization(mxfp4_stems, unpack_mxfp4, pack_mxfp4)
    raise ValueError(
        f"quantization_config names quant_method {method!r}; this loader reads DeepSeek's "
        f"fp8 blocks and GPT OSS's mxfp4 and nothing else")


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

    def text_generation(self, *, sampling: Sampling | None = None) -> TextGeneration:
        """The source's decoding components, or the caller's explicit policy.

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
        if self.export_adapter is not None:
            tensors = self.export_adapter(self.model, values, self.config)
        elif self.weight_layouts:
            # Source names and geometry first; the packed format goes back over them.
            text = self.model.language_model if isinstance(self.model, MultimodalTransformer) else self.model
            scalar_mode = text.layer_scalar if isinstance(text, CausalTransformer) else None
            tensors = {**self.retained_tensors,
                       **{layout.name: layout.export(values, scalar_mode) for layout in self.weight_layouts}}
        elif isinstance(self.model, CausalTransformer):
            # A source with no layout to run backwards is written by the
            # decoder export, which writes the whole directory: weights, the
            # config it derives, this processor's files and this generation
            # config. One export path, so a decoder saved here and one saved
            # directly leave the same files behind.
            decoders.save_pretrained_decoder(self.model, values, destination,
                                             tokenizer=self.processor,
                                             generation_config=generation_config)
            return
        else:
            raise ValueError("this source has no reversible weight layout")
        if quantization is not None:
            tensors = quantization.requantize(tensors, self.quantized_tensors)
        save_hf_layout(tensors, dict(self.config), destination)
        decoders.save_export_assets(destination, tokenizer=self.processor,
                                    generation_config=generation_config)





def _native_variables(parts: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    collections: dict[str, dict[str, object]] = {}
    for component, variables in parts.items():
        for collection, tree in variables.items():
            collections.setdefault(collection, {})[component] = tree
    return collections


def _generation_value(config: Mapping[str, object], generation_config: Mapping[str, object],
                      name: str, default: object = None) -> object:
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise ValueError("text_config must be a mapping")
    return generation_config.get(name, config.get(name, text.get(name, default)))


def _eos_ids(config: Mapping[str, object], generation_config: Mapping[str, object]) -> tuple[int, ...]:
    value = _generation_value(config, generation_config, "eos_token_id")
    if value is None:
        return ()
    values = (value,) if type(value) is int else value
    if not isinstance(values, (tuple, list)) or any(type(item) is not int or item < 0 for item in values):
        raise ValueError("eos_token_id must be an integer or a sequence of integers")
    return tuple(values)


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
    neutral: tuple[object, ...] = ()
    mode: Literal["always", "sampling", "beam"] = "always"
    refusal: str | None = None


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
        refusal="the native cache is the fixed-capacity static one"),
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
    "do_sample": _Control("policy"),
    "dola_layers": _Control("unsupported", refusal="DoLa is a decoding strategy that is not implemented"),
    "early_stopping": _Control("strategy", neutral=(False,), mode="beam"),
    "encoder_no_repeat_ngram_size": _Control("transform", neutral=(0,)),
    "encoder_repetition_penalty": _Control("transform", neutral=(1.0,)),
    "eos_token_id": _Control("policy"),
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
    "min_p": _Control("policy", mode="sampling"),
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
    "pad_token_id": _Control("policy"),
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
    "temperature": _Control("policy"),
    "token_healing": _Control(
        "unsupported",
        neutral=(False,),
        refusal="retokenizing the prompt is prompt construction, not decoding"),
    "tokenizer_name": _Control("metadata"),
    "top_h": _Control("transform", mode="sampling"),
    "top_k": _Control("policy"),
    "top_p": _Control("policy", mode="sampling"),
    "transformers_version": _Control("metadata"),
    "typical_p": _Control("transform", neutral=(1.0,), mode="sampling"),
    "use_cache": _Control(
        "unsupported",
        neutral=(True,),
        refusal="native decoding always runs through its own cache"),
    "use_mtp": _Control("strategy", neutral=(False,)),
    "watermarking_config": _Control("transform", refusal="no watermarking transform is implemented"),
}



def _neutral(value: object, neutral: tuple[object, ...]) -> bool:
    if value is None:
        return True
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    return any(value == item and (numeric and not isinstance(item, bool) or type(value) is type(item))
               for item in neutral)


def _active(config: Mapping[str, object], generation_config: Mapping[str, object],
            name: str) -> object:
    """The control's value when it is active, None when it changes nothing."""
    value = _generation_value(config, generation_config, name)
    rule = _CONTROLS.get(name)
    return None if _neutral(value, () if rule is None else rule.neutral) else value


def _audit(config: Mapping[str, object], generation_config: Mapping[str, object],
           model: nn.Module, do_sample: bool, beams: object, overridden: bool) -> None:
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



def _cache_capacity(config: Mapping[str, object], generation_config: Mapping[str, object],
                    model: nn.Module) -> None:
    """A declared cache length is real, and has to fit the model's own."""
    value = _generation_value(config, generation_config, "max_cache_len")
    if value is None:
        return
    capacity = getattr(model, "max_seq_len", None)
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
    if type(value) is not int:
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
    for item in strings:
        if not isinstance(item, str) or not item:
            raise ValueError("stop_strings must hold non-empty strings")
    return tuple(item for item in strings if isinstance(item, str))


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
    if not int(getattr(model, "num_nextn_predict_layers", 0) or 0):
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
    criteria = _source_stopping(config, generation_config, processor,
                                getattr(model, "vocab_size", None))
    policy = override if override is not None else _source_sampling(config, generation_config, do_sample)
    transforms = (None if override is not None else
                  _source_transforms(config, generation_config, policy, do_sample, isinstance(strategy, Beam)))
    return policy, transforms, criteria, strategy


class _Call(NamedTuple):
    """One pinned pipeline's own `__call__` policy, read from Diffusers
    0.34.0: the denoiser it drives, the steps and guidance scale it defaults
    to, whether that scale guides two branches or is the value the model
    embeds, and the text sequence budget it pads its T5 tower to."""

    component: str
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
    "StableDiffusionPipeline": _Call("unet", 50, 7.5, True),
    "StableDiffusionImg2ImgPipeline": _Call("unet", 50, 7.5, True),
    "StableDiffusionInpaintPipeline": _Call("unet", 50, 7.5, True),
    "StableDiffusionXLPipeline": _Call("unet", 50, 5.0, True),
    "StableDiffusionXLImg2ImgPipeline": _Call("unet", 50, 5.0, True),
    "StableDiffusionXLInpaintPipeline": _Call("unet", 50, 7.5, True),
    "StableDiffusion3Pipeline": _Call("transformer", 28, 7.0, True, 256),
    "FluxPipeline": _Call("transformer", 28, 3.5, False, 512),
    "FlaxStableDiffusionPipeline": _Call("unet", 50, 7.5, True),
    "FlaxStableDiffusionImg2ImgPipeline": _Call("unet", 50, 7.5, True),
    "FlaxStableDiffusionInpaintPipeline": _Call("unet", 50, 7.5, True),
    "FlaxStableDiffusionXLPipeline": _Call("unet", 50, 7.5, True),
})


def _call_policy(index: Mapping[str, object], denoiser: _Denoiser) -> _Call:
    """The call policy this file's own pipeline carries.

    A directory that declares no pipeline - a bare component tree - takes its
    family's reference pipeline. A directory that declares one Dew does not
    implement is refused rather than run under another pipeline's defaults,
    and a declared pipeline that drives a different denoiser than the one the
    directory holds is refused too: a component this loader reads does not
    qualify a workflow it does not.
    """
    published = index.get("_class_name")
    if published is None:
        policy = _PIPELINE_POLICY[denoiser.pipeline]
    else:
        found = _PIPELINE_POLICY.get(published) if isinstance(published, str) else None
        if found is None:
            raise ValueError(f"Native diffusion does not implement the published pipeline "
                             f"{published!r}")
        policy = found
    if policy.component != denoiser.component:
        raise ValueError(f"The published pipeline {published!r} drives a {policy.component}, and "
                         f"this directory holds a {denoiser.component}")
    return policy


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

    The native model and the variables its component's tensors translate to,
    and the reading conventions the rest of the source follows from it: which
    text towers it takes and how they compose, the width it reads their
    sequence at, how many latent pixels one of its positions covers, its own
    input channel count, the pipeline class its family's call policy comes
    from and where that pipeline's sigmas start.
    """

    component: str
    model: nn.Module
    variables: Variables
    layouts: tuple[WeightLayout, ...]
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
                           attention_impl: str) -> Pretrained:
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
    policy = _call_policy(index, denoiser)
    autoencoder, vae_params, vae_layouts, vae_config = _diffusion_vae(directory, compute)
    names = tuple(name for name in denoiser.towers if _present(index, name))
    if not names:
        raise ValueError("A latent diffusion source needs at least one text encoder")
    towers, tokenizers, text_params, text_layouts = _clip_towers(directory, names, compute)
    components = {denoiser.component: denoiser.config, "vae": vae_config,
                  **{name: _component_config(directory, name) for name in names}}
    t5 = None
    if denoiser.t5_tower is not None and _present(index, denoiser.t5_tower):
        t5, t5_params, t5_layouts, components[denoiser.t5_tower] = _t5_tower(
            directory, compute, denoiser.t5_tower, policy.sequence)
        text_params[denoiser.t5_tower] = t5_params
        text_layouts += t5_layouts
    size = denoiser.sample_size * autoencoder.downscale_factor
    height, width = index.get("dew_height", size), index.get("dew_width", size)
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError("Image geometry must contain positive integer dimensions")
    # A guidance-embedded model takes the pipeline's scale as an input; a
    # model that reads none carries nothing.
    encoder = DiffusionConditioner(
        towers, tokenizers, names, text_params, str(directory), height, width,
        denoiser.context_width, composition=denoiser.composition, t5=t5,
        guidance=policy.guidance if denoiser.embeds_guidance and not policy.guided else None,
        aesthetics=bool(index.get("requires_aesthetics_score", False)))
    inpaint = denoiser.latent_input == autoencoder.latent_channels * 2 + 1
    inputs = InputSpec(Field("image", (height, width, 3)),
                       {"conditioning": Condition(encoder, unconditional=_unconditional(
                           denoiser.composition, index))},
                       mask=Field("mask", (height, width, 1)) if inpaint else None)
    encoders: dict[str, object] = {"conditioning": text_params}
    finish, safety_layouts = None, ()
    if _present(index, "safety_checker"):
        finish, encoders["safety"], safety_layouts, safety_configs = _image_safety(
            directory, compute)
        components.update(safety_configs)
    schedule = SourceSchedule.from_config(_component_config(directory, "scheduler"))
    components["scheduler"] = dict(schedule.config)
    patch = denoiser.patch * autoencoder.downscale_factor
    task = SourceTask(min(policy.steps, schedule.train_steps),
                      CFG(policy.guidance) if policy.guided and policy.guidance > 1 else None,
                      functools.partial(schedule.sampling, origin=denoiser.origin,
                                        tokens=(height // patch) * (width // patch)))
    variables = {**denoiser.variables, "encoders": encoders, "autoencoder": vae_params}
    config = {"model_index": {**index, "dew_height": height, "dew_width": width}, **components}
    return Pretrained(denoiser.model, variables, None, config, directory, denoiser.built,
                      weight_layouts=denoiser.layouts + vae_layouts + text_layouts + safety_layouts,
                      process=schedule.training_process(), inputs=inputs, autoencoder=autoencoder,
                      schedule=schedule, finish=finish, task=task)


def _unet_denoiser(directory: Path, *, dtype: str, attention_impl: str) -> _Denoiser:
    """The published UNet: cross attention over one or two CLIP towers, whose
    pooled text conditioning is the one its added time features ask for."""
    from dew.interop import diffusion
    from dew.interop.diffusion import _integer
    from dew.nn.backbones.unet_condition import UNet2DCondition

    config = _component_config(directory, "unet")
    fields = diffusion.unet_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = UNet2DCondition(**fields)
    params, layouts = diffusion.translate_unet_weights(
        diffusion.component_tensors(directory, "unet"), model)
    pooled = model.additional_time_features > 0
    built = {"name": "unet_2d_condition",
             "fields": {**fields, "dtype": dtype,
                        "stages": [asdict(stage) for stage in model.stages]}}
    return _Denoiser(
        component="unet", model=model, variables={"params": params}, layouts=layouts,
        built=built, config=config, composition="clip_pooled" if pooled else "clip",
        towers=("text_encoder", "text_encoder_2"), patch=1, latent_input=model.in_channels,
        sample_size=_integer(config["sample_size"], "sample_size"),
        context_width=_integer(config.get("cross_attention_dim", 1280), "cross_attention_dim"),
        pipeline="StableDiffusionXLPipeline" if pooled else "StableDiffusionPipeline")


def _transformer_denoiser(directory: Path, *, dtype: str, attention_impl: str) -> _Denoiser:
    """The published transformer this directory holds, by the class it names."""
    config = _component_config(directory, "transformer")
    published = config.get("_class_name")
    if published == "SD3Transformer2DModel":
        return _sd3_denoiser(config, directory, dtype=dtype, attention_impl=attention_impl)
    if published == "FluxTransformer2DModel":
        return _flux_denoiser(config, directory, dtype=dtype, attention_impl=attention_impl)
    raise ValueError(f"Native diffusion does not implement the published transformer "
                     f"{published!r}")


def _sd3_denoiser(config: dict, directory: Path, *, dtype: str, attention_impl: str) -> _Denoiser:
    """SD3's MM-DiT: both CLIP towers and the T5 tower read jointly, with the
    stored position buffer in its own frozen collection."""
    from dew.interop import diffusion
    from dew.interop.diffusion import _integer
    from dew.nn.backbones.sd3 import SD3Transformer

    fields = diffusion.sd3_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = SD3Transformer(**fields)
    params, buffers, layouts = diffusion.translate_sd3_weights(
        diffusion.component_tensors(directory, "transformer"))
    built = {"name": "sd3_transformer",
             "fields": {**fields, "dtype": dtype,
                        "dual_attention_layers": list(fields["dual_attention_layers"])}}
    return _Denoiser(
        component="transformer", model=model, variables={"params": params, "buffers": buffers},
        layouts=layouts, built=built, config=config, composition="sd3",
        towers=("text_encoder", "text_encoder_2"), patch=fields["patch_size"],
        latent_input=fields["in_channels"], sample_size=_integer(config["sample_size"], "sample_size"),
        context_width=fields["joint_attention_dim"], pipeline="StableDiffusion3Pipeline",
        t5_tower="text_encoder_3")


def _flux_denoiser(config: dict, directory: Path, *, dtype: str, attention_impl: str) -> _Denoiser:
    """Flux's transformer: one CLIP tower for the pooled vector, the T5 tower
    for the sequence, and a latent its pipeline packs in 2x2 patches.

    The class declares no sample size; its pipeline's `default_sample_size`
    is 128 latent positions, which a directory overrides with its own
    geometry. It starts from the sigmas its pipeline hands the scheduler.
    """
    from dew.interop import diffusion
    from dew.interop.diffusion import _integer
    from dew.nn.backbones.flux import FluxTransformer

    fields = diffusion.flux_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = FluxTransformer(**fields)
    params, layouts = diffusion.translate_flux_weights(
        diffusion.component_tensors(directory, "transformer"))
    built = {"name": "flux_transformer",
             "fields": {**fields, "dtype": dtype,
                        "axes_dims_rope": list(fields["axes_dims_rope"])}}
    return _Denoiser(
        component="transformer", model=model, variables={"params": params}, layouts=layouts,
        built=built, config=config, composition="flux", towers=("text_encoder",), patch=2,
        latent_input=fields["in_channels"] // 4,
        sample_size=_integer(config.get("sample_size", 128), "sample_size"),
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


def _diffusion_vae(directory: Path, compute) -> tuple[StableDiffusionVAE, Variables,
                                                      tuple[WeightLayout, ...], dict]:
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
        "vae", tensors, lambda name: _vae_path(name, np.ndim(tensors[name])), ("autoencoder",))
    autoencoder = StableDiffusionVAE(str(directory), dtype=compute, params=params, model=model,
                                     latent_shift=config.get("shift_factor") or 0.0,
                                     latent_scale=config.get("scaling_factor", 0.18215))
    return autoencoder, params, layouts, config


def _clip_towers(directory: Path, names: tuple[str, ...], compute):
    """The published CLIP text towers, their tokenizers, their parameters and
    the layouts those parameters came from."""
    from transformers import CLIPTokenizer
    from dew.interop import diffusion
    from dew.nn.text_encoders import CLIPTextTransformer, translate_config

    towers, tokenizers, params, layouts = [], [], {}, ()
    for name in names:
        config = _component_config(directory, name)
        towers.append(CLIPTextTransformer(**translate_config(config), dtype=compute))
        tower, recorded = diffusion.record_layouts(
            name, diffusion.component_tensors(directory, name), _text_head_path,
            ("encoders", "conditioning", name))
        params[name] = tower
        layouts += recorded
        tokenizers.append(CLIPTokenizer.from_pretrained(
            directory / ("tokenizer" + name.removeprefix("text_encoder"))))
    return tuple(towers), tuple(tokenizers), params, layouts


def _t5_tower(directory: Path, compute, component: str, tokens: int):
    """The published T5 encoder as the conditioner's segment, with its
    parameters, their layouts and its config.

    `component` is where the family keeps it: an SD3 directory's third text
    encoder, a Flux directory's second one; `tokens` is the sequence budget
    the pipeline pads to.
    """
    from transformers import AutoTokenizer
    from dew.interop import diffusion
    from dew.nn.text_encoders import (
        T5EncoderTransformer, _t5_path, t5_embedding, translate_t5_config)

    config = _component_config(directory, component)
    tower = T5EncoderTransformer(**translate_t5_config(config), dtype=compute)
    tensors = diffusion.component_tensors(directory, component)
    t5_embedding(tensors)
    params, layouts = diffusion.record_layouts(
        component, tensors, _t5_path, ("encoders", "conditioning", component))
    tokenizer = AutoTokenizer.from_pretrained(
        directory / ("tokenizer" + component.removeprefix("text_encoder")))
    return T5Segment(tower, tokenizer, component, tokens), params, layouts, config


def _unconditional(composition: str, index: Mapping[str, object]) -> dict:
    """The empty-prompt row a file's own pipeline guides against: the XL
    pipelines zero it where their index says so, and the SD3 pipeline encodes
    it with its towers, having no such control."""
    zero = composition == "clip_pooled" and bool(index.get("force_zeros_for_empty_prompt", True))
    return {"text": "", "negative": True, "zero": zero}


def _image_safety(directory: Path, compute):
    """The safety head a file declares: the finish, its parameters, their
    layouts and the two configs it ships."""
    from dew.inputs.diffusion import CLIPImageTransform, CLIPSafetyHead, ImageSafety
    from dew.interop import diffusion
    from dew.nn.text_encoders import CLIPVisionTransformer, translate_vision_config

    config = _component_config(directory, "safety_checker")
    with open(directory / "feature_extractor" / "preprocessor_config.json") as handle:
        transform = json.load(handle)
    params, layouts = diffusion.record_layouts(
        "safety_checker", diffusion.component_tensors(directory, "safety_checker"), _safety_path,
        ("encoders", "safety"))
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


def load_pretrained(name_or_dir: str | Path, *, dtype: str = "bfloat16",
                    attention_impl: str = "auto", max_seq_len: int | None = None,
                    revision: str | None = None) -> Pretrained:
    """Load a source into a native Flax model with explicit parameter trees.

    ``name_or_dir`` is a local HF directory or a Hub model identifier. The
    decoder/tower/projector maps preserve their established internal paths;
    wrapper variables join under their existing component names. Processor
    artifacts are loaded only when the source contains them.
    """
    directory = decoders._snapshot(str(name_or_dir), revision)
    if (directory / "model_index.json").is_file():
        with open(directory / "model_index.json") as handle:
            return _load_diffusion_source(directory, json.load(handle), dtype=dtype, attention_impl=attention_impl)
    with open(directory / "config.json") as handle:
        config = json.load(handle)
    tensors = decoders._load_shards(directory)
    family = config.get("model_type")
    quantization = _source_quantization(config)
    quantized_tensors = () if quantization is None else quantization.names(tensors)
    if quantization is not None:
        tensors = quantization.dequantize(tensors)
    layouts: tuple[WeightLayout, ...] = ()
    retained: dict[str, np.ndarray] = {}
    export_adapter = None
    if family == "diffusion_gemma":
        from dew.interop import diffusion_gemma
        model = diffusion_gemma.build(config, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len)
        variables = diffusion_gemma.translate_weights(tensors, config)
        record = config
        built: Mapping[str, object] = {**config, "dtype": dtype, "attention_impl": attention_impl}
        export_adapter = diffusion_gemma.export_weights
    elif "text_config" in config:
        record = decoders.translate_wrapper_config(config)
        text_fields = dict(record["text"])
        if max_seq_len is not None:
            text_fields["max_seq_len"] = max_seq_len
        if family == "gemma3":
            # Gemma3ForConditionalGeneration projects logits without the
            # causal-LM class's optional final tanh cap.
            text_fields["final_logit_softcap"] = None
            text_fields["mixer"] = {"kind": "attention", "bidirectional_images": True}
        if family == "gemma4" and config["text_config"].get("use_bidirectional_attention") == "vision":
            kinds = dict(text_fields["kinds"])
            sliding = dict(kinds.get("sliding_attention", {}))
            sliding["mixer"] = {"kind": "attention", "bidirectional_images": True}
            kinds["sliding_attention"] = sliding
            text_fields["kinds"] = kinds
        if family == "qwen3_5":
            rope = config["text_config"].get("rope_parameters") or {}
            sections = rope.get("mrope_section", [11, 11, 10])
            if (not isinstance(sections, (list, tuple)) or len(sections) != 3
                    or any(type(value) is not int or value < 0 for value in sections)):
                raise ValueError("mrope_section must contain three nonnegative integer widths")
            kinds = dict(text_fields["kinds"])
            full = dict(kinds.get("full_attention", {}))
            full["mixer"] = {"kind": "attention", "mrope_section": [sections[0], sections[1], sections[2]]}
            kinds["full_attention"] = full
            text_fields["kinds"] = kinds

        built = {**record, "text": with_precision(
            "causal_transformer", text_fields, dtype=dtype, attention_impl=attention_impl)}
        language_model = models.build("causal_transformer", **built["text"])
        if not isinstance(language_model, CausalTransformer):
            raise TypeError("causal_transformer registry entry must build CausalTransformer")
        audio_record = record["audio"]
        model = MultimodalTransformer(
            language_model, tower_from_record(record["tower"]),
            projector_from_record(record["projector"]), family,
            record["image_token_id"], dtype=resolve_dtype(dtype),
            pad_token_id=config["text_config"].get("pad_token_id", 0),
            extra_placeholder_ids=(tuple(config.get(name, default) for name, default in
                (("video_token_id", 258884), ("audio_token_id", 258881))) if family == "gemma4" else ()),
            audio=None if audio_record is None else tower_from_record(audio_record),
            audio_projection=None if audio_record is None else projector_from_record(record["audio_projector"]),
            audio_soft_tokens=record["audio_soft_tokens"],
            attention_impl=None if attention_impl == "reference" else attention_impl)
        variables = _native_variables(decoders.translate_wrapper_weights(tensors, record))
        layouts, retained = _wrapper_layouts(tensors, record)
    else:
        record = decoders.translate_config(config)
        if max_seq_len is not None:
            record["max_seq_len"] = max_seq_len
        built = with_precision("causal_transformer", record, dtype=dtype, attention_impl=attention_impl)
        model = models.build("causal_transformer", **built)
        variables = decoders.translate_weights(tensors, record)
        decoders._check_tree(variables, model)
        if decoders._FAMILIES[family].preserve_source_layout or quantized_tensors:
            bindings = []
            for name, tensor in tensors.items():
                layout = _language_layout(name, name, tensor, record, family)
                if layout is None:
                    retained[name] = tensor
                else:
                    bindings.append(layout)
            layouts = tuple(bindings)
    processor = None
    if any((directory / name).exists() for name in ("processor_config.json", "preprocessor_config.json")):
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
