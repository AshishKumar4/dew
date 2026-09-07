"""Load native models and their host processors from a Hugging Face source."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.artifacts import agree_process_phase
from dew.interop import hf_decoders as decoders
from dew.interop.quantized import dequantize_checkpoint, fp8_block
from dew.inference import BlockGeneration, TextGeneration
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.sampling.text import Sampling
from dew.nn import audio as audio_nn
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.mixers.attention import AttentionMixer
from dew.nn.vision import projector_from_record, tower_from_record
from dew.objectives.base import Variables
from dew.registry import models, resolve_dtype, with_precision


class HostProcessor(Protocol):
    """The HF processor operations kept outside compiled model computation."""

    def __call__(self, **kwargs: object) -> Mapping[str, object]: ...
    def save_pretrained(self, save_directory: str) -> object: ...
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

    def __call__(self, text: str | Sequence[str], *, images: object | None = None,
                 audio: object | None = None) -> ModelInputs:
        import torch

        # truncation is off for text anyway; reloaded Gemma processors forward
        # the tokenizer's unset max_length into audio kwargs otherwise.
        arguments: dict[str, object] = {
            "text": text if isinstance(text, str) else list(text),
            "padding": not isinstance(text, str) and len(text) > 1, "truncation": False, "return_tensors": "pt"}
        if images is not None:
            arguments["images"] = images
        if audio is not None:
            arguments["audio"] = audio
        arrays: dict[str, np.ndarray] = {}
        for name, value in self.reference(**arguments).items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"processor field {name!r} is not a tensor")
            # Llama 4 normalizes pixels in bfloat16 as its original
            # implementation does; widening to float32 is exact.
            arrays[name] = (value.float() if value.dtype == torch.bfloat16 else value).numpy()
        return self.from_hf(arrays)

    def from_hf(self, values: Mapping[str, object]) -> ModelInputs:
        """Validate and normalize actual processor outputs before device use."""
        known = {"input_ids", "attention_mask", "pixel_values", "token_type_ids", "mm_token_type_ids",
                 "image_position_ids", "image_grid_thw", "input_features", "input_features_mask"}
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
        token_fields = {"attention_mask": jnp.asarray(valid), "positions": jnp.asarray(positions)}
        conditioning: dict[str, jax.Array] = {}
        if "pixel_values" in values:
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
        # Where a record keeps its text half: a translated multimodal wrapper
        # under 'text', a DiffusionGemma source, whose record is its own HF
        # config, under 'text_config', and a decoder-only record is the text
        # record. A processor a caller built with no record at all names no
        # vocabulary, so there is nothing to hold its ids to.
        text = self.record.get("text") or self.record.get("text_config") or self.record
        if text:
            if not isinstance(text, Mapping) or type(text.get("vocab_size")) is not int:
                raise ValueError("the text record must carry its integer vocab_size")
            if np.any(tokens < 0) or np.any(tokens >= text["vocab_size"]):
                raise ValueError("input_ids must lie in the text vocabulary, including any hard media ranges")
        result = ModelInputs(jnp.asarray(tokens, jnp.int32), token_fields, conditioning)
        result.validate()
        return result

    def _images(self, values: Mapping[str, object], tokens: np.ndarray
                ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        image_id = self.record.get("image_token_id", self.config.get("image_token_id"))
        if type(image_id) is not int:
            raise ValueError("image_token_id must be an integer")
        pixels = np.asarray(values["pixel_values"])
        if pixels.ndim not in (2, 3, 4):
            raise ValueError("pixel_values must contain image tensors or patch vectors")
        if not np.issubdtype(pixels.dtype, np.floating):
            raise ValueError("pixel_values must be floating processor output")
        runs = []
        for row in tokens:
            locations = np.flatnonzero(row == image_id)
            runs.append([] if not len(locations) else np.split(locations, np.flatnonzero(np.diff(locations) != 1) + 1))
        image_counts = np.array([len(row) for row in runs], np.int32)
        patch_positions = None
        grid = None
        merge = 1
        if "image_grid_thw" in values:
            vision = self.config.get("vision_config")
            if not isinstance(vision, Mapping):
                raise ValueError("image_grid_thw requires a vision config")
            merge = vision.get("spatial_merge_size")
            if type(merge) is not int or merge < 1:
                raise ValueError("spatial_merge_size must be a positive integer")
            grid = np.asarray(values["image_grid_thw"])
            if (grid.ndim != 2 or grid.shape[1] != 3 or not np.issubdtype(grid.dtype, np.integer)
                    or np.any(grid <= 0) or np.any(grid[:, 1:] % merge)):
                raise ValueError("image_grid_thw must contain positive integer grids tiled by spatial_merge_size")
            counts = np.prod(grid, axis=1)
            if pixels.ndim != 2 or int(counts.sum()) != pixels.shape[0]:
                raise ValueError("packed pixel_values do not match image_grid_thw")
            offsets = np.concatenate([[0], np.cumsum(counts)])
            chunks = [pixels[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]
            shape = (int(counts.max()), pixels.shape[-1])
            lengths = counts // merge ** 2
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
    """An existing source tensor's location and reversible storage layout."""

    name: str
    paths: tuple[tuple[str, ...], ...]
    shape: tuple[int, ...]
    transpose: tuple[int, ...] | None = None
    concatenate: int | None = None

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
            leaves.append(np.asarray(node))
        value = leaves[0] if self.concatenate is None else np.concatenate(leaves, axis=self.concatenate)
        if self.transpose is not None:
            value = value.transpose(self.transpose)
        return np.ascontiguousarray(value).reshape(self.shape)


def _language_layout(name: str, text_name: str, tensor: np.ndarray,
                     config, model_type: str, component: str | None = None) -> WeightLayout | None:
    """The text family's existing leaf map plus its inverse storage operations."""
    family = decoders._FAMILIES[model_type]

    def nested(path: tuple[str, ...]) -> tuple[str, ...]:
        return path if component is None else (path[0], component, *path[1:])

    transpose = None
    concatenate = None
    if text_name == "lm_head.weight" and config["tie_embeddings"]:
        paths = (nested(("params", "embed_tokens", "embedding")),)
    elif text_name.endswith(".experts.gate_up_proj"):
        names = [text_name.removesuffix("gate_up_proj") + projection for projection in ("gate_proj", "up_proj")]
        paths_list = []
        for key in names:
            path = family.weight_path(key, config)
            if path is None:
                raise ValueError(f"fused expert tensor {name!r} has no parameter path")
            paths_list.append(nested(path))
        paths = tuple(paths_list)
        concatenate = -1
        if model_type == "gemma4_text":
            transpose = (0, 2, 1)
    else:
        path = family.weight_path(text_name, config)
        if path is None:
            return None
        paths = (nested(path),)
        if path[-1] == "kernel" and tensor.ndim == 2:
            transpose = (1, 0)
        elif text_name.endswith(".experts.down_proj") and model_type == "gemma4_text":
            transpose = (0, 2, 1)
    return WeightLayout(name, paths, tensor.shape, transpose, concatenate)



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
        elif bare.startswith("language_model.") or bare == "lm_head.weight":
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

    def text_generation(self, *, sampling: Sampling | None = None) -> TextGeneration:
        """Use the source policy, or an explicit supported policy supplied by the caller."""
        if isinstance(self.model, DiffusionGemma):
            raise TypeError("a DiffusionGemma source generates through block_generation")
        return TextGeneration(self.model, self.variables, self.processor, sampling if sampling is not None
                              else _source_sampling(self.config, self.generation_config))

    def block_generation(self) -> BlockGeneration:
        """The DiffusionGemma as a canvas task, defaulting to the source's sampler config."""
        from dew.interop import diffusion_gemma
        if not isinstance(self.model, DiffusionGemma):
            raise TypeError("block generation needs a DiffusionGemma source")
        return BlockGeneration(self.model, self.variables,
                               diffusion_gemma.generation_process(self.config, self.generation_config),
                               self.processor, _eos_ids(self.config, self.generation_config),
                               _pad_id(self.config, self.generation_config))

    def save(self, directory: str | Path, *, variables: Mapping[str, object] | None = None) -> None:
        """Write trained variables back to the source layout with its tokenizer assets."""
        from dew.interop.safetensors_io import save_hf_layout
        values = self.variables if variables is None else variables
        destination = Path(directory)
        generation_config = dict(self.generation_config)
        if self.export_adapter is not None:
            tensors = self.export_adapter(self.model, values, self.config)
        elif self.weight_layouts:
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


def _probability_control(config: Mapping[str, object], generation_config: Mapping[str, object],
                         name: str, default: float) -> float:
    value = _generation_value(config, generation_config, name, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _source_sampling(config: Mapping[str, object], generation_config: Mapping[str, object]) -> Sampling:
    """Construct only policies whose active controls the native sampler implements."""
    from transformers import GenerationConfig

    defaults = GenerationConfig().to_dict()
    do_sample = _generation_value(config, generation_config, "do_sample", False)
    if type(do_sample) is not bool:
        raise ValueError("do_sample must be a boolean")
    supported = {"do_sample", "temperature", "top_k", "top_p", "min_p", "eos_token_id", "pad_token_id"}
    # Prompt creation, requested output size and output representation belong
    # to the task. These source fields do not transform its token distribution.
    task_owned = {"bos_token_id", "max_length", "max_new_tokens", "use_cache",
                  "cache_implementation", "cache_config", "return_legacy_cache", "compile_config",
                  "disable_compile", "output_attentions", "output_hidden_states", "output_scores",
                  "output_logits", "return_dict_in_generate", "transformers_version",
                  "_from_model_config", "_commit_hash", "tokenizer_name"}
    sampled_only = {"top_p", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff", "top_h"}
    unsupported = []
    for name in set(defaults) | set(generation_config):
        if name in supported or name in task_owned:
            continue
        if not do_sample and name in sampled_only:
            continue
        default = defaults.get(name)
        value = _generation_value(config, generation_config, name, default)
        if value != default:
            unsupported.append(name)
    if unsupported:
        raise ValueError(f"native sampling cannot honor active source controls {sorted(unsupported)}; "
                         "pass an explicit sampling=Sampling(...) policy to text_generation")
    temperature = _generation_value(config, generation_config, "temperature", 1.0)
    if not isinstance(temperature, (float, int)) or isinstance(temperature, bool):
        raise ValueError("temperature must be numeric")
    top_k = _generation_value(config, generation_config, "top_k", defaults["top_k"])
    if top_k is not None and type(top_k) is not int:
        raise ValueError("top_k must be an integer")
    return Sampling(
        temperature=float(temperature) if do_sample else 0.0,
        top_k=top_k if top_k else None, eos_id=(_eos_ids(config, generation_config) or None),
        pad_id=_pad_id(config, generation_config),
        top_p=_probability_control(config, generation_config, "top_p", 1.0),
        min_p=_probability_control(config, generation_config, "min_p", 0.0))


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
    with open(directory / "config.json") as handle:
        config = json.load(handle)
    tensors = dequantize_checkpoint(decoders._load_shards(directory), fp8_block(config))
    family = config.get("model_type")
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
        if family in ("gemma4_text", "gemma3n_text", "qwen3_5_text"):
            bindings = []
            for name, tensor in tensors.items():
                layout = _language_layout(name, name, tensor, record, family)
                if layout is None:
                    retained[name] = tensor
                else:
                    bindings.append(layout)
            layouts = tuple(bindings)
    processor = None
    if (directory / "processor_config.json").exists():
        from transformers import AutoProcessor
        options = {"backend": "pil"} if family == "gemma3" else {}
        reference = AutoProcessor.from_pretrained(str(directory), local_files_only=True, **options)
        processor = Processor(reference, config, record)
    elif (directory / "tokenizer_config.json").exists():
        from transformers import AutoTokenizer
        reference = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        processor = Processor(reference, config, record)
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
                      layouts, retained, export_adapter)
