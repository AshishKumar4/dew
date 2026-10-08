"""Host processor calls and media placeholder alignment for native model inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, TypeGuard, Unpack

import jax
import jax.numpy as jnp
import numpy as np

from dew import records
from dew._model_types import QWEN35_TYPES
from dew.nn import audio as audio_nn
from dew.nn.inputs import Media, ModelInputs, pad_token_rows
from dew.records import JSON
from dew.registry import towers

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


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


def hosts(reference: PreTrainedTokenizerBase) -> TypeGuard[HostProcessor]:
    """Whether `reference` offers every HostProcessor operation, checked by
    name because a tokenizer's own annotations are narrower than what dew
    passes (see `data.chat._token_ids`)."""
    return all(callable(getattr(reference, name, None))
               for name in ("__call__", "save_pretrained", "apply_chat_template", "batch_decode"))


def _media_token_ids(record: Mapping[str, object], config: Mapping[str, object], *,
                     qwen: bool, gemma: bool) -> tuple[int, int | None]:
    """Read the source's image/video placeholder IDs before aligning features."""
    image_id = record.get("image_token_id", config.get("image_token_id"))
    if type(image_id) is not int:
        raise ValueError("image_token_id must be an integer")
    video_id = (config.get("video_token_id", 258884) if gemma else
                config.get("video_token_id") if qwen else None)
    if video_id is not None and type(video_id) is not int:
        raise ValueError("video_token_id must be an integer")
    return image_id, video_id


def _placeholder_runs(tokens: np.ndarray, image_id: int, video_id: int | None) -> list[list[np.ndarray]]:
    """Contiguous runs of one media kind, in prompt-row and placeholder order."""
    runs = []
    for row in tokens:
        locations = np.flatnonzero((row == image_id) | (row == video_id if video_id is not None else False))
        boundaries = (
            np.flatnonzero((np.diff(locations) != 1) | (row[locations[1:]] != row[locations[:-1]])) + 1
        )
        runs.append([] if not len(locations) else np.split(locations, boundaries))
    return runs


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
    """Runs a checkpoint's own processor on the host and arranges its outputs as native model inputs.

    The checkpoint's processor does the resizing, normalization and
    special-token expansion. Dew arranges its outputs into row-aligned
    arrays; it does not reproduce the checkpoint's image preprocessing
    algorithms. Calling it with text, and optionally `images`, `audio` or
    `videos`, returns `ModelInputs`.
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
            "padding": not isinstance(text, str) and len(text) > 1,
            "truncation": False,
            "return_tensors": "pt",
        }
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
        """Apply the checkpoint's own chat template and processor to `messages` and return numeric inputs.

        The checkpoint's template interprets template options such as
        `reasoning_effort` and `preserve_thinking`. Messages with media go
        through the same processor and numeric normalization as plain text.
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
        """Check a processor's raw outputs and turn them into `ModelInputs` for the device.

        A field with no native model input, or `input_ids` that are not
        nonempty integer `[B, S]` rows, raises ValueError.
        """
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
                    and self.config.get("model_type") not in (*QWEN35_TYPES, "gemma4")):
                raise ValueError("video patch inputs require a Qwen3.5 or Gemma4 visual tower")
            image_fields, conditioning = self._images(values, tokens)
            token_fields.update(image_fields)
            if self.config.get("model_type") in QWEN35_TYPES:
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
        qwen = self.config.get("model_type") in QWEN35_TYPES
        gemma = self.config.get("model_type") == "gemma4"
        image_id, video_id = _media_token_ids(self.record, self.config, qwen=qwen, gemma=gemma)
        pixels = np.asarray(values.get("pixel_values", values.get("pixel_values_videos")))
        if pixels.ndim not in (2, 3, 4) or not np.issubdtype(pixels.dtype, np.floating):
            raise ValueError("pixel_values must contain floating image tensors or patch vectors")
        pixel_dtype = pixels.dtype
        runs = _placeholder_runs(tokens, image_id, video_id)
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

    def _positioned_chunks(
        self,
        values: Mapping[str, object],
        tokens: np.ndarray,
        runs: list[list[np.ndarray]],
        image_id: int,
        video_id: int | None,
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
        runs = _placeholder_runs(tokens, audio_id, None)
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
        conditioning = {
            "input_features": jnp.asarray(padded),
            "input_features_mask": jnp.asarray(padded_mask),
            "audio_lengths": jnp.asarray(counts),
        }
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
        """Decode integer `[B, S]` token rows with the processor's tokenizer, skipping special tokens."""
        array = np.asarray(tokens)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("decode expects integer [B, S] token rows")
        return self.reference.batch_decode(array.tolist(), skip_special_tokens=True)

    @property
    def bos_id(self) -> int | None:
        """The id the source's tokenizer starts a sequence with, or None."""
        return _row_start(self.reference)

    def save_pretrained(self, directory: str | Path) -> None:
        """Save the processor and tokenizer this object uses to `directory`."""
        self.reference.save_pretrained(str(directory))
