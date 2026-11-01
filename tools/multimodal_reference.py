"""Write processor-to-wrapper references for the native multimodal wrappers.

Run with transformers 5.16.1, Torch and tokenizers installed:
    PYTHONPATH=src python tools/multimodal_reference.py
    PYTHONPATH=src:. python tools/multimodal_reference.py float64 [family ...]  # the float64 truths

No network access or pretrained downloads are used. Each family's existing
tiny fixture supplies the weights, and every reference lands in
tests/fixtures/hf/<family>-native-tiny: the padded batch encoding, the
reference logits, a cached greedy continuation, the pixel gradient and the
logits after one SGD step. Unequal image counts per row exercise numeric
batch alignment.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AddedToken,
    AutoProcessor,
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3Processor,
    GemmaTokenizer,
)
from transformers.models.gemma3.image_processing_pil_gemma3 import Gemma3ImageProcessorPil

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
LEARNING_RATE = 1e-4


def _tokenizer(vocab_size: int, special: dict[str, int], **token_attributes: str) -> GemmaTokenizer:
    """A local vocabulary with stable IDs under the public tokenizer serializer."""
    vocab = {f"token{i}": i for i in range(vocab_size)}
    for token, index in special.items():
        del vocab[f"token{index}"]
        vocab[token] = index
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = GemmaTokenizer(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
        unk_token="<unk>", boi_token="<start_of_image>", eoi_token="<end_of_image>",
        image_token="<image_soft_token>", padding_side="left", **token_attributes)
    tokenizer.add_tokens([AddedToken(word, single_word=True, normalized=False)
                          for word in vocab if word.startswith("token")])
    tokenizer.add_tokens([AddedToken(word, special=True, normalized=False)
                          for word in special if word not in tokenizer.all_special_tokens])
    return tokenizer



def _runs(row: torch.Tensor, token_id: int) -> int:
    """Contiguous placeholder runs in one row: one per image, tile or clip."""
    marks = (row == token_id).long()
    return int(((marks[1:] - marks[:-1]) == 1).sum() + marks[0])


def _row_slices(encoded: dict, model) -> list[dict]:
    """Each row of the batch encoding on its own, without padding.

    Media tensors are sliced in placeholder-run order, which is the order the
    processor emits them: images or tiles for pixel_values (patches for Qwen's
    packed grids), clips for input_features. The model then sees exactly the
    numbers the batch carries, so the reference does not depend on how the
    reference model batches padded rows (Llama 4 scales attention by the
    padded slot index) nor on how the processor frames padded audio.
    """
    valid = encoded["attention_mask"].bool()
    image_id = model.config.image_token_id
    grid = encoded.get("image_grid_thw")
    image_offset = video_offset = clip_offset = 0
    video_pixels = encoded.get("pixel_values_videos")
    video_positions = encoded.get("video_position_ids")
    if video_pixels is not None:
        video_pixels = video_pixels.flatten(0, 1)
        video_positions = video_positions.flatten(0, 1)
    singles = []
    for row in range(encoded["input_ids"].shape[0]):
        ids = encoded["input_ids"][row][valid[row]]
        single = {"input_ids": ids[None], "attention_mask": torch.ones_like(ids)[None]}
        for name in ("token_type_ids", "mm_token_type_ids"):
            if name in encoded:
                single[name] = encoded[name][row][valid[row]][None]
        images = _runs(ids, image_id)
        if "pixel_values" in encoded and images and grid is None:
            single["pixel_values"] = encoded["pixel_values"][image_offset:image_offset + images]
            if "image_position_ids" in encoded:
                single["image_position_ids"] = encoded["image_position_ids"][
                    image_offset:image_offset + images]
        elif "pixel_values" in encoded and images:
            patches = grid.prod(dim=1)
            start = int(patches[:image_offset].sum())
            stop = int(patches[:image_offset + images].sum())
            single["pixel_values"] = encoded["pixel_values"][start:stop]
            single["image_grid_thw"] = grid[image_offset:image_offset + images]
        image_offset += images
        if video_pixels is not None:
            frames = _runs(ids, model.config.video_token_id)
            if frames:
                single["pixel_values_videos"] = video_pixels[video_offset:video_offset + frames, None]
                single["video_position_ids"] = video_positions[video_offset:video_offset + frames, None]
            video_offset += frames
        if "input_features" in encoded:
            clips = _runs(ids, model.config.audio_token_id)
            if clips:
                single["input_features"] = encoded["input_features"][clip_offset:clip_offset + clips]
                single["input_features_mask"] = encoded["input_features_mask"][
                    clip_offset:clip_offset + clips]
            clip_offset += clips
        singles.append(single)
    if ("pixel_values" in encoded
            and image_offset != (encoded["pixel_values"].shape[0] if grid is None else grid.shape[0])):
        raise ValueError("the reference's image rows do not match its placeholder runs")
    if video_pixels is not None and video_offset != len(video_pixels):
        raise ValueError("the reference's video frames do not match its placeholder runs")
    return singles


def _encode(processor, images: np.ndarray, prompts: list[str], audio: list[np.ndarray] | None) -> dict:
    """The padded batch encoding: one image in the first row, two in the second."""
    rows = [[images[0]], [images[1], images[2]]]
    media: dict[str, object] = {} if audio is None else {"audio": list(audio)}
    encoded = processor(text=prompts, images=rows, padding=True, truncation=False,
                        return_tensors="pt", **media)
    # The fp32 reference widens bfloat16 processor pixels exactly.
    encoded["pixel_values"] = encoded["pixel_values"].float()
    return encoded


def _sgd_step(model, singles: list[dict], valid: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The summed next-token loss over every row, its backward, and one SGD
    step of learning rate LEARNING_RATE on every parameter, in place; returns
    the loss and the pixel inputs, which hold their gradients."""
    targets = int((valid[:, :-1] & valid[:, 1:]).sum())
    loss = torch.zeros((), device=model.device)
    pixel_inputs = []
    for single in singles:
        pixels = single["pixel_values"].clone().requires_grad_(requires_grad=True)
        pixel_inputs.append(pixels)
        predictions = model(**{**single, "pixel_values": pixels}, use_cache=False).logits[0, :-1]
        loss = loss + torch.nn.functional.cross_entropy(
            predictions, single["input_ids"][0, 1:], reduction="sum") / targets
    loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
    return loss.cpu(), pixel_inputs


def _write_forward_backward(model, processor, destination: Path, images: np.ndarray, prompts: list[str],
                            audio: list[np.ndarray] | None = None) -> None:
    """Actual wrapper forward, cached continuation, pixel gradient and SGD output.

    The padded batch encoding is what Dew consumes; every reference quantity
    comes from the reference model over one row of it at a time.
    """
    encoded = _encode(processor, images, prompts, audio)
    valid = encoded["attention_mask"].bool()
    vocab_size = model.config.get_text_config().vocab_size
    logits = torch.zeros((*encoded["input_ids"].shape, vocab_size))
    updated = torch.zeros_like(logits)
    continuation = []
    singles = _row_slices(encoded, model)
    for row, single in enumerate(singles):
        with torch.no_grad():
            logits[row, valid[row]] = model(**single, use_cache=False).logits[0]
            generated = model.generate(**single, max_new_tokens=3, do_sample=False,
                                       eos_token_id=None, use_cache=True, return_dict_in_generate=False)
            continuation.append(generated[0, single["input_ids"].shape[1]:])
    loss, pixel_inputs = _sgd_step(model, singles, valid)
    np.save(destination / "pixel_gradient.npy", torch.cat([pixels.grad for pixels in pixel_inputs]).numpy())
    with torch.no_grad():
        for row, single in enumerate(singles):
            updated[row, valid[row]] = model(**single, use_cache=False).logits[0]
    np.save(destination / "updated_logits.npy", updated.numpy())
    (destination / "training.json").write_text(json.dumps(
        {"loss": float(loss.detach()), "learning_rate": LEARNING_RATE}) + "\n")

    np.save(destination / "raw_images.npy", images)
    np.save(destination / "logits.npy", logits.numpy())
    np.save(destination / "continuation.npy", torch.stack(continuation).numpy())
    (destination / "prompts.json").write_text(json.dumps(prompts) + "\n")
    for key, value in encoded.items():
        np.save(destination / f"{key}.npy", value.numpy())
    print(destination, "bytes", sum(p.stat().st_size for p in destination.iterdir()),
          "tokens", tuple(encoded["input_ids"].shape), "pixel_values", tuple(encoded["pixel_values"].shape))




def write_gemma3n_native() -> None:
    """The actual Gemma3nProcessor over images and audio: MobileNet-v5 features,
    hard vocabulary ranges, fixed audio slots and per-layer input masking.

    The complete checkpoint and processor stay in gemma-3n-audio-tiny, written
    by tools/audio_wrapper_reference.py; only the references land here. This
    run needs timm for the vision encoder.
    """
    from transformers import Gemma3nForConditionalGeneration

    source = ROOT / "gemma-3n-audio-tiny"
    destination = ROOT / "gemma3n-native-tiny"
    destination.mkdir(parents=True, exist_ok=True)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        source, attn_implementation="eager").float().eval()
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    audio = [np.load(source / "waveform_0.npy"), np.load(source / "waveform_1.npy")]
    images = np.random.default_rng(2604).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    prompts = ["listen <audio> <image> ok", "say <image> <audio> again <image> ok"]
    _write_forward_backward(model, processor, destination, images, prompts, audio=audio)



def _copied(source: str, destination: str, config_class, model_class, **changes):
    """The fixture `source`'s model, its config's top-level fields set to `changes`, built eager
    in float32 with its weights loaded strictly, and saved with its config under `destination`.
    Returns the model and that directory."""
    directory = ROOT / destination
    directory.mkdir(parents=True, exist_ok=True)
    hf_config = config_class.from_dict(json.loads((ROOT / source / "config.json").read_text()) | changes)
    hf_config._attn_implementation = "eager"
    model = model_class(hf_config).float().eval()
    tensors = load_file(str(ROOT / source / "model.safetensors"))
    model.load_state_dict(tensors, strict=True)
    save_file(tensors, str(directory / "model.safetensors"))
    hf_config.save_pretrained(directory)
    return model, directory


def qwen3_vl_processor(directory: Path):
    """The Qwen3-VL processor over a 256-token tokenizer, saved to `directory` and read back as
    a checkpoint's is, and the tokenizer."""
    from transformers import Qwen3VLProcessor, Qwen3VLVideoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    special = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
               "<image_soft_token>": 200, "<video>": 201,
               "<start_of_image>": 202, "<end_of_image>": 203}
    tokenizer = _tokenizer(256, special, video_token="<video>",
                           vision_start_token="<start_of_image>", vision_end_token="<end_of_image>")
    processor = Qwen3VLProcessor(
        Qwen2VLImageProcessor(patch_size=8, temporal_patch_size=2, merge_size=2, do_resize=False),
        tokenizer, Qwen3VLVideoProcessor(patch_size=8, temporal_patch_size=2, merge_size=2))
    processor.save_pretrained(directory)
    return AutoProcessor.from_pretrained(directory, local_files_only=True), tokenizer


def write_gemma3_native() -> None:
    """The actual public processor and conditional model over unequal image rows."""
    model, destination = _copied("gemma3-tiny-mm", "gemma3-native-tiny", Gemma3Config,
                                 Gemma3ForConditionalGeneration, mm_tokens_per_image=4)
    special = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
               "<start_of_image>": 200, "<end_of_image>": 201, "<image_soft_token>": 202}
    tokenizer = _tokenizer(256, special)
    processor = Gemma3Processor(
        Gemma3ImageProcessorPil(size={"height": 28, "width": 28}), tokenizer, image_seq_length=4)
    processor.save_pretrained(destination)
    processor = AutoProcessor.from_pretrained(destination, local_files_only=True, backend="pil")
    generator = np.random.default_rng(903)
    images = generator.integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    prompts = ["token7 <start_of_image> token9",
               "token5 <start_of_image> token8 <start_of_image> token6"]
    _write_forward_backward(model, processor, destination, images, prompts)


def write_gemma4_native() -> None:
    """Released patch preprocessing, active clipping and frozen standardization."""
    from transformers import (
        Gemma4AudioFeatureExtractor,
        Gemma4Config,
        Gemma4ForConditionalGeneration,
        Gemma4Processor,
        Gemma4VideoProcessor,
    )
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

    source = ROOT / "gemma4-tiny-mm"
    destination = ROOT / "gemma4-native-tiny"
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    config["vision_config"]["use_clipped_linears"] = True
    config.update(video_token_id=56, audio_token_id=57, boa_token_id=58, eoa_token_index=59)
    hf_config = Gemma4Config.from_dict(config)
    hf_config._attn_implementation = "eager"
    torch.manual_seed(1303)
    model = Gemma4ForConditionalGeneration(hf_config).float().eval()
    tensors = load_file(str(source / "model.safetensors"))
    limits = {"input_min": -2.0, "input_max": 2.0, "output_min": -3.0, "output_max": 3.0}
    for name, buffer in model.named_buffers():
        leaf = name.split(".")[-1]
        if leaf in limits:
            buffer.fill_(limits[leaf])
            tensors[name] = buffer
    model.load_state_dict(tensors, strict=True)
    save_file(tensors, str(destination / "model.safetensors"))
    hf_config.save_pretrained(destination)
    special = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
               "<|video|>": 56, "<audio>": 57, "<start_of_audio>": 58, "<end_of_audio>": 59,
               "<image_soft_token>": 60, "<start_of_image>": 61, "<end_of_image>": 62}
    tokenizer = _tokenizer(64, special, audio_token="<audio>",
                           boa_token="<start_of_audio>", eoa_token="<end_of_audio>")
    processor = Gemma4Processor(
        Gemma4AudioFeatureExtractor(),
        Gemma4ImageProcessorPil(patch_size=8, pooling_kernel_size=2,
                                max_soft_tokens=70, do_resize=False),
        tokenizer, Gemma4VideoProcessor())
    processor.save_pretrained(destination)
    processor = AutoProcessor.from_pretrained(destination, local_files_only=True)
    images = np.random.default_rng(1304).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    prompts = ["token7 <image_soft_token> token9",
               "token5 <image_soft_token> token8 <image_soft_token> token6"]
    _write_forward_backward(model, processor, destination, images, prompts)



def unified_draw(config, seed: int, *, side: int = 2) -> dict[str, np.ndarray]:
    """Unequal image, video and audio runs, including holes in each media item's padded slots."""
    rng = np.random.default_rng(seed)
    image_tokens = side ** 2
    tokens = rng.integers(4, 50, (2, image_tokens + 10), dtype=np.int32)
    tokens[:, 0] = config.bos_token_id
    tokens[0, 2:2 + image_tokens], tokens[1, 2:5] = config.image_token_id, config.video_token_id
    tokens[0, image_tokens + 3:image_tokens + 6] = config.audio_token_id
    tokens[1, 6:8] = config.audio_token_id
    tokens[1, 10:] = config.pad_token_id
    types = np.where(tokens == config.image_token_id, 1,
                     np.where(tokens == config.video_token_id, 2,
                              np.where(tokens == config.audio_token_id, 3, 0))).astype(np.int32)
    vision = config.vision_config
    patch_width = 3 * (vision.patch_size * vision.pooling_kernel_size) ** 2
    positions = np.full((1, image_tokens + 2, 2), -1, np.int32)
    valid_slots = np.delete(np.arange(image_tokens + 2), [1, image_tokens])
    positions[0, valid_slots] = np.stack(
        [np.tile(np.arange(side), side), np.repeat(np.arange(side), side)], -1)
    positions[0, 1] = (-1, 0)
    return {"input_ids": tokens, "attention_mask": tokens != config.pad_token_id,
            "mm_token_type_ids": types,
            "pixel_values": rng.random((1, image_tokens + 2, patch_width), dtype=np.float32),
            "image_position_ids": positions,
            "pixel_values_videos": rng.random((1, 1, 4, patch_width), dtype=np.float32),
            "video_position_ids": np.asarray([[[[0, 0], [1, 0], [0, -1], [0, 1]]]], np.int32),
            "input_features": rng.standard_normal((2, 4, config.audio_config.audio_embed_dim), np.float32),
            "input_features_mask": np.asarray([[1, 0, 1, 1], [1, 1, 0, 0]], bool)}


def write_gemma4_unified_native() -> None:
    """The encoder-free wrapper's tiny source, with the release's public image/audio processor."""
    from hf_reference import scatter_weights
    from transformers import (
        Gemma4UnifiedAudioFeatureExtractor,
        Gemma4UnifiedConfig,
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedImageProcessor,
        Gemma4UnifiedProcessor,
        Gemma4UnifiedVideoProcessor,
    )

    destination = ROOT / "gemma4-unified-native-tiny"
    destination.mkdir(parents=True, exist_ok=True)
    config = Gemma4UnifiedConfig(
        text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                     "global_head_dim": 16, "num_global_key_value_heads": 1, "max_position_embeddings": 128,
                     "layer_types": ["sliding_attention", "full_attention"] * 2, "sliding_window": 8,
                     "num_kv_shared_layers": 2, "attention_k_eq_v": True, "use_double_wide_mlp": True,
                     "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2},
        vision_config={"patch_size": 2, "pooling_kernel_size": 2, "mm_embed_dim": 24,
                       "output_proj_dims": 24, "mm_posemb_size": 32},
        audio_config={"audio_embed_dim": 16}, image_token_id=60, video_token_id=56,
        audio_token_id=57, boa_token_id=58, eoa_token_index=59,
        pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = "eager"
    torch.manual_seed(2410)
    model = Gemma4UnifiedForConditionalGeneration(config).float().eval()
    scatter_weights(model)
    with torch.no_grad():
        for index, layer in enumerate(model.model.language_model.layers):
            layer.layer_scalar.fill_(0.8 + index * 0.125)
    model.save_pretrained(destination, safe_serialization=True)
    config.to_json_file(destination / "config.json", use_diff=False)
    # The decoder alone, as the wrapper states it: the committed config its text family's kernel routes
    # trace (tools/kernel_routes.py), which find a family's fixture by its top-level model_type.
    stated = json.loads((destination / "config.json").read_text())
    text = ROOT / "gemma4-unified-text-tiny"
    text.mkdir(parents=True, exist_ok=True)
    (text / "config.json").write_text(json.dumps(
        {**stated["text_config"], "transformers_version": stated["transformers_version"]},
        indent=2, sort_keys=True) + "\n")
    special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
               "<|video|>": 56, "<audio>": 57, "<start_of_audio>": 58, "<end_of_audio>": 59,
               "<image_soft_token>": 60, "<start_of_image>": 61, "<end_of_image>": 62}
    tokenizer = _tokenizer(64, special, audio_token="<audio>", video_token="<|video|>",
                           boa_token="<start_of_audio>", eoa_token="<end_of_audio>")
    processor = Gemma4UnifiedProcessor(
        Gemma4UnifiedAudioFeatureExtractor(feature_size=16, audio_samples_per_token=16),
        Gemma4UnifiedImageProcessor(patch_size=2, pooling_kernel_size=2, max_soft_tokens=70, do_resize=False),
        tokenizer, Gemma4UnifiedVideoProcessor(patch_size=2, pooling_kernel_size=2, max_soft_tokens=70),
        audio_ms_per_token=1)
    processor.save_pretrained(destination)
    inputs = unified_draw(config, 0)
    with torch.no_grad():
        result = model(**{name: torch.from_numpy(value) for name, value in inputs.items()}, use_cache=False)
    print(f"{destination.name}: forward shape {tuple(result.logits.shape)}", flush=True)


def write_float64_truths(references: str, checkpoint: str) -> None:
    """The float64 truth of what `_write_forward_backward` recorded in `references` from the fixture
    `checkpoint`'s model: its float64 twin (tools/torch_precision.py) over the same stored batch
    encoding, one row at a time: its logits, then its summed next-token loss, pixel gradient and logits
    after one SGD step of LEARNING_RATE, in logits_f64.npy, training_f64.json, pixel_gradient_f64.npy
    and updated_logits_f64.npy. The Dew side's checks hold Dew and the fp32 reference to it
    (tests/test_multimodal_model.py)."""
    from transformers import AutoModelForImageTextToText

    from tools.torch_precision import float64, float64_twin

    destination, source = ROOT / references, ROOT / checkpoint
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.float32,
                                                        attn_implementation="eager").eval()
    recorded = ("logits", "updated_logits", "pixel_gradient", "continuation", "raw_images")
    encoded = {path.stem: torch.from_numpy(np.load(path)) for path in destination.glob("*.npy")
               if path.stem not in recorded and not path.stem.endswith("_f64")}
    twin = float64_twin(model)
    with float64():
        widened = {name: value.double() if value.is_floating_point() else value
                   for name, value in encoded.items()}
        valid = widened["attention_mask"].bool()
        singles = _row_slices(widened, twin)
        shape = (*widened["input_ids"].shape, twin.config.get_text_config().vocab_size)
        logits, updated = torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64)
        with torch.no_grad():
            for row, single in enumerate(singles):
                logits[row, valid[row]] = twin(**single, use_cache=False).logits[0]
        loss, pixel_inputs = _sgd_step(twin, singles, valid)
        with torch.no_grad():
            for row, single in enumerate(singles):
                updated[row, valid[row]] = twin(**single, use_cache=False).logits[0]
    np.save(destination / "pixel_gradient_f64.npy",
            torch.cat([pixels.grad for pixels in pixel_inputs]).numpy())
    np.save(destination / "logits_f64.npy", logits.numpy())
    np.save(destination / "updated_logits_f64.npy", updated.numpy())
    (destination / "training_f64.json").write_text(json.dumps({"loss": float(loss.detach())}) + "\n")
    print(destination, "float64 loss", float(loss.detach()), flush=True)


def write_llama4_native() -> None:
    """The actual Llama4Processor tiling: local tiles, a global tile and separators."""
    from transformers import Llama4Config, Llama4ForConditionalGeneration, Llama4Processor
    from transformers.models.llama4.image_processing_llama4 import Llama4ImageProcessor

    model, destination = _copied("llama4-tiny-mm", "llama4-native-tiny", Llama4Config,
                                 Llama4ForConditionalGeneration)
    special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
               "<|image_start|>": 90, "<|image_end|>": 91, "<|patch|>": 92, "<|image|>": 93,
               "<|tile_x_separator|>": 94, "<|tile_y_separator|>": 95}
    tokenizer = _tokenizer(model.config.text_config.vocab_size, special)
    processor = Llama4Processor(
        Llama4ImageProcessor(size={"height": 28, "width": 28}, max_patches=4, resize_to_max_canvas=False),
        tokenizer, patch_size=14, pixel_shuffle_ratio=0.5)
    processor.save_pretrained(destination)
    processor = AutoProcessor.from_pretrained(destination, local_files_only=True)
    images = np.random.default_rng(1707).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    prompts = ["token7 <|image|> token9", "token5 <|image|> token8 <|image|> token6"]
    _write_forward_backward(model, processor, destination, images, prompts)



def write_qwen35_native() -> None:
    """The actual Qwen3VLProcessor and hybrid model, including spatial M-RoPE."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    model, destination = _copied("qwen35-tiny-mm", "qwen35-native-tiny", Qwen3_5Config,
                                 Qwen3_5ForConditionalGeneration)
    processor, _ = qwen3_vl_processor(destination)
    images = np.random.default_rng(2305).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    prompts = ["token7 <start_of_image><image_soft_token><end_of_image> token9",
               "token5 <start_of_image><image_soft_token><end_of_image> token8 "
               "<start_of_image><image_soft_token><end_of_image> token6"]
    _write_forward_backward(model, processor, destination, images, prompts)



if __name__ == "__main__":
    if sys.argv[1:] == ["gemma4-unified"]:
        write_gemma4_unified_native()
        sys.exit()
    if sys.argv[1:2] == ["float64"]:
        # Which references each family's test reads, and the checkpoint they were written from: the
        # tests' declaration, data alone.
        sys.path.insert(0, str(ROOT.parents[1]))
        from multimodal_families import FAMILIES

        for family in sys.argv[2:] or FAMILIES:
            write_float64_truths(*FAMILIES[family])
        sys.exit()
    write_gemma3_native()
    write_gemma3n_native()
    write_gemma4_native()
    write_llama4_native()
    write_qwen35_native()
