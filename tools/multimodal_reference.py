"""Write processor-to-wrapper references from local tiny Gemma3 weights.

Run with transformers 5.16.1, Torch and tokenizers installed:
    PYTHONPATH=src python tools/multimodal_reference.py

No network access or pretrained downloads are used. The existing Gemma3
fixture supplies weights; four soft tokens per image exercise bidirectional
image attention, and unequal image counts exercise numeric batch alignment.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AddedToken, AutoProcessor, Gemma3Config, Gemma3ForConditionalGeneration, Gemma3Processor, GemmaTokenizer
from transformers.models.gemma3.image_processing_pil_gemma3 import Gemma3ImageProcessorPil


ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"


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
    image_offset = clip_offset = 0
    singles = []
    for row in range(encoded["input_ids"].shape[0]):
        ids = encoded["input_ids"][row][valid[row]]
        single = {"input_ids": ids[None], "attention_mask": torch.ones_like(ids)[None]}
        for name in ("token_type_ids", "mm_token_type_ids"):
            if name in encoded:
                single[name] = encoded[name][row][valid[row]][None]
        images = _runs(ids, image_id)
        if grid is None:
            single["pixel_values"] = encoded["pixel_values"][image_offset:image_offset + images]
            if "image_position_ids" in encoded:
                single["image_position_ids"] = encoded["image_position_ids"][image_offset:image_offset + images]
        else:
            patches = grid.prod(dim=1)
            start = int(patches[:image_offset].sum())
            stop = int(patches[:image_offset + images].sum())
            single["pixel_values"] = encoded["pixel_values"][start:stop]
            single["image_grid_thw"] = grid[image_offset:image_offset + images]
        image_offset += images
        if "input_features" in encoded:
            clips = _runs(ids, model.config.audio_token_id)
            single["input_features"] = encoded["input_features"][clip_offset:clip_offset + clips]
            single["input_features_mask"] = encoded["input_features_mask"][clip_offset:clip_offset + clips]
            clip_offset += clips
        singles.append(single)
    assert image_offset == (encoded["pixel_values"].shape[0] if grid is None else grid.shape[0])
    return singles


def _write_forward_backward(model, processor, destination: Path, images: np.ndarray, prompts: list[str],
                            audio: list[np.ndarray] | None = None) -> None:
    """Actual wrapper forward, cached continuation, pixel gradient and SGD output.

    The padded batch encoding is what Dew consumes; every reference quantity
    comes from the reference model over one row of it at a time.
    """
    rows = [[images[0]], [images[1], images[2]]]
    media: dict[str, object] = {} if audio is None else {"audio": list(audio)}
    encoded = processor(text=prompts, images=rows, padding=True, truncation=False, return_tensors="pt", **media)
    # The fp32 reference widens bfloat16 processor pixels exactly.
    encoded["pixel_values"] = encoded["pixel_values"].float()
    valid = encoded["attention_mask"].bool()
    vocab_size = model.config.get_text_config().vocab_size
    logits = torch.zeros((*encoded["input_ids"].shape, vocab_size))
    updated = torch.zeros_like(logits)
    targets = int((valid[:, :-1] & valid[:, 1:]).sum())
    loss = torch.zeros(())
    continuation, pixel_inputs = [], []
    singles = _row_slices(encoded, model)
    for row, single in enumerate(singles):
        with torch.no_grad():
            logits[row, valid[row]] = model(**single, use_cache=False).logits[0]
            generated = model.generate(**single, max_new_tokens=3, do_sample=False,
                                       eos_token_id=None, use_cache=True, return_dict_in_generate=False)
            continuation.append(generated[0, single["input_ids"].shape[1]:])
        pixels = single["pixel_values"].clone().requires_grad_(True)
        pixel_inputs.append(pixels)
        predictions = model(**{**single, "pixel_values": pixels}, use_cache=False).logits[0, :-1]
        loss = loss + torch.nn.functional.cross_entropy(
            predictions, single["input_ids"][0, 1:], reduction="sum") / targets
    loss.backward()
    np.save(destination / "pixel_gradient.npy", torch.cat([pixels.grad for pixels in pixel_inputs]).numpy())
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1e-4)
        for row, single in enumerate(singles):
            updated[row, valid[row]] = model(**single, use_cache=False).logits[0]
    np.save(destination / "updated_logits.npy", updated.numpy())
    (destination / "training.json").write_text(json.dumps({"loss": float(loss.detach()), "learning_rate": 1e-4}) + "\n")

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
    model = Gemma3nForConditionalGeneration.from_pretrained(source, attn_implementation="eager").float().eval()
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    audio = [np.load(source / "waveform_0.npy"), np.load(source / "waveform_1.npy")]
    images = np.random.default_rng(2604).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    prompts = ["listen <audio> <image> ok", "say <image> <audio> again <image> ok"]
    _write_forward_backward(model, processor, destination, images, prompts, audio=audio)



def write_gemma3_native() -> None:
    """The actual public processor and conditional model over unequal image rows."""
    source = ROOT / "gemma3-tiny-mm"
    destination = ROOT / "gemma3-native-tiny"
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    config["mm_tokens_per_image"] = 4
    hf_config = Gemma3Config.from_dict(config)
    hf_config._attn_implementation = "eager"
    model = Gemma3ForConditionalGeneration(hf_config).float().eval()
    tensors = load_file(str(source / "model.safetensors"))
    model.load_state_dict(tensors, strict=True)
    save_file(tensors, str(destination / "model.safetensors"))
    hf_config.save_pretrained(destination)

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
    from transformers import (Gemma4AudioFeatureExtractor, Gemma4Config,
                              Gemma4ForConditionalGeneration, Gemma4Processor,
                              Gemma4VideoProcessor)
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



def write_llama4_native() -> None:
    """The actual Llama4Processor tiling: local tiles, a global tile and separators."""
    from transformers import Llama4Config, Llama4ForConditionalGeneration, Llama4Processor
    from transformers.models.llama4.image_processing_llama4 import Llama4ImageProcessor

    source = ROOT / "llama4-tiny-mm"
    destination = ROOT / "llama4-native-tiny"
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    hf_config = Llama4Config.from_dict(config)
    hf_config._attn_implementation = "eager"
    model = Llama4ForConditionalGeneration(hf_config).float().eval()
    tensors = load_file(str(source / "model.safetensors"))
    model.load_state_dict(tensors, strict=True)
    save_file(tensors, str(destination / "model.safetensors"))
    hf_config.save_pretrained(destination)
    special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
               "<|image_start|>": 90, "<|image_end|>": 91, "<|patch|>": 92, "<|image|>": 93,
               "<|tile_x_separator|>": 94, "<|tile_y_separator|>": 95}
    tokenizer = _tokenizer(config["text_config"]["vocab_size"], special)
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
    from transformers import Qwen3VLProcessor, Qwen3VLVideoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    source = ROOT / "qwen35-tiny-mm"
    destination = ROOT / "qwen35-native-tiny"
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    hf_config = Qwen3_5Config.from_dict(config)
    hf_config._attn_implementation = "eager"
    model = Qwen3_5ForConditionalGeneration(hf_config).float().eval()
    tensors = load_file(str(source / "model.safetensors"))
    model.load_state_dict(tensors, strict=True)
    save_file(tensors, str(destination / "model.safetensors"))
    hf_config.save_pretrained(destination)
    special = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
               "<image_soft_token>": 200, "<video>": 201,
               "<start_of_image>": 202, "<end_of_image>": 203}
    tokenizer = _tokenizer(256, special, video_token="<video>",
                           vision_start_token="<start_of_image>", vision_end_token="<end_of_image>")
    processor = Qwen3VLProcessor(
        Qwen2VLImageProcessor(patch_size=8, temporal_patch_size=2, merge_size=2, do_resize=False),
        tokenizer, Qwen3VLVideoProcessor(patch_size=8, temporal_patch_size=2, merge_size=2))
    processor.save_pretrained(destination)
    processor = AutoProcessor.from_pretrained(destination, local_files_only=True)
    images = np.random.default_rng(2305).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    prompts = ["token7 <start_of_image><image_soft_token><end_of_image> token9",
               "token5 <start_of_image><image_soft_token><end_of_image> token8 <start_of_image><image_soft_token><end_of_image> token6"]
    _write_forward_backward(model, processor, destination, images, prompts)



if __name__ == "__main__":
    write_gemma3_native()
    write_gemma3n_native()
    write_gemma4_native()
    write_llama4_native()
    write_qwen35_native()
