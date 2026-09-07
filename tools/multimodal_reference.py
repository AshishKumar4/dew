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
    return tokenizer



def _write_forward_backward(model, processor, destination: Path, images: np.ndarray, prompts: list[str]) -> None:
    """Actual wrapper forward, cached continuation, pixel gradient and SGD output."""
    encoded = processor(text=prompts, images=[[images[0]], [images[1], images[2]]],
                        padding=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded, use_cache=True)
        logits = output.logits
        cache = output.past_key_values
        attention_mask = encoded["attention_mask"]
        next_logits = logits[:, -1]
        continuation = []
        for _ in range(3):
            token = next_logits.argmax(-1)
            continuation.append(token)
            attention_mask = torch.cat([attention_mask, torch.ones_like(token[:, None])], dim=1)
            output = model(input_ids=token[:, None], attention_mask=attention_mask,
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            next_logits = output.logits[:, -1]
    train_pixels = encoded["pixel_values"].clone().requires_grad_(True)
    training = {**encoded, "pixel_values": train_pixels}
    predictions = model(**training, use_cache=False).logits[:, :-1]
    labels = encoded["input_ids"][:, 1:]
    valid = encoded["attention_mask"][:, :-1].bool() & encoded["attention_mask"][:, 1:].bool()
    losses = torch.nn.functional.cross_entropy(predictions.reshape(-1, predictions.shape[-1]),
                                                labels.reshape(-1), reduction="none").reshape(labels.shape)
    loss = (losses * valid).sum() / valid.sum()
    loss.backward()
    np.save(destination / "pixel_gradient.npy", train_pixels.grad.numpy())
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1e-4)
        updated = model(**encoded, use_cache=False).logits
    np.save(destination / "updated_logits.npy", updated.numpy())
    (destination / "training.json").write_text(json.dumps({"loss": float(loss.detach()), "learning_rate": 1e-4}) + "\n")

    np.save(destination / "raw_images.npy", images)
    np.save(destination / "logits.npy", logits.numpy())
    np.save(destination / "continuation.npy", torch.stack(continuation, dim=1).numpy())
    (destination / "prompts.json").write_text(json.dumps(prompts) + "\n")
    for key, value in encoded.items():
        np.save(destination / f"{key}.npy", value.numpy())
    print(destination, "bytes", sum(p.stat().st_size for p in destination.iterdir()),
          "tokens", tuple(encoded["input_ids"].shape), "pixel_values", tuple(encoded["pixel_values"].shape))


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



if __name__ == "__main__":
    write_gemma3_native()
