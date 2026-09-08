"""Write tiny Qwen3.8 checkpoint references using the released Qwen3.5 classes.

No full weights are downloaded. Source metadata and revisions are committed in
qwen38-source. Tiny model geometry, vocabulary and processor size bounds keep
the reference runs local; video frame sampling is disabled for temporal-padding
coverage. Run with the CPU vision environment and Transformers 5.16.1. MTP
is recorded separately because Transformers does not execute its tensors.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, Qwen2Tokenizer, Qwen3_5Config
from transformers import Qwen3_5MoeTextConfig
from qwen_mtp_reference import QwenMTP, prediction_inputs

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
SOURCES = ROOT / "qwen38-source"


def tokenizer(source: Path) -> Qwen2Tokenizer:
    metadata = json.loads((source / "tokenizer_config.json").read_text())
    special = [item["content"] for item in metadata["added_tokens_decoder"].values()]
    vocab = {word: index for index, word in enumerate(sorted(ByteLevel.alphabet()))}
    for word in special:
        if word not in vocab:
            vocab[word] = len(vocab)
    template_file = source / "chat_template.jinja"
    template = template_file.read_text() if template_file.exists() else metadata["chat_template"]
    return Qwen2Tokenizer(vocab=vocab, merges=[], eos_token=metadata["eos_token"],
                          pad_token=metadata["pad_token"], bos_token=metadata.get("bos_token"),
                          additional_special_tokens=special, chat_template=template, padding_side="left")


def tiny_config(source: Path, tokenizer: Qwen2Tokenizer) -> dict:
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    text.update(hidden_size=32, num_hidden_layers=4, num_attention_heads=3,
                num_key_value_heads=1, head_dim=32, max_position_embeddings=512,
                linear_num_key_heads=1, linear_num_value_heads=3,
                linear_key_head_dim=8, linear_value_head_dim=8,
                vocab_size=len(tokenizer), bos_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    text["layer_types"] = text["layer_types"][:4]
    if "num_experts" in text:
        text.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
                    shared_expert_intermediate_size=24)
    else:
        text["intermediate_size"] = 112
    if "vision_config" in config:
        config["vision_config"].update(depth=2, hidden_size=32, intermediate_size=64,
                                       num_heads=4, out_hidden_size=32, num_position_embeddings=16)
        for name, symbol in (("image_token_id", "<|image_pad|>"), ("video_token_id", "<|video_pad|>"),
                             ("vision_start_token_id", "<|vision_start|>"), ("vision_end_token_id", "<|vision_end|>")):
            config[name] = tokenizer.convert_tokens_to_ids(symbol)
    return config


def scatter(model: torch.nn.Module) -> None:
    """Nondegenerate parameters expose routing, norms and both output gates."""
    generator = torch.Generator().manual_seed(3181)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("A_log"):
                parameter.uniform_(-1, 1, generator=generator)
            elif parameter.ndim == 1:
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.025)
            else:
                parameter.normal_(0, 0.08, generator=generator)


def write_video_reference(model, processor, destination: Path) -> None:
    """Interleave images and timestamped videos with odd frame counts and unequal grids."""
    rng = np.random.default_rng(3883)
    videos = [rng.integers(0, 256, shape, dtype=np.uint8)
              for shape in ((5, 64, 96, 3), (3, 96, 64, 3))]
    metadata = [{"fps": fps, "total_num_frames": len(video), "frames_indices": list(range(len(video)))}
                for video, fps in zip(videos, (5.0, 3.0))]
    images = np.load(destination / "images.npy")
    image = "<|vision_start|><|image_pad|><|vision_end|>"
    video = "<|vision_start|><|video_pad|><|vision_end|>"
    prompts = [f"a {image} b {video} c", f"d {video} e {image} f"]
    encoded = processor(text=prompts, images=[[images[0]], [images[1]]], videos=videos,
                        video_metadata=metadata, padding=True, return_tensors="pt")
    for index, frames in enumerate(videos):
        np.save(destination / f"video_{index}.npy", frames)
    (destination / "video_inputs.json").write_text(json.dumps({"prompts": prompts, "metadata": metadata}, indent=2) + "\n")
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=3, do_sample=False,
                                   eos_token_id=None, pad_token_id=processor.tokenizer.pad_token_id)
    logits, hidden, embeddings, positions = prediction_inputs(model, encoded)
    valid = encoded["attention_mask"].bool()
    admitted = valid[:, :-1] & valid[:, 1:]
    mtp_logits = model.lm_head(model.mtp(hidden[:, :-1], embeddings[:, 1:], positions[..., 1:], admitted))
    main_ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                encoded["input_ids"][:, 1:].reshape(-1), reduction="none")
    depth_ce = torch.nn.functional.cross_entropy(mtp_logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                 encoded["input_ids"][:, 2:].reshape(-1), reduction="none")
    depth_valid = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    loss = ((main_ce.reshape(admitted.shape) * admitted).sum()
            + .2 * (depth_ce.reshape(depth_valid.shape) * depth_valid).sum()) / admitted.sum()
    loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1e-4)
        updated = model(**encoded, use_cache=False).logits
    np.savez(destination / "video_reference.npz", **{key: value.numpy() for key, value in encoded.items()},
             logits=logits.detach().numpy(), mtp_logits=mtp_logits.detach().numpy(),
             loss=loss.detach().numpy(), updated_logits=updated.numpy(), generated=generated.numpy())



def write_reference(kind: str) -> None:
    source = SOURCES / kind
    destination = ROOT / f"qwen38-{kind}-tiny"
    destination.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(source)
    config = tiny_config(source, tok)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.manual_seed(3180)
    processor = None
    if kind == "dense":
        hf_config = Qwen3_5Config.from_dict(copy.deepcopy(config))
        hf_config._attn_implementation = "eager"
        model = AutoModelForImageTextToText.from_config(hf_config).float().eval()
        image_config = json.loads((source / "preprocessor_config.json").read_text())
        image_config["size"] = {"shortest_edge": 4096, "longest_edge": 24576}
        # Retain the actual split processor-file layout from the source.
        tok.save_pretrained(destination)
        (destination / "preprocessor_config.json").write_text(json.dumps(image_config, indent=2) + "\n")
        video_config = json.loads((source / "video_preprocessor_config.json").read_text())
        video_config.update(size={"shortest_edge": 4096, "longest_edge": 24576}, do_sample_frames=False)
        (destination / "video_preprocessor_config.json").write_text(json.dumps(video_config, indent=2) + "\n")
        images = np.random.default_rng(3182).integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
        prompts = ["a <|vision_start|><|image_pad|><|vision_end|> b",
                   "xy <|vision_start|><|image_pad|><|vision_end|> z"]
        processor = AutoProcessor.from_pretrained(destination, local_files_only=True)
        encoded = processor(text=prompts, images=[[images[0]], [images[1]]], padding=True, return_tensors="pt")
        np.save(destination / "images.npy", images)
    else:
        hf_config = Qwen3_5MoeTextConfig.from_dict(copy.deepcopy(config))
        hf_config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(hf_config).float().eval()
        tok.save_pretrained(destination)
        prompts = ["abcd ef", "wx yz uv"]
        encoded = tok(prompts, padding=True, return_tensors="pt")
    model.mtp = QwenMTP(hf_config.get_text_config())
    scatter(model)
    initial_state = {name: value.detach().contiguous().clone() for name, value in model.state_dict().items()}
    save_file(initial_state, str(destination / "model.safetensors"))
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "prompts.json").write_text(json.dumps(prompts) + "\n")
    generation = json.loads((source / "generation_config.json").read_text())
    generation.update(eos_token_id=[tok.eos_token_id, tok.pad_token_id], pad_token_id=tok.pad_token_id,
                      bos_token_id=tok.pad_token_id)
    (destination / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")
    with torch.no_grad():
        logits = model(**encoded, use_cache=False).logits
        sequence = model.generate(**encoded, max_new_tokens=3, do_sample=False, eos_token_id=None,
                                  pad_token_id=tok.pad_token_id)
    valid = encoded["attention_mask"].bool()
    target_valid = valid[:, 1:] & valid[:, :-1]
    predicted = model(**encoded, use_cache=False).logits[:, :-1]
    ce = torch.nn.functional.cross_entropy(predicted.reshape(-1, predicted.shape[-1]),
                                            encoded["input_ids"][:, 1:].reshape(-1), reduction="none")
    loss = (ce.reshape(target_valid.shape) * target_valid).sum() / target_valid.sum()
    loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1e-4)
        updated = model(**encoded, use_cache=False).logits
    np.savez(destination / "reference.npz", **{name: value.numpy() for name, value in encoded.items()},
             logits=logits.numpy(), updated_logits=updated.numpy(), generated=sequence.numpy(),
             loss=np.float32(loss.detach()), learning_rate=np.float32(1e-4))
    print(kind, "parameters", sum(parameter.numel() for parameter in model.parameters()),
          "tokens", tuple(encoded["input_ids"].shape))
    # Independently qualify the shipped MTP composition and its trainable loss.
    model.load_state_dict(initial_state, strict=True)
    model.zero_grad(set_to_none=True)
    main, hidden, embeddings, positions = prediction_inputs(model, encoded)
    admitted = valid[:, :-1] & valid[:, 1:]
    mtp_hidden = model.mtp(hidden[:, :-1], embeddings[:, 1:], positions[..., 1:], admitted)
    mtp_logits = model.lm_head(mtp_hidden)
    mtp_expected = mtp_logits.detach().numpy()
    main_ce = torch.nn.functional.cross_entropy(main[:, :-1].reshape(-1, main.shape[-1]),
                                                encoded["input_ids"][:, 1:].reshape(-1), reduction="none")
    depth_ce = torch.nn.functional.cross_entropy(mtp_logits[:, :-1].reshape(-1, main.shape[-1]),
                                                 encoded["input_ids"][:, 2:].reshape(-1), reduction="none")
    depth_valid = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    denominator = target_valid.sum()
    total = ((main_ce.reshape(target_valid.shape) * target_valid).sum()
             + 0.2 * (depth_ce.reshape(depth_valid.shape) * depth_valid).sum()) / denominator
    total.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1e-4)
        main_after, hidden_after, embed_after, pos_after = prediction_inputs(model, encoded)
        mtp_after = model.lm_head(model.mtp(hidden_after[:, :-1], embed_after[:, 1:],
                                           pos_after[..., 1:], admitted))
    np.savez(destination / "mtp_reference.npz", logits=mtp_expected,
             hidden=hidden.detach().numpy(), embeddings=embeddings.detach().numpy(),
             positions=positions.detach().numpy(), valid=admitted.numpy(),
             loss=total.detach().numpy(), updated_logits=main_after.numpy(),
             updated_mtp_logits=mtp_after.numpy())
    if processor is not None:
        model.load_state_dict(initial_state, strict=True)
        model.zero_grad(set_to_none=True)
        write_video_reference(model, processor, destination)



if __name__ == "__main__":
    torch.set_num_threads(1)
    for family in ("dense", "moe"):
        write_reference(family)
