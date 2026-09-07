#!/usr/bin/env python3
"""Write tiny audio-bearing Gemma 3n / Gemma 4 source directories and references.

Every weight initializes locally. Each directory contains config.json,
model.safetensors, generation_config.json, and the real checkpoint
processor (tokenizer, feature extractor, image processor) saved through
Transformers so the shared loader reads it offline. reference.npz holds
the processor outputs, the conditional model's logits and greedy
continuation over two waveforms of unequal length.

The Gemma 4 audio tower runs with sdpa. Under eager, Transformers 5.16.1
hands Gemma4AudioModel an additive float mask and the tower negates it with
logical_not (modeling_gemma4.py:343-346), which inverts the audio mask.
The text model stays eager.

Run with Transformers 5.16.1 and timm 1.0.29 / torchvision 0.29.0 on the path:
  python tools/audio_wrapper_reference.py --out tests/fixtures/hf
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (Gemma3nAudioConfig, Gemma3nAudioFeatureExtractor, Gemma3nConfig,
                          Gemma3nProcessor, Gemma3nTextConfig, Gemma3nVisionConfig, Gemma4AudioConfig,
                          Gemma4AudioFeatureExtractor, Gemma4Config, Gemma4ImageProcessor, Gemma4Processor,
                          Gemma4TextConfig, Gemma4VideoProcessor, Gemma4VisionConfig, PreTrainedTokenizerFast,
                          SiglipImageProcessor)
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nForConditionalGeneration
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


SEED = 2718
WORDS = {0: "<pad>", 1: "</s>", 2: "<s>", 3: "<boi>", 4: "<boa>", 5: "listen", 6: "<unk>", 7: "ok",
         8: "say", 9: "again", 48: "<eoi>", 49: "<image>", 50: "<|video|>", 56: "<eoa>", 57: "<audio>"}


def tokenizer():
    vocab = {f"tok{i}": i for i in range(64)}
    for index, word in WORDS.items():
        del vocab[f"tok{index}"]
        vocab[word] = index
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        padding_side="left",
        extra_special_tokens={"image_token": "<image>", "audio_token": "<audio>", "boi_token": "<boi>",
                              "eoi_token": "<eoi>", "boa_token": "<boa>", "eoa_token": "<eoa>"})


def gemma3n():
    text = Gemma3nTextConfig(
        vocab_size=64, vocab_size_per_layer_input=48, hidden_size=32,
        intermediate_size=[48, 48, 64, 64], num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
        layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        sliding_window=4, max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=1e6,
        rope_local_base_freq=1e4, final_logit_softcapping=5.0, hidden_size_per_layer_input=8,
        altup_num_inputs=3, altup_active_idx=0, altup_coef_clip=120.0, altup_correct_scale=True,
        num_kv_shared_layers=1, laurel_rank=8, activation_sparsity_pattern=[0.0, 0.0, 0.0, 0.0],
        tie_word_embeddings=True)
    vision = Gemma3nVisionConfig(hidden_size=2048, vocab_size=8, vocab_offset=48,
                                 model_args={"channel_multiplier": 1 / 64, "stem_size": 8, "msfa_output_resolution": 2})
    audio = Gemma3nAudioConfig(
        input_feat_size=16, hidden_size=16, conf_num_hidden_layers=2, conf_num_attention_heads=2,
        conf_attention_chunk_size=4, conf_attention_context_left=5, conf_attention_context_right=2,
        sscp_conv_channel_size=(8, 4), conf_reduction_factor=2, vocab_offset=56, vocab_size=8, rms_norm_eps=1e-5)
    config = Gemma3nConfig(text_config=text, vision_config=vision, audio_config=audio,
                           image_token_id=49, audio_token_id=57, boi_token_id=3, eoi_token_id=48,
                           boa_token_id=4, eoa_token_id=56, vision_soft_tokens_per_image=4,
                           audio_soft_tokens_per_image=3, pad_token_id=0, bos_token_id=2, eos_token_id=1)
    config._attn_implementation = "eager"
    torch.manual_seed(SEED)
    model = Gemma3nForConditionalGeneration(config).eval()
    processor = Gemma3nProcessor(Gemma3nAudioFeatureExtractor(feature_size=16),
                                 SiglipImageProcessor(size={"height": 32, "width": 32}), tokenizer(),
                                 audio_seq_length=3, image_seq_length=4)
    return config, model, processor


def gemma4():
    # The same dense decoder shape as tests/fixtures/hf/gemma4-tiny-mm.
    text = Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_activation="gelu_pytorch_tanh",
        attention_k_eq_v=True, sliding_window=4, hidden_size_per_layer_input=0, num_kv_shared_layers=0,
        per_layer_config={"2": {"head_dim": 16, "num_key_value_heads": 1}}, max_position_embeddings=64,
        rms_norm_eps=1e-6, final_logit_softcapping=30.0, tie_word_embeddings=True, bos_token_id=2,
        eos_token_id=1, pad_token_id=0, use_bidirectional_attention="vision", use_double_wide_mlp=False,
        rope_parameters={"full_attention": {"rope_type": "proportional", "rope_theta": 1e6,
                                            "partial_rotary_factor": 0.25},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}})
    vision = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                pooling_kernel_size=2, patch_size=8, position_embedding_size=64,
                                standardize=True)
    audio = Gemma4AudioConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2, subsampling_conv_channels=(16, 4),
        output_proj_dims=24, attention_chunk_size=4, attention_context_left=5, attention_context_right=2,
        use_clipped_linears=True, rms_norm_eps=1e-5)
    config = Gemma4Config(text_config=text, vision_config=vision, audio_config=audio,
                          image_token_id=49, video_token_id=50, audio_token_id=57, boi_token_id=3,
                          eoi_token_id=48, boa_token_id=4, eoa_token_index=56,
                          pad_token_id=0, bos_token_id=2, eos_token_id=1)
    config._attn_implementation = "eager"
    torch.manual_seed(SEED)
    model = Gemma4ForConditionalGeneration(config).eval()
    model.model.audio_tower.config._attn_implementation = "sdpa"
    if model.model.audio_tower.config._attn_implementation != "sdpa":
        raise RuntimeError("the audio tower must run with the boolean sdpa mask")
    processor = Gemma4Processor(Gemma4AudioFeatureExtractor(feature_size=16), Gemma4ImageProcessor(),
                                tokenizer(), Gemma4VideoProcessor())
    return config, model, processor


def randomize(model):
    torch.manual_seed(SEED)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 1 and name.endswith("weight"):
                parameter.uniform_(0.7, 1.3)
        for name, buffer in model.named_buffers():
            if name.endswith("input_min"):
                buffer.fill_(-1.2)
            elif name.endswith("input_max"):
                buffer.fill_(1.2)
            elif name.endswith("output_min"):
                buffer.fill_(-0.08)
            elif name.endswith("output_max"):
                buffer.fill_(0.08)


def write(root: Path):
    if transformers.__version__ != "5.16.1":
        raise RuntimeError("Reference fixtures require Transformers 5.16.1")
    torch.set_num_threads(2)
    time = np.arange(4096, dtype=np.float32) / 16000
    waveforms = [0.3 * np.sin(2 * np.pi * 437 * time) + 0.15 * np.cos(2 * np.pi * 113 * time),
                 0.4 * np.sin(2 * np.pi * 283 * time[:1693])]
    for kind, build in (("gemma-3n-audio-tiny", gemma3n), ("gemma-4-audio-tiny", gemma4)):
        directory = root / kind
        directory.mkdir(parents=True, exist_ok=True)
        config, model, processor = build()
        randomize(model)
        prompts = ["listen <audio> ok", "say <audio> again ok"]
        inputs = processor(text=prompts, audio=waveforms, padding=True, return_tensors="np")
        tensors = {key: torch.from_numpy(np.asarray(value)) for key, value in inputs.items()}
        forward = {key: value for key, value in tensors.items()
                   if key in ("input_ids", "attention_mask", "input_features", "input_features_mask",
                              "token_type_ids", "mm_token_type_ids")}
        with torch.no_grad():
            logits = model(**forward, use_cache=False).logits
            generated = model.generate(**forward, max_new_tokens=3, do_sample=False)
        save_file({name: value.contiguous().clone() for name, value in model.state_dict().items()},
                  str(directory / "model.safetensors"), metadata={"format": "pt"})
        (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        (directory / "generation_config.json").write_text(json.dumps(
            {"do_sample": False, "eos_token_id": 1, "pad_token_id": 0, "bos_token_id": 2}, indent=2) + "\n")
        processor.save_pretrained(str(directory))
        for index, waveform in enumerate(waveforms):
            np.save(directory / f"waveform_{index}.npy", waveform)
        np.savez(directory / "reference.npz", logits=logits.numpy(), generated=generated.numpy(),
                 **{key: np.asarray(value) for key, value in inputs.items()})
        (directory / "meta.json").write_text(json.dumps({
            "torch": torch.__version__, "transformers": transformers.__version__, "seed": SEED,
            "prompts": prompts, "sampling_rate": 16000, "backend": "cpu", "dtype": "float32",
            "parameters": sum(p.numel() for p in model.parameters())}, indent=2) + "\n")
        print(kind, "ids", inputs["input_ids"].shape, "features", inputs["input_features"].shape,
              "logits", tuple(logits.shape), "generated", generated.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/hf")
    write(parser.parse_args().out)
