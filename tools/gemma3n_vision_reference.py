#!/usr/bin/env python3
"""Write local Gemma 3n vision fixtures with timm 1.0.29 / transformers 5.16.1.

No pretrained weights are read. The timm encoder retains all 84 blocks and
all attention heads; channel_multiplier=1/64 and an eight-channel stem keep
it at 2,140,080 parameters. The fusion output is four 2048-wide tokens.
The image-only wrapper fixture excludes audio config/weights explicitly;
its reference logits come from the real conditional-generation model with
a tiny audio tower and no audio input.

Run with Torch 2.14.0 CPU, torchvision 0.29.0 and the versions above:
  python tools/gemma3n_vision_reference.py --out tests/fixtures/hf/gemma3n-vision-tiny
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import timm
import torch
import transformers
from safetensors.numpy import save_file
from transformers import Gemma3nAudioConfig, Gemma3nConfig, Gemma3nTextConfig, Gemma3nVisionConfig
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nForConditionalGeneration


MODEL_ARGS = {"channel_multiplier": 1 / 64, "stem_size": 8, "msfa_output_resolution": 2}
SEED = 1729
LEARNING_RATE = 0.01


def reference_model():
    text = Gemma3nTextConfig(
        vocab_size=64, vocab_size_per_layer_input=48, hidden_size=32,
        intermediate_size=[48, 48, 64, 64], num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        sliding_window=4, max_position_embeddings=64, rms_norm_eps=1e-6,
        rope_theta=1e6, rope_local_base_freq=1e4, final_logit_softcapping=5.0,
        hidden_size_per_layer_input=8, altup_num_inputs=3, altup_active_idx=0,
        altup_coef_clip=120.0, altup_correct_scale=True, num_kv_shared_layers=1,
        laurel_rank=8, activation_sparsity_pattern=[0.0, 0.0, 0.0, 0.0],
        tie_word_embeddings=True)
    vision = Gemma3nVisionConfig(hidden_size=2048, vocab_size=8, vocab_offset=48,
                                 model_args=MODEL_ARGS)
    audio = Gemma3nAudioConfig(
        hidden_size=32, input_feat_size=16, vocab_size=8, vocab_offset=56,
        conf_num_attention_heads=4, conf_num_hidden_layers=1,
        sscp_conv_channel_size=(8, 8))
    config = Gemma3nConfig(
        text_config=text, vision_config=vision, audio_config=audio,
        image_token_id=49, audio_token_id=57, boi_token_id=3, eoi_token_id=48,
        boa_token_id=4, eoa_token_id=56, vision_soft_tokens_per_image=4,
        pad_token_id=0, bos_token_id=2, eos_token_id=1)
    config._attn_implementation = "eager"
    torch.manual_seed(SEED)
    model = Gemma3nForConditionalGeneration(config).eval()
    with torch.no_grad():
        # Learned checkpoints need not retain the near-zero layer-scale init.
        # Active residual branches make a missing block observable in fixtures.
        for name, parameter in model.model.vision_tower.named_parameters():
            if name.endswith("layer_scale.gamma"):
                parameter.uniform_(0.25, 0.75)
    return model


def arrays(state):
    return {name: value.detach().cpu().float().numpy().copy() for name, value in state.items()}


def write_reference(directory: Path):
    if timm.__version__ != "1.0.29" or transformers.__version__ != "5.16.1":
        raise RuntimeError("This fixture is pinned to timm 1.0.29 and transformers 5.16.1")
    torch.set_num_threads(2)
    directory.mkdir(parents=True, exist_ok=True)
    model = reference_model()
    tower = model.model.vision_tower.timm_model
    projector = model.model.embed_vision

    config = model.config.to_dict()
    config["audio_config"] = None
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    image_state = {name: tensor for name, tensor in model.state_dict().items()
                   if not name.startswith(("model.audio_tower.", "model.embed_audio."))}
    save_file(arrays(image_state), str(directory / "model.safetensors"))
    generator = torch.Generator().manual_seed(SEED + 1)
    for name, height, width in (("odd", 33, 29), ("even", 32, 32), ("pooled", 64, 64)):
        pixels = torch.randn((1, 3, height, width), generator=generator)
        with torch.no_grad():
            features = tower(pixels).permute(0, 2, 3, 1).reshape(1, 4, 2048)
            soft = model.get_image_features(pixels).pooler_output
        np.save(directory / f"pixels_{name}.npy", pixels.numpy())
        np.save(directory / f"tower_{name}.npy", features.numpy())
        np.save(directory / f"soft_{name}.npy", soft.numpy())

    ids = torch.tensor([[2, 3, 49, 49, 49, 49, 48, 7, 54, 5, 1]])
    pixels = torch.from_numpy(np.load(directory / "pixels_odd.npy"))
    with torch.no_grad():
        logits = model(input_ids=ids, pixel_values=pixels, use_cache=False).logits
        hard = projector(input_ids=torch.tensor([[48, 50, 55]]))
    np.save(directory / "input_ids.npy", ids.numpy().astype(np.int32))
    np.save(directory / "wrapper_ref.npy", logits.numpy())
    np.save(directory / "hard_ref.npy", hard.numpy())

    pixels = pixels.clone().requires_grad_(True)
    weights = torch.linspace(-1, 1, 4 * 2048).reshape(1, 4, 2048)
    output = tower(pixels).permute(0, 2, 3, 1).reshape(1, 4, 2048)
    loss = (output * weights).mean()
    loss.backward()
    np.save(directory / "input_grad.npy", pixels.grad.numpy())
    with torch.no_grad():
        for parameter in tower.parameters():
            parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
        stepped = tower(pixels).permute(0, 2, 3, 1).reshape(1, 4, 2048)
    np.save(directory / "stepped_ref.npy", stepped.numpy())

    bf16_tower = copy.deepcopy(tower).bfloat16()
    # Reload the pre-update fixture so dtype comparisons share the fp32 weights.
    from safetensors.numpy import load_file
    prefix = "model.vision_tower.timm_model."
    bf16_tower.load_state_dict({name.removeprefix(prefix): torch.from_numpy(value)
                                for name, value in load_file(str(directory / "model.safetensors")).items()
                                if name.startswith(prefix)})
    with torch.no_grad():
        bf16 = bf16_tower(pixels.detach().bfloat16()).permute(0, 2, 3, 1).reshape(1, 4, 2048)
    np.save(directory / "tower_bf16.npy", bf16.float().numpy())
    (directory / "meta.json").write_text(json.dumps({
        "torch": torch.__version__, "transformers": transformers.__version__,
        "timm": timm.__version__, "seed": SEED, "dtype": "float32", "backend": "cpu",
        "tower_parameters": sum(p.numel() for p in tower.parameters()),
        "learning_rate": LEARNING_RATE, "loss": float(loss.detach()),
        "gradient_loss": "mean(tower(pixels_odd) * linspace(-1, 1, 8192).reshape(1, 4, 2048))",
        "scope": "Image-only wrapper bundle; real conditional reference received no audio input",
    }, indent=2) + "\n")
    print(f"Wrote {directory}; tower parameters {sum(p.numel() for p in tower.parameters())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]
                        / "tests" / "fixtures" / "hf" / "gemma3n-vision-tiny")
    write_reference(parser.parse_args().out)
