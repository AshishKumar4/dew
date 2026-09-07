#!/usr/bin/env python3
"""Generate native audio fixtures from Transformers 5.16.1 on Torch CPU.

Every weight is initialized locally. The fixtures contain two unequal-length
16 kHz waveforms, the owner's NumPy preprocessing, encoder/projector outputs,
input gradients and one SGD update. No pretrained weights are downloaded.

  python tools/audio_reference.py --out tests/fixtures/audio
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.numpy import save_file
from transformers import (Gemma3nAudioConfig, Gemma3nAudioFeatureExtractor, Gemma3nTextConfig,
                          Gemma4AudioConfig, Gemma4AudioFeatureExtractor, Gemma4TextConfig)
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nAudioEncoder, Gemma3nMultimodalEmbedder
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder


LEARNING_RATE = 0.002


def reference(kind):
    torch.manual_seed(314)
    if kind == "gemma3n":
        config = Gemma3nAudioConfig(
            input_feat_size=16, hidden_size=16, conf_num_hidden_layers=2,
            conf_num_attention_heads=2, conf_attention_chunk_size=4,
            conf_attention_context_left=5, conf_attention_context_right=2,
            sscp_conv_channel_size=(8, 4), conf_reduction_factor=2,
            vocab_offset=56, vocab_size=8, rms_norm_eps=1e-5)
        model = Gemma3nAudioEncoder(config).eval()
        projector = Gemma3nMultimodalEmbedder(config, Gemma3nTextConfig(hidden_size=32)).eval()
        extractor = Gemma3nAudioFeatureExtractor(feature_size=16)
    else:
        config = Gemma4AudioConfig(
            hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            subsampling_conv_channels=(16, 4), output_proj_dims=24,
            attention_chunk_size=4, attention_context_left=5, attention_context_right=2,
            use_clipped_linears=True, rms_norm_eps=1e-5)
        config._attn_implementation = "sdpa"
        model = Gemma4AudioModel(config).eval()
        projector = Gemma4MultimodalEmbedder(config, Gemma4TextConfig(hidden_size=32)).eval()
        extractor = Gemma4AudioFeatureExtractor(feature_size=16)
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
        for name, parameter in projector.named_parameters():
            if parameter.ndim > 1:
                parameter.normal_(std=0.1)
            else:
                parameter.uniform_(0.7, 1.3)
    return config, model, projector, extractor


def to_arrays(state):
    return {name: tensor.detach().cpu().float().numpy().copy() for name, tensor in state.items()}


def write_reference(root):
    if transformers.__version__ != "5.16.1":
        raise RuntimeError("Reference fixtures require Transformers 5.16.1")
    torch.set_num_threads(2)
    time = np.arange(4096, dtype=np.float32) / 16000
    waveforms = [0.3 * np.sin(2 * np.pi * 437 * time) + 0.15 * np.cos(2 * np.pi * 113 * time),
                 0.4 * np.sin(2 * np.pi * 283 * time[:1693])]
    for kind in ("gemma3n", "gemma4"):
        directory = root / kind
        directory.mkdir(parents=True, exist_ok=True)
        config, model, projector, extractor = reference(kind)
        for index, waveform in enumerate(waveforms):
            np.save(directory / f"waveform_{index}.npy", waveform)
        processed = extractor(waveforms, padding="longest", return_tensors="np")
        features = torch.tensor(processed["input_features"], requires_grad=True)
        mask = torch.tensor(processed["input_features_mask"])
        output = model(features, ~mask if kind == "gemma3n" else mask)
        valid = ~output.audio_mel_mask if kind == "gemma3n" else output.attention_mask
        hidden = output.last_hidden_state
        soft = projector(inputs_embeds=hidden)
        coefficient = torch.linspace(-1, 1, soft.numel()).reshape_as(soft)
        loss = torch.mean(soft * coefficient)
        loss.backward()
        save_file(to_arrays(model.state_dict()), str(directory / "encoder.safetensors"))
        save_file(to_arrays(projector.state_dict()), str(directory / "projector.safetensors"))
        save_file({name: p.grad.detach().numpy().copy() for name, p in model.named_parameters()},
                  str(directory / "encoder_grad.safetensors"))
        np.savez(directory / "reference.npz", input_features=features.detach().numpy(),
                 input_features_mask=mask.numpy(), encoded=hidden.detach().numpy(), valid=valid.numpy(),
                 projected=soft.detach().numpy(), input_gradient=features.grad.detach().numpy(),
                 coefficient=coefficient.numpy(), loss=loss.detach().numpy())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
            for parameter in projector.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
            stepped = model(features.detach(), ~mask if kind == "gemma3n" else mask)
            projected = projector(inputs_embeds=stepped.last_hidden_state)
        np.save(directory / "stepped.npy", projected.numpy())
        (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        (directory / "preprocessor_config.json").write_text(extractor.to_json_string())
        (directory / "meta.json").write_text(json.dumps({
            "torch": torch.__version__, "transformers": transformers.__version__,
            "seed": 314, "sampling_rate": 16000, "learning_rate": LEARNING_RATE,
            "text_width": 32, "backend": "cpu", "dtype": "float32",
            "parameters": sum(p.numel() for p in model.parameters()),
        }, indent=2) + "\n")
        print(kind, "mel", tuple(features.shape), "encoded", tuple(hidden.shape),
              "valid", valid.sum(1).tolist(), "params", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/audio")
    write_reference(parser.parse_args().out)
