"""MiMo-V2.6's audio path at a tiny scale, run by its references, for tests/test_mimo_audio.py.

Two parts, each in the environment that has its reference:

- `--mel`: torchaudio's MelSpectrogram with MiMo-Audio-Tokenizer's `mel_spectrogram` settings (24 kHz,
  nfft 960, hop 240, 128 mels, magnitudes, log floored at 1e-7) on a fixed waveform, fp32 and
  float64, into `mel.npz`. torchaudio stops at torch 2.11, so this part runs in an environment of
  torch 2.11 and torchaudio 2.11.
- default: the audio tokenizer's encoder (AudioTokenizerEncoder, its codes by tokenize_audio_batch) and
  the audio encoder (MiMoAudioEncoder over the codes) from XiaomiMiMo/MiMo-V2.6-Pro-MOPD's
  `modeling_mimo_v2.py` at `MIMO_REVISION`, at a tiny scale with scattered weights, into
  `model.npz`: the tensors as the release names them, the inputs, the pre-quantization features,
  the codes and the embeddings, fp32 and float64.

    python tools/mimo_audio_reference.py
    <torch 2.11 + torchaudio python> tools/mimo_audio_reference.py --mel
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf" / "mimo-audio-tiny"
MIMO_REPO = "XiaomiMiMo/MiMo-V2.6-Pro-MOPD"
MIMO_REVISION = "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8"

TOKENIZER = {"n_mels": 16, "sampling_rate": 24000, "nfft": 960, "hop_length": 240, "window_size": 960,
             "fmin": 0, "fmax": None, "d_model": 32, "kernel_size": 3, "stride_size": 2, "avg_pooler": 2,
             "encoder_layers": 4, "encoder_attention_heads": 4, "encoder_ffn_dim": 48, "encoder_causal": True,
             "encoder_attn_window_size": [3, 0], "encoder_skip_layer_id": 2, "hybrid_attention": True,
             "swa_per_block": 2, "rope_theta": 10000, "rope_type": "default", "ln_type": "LayerNorm",
             "activation_function": "gelu", "num_quantizers": 4, "codebook_size": [16, 16, 8, 8],
             "max_audio_seconds": 30, "scale_embedding": False, "threshold_ema_dead_code": 2,
             "position_embedding_type": "rope"}
"""Two windowed and two full causal layers, a window of three frames back, a skip from layer two."""

AUDIO = {"audio_channels": 4, "group_size": 4, "input_local_dim": 32, "input_local_layers": 2,
         "input_local_attn_heads": 4, "input_local_intermediate_size": 48, "input_full_attention": True,
         "rope_theta": 640000, "partial_rotary_factor": 1.0, "add_post_norm": True, "projection_layers": 2,
         "out_hidden_size": 24, "speech_vocab_size": "20", "speech_zeroemb_idx": "16",
         "audio_segment_size": 20, "input_local_hidden_dropout": 0.0}

LENGTHS = (45, 13)
"""Mel frames of two clips: one cut into segments of 20, 20 and 5 frames, one of 13. Their
convolved lengths, 10, 10, 3 and 7, put two odd ones beside the batch's longest."""


def mimo_module():
    """MiMo's modeling module, fetched at MIMO_REVISION and imported as a package beside its config."""
    from huggingface_hub import hf_hub_download

    package = types.ModuleType("mimo_v2")
    package.__path__ = []
    sys.modules["mimo_v2"] = package
    for name in ("configuration_mimo_v2", "modeling_mimo_v2"):
        path = hf_hub_download(MIMO_REPO, f"{name}.py", revision=MIMO_REVISION)
        spec = importlib.util.spec_from_file_location(f"mimo_v2.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"mimo_v2.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules["mimo_v2.modeling_mimo_v2"]


def scatter(module: torch.nn.Module, seed: int) -> None:
    """Something in every tensor: norms near one, biases small, maps random; codebooks random."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
            if not tensor.is_floating_point() or "inv_freq" in name:
                continue
            if "norm" in name:
                tensor.copy_(1.0 + torch.randn(tensor.shape, generator=generator) * 0.05)
            elif "bias" in name:
                tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.2)
            else:
                tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.4)


def write_mel() -> None:
    """torchaudio's mel frames of a fixed half second at MiMo's settings, fp32 and float64."""
    from torchaudio.transforms import MelSpectrogram

    waveform = np.random.RandomState(5).randn(12000).astype(np.float32) * 0.3

    def mel(dtype):
        transform = MelSpectrogram(sample_rate=24000, n_fft=960, hop_length=240, win_length=960, f_min=0,
                                   f_max=None, n_mels=128, power=1.0, center=True).to(dtype)
        spec = transform(torch.from_numpy(waveform).to(dtype)[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze(0).T.numpy()

    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "mel.npz", precision="ieee", waveform=waveform, mel=mel(torch.float32),
             mel_f64=mel(torch.float64))


def write_model() -> None:
    from diffusers_wan_reference import float64

    modeling = mimo_module()

    def build():
        tokenizer = modeling.AudioTokenizerEncoder(modeling.MiMoAudioTokenizerConfig(**TOKENIZER)).eval()
        config = SimpleNamespace(**AUDIO)
        encoder = modeling.MiMoAudioEncoder(config).eval()
        embeddings = modeling._build_speech_embeddings(config)
        return tokenizer, encoder, embeddings

    torch.manual_seed(1234)
    tokenizer, encoder, embeddings = build()
    for index, module in enumerate((tokenizer, encoder, embeddings)):
        scatter(module, 1234 + index)
    mels = [np.random.RandomState(7 + i).randn(length, TOKENIZER["n_mels"]).astype(np.float32)
            for i, length in enumerate(LENGTHS)]

    def run(modules, dtype):
        tokens, encode, embed = modules
        caught = []
        hook = tokens.down_sample_norm.register_forward_hook(lambda module, args, output: caught.append(output))
        codes = modeling.tokenize_audio_batch([torch.from_numpy(m).to(dtype) for m in mels], tokens,
                                              segment_size=AUDIO["audio_segment_size"])
        hook.remove()
        with torch.no_grad():
            grouped = [modeling._pad_and_group_audio_codes(c, AUDIO["audio_channels"], AUDIO["group_size"])
                       for c in codes]
            hidden = encode._apply_input_local_transformer(
                encode._apply_speech_embeddings(torch.cat(grouped), embed))
            features = encode.projection(hidden.reshape(hidden.shape[0], -1))
        counts = [len(g) for g in grouped]
        return caught[0].numpy(), [c.numpy() for c in codes], np.split(features.numpy(), np.cumsum(counts)[:-1])

    features, codes, embedded = run((tokenizer, encoder, embeddings), torch.float32)
    with float64():
        wide = build()
        for source, target in zip((tokenizer, encoder, embeddings), wide, strict=True):
            target.load_state_dict({name: tensor.double() if tensor.is_floating_point() else tensor
                                    for name, tensor in source.state_dict().items()})
        features_f64, codes_f64, _ = run(wide, torch.float64)
        # The encoder in float64 over the fp32 run's codes, the codes it reads.
        with torch.no_grad():
            grouped = [modeling._pad_and_group_audio_codes(torch.from_numpy(c), AUDIO["audio_channels"],
                                                           AUDIO["group_size"]) for c in codes]
            hidden = wide[1]._apply_input_local_transformer(wide[1]._apply_speech_embeddings(torch.cat(grouped),
                                                                                              wide[2]))
            embedded_f64 = np.split(wide[1].projection(hidden.reshape(hidden.shape[0], -1)).numpy(),
                                    np.cumsum([len(g) for g in grouped])[:-1])
    assert all(np.array_equal(a, b) for a, b in zip(codes, codes_f64, strict=True)), "a code flips in float64"
    tensors = {**{f"tokenizer/{name}": tensor for name, tensor in tokenizer.state_dict().items()},
               **{f"encoder/audio_encoder.{name}": tensor for name, tensor in encoder.state_dict().items()
                  if "embed_tokens" not in name and "inv_freq" not in name},
               **{f"encoder/speech_embeddings.{name}": tensor for name, tensor in embeddings.state_dict().items()}}
    FIXTURE.mkdir(parents=True, exist_ok=True)
    width = max(len(m) for m in mels)
    np.savez(FIXTURE / "model.npz", precision="ieee",
             **{name: tensor.numpy() for name, tensor in tensors.items()},
             mels=np.stack([np.pad(m, ((0, width - len(m)), (0, 0))) for m in mels]),
             lengths=np.asarray(LENGTHS, np.int32), features=features, features_f64=features_f64,
             **{f"codes_{i}": c for i, c in enumerate(codes)},
             **{f"embedded_{i}": e for i, e in enumerate(embedded)},
             **{f"embedded_f64_{i}": e for i, e in enumerate(embedded_f64)})
    import json

    (FIXTURE / "config.json").write_text(json.dumps({"audio_config": AUDIO, "tokenizer": TOKENIZER}, indent=1) + "\n")
    print(FIXTURE, sorted(p.name for p in FIXTURE.iterdir()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel", action="store_true", help="write mel.npz, under torchaudio")
    if parser.parse_args().mel:
        write_mel()
    else:
        write_model()


if __name__ == "__main__":
    main()
